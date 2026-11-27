#!/usr/bin/env python3
"""从清洗版知乎回答里按问题标题粗筛，再用 DeepSeek 判文体，挑出稀缺文体候选。

来源：corpus/work/zhihu_genre/zhihu_kol.clean-000.jsonl.gz（每行 JSON）
产出：corpus/work/zhihu_genre/labels.jsonl（逐条追加，可断点续跑）
      data/v3/extra/zhihu_genre/<genre>.jsonl（每次整体重写，派生产物）

用法：
  .venv/bin/python scripts/genre_scout.py --limit 100   # 试跑
  .venv/bin/python scripts/genre_scout.py               # 全量
"""
from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srh import llm  # noqa: E402

SRC = ROOT / "corpus/work/zhihu_genre/zhihu_kol.clean-000.jsonl.gz"
RAW = ROOT / "data/v3/raw/zhihu.jsonl"
WORK = ROOT / "corpus/work/zhihu_genre"
LABELS = WORK / "labels.jsonl"
FAILED = WORK / "failed_batches.jsonl"
OUTDIR = ROOT / "data/v3/extra/zhihu_genre"
MODEL = os.environ.get("VERNA_DS_MODEL", "deepseek-chat@https://api.deepseek.com/v1")  # <模型>@<OpenAI 兼容接口>
MODEL_LABEL = llm.label(MODEL)

GENRES = ["academic", "news", "official", "business", "tech", "popsci", "opinion",
          "forum", "essay", "fiction", "marketing", "speech", "other"]
SCARCE = ["business", "fiction", "marketing", "popsci", "essay"]
PER_BUCKET_CAP = 4000
MIN_UPVOTES = 300
DATE_CUTOFF = "2022-11-30"
MIN_CHARS = 300
CONF_MIN = 0.7

# 粗筛桶。顺序即报告和 --limit 的取用顺序；排除词命中则整个桶不记。
BUCKETS: list[tuple[str, re.Pattern, re.Pattern | None]] = [
    ("fiction", re.compile(
        r"故事|小说|短篇|脑洞|写一个|编一个|以.{1,20}(为开头|开头|结尾)|续写|微小说|灵异|恐怖经历"),
     re.compile(r"如何评价|怎么评价|推荐|哪本|哪些.{0,6}小说|改编|电影|电视剧|动漫|游戏")),
    ("popsci", re.compile(
        r"原理|是怎么(形成|产生|做到|实现|工作|来的)|为什么|科学|物理|化学|生物|医学|天文|宇宙|"
        r"进化|基因|病毒|细胞|量子|地球|动物|植物|大脑|人体"),
     re.compile(r"历史|朝|皇帝|游戏|英雄联盟|电影|电视剧|动漫|明星|如何评价|怎么评价")),
    ("essay", re.compile(
        r"怎样的体验|什么体验|什么感受|什么感觉|最难忘|回忆|童年|故乡|家乡|父亲|母亲|爸爸|妈妈|"
        r"外婆|外公|奶奶|爷爷|离别|青春|那一刻"),
     re.compile(r"如何评价|怎么评价|就业|怎么样|值得|推荐")),
    ("marketing", re.compile(r"文案|广告语|slogan|标语|宣传语|产品介绍|卖点", re.I),
     re.compile(r"沙雕|说说|朋友圈|网名|签名|情话")),
    ("business", re.compile(r"邮件|周报|述职|工作总结|工作汇报|汇报材料|商务信函|商业计划书|报告怎么写"),
     None),
]

PROMPT_HEAD = """你是中文文本的文体分类器。下面每条是一条知乎回答，给出序号、问题标题和正文开头。\
请判断**正文本身**属于哪种文体，只从下面 13 个里选一个：

academic 学术论文；news 新闻报道；official 公文通知；business 商务邮件与工作报告；\
tech 技术文档与教程；popsci 科普说明；opinion 评论观点（书面）；forum 网络口语（论坛、社交媒体、问答）；\
essay 散文随笔；fiction 小说叙事；marketing 产品与营销文案；speech 演讲稿与口播；other 其他。

判别要点：
- fiction：正文本身在讲虚构故事；书评、影评、推荐书单不算。
- essay：作者写自己的经历和感受，有叙事有抒情；干巴巴列清单的经验贴算 forum。
- popsci：给普通人讲清科学、技术或医学原理；历史、娱乐、八卦不算。
- marketing：正文本身就是产品介绍或广告文案；讨论品牌好坏的算 opinion 或 forum。
- business：正文本身就是商务邮件、工作报告、商业信函；讲怎么写邮件的经验贴不算。
- 问答平台的回答默认偏 forum，只有正文确实呈现上述文体特征时才选对应类别。

每条给 genre 和 conf（0 到 1，判断有多确定）。只回 JSON，不要任何解释，格式：
{"items": [{"i": 1, "genre": "forum", "conf": 0.8}]}

共 N 条：
"""

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_raw_ids() -> set[str]:
    ids = set()
    with RAW.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def as_int(v) -> int:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return 0


def scan(raw_ids: set[str]) -> tuple[dict[str, dict], dict[str, int], dict[str, int], dict[str, list[str]]]:
    """流式读 clean 语料，返回 (候选 id→doc, 各桶命中数, 各桶截断后条数, 各桶保留的 id 表)。"""
    matched: dict[str, list[str]] = {name: [] for name, _, _ in BUCKETS}
    docs: dict[str, dict] = {}
    n_total = 0
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rid, q, text = d["id"], d.get("question") or "", d.get("text") or ""
            except (json.JSONDecodeError, KeyError):
                continue
            if rid in raw_ids:
                continue
            if as_int(d.get("upvotes")) < MIN_UPVOTES:
                continue
            if str(d.get("created") or "") >= DATE_CUTOFF:
                continue
            if len(text) < MIN_CHARS:
                continue
            hits = []
            for name, inc, exc in BUCKETS:
                if inc.search(q) and not (exc and exc.search(q)):
                    hits.append(name)
            if not hits:
                continue
            docs[rid] = d
            for name in hits:
                matched[name].append(rid)
            if n_total % 20000 == 0:
                log(f"  扫描 {n_total} 行，候选 {len(docs)}")
    log(f"扫描完成：{n_total} 行，粗筛候选 {len(docs)} 条")
    counts = {name: len(v) for name, v in matched.items()}
    kept = {}
    for name, ids in matched.items():
        ids.sort(key=md5)
        kept[name] = len(ids[:PER_BUCKET_CAP])
        matched[name] = ids[:PER_BUCKET_CAP]
    return docs, counts, kept, matched


def load_labels() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not LABELS.exists():
        return out
    with LABELS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id"):
                out[r["id"]] = r
    return out


def build_prompt(batch: list[dict]) -> str:
    parts = [PROMPT_HEAD.replace("共 N 条：", f"共 {len(batch)} 条：")]
    for i, d in enumerate(batch, 1):
        body = re.sub(r"\s+", " ", (d.get("text") or ""))[:400]
        parts.append(f"[{i}] 标题：{d.get('question', '')}\n正文开头：{body}\n")
    return "\n".join(parts)


def valid_items(j: dict, n: int) -> dict[int, dict] | None:
    items = j.get("items")
    if not isinstance(items, list):
        return None
    out = {}
    for it in items:
        try:
            i = int(it["i"])
            g = str(it["genre"]).strip().lower()
            c = float(it["conf"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (1 <= i <= n) or g not in GENRES:
            return None
        out[i] = {"genre": g, "conf": max(0.0, min(1.0, c))}
    return out if len(out) == n else None


def label_batch(batch: list[dict], widx: int) -> tuple[list[dict], str | None, int]:
    """判一批；解析失败重试一次。返回 (labels, 错误描述或 None, 实际调用次数)。"""
    prompt = build_prompt(batch)
    err = ""
    for attempt in (1, 2):
        try:
            items = valid_items(llm.ask_json(MODEL, prompt, tag="genre_scout", timeout=240), len(batch))
            if items is not None:
                out = []
                for i, d in enumerate(batch, 1):
                    it = items[i]
                    out.append({"id": d["id"], "question": d.get("question") or "",
                                "bucket": "+".join(d["_buckets"]), "genre": it["genre"],
                                "conf": round(it["conf"], 3), "model": MODEL_LABEL})
                return out, None, attempt
            err = "格式不符"
        except Exception as e:  # 网络、解析都算，重试一次
            err = f"{type(e).__name__}: {e}"[:300]
        if attempt == 1:
            log(f"  批 {widx} 解析失败（{err}），重试")
    return [], err, 2


def write_outputs(docs: dict[str, dict], labels: dict[str, dict]) -> dict[str, int]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {g: [] for g in SCARCE}
    for rid, lab in labels.items():
        d = docs.get(rid)
        if not d:
            continue
        g = lab.get("genre")
        if g in SCARCE and float(lab.get("conf") or 0) >= CONF_MIN:
            rows[g].append({
                "id": rid,
                "question": d.get("question") or "",
                "text": d.get("text") or "",
                "created": d.get("created") or "",
                "upvotes": d.get("upvotes") or "",
                "url": d.get("url") or "",
                "scout_genre": g,
                "scout_conf": lab.get("conf"),
                "scout_model": lab.get("model") or MODEL_LABEL,
            })
    counts = {}
    for g, rs in rows.items():
        rs.sort(key=lambda r: md5(r["id"]))
        tmp = OUTDIR / f".{g}.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(OUTDIR / f"{g}.jsonl")
        counts[g] = len(rs)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多判多少条（按桶顺序+md5，试跑用）")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    t_start = time.time()
    log(f"模型 {MODEL_LABEL}")
    raw_ids = load_raw_ids()
    log(f"已收录于 data/v3/raw/zhihu.jsonl：{len(raw_ids)} 条，将跳过")

    docs, counts, kept, matched = scan(raw_ids)

    # 候选顺序：按桶顺序、每桶 md5 序，跨桶去重
    ordered: list[str] = []
    seen: set[str] = set()
    buckets_of: dict[str, list[str]] = {}
    for name, _, _ in BUCKETS:
        for rid in matched[name]:
            buckets_of.setdefault(rid, []).append(name)
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
    for rid in ordered:
        docs[rid]["_buckets"] = buckets_of[rid]
    log(f"截断后候选合计 {len(ordered)} 条（去重后）")
    log("粗筛各桶：" + "，".join(f"{n} {counts[n]}→{kept[n]}" for n, _, _ in BUCKETS))

    done = load_labels()
    pending = [rid for rid in ordered if rid not in done]
    log(f"已有标注 {len(done)} 条，待判 {len(pending)} 条")
    if args.limit:
        pending = pending[:args.limit]
        log(f"--limit {args.limit}：本次只判 {len(pending)} 条")

    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    n_calls = 0
    n_failed = 0
    lock = threading.Lock()
    done_n = 0

    def run(widx: int, ids: list[str]) -> None:
        nonlocal n_calls, n_failed, done_n
        batch = [docs[r] for r in ids]
        rows, err, attempts = label_batch(batch, widx)
        with lock:
            n_calls += attempts
            if err is not None:
                n_failed += 1
                WORK.mkdir(parents=True, exist_ok=True)
                with FAILED.open("a") as f:
                    f.write(json.dumps({"ts": round(time.time()), "ids": ids, "error": err},
                                       ensure_ascii=False) + "\n")
                log(f"  批 {widx} 失败，跳过（{err}）")
            elif rows:
                with LABELS.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done_n += len(batch)
            if done_n % 100 < len(batch):
                rate = done_n / max(time.time() - t_start, 1e-6)
                eta = (len(pending) - done_n) / rate if rate else 0
                log(f"  进度 {done_n}/{len(pending)}  批失败 {n_failed}  "
                    f"{rate:.1f} 条/秒  剩余约 {eta/60:.0f} 分钟")

    WORK.mkdir(parents=True, exist_ok=True)
    if batches:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda p: run(*p), enumerate(batches)))

    labels = load_labels()
    out_counts = write_outputs(docs, labels)
    elapsed = time.time() - t_start
    log("")
    log(f"DeepSeek 调用批次 {n_calls}，失败批次 {n_failed}，本次耗时 {elapsed/60:.1f} 分钟")
    log(f"labels.jsonl 累计 {len(labels)} 条")
    log("输出：" + "，".join(f"{g} {out_counts[g]}" for g in SCARCE))

    # 交叉表：桶 × genre（同一 id 命中多桶时每个桶各计一次）
    cross: dict[str, dict[str, int]] = {n: {} for n, _, _ in BUCKETS}
    for rid, lab in labels.items():
        if rid not in docs:
            continue
        for b in docs[rid]["_buckets"]:
            cross[b][lab["genre"]] = cross[b].get(lab["genre"], 0) + 1
    log("")
    log("桶 × genre 交叉表：")
    for name, _, _ in BUCKETS:
        row = cross[name]
        tot = sum(row.values())
        cells = "，".join(f"{g} {row[g]}" for g in sorted(row, key=row.get, reverse=True) if row[g])
        log(f"  {name}（{tot}）：{cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
