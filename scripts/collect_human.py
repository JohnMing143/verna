#!/usr/bin/env python3
"""第 1 步：收集人写文字，切成 150–800 字的段，用规则粗筛。

  python scripts/collect_human.py --source zhihu --limit 30000 --out data/human_raw.jsonl
  python scripts/collect_human.py --source local:~/my_articles --out data/human_local.jsonl
  python scripts/collect_human.py --source cqia:douban --out data/human_douban.jsonl

来源：
  zhihu        wangrui6/Zhihu-KOL，按 answer_creation_time 只要 2022-11-30 之前、点赞 ≥ --min-upvotes
  cqia:<子集>  m-a-p/COIG-CQIA 的某个子集（douban、xhs、wikihow 等，2023 年整理，没有时间戳）
  local:<目录> 自己攒的 .txt/.md（喜欢的文章、自己写的东西），最能决定模型往哪种文风走

需要：pip install datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import tells  # noqa: E402

CUTOFF = "2022-11-30"
BAD = re.compile(
    r"https?://|www\.|编辑于|发布于|著作权归|谢邀|利益相关|如图|上图|下图|见图|\[图片\]|图源|"
    r"关注我|点赞|私信|评论区|公众号|扫码|二维码|淘宝|拼多多|优惠券|链接|"
    r"~{2,}|！{3,}|\?{3,}|？{3,}|\.{6,}|哈{4,}|呜{3,}"
)
SENT_END = re.compile(r"[。！？!?]")


def upvotes(s) -> float:
    m = re.search(r"([\d.]+)\s*(万|[kK])?", str(s or ""))
    if not m:
        return 0.0
    return float(m.group(1)) * {"万": 1e4, "k": 1e3, "K": 1e3}.get(m.group(2) or "", 1)


def paragraphs(text: str) -> list[str]:
    text = text.replace("\r", "")
    # Zhihu-KOL 把换行压成了空格：句末标点后的空白当段落分隔
    text = re.sub(r"(?<=[。！？!?”」])[ 　]+", "\n", text)
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def chunks(paras: list[str], lo: int = 150, hi: int = 700) -> list[str]:
    out, cur = [], ""
    for p in paras:
        if len(p) > hi:  # 超长段落按句切
            for s in re.split(r"(?<=[。！？!?])", p):
                if cur and len(cur) + len(s) > hi:
                    out.append(cur)
                    cur = ""
                cur += s
            continue
        if cur and len(cur) + len(p) + 1 > hi:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return [c.strip() for c in out if lo <= len(c.strip()) <= hi + 100]


def good(chunk: str, max_density: float) -> str | None:
    """返回 None 表示通过，否则返回拒绝原因。"""
    if BAD.search(chunk):
        return "噪声词"
    han = len(re.findall(r"[一-鿿]", chunk))
    if han / max(len(chunk), 1) < 0.6:
        return "中文比例低"
    if len(SENT_END.findall(chunk)) < 3:
        return "句子太少"
    if re.match(r"^(楼上|楼主|题主说|同上|补充|更新|追加|以上|另外|还有|所以|因此|但是|不过|而且)", chunk):
        return "依赖上文"
    if re.search(r"\S {2,}\S", chunk):  # 原文的图片/列表被删掉后留下的空洞
        return "内容残缺"
    if not re.search(r"[。！？!?”」…）)]$", chunk):
        return "没说完"
    if tells.banned(chunk):
        return "含禁用句式"
    # 短段落按绝对命中算：一段里最多 2 分（比如一个"总的来说"），长段落按每千字 max_density
    if tells.weighted(chunk) > max(2.0, max_density * len(chunk) / 1000):
        return "AI腔偏高"
    if re.search(r"(?m)^\s*(?:[-*•]|\d{1,2}[.、)）])\s*", chunk):
        return "列表"
    return None


def iter_zhihu(min_up: float):
    from datasets import load_dataset

    for i, r in enumerate(load_dataset("wangrui6/Zhihu-KOL", split="train", streaming=True)):
        try:
            m = json.loads(r.get("METADATA") or "{}")
        except json.JSONDecodeError:
            continue
        t = m.get("answer_creation_time") or ""
        if not t or t >= CUTOFF or upvotes(m.get("upvotes")) < min_up:
            continue
        doc = str(int(m.get("answer_id") or i))
        yield f"zhihu:{doc}", r["RESPONSE"] or "", {"time": t[:10], "upvotes": m.get("upvotes"), "url": m.get("url"), "q": r["INSTRUCTION"]}


def iter_cqia(cfg: str):
    from datasets import load_dataset

    for i, r in enumerate(load_dataset("m-a-p/COIG-CQIA", cfg, split="train", streaming=True)):
        if r.get("answer_from") != "human":
            continue
        yield f"cqia-{cfg}:{i}", r.get("output") or "", {"q": r.get("instruction")}


def iter_local(d: str):
    for p in sorted(Path(d).expanduser().rglob("*")):
        if p.suffix.lower() in (".txt", ".md") and p.is_file():
            text = p.read_text(errors="ignore")
            text = re.sub(r"(?m)^#{1,6}\s.*$|^>.*$|!\[.*?\]\(.*?\)|```.*?```", "", text, flags=re.S)
            yield f"local:{hashlib.md5(str(p).encode()).hexdigest()[:10]}", text, {"file": str(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20000, help="最多输出多少段")
    ap.add_argument("--min-upvotes", type=float, default=100)
    ap.add_argument("--max-density", type=float, default=3.0, help="AI 腔密度上限，人写知乎回答中位数约 1.5")
    ap.add_argument("--per-doc", type=int, default=2, help="每篇最多取几段，避免一个作者占太多")
    a = ap.parse_args()

    src = a.source
    it = iter_zhihu(a.min_upvotes) if src == "zhihu" else iter_cqia(src.split(":", 1)[1]) if src.startswith("cqia:") \
        else iter_local(src.split(":", 1)[1]) if src.startswith("local:") else None
    if it is None:
        raise SystemExit(f"不认识的来源 {src}")

    seen, n_doc, n_out, why = set(), 0, 0, {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for doc_id, text, meta in it:
            n_doc += 1
            took = 0
            for k, c in enumerate(chunks(paragraphs(text))):
                r = good(c, a.max_density)
                if r:
                    why[r] = why.get(r, 0) + 1
                    continue
                h = hashlib.md5(re.sub(r"\W", "", c)[:60].encode()).hexdigest()
                if h in seen:
                    why["重复"] = why.get("重复", 0) + 1
                    continue
                seen.add(h)
                f.write(json.dumps({"id": f"{doc_id}:{k}", "doc": doc_id, "text": c, "meta": meta}, ensure_ascii=False) + "\n")
                n_out += 1
                took += 1
                if took >= a.per_doc:
                    break
            if n_out >= a.limit:
                break
            if n_doc % 2000 == 0:
                print(f"已扫 {n_doc} 篇，输出 {n_out} 段", file=sys.stderr)
    print(json.dumps({"docs": n_doc, "chunks": n_out, "rejected": why}, ensure_ascii=False), flush=True)
    sys.stderr.flush()
    os._exit(0)  # datasets 流式读取的后台线程在解释器退出时偶尔会崩，直接退出


if __name__ == "__main__":
    main()
