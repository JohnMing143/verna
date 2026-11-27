#!/usr/bin/env python3
"""Verna（内部代号说人话）小模型的无人值守数据管线。用户休息时长时间运行，到需要判断的节点用 `claude -p` 唤醒 Claude 处理。

流程（每一轮都从断点继续，所有产出只追加）：
  收集（后台线程，按 config.sources）→ 切段粗筛 → 批量分级 → 范本复审
  → 生成配对：反向样本（范本改坏→范本）、保留样本（范本→范本）、打磨（一般人写 / AI 初稿 → Argon 最好的版本）
  → 抽检 → 老师口头禅统计 → 训练集快照 → 进度报告

额度：每批 agy 调用前查 /quota；5 小时窗口低于下限就睡到重置；周额度低于下限就停掉所有 agy 环节（留给用户日常使用）。
唤醒节点：新来源分级后的复查、老师口头禅清单、抽检不合格、同一环节连续报错、每日小结。每天有次数上限。

  运行：  .venv/bin/python autopilot/run.py
  常驻：  见 autopilot/start.sh（systemd-run 托管，日志 data/v3/autopilot/run.log）
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from srh import facts, genres, harvest, llm, prompts, segment, tells  # noqa: E402
from srh.runner import read_jsonl  # noqa: E402
from verify_pairs import overlap  # noqa: E402

CFG = Path(os.environ.get("SRH_AP_CONFIG", ROOT / "autopilot" / "config.json"))   # 试跑时指向小配置
D = Path(os.environ.get("SRH_AP_DATA", ROOT / "data" / "v3"))                    # 试跑时指向临时目录
ONCE = bool(os.environ.get("SRH_AP_ONCE"))  # 试跑：同步收集一遍、跑一轮、不唤醒、不发报告，然后退出
AP = D / "autopilot"
for p in (D / "raw", D / "pairs", AP / "wakes"):
    p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SRH_USAGE_LOG", str(AP / "usage.jsonl"))
llm.USAGE_LOG = Path(os.environ["SRH_USAGE_LOG"])  # llm 导入时已经读过环境变量，主进程里的调用要在这里改过来
_ENV = ROOT / "autopilot" / "env"  # 可选：接口 key 之类，一行一个 KEY=VALUE（例如 SRH_API_KEY_FLOWAY=...）
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

F = {  # 所有产出文件
    "segments": D / "segments.jsonl", "graded": D / "graded.jsonl", "polish": D / "polish.jsonl", "confirm": D / "confirm.jsonl",
    "reverse_raw": D / "pairs" / "reverse_raw.jsonl", "reverse": D / "pairs" / "reverse.jsonl",
    "keep": D / "pairs" / "keep.jsonl", "keep_fact": D / "pairs" / "keep_fact.jsonl",
    "keep_edit": D / "pairs" / "keep_edit.jsonl", "reverse_check": D / "pairs" / "reverse_check.jsonl", "distill_check": D / "pairs" / "distill_check.jsonl",
    "distill_human": D / "pairs" / "distill_human.jsonl",
    "tasks": D / "pairs" / "tasks.jsonl", "drafts": D / "pairs" / "ai_drafts.jsonl", "distill_ai": D / "pairs" / "distill_ai.jsonl",
    "distill_ai2": D / "pairs" / "distill_ai_v2.jsonl",
    "audit": D / "audit.jsonl", "status": AP / "status.json", "control": AP / "control.json",
    "log": AP / "run.log", "decisions": AP / "decisions.md", "for_user": AP / "for_user.md",
}
_lock = threading.Lock()


def log(msg: str):
    line = f"{dt.datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with _lock, open(F["log"], "a") as f:
        f.write(line + "\n")


def cfg() -> dict:
    c = json.loads(CFG.read_text())
    ctl = json.loads(F["control"].read_text()) if F["control"].exists() else {}
    c["_control"] = ctl
    return c


def status() -> dict:
    return json.loads(F["status"].read_text()) if F["status"].exists() else {}


def save_status(**kw):
    with _lock:
        s = status()
        s.update(kw)
        s["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        F["status"].write_text(json.dumps(s, ensure_ascii=False, indent=1))


def append(path: Path, rows: list[dict]):
    if not rows:
        return
    with _lock, open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_children: set = set()
_stopping = threading.Event()


def _sub(cmd: list[str]) -> subprocess.CompletedProcess:
    """跑子进程并登记。重启自身前要先结束它们，免得旧的 distill 和新进程同时写同一个文件。"""
    if _stopping.is_set():
        raise RuntimeError("正在重启，不再起新子进程")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, start_new_session=True)
    with _lock:
        _children.add(p)
    try:
        out, err = p.communicate()
    finally:
        with _lock:
            _children.discard(p)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _kill_children():
    _stopping.set()
    with _lock:
        ps = list(_children)
    for p in ps:
        try:
            os.killpg(p.pid, 15)  # 连同它起的 agy 一起结束
        except ProcessLookupError:
            pass
    for p in ps:
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(3)
    for p in ps:  # distill 收到 TERM 会马上退，但 agy 不理 TERM，留下来成孤儿（11-05 05:34 留了 3 个跑了 37 分钟）；整组再补一刀
        try:
            os.killpg(p.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage(D).free / 2**30


def prune_agy_logs(days: float):
    """agy 每次调用在 ~/.gemini/antigravity-cli/log 留一个约 40KB 的日志（一夜 5800 个、240MB），只留最近几天。"""
    d = Path.home() / ".gemini" / "antigravity-cli" / "log"
    cut = time.time() - days * 86400
    n = 0
    for f in d.glob("cli-*.log") if d.exists() else []:
        try:
            if f.is_file() and not f.is_symlink() and f.stat().st_mtime < cut:
                f.unlink()
                n += 1
        except OSError:
            pass
    if n:
        log(f"清理 agy 日志：删掉 {days:g} 天前的 {n} 个")


def ids(path: Path, key: str = "id") -> set:
    return {r[key] for r in read_jsonl(path)}


BLOCK_AFTER = 3  # 同一条同一环节连续失败这么多次就不再送


def _block_fails(stage: str) -> Counter:
    return Counter(r["id"] for r in read_jsonl(AP / "blocked_ids.jsonl") if r.get("stage") == stage)


def _blocked(stage: str) -> set:
    """11-16 daily：复审/精修/改动核对/反向补查/反向样本出错时只记日志不记 id，同一条被内容政策拦下的段落
    每 6 分钟重试一次、永远不过（24 小时约 2,200 次调用全失败）。连续失败 BLOCK_AFTER 次的记进
    blocked_ids.jsonl 不再送，做法同 grade_blocked_ids.txt；只是不再重试，不放宽任何判定。"""
    return {k for k, n in _block_fails(stage).items() if n >= BLOCK_AFTER}


def _note_fail(stage: str, rid: str, err: Exception):
    n = _block_fails(stage)[rid] + 1
    append(AP / "blocked_ids.jsonl", [{"stage": stage, "id": rid, "n": n, "err": str(err)[:160],
                                       "t": dt.datetime.now().strftime("%m-%d %H:%M")}])
    if n == BLOCK_AFTER:
        log(f"{stage} {rid} 连续失败 {n} 次，不再送（记入 blocked_ids.jsonl）：{str(err)[:100]}")


# ---------------- 额度 ----------------

_quota_cache: dict = {}


def quota() -> dict:
    if _quota_cache and time.time() - _quota_cache["t"] < 300:
        return _quota_cache["q"]
    if llm.agy_auth_down():  # 鉴权失效熔断中，别再去起 agy 等 OAuth
        raise llm.AgyAuthError("agy 未登录（熔断中），等用户重新登录")
    r = subprocess.run(["agy", "-p", "/quota"], capture_output=True, text=True, timeout=300, cwd="/tmp")
    q = {}
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].startswith("Gemini"):
            k = "weekly" if "Weekly" in parts[1] else "five"
            q[k] = int(parts[2].rstrip("%"))
            q[k + "_reset"] = parts[3]
    if not q:
        if llm.note_agy_auth_fail(r.stdout, r.stderr):
            raise llm.AgyAuthError(f"agy 未登录，等用户重新登录：{r.stderr[-150:]}")
        raise RuntimeError(f"读不到 agy 额度：{r.stdout[-200:]} {r.stderr[-200:]}")
    _quota_cache.update(t=time.time(), q=q)
    # 11-06：额度读数留历史（之前只存最新一条，反推每个环节花多少额度只能靠巡检记录里零星的几个数）
    try:
        with open(AP / "quota.jsonl", "a") as f:
            f.write(json.dumps({"ts": int(time.time()), **q}) + "\n")
    except OSError:
        pass
    return q


def quota_gate(c: dict) -> bool:
    """可以继续调 agy 就返回 True；周额度到下限返回 False；5 小时窗口到下限就睡到重置。"""
    while True:
        q = quota()
        save_status(quota=q)
        if q["weekly"] <= c["quota"]["floor_weekly"]:
            # 11-06：到下限后每个环节每轮都记一行，2 小时攒了 340 行；改成 30 分钟最多记一次
            if time.time() - _bg.get("floor_logged", 0) > 1800:
                _bg["floor_logged"] = time.time()
                log(f"周额度剩 {q['weekly']}%，到了下限 {c['quota']['floor_weekly']}%，暂停所有 agy 环节（{q.get('weekly_reset')} 重置）")
            return False
        if q["five"] > c["quota"]["floor_5h"]:
            return True
        reset = dt.datetime.fromisoformat(q["five_reset"].replace("Z", "+00:00"))
        wait = max(60, (reset - dt.datetime.now(dt.timezone.utc)).total_seconds() + 90)
        log(f"5 小时窗口剩 {q['five']}%，睡 {wait / 60:.0f} 分钟到重置")
        save_status(state=f"等 5 小时窗口重置（{q['five_reset']}）")
        time.sleep(wait)
        _quota_cache.clear()


# ---------------- 收集（后台线程） ----------------

def harvest_source(src: dict):
    out = D / "raw" / f"{src['name']}.jsonl"
    done = out.with_suffix(".done")
    if done.exists():
        return
    have = ids(out)
    n, target = len(have), src.get("max_docs", 2000)
    min_free = cfg().get("disk", {}).get("min_free_gb_harvest", 10)
    if disk_free_gb() < min_free:
        log(f"磁盘只剩 {disk_free_gb():.1f}GB（下限 {min_free}GB），暂不收集 {src['name']}")
        return
    log(f"收集 {src['name']}：已有 {n}，目标 {target}")
    buf = []
    for doc in harvest.HARVESTERS[src["type"]](src):
        if doc["id"] in have or len(doc.get("text") or "") < 150:
            continue
        buf.append(doc)
        n += 1
        if len(buf) >= 50:
            append(out, buf)
            buf = []
            if disk_free_gb() < min_free:
                log(f"磁盘只剩 {disk_free_gb():.1f}GB，{src['name']} 收到 {n} 篇先停（没标完成，空间够了会接着收）")
                return
        if n >= target:
            break
    append(out, buf)
    done.touch()
    log(f"收集 {src['name']} 完成：{n} 篇")


def harvest_thread():
    errors = Counter()
    while True:
        c = cfg()
        # needs_harvest_version：来源要用新的收集功能时写上（同时 enabled 写 false，老进程就跳过），新代码加载后才收
        todo = [s for s in c["sources"] if (s.get("enabled", True) or s.get("needs_harvest_version", 99) <= harvest.VERSION)
                and s["name"] not in c["_control"].get("disabled_sources", []) and not (D / "raw" / f"{s['name']}.done").exists()]
        if not todo:
            time.sleep(600)
            continue
        with ThreadPoolExecutor(min(6, len(todo))) as ex:  # 不同网站并行，同一网站内部限速
            futs = {ex.submit(harvest_source, s): s for s in todo}
            for fu, s in futs.items():
                try:
                    fu.result()
                except Exception:
                    errors[s["name"]] += 1
                    log(f"收集 {s['name']} 出错（第 {errors[s['name']]} 次）：{traceback.format_exc()[-600:]}")
                    if errors[s["name"]] >= 3:
                        (D / "raw" / f"{s['name']}.done").write_text("连续出错 3 次，跳过")
                        save_status(harvest_errors=dict(errors))
        time.sleep(60)


# ---------------- 切段 ----------------

def stage_segment(c: dict):
    segs = read_jsonl(F["segments"])
    # 停用的旧来源（config segment.retired，比如换了收集代码重收的 wangdoc → wangdoc2）：没分级或判 unusable 的段落不参与去重，
    # 不然重收的同一篇文章开头 80 字没变就被当重复扔掉；已分级能用的照旧参与，免得同一段进两次训练集
    retired = set(c["segment"].get("retired", []))
    usable = {r["id"] for r in read_jsonl(F["graded"]) if r["source"] in retired and r.get("tier") != "unusable"} if retired else set()
    # 11-05 13:43：旧来源同时停用（control 的 disabled_sources，如 wangdoc/wangdoc2 → wangdoc3）或后加的切段规则不过的，已分级的也不再往下游送，
    # 就不该再占着去重的位置；已经进了配对的照旧参与，免得同一段进两次训练集
    if usable:
        off = set(c.get("_control", {}).get("disabled_sources", []))
        paired = {"#".join(p["id"].split("#")[:2]) for p in _all_pairs()}
        by_id = {s["id"]: s for s in segs if s["id"] in usable}
        usable = {i for i in usable if i in paired or (i in by_id and by_id[i]["source"] not in off and segment.late_check_seg(by_id[i]) is None)}
    seen = {segment.dedupe_key(s["text"]) for s in segs if s["source"] not in retired or s["id"] in usable}
    done_docs = {s["doc"] for s in segs}
    seen_docs_file = AP / "segmented_docs.txt"
    done_docs |= set(seen_docs_file.read_text().split()) if seen_docs_file.exists() else set()
    why, new, new_docs = Counter(), [], []
    for raw in sorted((D / "raw").glob("*.jsonl")):
        for doc in read_jsonl(raw):
            if doc["id"] in done_docs:
                continue
            new_docs.append(doc["id"])
            w = {}
            new += segment.segment_doc(doc, per_doc=c["segment"]["per_doc"], seen=seen, why=w)
            why.update(w)
    append(F["segments"], new)
    with open(seen_docs_file, "a") as f:
        f.write("".join(d + "\n" for d in new_docs))
    if new_docs:
        st = status().get("segment_reject", {})
        for k, v in why.items():
            st[k] = st.get(k, 0) + v
        save_status(segment_reject=st)
        log(f"切段：{len(new_docs)} 篇新文档 → {len(new)} 段")


# ---------------- 分级 ----------------

def _grade_batch(model: str, batch: list[dict]) -> list[dict]:
    seg_txt = "\n\n".join(f"【第 {i} 段】\n{s['text']}" for i, s in enumerate(batch, 1))
    j = llm.ask_json(model, prompts.GRADE_BATCH_PROMPT.format(n=len(batch), principles=prompts.principles(),
                                                             genre_names=genres.names(), segments=seg_txt), tag="grade")
    items = {int(x.get("i", 0)): x for x in j.get("items", [])}
    if len(items) != len(batch):
        raise ValueError(f"分级条数不对：{len(items)}/{len(batch)}")
    return [{"id": s["id"], "source": s["source"], **{k: items[i].get(k) for k in ("genre", "tier", "ai_likelihood", "modern", "standalone", "flaws")}}
            for i, s in enumerate(batch, 1)]


def stage_grade(c: dict):
    g = c["grade"]
    graded = read_jsonl(F["graded"])
    done = {r["id"] for r in graded}
    off = set(c["_control"].get("disabled_sources", []))
    # 单一题材的大来源设分级总数上限（11-05 复查：thucnews 全是体育、people_daily 全是 1995–96），免得占满分级额度
    room = {k: v - sum(1 for r in graded if r["source"] == k) for k, v in g.get("source_caps", {}).items()}
    bf = AP / "grade_blocked_ids.txt"  # 单独送也被 Google 内容政策拦下的段落，不再重试
    done |= set(bf.read_text().split()) if bf.exists() else set()
    todo = [s for s in read_jsonl(F["segments"]) if s["id"] not in done and s["source"] not in off
            and room.get(s["source"], 1) > 0 and segment.late_check_seg(s) is None]
    if not todo:
        return
    by_src = defaultdict(list)  # 各来源轮流，新来源也能尽快拿到分级结果
    # 来源内按 id 的哈希排（固定顺序，每轮接着上轮），不按文件顺序：11-05 复查时 people_daily_span 分了 238 段全是 1998 年
    # （9 个年份文件里的第一个），thucnews_mix 的 324 段全是排在最前的体育。打散之后分级结果一开始就能代表整个来源。
    todo.sort(key=lambda s: hashlib.md5(s["id"].encode()).hexdigest())
    for s in todo:
        by_src[s["source"]].append(s)
    order = []
    while any(by_src.values()) and len(order) < g["per_cycle"]:
        for k in list(by_src):
            if by_src[k] and room.get(k, 1) > 0:
                order.append(by_src[k].pop(0))
                if k in room:
                    room[k] -= 1
            elif by_src[k]:
                by_src[k] = []
    batches = [order[i:i + g["batch"]] for i in range(0, len(order), g["batch"])]
    for i in range(0, len(batches), g["workers"] * 3):
        if not quota_gate(c):
            return
        chunk = batches[i:i + g["workers"] * 3]
        singles = []
        with ThreadPoolExecutor(g["workers"]) as ex:
            for b, fu in [(b, ex.submit(_grade_batch, g["model"], b)) for b in chunk]:
                try:
                    append(F["graded"], fu.result())
                except Exception as e:
                    if "Prohibited Use" in str(e):  # 11-05：一段触发内容政策，整批 6 段每轮都失败；拆开单独送
                        singles += [[s] for s in b]
                    else:
                        log(f"分级一批失败：{str(e)[:200]}")
        if singles:
            blocked = []
            with ThreadPoolExecutor(g["workers"]) as ex:
                for b, fu in [(b, ex.submit(_grade_batch, g["model"], b)) for b in singles]:
                    try:
                        append(F["graded"], fu.result())
                    except Exception as e:
                        if "Prohibited Use" in str(e):
                            blocked.append(b[0]["id"])
            with _lock, open(bf, "a") as f:
                f.writelines(x + "\n" for x in blocked)
            log(f"分级：有批次被内容政策拦下，拆开单独送 {len(singles)} 段，{len(blocked)} 段单独也被拦（记入 grade_blocked_ids.txt，不再送）")
    log(f"分级：本轮 {len(order)} 段")


def _ai(x) -> float:
    """ai_likelihood 转成数；缺失或不是数时按 1 算。
    11-05 修：之前写成 float(x or 1)，0.0 也变成了 1——判为"完全不像 AI"的段落（分级里 1061/2072）进不了精修、复审、打磨，
    复审给 0.0 的一律被拒（88 条全拒）。"""
    try:
        return float(x) if x is not None and x != "" else 1.0
    except (TypeError, ValueError):
        return 1.0


def _major(flaws) -> list:
    return [f for f in (flaws or []) if not isinstance(f, dict) or f.get("severity") != "minor"]  # 没标轻重的按 major 算


def _rc_rows() -> list[dict]:
    """因 _ai 的 bug 被误拒的复审记录：复审给了 0.0、没有 major 问题、当时没存 verdict（11-05 之前的记录），
    且这一段还没有任何通过的复审记录。每段只取最早的一条，带后缀 @rc 原样重审一次（当时的 verdict 没存，不能直接改判）。"""
    rows = read_jsonl(F["confirm"])
    passed = {r["id"].split("@")[0] for r in rows if r["ok"]}
    done = {r["id"] for r in rows}
    out, seen = [], set()
    for r in rows:
        sid = r["id"].split("@")[0]
        if (r["ok"] or "verdict" in r or r["id"].endswith("@rc") or sid in passed or sid in seen
                or r.get("ai_likelihood") is None or _ai(r.get("ai_likelihood")) != 0 or _major(r.get("flaws"))):
            continue
        seen.add(sid)
        if r["id"] + "@rc" not in done:
            out.append(r)
    return out


def _good_enough(r: dict, g: dict) -> bool:
    return (_ai(r.get("ai_likelihood")) <= g["ai_max"] and int(r.get("standalone") or 0) >= 4
            and r.get("genre") in genres.GENRES and (r.get("modern") or r["genre"] in g["old_ok_genres"]))


def _flaw_line(x) -> str:
    """复审意见可能是字符串（旧格式）或 {severity, text}（11-05 起）。"""
    if isinstance(x, dict):
        return f"- {'【必须改】' if x.get('severity') != 'minor' else '【可改可不改】'}{x.get('text', '')}"
    return f"- {x}"


def stage_polish(c: dict):
    """精修：初评为 good 的人写段落，只改审稿人指出的小毛病和错别字，其他一字不动；改动量和事实都要过规则，之后照样复审。"""
    g = c["grade"]
    off = set(c["_control"].get("disabled_sources", []))
    seg = {s["id"]: s for s in read_jsonl(F["segments"]) if s["source"] not in off and segment.late_check_seg(s) is None}
    done = ids(F["polish"])
    blk = _blocked("polish")
    cands = [{**r, "base": seg[r["id"]]["text"]} for r in read_jsonl(F["graded"]) if r["id"] not in done and r.get("tier") == "good"
             and r["id"] in seg and _good_enough(r, g) and r.get("modern") and r["id"] not in blk]
    # 第二轮：复审没过的（不论是原文还是精修稿），带着复审意见再精修一次，再复审一次；只给这一次机会
    rc = {r["id"].split("@")[0] for r in _rc_rows()}  # 误拒的先原样重审，不进第二轮精修
    cands += [{**r, "id": r["id"] + "@2", "base": r["text"]} for r in read_jsonl(F["confirm"])
              if not r["ok"] and "@" not in r["id"] and r["id"] + "@2" not in done and r["id"] in seg and r.get("modern")
              and r["id"] not in rc and r["id"] not in blk]
    cands = cands[: g["confirm_per_cycle"]]
    if not cands:
        return
    from verify_pairs import overlap

    def one(r):
        orig = seg[r["id"].split("@")[0]]["text"]
        flaws = "\n".join(_flaw_line(x) for x in (r.get("flaws") or [])) or "- （无具体意见，只查错别字和病句）"
        fixed = llm.chat(c["pairs"]["teacher"], [{"role": "user", "content": prompts.MINOR_FIX_PROMPT.format(
            genre_name=genres.GENRES[r["genre"]]["name"], flaws=flaws, text=r["base"])}], tag="polish").strip()
        cmp = facts.compare(orig, fixed)
        keep = overlap(orig, fixed)
        ratio = len(fixed) / max(len(orig), 1)
        new = facts.added_clauses(orig, fixed)  # 补内容、整句改写（复审只看精修稿，看不出和原文的差别，只能在这里拦）
        ok = cmp["ok"] and keep >= 0.75 and 0.85 <= ratio <= 1.1 and not new
        return {"id": r["id"], "source": r["source"], "genre": r["genre"], "modern": r.get("modern"), "orig": orig, "text": fixed,
                "ok": ok, "why": None if ok else f"规则 {cmp['hard'][:2]} 保留 {keep:.2f} 长度比 {ratio:.2f} 新句 {new[:2]}"}

    for i in range(0, len(cands), g["workers"] * 3):
        if not quota_gate(c):
            return
        with ThreadPoolExecutor(g["workers"]) as ex:
            for fu, r in [(ex.submit(one, r), r) for r in cands[i:i + g["workers"] * 3]]:
                try:
                    append(F["polish"], [fu.result()])
                except Exception as e:
                    log(f"精修失败：{str(e)[:200]}")
                    _note_fail("polish", r["id"], e)
    log(f"精修：{len(cands)} 段")


def stage_confirm(c: dict):
    g = c["grade"]
    off = set(c["_control"].get("disabled_sources", []))
    seg = {s["id"]: s for s in read_jsonl(F["segments"]) if s["source"] not in off and segment.late_check_seg(s) is None}
    done = ids(F["confirm"])
    blk = _blocked("confirm")
    cands = [{**r, "text": seg[r["id"]]["text"]} for r in read_jsonl(F["graded"])
             if r["id"] not in done and r.get("tier") == "exemplar" and r["id"] in seg and _good_enough(r, g)]
    cands += [r for r in read_jsonl(F["polish"]) if r["ok"] and r["id"] not in done and r["id"].split("@")[0] in seg]
    cands = [{**r, "id": r["id"] + "@rc"} for r in _rc_rows() if r["id"].split("@")[0] in seg] + cands
    cands = [r for r in cands if r["id"] not in blk]
    cands = cands[: g["confirm_per_cycle"]]
    if not cands:
        return

    def one(r):
        j = llm.ask_json(g["confirm_model"], prompts.CONFIRM_PROMPT.format(
            principles=prompts.principles(), genre_name=genres.GENRES[r["genre"]]["name"], text=r["text"]), tag="confirm")
        ok = j.get("verdict") == "pass" and not _major(j.get("flaws")) and _ai(j.get("ai_likelihood")) <= g["ai_max"]
        return {"id": r["id"], "source": r["source"], "genre": r["genre"], "modern": r.get("modern"), "ok": ok,
                "polished": "orig" in r or bool(r.get("polished")), "text": r["text"], "flaws": j.get("flaws"),
                "ai_likelihood": j.get("ai_likelihood"), "verdict": j.get("verdict")}

    for i in range(0, len(cands), g["workers"] * 3):
        if not quota_gate(c):
            return
        with ThreadPoolExecutor(g["workers"]) as ex:
            for fu, r in [(ex.submit(one, r), r) for r in cands[i:i + g["workers"] * 3]]:
                try:
                    append(F["confirm"], [fu.result()])
                except Exception as e:
                    log(f"复审失败：{str(e)[:200]}")
                    _note_fail("confirm", r["id"], e)
    log(f"复审：{len(cands)} 段")


def exemplars(c: dict) -> list[dict]:
    seg = {s["id"]: s for s in read_jsonl(F["segments"])}
    out, seen = [], set()
    for r in read_jsonl(F["confirm"]):
        sid = r["id"].split("@")[0]  # 第二轮精修的记录 id 带 @2，误拒重审的带 @rc
        if (r["ok"] and sid in seg and sid not in seen and r["source"] not in c["_control"].get("disabled_sources", [])
                and segment.late_check_seg(seg[sid]) is None  # 后加的切段规则对已通过的范本也生效
                and not facts.added_clauses(seg[sid]["text"], r.get("text") or seg[sid]["text"])):  # 精修时补了内容或整句改写的（11-05 加）
            seen.add(sid)
            out.append({**seg[sid], "text": r.get("text") or seg[sid]["text"], "orig_text": seg[sid]["text"],
                        "genre": r["genre"], "modern": r.get("modern")})
    return out


# ---------------- 配对 ----------------

def _plain_reverse(rows: list[dict]) -> list[dict]:
    """普通反向样本：id 不以 #revai 结尾。AI 腔反向样本（#revai）单独计数，不占普通反向 2,000 的目标和文体配额。"""
    return [r for r in rows if not str(r.get("id", "")).endswith("#revai")]


def _balanced(rows: list[dict], have: Counter, total: int, limit: int) -> list[dict]:
    """按文体配额挑：离配额差得最多的文体先来；旧白话（modern=false）最多占该文体的 old_share。"""
    pool = defaultdict(list)
    for r in rows:
        pool[r["genre"]].append(r)
    for g in pool:
        random.Random(g).shuffle(pool[g])
    out = []
    while len(out) < limit and any(pool.values()):
        def deficit(g):
            return have[g] / max(genres.GENRES[g]["quota"] * total, 1)
        g = min((g for g in pool if pool[g]), key=deficit)
        out.append(pool[g].pop())
        have[g] += 1
    return out


ASSIST_OPENER = re.compile(r"^(?:好的|当然(?:可以)?|没问题|以下是|下面是)[^\n]{0,40}")


def rule_fail(orig: str, d: str) -> str | None:
    """反向样本的规则核对（11-10 从 stage_reverse_keep 里提出来，AI 腔反向样本共用，行为不变）。"""
    if ASSIST_OPENER.match(d):
        return "规则 助手腔开场白"
    cmp = facts.compare(orig, d)
    ratio = len(d) / max(len(orig), 1)
    cp = overlap(orig, d)
    if not cmp["ok"] or not 0.8 <= ratio <= 2.0 or cp > 0.45 or "UNUSABLE" in d[:20]:
        return f"规则 {cmp['hard'][:2]} 长度比 {ratio:.2f} 照抄 {cp:.2f}"
    return None

# DeepSeek 批量攒的 AI 一侧候选（scripts/bulk_deepseek.py，data/v3/extra/），反向样本先用这些。渠道分开记：
# 11-05 的临时免费渠道和 11-06 起自己网关上的 DeepSeek。两个标签（模型@主机名）要和数据记录里的一致，
# 放在 autopilot/env（SRH_DS_POOL_LABEL、SRH_DS_GATEWAY_LABEL），不写进代码
DS_POOL_LABEL = llm._env_file_key("SRH_DS_POOL_LABEL") or "deepseek-v4.1-flash@free-channel"
DS_FLOWAY_LABEL = llm._env_file_key("SRH_DS_GATEWAY_LABEL") or "deepseek-v4.1-flash@gateway"
DS_POOL = [("restyle", "ds_moxuna_restyle.jsonl", DS_POOL_LABEL), ("restyle", "ds_floway_restyle.jsonl", DS_FLOWAY_LABEL),
           ("polish", "ds_moxuna_polish.jsonl", DS_POOL_LABEL), ("degrade", "ds_moxuna_degrade.jsonl", DS_POOL_LABEL)]


def _ds_pool() -> dict:
    pool = defaultdict(list)
    for job, name, label in DS_POOL:
        p = D / "extra" / name
        if p.exists():
            for r in read_jsonl(p):
                if r.get("draft"):
                    pool[r["src_id"]].append((job, r["draft"], label))
    return pool


def stage_reverse_keep(c: dict):
    pc = c["pairs"]
    ex = exemplars(c)
    # 民国白话（modern=false，维基文库的朱自清、老舍等）只做保留样本：当反向样本的答案，模型会学着把当代写法改回"著""甚么"；
    # 当保留样本，教的是遇到这类文字不要乱动，正合"作者语气不变"（11-05 来源复查）
    # 11-06 用户定：保留样本一律原文→原文。原文里有真错（改动核对找到 fix）的不能当答案，这些范本不做保留样本，
    # 改去做反向样本（答案是修好的范本）；民国白话只做保留样本
    err_ids = {r["id"].removesuffix("#keep") for r in read_jsonl(F["keep_edit"]) if (r.get("n_fix") or 0) > 0}
    # 11-10 用户定：同一个范本可以既做反向样本又做保留样本。原来按 crc32 把 keep_share（15%）的范本划给保留、不做反向，
    # 保留样本停在 248 条（目标 700）没有新候选，稀缺文体的范本又有 15% 不能做反向。现在保留样本从全部范本里挑，
    # 每个文体封顶在 配额 × 保留目标，差得多的文体先来；反向只排除民国白话。同一范本的 #rev 和 #keep 按 doc 落在同一边。
    no_rev = {e["id"] for e in ex if e.get("modern") is False}
    # 11-06 修：keep.jsonl 里的 id 带 #keep 后缀，之前直接拿去和范本 id 比，永远对不上，每轮把全部保留样本重写一遍
    # （到 11-06 00:50 共 1167 行、182 个不同 id）。all_pairs 按 id 去重，训练快照没受影响；旧的重复行按规矩不删。
    have_keep = {i.removesuffix("#keep") for i in ids(F["keep"])}
    kt = pc["targets"]["keep"]
    gk = Counter(e["genre"] for e in ex if e["id"] in have_keep - err_ids)
    room = {g: max(0, math.ceil(gi["quota"] * kt) - gk[g]) for g, gi in genres.GENRES.items()}
    cands = []
    for g in room:
        pool_g = [e for e in ex if e["genre"] == g and e["id"] not in have_keep and e["id"] not in err_ids]
        random.Random(f"keep:{g}").shuffle(pool_g)
        cands += pool_g[:room[g]]
    keep_new = _balanced(cands, Counter(gk), kt, max(0, kt - len(have_keep - err_ids)))
    # 输入是精修前的原文（可能带错别字），答案是精修后的版本；all_pairs 里改成原文→原文（见 _all_pairs_filtered）
    rows = [{"id": e["id"] + "#keep", "doc": e["doc"], "kind": "keep", "genre": e["genre"], "draft": e["orig_text"], "original": e["text"],
             "source": e["source"]} for e in keep_new]
    append(F["keep"], rows)
    _keep_edit_check(c)
    _reverse_recheck(c)

    raw = read_jsonl(F["reverse_raw"])
    by_src = defaultdict(list)
    for r in raw:
        by_src[r["src_id"]].append(r)
    acc = read_jsonl(F["reverse"])
    # 11-07 用户定（主会话）：总目标 4,000 → 2,000，每个文体封顶在 配额 × reverse_genre_basis（原来的 4,000）。
    # 只数还有效的（补查、抽检不过和 exclude_ids 里的不算），不然被 V3 拒掉的旧样本也占着名额
    invalid = _rev_invalid()
    acc_valid = _plain_reverse([r for r in acc if r["id"] not in invalid])
    if len(acc_valid) >= pc["targets"]["reverse"]:
        return
    dms = pc.get("degrade_models") or [pc["teacher"]]
    pool = _ds_pool()

    def plan(e):
        """下一次怎么做：("pool", 任务名, 草稿) / ("model", 模型) / None（不再做）。
        11-06 改：管线里 DeepSeek 改坏只有 17% 过规则（多半照抄过多，Gemini 是 75%），原来重试还用同一个模型，
        分给 DeepSeek 的范本大多白白作废。现在分给 DeepSeek 的先用免费渠道批量攒的候选（restyle/polish/degrade，
        要先过同样的规则），没有再现调一次 DeepSeek，还不行换 Gemini 改坏一次。分给 Gemini 的照旧（规则拒了重试一次）。
        已经成功的、被事实裁判拒过的不再做；每段最多三次。"""
        # 第一轮（#rev）；第二轮见 plan2。#revai 是 AI 腔反向样本，不算进普通反向的尝试次数（11-10）
        rs = [r for r in by_src.get(e["id"], []) if not r["id"].endswith(("#rev2", "#revai"))]
        if any(r["ok"] or not str(r.get("why") or "").startswith("规则") for r in rs) or len(rs) >= 3:
            return None
        ds_tries = sum(1 for r in rs if "deepseek" in str(r.get("degrader") or "") and not r.get("draft_src"))
        gem_tries = sum(1 for r in rs if "deepseek" not in str(r.get("degrader") or ""))
        dm = dms[zlib.crc32(e["id"].encode()) % len(dms)]
        if "deepseek" not in dm:
            return ("model", dm) if gem_tries < 2 else None
        if not any(r.get("draft_src") for r in rs):
            for job, d, label in pool.get(e["id"], []):
                if rule_fail(e["text"], d) is None:
                    return ("pool", job, d, label)
        if ds_tries == 0:
            return ("model", dm)
        gem = next((m for m in dms if "deepseek" not in m), pc["teacher"])
        return ("model", gem) if gem_tries == 0 else None

    second = set(pc.get("reverse_second_genres", []))
    gem2 = next((m for m in dms if "deepseek" not in m), pc["teacher"])

    def plan2(e):
        """11-07 第二轮（id 后缀 #rev2，只用 Gemini 按 DEGRADE_PROMPT 改坏）：范本稀缺，第一轮做完却没留下有效样本的
        （裁判拒、规则拒满次数、补查或抽检不过）再改坏一次；reverse_second_genres 里的稀缺文体，第一轮成功了也再做一版，
        提示词里附上第一版，要求换一种改法。第二轮规则拒了重试一次，裁判拒了就不再做。"""
        if plan(e) is not None:
            return None  # 第一轮还没做完
        rs1 = [r for r in by_src.get(e["id"], []) if not r["id"].endswith(("#rev2", "#revai"))]
        rs2 = [r for r in by_src.get(e["id"], []) if r["id"].endswith("#rev2")]
        if not rs1 or any(r["ok"] or not str(r.get("why") or "").startswith("规则") for r in rs2) or len(rs2) >= 2:
            return None
        v1 = [r for r in rs1 if r["ok"] and r["id"] not in invalid]
        if v1 and e["genre"] not in second:
            return None
        return ("model2", gem2, v1[0]["draft"] if v1 else None)

    plans = {}
    rev_blk = _blocked("reverse")
    for e in ex:
        if e["id"] in rev_blk:
            continue
        if e["id"] not in no_rev or e["id"] in err_ids and e.get("modern") is not False:
            p = plan(e) or plan2(e)
            if p:
                plans[e["id"]] = p
    basis = pc.get("reverse_genre_basis", pc["targets"]["reverse"])
    have = Counter(r["genre"] for r in acc_valid)
    todo = [e for e in ex if e["id"] in plans and have[e["genre"]] < genres.GENRES[e["genre"]]["quota"] * basis]
    todo = _balanced(todo, Counter(have), basis, min(pc["per_cycle"], pc["targets"]["reverse"] - len(acc_valid)))
    if not todo:
        return

    def one(e):
        gg = genres.GENRES[e["genre"]]
        p = plans[e["id"]]
        rec = {"id": e["id"] + ("#rev2" if p[0] == "model2" else "#rev"), "src_id": e["id"], "doc": e["doc"], "kind": "reverse",
               "genre": e["genre"], "source": e["source"], "original": e["text"]}
        if p[0] == "model2":
            rec["round"] = 2
        if p[0] == "pool":
            d = p[2]
            rec.update(draft=d, degrader=p[3], draft_src=f"pool:{p[1]}")
        elif "deepseek" in p[1]:
            # 11-06 用户定：现调 DeepSeek 改成普通用户的改写请求（prompts.RESTYLE_PROMPTS），规则通过率 17% → 约 28%。
            # 不用 2“改正式”：规则通过率最低（20%），裁判也只过 6/10
            v = (0, 1, 3)[zlib.crc32(e["id"].encode()) % 3]
            raw = llm.chat(p[1], [{"role": "user", "content": prompts.RESTYLE_PROMPTS[v].format(text=e["text"])}],
                           temperature=0.8, tag="degrade").strip()
            d = prompts.clean_reply(raw)
            rec.update(draft=d, degrader=llm.label(p[1]), draft_prompt=f"restyle{v}")
        else:
            # 11-05 起改坏这一步在 degrade_models 里按段落 id 轮流选（Gemini Argon + DeepSeek），AI 腔不只来自一家
            q = prompts.DEGRADE_PROMPT.format(genre_name=gg["name"], faults=gg["faults"], text=e["text"])
            if p[0] == "model2" and p[2]:
                q += prompts.DEGRADE_AGAIN.format(prev=p[2])
            d = llm.chat(p[1], [{"role": "user", "content": q}], tag="degrade").strip()
            rec.update(draft=d, degrader=llm.label(p[1]))
        why = rule_fail(e["text"], d)
        if why:
            return {**rec, "ok": False, "why": why}
        ok, why = _reverse_verdict(llm.ask_json(pc["judge"], prompts.REVERSE_JUDGE.format(a=e["text"], b=d), tag="judge"))
        return {**rec, "ok": ok, "why": why, "judge_v": REVERSE_JUDGE_V}

    for i in range(0, len(todo), pc["workers"] * 3):
        if not quota_gate(c):
            return
        with ThreadPoolExecutor(pc["workers"]) as ex_:
            for fu, e in [(ex_.submit(one, e), e) for e in todo[i:i + pc["workers"] * 3]]:
                try:
                    r = fu.result()
                    append(F["reverse_raw"], [r])
                    if r["ok"]:
                        append(F["reverse"], [{k: r[k] for k in ("id", "doc", "kind", "genre", "source", "draft", "original")}])
                except Exception as err:
                    log(f"反向样本失败：{str(err)[:200]}")
                    _note_fail("reverse", e["id"], err)
    log(f"反向样本：本轮处理 {len(todo)} 段")


def _ai_family(src_id: str, genre: str, attempt: int = 0) -> str:
    """AI 腔反向样本的 family：按 id 的 crc32 在文体推荐的几类里挑；第二次尝试换下一个。"""
    fams = prompts.AI_STYLE_BY_GENRE.get(genre) or list(prompts.AI_STYLE_FAMILIES)
    return fams[(zlib.crc32((src_id + "#revai").encode()) + attempt) % len(fams)]


def _reverse_ai_pick(c: dict) -> dict:
    """AI 腔反向样本（#revai）这一轮挑哪些，不调模型。返回 todo 和几个数：
    已尝试条数（试点按它到量就停）、有效条数（到 target 停）、本轮上限。"""
    pc = c["pairs"]
    ra = pc.get("reverse_ai") or {}
    info = {"todo": [], "enabled": bool(ra.get("enabled")), "pilot_done": False, "tries": 0, "valid": 0, "limit": 0,
            "have": Counter()}
    if not ra.get("enabled"):
        return info
    ai_raw = [r for r in read_jsonl(F["reverse_raw"]) if str(r.get("id", "")).endswith("#revai")]
    pilot = int(ra.get("pilot", 0))
    info["tries"] = len(ai_raw)
    if not ra.get("pilot_ok", False) and pilot > 0 and len(ai_raw) >= pilot:
        info["pilot_done"] = True
        return info
    target = int(ra.get("target", 0))
    invalid = _rev_invalid()
    acc_ai = [r for r in read_jsonl(F["reverse"]) if str(r["id"]).endswith("#revai") and r["id"] not in invalid]
    info["valid"] = len(acc_ai)
    if len(acc_ai) >= target:
        return info
    by_src = defaultdict(list)
    for r in ai_raw:
        by_src[r.get("src_id")].append(r)
    cands = []
    for e in exemplars(c):
        if e.get("modern") is False:  # 民国白话不做反向（当答案会教模型把当代写法改回去）
            continue
        rs = by_src.get(e["id"], [])
        # 已经 ok、被裁判拒过、或试满 2 次的不再做；规则拒了可以再试一次
        if any(r.get("ok") or not str(r.get("why") or "").startswith("规则") for r in rs) or len(rs) >= 2:
            continue
        cands.append(e)
    have = Counter(r["genre"] for r in acc_ai)
    cap = {g: math.ceil(gi["quota"] * target) for g, gi in genres.GENRES.items()}  # 每个文体封顶 配额 × 目标
    capped = [e for e in cands if have[e["genre"]] < cap[e["genre"]]]
    limit = min(int(ra.get("per_cycle", pc["per_cycle"])), target - len(acc_ai))
    if not ra.get("pilot_ok", False) and pilot:
        limit = min(limit, pilot - len(ai_raw))  # 试点：尝试总数到 pilot 就停，不超做
    info["limit"] = max(0, limit)
    info["have"] = have
    info["todo"] = _balanced(capped, Counter(have), target, max(0, limit))
    return info


def stage_reverse_ai(c: dict):
    """11-10 冒烟训练后加（主会话设计）：反向样本原来只有 DEGRADE_PROMPT（“平庸第一稿”），模型学会了修这种稿子，
    却认不出真实 AI 的腔调。这里拿人写范本让 Gemini 按一类 AI 腔改写（prompts.AI_STYLE_PROMPT），答案仍是人写原文，
    再过 REVERSE_JUDGE；通过的进反向线（id 后缀 #revai，draft 是 AI 腔版本）。普通反向样本的计数不数它们。"""
    pc = c["pairs"]
    ra = pc.get("reverse_ai") or {}
    info = _reverse_ai_pick(c)
    if not info["enabled"]:
        return
    if info["pilot_done"]:
        if not _bg.get("revai_pilot_logged"):
            _bg["revai_pilot_logged"] = True
            log(f"AI 腔反向样本试点 {int(ra.get('pilot', 0))} 条已做完，等主会话复核（config pairs.reverse_ai.pilot_ok）")
        return
    todo = info["todo"]
    if not todo:
        return
    by_src = defaultdict(list)
    for r in read_jsonl(F["reverse_raw"]):
        if str(r.get("id", "")).endswith("#revai"):
            by_src[r.get("src_id")].append(r)

    def one(e):
        gg = genres.GENRES[e["genre"]]
        family = _ai_family(e["id"], e["genre"], len(by_src.get(e["id"], [])))
        q = prompts.AI_STYLE_PROMPT.format(genre_name=gg["name"], family=prompts.AI_STYLE_FAMILIES[family], text=e["text"])
        d = prompts.clean_reply(llm.chat(pc["teacher"], [{"role": "user", "content": q}], tag="degrade_ai"))
        rec = {"id": e["id"] + "#revai", "src_id": e["id"], "doc": e["doc"], "kind": "reverse", "genre": e["genre"],
               "source": e["source"], "original": e["text"], "draft": d, "degrader": llm.label(pc["teacher"]),
               "ai_family": family, "round": "ai"}
        why = rule_fail(e["text"], d)
        if why:
            return {**rec, "ok": False, "why": why}
        ok, why = _reverse_verdict(llm.ask_json(pc["judge"], prompts.REVERSE_JUDGE.format(a=e["text"], b=d), tag="judge"))
        return {**rec, "ok": ok, "why": why, "judge_v": REVERSE_JUDGE_V}

    for i in range(0, len(todo), pc["workers"] * 3):
        if not quota_gate(c):
            return
        with ThreadPoolExecutor(pc["workers"]) as ex_:
            for fu in [ex_.submit(one, e) for e in todo[i:i + pc["workers"] * 3]]:
                try:
                    r = fu.result()
                    append(F["reverse_raw"], [r])
                    if r["ok"]:
                        append(F["reverse"], [{"id": r["id"], "doc": r["doc"], "kind": "reverse", "genre": r["genre"],
                                                "source": r["source"], "draft": r["draft"], "original": r["original"],
                                                "variant": "ai", "ai_family": r["ai_family"]}])
                except Exception as e_:
                    log(f"AI 腔反向样本失败：{str(e_)[:200]}")
    log(f"AI 腔反向样本：本轮处理 {len(todo)} 段")


REVERSE_JUDGE_V = 3


def _nonempty(v) -> bool:
    if isinstance(v, list):
        return any(str(x).strip() for x in v)
    return bool(str(v or "").strip())


def _reverse_verdict(j: dict) -> tuple[bool, str | None]:
    """反向样本裁判（prompts.REVERSE_JUDGE）。11-06 audit_fail 起：除了信息不等价，B（改坏稿）把 A（范本）的硬伤改好了也拒——
    训练时 B 是输入、A 是答案，模型会学着把对的改回错的。旧裁判（FACT_JUDGE）放过 98%，抽检却有 12% 不合格。
    11-07 audit_fail 起（V3）：B 加的限定和细节、B 换了语体、B 有比 A 好的地方、B 的改动多半只是同义替换（changes 不是 worse），都拒。"""
    bad = {k: j.get(k) for k in ("missing", "added", "distorted", "a_worse", "register_shift", "b_better") if _nonempty(j.get(k))}
    if str(j.get("changes") or "").strip().lower() != "worse":
        bad["changes"] = j.get("changes")
    ok = j.get("verdict") == "pass" and not bad
    return ok, None if ok else f"裁判 {json.dumps(bad or {'verdict': j.get('verdict')}, ensure_ascii=False)[:300]}"


def _keep_verdict(changes) -> dict:
    """KEEP_EDIT_JUDGE 的逐处结论 → pair（照用：原文→精修稿）/ orig（精修没修掉真错、还丢了东西：改成原文→原文）/ drop。"""
    if not isinstance(changes, list):
        raise ValueError(f"changes 不是列表：{str(changes)[:100]}")
    n = Counter(str(x.get("type", "")).strip().lower() if isinstance(x, dict) else "" for x in changes)
    n_fix, n_ok = n.get("fix", 0), n.get("ok", 0)
    n_loss = len(changes) - n_fix - n_ok  # 认不出的类型按 loss 算
    verdict = "pair" if n_loss == 0 else ("orig" if n_fix == 0 else "drop")
    return {"verdict": verdict, "n_fix": n_fix, "n_ok": n_ok, "n_loss": n_loss}


def _keep_edit_check(c: dict):
    """保留样本的输入是精修前的原文、答案是精修稿；精修改了内容或语气就等于教模型改好文字。
    11-05 起用 FACT_JUDGE 核对（keep_fact.jsonl），但它只看事实；11-06 抽检保留样本 21 条 4 条不合格，都是精修照分级意见
    抹掉了作者的修饰和分寸（"并肩相依"→"相连"、删"完全""赤裸裸地"、"代表团"→"一行"），FACT_JUDGE 全放过。
    改成 KEEP_EDIT_JUDGE 逐处归类（fix / ok / loss），结论记在 keep_edit.jsonl：没有 loss 照用；有 loss 但没修掉任何真错，
    说明原文本来就好，改成原文→原文的保留样本；既有 loss 又有真错的扔掉。旧的 keep_fact 记录只用来否决（不再新做）。
    11-06 用户定保留样本一律原文→原文后，这一步只用来找原文里的真错（fix）：有真错的范本改去做反向样本。"""
    pc = c["pairs"]
    checked = ids(F["keep_edit"])
    blk = _blocked("keep_edit")
    todo, seen = [], set()
    for r in read_jsonl(F["keep"]):
        if r["draft"] != r["original"] and r["id"] not in checked and r["id"] not in seen and r["id"] not in blk:
            seen.add(r["id"])
            todo.append(r)
    if not todo:
        return

    def one(r):
        j = llm.ask_json(pc["judge"], prompts.KEEP_EDIT_JUDGE.format(
            genre_name=genres.GENRES.get(r["genre"], {}).get("name", r["genre"]), a=r["draft"], b=r["original"]), tag="keep_edit")
        return {"id": r["id"], **_keep_verdict(j.get("changes")), "changes": j.get("changes")}

    n = Counter()
    for i in range(0, len(todo), pc["workers"] * 3):
        if not quota_gate(c):
            break
        with ThreadPoolExecutor(pc["workers"]) as ex:
            for fu, r in [(ex.submit(one, r), r) for r in todo[i:i + pc["workers"] * 3]]:
                try:
                    rec = fu.result()
                    append(F["keep_edit"], [rec])
                    n[rec["verdict"]] += 1
                except Exception as e:
                    log(f"保留样本改动核对失败：{str(e)[:200]}")
                    _note_fail("keep_edit", r["id"], e)
    log(f"保留样本改动核对：{sum(n.values())} 条（{dict(n)}）")


def _rev_risky(r: dict) -> bool:
    """DeepSeek 日常改写（换个说法/改高级/改生动/润色）当改坏稿的：11-07 抽检 5 条坏 4 条（改写稿常常不比原文差），补查排最前。"""
    return str(r.get("draft_src") or "").startswith("pool:") or str(r.get("draft_prompt") or "").startswith("restyle")


def _rev_invalid() -> set:
    """已经判为无效的反向样本 id：补查不过、抽检不过、exclude_ids.txt 里列的。"""
    ex_file = AP / "exclude_ids.txt"
    return ({r["id"] for r in read_jsonl(F["reverse_check"]) if not r["ok"]}
            | {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped") and r.get("ok") is False}
            | (set(ex_file.read_text().split()) if ex_file.exists() else set()))


def _rev_checked_v() -> set:
    """已经用当前版本（REVERSE_JUDGE_V）裁判判过的反向样本 id：生成时判的（reverse_raw.judge_v）和补查的（reverse_check.v）。"""
    return ({r["id"] for r in read_jsonl(F["reverse_check"]) if r.get("v") == REVERSE_JUDGE_V}
            | {r["id"] for r in read_jsonl(F["reverse_raw"]) if r.get("judge_v") == REVERSE_JUDGE_V})


def _reverse_recheck(c: dict):
    """11-06 audit_fail：已通过的反向样本是旧裁判放过的，用 REVERSE_JUDGE 补查，不过的不进快照（all_pairs 查 reverse_check）。
    每轮最多 pairs.reverse_recheck_per_cycle 条（0 = 不查），免得一下子吃掉额度。
    11-07 audit_fail：裁判升到 V3，V3 没判过的全部重查（V2 补查过的、抽检通过的也查；已经不过的不再查），
    DeepSeek 日常改写当改坏稿的排最前，其次学术文体。查完之前 all_pairs 不放它们进快照。"""
    pc = c["pairs"]
    cap = int(pc.get("reverse_recheck_per_cycle", 0))
    if cap <= 0:
        return
    failed = {r["id"] for r in read_jsonl(F["reverse_check"]) if not r["ok"]} | \
             {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped") and r.get("ok") is False}
    checked = _rev_checked_v() | failed
    blk = _blocked("reverse_recheck")
    meta = {r["id"]: r for r in read_jsonl(F["reverse_raw"]) if r.get("ok")}
    todo, seen = [], set()
    for r in read_jsonl(F["reverse"]):
        if r["id"] not in checked and r["id"] not in seen and r["id"] not in blk:
            seen.add(r["id"])
            todo.append(r)
    todo.sort(key=lambda r: (not _rev_risky(meta.get(r["id"], {})), r.get("genre") != "academic"))
    todo = todo[:cap]
    if not todo or not quota_gate(c):
        return

    def one(r):
        ok, why = _reverse_verdict(llm.ask_json(pc["judge"], prompts.REVERSE_JUDGE.format(a=r["original"], b=r["draft"]), tag="judge_recheck"))
        return {"id": r["id"], "ok": ok, "why": why, "v": REVERSE_JUDGE_V}

    n = Counter()
    with ThreadPoolExecutor(pc["workers"]) as ex:
        for fu, r in [(ex.submit(one, r), r) for r in todo]:
            try:
                rec = fu.result()
                append(F["reverse_check"], [rec])
                n[rec["ok"]] += 1
            except Exception as e:
                log(f"反向样本补查失败：{str(e)[:200]}")
                _note_fail("reverse_recheck", r["id"], e)
    log(f"反向样本补查：{sum(n.values())} 条，不过 {n[False]} 条")


_distill_fail: dict = {}


def _run_distill(c: dict, rows: list[dict], out: Path, tag: str, ai_mode: bool = False):
    pc = c["pairs"]
    tmp = AP / f"distill_in_{tag}.jsonl"
    for i in range(0, len(rows), pc["workers"] * 5):
        if not quota_gate(c):
            return i > 0
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + pc["workers"] * 5]))
        # 11-12（主会话，Carbon 实测后用户同意）：AI 线 v2 的老师可以单独配（pairs.ai2.teacher），
        # 16 段真实 AI 文字上 Carbon medium 改错事实 2–3 段、Argon medium 10–12 段、Carbon low（不思考）9 段。
        # 裁判、分级、终审和人写线的老师都不变。
        ai2 = pc.get("ai2") or {}
        teacher = (ai2.get("teacher") or pc["teacher"]) if ai_mode else pc["teacher"]
        t_out = int(ai2.get("teacher_timeout", 300)) if ai_mode else 300
        cmd = [sys.executable, str(ROOT / "scripts" / "distill.py"), "--in", str(tmp), "--out", str(out),
               "--teacher", teacher, "--teacher-timeout", str(t_out), "--judge", pc["judge"], "--workers", str(pc["workers"]),
               "--final-gate", pc.get("final_gate", "")]
        if ai_mode:
            cmd.append("--ai-mode")
        r = _sub(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"distill 失败：{r.stderr[-800:]}")
        # 11-06：单条出错 runner 不写记录，下一轮又挑中它；tech-004-01 连着 7 轮每轮白花一次老师调用。
        # 同一条连续 3 轮没出记录就记成 reject，带上报错
        have = ids(out)
        errs = dict(re.findall(r"^失败 (\S+): (.*)$", r.stderr or "", re.M))
        for x in rows[i:i + pc["workers"] * 5]:
            if x["id"] in have:
                _distill_fail.pop((tag, x["id"]), None)
                continue
            n = _distill_fail[(tag, x["id"])] = _distill_fail.get((tag, x["id"]), 0) + 1
            if n >= 3:
                rec = {"id": x["id"], "kind": x.get("kind"), "input": x["text"], "source": x.get("source"), "genre": x.get("genre"),
                       "status": "reject", "reason": f"distill 连续 {n} 轮出错", "error": errs.get(x["id"], "")[:300]}
                with open(out, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _distill_fail.pop((tag, x["id"]), None)
                log(f"distill {x['id']} 连续 {n} 轮出错，记成 reject：{rec['error'][:120]}")


def _distill_invalid() -> set:
    """已经判为无效的打磨样本 id：补查不过、抽检不过、exclude_ids.txt 里列的。和 _rev_invalid 同一套。"""
    ex_file = AP / "exclude_ids.txt"
    return ({r["id"] for r in read_jsonl(F["distill_check"]) if not r["ok"]}
            | {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped") and r.get("ok") is False}
            | (set(ex_file.read_text().split()) if ex_file.exists() else set()))


def stage_distill_human(c: dict):
    pc = c["pairs"]
    # 11-20 daily：原来只数文件里的 ok 行（2,577 ≥ 2,500 就停），但补查不过 290、抽检不过 22、exclude_ids 30 共 338 条
    # 不进快照，实际只有 2,239，差 261 条；候选还有 1 万多段。改成只数还有效的，和反向、保留的算法一致。
    bad = _distill_invalid()
    acc = [r for r in read_jsonl(F["distill_human"]) if r["status"] in ("ok", "keep") and r["id"] not in bad]
    if len(acc) >= pc["targets"]["distill_human"]:
        return
    done = ids(F["distill_human"])
    seg = {s["id"]: s for s in read_jsonl(F["segments"]) if segment.late_check_seg(s) is None}
    cands = [{"id": r["id"], "kind": "human", "genre": r["genre"], "quality": "improvable", "text": seg[r["id"]]["text"],
              "source": r["source"]}
             for r in read_jsonl(F["graded"]) if r["id"] not in done and r.get("tier") == "improvable" and r["id"] in seg
             and r.get("genre") in genres.GENRES and r.get("modern") and _ai(r.get("ai_likelihood")) <= pc["human_ai_max"]
             and int(r.get("standalone") or 0) >= 3 and r["source"] not in c["_control"].get("disabled_sources", [])]
    # 主循环里这一步最慢（120 段约 70 分钟），会拖住范本→反向/保留那条核心线；单独设小一点的每轮数量
    todo = _balanced(cands, Counter(r.get("genre") for r in acc), pc["targets"]["distill_human"], pc.get("human_per_cycle", pc["per_cycle"]))
    if todo and _run_distill(c, todo, F["distill_human"], "human") is not False:
        log(f"打磨人写文字：本轮 {len(todo)} 段")


def stage_ai_drafts(c: dict) -> bool:
    """返回这一轮有没有干活。出题、写初稿、打磨都按文体配额轮流挑（题目文件是按文体顺序写的）。"""
    pc = c["pairs"]
    target = pc["targets"]["distill_ai"]
    acc = [r for r in read_jsonl(F["distill_ai"]) if r["status"] in ("ok", "keep")]
    if len(acc) >= target or not pc.get("writers"):
        return False
    tasks = read_jsonl(F["tasks"])
    if len(tasks) < target * 1.3:
        if not quota_gate(c):
            return False
        # 一次只出 10 个：一次 30 个时输出太长，JSON 经常坏掉
        r = _sub([sys.executable, str(ROOT / "scripts" / "make_inputs.py"), "tasks", "--total", str(int(target * 1.4)),
                  "--per-call", "10", "--model", pc["teacher"], "--out", str(F["tasks"]), "--workers", str(pc["workers"])])
        if r.returncode != 0:
            raise RuntimeError(f"出题失败：{r.stderr[-800:]}")
    done = ids(F["distill_ai"])
    drafts = read_jsonl(F["drafts"])
    backlog = [d for d in drafts if d["id"] not in done]
    if len(backlog) < pc["per_cycle"]:
        # 已通过的 + 还没打磨的 一起算，通过率低的文体会多分到初稿
        have = Counter(r.get("genre") for r in acc) + Counter(d.get("genre") for d in backlog)
        done_d = {d["id"] for d in drafts}
        todo_t = _balanced([t for t in read_jsonl(F["tasks"]) if t["id"] not in done_d and t.get("genre") in genres.GENRES],
                           have, target, pc["per_cycle"] - len(backlog))
        if todo_t:
            if not quota_gate(c):
                return False
            tmp = AP / "tasks_todo.jsonl"
            tmp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo_t))
            r = _sub([sys.executable, str(ROOT / "scripts" / "make_inputs.py"), "drafts", "--tasks", str(tmp), "--out", str(F["drafts"]),
                      "--writers", ",".join(pc["writers"]), "--workers", str(pc["workers"])])
            if r.returncode != 0:
                raise RuntimeError(f"写初稿失败：{r.stderr[-800:]}")
            backlog = [d for d in read_jsonl(F["drafts"]) if d["id"] not in done]
    todo = _balanced([d for d in backlog if d.get("genre") in genres.GENRES], Counter(r.get("genre") for r in acc), target, pc["per_cycle"])
    if todo:
        if _run_distill(c, todo, F["distill_ai"], "ai") is False:  # 一开头就被额度挡住：没干活，AI 线程按空闲歇 15 分钟
            return False
        log(f"打磨 AI 初稿：本轮 {len(todo)} 段（{dict(Counter(d['genre'] for d in todo))}）")
    elif not _bg.get("tasks_out_logged"):
        # 题目数只在少于目标 1.3 倍时补；通过率低（约 65%）时题目全写完了也到不了目标，这里只记一笔，不自动加题
        _bg["tasks_out_logged"] = True
        log(f"AI 初稿：题目已全部写成初稿并打磨完，通过 {len(acc)}/{target}，没有新题，这条线先停")
    return bool(todo)


def _ai2_pick(c: dict) -> dict:
    """AI 线 v2（distill_ai_v2.jsonl，id 后缀 #ai2）这一轮挑哪些，不调模型。
    输入是 ai_drafts.jsonl 里的初稿，文体配额按 v2 里 status=ok 的计数；试点没复核时按记录数到 pilot 就停。"""
    pc = c["pairs"]
    ai2 = pc.get("ai2") or {}
    info = {"enabled": bool(ai2.get("enabled")), "todo": [], "pilot_done": False, "ok": 0, "records": 0, "limit": 0}
    if not info["enabled"]:
        return info
    rows = read_jsonl(F["distill_ai2"])
    acc = [r for r in rows if r.get("status") == "ok"]
    info.update(ok=len(acc), records=len(rows))
    pilot = int(ai2.get("pilot", 0))
    if not ai2.get("pilot_ok", False) and pilot > 0 and len(rows) >= pilot:
        info["pilot_done"] = True
        return info
    target = int(ai2.get("target", 0))
    if len(acc) >= target:
        return info
    done = {r["id"] for r in rows}
    cands = [{**d, "id": d["id"] + "#ai2"} for d in read_jsonl(F["drafts"])
             if d.get("genre") in genres.GENRES and d["id"] + "#ai2" not in done]
    limit = min(pc["per_cycle"], target - len(acc))
    if not ai2.get("pilot_ok", False) and pilot:
        limit = min(limit, pilot - len(rows))  # 试点：记录数到 pilot 就停，不超做
    info["limit"] = max(0, limit)
    info["todo"] = _balanced(cands, Counter(r.get("genre") for r in acc), target, max(0, limit))
    return info


def stage_ai2(c: dict) -> bool:
    """11-10 冒烟训练后加（主会话设计）：用 AI 专用的改写/裁判/终审提示词，把 ai_drafts.jsonl 里的 AI 初稿重新打磨一遍
    （distill.py --ai-mode），写进 distill_ai_v2.jsonl。旧 AI 线（distill_ai.jsonl，EDIT_PROMPT）保留不动。"""
    pc = c["pairs"]
    ai2 = pc.get("ai2") or {}
    info = _ai2_pick(c)
    if not info["enabled"]:
        return False
    if info["pilot_done"]:
        if not _bg.get("ai2_pilot_logged"):
            _bg["ai2_pilot_logged"] = True
            log(f"AI 线 v2 试点 {int(ai2.get('pilot', 0))} 条已做完，等主会话复核（config pairs.ai2.pilot_ok）")
        return False
    todo = info["todo"]
    if not todo:
        return False
    if _run_distill(c, todo, F["distill_ai2"], "ai2", ai_mode=True) is False:  # 一开头就被额度挡住
        return False
    log(f"打磨 AI 初稿 v2：本轮 {len(todo)} 段（{dict(Counter(d['genre'] for d in todo))}）")
    return True


def _ai_stage(c: dict) -> bool:
    """AI 线入口：v2 开着跑 v2（旧线的题目早就用完了），否则跑旧线。"""
    return stage_ai2(c) if (c["pairs"].get("ai2") or {}).get("enabled") else stage_ai_drafts(c)


_bg = {"errors": 0, "wake": None}


def ai_thread():
    """AI 初稿这条线单独跑：出题、写初稿、打磨一轮要一个多小时，放在主循环里人写文字那条线就得干等。"""
    while not _stopping.is_set():
        c = cfg()
        if c["_control"].get("pause") or disk_free_gb() < c.get("disk", {}).get("min_free_gb_run", 5):
            time.sleep(300)
            continue
        try:
            busy = _ai_stage(c)
            _bg["errors"] = 0
        except llm.AgyAuthError:
            busy = False  # 主循环那边会记一行“agy 未登录”，这里不重复记错、不推唤醒
        except Exception:
            if _stopping.is_set():  # 重启前子进程被结束，distill 报失败是预期的，不算出错
                return
            busy = False
            _bg["errors"] += 1
            tb = traceback.format_exc()
            log(f"ai 出错（连续第 {_bg['errors']} 次）：{tb[-800:]}")
            if _bg["errors"] >= 3:
                _bg["wake"], _bg["errors"] = ("error:ai", tb[-3000:]), 0
        time.sleep(c["cycle_sleep_sec"] if busy else 900)


# ---------------- 抽检、口头禅、快照 ----------------

def _gate_problems() -> dict:
    """打磨配对 id → 终审最后一轮列出的问题条数（和 _all_pairs 一样取每个 id 第一条 ok/keep 记录）。"""
    out = {}
    for key in ("distill_human", "distill_ai", "distill_ai2"):
        for r in read_jsonl(F[key]):
            if r["status"] in ("ok", "keep") and r["id"] not in out:
                g = r.get("gate") or []
                out[r["id"]] = sum(1 for x in (g[-1].get("problems") or []) if str(x).strip()) if g else 0
    return out


def _recheck_rules(c: dict) -> list[dict]:
    rc = c["audit"].get("recheck") or []
    return [r for r in (rc if isinstance(rc, list) else [rc]) if int(r.get("per_cycle", 0)) > 0]


def _rule_match(rule: dict, p: dict, gp: dict) -> bool:
    origins = rule.get("origins") or ([rule["origin"]] if rule.get("origin") else ["distill_human", "distill_ai", "distill_ai2"])
    return (p.get("origin") in origins and p["kind"] == "distill" and (not rule.get("genres") or p["genre"] in rule["genres"])
            and gp.get(p["id"], 0) >= int(rule.get("min_gate_problems", 0)))


def all_pairs(hold: bool = True) -> list[dict]:
    """所有合格配对；data/v3/autopilot/exclude_ids.txt 里列出的（抽检或复查后判为不合格的）跳过。
    hold=True（快照、抽检、统计用）时，还要暂扣两类没查完的（11-07 audit_fail）：当前版本反向裁判没判过的反向样本；
    命中 audit.recheck 里 hold 规则、还没补查也没抽检过的打磨样本。补查（_reverse_recheck、_distill_recheck）用 hold=False 挑。"""
    out = _all_pairs_filtered()
    if not hold:
        return out
    rev_ok = _rev_checked_v()
    c = cfg()
    rules = [r for r in _recheck_rules(c) if r.get("hold")]
    held_d = set()
    if rules:
        gp = _gate_problems()
        done = ids(F["distill_check"]) | {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped")}
        held_d = {p["id"] for p in out if p["id"] not in done and any(_rule_match(r, p, gp) for r in rules)}
    # 11-10：试点没经主会话复核前，AI 腔反向样本（#revai）和 AI 线 v2 的配对先不进快照
    hide_revai = not (c["pairs"].get("reverse_ai") or {}).get("pilot_ok", False)
    hide_ai2 = not (c["pairs"].get("ai2") or {}).get("pilot_ok", False)
    return [p for p in out if not (p["kind"] == "reverse" and p["id"] not in rev_ok) and p["id"] not in held_d
            and not (hide_revai and str(p["id"]).endswith("#revai"))
            and not (hide_ai2 and p.get("origin") == "distill_ai2")]


def _all_pairs_filtered() -> list[dict]:
    ex_file = AP / "exclude_ids.txt"
    excluded = set(ex_file.read_text().split()) if ex_file.exists() else set()
    # 11-05：抽检不合格的自动排除（之前只有不合格率超 20% 唤醒时才有人写 exclude_ids，10% 的那两条一直留在快照里）
    excluded |= {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped") and r.get("ok") is False}
    ke = {r["id"]: {"verdict": r["verdict"], "n_fix": r.get("n_fix") or 0} for r in read_jsonl(F["keep_edit"])}
    excluded |= {r["id"] for r in read_jsonl(F["reverse_check"]) if not r["ok"]}
    excluded |= {r["id"] for r in read_jsonl(F["distill_check"]) if not r["ok"]}  # 11-06 daily：AI 线科普补查
    # 11-10：pairs.ai2.exclude_v1 为 true 时，旧的 AI 线（origin=distill_ai）不再进快照，由 v2 代替
    drop_v1 = bool((cfg()["pairs"].get("ai2") or {}).get("exclude_v1"))
    out, seen = [], set()
    for p in _all_pairs():  # 11-05：keep.jsonl 里有 12 条同 id 重复（02:40 前后两个进程同时在跑），按 id 去重
        if drop_v1 and p.get("origin") == "distill_ai":
            continue
        if p.get("origin") is None and p["kind"] == "keep" and p["src"] != p["tgt"]:
            # 精修过的保留样本。11-06 用户定：一律原文→原文（答案就是输入，教模型好文字不动）。
            # 改动核对（_keep_edit_check）找到真错（n_fix>0）的不要：答案里会留着错别字、病句；没核对完的也先不要
            v = ke.get(p["id"])
            if v is None or v["n_fix"] > 0:
                continue
            p = {**p, "tgt": p["src"]}
        if p["id"] not in excluded and p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out


def _match_breaks(draft: str, original: str) -> str:
    """11-16 主会话：人写原文切段后段间只换一行，Gemini 改写出的稿子段间空一行。反向样本里 1,421 条是这样，
    模型会顺带学“把空行删掉”，和 AI 线（空行→空行）打架。原文没有空行时，把稿子的连续换行压成一个。
    只在导出配对时改，jsonl 原样不动。"""
    if "\n\n" in draft and "\n\n" not in original:
        return re.sub(r"\n\s*\n+", "\n", draft)
    return draft


def _all_pairs() -> list[dict]:
    out = []
    for r in read_jsonl(F["reverse"]) + read_jsonl(F["keep"]):
        out.append({"id": r["id"], "kind": r["kind"], "genre": r["genre"],
                    "src": _match_breaks(r["draft"], r["original"]), "tgt": r["original"]})
    for key in ("distill_human", "distill_ai", "distill_ai2"):
        for r in read_jsonl(F[key]):
            if r["status"] in ("ok", "keep"):
                out.append({"id": r["id"], "kind": "distill" if r["status"] == "ok" else "keep", "genre": r.get("genre"),
                            "src": r["input"], "tgt": r["target"], "origin": key})
    return out


def _audit_prompt(p: dict) -> str:
    """抽检/补查用的提示词：AI 线 v2 的配对（origin=distill_ai2）用 AUDIT_AI_PROMPT，别的照旧。"""
    gn = genres.GENRES.get(p["genre"], {}).get("name", p["genre"])
    if p.get("origin") == "distill_ai2":
        return prompts.AUDIT_AI_PROMPT.format(principles=prompts.principles(), genre_name=gn, src=p["src"], tgt=p["tgt"])
    return prompts.AUDIT_PROMPT.format(principles=prompts.principles(), kind=p["kind"], genre_name=gn, src=p["src"], tgt=p["tgt"])


def _distill_recheck(c: dict):
    """11-06 daily：打磨配对按 audit.recheck 选的子集（origin + genres）用抽检同一提示词全部补查，不过的不进快照（all_pairs 查 distill_check）。
    起因：AI 线科普抽检 6/14 不合格（AI 线其他文体 7/83）——题目要求写进的“硬数字”和角色设定（“各位义乌老板好”）本身就站不住，
    打磨不许改信息，修不好。结果单独记在 distill_check.jsonl，不算进抽检不合格率、不触发 audit_fail。每轮最多 per_cycle 条。
    11-07 audit_fail：recheck 可以是规则列表，每条规则可选 origins/genres/min_gate_problems/per_cycle/hold。
    起因：终审最后一轮列了 ≥2 条（minor）问题的打磨样本抽检 18/81 不合格（22%），一条没列的 0/50。hold=true 的规则命中、
    还没查的，all_pairs 先不放进快照。"""
    rules = _recheck_rules(c)
    if not rules:
        return
    done = ids(F["distill_check"]) | {r["id"] for r in read_jsonl(F["audit"]) if not r.get("skipped")}  # 正式抽检过的不再查
    pairs = [p for p in all_pairs(hold=False) if p["id"] not in done]
    gp = _gate_problems()
    todo, picked = [], set()
    for rule in rules:
        sel = [p for p in pairs if p["id"] not in picked and _rule_match(rule, p, gp)][:int(rule["per_cycle"])]
        picked |= {p["id"] for p in sel}
        todo += sel
    if not todo or not quota_gate(c):
        return
    a = c["audit"]

    def one(p):
        j = llm.ask_json(a["model"], _audit_prompt(p), tag="audit_recheck")
        ok = all(j.get(k, True) for k in ("fact_ok", "quality_ok", "register_ok", "edit_ok"))
        return {"id": p["id"], "genre": p["genre"], "ok": ok, **{k: j.get(k) for k in ("fact_ok", "quality_ok", "register_ok", "edit_ok", "problems")}}

    n = Counter()
    with ThreadPoolExecutor(a.get("workers", 6)) as ex:
        for fu in [ex.submit(one, p) for p in todo]:
            try:
                rec = fu.result()
                append(F["distill_check"], [rec])
                n[rec["ok"]] += 1
            except Exception as e:
                log(f"打磨配对补查失败：{str(e)[:200]}")
    log(f"打磨配对补查：{sum(n.values())} 条，不过 {n[False]} 条")


def stage_audit(c: dict) -> str | None:
    a = c["audit"]
    _distill_recheck(c)
    pairs = all_pairs()
    audited = ids(F["audit"])
    new = [p for p in pairs if p["id"] not in audited]
    if len(new) < a["every"]:
        return None
    if not quota_gate(c):
        return None
    sample = random.Random(len(pairs)).sample(new, min(a["n"], len(new)))

    def one(p):
        j = llm.ask_json(a["model"], _audit_prompt(p), tag="audit")
        ok = all(j.get(k, True) for k in ("fact_ok", "quality_ok", "register_ok", "edit_ok"))
        return {"id": p["id"], "kind": p["kind"], "genre": p["genre"], "ok": ok, **{k: j.get(k) for k in ("fact_ok", "quality_ok", "register_ok", "edit_ok", "problems")}}

    with ThreadPoolExecutor(a.get("workers", 6)) as ex:
        res = [fu.result() for fu in [ex.submit(one, p) for p in sample]]
    append(F["audit"], res)
    # 已抽检过的不再抽：把这一轮没抽到的也记一笔"跳过"，下次只看更新的配对
    append(F["audit"], [{"id": p["id"], "skipped": True} for p in new if p["id"] not in {r["id"] for r in res}])
    bad = [r for r in res if not r["ok"]]
    rate = len(bad) / max(len(res), 1)
    log(f"抽检 {len(res)} 条，不合格 {len(bad)} 条（{rate:.0%}）")
    hist = status().get("audits", []) + [{"t": dt.datetime.now().isoformat(timespec="minutes"), "n": len(res), "bad": len(bad)}]
    save_status(audits=hist[-50:])
    if rate > a["fail_max"]:
        return "audit_fail"
    return None


def stage_slop(c: dict) -> str | None:
    n_teacher = sum(1 for k in ("distill_human", "distill_ai", "distill_ai2") for r in read_jsonl(F[k]) if r["status"] == "ok")
    done = status().get("slop_done", [])
    nxt = [t for t in c["slop_thresholds"] if t not in done and n_teacher >= t]
    if not nxt:
        return None
    human = AP / "human_ref.jsonl"
    seg = {s["id"]: s for s in read_jsonl(F["segments"])}
    # 人写参照：复审通过的范本 + 分级为 good/exemplar 的段落（11-05 前只有前者，183 段太少，没出现过的短语倍数都被放大）
    hid = {e["id"] for e in read_jsonl(F["confirm"]) if e["ok"]} | {g["id"] for g in read_jsonl(F["graded"]) if g.get("tier") in ("good", "exemplar")}
    human.write_text("".join(json.dumps({"text": seg[i]["text"]}, ensure_ascii=False) + "\n" for i in sorted(hid) if i in seg))
    teacher = AP / "teacher_targets.jsonl"
    teacher.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for k in ("distill_human", "distill_ai", "distill_ai2") for r in read_jsonl(F[k])))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "slop_profile.py"), "--teacher", str(teacher), "--human", str(human), "--top", "120"],
                       capture_output=True, text=True, cwd=ROOT)
    (AP / f"slop_profile_{nxt[0]}.txt").write_text(r.stdout + r.stderr)
    save_status(slop_done=done + [nxt[0]])
    return f"slop_profile:{nxt[0]}"


def stage_snapshot(c: dict):
    pairs = all_pairs()
    snap = AP / "pairs_snapshot.jsonl"
    snap.write_text("".join(json.dumps({"id": p["id"], "doc": p["id"].split("#")[0], "genre": p["genre"], "recipe": p["kind"],
                                        "draft": p["src"], "original": p["tgt"]}, ensure_ascii=False) + "\n" for p in pairs))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_sft.py"), "--pairs", str(snap), "--outdir", str(D / "sft")],
                   capture_output=True, text=True, cwd=ROOT)


# ---------------- 唤醒 ----------------

def wake(kind: str, c: dict, extra: str = "") -> bool:
    w = c["wake"]
    today = dt.date.today().isoformat()
    hist = status().get("wakes", [])
    if w["max_per_day"] <= 0:
        log(f"唤醒已关闭（wake.max_per_day=0），跳过 {kind}")  # 11-07 用户定：Claude 额度不够，先不唤醒
        return False
    if sum(1 for x in hist if x["t"].startswith(today) and x.get("rc") == 0) >= w["max_per_day"]:
        log(f"今天唤醒次数已满，跳过 {kind}")
        return False
    tpl = (ROOT / "autopilot" / "wake_prompt.md").read_text()
    prompt = tpl.format(kind=kind, now=dt.datetime.now().isoformat(timespec="minutes"), stats=stats_md(), extra=extra or "（无）",
                        log_tail="\n".join(F["log"].read_text().splitlines()[-40:]) if F["log"].exists() else "")
    ts = dt.datetime.now().strftime("%m%d-%H%M")
    (AP / "wakes" / f"{ts}-{kind.replace(':', '_')}.prompt.md").write_text(prompt)
    t0 = time.time()
    if w.get("agent") == "agy":
        # 11-07 用户定：Claude Code 周额度紧，唤醒改用 agy 里的 Claude（单独的额度池）。模型按顺序试，
        # 只有很快就失败的（额度用完、资格校验这类）才换下一个，跑了一阵才失败的可能已经改了一半，不再叠一个上去。
        # 环境变量去掉密钥（agy 会把内容发到 Google）
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "SRH_API_KEY"))}
        out, rc, who = "", -1, ""
        for m in [w.get("model") or "claude-opus-5-5-medium"] + list(w.get("fallback_models", [])):
            who, t1 = "agy:" + m, time.time()
            log(f"唤醒 {who}：{kind}")
            try:
                # 参数从 wake.agy_flags 读，默认 auto 模式。11-07 实测：agy 无人值守的 auto 模式会拒掉所有 shell 命令，
                # 唤醒要能跑检查就得由用户自己在配置里写 ["--dangerously-skip-permissions"]（主会话的 Claude 被安全检查拦下，不代设）
                out, err, _ = llm._run_agy(["agy", "-p", prompt, "--model", m, "--output-format", "json",
                                            *w.get("agy_flags", ["--mode", "auto"]), "--disable-slash-commands",
                                            "--print-timeout", f"{w['timeout_min'] * 60}s"],
                                           w["timeout_min"] * 60 + 120, cwd=ROOT, env=env)
                rc = 0 if json.loads(out).get("status") == "SUCCESS" else 1
                if rc:
                    out = out or err[-500:]
            except Exception as e:
                out, rc = f"失败：{e!r}"[:500], -1
            if rc == 0 or time.time() - t1 > 180:
                break
        try:
            summary = json.loads(out).get("response", "")[:500]
        except Exception:
            summary = out[-300:]
    elif w.get("agent") == "claude-ds":
        # 11-07 22:10 用户定：唤醒交给 DeepSeek（/usr/local/bin/claude-ds：Claude Code 经 Floway 换成 DeepSeek，独立配置目录
        # ~/.claude-ds，auto 模式的权限检查也由 DeepSeek 在本地做）。模型用 DS_MODEL 换，按顺序试，只有 3 分钟内就失败的才换下一个
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
        env.setdefault("HOME", "/root")  # 包装脚本靠 $HOME 找 ~/.claude-ds
        out, rc, who, summary = "", -1, "", ""
        for m in [w.get("model") or "deepseek-v4.1-flash"] + list(w.get("fallback_models", [])):
            who, t1 = "claude-ds:" + m, time.time()
            log(f"唤醒 {who}：{kind}")
            try:
                r = subprocess.run(["claude-ds", "-p", prompt, "--permission-mode", "auto", "--output-format", "json"],
                                   cwd=ROOT, capture_output=True, text=True, timeout=w["timeout_min"] * 60,
                                   env={**env, "DS_MODEL": m})
                out, rc = r.stdout or r.stderr[-500:], r.returncode
                j = json.loads(r.stdout)
                summary = str(j.get("result", ""))[:500]
                if j.get("is_error"):
                    rc = rc or 1
            except subprocess.TimeoutExpired:
                out, rc = "超时", -1
            except Exception as e:
                out, rc = (out or "") + f" 失败：{e!r}"[:300], rc if rc else -1
            if rc == 0 or time.time() - t1 > 180:
                break
        summary = summary or out[-300:]
    else:
        who = "Claude"
        log(f"唤醒 Claude：{kind}")
        cmd = ["claude", "-p", prompt, "--permission-mode", "auto", "--output-format", "json"]
        if w.get("model"):
            cmd += ["--model", w["model"]]
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}  # 不继承上层会话，避免被当成子会话
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=w["timeout_min"] * 60, env=env)
            out, rc = r.stdout, r.returncode
        except subprocess.TimeoutExpired:
            out, rc = "超时", -1
        try:
            summary = json.loads(out).get("result", "")[:500]
        except Exception:
            summary = out[-300:]
    (AP / "wakes" / f"{ts}-{kind.replace(':', '_')}.result.json").write_text(out)
    hist.append({"t": dt.datetime.now().isoformat(timespec="minutes"), "kind": kind, "rc": rc, "secs": round(time.time() - t0),
                 "who": who, "summary": summary})
    save_status(wakes=hist[-60:])
    log(f"{who} 返回（{rc}，{time.time() - t0:.0f} 秒）：{summary[:200]}")
    # 11-07：14:50 的 daily 撞上 Claude 5 小时额度，2 秒就失败了，管线照样当成唤醒过、重启，日报那天就没了。
    # 现在失败不重启（调用方据此稍后重试 daily）；跑了很久才失败的可能改过东西，照常重启加载
    return rc == 0 or time.time() - t0 > 180


# ---------------- 统计与报告 ----------------

def stats() -> dict:
    raw = {p.stem: sum(1 for _ in open(p)) for p in (D / "raw").glob("*.jsonl")}
    segs = read_jsonl(F["segments"])
    graded = read_jsonl(F["graded"])
    conf = read_jsonl(F["confirm"])
    by_src = defaultdict(Counter)
    for s in segs:
        by_src[s["source"]]["segments"] += 1
    for g in graded:
        by_src[g["source"]]["graded"] += 1
        by_src[g["source"]][f"tier_{g.get('tier')}"] += 1
    for r in conf:
        by_src[r["source"]]["exemplar" if r["ok"] else "exemplar_rejected"] += 1
    pairs = all_pairs()
    pairs_ids = {p["id"] for p in pairs}
    held = Counter(p["kind"] for p in all_pairs(hold=False) if p["id"] not in pairs_ids)  # 11-07：补查完之前暂扣的
    audits = [r for r in read_jsonl(F["audit"]) if not r.get("skipped")]
    return {"raw_docs": raw, "segments": len(segs), "graded": len(graded), "by_source": {k: dict(v) for k, v in by_src.items()},
            "pairs_by_kind": dict(Counter(p["kind"] for p in pairs)), "pairs_by_genre": dict(Counter(p["genre"] for p in pairs)),
            "pairs_total": len(pairs), "pairs_held": dict(held), "audit_n": len(audits), "audit_bad": sum(1 for r in audits if not r["ok"]),
            "reverse_tried": sum(1 for _ in read_jsonl(F["reverse_raw"]))}


def stats_md() -> str:
    s = stats()
    lines = [f"- 收集：{json.dumps(s['raw_docs'], ensure_ascii=False)}", f"- 切段 {s['segments']}，已分级 {s['graded']}",
             f"- 配对：{s['pairs_total']}（{json.dumps(s['pairs_by_kind'], ensure_ascii=False)}）"
             + (f"；另有 {sum(s['pairs_held'].values())} 条暂扣待补查（{json.dumps(s['pairs_held'], ensure_ascii=False)}）" if s.get("pairs_held") else ""), f"- 按文体：{json.dumps(s['pairs_by_genre'], ensure_ascii=False)}",
             f"- 抽检：{s['audit_n']} 条，不合格 {s['audit_bad']}", "- 各来源："]
    for k, v in sorted(s["by_source"].items()):
        lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines)


def stage_report(c: dict):
    last = status().get("report_at", 0)
    if time.time() - last < c["report_every_min"] * 60:
        return
    s = stats()
    save_status(report_at=time.time(), stats=s)
    try:
        r = subprocess.run([sys.executable, str(ROOT / "autopilot" / "report.py")], capture_output=True, text=True, cwd=ROOT, timeout=300)
        if r.returncode:
            log(f"报告生成失败：{r.stderr[-300:]}")
    except Exception as e:
        log(f"报告生成失败：{e}")


# ---------------- 主循环 ----------------

def main():
    log(f"启动（pid {os.getpid()}）")
    now = dt.datetime.now().isoformat(timespec="seconds")
    save_status(pid=os.getpid(), started=now, state="运行中")
    if not status().get("first_started"):
        save_status(first_started=now, daily_at=time.time())
    if ONCE:
        for s in cfg()["sources"]:
            harvest_source(s)
    else:
        threading.Thread(target=harvest_thread, daemon=True).start()
        threading.Thread(target=ai_thread, daemon=True).start()
    errors = Counter()
    while True:
        c = cfg()
        if c["_control"].get("pause"):
            save_status(state="暂停（control.json）")
            time.sleep(300)
            continue
        dk = c.get("disk", {})
        prune_agy_logs(dk.get("agy_log_keep_days", 3))
        free = disk_free_gb()
        save_status(disk_free_gb=round(free, 1))
        if free < dk.get("min_free_gb_run", 5):
            log(f"磁盘只剩 {free:.1f}GB（下限 {dk.get('min_free_gb_run', 5)}GB），所有环节暂停，30 分钟后再看")
            save_status(state=f"暂停：磁盘只剩 {free:.1f}GB")
            time.sleep(1800)
            continue
        wake_kind, wake_extra, auth_down = None, "", False
        for name, fn in [("segment", stage_segment), ("grade", stage_grade), ("polish", stage_polish), ("confirm", stage_confirm),
                         ("reverse", stage_reverse_keep), ("reverse_ai", stage_reverse_ai),
                         ("distill_human", stage_distill_human), ("ai", _ai_stage),
                         ("audit", stage_audit), ("slop", stage_slop), ("snapshot", stage_snapshot), ("report", stage_report)]:
            if ONCE and name == "report":
                continue
            if not ONCE and name == "ai":  # 平时由 ai_thread 单独跑
                continue
            save_status(state=f"运行中：{name}")
            try:
                res = fn(c)
                errors[name] = 0
                if isinstance(res, str) and not wake_kind:
                    wake_kind = res
            except llm.AgyAuthError as e:
                # 11-16 20:19 agy 掉登录：所有环节都会立刻报这个。重试和唤醒都修不了（要用户本人登录），
                # 按额度到下限的做法处理：半小时记一行、这个循环结束时歇 30 分钟，不唤醒。
                errors[name] = 0
                auth_down = True
                if time.time() - _bg.get("auth_logged", 0) > 1800:
                    _bg["auth_logged"] = time.time()
                    log(f"agy 未登录，所有 agy 环节暂停，等用户重新登录：{e}")
                save_status(state="暂停：agy 未登录，等用户重新登录")
            except Exception:
                errors[name] += 1
                tb = traceback.format_exc()
                log(f"{name} 出错（连续第 {errors[name]} 次）：{tb[-800:]}")
                if errors[name] >= 3 and not wake_kind:
                    wake_kind, wake_extra = f"error:{name}", tb[-3000:]
                    errors[name] = 0
        if ONCE:
            log(f"试跑结束，各环节出错次数 {dict(errors)}")
            return
        if auth_down:  # 见上面：agy 掉登录时整个循环歇 30 分钟再探，不空转
            time.sleep(1800)
            continue
        if not wake_kind and _bg["wake"]:
            (wake_kind, wake_extra), _bg["wake"] = _bg["wake"], None
        # 新来源分级到一定数量：统一复查一次
        st = status()
        reviewed = set(st.get("reviewed_sources", []))
        by_src = Counter(r["source"] for r in read_jsonl(F["graded"]))
        ready = [k for k, n in by_src.items() if n >= c["grade"]["review_after"] and k not in reviewed]
        if ready and not wake_kind:
            wake_kind, wake_extra = "source_review", "待复查的来源：" + "、".join(ready)
            save_status(reviewed_sources=sorted(reviewed | set(ready)))
        # 周额度到下限时各环节都在空转：歇 30 分钟再看（额度缓存 5 分钟，下限附近 5%/6% 来回跳也不会白转）
        floored = _quota_cache and _quota_cache["q"].get("weekly", 100) <= c["quota"]["floor_weekly"]
        last_daily = st.get("daily_at", 0)
        # 11-07：额度到下限时管线不干活，daily 没东西可看，不唤醒（省 agy 的 Claude 额度），恢复后再补
        if not wake_kind and not floored and c["wake"]["max_per_day"] > 0 and time.time() - last_daily > c["wake"]["daily_every_h"] * 3600:
            wake_kind = "daily"
            save_status(daily_at=time.time())
        if wake_kind:
            if wake(wake_kind, c, wake_extra):
                log("唤醒结束，重启自身以加载改动")
                _kill_children()  # AI 初稿线程起的 distill 也要停，不然重启后两份同时写
                llm.shutdown_agy()  # execv 不跑 atexit，不停的话旧的 agy acp 会一直留着
                os.execv(sys.executable, [sys.executable] + sys.argv)
            if wake_kind == "daily":
                save_status(daily_at=time.time() - (c["wake"]["daily_every_h"] - 2) * 3600)  # 没成功，2 小时后再试
                log("daily 唤醒没成功，2 小时后再试")
        save_status(state="周额度到下限，30 分钟后再看" if floored else "本轮完成，休息中")
        time.sleep(1800 if floored else c["cycle_sleep_sec"])


if __name__ == "__main__":
    main()
