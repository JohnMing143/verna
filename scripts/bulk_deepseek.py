#!/usr/bin/env python3
"""用临时的 DeepSeek 渠道批量攒 AI 一侧的文本（不花 Gemini 额度），断了重跑会跳过已完成的。

  drafts ：管线已有的 1740 道题（data/v3/pairs/tasks.jsonl，12 种文体）各写一篇初稿
  degrade：分级为 good/exemplar 的人写段落用 DEGRADE_PROMPT 改坏；只跑规则检查（事实、长度比、照抄率），
           不调 Gemini 裁判——以后要进训练前再裁判
  polish ：像普通用户那样请 AI“润色”人写段落（四种日常说法轮流用），得到现实里最常见的 AI 润色腔；
           同样只做规则检查、judged=false。去掉开头的助手腔和结尾的“修改说明”，原始回复留在 raw
  restyle：同样取人写段落，但用更重的日常改写请求（换个说法 / 改高级一点 / 改正式 / 改生动），改动比 polish 大
  humanize：把 AI 初稿交给它“去 AI 味 / 降 AI 率”（四种说法），得到“直接让大模型去 AI 味”的基线：
           评估时 Verna 要比它好；也是现实里常见的输入（用户先用 AI 降过一遍）。--inputs 指定初稿文件
  --tasks 换题目文件（比如用 DeepSeek 新出的题）；--tiers 指定段落等级；--suffix 区分输出文件
  --rows 直接给段落（jsonl，含 id/src_id/genre/source/text），不从 graded 里挑；restyle 可用 --variants 0,1,3 每段每种说法各写一份

产出放 data/v3/extra/，writer 记成“模型@渠道”，和 Floway 的 DeepSeek 分开，来源可追溯。
用途：统计 DeepSeek 的高频句式补 AI 腔规则；当以后打磨用的初稿/改坏候选；带标签的 AI 文本归档。

  .venv/bin/python scripts/bulk_deepseek.py drafts  --model deepseek-chat@https://api.deepseek.com/v1 --tag official --until 23:50
  .venv/bin/python scripts/bulk_deepseek.py degrade --model ... --tag official --until 23:50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from srh import facts, genres, llm, prompts  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402

V3 = ROOT / "data" / "v3"
OPENER = re.compile(r"^(?:好的|当然(?:可以)?|没问题|以下是|下面是)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["drafts", "degrade", "polish", "restyle", "humanize"])
    ap.add_argument("--tasks", default=str(V3 / "pairs" / "tasks.jsonl"))
    ap.add_argument("--inputs", nargs="*", default=[], help="humanize 用的初稿文件（含 id/text/genre/writer）")
    ap.add_argument("--tiers", default="good,exemplar")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="渠道名，记进 writer 字段")
    ap.add_argument("--until", default="23:50", help="到这个时刻（当天）就不再发新请求")
    ap.add_argument("--out", default=None, help="覆盖输出路径（测试用）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-thinking", action="store_true", help='请求里加 thinking: {type: disabled}')
    ap.add_argument("--rows", default=None, help="degrade/polish/restyle：直接给段落文件")
    ap.add_argument("--variants", default=None, help="restyle：每段用哪几种说法（逗号分隔的下标），默认按 id 轮流一种")
    a = ap.parse_args()
    hh, mm = map(int, a.until.split(":"))
    deadline = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    extra = {"thinking": {"type": "disabled"}} if a.no_thinking else {}
    writer = f"{a.model.split('@')[0]}@{a.tag}"
    out = Path(a.out) if a.out else V3 / "extra" / f"ds_{a.tag}_{a.job}{a.suffix}.jsonl"
    tiers = set(a.tiers.split(","))

    def chat(content: str, temperature: float) -> str:
        t = llm.chat(a.model, [{"role": "user", "content": content}], temperature=temperature, max_tokens=3000,
                     retries=3, timeout=180, tag=f"bulk_{a.job}", **extra).strip()
        if not t:
            raise RuntimeError("空回复")
        return t

    if a.job == "drafts":
        rows = [{**t, "id": t["id"] + "#" + a.tag} for t in read_jsonl(a.tasks) if t.get("genre") in genres.GENRES]

        def gen(r):
            if datetime.now() >= deadline:
                return None
            text = chat(r["task"] + "\n\n（直接写正文，不要使用任何工具）", 0.8)
            first, _, rest = text.partition("\n")
            if rest.strip() and len(first) <= 60 and OPENER.match(first):
                text = rest.strip()
            return {"id": r["id"], "kind": "ai_draft", "genre": r["genre"], "writer": writer, "task": r["task"], "text": text,
                    "opener": bool(OPENER.match(text))}
    elif a.job == "humanize":
        rows = []
        for f in a.inputs:
            for r in read_jsonl(f):
                if r.get("text") and r.get("genre") in genres.GENRES:
                    rows.append({"id": f"{r['id']}#h-{a.tag}", "src_id": r["id"], "src_file": Path(f).name, "genre": r["genre"],
                                 "src_writer": r.get("writer"), "text": r["text"]})
    elif a.rows:
        from verify_pairs import overlap
        rows = []
        for r in read_jsonl(a.rows):
            for v in ([int(x) for x in a.variants.split(",")] if a.variants and a.job == "restyle" else [None]):
                rows.append({**r, "id": f"{r['src_id']}#{a.tag}{ {'polish': 'p', 'restyle': 'r'}.get(a.job, '') }{'' if v is None else v}",
                             "variant": v, "tier": r.get("tier")})
    else:
        from verify_pairs import overlap
        seg = {s["id"]: s for s in read_jsonl(V3 / "segments.jsonl")}
        rows = []
        for g in read_jsonl(V3 / "graded.jsonl"):
            s = seg.get(g["id"])
            if s and g.get("tier") in tiers and g.get("genre") in genres.GENRES:
                rows.append({"id": g["id"] + "#" + a.tag + {"polish": "p", "restyle": "r"}.get(a.job, ""), "src_id": g["id"], "genre": g["genre"],
                             "tier": g["tier"], "source": g["source"], "text": s["text"]})

    def rules(orig: str, d: str) -> tuple[bool, dict]:
        cmp = facts.compare(orig, d)
        ratio = len(d) / max(len(orig), 1)
        cp = overlap(orig, d)
        ok = cmp["ok"] and 0.8 <= ratio <= 2.0 and cp <= 0.45 and "UNUSABLE" not in d[:20] and not OPENER.match(d)
        return ok, {"facts": cmp["hard"][:3], "ratio": round(ratio, 2), "overlap": round(cp, 2)}

    POLISH = ["帮我润色一下这段文字，让它读起来更流畅：\n\n{text}",
              "请优化下面这段话的表达，让它更专业、更有条理：\n\n{text}",
              "把这段文字改得更有文采一些：\n\n{text}",
              "帮我修改一下这段话，让它更通顺易读，意思不要变：\n\n{text}"]
    TAIL = re.compile(r"\n+(?:[#*\s]*)(?:修改说明|润色说明|优化说明|主要修改|修改要点|改动说明|说明)[：:*].*$", re.S)

    RESTYLE = prompts.RESTYLE_PROMPTS  # 11-06 挪到 srh/prompts.py，管线里现调 DeepSeek 也用
    HUMANIZE = ["帮我把这段文字改得更像人写的，去掉AI味：\n\n{text}",
                "这段是AI写的，帮我降低一下AI率，改得自然一点：\n\n{text}",
                "请用更口语、更自然的方式重写下面这段，别让人看出是AI写的：\n\n{text}",
                "帮我把这段改得说人话一点，接地气些：\n\n{text}"]

    clean = prompts.clean_reply

    import zlib
    if a.job == "humanize":
        from srh import tells

        def gen(r):
            if datetime.now() >= deadline:
                return None
            v = zlib.crc32(r["id"].encode()) % len(HUMANIZE)
            raw = chat(HUMANIZE[v].format(text=r["text"]), 0.7)
            d = clean(raw)
            return {"id": r["id"], "src_id": r["src_id"], "src_file": r["src_file"], "kind": "humanize_baseline", "genre": r["genre"],
                    "src_writer": r["src_writer"], "writer": writer, "prompt_variant": v, "input": r["text"], "output": d,
                    "raw": raw if raw != d else None, "ratio": round(len(d) / max(len(r["text"]), 1), 2),
                    "tells_in": round(tells.density(r["text"]), 2), "tells_out": round(tells.density(d), 2)}

    elif a.job == "restyle":
        def gen(r):
            if datetime.now() >= deadline:
                return None
            v = r["variant"] if r.get("variant") is not None else zlib.crc32(r["id"].encode()) % len(RESTYLE)
            raw = chat(RESTYLE[v].format(text=r["text"]), 0.8)
            d = clean(raw)
            ok, info = rules(r["text"], d)
            return {"id": r["id"], "src_id": r["src_id"], "kind": "restyle_candidate", "genre": r["genre"], "tier": r["tier"],
                    "source": r["source"], "writer": writer, "prompt_variant": v, "original": r["text"], "draft": d,
                    "raw": raw if raw != d else None, "rule_ok": ok, "rule": info, "judged": False}

    elif a.job == "polish":

        def gen(r):
            if datetime.now() >= deadline:
                return None
            v = zlib.crc32(r["id"].encode()) % len(POLISH)
            raw = chat(POLISH[v].format(text=r["text"]), 0.7)
            d = raw
            first, _, rest = d.partition("\n")
            if rest.strip() and len(first) <= 60 and (OPENER.match(first) or first.rstrip().endswith(("：", ":"))):
                d = rest.strip()
            d = TAIL.sub("", d).strip().strip("“”\"").strip()
            ok, info = rules(r["text"], d)
            return {"id": r["id"], "src_id": r["src_id"], "kind": "polish_candidate", "genre": r["genre"], "tier": r["tier"],
                    "source": r["source"], "writer": writer, "prompt_variant": v, "original": r["text"], "draft": d,
                    "raw": raw if raw != d else None, "rule_ok": ok, "rule": info, "judged": False}

    elif a.job == "degrade":
        def gen(r):
            if datetime.now() >= deadline:
                return None
            gg = genres.GENRES[r["genre"]]
            d = chat(prompts.DEGRADE_PROMPT.format(genre_name=gg["name"], faults=gg["faults"], text=r["text"]), 0.7)
            ok, info = rules(r["text"], d)
            return {"id": r["id"], "src_id": r["src_id"], "kind": "degrade_candidate", "genre": r["genre"], "tier": r["tier"],
                    "source": r["source"], "writer": writer, "original": r["text"], "draft": d, "rule_ok": ok,
                    "rule": info, "judged": False}

    if a.limit:
        rows = rows[:a.limit]
    print(f"{a.job}：{len(rows)} 条 → {out}（{writer}，截止 {deadline:%H:%M}）", file=sys.stderr)
    run(rows, str(out), gen, a.workers)


if __name__ == "__main__":
    main()
