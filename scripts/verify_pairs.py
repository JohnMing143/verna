#!/usr/bin/env python3
"""第 4 步：核对 (AI 腔草稿, 人写原文) 是否信息等价。这一步决定小模型会不会编内容。

为什么要两个方向都查：
  原文有、草稿丢了 → 训练时模型看到"输入没有、输出有"，学会凭空补细节（幻觉）
  草稿有、原文没有 → 模型学会删掉输入里的事实
两种都必须扔。先跑便宜的规则，过了再花钱让大模型裁判看。

  python scripts/verify_pairs.py --in data/drafts.jsonl --checked data/drafts_checked.jsonl \
      --out data/pairs.jsonl --judge deepseek-chat@https://api.deepseek.com/v1
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import facts, llm, prompts, tells  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402

PREAMBLE = re.compile(r"^(?:好的|当然|没问题|以下是|下面是)[^\n]{0,40}[：:]\s*\n+")


def grams(s: str, n: int) -> set[str]:
    s = re.sub(r"\s", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def overlap(src: str, out: str, n: int = 6) -> float:
    """out 里有多少 n 字片段是从 src 原样搬来的。"""
    g = grams(out, n)
    return len(g & grams(src, n)) / max(len(g), 1)


def rule_check(r: dict, max_copy: float) -> tuple[str | None, dict]:
    d = PREAMBLE.sub("", r["draft"]).strip()
    r["draft"] = d
    m = {"len_ratio": round(len(d) / max(len(r["original"]), 1), 2),
         "copy": round(overlap(r["original"], d), 3),
         "tells_gain": round(tells.weighted(d) - tells.weighted(r["original"]), 1)}
    if not d or "UNUSABLE" in d[:20]:
        return "unusable", m
    if not 0.7 <= m["len_ratio"] <= 2.2:
        return "长度异常", m
    if m["copy"] > max_copy:
        return "照抄原文", m
    c = facts.compare(r["original"], d)
    m["facts_soft"] = c["soft"]
    if not c["ok"]:
        m["facts_hard"] = c["hard"]
        return "规则:" + c["hard"][0].split(" ")[0], m
    return None, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--checked", required=True, help="每条的核对明细（可断点续跑）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="", help="不填就只跑规则（试跑用；正式数据一定要加裁判）")
    ap.add_argument("--max-copy", type=float, default=0.45)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    def check(r):
        why, m = rule_check(r, a.max_copy)
        verdict = None
        if why is None and a.judge:
            j = llm.ask_json(a.judge, prompts.FACT_JUDGE.format(a=r["original"], b=r["draft"]))
            verdict = j
            if j.get("verdict") != "pass" or j.get("missing") or j.get("added") or j.get("distorted"):
                why = "裁判:" + ("丢信息" if j.get("missing") else "加信息" if j.get("added") else "改意思" if j.get("distorted") else "fail")
        return {**r, "reject": why, "metrics": m, "judge": verdict}

    run(read_jsonl(a.inp), a.checked, check, a.workers)

    rows = read_jsonl(a.checked)
    stat = collections.defaultdict(collections.Counter)
    kept = []
    for r in rows:
        stat["总"][r["reject"] or "通过"] += 1
        stat[f"风格 {r['recipe']}"][r["reject"] or "通过"] += 1
        stat[f"模型 {r['model']}"][r["reject"] or "通过"] += 1
        if r["reject"] is None:
            kept.append(r)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("id", "doc", "recipe", "model", "genre", "draft", "original", "metrics")},
                               ensure_ascii=False) + "\n")
    for k, c in stat.items():
        n = sum(c.values())
        print(f"{k:24s} {n:6d} 条  通过率 {c['通过'] / n:5.1%}  主要原因 {dict(c.most_common(4))}")


if __name__ == "__main__":
    main()
