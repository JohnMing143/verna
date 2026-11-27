#!/usr/bin/env python3
"""第 2 步：大模型给人写段落打分，只留"具体、自然、清楚"的。目标端的质量就是小模型的天花板，这一步宁严勿松。

  python scripts/select_human.py --in data/human_raw.jsonl --scored data/human_scored.jsonl \
      --out data/human.jsonl --judge deepseek-chat@https://api.deepseek.com/v1
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import llm, prompts  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--scored", required=True, help="打分明细（可断点续跑）")
    ap.add_argument("--out", required=True, help="通过的段落")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min", default="specific=3,natural=4,clear=4,standalone=3")
    a = ap.parse_args()

    def score(r):
        j = llm.ask_json(a.judge, prompts.QUALITY_JUDGE.format(text=r["text"]))
        return {**r, "quality": j}

    run(read_jsonl(a.inp), a.scored, score, a.workers)

    need = {k: int(v) for k, v in (x.split("=") for x in a.min.split(","))}
    kept, why, genres = [], collections.Counter(), collections.Counter()
    for r in read_jsonl(a.scored):
        q = r["quality"]
        if q.get("problem"):
            why["problem"] += 1
            continue
        low = [k for k, v in need.items() if int(q.get(k, 0)) < v]
        if low:
            why["低分:" + ",".join(low)] += 1
            continue
        kept.append(r)
        genres[q.get("genre", "?")] += 1
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"kept": len(kept), "rejected": dict(why.most_common(8)), "genres": dict(genres)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
