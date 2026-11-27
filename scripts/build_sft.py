#!/usr/bin/env python3
"""第 5 步：生成训练文件。按原文所属文章切分 train/val/test，同一篇文章的段落不会同时出现在训练和测试里。

  python scripts/build_sft.py --pairs data/pairs.jsonl --outdir data/sft
产出：
  train.jsonl / val.jsonl   {"messages": [...]}，直接喂 notebook
  test.jsonl                {"id", "src", "ref"}，给 rewrite.py / eval.py 用
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import prompts  # noqa: E402
from srh.runner import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val", type=float, default=0.03)
    ap.add_argument("--test", type=float, default=0.03)
    ap.add_argument("--max-per-doc", type=int, default=3)
    a = ap.parse_args()

    rows = []
    for p in a.pairs:
        for r in read_jsonl(p):
            # v1/反向配对：draft→original；v2 distill：input→target（只要 status 为 ok 或 keep 的）
            if "status" in r and r["status"] not in ("ok", "keep"):
                continue
            src, tgt = r.get("input") or r.get("draft"), r.get("target") or r.get("original")
            if not src or not tgt:
                continue
            rows.append({"id": r["id"], "doc": r.get("doc") or r["id"].split("#")[0], "draft": src, "original": tgt,
                         "recipe": r.get("recipe") or r.get("kind") or "?", "genre": r.get("genre") or "?"})
    per_doc = collections.Counter()
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        if per_doc[r["doc"]] >= a.max_per_doc:
            continue
        per_doc[r["doc"]] += 1
        h = (zlib.crc32(r["doc"].encode()) % 10000) / 10000
        split = "test" if h < a.test else "val" if h < a.test + a.val else "train"
        out[split].append(r)

    d = Path(a.outdir)
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with open(d / f"{split}.jsonl", "w") as f:
            for r in out[split]:
                msgs = prompts.rewrite_messages(r["draft"]) + [{"role": "assistant", "content": r["original"]}]
                f.write(json.dumps({"id": r["id"], "messages": msgs}, ensure_ascii=False) + "\n")
    with open(d / "test.jsonl", "w") as f:
        for r in out["test"]:
            f.write(json.dumps({"id": r["id"], "src": r["draft"], "ref": r["original"], "recipe": r["recipe"], "genre": r["genre"]}, ensure_ascii=False) + "\n")
    recipes = collections.Counter(r["recipe"] for r in out["train"])
    gen = collections.Counter(r["genre"] for r in out["train"])
    print(json.dumps({k: len(v) for k, v in out.items()} | {"train_recipes": dict(recipes), "train_genres": dict(gen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
