#!/usr/bin/env python3
"""真实 AI 输出测试集：让几个大模型按 data/native_prompts.txt 各写一段。没有人写参照，只能测"准确"和"好读"。

  python scripts/make_native_test.py --models "deepseek-chat@...,gemini-3.8-flash@..." --out data/native_test.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import llm  # noqa: E402
from srh.runner import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "data/native_prompts.txt"))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ps = [l.strip() for l in open(a.prompts) if l.strip() and not l.startswith("#")]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    jobs = [{"id": f"native{i:03d}-{m.split('@')[0]}", "prompt": p, "model": m} for i, p in enumerate(ps) for m in models]

    def gen(j):
        src = llm.chat(j["model"], [{"role": "user", "content": j["prompt"]}], temperature=0.8)
        return {"id": j["id"], "prompt": j["prompt"], "model": j["model"].split("@")[0], "src": src}

    run(jobs, a.out, gen, a.workers)


if __name__ == "__main__":
    main()
