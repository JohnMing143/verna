#!/usr/bin/env python3
"""第 3 步：让几个大模型把人写段落"改坏"成 AI 腔（信息不变）。训练时反过来学：AI 腔 → 人写原文。

  python scripts/aiify.py --in data/human.jsonl --out data/drafts.jsonl \
      --models "deepseek-chat@https://api.deepseek.com/v1,gemini-3.8-flash@https://generativelanguage.googleapis.com/v1beta/openai"

模型越杂越好：真实输入可能来自任何 AI，只用一家造数据，小模型只会认那一家的腔调。
每段随机抽一种风格（见 srh/prompts.py RECIPES），按 id 固定随机种子，重跑结果不变。
"""
from __future__ import annotations

import argparse
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import llm, prompts  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True, help="逗号分隔，name@base_url")
    ap.add_argument("--per-text", type=int, default=1, help="每段造几个版本（不同模型/风格）")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    jobs = []
    for r in read_jsonl(a.inp):
        for k in range(a.per_text):
            rng = random.Random(zlib.crc32(f"{r['id']}#{k}".encode()))
            jobs.append({"id": f"{r['id']}#{k}", "src": r, "recipe": prompts.pick_recipe(rng),
                         "model": models[rng.randrange(len(models))], "seed": rng.random()})

    def make(j):
        rng = random.Random(j["seed"])
        prompt = prompts.aiify_prompt(j["src"]["text"], j["recipe"], rng)
        draft = llm.chat(j["model"], [{"role": "user", "content": prompt}], temperature=0.8, max_tokens=2000)
        return {"id": j["id"], "doc": j["src"]["doc"], "recipe": j["recipe"], "model": j["model"].split("@")[0],
                "original": j["src"]["text"], "draft": draft.strip(), "genre": j["src"].get("quality", {}).get("genre")}

    run(jobs, a.out, make, a.workers)


if __name__ == "__main__":
    main()
