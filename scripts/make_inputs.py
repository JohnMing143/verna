#!/usr/bin/env python3
"""v2 第 1–2 步：造输入。

  # 1. 按文体配比生成写作任务（老师模型出题）
  python scripts/make_inputs.py tasks --total 3000 --model agy:gemini-4-argon-medium --out data/tasks.jsonl
  # 2. 让多家模型按任务写初稿，得到真实的 AI 输入（写手越杂越好；不用 GPT 系）
  python scripts/make_inputs.py drafts --tasks data/tasks.jsonl --out data/ai_drafts.jsonl \\
      --writers "agy:gemini-3.8-flash-low,deepseek-chat@https://api.deepseek.com/v1,<模型>@https://your-gateway.example.com/v1"
  # 3. 人写文字转成输入格式（collect_human.py 的输出；任何年代都行，质量由 distill 的分类把关）
  python scripts/make_inputs.py human --src data/human_raw.jsonl --out data/human_inputs.jsonl
  # 4. 人写好文字反向改坏（保持文体，只加这一文体的毛病），产出的配对再交给 verify_pairs.py 核对
  python scripts/make_inputs.py degrade --src data/distilled.jsonl --out data/degraded.jsonl --model agy:gemini-4-argon-medium
"""
from __future__ import annotations

import argparse
import json
import re
import math
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import genres, llm, prompts  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402


def cmd_tasks(a):
    jobs = []
    for key, g in genres.GENRES.items():
        n_calls = max(1, math.ceil(a.total * g["quota"] / a.per_call))
        jobs += [{"id": f"{key}-{i:03d}", "genre": key} for i in range(n_calls)]

    def gen(j):
        g = genres.GENRES[j["genre"]]
        out = llm.ask_json(a.model, prompts.TASKS_PROMPT.format(genre_name=g["name"], n=a.per_call), tag="tasks")
        return {"id": j["id"], "genre": j["genre"], "tasks": [t for t in out.get("tasks", []) if isinstance(t, str)]}

    tmp = a.out + ".calls"
    run(jobs, tmp, gen, a.workers)
    n = 0
    with open(a.out, "w") as f:
        for r in read_jsonl(tmp):
            for k, t in enumerate(r["tasks"]):
                f.write(json.dumps({"id": f"{r['id']}-{k:02d}", "genre": r["genre"], "task": t}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} 个任务 → {a.out}")


def cmd_drafts(a):
    writers = [w.strip() for w in a.writers.split(",") if w.strip()]
    if any("gpt" in w.lower() for w in writers):
        raise SystemExit("按约定不用 GPT 系模型写初稿")
    rows = read_jsonl(a.tasks)
    for r in rows:
        r["writer"] = writers[zlib.crc32(r["id"].encode()) % len(writers)]

    def gen(r):
        text = llm.chat(r["writer"], [{"role": "user", "content": r["task"] + "\n\n（直接写正文，不要使用任何工具）"}], temperature=0.8, tag="draft")
        text = text.strip()
        first, _, rest = text.partition("\n")
        if rest.strip() and len(first) <= 60 and re.match(r"^(?:好的|当然(?:可以)?|没问题|以下是|下面是)", first):
            text = rest.strip()  # 去掉单独一行的助手腔开场白
        return {"id": r["id"], "kind": "ai_draft", "genre": r["genre"], "writer": llm.label(r["writer"]), "task": r["task"], "text": text}

    run(rows, a.out, gen, a.workers)


def cmd_human(a):
    n = 0
    with open(a.out, "w") as f:
        for r in read_jsonl(a.src):
            f.write(json.dumps({"id": r["id"], "kind": "human", "source": r["id"].split(":")[0], "text": r["text"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} 条 → {a.out}")


def cmd_degrade(a):
    # 只拿 distill 判为 excellent 的人写文字来改坏：答案端是真人的好文字，给数据锚住"人写"的分布
    rows = [r for r in read_jsonl(a.src) if r.get("kind") == "human" and r.get("quality") == "excellent" and r.get("genre") in genres.GENRES]

    def gen(r):
        g = genres.GENRES[r["genre"]]
        d = llm.chat(a.model, [{"role": "user", "content": prompts.DEGRADE_PROMPT.format(
            genre_name=g["name"], faults=g["faults"], text=r["input"])}], tag="degrade")
        # 字段和 v1 的 aiify.py 一致，直接交给 verify_pairs.py 核对
        return {"id": r["id"] + "#deg", "doc": r["id"], "recipe": "degrade", "model": a.model, "genre": r["genre"],
                "original": r["input"], "draft": d.strip()}

    run(rows, a.out, gen, a.workers)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("--total", type=int, required=True); t.add_argument("--per-call", type=int, default=30)
    t.add_argument("--model", required=True); t.add_argument("--out", required=True); t.add_argument("--workers", type=int, default=4)
    d = sub.add_parser("drafts"); d.add_argument("--tasks", required=True); d.add_argument("--writers", required=True)
    d.add_argument("--out", required=True); d.add_argument("--workers", type=int, default=6)
    h = sub.add_parser("human"); h.add_argument("--src", required=True); h.add_argument("--out", required=True)
    g = sub.add_parser("degrade"); g.add_argument("--src", required=True); g.add_argument("--model", required=True)
    g.add_argument("--out", required=True); g.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"tasks": cmd_tasks, "drafts": cmd_drafts, "human": cmd_human, "degrade": cmd_degrade}[a.cmd](a)


if __name__ == "__main__":
    main()
