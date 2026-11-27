#!/usr/bin/env python3
"""评测：同一批输入，比较几份改写结果（原文不改、大模型+提示词、现成 humanizer、你的 4B/9B……）。

  python scripts/eval.py preds/raw9b.jsonl preds/guard9b.jsonl preds/claude_prompt.jsonl --judge deepseek-chat@... --n 200

每个文件是 jsonl：{"id", "src", "out", "ref"(可选)}。
指标（"准确"和"好读"两组，对应这个模型只管的两件事）：
  准确  硬事实错误率（规则）、裁判判定事实有问题的比例、守卫退回原文的比例
  好读  AI 腔密度、禁用句式出现率、句长变化、照抄输入的比例、裁判盲评胜率（对输入 / 对人写原文）
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import facts, llm, prompts, tells  # noqa: E402
from srh.runner import read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_pairs import overlap  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def judge_row(model: str, r: dict, rng: random.Random) -> dict:
    j = {}
    f = llm.ask_json(model, prompts.FACT_JUDGE.format(a=r["src"], b=r["out"]))
    j["fact_ok"] = f.get("verdict") == "pass" and not (f.get("missing") or f.get("added") or f.get("distorted"))
    j["fact_detail"] = {k: f.get(k) for k in ("missing", "added", "distorted")}
    for other, key in ((r["src"], "win_vs_src"), (r.get("ref"), "win_vs_ref")):
        if not other:
            continue
        flip = rng.random() < 0.5  # 随机左右位置，避免裁判偏向第一个
        a, b = (r["out"], other) if not flip else (other, r["out"])
        s = llm.ask_json(model, prompts.STYLE_JUDGE.format(a=a, b=b))
        better = s.get("better")
        j[key] = 0.5 if better in (0, "0") else float((better in (1, "1")) != flip)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--judge", default="")
    ap.add_argument("--n", type=int, default=200, help="裁判只看前 n 条，省钱")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    table = []
    for p in a.preds:
        rows = [r for r in read_jsonl(p) if r.get("out")]
        for r in rows:
            c = facts.compare(r["src"], r["out"])
            r["_hard"] = c["hard"]
        res = {
            "file": Path(p).name, "n": len(rows),
            "硬事实错误率": mean([bool(r["_hard"]) for r in rows]),
            "守卫退回段落比例": mean([sum(n["fallback"] for n in r["guard"]) / max(len(r["guard"]), 1) for r in rows if r.get("guard")]),
            "AI腔密度 输入→输出": f"{mean([tells.density(r['src']) for r in rows])} → {mean([tells.density(r['out']) for r in rows])}",
            "禁用句式出现率": mean([bool(tells.banned(r["out"])) for r in rows]),
            "句长变化 输入→输出": f"{mean([tells.burstiness(r['src']) for r in rows])} → {mean([tells.burstiness(r['out']) for r in rows])}",
            "长度比": mean([len(r["out"]) / max(len(r["src"]), 1) for r in rows]),
            "照抄输入比例": mean([overlap(r["src"], r["out"]) for r in rows]),
        }
        if any(r.get("ref") for r in rows):
            res["人写原文的 AI腔密度"] = mean([tells.density(r["ref"]) for r in rows if r.get("ref")])
        if a.judge:
            sub = rows[: a.n]
            rng = random.Random(0)
            with ThreadPoolExecutor(a.workers) as ex:
                js = list(ex.map(lambda r: judge_row(a.judge, r, random.Random(rng.random())), sub))
            for r, j in zip(sub, js):
                r["_judge"] = j
            res["裁判:事实无误率"] = mean([j["fact_ok"] for j in js])
            res["裁判:胜过输入"] = mean([j.get("win_vs_src") for j in js])
            if any("win_vs_ref" in j for j in js):
                res["裁判:不输人写原文"] = mean([j.get("win_vs_ref") for j in js])
        with open(p.replace(".jsonl", "") + ".eval.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in ("id", "src", "out", "ref", "_hard", "_judge")}, ensure_ascii=False) + "\n")
        table.append(res)

    keys = list(dict.fromkeys(k for r in table for k in r))
    for k in keys:
        print(f"{k:18s}", " | ".join(f"{str(r.get(k, '')):>22s}" for r in table))


if __name__ == "__main__":
    main()
