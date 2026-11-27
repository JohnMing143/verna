#!/usr/bin/env python3
"""用训练好的小模型改写，带事实守卫：按段改写，数字/英文/书名/长引语对不上就重试，三次都不行就保留原段。

  # 先起服务（任选其一）
  llama-server -m verna-qwen3.5-9b-v1-q8_0.gguf -c 8192 --port 8080 --jinja
  # 单篇
  python scripts/rewrite.py --model verna@http://127.0.0.1:8080/v1 < draft.md > out.md
  # 批量评测（--raw 关掉守卫，看模型本身的事实准确率）
  python scripts/rewrite.py --model verna@http://127.0.0.1:8080/v1 --in data/sft/test.jsonl --out preds/verna-9b.jsonl

守卫只认得出数字、英文/型号、书名、长引语这几类硬信息；"可能"改成"一定"这种要靠评测时的裁判和你自己看。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import facts, llm, prompts, tells  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402


def blocks(text: str, max_chars: int) -> list[tuple[str, str]]:
    """切成 [(kind, text)]：kind=prose 要改写，kind=keep 原样保留（代码、表格、标题）。"""
    out: list[tuple[str, str]] = []
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for part in parts:
        if part.startswith("```"):
            out.append(("keep", part))
            continue
        cur = ""
        for para in re.split(r"\n\s*\n", part):
            p = para.strip()
            if not p:
                continue
            if re.match(r"^(#{1,6}\s|\|)", p):
                if cur:
                    out.append(("prose", cur))
                    cur = ""
                out.append(("keep", p))
                continue
            if cur and len(cur) + len(p) > max_chars:
                out.append(("prose", cur))
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            out.append(("prose", cur))
    return out


INSTRUCTION: str | None = None  # --instruction 覆盖；对照组（大模型+提示词、现成 humanizer）用
TEMPS = [0.7, 0.7, 0.5, 0.3]
TOP_P = 0.8


def rewrite_chunk(model: str, chunk: str, tries: int, guard: bool) -> tuple[str, dict]:
    temps = (TEMPS * tries)[:tries] or TEMPS[:1]
    best, log = None, []
    for t in temps:
        out = llm.chat(model, prompts.rewrite_messages(chunk, INSTRUCTION), temperature=t, max_tokens=2048, top_p=TOP_P)
        c = facts.compare(chunk, out)
        ratio = len(out) / max(len(chunk), 1)
        bad = tells.banned(out)
        log.append({"t": t, "hard": c["hard"], "ratio": round(ratio, 2), "banned": bad})
        if not guard:
            return out, {"tries": log, "fallback": False}
        if c["ok"] and 0.4 <= ratio <= 1.4:
            if not bad:
                return out, {"tries": log, "fallback": False}
            best = best or out  # 事实没问题但还有禁用句式：先记着，继续试
    if best:
        return best, {"tries": log, "fallback": False}
    return chunk, {"tries": log, "fallback": True}


def rewrite_text(model: str, text: str, max_chars: int, tries: int, guard: bool) -> tuple[str, list[dict]]:
    pieces, notes = [], []
    for kind, b in blocks(text, max_chars):
        if kind == "keep":
            pieces.append(b)
            continue
        out, info = rewrite_chunk(model, b, tries, guard)
        pieces.append(out)
        notes.append({"chars": len(b), **info})
    return "\n\n".join(pieces), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", help="jsonl，每行有 id 和 src；不填就从 stdin 读一篇")
    ap.add_argument("--out")
    ap.add_argument("--max-chars", type=int, default=600, help="每次喂给模型的最大字数，小模型段落越短越不容易漏")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--raw", action="store_true", help="关掉守卫，只改写一次（评测模型本身用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instruction", help="指令文件路径，替换默认指令；传 none 表示只发原文（jialinyyzz/humanizer 的 GGUF 要这样）")
    ap.add_argument("--temps", help="每次重试的温度，逗号分隔，默认 0.7,0.7,0.5,0.3；humanizer 用 1.0")
    ap.add_argument("--top-p", type=float, default=0.8, help="humanizer 用 0.95")
    a = ap.parse_args()
    global INSTRUCTION, TEMPS, TOP_P
    if a.temps:
        TEMPS = [float(x) for x in a.temps.split(",")]
    TOP_P = a.top_p
    if a.instruction:
        INSTRUCTION = "" if a.instruction == "none" else Path(a.instruction).read_text()

    if not a.inp:
        out, notes = rewrite_text(a.model, sys.stdin.read(), a.max_chars, a.tries, not a.raw)
        sys.stdout.write(out + "\n")
        fb = [n for n in notes if n["fallback"]]
        print(f"[守卫] {len(notes)} 段，{len(fb)} 段因事实对不上保留了原文", file=sys.stderr)
        for n in fb:
            print("  ", n["tries"][-1]["hard"][:3], file=sys.stderr)
        return

    def one(r):
        out, notes = rewrite_text(a.model, r["src"], a.max_chars, a.tries, not a.raw)
        return {**r, "out": out, "guard": notes}

    run(read_jsonl(a.inp), a.out, one, a.workers)


if __name__ == "__main__":
    main()
