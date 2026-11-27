#!/usr/bin/env python3
"""把 config.json 里等于 OLD 的模型串全部换成 NEW（临时渠道到期时用），并在 run.log、decisions.md 记一笔。
用法：.venv/bin/python autopilot/swap_model.py OLD NEW "原因"
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old, new, why = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""
p = ROOT / "autopilot" / "config.json"
c = json.loads(p.read_text())
n = 0


def walk(x):
    global n
    if isinstance(x, dict):
        return {k: walk(v) for k, v in x.items()}
    if isinstance(x, list):
        return [walk(v) for v in x]
    if x == old:
        n += 1
        return new
    return x


c = walk(c)
p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
msg = f"{datetime.now():%m-%d %H:%M:%S} 配置换模型：{old} → {new}，{n} 处。{why}"
for f in (ROOT / "data/v3/autopilot/run.log", ROOT / "data/v3/autopilot/decisions.md"):
    with open(f, "a") as fh:
        fh.write(("\n" if f.suffix == ".md" else "") + msg + "\n")
print(msg)
