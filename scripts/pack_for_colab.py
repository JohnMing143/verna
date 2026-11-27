#!/usr/bin/env python3
"""把 Colab 需要的东西打成一个 zip：srh/ 代码 + data/sft/ + 真实 AI 测试集。解压到 Drive 的 MyDrive/shuorenhua/。"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "shuorenhua-colab.zip"
files = [p for p in (ROOT / "srh").glob("*.py")] + list((ROOT / "data" / "sft").glob("*.jsonl"))
native = ROOT / "data" / "native_test.jsonl"
if native.exists():
    files.append(native)
if not (ROOT / "data" / "sft" / "train.jsonl").exists():
    raise SystemExit("还没有 data/sft/train.jsonl，先跑 build_sft.py")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(ROOT))
print(out, f"{out.stat().st_size / 1e6:.1f} MB，{len(files)} 个文件")
