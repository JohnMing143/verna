#!/usr/bin/env python3
"""拉回 Colab 训练结果：notebook 最后一格把预测和运行信息打成 zip，盖写到 gdrive:verna-train/inbox/<底座>.zip。

  .venv/bin/python scripts/pull_results.py --init   # 给每个底座建占位文件（已有的不动）
  .venv/bin/python scripts/pull_results.py          # 有新内容就下载，解压到 runs/<RUN>/<完成时间>/

为什么要占位：rclone 远端 gdrive: 是 drive.file 权限，只看得到自己建的文件，Colab 新建的（比如 runs/ 下面的）看不到；
Colab 盖写服务器建的文件时文件 ID 不变，服务器就能读到新内容。11-10 实测 open 写、shutil.copyfile、cp 三种都行。
同一个底座再跑一次会盖掉上一次的 zip，所以备份（每 6 小时）顺带拉一次；跑完马上要结果就手动跑这个脚本。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = "gdrive:verna-train/inbox"
RUNS = ROOT / "runs"
STATE = RUNS / "pull_state.json"
CHOICES = ("qwen3.5-9b", "gemma4-12b", "qwen3.5-4b")  # 和 notebooks/build_notebook.py 的 MODELS 一致
PLACEHOLDER = "Verna 回传占位文件：Colab 训练完会把结果盖写到这里，服务器 scripts/pull_results.py 拉取。别删、别改名。\n"
ALLOWED = {"meta.json", "pred_test.jsonl", "pred_native.jsonl"}
RC = ["rclone", "--low-level-retries", "30", "--retries", "5", "--contimeout", "30s", "--timeout", "10m"]


def rc(*args: str) -> str:
    r = subprocess.run(RC + list(args), capture_output=True, text=True, timeout=3600)
    if r.returncode:
        raise RuntimeError(f"rclone {args[0]} 失败（{r.returncode}）：{r.stderr.strip()[-500:]}")
    return r.stdout


def listing() -> dict[str, dict]:
    try:
        rows = json.loads(rc("lsjson", "--hash", INBOX))
    except RuntimeError as e:
        if "directory not found" in str(e):
            return {}
        raise
    return {r["Name"]: r for r in rows if not r.get("IsDir")}


def init() -> None:
    have = listing()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "placeholder"
        p.write_text(PLACEHOLDER)
        for c in CHOICES:
            name = f"{c}.zip"
            if name in have:
                print(f"已有 {name}，不动")
                continue
            rc("copyto", str(p), f"{INBOX}/{name}")
            print(f"建好占位文件 {INBOX}/{name}")


def pull() -> list[Path]:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    got, have = [], listing()
    for c in CHOICES:
        info = have.get(f"{c}.zip")
        if info is None:
            print(f"{c}.zip 不存在：先跑 --init（Colab 新建的文件服务器看不到）")
            continue
        md5 = (info.get("Hashes") or {}).get("md5") or f"{info['Size']}:{info['ModTime']}"
        if state.get(c) == md5:
            continue
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "r.zip"
            rc("copyto", f"{INBOX}/{c}.zip", str(z))
            if not zipfile.is_zipfile(z):
                state[c] = md5  # 还是占位文件
                continue
            with zipfile.ZipFile(z) as zf:
                names = set(zf.namelist())
                if "meta.json" not in names:
                    print(f"{c}.zip 里没有 meta.json，跳过")
                    continue
                meta = json.loads(zf.read("meta.json"))
                run = str(meta.get("run") or c)
                if not run.replace(".", "").replace("-", "").replace("_", "").isalnum():
                    raise ValueError(f"RUN 名字不合规：{run!r}")
                fin = meta.get("finished_utc") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                stamp = dt.datetime.strptime(fin, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
                dest = RUNS / run / stamp
                dest.mkdir(parents=True, exist_ok=True)
                for n in names & ALLOWED:  # 只认这几个文件名，不按 zip 里的路径解压
                    (dest / n).write_bytes(zf.read(n))
                skipped = names - ALLOWED
                if skipped:
                    print(f"{c}.zip 里多出的文件没解压：{sorted(skipped)}")
            shutil.copyfile(z, dest / "result.zip")
        state[c] = md5
        RUNS.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        got.append(dest)
        s = meta.get("summary", {})
        print(f"拉到 {run}（完成于 {stamp}）→ {dest.relative_to(ROOT)}：训练 {meta.get('n_train')} 条、"
              f"{meta.get('train_runtime_min')} 分钟、最好 eval_loss {meta.get('best_eval_loss')}；"
              + "；".join(f"{k} {v.get('n')} 条 硬事实错误 {v.get('hard_fact_err', 0):.1%} AI腔 {v.get('tells_src', 0):.1f}→{v.get('tells_out', 0):.1f}"
                         for k, v in s.items()))
    RUNS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()
    if a.init:
        init()
    else:
        got = pull()
        if not got:
            print("没有新结果")
    sys.exit(0)
