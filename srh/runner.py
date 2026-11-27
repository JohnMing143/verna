"""并发跑 jsonl：断了重跑会跳过已完成的 id。"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            if not line.endswith("\n"):  # 别的进程正在写的半行（大记录会分几次写盘），下次再读
                break
            if line.strip():
                out.append(json.loads(line))
    return out


def run(rows: Iterable[dict], out_path: str, fn: Callable[[dict], dict | None], workers: int = 8, key: str = "id") -> None:
    done = {r[key] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r[key] not in done]
    print(f"已完成 {len(done)}，待处理 {len(todo)}", file=sys.stderr)
    lock = threading.Lock()
    ok = fail = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, r): r for r in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # 单条失败不影响整体，重跑时会再试
                fail += 1
                if fail <= 5 or fail % 50 == 0:
                    print(f"失败 {futs[fu][key]}: {e}", file=sys.stderr)
                continue
            if res is not None:
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                ok += 1
            if i % 100 == 0:
                print(f"{i}/{len(todo)}  成功 {ok}  失败 {fail}", file=sys.stderr)
    print(f"完成：成功 {ok}，失败 {fail}（重跑同一命令会补上失败的）", file=sys.stderr)
