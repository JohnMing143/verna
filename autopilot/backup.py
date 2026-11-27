#!/usr/bin/env python3
"""异地备份到 Google Drive（rclone 远端 gdrive:，scope=drive.file，只能看到自己建的文件）。

- 打包整个项目（代码 + data/），不含 .venv、data/v3/git（能重新克隆）、autopilot/env（放接口 key 的地方）
- 用 openssl AES-256 加密后传到 gdrive:shuorenhua-backup/snapshots/，传完比对 md5。
  加密只为了不让 Drive 扫描内容：11-05 第一份明文 tar.gz 一上传就被判成"恶意软件或垃圾内容"，不能直接下载；
  每 6 小时传一个被标记的文件，可能牵连这个 Google 账号。数据本身不保密，密钥和恢复方法放在同目录 RESTORE.txt
- 训练要用的同步到 gdrive:verna-train/：train_sft.ipynb、srh/（评测代码）、data/sft/、data/native_test.jsonl，
  Colab 打开 notebook 直接用。notebook 只在本机版本变了时才传；如果 Drive 上那份被改过（在 Colab 里改了 CHOICE 之类），
  不覆盖，另存成 train_sft-<日期>.ipynb
- 保留：最近 12 份 + 30 天内每天最早的一份；本地只留最新一份 /root/archives/shuorenhua/backup-latest.tar.gz
- 结果写一行到 data/v3/autopilot/run.log（巡检看得到），细节在 backup.log

由 systemd 定时器 srh-backup.timer 每 6 小时跑一次；手动：.venv/bin/python autopilot/backup.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gdrive:shuorenhua-backup"
TRAIN = "gdrive:verna-train"
LOCAL = Path("/root/archives/shuorenhua")
AP = ROOT / "data" / "v3" / "autopilot"
RC = ["rclone", "--low-level-retries", "30", "--retries", "5", "--contimeout", "30s", "--timeout", "10m"]  # 公共 client_id 偶尔被限流，多重试
SKIP_DIRS = {".venv", "__pycache__"}
SKIP_PATHS = {"data/v3/git", "autopilot/env", "corpus/work", "corpus/samples"}  # 语料原文在 gdrive:verna-corpus，不进备份
KEEP_RECENT, KEEP_DAYS = 12, 30
KEY = Path("/root/.config/srh-backup.key")
RESTORE = """Verna（内部代号 shuorenhua / 说人话）备份的恢复方法

snapshots/ 里是 openssl 加密的 tar.gz（加密只为了不让 Google Drive 误判成恶意文件，数据本身不保密）。
密钥：{key}

在 Mac 或 Linux 上：
  printf '%s' '{key}' > key.txt
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass file:key.txt -in shuorenhua-YYYYMMDD-HHMM.tar.gz.enc -out backup.tar.gz
  tar xzf backup.tar.gz        # 解出 shuorenhua/ 目录（代码 + data/，不含 .venv 和 data/v3/git）
  # 然后 python3 -m venv .venv && .venv/bin/pip install -r requirements.txt（或照 README 装依赖）

训练用的文件（notebook、数据、评测代码）在同级的 verna-train/ 文件夹，没加密，Colab 直接打开。
"""


def note(msg: str, run_log: bool = False):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}\n"
    with open(AP / "backup.log", "a") as f:
        f.write(line)
    if run_log:
        with open(AP / "run.log", "a") as f:
            f.write(line)
    print(line, end="")


def rc(*args: str, capture: bool = True) -> str:
    r = subprocess.run(RC + list(args), capture_output=True, text=True, timeout=3600)
    if r.returncode:
        raise RuntimeError(f"rclone {args[0]} 失败（{r.returncode}）：{r.stderr.strip()[-500:]}")
    return r.stdout


def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
    rel = ti.name.split("/", 1)[1] if "/" in ti.name else ""
    parts = set(rel.split("/"))
    if parts & SKIP_DIRS or any(rel == p or rel.startswith(p + "/") for p in SKIP_PATHS) or rel.endswith(".pyc"):
        return None
    return ti


def snapshot_names(listing: str) -> list[str]:
    return sorted(x for x in listing.split() if re.fullmatch(r"shuorenhua-\d{8}-\d{4}\.tar\.gz(\.enc)?", x))


def to_delete(names: list[str], now: dt.datetime) -> list[str]:
    keep = set(names[-KEEP_RECENT:])
    first_of_day: dict[str, str] = {}
    for n in names:
        day = n[11:19]
        first_of_day.setdefault(day, n)
    for day, n in first_of_day.items():
        if (now - dt.datetime.strptime(day, "%Y%m%d")).days <= KEEP_DAYS:
            keep.add(n)
    return [n for n in names if n not in keep]


def ensure_key() -> str:
    if not KEY.exists():
        import secrets
        alphabet = "abcdefghjkmnpqrstuvwxyACDEFGHJKLMNPQRTUVWXY34679"  # 去掉 l/I/1、O/0、S/5、Z/2、B/8
        KEY.parent.mkdir(parents=True, exist_ok=True)
        KEY.write_text("".join(secrets.choice(alphabet) for _ in range(40)))
        KEY.chmod(0o600)
    return KEY.read_text().strip()


def sync_train():
    rc("sync", str(ROOT / "data" / "v3" / "sft"), f"{TRAIN}/data/sft")
    rc("sync", str(ROOT / "srh"), f"{TRAIN}/srh", "--exclude", "__pycache__/**")
    native = ROOT / "data" / "native_test.jsonl"
    if native.exists():
        rc("copyto", str(native), f"{TRAIN}/data/native_test.jsonl")
    push_notebook()
    # 11-10：顺带拉 Colab 回传的训练结果（inbox/<底座>.zip → runs/），失败不影响备份
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pull_results.py")], capture_output=True, text=True, timeout=1800)
    if r.returncode:
        note(f"拉训练结果失败：{(r.stderr or r.stdout).strip()[-300:]}", run_log=True)
    elif "拉到" in r.stdout:
        note(r.stdout.strip()[-600:], run_log=True)


PARAM_RE = re.compile(r'^(CHOICE|LORA_R|RUN_TAG) = ("[^"\n]*"|\d+)', re.M)  # 你在 Colab 里会改的参数（表单），只认值，后面的注释不管


def _code_sig(nb: dict) -> str:
    """只看代码格的源码，去掉参数行；Colab 存回去的运行输出、元数据不算改动。"""
    src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    return hashlib.md5(PARAM_RE.sub("", "\n\x00\n".join(src)).encode()).hexdigest()


def _params(nb: dict) -> dict:
    out = {}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for m in PARAM_RE.finditer(src):
                out[m.group(1)] = m.group(2)
    return out


def push_notebook():
    """把本机 notebook 推到 Drive。
    - Drive 上没有：直接传
    - Drive 上那份只是跑过（带输出）或改了 CHOICE/LORA_R/RUN_TAG：旧版先存到 history/，新版带上你选的参数覆盖
    - Drive 上那份改了别的代码：不覆盖，新版另存 train_sft-<时间>.ipynb（每个新版本只另存一次）"""
    nb_path = ROOT / "notebooks" / "train_sft.ipynb"
    last_f = AP / "backup_nb_last.ipynb"   # 上次推上去的版本（不含参数合并），用来判断 Drive 上改了什么
    state_f = AP / "backup_nb_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    local = json.loads(nb_path.read_text())
    local_md5 = hashlib.md5(nb_path.read_bytes()).hexdigest()
    try:
        remote_raw = rc("cat", f"{TRAIN}/train_sft.ipynb")
        remote = json.loads(remote_raw)
    except (RuntimeError, json.JSONDecodeError):
        remote_raw, remote = "", None
    touched = hashlib.md5(remote_raw.encode()).hexdigest() != state.get("uploaded")  # Colab 存过（跑过或改过参数）
    if remote is None:
        dest = "train_sft.ipynb"
    else:
        last = json.loads(last_f.read_text()) if last_f.exists() else None
        if local_md5 == state.get("uploaded") or local_md5 == state.get("side"):
            return  # 这一版已经推过
        if last is not None and _code_sig(remote) == _code_sig(last):
            p = _params(remote)
            for c in local["cells"]:  # 带上你在 Colab 里选的参数
                if c["cell_type"] == "code":
                    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    c["source"] = PARAM_RE.sub(lambda m: f"{m.group(1)} = {p.get(m.group(1), m.group(2))}", src)
            if touched:  # 带运行输出的旧版本留一份
                rc("copyto", f"{TRAIN}/train_sft.ipynb", f"{TRAIN}/history/train_sft-{dt.datetime.now():%Y%m%d-%H%M}.ipynb")
            dest = "train_sft.ipynb"
        else:
            dest = f"train_sft-{dt.datetime.now():%Y%m%d-%H%M}.ipynb"
            note(f"Drive 上的 train_sft.ipynb 代码被改过，新版本另存为 {dest}", run_log=True)
    tmp = LOCAL / "push_tmp.ipynb"
    tmp.write_text(json.dumps(local, ensure_ascii=False, indent=1))
    rc("copyto", str(tmp), f"{TRAIN}/{dest}")
    tmp.unlink()
    if dest == "train_sft.ipynb":
        last_f.write_text(nb_path.read_text())
        state = {"uploaded": local_md5}
    else:
        state["side"] = local_md5
    state_f.write_text(json.dumps(state))
    note(f"notebook 已推到 Drive：{dest}")


def main():
    t0 = time.time()
    now = dt.datetime.now()
    name = f"shuorenhua-{now:%Y%m%d-%H%M}.tar.gz.enc"
    LOCAL.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL / "backup-tmp.tar.gz"
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:  # 正在追加的 jsonl 可能带半行，read_jsonl 会跳过
        tf.add(ROOT, arcname="shuorenhua", filter=_filter)
    note(f"备份阶段：打包 {time.time() - t0:.0f} 秒")
    key = ensure_key()
    enc = LOCAL / "backup-tmp.tar.gz.enc"
    subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt", "-pass", f"file:{KEY}",
                    "-in", str(tmp), "-out", str(enc)], check=True, capture_output=True, timeout=600)
    md5 = hashlib.md5(enc.read_bytes()).hexdigest()
    size = enc.stat().st_size
    readme = LOCAL / "RESTORE.txt"
    readme.write_text(RESTORE.format(key=key))
    rc("copyto", str(readme), f"{REMOTE}/RESTORE.txt")
    readme.unlink()
    t_up = time.time()
    rc("copyto", str(enc), f"{REMOTE}/snapshots/{name}")
    remote_md5 = rc("md5sum", f"{REMOTE}/snapshots/{name}").split()[0]
    if remote_md5 != md5:
        raise RuntimeError(f"上传后 md5 不一致：本地 {md5}，远端 {remote_md5}")
    note(f"备份阶段：上传+校验 {time.time() - t_up:.0f} 秒")
    tmp.replace(LOCAL / "backup-latest.tar.gz")  # 本地留一份明文的
    enc.unlink()
    t_sync = time.time()
    sync_train()
    note(f"备份阶段：sync_train {time.time() - t_sync:.0f} 秒")
    names = snapshot_names(rc("lsf", f"{REMOTE}/snapshots"))
    old = to_delete(names, now)
    for n in old:
        rc("deletefile", f"{REMOTE}/snapshots/{n}")
    (AP / "backup_status.json").write_text(json.dumps(
        {"t": now.isoformat(timespec="seconds"), "ok": True, "name": name, "mb": round(size / 2**20, 1), "md5": md5,
         "remote_snapshots": len(names) - len(old), "secs": round(time.time() - t0)}, ensure_ascii=False))
    note(f"备份完成：{name}（{size / 2**20:.0f}MB，md5 已核对），Drive 上共 {len(names) - len(old)} 份，删旧 {len(old)} 份，{time.time() - t0:.0f} 秒", run_log=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--notebook"]:  # 改完 notebook 马上推：.venv/bin/python autopilot/backup.py --notebook
        push_notebook()
        sys.exit(0)
    try:
        main()
    except Exception as e:
        note(f"备份失败：{str(e)[:300]}", run_log=True)
        note(traceback.format_exc()[-1500:])
        try:
            (AP / "backup_status.json").write_text(json.dumps(
                {"t": dt.datetime.now().isoformat(timespec="seconds"), "ok": False, "error": str(e)[:300]}, ensure_ascii=False))
        except OSError:
            pass
        sys.exit(1)
