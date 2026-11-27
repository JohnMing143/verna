#!/usr/bin/env python3
"""Verna 语料库收集器：按 sources.json 把原始语料原样归档到 Google Drive，本地只放临时文件。

归档位置：gdrive:verna-corpus/<lang>/<名字>/<原路径>；目录 CATALOG.md、清单 manifest.json、评估 reviews.json 在 verna-corpus/ 根下。

- hf：逐个文件下载到 corpus/work/，边下边算 sha256/md5，和 Hugging Face 记录的 LFS sha256（小文件是 git blob sha1）核对；
  传 Drive 后再用 Drive 算的 md5 核一遍，然后删本地。下载地址固定到当时的提交号，以后能复现。
- archive：rclone copyurl 从 archive.org 直接流式传到 Drive（不落本地盘），传完用 Drive 的 md5 和 archive.org 公布的 md5 核对。
- git_local：data/v3/git 下已克隆的仓库按文件转成 jsonl.gz（repo、commit、path、text）。不打 tar 包：Drive 会把含脚本的压缩包当恶意软件拦。

每个文件做完记一行到 corpus/state/files.jsonl，重跑时跳过。每个来源从本地文件或远端抽样到 corpus/samples/<id>.jsonl，给评估用。
收来的东西一律是 pending_review，评估结论写在 reviews.json（见 corpus/README.md），没评估或评估没过的不能进训练管线。

用法：.venv/bin/python corpus/collect.py [--only id,id] [--lane local|stream] [--dry]
     .venv/bin/python corpus/collect.py --catalog        # 只重新生成目录并上传元数据
     .venv/bin/python corpus/collect.py --sample id      # 只给某个来源重新抽样
"""
from __future__ import annotations

import argparse
import bz2
import fnmatch
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

C = Path(__file__).resolve().parent
ROOT = C.parent
WORK = C / "work"
STATE = C / "state" / "files.jsonl"
SAMPLES = C / "samples"
LOG = C / "collect.log"
SOURCES = C / "sources.json"
REVIEWS = C / "reviews.json"
# 11-05 起用用户自建 OAuth client 的 remote gcorpus（drive.file 权限，只看得到自己建的文件；不和备份/训练的 gdrive 共用配额）
REMOTE = os.environ.get("VERNA_CORPUS_REMOTE", "gcorpus:verna-corpus")
# rclone 默认用全体用户共享的 Google 项目，每分钟请求数常被挤满（403 rateLimitExceeded），大文件最后一步被拒就整份重传；
# 低层重试调高，让它在原地多等几轮。根治要换自己的 OAuth client_id（见 corpus/README.md）
RC = ["rclone", "--low-level-retries", "200", "--retries", "10", "--contimeout", "60s", "--timeout", "30m", "--drive-chunk-size", "64M",
      "--drive-pacer-min-sleep", "200ms"]
UA = "verna-corpus/0.1 (personal research archive)"
MIN_FREE_GB = 20       # 本地至少留这么多（管线低于 10GB 停收集、低于 5GB 全停）
SAMPLE_N = 300         # 每个来源抽多少条
SAMPLE_CHARS = 2500    # 每条截到多少字
_lock = threading.Lock()
_stop = threading.Event()


def log(msg: str) -> None:
    line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with _lock, open(LOG, "a") as f:
        f.write(line + "\n")


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def load_sources() -> list[dict]:
    return json.loads(SOURCES.read_text())["sources"]


def done_map() -> dict:
    d = {}
    if STATE.exists():
        for line in STATE.read_text().splitlines():
            try:
                r = json.loads(line)
                d[(r["id"], r["path"])] = r
            except Exception:
                pass
    return d


def record(r: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(STATE, "a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def remote_dir(src: dict) -> str:
    return f"{REMOTE}/{src['lang']}/{src['id'].split('.', 1)[1]}"


def _open_url(u: str, timeout: int = 120, headers: dict | None = None):
    req = urllib.request.Request(u, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def http_json(u: str):
    for i in range(6):
        try:
            with _open_url(u) as r:
                return json.load(r), r.headers
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(30 * (i + 1))
        except Exception:
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"拿不到 {u}")


def rclone(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(RC + list(args), capture_output=True, text=True, timeout=timeout)


def remote_md5(remote: str) -> str | None:
    r = rclone("md5sum", remote, timeout=600)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def remote_md5s(rdir: str) -> dict[str, str]:
    """一次列出 Drive 目录下所有文件的 md5（共享的 rclone 配额常被限流，能少调一次是一次）。"""
    for attempt in range(3):
        r = rclone("md5sum", rdir, timeout=1800)
        if r.returncode == 0:
            out = {}
            for line in r.stdout.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    out[parts[1]] = parts[0]
            return out
        time.sleep(60 * (attempt + 1))
    return {}


def upload(local: Path, remote: str, md5: str) -> None:
    for attempt in range(3):
        r = rclone("copyto", str(local), remote)
        if r.returncode == 0:
            got = remote_md5(remote)
            if got == md5:
                return
            log(f"  Drive 上的 md5 对不上（{got} ≠ {md5}），重传 {remote}")
        else:
            log(f"  上传失败 {remote}：{r.stderr.strip()[-300:]}")
        time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"上传 {remote} 三次都没成功")


def wait_disk(need_gb: float, what: str) -> None:
    while free_gb() < MIN_FREE_GB + need_gb and not _stop.is_set():
        log(f"  本地剩 {free_gb():.0f}GB，放 {what}（{need_gb:.1f}GB）后会低于 {MIN_FREE_GB}GB，等 10 分钟")
        time.sleep(600)


# ---------------- Hugging Face ----------------

def hf_list(spec: dict) -> tuple[str, list[dict]]:
    repo = spec["repo"]
    info, _ = http_json(f"https://huggingface.co/api/datasets/{repo}")
    sha = info["sha"]
    files = []
    for p in spec.get("paths", [""]):
        u = f"https://huggingface.co/api/datasets/{repo}/tree/{sha}" + (f"/{urllib.parse.quote(p)}" if p else "") + "?recursive=true"
        while u:
            with _open_url(u) as r:
                files += json.load(r)
                link = r.headers.get("Link", "")
            u = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    inc, exc = spec.get("include", ["*"]), spec.get("exclude", [])
    out = [f for f in files if f["type"] == "file"
           and any(fnmatch.fnmatch(f["path"], x) for x in inc) and not any(fnmatch.fnmatch(f["path"], x) for x in exc)]
    return sha, sorted(out, key=lambda f: f["path"])


BATCH_BYTES, BATCH_FILES = 4e9, 40


def hf_fetch(src: dict, sha: str, f: dict) -> tuple[dict, Path]:
    rel, size, lfs = f["path"], f.get("size", 0), f.get("lfs")
    wait_disk(size / 1e9, rel)
    local = WORK / src["id"] / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{src['hf']['repo']}/resolve/{sha}/{urllib.parse.quote(rel)}"
    for attempt in range(5):
        h256, hmd5, hsha1 = hashlib.sha256(), hashlib.md5(), hashlib.sha1()
        if not lfs:
            hsha1.update(f"blob {size}\0".encode())
        try:
            with _open_url(url, timeout=300) as r, open(local, "wb") as out:
                while chunk := r.read(8 << 20):
                    out.write(chunk)
                    h256.update(chunk)
                    hmd5.update(chunk)
                    if not lfs:
                        hsha1.update(chunk)
            n = local.stat().st_size
            if n != size:
                raise RuntimeError(f"大小不对 {n} ≠ {size}")
            if lfs and h256.hexdigest() != lfs["oid"]:
                raise RuntimeError("sha256 和 HF 记录的不一致")
            if not lfs and hsha1.hexdigest() != f["oid"]:
                raise RuntimeError("git blob sha1 和 HF 记录的不一致")
            break
        except Exception as e:
            log(f"  下载 {rel} 第 {attempt + 1} 次失败：{e}")
            time.sleep(60 * (attempt + 1))
    else:
        raise RuntimeError(f"下载 {rel} 五次都失败")
    remote = f"{remote_dir(src)}/{rel}"
    rec = {"id": src["id"], "path": rel, "remote": remote, "bytes": size, "md5": hmd5.hexdigest(), "sha256": h256.hexdigest(),
           "source": f"hf:{src['hf']['repo']}@{sha}", "fetched_at": datetime.now().isoformat(timespec="seconds")}
    return rec, local


def push_batch(src: dict, batch: list[tuple[dict, Path]]) -> list[dict]:
    """把 work/<id>/ 下这一批文件一次传上去，再一次性取 Drive 的 md5 和下载时算的核对，返回核对通过的。"""
    base = WORK / src["id"]
    for attempt in range(3):
        r = rclone("copy", str(base), remote_dir(src), "--exclude", "_sample/**")
        if r.returncode == 0:
            break
        log(f"  上传失败（第 {attempt + 1} 次）：{r.stderr.strip()[-300:]}")
        time.sleep(60 * (attempt + 1))
    sums = remote_md5s(remote_dir(src))
    ok = []
    for rec, _ in batch:
        if sums.get(rec["path"]) == rec["md5"]:
            ok.append(rec)
        else:
            log(f"  ✗ {rec['path']}：Drive 上的 md5 对不上或没传上（{sums.get(rec['path'])}），下次重跑再传")
    return ok


def run_hf(src: dict, dm: dict) -> None:
    sha, files = hf_list(src["hf"])
    todo = [f for f in files if (src["id"], f["path"]) not in dm]
    tot = sum(f.get("size", 0) for f in files)
    log(f"{src['id']}：{len(files)} 个文件 {tot / 1e9:.2f}GB，还剩 {len(todo)} 个（提交 {sha[:10]}）")
    data = [f for f in files if _sampleable(f["path"])]
    need_sample = not (SAMPLES / f"{src['id']}.jsonl").exists()
    picks = {data[i]["path"] for i in sorted({0, len(data) // 2, len(data) - 1})} if data and need_sample else set()
    got: list[dict] = []
    batch: list[tuple[dict, Path]] = []

    def flush() -> None:
        if batch:
            ok = push_batch(src, batch)
            for rec in ok:
                record(rec)
            log(f"  ✓ 传完并核对 {len(ok)}/{len(batch)} 个文件（{sum(r['bytes'] for r in ok) / 1e6:.0f}MB），最后一个 {batch[-1][0]['path']}")
            batch.clear()
        shutil.rmtree(WORK / src["id"], ignore_errors=True)

    for f in todo:
        if _stop.is_set():
            return
        try:
            rec, local = hf_fetch(src, sha, f)
        except Exception as e:
            log(f"  ✗ {f['path']}：{e}（跳过，下次重跑再试）")
            continue
        if f["path"] in picks:
            try:
                got += sample_file(local, SAMPLE_N // len(picks) + 1)
            except Exception as e:
                log(f"  抽样 {f['path']} 失败：{e}")
        batch.append((rec, local))
        if sum(r["bytes"] for r, _ in batch) >= BATCH_BYTES or len(batch) >= BATCH_FILES:
            flush()
    flush()
    if need_sample and not got and picks:
        # 抽样的文件之前已经传完（重跑），从 Drive 拉回来抽
        for p in picks:
            got += sample_remote_file(src, p, SAMPLE_N // len(picks) + 1)
    if got:
        save_samples(src["id"], got)


# ---------------- archive.org ----------------

def archive_list(spec: dict) -> list[dict]:
    m, _ = http_json(f"https://archive.org/metadata/{spec['item']}")
    inc, exc = spec.get("include", ["*"]), spec.get("exclude", [])
    fs = [f for f in m.get("files", []) if any(fnmatch.fnmatch(f["name"], x) for x in inc)
          and not any(fnmatch.fnmatch(f["name"], x) for x in exc)]
    return sorted(fs, key=lambda f: int(f.get("size", 0)))


def run_archive(src: dict, dm: dict) -> None:
    spec = src["archive"]
    files = archive_list(spec)
    todo = [f for f in files if (src["id"], f["name"]) not in dm]
    log(f"{src['id']}：{len(files)} 个文件 {sum(int(f.get('size', 0)) for f in files) / 1e9:.1f}GB，还剩 {len(todo)} 个")
    pending: list[dict] = []

    def verify() -> None:
        if not pending:
            return
        sums = remote_md5s(remote_dir(src))
        n = 0
        for f in pending:
            if sums.get(f["name"]) == f.get("md5"):
                record({"id": src["id"], "path": f["name"], "remote": f"{remote_dir(src)}/{f['name']}", "bytes": int(f.get("size", 0)),
                        "md5": f["md5"], "source": f"archive.org:{spec['item']}", "fetched_at": datetime.now().isoformat(timespec="seconds")})
                n += 1
            else:
                log(f"  ✗ {f['name']}：md5 对不上（{sums.get(f['name'])} ≠ {f.get('md5')}），下次重跑再传")
        log(f"  ✓ 核对通过 {n}/{len(pending)} 个，最后一个 {pending[-1]['name']}")
        pending.clear()

    for f in todo:
        if _stop.is_set():
            break
        url = f"https://archive.org/download/{spec['item']}/{urllib.parse.quote(f['name'])}"
        remote = f"{remote_dir(src)}/{f['name']}"
        size = int(f.get("size", 0))
        for attempt in range(3):
            t0 = time.time()
            r = rclone("copyurl", url, remote)
            if r.returncode == 0:
                pending.append(f)
                if size > 5e8:
                    log(f"  传完 {f['name']}（{size / 1e9:.1f}GB，{size / 1e6 / max(time.time() - t0, 1):.0f}MB/s）")
                break
            log(f"  {f['name']} 传输失败（第 {attempt + 1} 次）：{r.stderr.strip()[-300:]}")
            time.sleep(120 * (attempt + 1))
        else:
            log(f"  ✗ {f['name']}：三次都没成功（跳过，下次重跑再试）")
        if len(pending) >= 15:
            verify()
    verify()
    if _stop.is_set():
        return
    if not (SAMPLES / f"{src['id']}.jsonl").exists():
        try:
            if src.get("sample") == "wiki_bz2":
                big = max(files, key=lambda f: int(f.get("size", 0)))
                save_samples(src["id"], sample_wiki_bz2(f"https://archive.org/download/{spec['item']}/{urllib.parse.quote(big['name'])}", int(big["size"])))
            elif src.get("sample") == "stackexchange":
                save_samples(src["id"], sample_stackexchange(src, spec))
        except Exception as e:
            log(f"  抽样失败：{e}")


# ---------------- 本地 git 仓库 ----------------

def run_git(src: dict, dm: dict) -> None:
    spec = src["git"]
    got: list[dict] = []
    for d in spec["dirs"]:
        name = d.replace("/", "__") + ".jsonl.gz"
        if (src["id"], name) in dm:
            continue
        repo = ROOT / "data" / "v3" / "git" / d
        if not repo.exists():
            log(f"  ✗ {d}：本地没有这个仓库")
            continue
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        url = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
        rows = []
        for p in sorted(repo.glob(spec.get("glob", "**/*"))):
            if ".git" in p.parts or not p.is_file():
                continue
            try:
                rows.append({"repo": url, "commit": commit, "path": str(p.relative_to(repo)), "text": p.read_text()})
            except UnicodeDecodeError:
                continue
        local = WORK / src["id"] / name
        local.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(local, "wt") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        md5 = hashlib.md5(local.read_bytes()).hexdigest()
        remote = f"{remote_dir(src)}/{name}"
        try:
            upload(local, remote, md5)
        except Exception as e:
            log(f"  ✗ {d}：{e}")
            continue
        got += random.Random(0).sample(rows, min(len(rows), 60))
        record({"id": src["id"], "path": name, "remote": remote, "bytes": local.stat().st_size, "md5": md5,
                "source": f"git:{url}@{commit}", "files": len(rows), "fetched_at": datetime.now().isoformat(timespec="seconds")})
        log(f"  ✓ {d}：{len(rows)} 个文件（提交 {commit[:10]}）")
        shutil.rmtree(WORK / src["id"], ignore_errors=True)
    if got and not (SAMPLES / f"{src['id']}.jsonl").exists():
        save_samples(src["id"], got)


# ---------------- 抽样 ----------------

def _sampleable(path: str) -> bool:
    return bool(re.search(r"\.(parquet|jsonl|jsonl\.gz|json|json\.gz)$", path))


def _trim(r: dict) -> dict:
    """每个字段截到 SAMPLE_CHARS。长文（书、年报）开头多是版权页/法定声明，另存一段中间的 _mid_<字段>，评估看正文用。"""
    out = {}
    for k, v in r.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) and "内容" in x for x in v):
            v = "\n".join(str(x.get("内容", "")) for x in v)   # MNBVC 的“段落”列表拼回正文
        if isinstance(v, str):
            out[k] = v[:SAMPLE_CHARS]
            if len(v) > 3 * SAMPLE_CHARS:
                m = len(v) // 2
                out[f"_mid_{k}"] = v[m:m + SAMPLE_CHARS]
                out[f"_len_{k}"] = len(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = [x[:SAMPLE_CHARS] for x in v[:5]] + ([f"…另有 {len(v) - 5} 条"] if len(v) > 5 else [])
        else:
            out[k] = json.dumps(v, ensure_ascii=False, default=str)[:SAMPLE_CHARS]
    return out


def _reservoir(it, n: int, rnd: random.Random) -> list:
    res, i = [], 0
    for x in it:
        i += 1
        if len(res) < n:
            res.append(x)
        else:
            j = rnd.randrange(i)
            if j < n:
                res[j] = x
    return res


def sample_file(path: Path, n: int, seed: int = 0) -> list[dict]:
    """从本地文件均匀抽 n 条。parquet 按行号抽；jsonl 用蓄水池抽样（整份读一遍）；json 整份读。"""
    rnd = random.Random(seed)
    name = path.name
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows
        want = sorted(rnd.sample(range(total), min(n, total)))
        out, base, wi = [], 0, 0
        for g in range(pf.num_row_groups):
            rows = pf.metadata.row_group(g).num_rows
            idx = []
            while wi < len(want) and want[wi] < base + rows:
                idx.append(want[wi] - base)
                wi += 1
            if idx:
                tab = pf.read_row_group(g).take(idx).to_pylist()
                out += [_trim(r) for r in tab]
            base += rows
        return out
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        head = f.read(1)
    with opener(path, "rt", errors="replace") as f:
        if head == "[":
            data = json.load(f)
            return [_trim(r) for r in rnd.sample(data, min(n, len(data))) if isinstance(r, dict)]
        res = _reservoir((ln.strip() for ln in f if ln.strip()), n, rnd)
        out = []
        for line in res:
            try:
                r = json.loads(line)
                out.append(_trim(r) if isinstance(r, dict) else {"value": str(r)[:SAMPLE_CHARS]})
            except Exception:
                out.append({"_unparsed": line[:SAMPLE_CHARS]})
        return out


def sample_remote_file(src: dict, rel: str, n: int) -> list[dict]:
    local = WORK / src["id"] / "_sample" / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = rclone("copyto", f"{remote_dir(src)}/{rel}", str(local))
        return sample_file(local, n) if r.returncode == 0 else []
    finally:
        shutil.rmtree(WORK / src["id"], ignore_errors=True)


_PAGE = re.compile(r"<page>.*?</page>", re.S)


def sample_wiki_bz2(url: str, size: int, n: int = SAMPLE_N) -> list[dict]:
    """multistream 的 bz2 每个流 100 个完整页面。在文件 5%/30%/55%/80% 处各取一段，找流头解压，抽正文页面（不含重定向）。"""
    pages = []
    for frac in (0.05, 0.30, 0.55, 0.80):
        start = int(size * frac)
        with _open_url(url, timeout=300, headers={"Range": f"bytes={start}-{start + 24 * 2**20}"}) as r:
            raw = r.read()
        pos = raw.find(b"BZh91AY&SY")
        text, buf = [], raw[pos:] if pos >= 0 else b""
        while buf:
            d = bz2.BZ2Decompressor()
            try:
                text.append(d.decompress(buf))
            except (OSError, EOFError):
                break
            if not d.eof:
                break
            buf = d.unused_data
        xml = b"".join(text).decode("utf-8", "replace")
        for m in _PAGE.finditer(xml):
            p = m.group(0)
            if "<ns>0</ns>" not in p or "<redirect" in p:
                continue
            t = re.search(r"<title>(.*?)</title>", p)
            body = re.search(r"<text[^>]*>(.*?)</text>", p, re.S)
            ts = re.search(r"<timestamp>(.*?)</timestamp>", p)
            if body:
                pages.append({"title": t.group(1) if t else "", "timestamp": ts.group(1) if ts else "",
                              "text": _unescape(body.group(1))[:SAMPLE_CHARS], "_at": frac})
    return random.Random(0).sample(pages, min(n, len(pages)))


def _unescape(s: str) -> str:
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")


def sample_stackexchange(src: dict, spec: dict, n: int = SAMPLE_N) -> list[dict]:
    """从几个中等大小的站点各解出 Posts.xml 抽样（只看评估用，不改归档）。"""
    out = []
    sites = ["writers.stackexchange.com.7z", "academia.stackexchange.com.7z", "english.stackexchange.com.7z", "cooking.stackexchange.com.7z"]
    for s in sites:
        d = WORK / src["id"] / "_sample"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        try:
            r = rclone("copyto", f"{remote_dir(src)}/{s}", str(d / s))
            if r.returncode != 0:
                continue
            subprocess.run(["7z", "e", "-y", str(d / s), "Posts.xml", f"-o{d}"], capture_output=True)
            with open(d / "Posts.xml", errors="replace") as f:
                lines = _reservoir((ln for ln in f if "<row " in ln), n // len(sites) + 1, random.Random(0))
            for line in lines:
                a = dict(re.findall(r'(\w+)="([^"]*)"', line))
                out.append({"site": s[:-3], "type": {"1": "question", "2": "answer"}.get(a.get("PostTypeId"), a.get("PostTypeId")),
                            "score": int(a.get("Score", 0) or 0), "created": a.get("CreationDate", ""),
                            "title": _unescape(a.get("Title", "")), "body": _unescape(a.get("Body", ""))[:SAMPLE_CHARS]})
        finally:
            shutil.rmtree(WORK / src["id"], ignore_errors=True)
    return out


def save_samples(sid: str, rows: list[dict]) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    p = SAMPLES / f"{sid}.jsonl"
    with open(p, "w") as f:
        for r in rows[:SAMPLE_N]:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    rclone("copyto", str(p), f"{REMOTE}/_samples/{p.name}", timeout=600)
    log(f"  抽样 {min(len(rows), SAMPLE_N)} 条 → corpus/samples/{p.name}")


# ---------------- 目录 ----------------

STATUS_ZH = {"pending_review": "待评估", "clean": "可用", "flagged": "有问题·清洗后才能用", "rejected": "不用"}


def load_reviews() -> dict:
    return json.loads(REVIEWS.read_text()) if REVIEWS.exists() else {}


_cat_lock = threading.Lock()


def catalog(upload_meta: bool = True) -> None:
    with _cat_lock:
        _catalog(upload_meta)


def _catalog(upload_meta: bool) -> None:
    srcs, dm, rv = load_sources(), done_map(), load_reviews()
    man: dict = {}
    for (sid, _), r in dm.items():
        m = man.setdefault(sid, {"files": 0, "bytes": 0, "sources": set(), "last": ""})
        m["files"] += 1
        m["bytes"] += r.get("bytes", 0)
        m["sources"].add(r.get("source", ""))
        m["last"] = max(m["last"], r.get("fetched_at", ""))
    lines = ["# Verna 语料库目录", "",
             f"更新于 {datetime.now():%Y-%m-%d %H:%M}。原始文件原样存放在本目录 `<语言>/<名字>/` 下；`_samples/` 是评估用的抽样。",
             "", "状态：**待评估**＝收了还没看过，不能用；**可用**＝评估通过，按常规流程处理后可进训练管线；"
             "**有问题·清洗后才能用**＝有价值但有具体问题（见“问题”列），清洗产物要作为新条目登记并复核通过后才能用；**不用**＝评估后放弃。", ""]
    for lang, head in (("zh", "## 中文"), ("en", "## English")):
        lines += [head, "", "| 名字 | 内容 | 已归档 | 时间 | 许可 | 可再分发 | AI 风险 | 状态 | 问题 / 清洗要求 |", "|---|---|---|---|---|---|---|---|---|"]
        for s in srcs:
            if s["lang"] != lang:
                continue
            m = man.get(s["id"])
            got = f"{m['files']} 个文件 {m['bytes'] / 1e9:.2f}GB" if m else ("候选，未收" if s.get("enabled") is False else "排队中")
            r = rv.get(s["id"], {})
            st = STATUS_ZH.get(r.get("status", "pending_review"), r.get("status"))
            issues = "；".join(r.get("issues", [])) or r.get("summary", "")
            row = [s["id"], s["title"], got, s.get("time_range", ""), s.get("license", ""), s.get("redistributable", ""),
                   s.get("ai_risk", ""), st, issues]
            lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in row) + " |")
        lines.append("")
    (C / "CATALOG.md").write_text("\n".join(lines))
    mj = {k: {**v, "sources": sorted(v["sources"])} for k, v in man.items()}
    (C / "manifest.json").write_text(json.dumps({"updated": datetime.now().isoformat(timespec="seconds"), "sources": mj,
                                                 "files": [r for r in dm.values()]}, ensure_ascii=False, indent=1))
    if upload_meta:
        inc = []
        for name in ("CATALOG.md", "manifest.json", "sources.json", "reviews.json", "README.md"):
            inc += ["--include", f"/{name}"]
        rclone("copy", str(C), REMOTE, *inc, timeout=1200)


# ---------------- 主流程 ----------------

def lane(kind: str, srcs: list[dict]) -> None:
    for s in srcs:
        if _stop.is_set():
            return
        try:
            dm = done_map()
            {"hf": run_hf, "archive": run_archive, "git_local": run_git}[s["method"]](s, dm)
            catalog()
        except Exception as e:
            log(f"{s['id']} 出错：{e}\n{traceback.format_exc()[-800:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--lane", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--sample", default="")
    a = ap.parse_args()
    if a.catalog:
        catalog()
        return
    srcs = [s for s in load_sources() if s.get("enabled", True) and s["method"] in ("hf", "archive", "git_local")]
    if a.only:
        want = set(a.only.split(","))
        srcs = [s for s in srcs if s["id"] in want]
    if a.sample:
        s = next(s for s in load_sources() if s["id"] == a.sample)
        (SAMPLES / f"{s['id']}.jsonl").unlink(missing_ok=True)
        {"hf": run_hf, "archive": run_archive, "git_local": run_git}[s["method"]](s, done_map())
        return
    srcs.sort(key=lambda s: (s.get("priority", 9), s["id"]))
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    stream = [s for s in srcs if s["method"] == "archive"]
    local = [s for s in srcs if s["method"] != "archive"]
    if a.lane:
        stream, local = (stream, []) if a.lane == "stream" else ([], local)
    if a.dry:
        for s in stream + local:
            print(s["priority"], s["method"], s["id"])
        return
    log(f"开始：流式 {len(stream)} 个来源，本地中转 {len(local)} 个；本地剩余 {free_gb():.0f}GB")
    ts = [threading.Thread(target=lane, args=(k, x), daemon=True) for k, x in (("stream", stream), ("local", local)) if x]
    for t in ts:
        t.start()
    try:
        for t in ts:
            while t.is_alive():
                t.join(5)
    except KeyboardInterrupt:
        _stop.set()
    catalog()
    log("这一轮做完了")


if __name__ == "__main__":
    main()
