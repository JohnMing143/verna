"""模型调用（只用标准库）。造数据、当裁判、调本地模型都走这里。

两种后端，按模型名区分：
  agy:<家族>-<档位>   走本机常驻的 agy acp 子进程（Antigravity 订阅额度），例如 agy:gemini-4-argon-medium
                      一个 Python 进程只起一个 agy，多线程共用；每次调用开一个 ACP 会话，用完关掉。
                      空闲 SRH_AGY_IDLE_SECS（默认 600）秒后自动停掉，省下约 1GB 内存。
                      agy 可执行文件用 SRH_AGY_BIN 指定（默认 agy）。
                      设了 SRH_AGY_AGENT=<名字> 就临时写一个没有内置工具的精简 agent（输入 token 从约 9.5k 降到约 400）。
                      agy 默认带 GOMEMLIMIT=600MiB（SRH_AGY_GOMEMLIMIT 可改，设成空串就不加），峰值从约 1.34GB 降到约 0.9GB。
  <模型>@<base_url>   OpenAI 兼容接口（DeepSeek、llama-server、Floway……）
                      key 用 SRH_API_KEY_<主机名第一段大写> 或统一的 SRH_API_KEY；环境里没有就读 autopilot/env
                      末尾可加 #nothink（请求里带 thinking: {type: disabled}，给默认开思考的渠道用）
记录来源统一用 label()：agy 模型原样，OpenAI 兼容接口写成 名字@主机名——同一个模型走不同渠道要分得开。

每次调用的 token 用量追加写到 SRH_USAGE_LOG（默认 data/usage.jsonl），用来核算额度。
"""
from __future__ import annotations

import atexit
import contextlib
import fcntl
import json
import os
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

USAGE_LOG = Path(os.environ.get("SRH_USAGE_LOG", Path(__file__).resolve().parents[1] / "data" / "usage.jsonl"))
_lock = threading.Lock()
_AGY_DIR = Path(tempfile.gettempdir()) / "srh-agy-work"  # agy 的工作目录，保持为空，免得它去读项目文件
# 跨进程 agy 并发上限：run.py 主循环和 ai_thread 起的 distill.py 子进程共享这组文件锁，
# 防止双线叠加拉起 12 个 agy 把整机 12GB 内存吃爆；进程异常退出或被 SIGKILL 时内核会自动释放 flock。
_AGY_SLOT_DIR = Path(tempfile.gettempdir()) / "srh-agy-slots"
_AGY_MAX_CONCURRENCY = max(1, int(os.environ.get("SRH_AGY_MAX_CONCURRENCY", "4")))
_agy_slot_tls = threading.local()


@contextlib.contextmanager
def _agy_slot():
    """跨进程（及同进程多线程）限制同时运行的 agy 进程数，同线程可重入。"""
    if getattr(_agy_slot_tls, "held", False):
        yield
        return
    _AGY_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    n = _AGY_MAX_CONCURRENCY
    fd = None
    while fd is None:
        start = random.randrange(n)
        for off in range(n):
            p = _AGY_SLOT_DIR / f"slot-{(start + off) % n}.lock"
            f = open(p, "a+")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fd = f
                break
            except BlockingIOError:
                f.close()
        if fd is None:
            time.sleep(0.25)
    _agy_slot_tls.held = True
    try:
        yield
    finally:
        _agy_slot_tls.held = False
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()

# 11-16 20:19：agy 的登录态失效（~/.gemini/antigravity-cli/antigravity-oauth-token 空了），之后每次调用
# 都会起一个交互式 OAuth 等 60 秒、并发时几个进程一起抢同一份令牌，重试没有用，只能用户本人跑 agy 登录。
# 这里记一个熔断时间：鉴权失败后 30 分钟内不再起 agy 进程，调用直接报 AgyAuthError 由调用方跳过这一轮。
_AUTH_DOWN = {"until": 0.0}
_AUTH_COOLDOWN = 1800
_AUTH_TELLS = ("authentication failed", "authentication required", "not logged into antigravity")


class AgyAuthError(RuntimeError):
    """agy 未登录（令牌失效）。要用户本人跑 agy 重新登录，重试没有用。"""


class AgyModelError(ValueError):
    """agy 没有这个模型（或这个档位）。重试没有用，直接抛给调用方。"""


def agy_auth_down() -> bool:
    return time.time() < _AUTH_DOWN["until"]


def note_agy_auth_fail(*texts: str) -> bool:
    """输出里出现鉴权失败的特征就记下熔断时间并返回 True。
    ACP 的报错是 "Authentication required"（首字母大写），所以统一转小写再比；
    JSON-RPC error code -32000 是 ACP 规定的 auth_required，也算。"""
    s = " ".join(t or "" for t in texts).lower()
    if "code=-32000" in s or any(k in s for k in _AUTH_TELLS):
        _AUTH_DOWN["until"] = time.time() + _AUTH_COOLDOWN
        return True
    return False


def _log(model: str, secs: float, usage: dict, tag: str = ""):
    rec = {"ts": round(time.time()), "model": model, "secs": round(secs, 1), "tag": tag, **usage}
    with _lock:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")


_ENV_FILE = Path(__file__).resolve().parents[1] / "autopilot" / "env"


def _env_file_key(name: str) -> str:
    try:
        for line in _ENV_FILE.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _flags(model: str) -> tuple[str, set[str]]:
    m, _, f = model.partition("#")
    return m, {x for x in f.split(",") if x}


def label(model: str) -> str:
    """记录用的模型名：agy:… 原样；名字@https://host/v1#flags → 名字@host。"""
    if model.startswith("agy:"):
        return model
    m, _ = _flags(model)
    if "@" not in m:
        return m
    name, base = m.split("@", 1)
    return f"{name}@{re.sub(r'^https?://', '', base).split('/')[0]}"


def _endpoint(model: str) -> tuple[str, str, str]:
    if "@" in model:
        name, base = model.split("@", 1)
    else:
        name, base = model, os.environ.get("SRH_BASE_URL", "")
    if not base:
        raise SystemExit("没有 base_url：设置 SRH_BASE_URL，或把模型写成 name@https://host/v1，或用 agy:<模型>")
    host = re.sub(r"^https?://", "", base).split("/")[0].split(".")[0].split(":")[0].upper().replace("-", "_")
    key = (os.environ.get(f"SRH_API_KEY_{host}") or _env_file_key(f"SRH_API_KEY_{host}")
           or os.environ.get("SRH_API_KEY", ""))
    return name, base.rstrip("/"), key


def _run_agy(cmd: list[str], timeout: float, cwd=None, env=None) -> tuple[str, str, bool]:
    """跑 agy，返回 (stdout, stderr, 是否卡在退出)。
    11-06 发现：并发多时约四分之一的 agy 进程把结果写完后卡在退出（退出时同步
    ~/.gemini/antigravity-cli/conversation_summaries.db，WAL 已 200MB+，日志里有 database is locked）。
    原来用 subprocess.run 一直等到超时（360 秒）才杀，结果跟着丢掉，管线再重做一遍，白花时间和额度。
    现在边跑边读 stdout，读到完整的 JSON 结果后最多再等 20 秒（正常约 5 秒就退出，期间在同步会话记录），
    进程还不退就连同子进程一起杀掉。"""
    import signal
    with _agy_slot():
        p = subprocess.Popen(cmd, cwd=cwd or _AGY_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True, env=env)
        out: list[str] = []
        errs: list[str] = []

        def pump(src, dst):
            for line in src:
                dst.append(line)

        ts = [threading.Thread(target=pump, args=(p.stdout, out), daemon=True),
              threading.Thread(target=pump, args=(p.stderr, errs), daemon=True)]
        for t_ in ts:
            t_.start()

        def kill():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()
            for t_ in ts:
                t_.join(2)

        t0, got_at, hung = time.time(), None, False
        while p.poll() is None:
            now = time.time()
            if got_at is None:
                s = "".join(out).strip()
                if s.endswith("}"):
                    try:
                        if "status" in json.loads(s):
                            got_at = now
                    except ValueError:
                        pass
            elif now - got_at > 20:
                hung = True
                kill()
                break
            if now - t0 > timeout:
                kill()
                raise subprocess.TimeoutExpired(cmd, timeout, output="".join(out), stderr="".join(errs))
            time.sleep(0.5)
        for t_ in ts:
            t_.join(2)
        return "".join(out), "".join(errs), hung


# ---------------------------------------------------------------------------
# agy acp：一个 Python 进程共用一个常驻的 agy acp 子进程
# 以前每次调用起一个 `agy -p`，每个进程峰值约 1.34GB，并发 4 个就 5GB 多，整机 OOM 过。
# 实测一个 `agy acp` 跑 8 个会话才 1.30GB，会话之间还是真并发，所以改成常驻。
# 协议：换行分隔的 JSON-RPC 2.0，走 stdin/stdout。
# ---------------------------------------------------------------------------

_ACP_INIT_TIMEOUT = 120
_ACP_NEW_TIMEOUT = 120
_ACP_SET_TIMEOUT = 60
_ACP_CLOSE_TIMEOUT = 15
_ACP_IDLE_DEFAULT = 600
_ACP_STDERR_LINES = 50
_PDEATHSIG_PRCTL = 1  # PR_SET_PDEATHSIG
_SIGKILL = 9
try:  # 提前加载，preexec_fn 里只调一次 prctl，别在 fork 后的子进程里 dlopen
    import ctypes as _ctypes
    _PRCTL = _ctypes.CDLL("libc.so.6", use_errno=True).prctl
except Exception:
    _PRCTL = None

_AGY_LEVELS = {"low", "medium", "high", "xhigh", "max"}

_AGENT_TMPL = """---
name: {name}
description: "Verna: plain text generation, no built-in tools"
tools: []
excludeDefaultComponents: true
mainAgent: true
---
Follow the user instructions exactly.
"""


def _split_agy_model(model: str) -> tuple[str, str | None]:
    """agy:gemini-4-argon-medium → ("gemini-4-argon", "medium")；没有档位后缀就只给家族。"""
    family, _, level = model.rpartition("-")
    if family and level in _AGY_LEVELS:
        return family, level
    return model, None


def _idle_secs() -> float:
    try:
        return max(1.0, float(os.environ.get("SRH_AGY_IDLE_SECS", _ACP_IDLE_DEFAULT)))
    except ValueError:
        return float(_ACP_IDLE_DEFAULT)


def _set_pdeathsig():
    """创建进程的那个线程没了就让 agy 跟着被 SIGKILL，别留孤儿占内存。失败就算了。
    注意它跟的是线程不是进程：所以固定由常驻的派生线程来 Popen，那个线程活到 Python 进程退出。"""
    if _PRCTL is None:
        return
    try:
        _PRCTL(_PDEATHSIG_PRCTL, _SIGKILL)
    except Exception:
        pass


class _RpcError(RuntimeError):
    def __init__(self, method: str, code, message):
        self.code = code
        super().__init__(f"{method}: {message} (code={code})")


class _Waiter:
    __slots__ = ("event", "result", "error", "gen", "method")

    def __init__(self, method: str, gen: int):
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.gen = gen
        self.method = method


def _opt_values(opt: dict) -> list[str]:
    """configOptions 里的可选值可能写 value 也可能写 id，两种都认。"""
    out = []
    for o in opt.get("options") or []:
        v = o.get("value", o.get("id")) if isinstance(o, dict) else None
        if v is not None:
            out.append(v)
    return out


def _find_opt(opts: list, cid: str) -> dict | None:
    for o in opts or []:
        if isinstance(o, dict) and o.get("id") == cid:
            return o
    return None


class _AcpClient:
    """常驻 agy acp 客户端。懒启动，线程安全；每个会话一个 ACP session，用完关掉。

    请求都带自增 id，reader 线程按 id 交给等待者。agy 有时自己发请求（例如要权限），
    一律回拒绝；读失败的行直接丢掉（不影响别的调用）。
    """

    def __init__(self):
        self._start_lock = threading.RLock()   # 起停进程；RLock：_start_locked 里会调 _stop
        self._write_lock = threading.Lock()    # stdin 一次一行
        self._state_lock = threading.Lock()    # pending / chunks / 计数
        self._id_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, _Waiter] = {}
        self._chunks: dict[str, list[str]] = {}
        self._stderr: deque[str] = deque(maxlen=_ACP_STDERR_LINES)
        self._proc = None
        self._gen = 0       # 每次起停都加一，reader 只对自己那一代负责
        self._alive = False
        self._inflight = 0  # 进行中的 chat 数，空闲关闭要看它
        self._last_used = time.time()
        self._monitor: threading.Thread | None = None
        # PR_SET_PDEATHSIG 是在“创建子进程的那个线程”退出时触发的（不是进程），
        # 从临时的工作线程里 Popen，那个线程一结束 agy 就被 SIGKILL。所以固定用一个常驻线程来起进程。
        self._spawn_q: queue.Queue = queue.Queue()
        self._spawner: threading.Thread | None = None

    # ---- 进程起停 ---------------------------------------------------------

    def _tail(self) -> str:
        return "".join(self._stderr).strip()[-2000:]

    def _ensure(self):
        with self._start_lock:
            p = self._proc
            if p is not None and self._alive and p.poll() is None:
                return
            self._start_locked()

    def _start_locked(self):
        _AGY_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [os.environ.get("SRH_AGY_BIN", "agy"), "acp", "--disable-telemetry"]
        agent = os.environ.get("SRH_AGY_AGENT", "").strip()
        if agent:
            self._write_agent_file(agent)
            cmd += ["--agent", agent]
        with self._state_lock:
            self._gen += 1
            gen = self._gen
            self._stderr.clear()
            self._chunks.clear()
        # agy 是 Go 程序，默认堆会涨到 1GB 以上才回收。11-26 实测 GOMEMLIMIT=600MiB 连跑 32 个会话峰值
        # 1339→877MB、速度不变；环境里已经设了 GOMEMLIMIT 就照用，SRH_AGY_GOMEMLIMIT= 设成空串可以关掉
        env = dict(os.environ)
        limit = os.environ.get("SRH_AGY_GOMEMLIMIT", "600MiB").strip()
        if limit and "GOMEMLIMIT" not in env:
            env["GOMEMLIMIT"] = limit
        p = self._spawn(cmd, env)
        self._proc = p
        self._alive = True
        threading.Thread(target=self._read_loop, args=(p, gen), daemon=True).start()
        threading.Thread(target=self._err_loop, args=(p, gen), daemon=True).start()
        self._ensure_monitor()
        try:
            self._request("initialize", {"protocolVersion": 1, "clientCapabilities": {},
                                         "clientInfo": {"name": "verna-srh", "version": "1"}},
                          _ACP_INIT_TIMEOUT)
        except TimeoutError:
            self._stop()
            raise RuntimeError(f"agy acp initialize 超时（{_ACP_INIT_TIMEOUT} 秒），已停掉进程") from None
        except Exception:
            self._stop()
            raise

    def _write_agent_file(self, name: str):
        p = _AGY_DIR / ".agents" / "agents" / f"{name}.md"
        body = _AGENT_TMPL.format(name=name)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if p.read_text() == body:
                return
        except OSError:
            pass
        p.write_text(body)

    def _spawn(self, cmd: list[str], env: dict | None = None):
        """让常驻的派生线程去 Popen（见 __init__ 里 PDEATHSIG 的说明）。"""
        with self._state_lock:
            if self._spawner is None or not self._spawner.is_alive():
                self._spawner = threading.Thread(target=self._spawn_loop, daemon=True)
                self._spawner.start()
        job = {"cmd": cmd, "env": env, "done": threading.Event(), "proc": None, "err": None}
        self._spawn_q.put(job)
        if not job["done"].wait(60):
            raise RuntimeError("agy acp 派生线程没有响应")
        if job["err"] is not None:
            raise job["err"]
        return job["proc"]

    def _spawn_loop(self):
        while True:
            job = self._spawn_q.get()
            try:
                job["proc"] = subprocess.Popen(
                    job["cmd"], cwd=str(_AGY_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True,
                    preexec_fn=_set_pdeathsig, env=job.get("env"))
            except BaseException as e:  # 包括 FileNotFoundError
                job["err"] = e
            job["done"].set()

    def _ensure_monitor(self):
        with self._state_lock:
            if self._monitor is not None and self._monitor.is_alive():
                return
            self._monitor = threading.Thread(target=self._idle_loop, daemon=True)
            self._monitor.start()

    def _idle_loop(self):
        """没有调用、且距上次使用超过 SRH_AGY_IDLE_SECS 就停进程（管线常一睡几个小时）。"""
        while True:
            idle = _idle_secs()
            time.sleep(max(0.5, min(5.0, idle / 4.0)))
            with self._start_lock:  # 和 chat 的 _inflight += 1 互斥，别把刚起来的调用杀掉
                with self._state_lock:
                    due = (self._proc is not None and self._inflight == 0
                           and time.time() - self._last_used > idle)
                if due:
                    self._stop()

    def _stop(self):
        """关 stdin，等 10 秒；不退就对进程组 SIGKILL，并 wait() 回收防僵尸。可重复调用。"""
        with self._start_lock:
            with self._state_lock:
                p, gen = self._proc, self._gen
                if p is None:
                    return
                self._proc = None
                self._alive = False
                self._gen += 1
                self._chunks.clear()
        if p.stdin is not None:
            try:
                p.stdin.close()
            except Exception:
                pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, _SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for f in (p.stdout, p.stderr):
            try:
                f.close()
            except Exception:
                pass
        self._fail_gen(gen, RuntimeError(f"agy acp 已停止：{self._tail()}"))

    def stop(self):
        self._stop()

    # ---- 读写 -------------------------------------------------------------

    def _write(self, msg: dict) -> bool:
        p = self._proc
        if p is None or p.stdin is None:
            return False
        data = json.dumps(msg, ensure_ascii=False) + "\n"
        with self._write_lock:
            try:
                p.stdin.write(data)
                p.stdin.flush()
            except (OSError, ValueError):
                return False
        return True

    def _err_loop(self, p, gen):
        try:
            for line in p.stderr:
                self._stderr.append(line)
        except Exception:
            pass
        finally:
            try:
                p.stderr.close()
            except Exception:
                pass

    def _read_loop(self, p, gen):
        try:
            for line in p.stdout:
                if self._gen != gen:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                try:
                    mid, method = msg.get("id"), msg.get("method")
                    if method is None and mid is not None:
                        self._on_response(mid, msg)
                    elif method is not None and mid is not None:
                        self._on_agent_request(mid, method, msg.get("params") or {})
                    elif method is not None:
                        self._on_update(msg.get("params") or {})
                except Exception:
                    continue  # 单条消息格式怪就丢掉，别把整条读线程带崩
        except Exception:
            pass
        finally:
            try:
                p.stdout.close()
            except Exception:
                pass
            self._on_eof(p, gen)

    def _on_response(self, mid, msg):
        with self._state_lock:
            w = self._pending.pop(mid, None)
        if w is None:
            return
        if "error" in msg:
            e = msg.get("error") or {}
            w.error = _RpcError(w.method, e.get("code"), e.get("message"))
        else:
            w.result = msg.get("result")
        w.event.set()

    def _on_agent_request(self, mid, method, params):
        if method == "session/request_permission":
            oid = None
            for o in params.get("options") or []:
                if isinstance(o, dict) and str(o.get("kind", "")).lower().startswith("reject"):
                    oid = o.get("optionId") or o.get("id")
                    break
            result = ({"outcome": {"outcome": "selected", "optionId": oid}} if oid
                      else {"outcome": {"outcome": "cancelled"}})
            self._write({"jsonrpc": "2.0", "id": mid, "result": result})
        else:
            self._write({"jsonrpc": "2.0", "id": mid,
                         "error": {"code": -32601, "message": f"Method not found: {method}"}})

    def _on_update(self, params):
        # params.update.sessionUpdate == "agent_message_chunk" 的 content.text 是正文片段
        u = params.get("update") or {}
        if not isinstance(u, dict) or u.get("sessionUpdate") != "agent_message_chunk":
            return
        content = u.get("content")
        text = content.get("text") if isinstance(content, dict) else None
        if not text:
            return
        sid = params.get("sessionId")
        with self._state_lock:
            if sid in self._chunks:  # 会话已经关掉/超时清掉了就不要再建回来
                self._chunks[sid].append(text)

    def _on_eof(self, p, gen):
        with self._state_lock:
            if self._gen == gen:
                self._alive = False
        if p.stdin is not None:  # 进程都退出了，输入管道也关掉，别留未关闭的文件对象
            try:
                p.stdin.close()
            except Exception:
                pass
        code = p.poll()
        if code is None:
            try:
                code = p.wait(timeout=5)
            except Exception:
                code = None
        self._fail_gen(gen, RuntimeError(f"agy acp 退出了（code={code}）：{self._tail()}"))

    def _fail_gen(self, gen: int, err: Exception):
        with self._state_lock:
            ws = [(k, w) for k, w in self._pending.items() if w.gen == gen]
            for k, _ in ws:
                self._pending.pop(k, None)
        for _, w in ws:
            if not w.event.is_set():
                w.error = err
                w.event.set()

    # ---- 请求 -------------------------------------------------------------

    def _send(self, method: str, params: dict, ensure: bool = True) -> tuple[int, _Waiter]:
        if ensure:
            self._ensure()
        else:
            p = self._proc
            if p is None or not self._alive:
                raise RuntimeError("agy acp 没在运行")
        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        with self._state_lock:
            w = _Waiter(method, self._gen)
            self._pending[rid] = w
        if not self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}):
            with self._state_lock:
                self._pending.pop(rid, None)
            raise RuntimeError(f"agy acp 写 stdin 失败（{method}）：{self._tail() or '进程已退出'}")
        return rid, w

    def _request(self, method: str, params: dict, timeout: float, ensure: bool = True):
        rid, w = self._send(method, params, ensure=ensure)
        if not w.event.wait(timeout):
            with self._state_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"{method} 超时（{timeout} 秒）")
        if w.error is not None:
            raise w.error
        return w.result

    def _notify(self, method: str, params: dict):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    # ---- 一次调用 ---------------------------------------------------------

    def chat(self, family: str, level: str | None, prompt: str, timeout: float) -> tuple[str, dict]:
        with self._start_lock:  # 见 _idle_loop：计数要在空闲关闭检查的同一把锁里加
            with self._state_lock:
                self._inflight += 1
        sid = None
        try:
            sid, opts = self._open_session()
            self._configure(sid, opts, family, level)
            return self._prompt(sid, prompt, timeout)
        finally:
            if sid is not None:
                self._close_session(sid)
            with self._state_lock:
                self._inflight -= 1
                self._last_used = time.time()

    def _open_session(self) -> tuple[str, list]:
        try:
            res = self._request("session/new", {"cwd": str(_AGY_DIR.resolve()), "mcpServers": []},
                                _ACP_NEW_TIMEOUT)
        except TimeoutError:
            self._stop()
            raise RuntimeError(f"agy acp session/new 超时（{_ACP_NEW_TIMEOUT} 秒），已停掉进程") from None
        if not isinstance(res, dict) or not res.get("sessionId"):
            raise RuntimeError(f"session/new 返回异常：{res!r}")
        sid = res["sessionId"]
        with self._state_lock:
            self._chunks[sid] = []
        return sid, res.get("configOptions") or []

    def _set_option(self, sid: str, opts: list, cid: str, value) -> tuple[list, bool]:
        """返回 (options, 是否拿到了服务器的新 configOptions)。"""
        res = self._request("session/set_config_option",
                            {"sessionId": sid, "configId": cid, "value": value}, _ACP_SET_TIMEOUT)
        if isinstance(res, dict) and isinstance(res.get("configOptions"), list):
            return res["configOptions"], True
        # 没回新的 configOptions 就本地改一下 currentValue
        out = []
        for o in opts:
            if isinstance(o, dict) and o.get("id") == cid:
                o = {**o, "currentValue": value}
            out.append(o)
        return out, False

    def _configure(self, sid: str, opts: list, family: str, level: str | None):
        # 先设 model 再设 level：level 的可选值跟着模型走
        lvl_vals_known, force_level = True, False  # session/new 给的是准的；没拿到新 options 就不校验档位
        model_opt = _find_opt(opts, "model")
        if model_opt is not None:
            vals = _opt_values(model_opt)
            if vals and family not in vals:
                raise AgyModelError(f"agy 没有模型 {family}，可选：{'、'.join(map(str, vals))}")
            if model_opt.get("currentValue") != family:
                opts, lvl_vals_known = self._set_option(sid, opts, "model", family)
                force_level = not lvl_vals_known  # 换模型后档位可能被重置，不知道新 options 就照设一遍
        if level:
            lvl_opt = _find_opt(opts, "thought_level")
            if lvl_opt is not None:
                vals = _opt_values(lvl_opt) if lvl_vals_known else []
                if vals and level not in vals:
                    raise AgyModelError(f"模型 {family} 不支持档位 {level}，可选：{'、'.join(map(str, vals))}")
                if force_level or lvl_opt.get("currentValue") != level:
                    self._set_option(sid, opts, "thought_level", level)

    def _prompt(self, sid: str, prompt: str, timeout: float) -> tuple[str, dict]:
        rid, w = self._send("session/prompt",
                            {"sessionId": sid, "prompt": [{"type": "text", "text": prompt}]})
        if not w.event.wait(timeout):
            self._notify("session/cancel", {"sessionId": sid})
            w.event.wait(10)  # 取消后最多再等 10 秒拿返回，拿不到也照样报超时
            with self._state_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"agy acp prompt 超时（{timeout} 秒）")
        if w.error is not None:
            raise w.error
        res = w.result if isinstance(w.result, dict) else {}
        stop = res.get("stopReason")
        if stop != "end_turn":
            raise RuntimeError(f"agy acp 没正常完成（stopReason={stop}）")
        with self._state_lock:
            text = "".join(self._chunks.get(sid) or []).strip()
        if not text:
            raise RuntimeError("agy acp 返回空正文")
        if text.startswith("⚠"):
            # agy 有时把失败当正文发出来，stopReason 还是 end_turn
            raise RuntimeError(f"agy 报错：{text[:300]}")
        u = res.get("usage") or {}
        info = {"input_tokens": u.get("inputTokens", 0), "output_tokens": u.get("outputTokens", 0),
                "thinking_tokens": u.get("thoughtTokens", 0), "cache_read_tokens": u.get("cachedReadTokens", 0)}
        return text, info

    def _close_session(self, sid: str):
        try:
            self._request("session/close", {"sessionId": sid}, _ACP_CLOSE_TIMEOUT, ensure=False)
        except Exception:
            pass  # 关会话失败只忽略，正文已经拿到了
        finally:
            with self._state_lock:
                self._chunks.pop(sid, None)


_ACP: _AcpClient | None = None
_ACP_LOCK = threading.Lock()


def _client() -> _AcpClient:
    global _ACP
    with _ACP_LOCK:
        if _ACP is None:
            _ACP = _AcpClient()
        return _ACP


def shutdown_agy():
    """停掉常驻 agy acp 进程（空闲关闭、进程重启前用）。可重复调用。"""
    c = _ACP
    if c is not None:
        c.stop()


atexit.register(shutdown_agy)


def _agy(model: str, messages: list[dict], timeout: int, retries: int, tag: str) -> str:
    """agy 走常驻的 agy acp（一个 Python 进程一个 agy），这里只留重试、熔断、限并发和记录。
    agy 不支持温度等采样参数，多样性靠提示词（一次要多个候选）。"""
    prompt = "\n\n".join(m["content"] for m in messages if m["role"] in ("system", "user"))
    _AGY_DIR.mkdir(exist_ok=True)
    family, level = _split_agy_model(model)
    err = None
    for i in range(retries):
        if agy_auth_down():
            raise AgyAuthError(f"agy 未登录（{_AUTH_COOLDOWN // 60} 分钟内不再调用），等用户重新登录")
        quota_sleep = False
        with _agy_slot():
            t = time.time()
            try:
                text, info = _client().chat(family, level, prompt, timeout)
                _log("agy:" + model, time.time() - t, info, tag)
                return text
            except AgyModelError:
                raise  # 模型/档位不存在，重试没有用
            except Exception as e:
                err = repr(e)[:300]
                _log("agy:" + model, time.time() - t, {"error": err[:200], "try": i + 1}, tag)  # 失败也记，卡住时好查
                if note_agy_auth_fail(err):
                    raise AgyAuthError(f"agy:{model} 未登录，{_AUTH_COOLDOWN // 60} 分钟内不再调用：{err}") from None
                if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    quota_sleep = True
        if quota_sleep:
            time.sleep(600)  # 额度用完：释放槽位后再等十分钟
        time.sleep(2 ** i * 5)
    raise RuntimeError(f"agy:{model} 调用失败：{err}")


def chat(model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048,
         retries: int = 4, timeout: int = 300, tag: str = "", **extra) -> str:
    if model.startswith("agy:"):
        return _agy(model[4:], messages, timeout, retries, tag)
    core, flags = _flags(model)
    name, base, key = _endpoint(core)
    body = {"model": name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
    if "nothink" in flags:
        body["thinking"] = {"type": "disabled"}
    # Floway 前面的 Cloudflare 会拦 Python-urllib 的默认 UA（error 1010）
    headers = {"Content-Type": "application/json", "User-Agent": "verna-pipeline/0.1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    err = None
    for i in range(retries):
        t = time.time()
        try:
            req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            u = j.get("usage") or {}
            _log(label(model), time.time() - t, {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}, tag)
            text = j["choices"][0]["message"].get("content") or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:  # 网络抖动、超时
            err = repr(e)
        time.sleep(2 ** i * 3)
    raise RuntimeError(f"{name} 调用失败：{err}")


def _repair_json(s: str) -> str:
    """修两种常见的坏 JSON：字符串里没转义的英文双引号（中文引文写成 "好" 而不是 “好”），和 } ] 前多余的逗号。
    字符串里遇到 "：后面（跳过空白）紧跟 , : } ] 才算收尾，否则当成正文里的引号转义掉。"""
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                out.append(s[i:i + 2]); i += 2; continue
            if ch == '"':
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j >= n or s[j] in ",:}]":
                    in_str = False; out.append(ch)
                else:
                    out.append('\\"')
                i += 1; continue
            if ch == "\n":
                out.append("\\n"); i += 1; continue
            out.append(ch); i += 1; continue
        if ch == '"':
            in_str = True
        out.append(ch); i += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


def parse_json(text: str) -> dict:
    """容忍 ```json 包裹和前后废话。11-06：解析失败时修一次引号和多余逗号再试（分级一天坏 10 批左右，多是正文里的英文引号）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"没有 JSON：{text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        try:
            return json.loads(_repair_json(m.group(0)))
        except json.JSONDecodeError:
            raise e from None


def ask_json(model: str, prompt: str, **kw) -> dict:
    text = chat(model, [{"role": "user", "content": prompt}], temperature=kw.pop("temperature", 0.0), **kw)
    return parse_json(text)
