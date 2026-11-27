"""agy acp 常驻客户端的测试（标准库 unittest，用 tests/fake_agy_acp.py 冒充 agy）。"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srh import llm  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_agy_acp.py"
MODEL = "agy:gemini-4-argon-medium"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # 僵尸算已退出（客户端会 wait 回收，这里只是保险）
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state != "Z"
    except OSError:
        return False


class AcpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pidfile = self.dir / "pids.txt"
        self.fakelog = self.dir / "fake.log"
        self.usage = self.dir / "usage.jsonl"
        os.chmod(FAKE, 0o755)
        self.env = mock.patch.dict(os.environ, {
            "SRH_AGY_BIN": str(FAKE),
            "FAKE_AGY_MODE": "normal",
            "FAKE_AGY_PIDFILE": str(self.pidfile),
            "FAKE_AGY_LOG": str(self.fakelog),
            "SRH_AGY_MAX_CONCURRENCY": "32",
            "SRH_AGY_IDLE_SECS": "600",
            "SRH_AGY_AGENT": "",
        })
        self.env.start()
        llm.shutdown_agy()
        llm._ACP = None
        llm._AUTH_DOWN["until"] = 0.0
        self.patches = [
            mock.patch.object(llm, "_AGY_DIR", self.dir / "agy-work"),
            mock.patch.object(llm, "_AGY_SLOT_DIR", self.dir / "slots"),
            mock.patch.object(llm, "_AGY_MAX_CONCURRENCY", 32),
            mock.patch.object(llm, "USAGE_LOG", self.usage),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        llm.shutdown_agy()
        llm._ACP = None
        llm._AUTH_DOWN["until"] = 0.0
        for p in reversed(self.patches):
            p.stop()
        self.env.stop()
        left = [pid for pid in self.pids() if _alive(pid)]
        self.assertEqual(left, [], "测试结束后还有假 agy 进程")
        self.tmp.cleanup()

    # ---- 小工具 -----------------------------------------------------------

    def pids(self) -> list[int]:
        if not self.pidfile.exists():
            return []
        return [int(x) for x in self.pidfile.read_text().split()]

    def rows(self) -> list[dict]:
        if not self.fakelog.exists():
            return []
        return [json.loads(x) for x in self.fakelog.read_text().splitlines() if x.strip()]

    def methods(self) -> list[str]:
        return [r.get("method") for r in self.rows() if r.get("method")]

    def wait_dead(self, pid: int, timeout: float = 15) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if not _alive(pid):
                return True
            time.sleep(0.2)
        return False

    # ---- 1. 并发共用进程 ------------------------------------------------

    def test_concurrent_sessions_share_one_process(self):
        outs = {}

        def work(i):
            outs[i] = llm.chat(MODEL, [{"role": "user", "content": f"MARK-{i} 请回答"}],
                               retries=1, timeout=60)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(90)

        self.assertEqual(len(outs), 8)
        for i in range(8):
            self.assertIn(f"MARK-{i}", outs[i])
            self.assertIn("model=gemini-4-argon", outs[i])
            self.assertIn("level=medium", outs[i])
        self.assertEqual(len(self.pids()), 1)
        sets = [r["params"] for r in self.rows() if r.get("method") == "session/set_config_option"]
        self.assertTrue(any(s.get("configId") == "model" and s.get("value") == "gemini-4-argon" for s in sets))
        self.assertTrue(any(s.get("configId") == "thought_level" and s.get("value") == "medium" for s in sets))

    # ---- 2. usage 字段 ---------------------------------------------------

    def test_usage_fields(self):
        llm.chat(MODEL, [{"role": "user", "content": "MARK-u"}], retries=1)
        rec = json.loads(self.usage.read_text().splitlines()[-1])
        self.assertEqual(rec["model"], MODEL)
        self.assertEqual(rec["input_tokens"], 9435)
        self.assertEqual(rec["output_tokens"], 7)
        self.assertEqual(rec["thinking_tokens"], 5)
        self.assertEqual(rec["cache_read_tokens"], 3)
        self.assertNotIn("exit_hang", rec)

    # ---- 3. 模型不存在：只调一次 ----------------------------------------

    def test_unknown_model_no_retry(self):
        t0 = time.time()
        with self.assertRaises(llm.AgyModelError) as cm:
            llm.chat("agy:gemini-9-nope-medium", [{"role": "user", "content": "x"}], retries=3)
        self.assertLess(time.time() - t0, 4)
        self.assertIn("gemini-4-argon", str(cm.exception))  # 错误信息里带可选值
        self.assertEqual(self.methods().count("session/new"), 1)
        self.assertNotIn("session/prompt", self.methods())

    # ---- 4. prompt 期间崩溃：本次失败，下次重起 --------------------------

    def test_crash_then_restart(self):
        os.environ["FAKE_AGY_MODE"] = "crash_on_prompt"
        with self.assertRaises(Exception):
            llm.chat(MODEL, [{"role": "user", "content": "MARK-c1"}], retries=1, timeout=30)
        os.environ["FAKE_AGY_MODE"] = "normal"
        text = llm.chat(MODEL, [{"role": "user", "content": "MARK-c2"}], retries=1, timeout=30)
        self.assertIn("MARK-c2", text)
        self.assertEqual(len(self.pids()), 2)

    # ---- 5. ⚠ 正文当失败 -------------------------------------------------

    def test_warn_text_is_failure(self):
        os.environ["FAKE_AGY_MODE"] = "warn"
        with self.assertRaises(RuntimeError) as cm:
            llm._client().chat("gemini-4-argon", "medium", "MARK-w", 30)
        self.assertIn("⚠", str(cm.exception))

    # ---- 6. 超时发 cancel ------------------------------------------------

    def test_timeout_sends_cancel(self):
        os.environ["FAKE_AGY_MODE"] = "slow"
        t0 = time.time()
        with self.assertRaises(TimeoutError):
            llm._client().chat("gemini-4-argon", "medium", "MARK-t", 1)
        self.assertLess(time.time() - t0, 15)
        self.assertIn("session/cancel", self.methods())

    # ---- 7. 额度错误的文字保留 ------------------------------------------

    def test_quota_error_text(self):
        os.environ["FAKE_AGY_MODE"] = "quota"
        with self.assertRaises(RuntimeError) as cm:
            llm._client().chat("gemini-4-argon", "medium", "MARK-q", 30)
        self.assertIn("RESOURCE_EXHAUSTED", str(cm.exception))

    # ---- 8. 鉴权失败与熔断 ----------------------------------------------

    def test_auth_failure_opens_circuit(self):
        for mode in ("auth_text", "auth_code"):
            os.environ["FAKE_AGY_MODE"] = mode
            llm.shutdown_agy()
            llm._ACP = None
            llm._AUTH_DOWN["until"] = 0.0
            with self.assertRaises(llm.AgyAuthError):
                llm.chat(MODEL, [{"role": "user", "content": "x"}], retries=2)
            self.assertTrue(llm.agy_auth_down(), mode)
            n = len(self.pids())
            with self.assertRaises(llm.AgyAuthError):
                llm.chat(MODEL, [{"role": "user", "content": "x"}], retries=2)
            self.assertEqual(len(self.pids()), n, f"{mode}：熔断中不该再起进程")

    # ---- 9. 空闲关闭 / shutdown_agy --------------------------------------

    def test_idle_shutdown(self):
        os.environ["SRH_AGY_IDLE_SECS"] = "1"
        llm.chat(MODEL, [{"role": "user", "content": "MARK-i1"}], retries=1)
        pids = self.pids()
        self.assertEqual(len(pids), 1)
        self.assertTrue(self.wait_dead(pids[0]), "空闲 1 秒后进程应该退出")

        llm.chat(MODEL, [{"role": "user", "content": "MARK-i2"}], retries=1)
        pids = self.pids()
        self.assertEqual(len(pids), 2)
        llm.shutdown_agy()
        llm.shutdown_agy()  # 可重复调用
        self.assertTrue(self.wait_dead(pids[1]), "shutdown_agy 后进程应该退出")

    # ---- 10. SRH_AGY_AGENT -----------------------------------------------

    def test_agent_file_and_flag(self):
        os.environ["SRH_AGY_AGENT"] = "srh-bare"
        llm.chat(MODEL, [{"role": "user", "content": "MARK-a"}], retries=1)
        p = self.dir / "agy-work" / ".agents" / "agents" / "srh-bare.md"
        self.assertTrue(p.exists())
        body = p.read_text()
        self.assertIn("name: srh-bare", body)
        self.assertIn("excludeDefaultComponents: true", body)
        self.assertIn("tools: []", body)
        argv = self.rows()[0]["argv"]
        self.assertIn("--agent", argv)
        self.assertEqual(argv[argv.index("--agent") + 1], "srh-bare")


    # ---- 11. GOMEMLIMIT ----------------------------------------------------

    def environ_of_fake(self) -> dict:
        pid = self.pids()[-1]
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return dict(x.decode().split("=", 1) for x in raw if b"=" in x)

    def test_gomemlimit_default(self):
        os.environ.pop("GOMEMLIMIT", None)
        llm.chat(MODEL, [{"role": "user", "content": "MARK-g"}], retries=1)
        self.assertEqual(self.environ_of_fake().get("GOMEMLIMIT"), "600MiB")

    def test_gomemlimit_off(self):
        os.environ.pop("GOMEMLIMIT", None)
        os.environ["SRH_AGY_GOMEMLIMIT"] = ""
        llm.chat(MODEL, [{"role": "user", "content": "MARK-g0"}], retries=1)
        self.assertNotIn("GOMEMLIMIT", self.environ_of_fake())

if __name__ == "__main__":
    unittest.main()
