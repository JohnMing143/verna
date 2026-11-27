#!/usr/bin/env python3
"""Tests for the Verna vote API.

Run from the repository root with:
    python -m unittest vote/server/test_server.py
"""

import contextlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import admin  # noqa: E402
import server as srv  # noqa: E402


def new_judge():
    return "j" + uuid.uuid4().hex[:12]


def serve_in_background(httpd):
    thread = threading.Thread(
        target=lambda: httpd.serve_forever(poll_interval=0.05), daemon=True
    )
    thread.start()
    return thread


def item(item_id, genre):
    return {
        "id": item_id,
        "genre": genre,
        "source": "这是一段用于测试的原文，**里面包含加粗**。\n\n第二段原文。",
        "a": "这是改写 A。",
        "b": "这是改写 B。",
        "a_system": "system-a",
        "b_system": "system-b",
        "kind": "normal",
        "meta": {"secret": "hidden-meta"},
    }


ITEMS = [
    item("it01", "职场"),
    item("it02", "职场"),
    item("it03", "职场"),
    item("it04", "评论"),
    item("it05", "评论"),
    item("it06", "评论"),
    item("it07", "科普"),
    item("it08", "科普"),
]

BASE_TEXT = (
    "这是一段用来测试投稿接口的示例文字，汉字数量需要超过五十个，"
    "重复提交会被识别为同一篇，因此每条内容都要略有不同。"
)


@contextlib.contextmanager
def running_server(items=None, salt="test-salt"):
    tmpdir = tempfile.mkdtemp(prefix="verna-test-")
    db_path = os.path.join(tmpdir, "verna.db")
    if items is not None:
        jsonl_path = os.path.join(tmpdir, "items.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for record in items:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        admin.import_items(db_path, jsonl_path)
    else:
        srv.init_db(db_path)

    old_salt = os.environ.get("VERNA_IP_SALT")
    os.environ["VERNA_IP_SALT"] = salt
    httpd = None
    thread = None
    try:
        httpd = srv.make_server(db_path, port=0)
        thread = serve_in_background(httpd)
        base = "http://127.0.0.1:%d/api" % httpd.server_address[1]
        yield db_path, base, tmpdir
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(5)
        if old_salt is None:
            os.environ.pop("VERNA_IP_SALT", None)
        else:
            os.environ["VERNA_IP_SALT"] = old_salt
        shutil.rmtree(tmpdir, ignore_errors=True)


class ApiTestCase(unittest.TestCase):
    items = ITEMS

    def setUp(self):
        self._ctx = running_server(self.items)
        self.db, self.base, self.tmpdir = self._ctx.__enter__()
        self._ip_seq = 0

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    # -- helpers ---------------------------------------------------------

    def request(self, method, path, payload=None, raw=None, ctype="application/json", ip=None):
        if ip is None:
            self._ip_seq += 1
            ip = "198.51.100.%d" % (self._ip_seq % 250 + 1)
        headers = {"X-Forwarded-For": ip}
        data = None
        if raw is not None:
            data = raw
            if ctype is not None:
                headers["Content-Type"] = ctype
        elif payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if ctype is not None:
                headers["Content-Type"] = ctype
        req = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read()
                return response.status, json.loads(body.decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                parsed = json.loads(body.decode("utf-8"))
            except ValueError:
                parsed = {"raw": body.decode("utf-8", "replace")}
            return exc.code, parsed

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, payload=None, **kwargs):
        return self.request("POST", path, payload=payload, **kwargs)

    def vote_payload(self, judge, item_id="it01", **overrides):
        payload = {
            "item": item_id,
            "judge": judge,
            "choice": "a",
            "left": "a",
            "ms": 1200,
            "reasons": ["读着顺"],
            "note": "",
        }
        payload.update(overrides)
        return payload

    def submit_payload(self, judge, text=None, **overrides):
        payload = {
            "judge": judge,
            "text": text if text is not None else BASE_TEXT,
            "model": "",
            "scene": "",
            "nickname": "",
            "consent": True,
        }
        payload.update(overrides)
        return payload

    def next_ids(self, judge, n=10, **kwargs):
        status, data = self.get(
            "/next?judge=%s&n=%d" % (judge, n), **kwargs
        )
        self.assertEqual(status, 200, data)
        self.assertTrue(data.get("ok"))
        return [row["id"] for row in data["items"]]

    def query(self, sql, params=()):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


class NextTests(ApiTestCase):
    def test_next_shape_and_no_private_fields(self):
        judge = new_judge()
        status, data = self.get("/next?judge=%s&n=5" % judge)
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["items"]), 5)
        raw = json.dumps(data, ensure_ascii=False)
        for leaked in ("system-a", "system-b", "hidden-meta", "a_system", "b_system", "probe_answer"):
            self.assertNotIn(leaked, raw)
        for row in data["items"]:
            self.assertEqual(set(row.keys()), {"id", "genre", "source", "a", "b"})

    def test_next_batch_is_unique(self):
        ids = self.next_ids(new_judge())
        self.assertEqual(len(ids), len(set(ids)))

    def test_next_records_judge_and_from_tag(self):
        judge = new_judge()
        status, _ = self.get("/next?judge=%s&n=2&from=wechat" % judge, ip="203.0.113.7")
        self.assertEqual(status, 200)
        rows = self.query("SELECT * FROM judges WHERE id = ?", (judge,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["ip_hash"]), 16)
        self.assertNotEqual(rows[0]["ip_hash"], "203.0.113.7")
        self.assertEqual(rows[0]["from_tag"], "wechat")
        # First sighting already happened, so a later tag must not replace it.
        self.get("/next?judge=%s&n=2&from=other" % judge)
        self.assertEqual(
            self.query("SELECT from_tag FROM judges WHERE id = ?", (judge,))[0]["from_tag"],
            "wechat",
        )

    def test_from_tag_truncated(self):
        judge = new_judge()
        long_tag = "x" * 80
        status, _ = self.get("/next?judge=%s&from=%s" % (judge, long_tag))
        self.assertEqual(status, 200)
        stored = self.query("SELECT from_tag FROM judges WHERE id = ?", (judge,))[0]["from_tag"]
        self.assertEqual(stored, "x" * 40)

    def test_genre_cap(self):
        for _ in range(5):
            judge = new_judge()
            status, data = self.get("/next?judge=%s&n=5" % judge)
            self.assertEqual(status, 200)
            self.assertEqual(len(data["items"]), 5)
            counts = {}
            for row in data["items"]:
                counts[row["genre"]] = counts.get(row["genre"], 0) + 1
            for genre, count in counts.items():
                self.assertLessEqual(count, 2, "genre %s appeared %d times" % (genre, count))

    def test_n_is_clamped(self):
        judge = new_judge()
        status, data = self.get("/next?judge=%s&n=50" % judge)
        self.assertEqual(status, 200)
        self.assertLessEqual(len(data["items"]), 10)
        status, data = self.get("/next?judge=%s&n=0" % judge)
        self.assertEqual(status, 200)
        self.assertEqual(len(data["items"]), 1)
        status, data = self.get("/next?judge=%s&n=abc" % judge)
        self.assertEqual(status, 200)
        self.assertEqual(len(data["items"]), 5)

    def test_bad_judge(self):
        for bad in ("", "x123456789", "jABCDEFGH", "j123", "j" + "a" * 41):
            status, data = self.get("/next?judge=%s" % bad)
            self.assertEqual(status, 400, bad)
            self.assertEqual(data["error"], "bad_judge")

    def test_low_vote_items_first(self):
        heavy = new_judge()
        status, _ = self.post("/vote", self.vote_payload(heavy, "it01"))
        self.assertEqual(status, 200)
        ids = self.next_ids(new_judge(), n=3)
        self.assertNotIn("it01", ids)


class VoteTests(ApiTestCase):
    def test_vote_then_hidden_from_next(self):
        judge = new_judge()
        self.assertIn("it01", self.next_ids(judge))
        status, data = self.post("/vote", self.vote_payload(judge, "it01"))
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertNotIn("it01", self.next_ids(judge))

    def test_vote_overwrite(self):
        judge = new_judge()
        self.post("/vote", self.vote_payload(judge, "it01"))
        status, data = self.post(
            "/vote",
            self.vote_payload(
                judge,
                "it01",
                choice="src",
                left="b",
                ms=9,
                reasons=["两份都不错"],
                note="补一句",
            ),
        )
        self.assertEqual(status, 200, data)
        rows = self.query("SELECT * FROM votes WHERE item_id = 'it01' AND judge_id = ?", (judge,))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["choice"], "src")
        self.assertEqual(row["left_side"], "b")
        self.assertEqual(row["ms"], 9)
        self.assertEqual(json.loads(row["reasons"]), ["两份都不错"])
        self.assertEqual(row["note"], "补一句")

    def test_unvote_restores_item(self):
        judge = new_judge()
        self.post("/vote", self.vote_payload(judge, "it01"))
        status, data = self.post("/unvote", {"item": "it01", "judge": judge})
        self.assertEqual(status, 200, data)
        self.assertIn("it01", self.next_ids(judge))
        self.assertEqual(len(self.query("SELECT 1 FROM votes WHERE judge_id = ?", (judge,))), 0)

    def test_unvote_unknown_is_ok(self):
        status, data = self.post("/unvote", {"item": "it01", "judge": new_judge()})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])

    def test_skip_hides_item(self):
        judge = new_judge()
        status, data = self.post("/skip", {"item": "it03", "judge": judge})
        self.assertEqual(status, 200, data)
        self.assertNotIn("it03", self.next_ids(judge))
        self.assertEqual(
            self.query("SELECT COUNT(*) FROM votes WHERE judge_id = ?", (judge,))[0][0],
            0,
        )

    def test_vote_validation(self):
        judge = new_judge()
        cases = [
            ({"judge": "bad"}, "bad_judge"),
            ({"judge": 123}, "bad_judge"),
            ({"item": ""}, "bad_item"),
            ({"item": 5}, "bad_item"),
            ({"choice": "c"}, "bad_choice"),
            ({"choice": "tie"}, "bad_choice"),
            ({"choice": "bad"}, "bad_choice"),
            ({"choice": None}, "bad_choice"),
            ({"left": "c"}, "bad_left"),
            ({"left": "left"}, "bad_left"),
            ({"ms": -1}, "bad_ms"),
            ({"ms": 86400001}, "bad_ms"),
            ({"ms": 1.5}, "bad_ms"),
            ({"ms": True}, "bad_ms"),
            ({"ms": "1000"}, "bad_ms"),
            ({"reasons": "读着顺"}, "bad_reasons"),
            ({"reasons": ["a"] * 13}, "bad_reasons"),
            ({"reasons": ["x" * 41]}, "bad_reasons"),
            ({"reasons": [1]}, "bad_reasons"),
            ({"note": "x" * 301}, "bad_note"),
            ({"note": 1}, "bad_note"),
            ({"item": "nope"}, None),
        ]
        for override, expected in cases:
            with self.subTest(override=override):
                payload = self.vote_payload(judge, "it01")
                payload.update(override)
                status, data = self.post("/vote", payload)
                if expected is None:
                    self.assertEqual(status, 404, data)
                    self.assertEqual(data["error"], "item_not_found")
                else:
                    self.assertEqual(status, 400, data)
                    self.assertEqual(data["error"], expected)

    def test_vote_retired_item(self):
        admin.retire_items(self.db, ["it02"])
        status, data = self.post("/vote", self.vote_payload(new_judge(), "it02"))
        self.assertEqual(status, 400)
        self.assertEqual(data["error"], "item_inactive")

    def test_vote_reasons_limit_boundaries(self):
        judge = new_judge()
        status, data = self.post(
            "/vote",
            self.vote_payload(judge, "it01", reasons=["y" * 40] * 12, note="n" * 300),
        )
        self.assertEqual(status, 200, data)


class HttpTests(ApiTestCase):
    def test_unknown_path(self):
        status, data = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertEqual(data, {"ok": False, "error": "not_found"})

    def test_method_not_allowed(self):
        status, data = self.get("/vote")
        self.assertEqual(status, 405)
        self.assertEqual(data["error"], "method_not_allowed")
        status, data = self.post("/next", {})
        self.assertEqual(status, 405)
        self.assertEqual(data["error"], "method_not_allowed")

    def test_bad_content_type(self):
        judge = new_judge()
        raw = json.dumps(self.vote_payload(judge)).encode("utf-8")
        status, data = self.post("/vote", raw=raw, ctype="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(data["error"], "bad_content_type")
        status, data = self.post("/vote", raw=raw, ctype=None)
        self.assertEqual(status, 415)

    def test_bad_json(self):
        for raw in (b"{", b"[]", b"null", b"\xff\xfe"):
            with self.subTest(raw=raw):
                status, data = self.post("/vote", raw=raw)
                self.assertEqual(status, 400)
                self.assertEqual(data["error"], "bad_json")

    def test_body_size_limit(self):
        # Exactly 16 KiB is accepted by the size check (and then rejected
        # as a bad judge), one byte more is a 413.
        ok_raw = b'{"pad":"' + b"a" * (16384 - 10) + b'"}'
        self.assertEqual(len(ok_raw), 16384)
        status, data = self.post("/vote", raw=ok_raw)
        self.assertEqual(status, 400)
        self.assertEqual(data["error"], "bad_judge")

        big_raw = b'{"pad":"' + b"a" * (16385 - 10) + b'"}'
        self.assertEqual(len(big_raw), 16385)
        status, data = self.post("/vote", raw=big_raw)
        self.assertEqual(status, 413)
        self.assertEqual(data["error"], "too_large")

    def test_stats(self):
        status, data = self.get("/stats")
        self.assertEqual(status, 200)
        self.assertEqual(
            data, {"ok": True, "judges": 0, "votes": 0, "items": 8, "submissions": 0}
        )
        judge = new_judge()
        self.next_ids(judge, n=1)
        self.post("/vote", self.vote_payload(judge, "it01"))
        status, data = self.get("/stats")
        self.assertEqual(data["judges"], 1)
        self.assertEqual(data["votes"], 1)


class SubmitTests(ApiTestCase):
    def test_submit_success_and_pending(self):
        judge = new_judge()
        status, data = self.post("/submit", self.submit_payload(judge))
        self.assertEqual(status, 200, data)
        self.assertTrue(data["ok"])
        self.assertEqual(data["status"], "pending")
        rows = admin.export_submissions(self.db, "pending")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["judge_id"], judge)
        self.assertEqual(rows[0]["model"], "")
        self.assertEqual(len(rows[0]["text_hash"]), 64)

    def test_submit_validation(self):
        judge = new_judge()
        cases = [
            ({"consent": False}, 400, "consent_required"),
            ({"consent": None}, 400, "consent_required"),
            ({"consent": "true"}, 400, "consent_required"),
            ({"text": "短" * 49}, 400, "too_short"),
            ({"text": ""}, 400, "too_short"),
            ({"text": "长" * 3001}, 400, "too_long"),
            ({"model": "GPT-4"}, 400, "bad_model"),
            ({"model": 1}, 400, "bad_model"),
            ({"scene": "场" * 101}, 400, "bad_scene"),
            ({"nickname": "名" * 31}, 400, "bad_nickname"),
            ({"text": 123}, 400, "bad_text"),
        ]
        for override, status_code, code in cases:
            with self.subTest(override=override):
                status, data = self.post("/submit", self.submit_payload(judge, **override))
                self.assertEqual(status, status_code, data)
                self.assertEqual(data["error"], code)

    def test_submit_models_accepted(self):
        judge = new_judge()
        for index, model in enumerate(srv.MODELS):
            text = BASE_TEXT + str(index)
            status, data = self.post(
                "/submit", self.submit_payload(judge, text=text, model=model)
            )
            self.assertEqual(status, 200, (model, data))

    def test_submit_duplicate(self):
        judge = new_judge()
        first, data = self.post("/submit", self.submit_payload(judge))
        self.assertEqual(first, 200, data)
        # Same text with different surrounding whitespace still duplicates.
        status, data = self.post(
            "/submit", self.submit_payload(new_judge(), text="  " + BASE_TEXT + "\n")
        )
        self.assertEqual(status, 409)
        self.assertEqual(data["error"], "duplicate")

    def test_submit_daily_limit_per_judge(self):
        judge = new_judge()
        for index in range(20):
            status, data = self.post(
                "/submit", self.submit_payload(judge, text=BASE_TEXT + str(index))
            )
            self.assertEqual(status, 200, (index, data))
        status, data = self.post("/submit", self.submit_payload(judge, text=BASE_TEXT + "20"))
        self.assertEqual(status, 429)
        self.assertEqual(data["error"], "limit")

    def test_submit_daily_limit_per_ip(self):
        ip = "192.0.2.99"
        count = 0
        for judge_no in range(3):
            judge = new_judge()
            for index in range(20):
                status, data = self.post(
                    "/submit",
                    self.submit_payload(judge, text=BASE_TEXT + "%d-%d" % (judge_no, index)),
                    ip=ip,
                )
                self.assertEqual(status, 200, (judge_no, index, data))
                count += 1
        self.assertEqual(count, 60)
        status, data = self.post(
            "/submit", self.submit_payload(new_judge(), text=BASE_TEXT + "overflow"), ip=ip
        )
        self.assertEqual(status, 429)
        self.assertEqual(data["error"], "limit")


class SaltTests(unittest.TestCase):
    def test_salt_file_created_with_0600_and_reused(self):
        tmpdir = tempfile.mkdtemp(prefix="verna-salt-")
        old_salt = os.environ.pop("VERNA_IP_SALT", None)
        try:
            db_path = os.path.join(tmpdir, "verna.db")
            first = srv.load_salt(db_path)
            second = srv.load_salt(db_path)
            path = os.path.join(tmpdir, "ip_salt")
            self.assertTrue(os.path.exists(path))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(first, second)
            self.assertGreaterEqual(len(first), 32)
            self.assertEqual(len(srv.hash_ip(first, "1.2.3.4")), 16)
            self.assertNotEqual(
                srv.hash_ip(first, "1.2.3.4"), srv.hash_ip(first, "1.2.3.5")
            )
        finally:
            if old_salt is not None:
                os.environ["VERNA_IP_SALT"] = old_salt
            shutil.rmtree(tmpdir, ignore_errors=True)


class AdminTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="verna-admin-")
        self.db = os.path.join(self.tmpdir, "verna.db")
        self.jsonl = os.path.join(self.tmpdir, "items.jsonl")
        with open(self.jsonl, "w", encoding="utf-8") as handle:
            for record in ITEMS:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_import_skips_existing(self):
        self.assertEqual(admin.import_items(self.db, self.jsonl), (8, 0, 0))
        self.assertEqual(admin.import_items(self.db, self.jsonl), (0, 8, 0))

    def test_import_bad_lines(self):
        bad_path = os.path.join(self.tmpdir, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as handle:
            handle.write("not json\n")
            handle.write(json.dumps({"id": "x1"}, ensure_ascii=False) + "\n")
            handle.write(json.dumps(dict(ITEMS[0], id="x2", kind="weird"), ensure_ascii=False) + "\n")
        imported, skipped, bad = admin.import_items(self.db, bad_path)
        self.assertEqual((imported, skipped, bad), (0, 0, 3))

    def test_retire(self):
        admin.import_items(self.db, self.jsonl)
        self.assertEqual(admin.retire_items(self.db, ["it01", "nope"]), (1, ["nope"]))
        self.assertEqual(admin.retire_items(self.db, ["it01"]), (0, []))

    def test_exports_and_stats(self):
        admin.import_items(self.db, self.jsonl)

        # Build activity directly through the API on this database.
        tmpdir = tempfile.mkdtemp(prefix="verna-admin-srv-")
        old_salt = os.environ.get("VERNA_IP_SALT")
        os.environ["VERNA_IP_SALT"] = "admin-salt"
        httpd = srv.make_server(self.db, port=0)
        thread = serve_in_background(httpd)
        try:
            base = "http://127.0.0.1:%d/api" % httpd.server_address[1]
            judge = new_judge()

            def post(path, payload):
                req = urllib.request.Request(
                    base + path,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-Forwarded-For": "203.0.113.9"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.loads(response.read().decode("utf-8"))

            post("/vote", {"item": "it01", "judge": judge, "choice": "b", "left": "a", "ms": 5, "reasons": ["读着顺"], "note": "ok"})
            post("/submit", {"judge": judge, "text": BASE_TEXT, "model": "", "scene": "", "nickname": "", "consent": True})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(5)
            if old_salt is None:
                os.environ.pop("VERNA_IP_SALT", None)
            else:
                os.environ["VERNA_IP_SALT"] = old_salt
            shutil.rmtree(tmpdir, ignore_errors=True)

        votes = admin.export_votes(self.db)
        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["choice"], "b")
        self.assertEqual(votes[0]["reasons"], ["读着顺"])

        pending = admin.export_submissions(self.db, "pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(admin.export_submissions(self.db, "approved"), [])

        counts = admin.stats(self.db)
        self.assertEqual(counts["items"], 8)
        self.assertEqual(counts["active_items"], 8)
        self.assertEqual(counts["votes"], 1)
        self.assertEqual(counts["submissions"], 1)
        self.assertEqual(counts["pending_submissions"], 1)
        self.assertEqual(counts["judges"], 1)

    def test_cli_stats(self):
        admin.import_items(self.db, self.jsonl)
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "admin.py"), "--db", self.db, "stats"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["items"], 8)

    def test_cli_import_and_export(self):
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "admin.py"),
                "import-items",
                self.jsonl,
                "--db",
                self.db,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "admin.py"),
                "export-submissions",
                "--status",
                "pending",
                "--db",
                self.db,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
