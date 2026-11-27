#!/usr/bin/env python3
"""Verna vote API.

Standard library only. Listens on 127.0.0.1 and speaks JSON under /api/.
nginx is expected to keep the /api/ prefix; paths are also accepted without
it so the server can be exercised directly.

Usage:
    python server.py --db /path/verna.db --port 8790
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

MAX_BODY = 16 * 1024
MAX_REASONS = 12
MAX_REASON_LEN = 40
MAX_NOTE_LEN = 300
MAX_TEXT_LEN = 3000
MIN_HAN = 50
MAX_MS = 86400000
SUBMIT_PER_JUDGE_DAY = 20
SUBMIT_PER_IP_DAY = 60

JUDGE_RE = re.compile(r"j[a-z0-9]{8,40}")
HAN_RE = re.compile("[\\u4e00-\\u9fff]")
WS_RE = re.compile(r"\s+")

CHOICES = ("a", "b", "src")
SIDES = ("a", "b")
MODELS = (
    "不清楚",
    "ChatGPT",
    "Claude",
    "Gemini",
    "DeepSeek",
    "豆包",
    "Kimi",
    "通义千问",
    "文心一言",
    "其他",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    genre TEXT,
    source TEXT,
    a TEXT,
    b TEXT,
    a_system TEXT,
    b_system TEXT,
    kind TEXT DEFAULT 'normal',
    probe_answer TEXT,
    status TEXT DEFAULT 'active',
    meta TEXT,
    created TEXT
);

CREATE TABLE IF NOT EXISTS judges (
    id TEXT PRIMARY KEY,
    created TEXT,
    from_tag TEXT,
    ip_hash TEXT
);

CREATE TABLE IF NOT EXISTS votes (
    item_id TEXT NOT NULL,
    judge_id TEXT NOT NULL,
    choice TEXT NOT NULL,
    left_side TEXT NOT NULL,
    ms INTEGER NOT NULL,
    reasons TEXT,
    note TEXT,
    ip_hash TEXT,
    created TEXT,
    updated TEXT,
    UNIQUE (item_id, judge_id)
);

CREATE TABLE IF NOT EXISTS skips (
    item_id TEXT NOT NULL,
    judge_id TEXT NOT NULL,
    created TEXT,
    UNIQUE (item_id, judge_id)
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    judge_id TEXT,
    text TEXT,
    model TEXT,
    scene TEXT,
    nickname TEXT,
    ip_hash TEXT,
    text_hash TEXT UNIQUE,
    status TEXT DEFAULT 'pending',
    created TEXT
);

CREATE INDEX IF NOT EXISTS idx_votes_item ON votes (item_id);
CREATE INDEX IF NOT EXISTS idx_votes_judge ON votes (judge_id);
CREATE INDEX IF NOT EXISTS idx_skips_judge ON skips (judge_id);
CREATE INDEX IF NOT EXISTS idx_items_status ON items (status);
CREATE INDEX IF NOT EXISTS idx_submissions_judge ON submissions (judge_id);
CREATE INDEX IF NOT EXISTS idx_submissions_ip ON submissions (ip_hash);
CREATE INDEX IF NOT EXISTS idx_submissions_created ON submissions (created);
"""


class ApiError(Exception):
    """An error that maps directly to a JSON error response."""

    def __init__(self, status, code, headers=None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.headers = headers or {}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db(db_path):
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def load_salt(db_path):
    """Salt from VERNA_IP_SALT, else an ip_salt file beside the database."""
    env_salt = os.environ.get("VERNA_IP_SALT", "")
    if env_salt:
        return env_salt
    path = os.path.join(os.path.dirname(os.path.abspath(db_path)), "ip_salt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            salt = f.read().strip()
        if salt:
            return salt
    except OSError:
        pass
    salt = secrets.token_hex(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(salt + "\n")
    return salt


def hash_ip(salt, ip):
    return hashlib.sha256((salt + (ip or "")).encode("utf-8")).hexdigest()[:16]


def han_count(text):
    return len(HAN_RE.findall(text or ""))


def text_hash(text):
    return hashlib.sha256(WS_RE.sub("", text).encode("utf-8")).hexdigest()


def ensure_judge(conn, judge_id, from_tag=None, ip_hash=None):
    """Insert the judge on first sight. Existing judges are never modified."""
    if from_tag:
        from_tag = from_tag[:40]
    else:
        from_tag = None
    cur = conn.execute(
        "INSERT OR IGNORE INTO judges (id, created, from_tag, ip_hash) "
        "VALUES (?, ?, ?, ?)",
        (judge_id, now_iso(), from_tag, ip_hash),
    )
    return cur.rowcount > 0


def require_judge(data):
    judge = data.get("judge")
    if not isinstance(judge, str) or not JUDGE_RE.fullmatch(judge):
        raise ApiError(400, "bad_judge")
    return judge


def _require_str(data, key, code, max_len=None, allow_empty=False):
    value = data.get(key)
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ApiError(400, code)
    value = value.strip()
    if not allow_empty and not value:
        raise ApiError(400, code)
    if max_len is not None and len(value) > max_len:
        raise ApiError(400, code)
    return value


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "VernaVote/1.0"
    protocol_version = "HTTP/1.1"

    db_path = None
    salt = ""
    routes = {
        "/next": ("GET", "handle_next"),
        "/stats": ("GET", "handle_stats"),
        "/vote": ("POST", "handle_vote"),
        "/unvote": ("POST", "handle_unvote"),
        "/skip": ("POST", "handle_skip"),
        "/submit": ("POST", "handle_submit"),
    }

    # -- plumbing ---------------------------------------------------------

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_OPTIONS(self):
        self._dispatch("OPTIONS")

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            sys.stderr.write(
                "%s %s\n" % (self.log_date_time_string(), fmt % args)
            )

    def client_ip(self):
        ip = self.headers.get("CF-Connecting-IP")
        if ip and ip.strip():
            return ip.strip()
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        return self.client_address[0]

    def _arg(self, name):
        values = self.query.get(name)
        if not values:
            return None
        return values[0]

    def _json_body(self):
        ctype = (self.headers.get("Content-Type") or "")
        ctype = ctype.split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(415, "bad_content_type")
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise ApiError(400, "bad_length")
        try:
            length = int(raw_len)
        except ValueError:
            raise ApiError(400, "bad_length")
        if length < 0:
            raise ApiError(400, "bad_length")
        if length > MAX_BODY:
            self.close_connection = True
            # Drain modest over-limit bodies so the client can still read
            # the 413 instead of getting a TCP reset.
            if length <= MAX_BODY * 4:
                try:
                    self.rfile.read(length)
                except OSError:
                    pass
            raise ApiError(413, "too_large")
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            raise ApiError(400, "bad_length")
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "bad_json")
        if not isinstance(data, dict):
            raise ApiError(400, "bad_json")
        return data

    def _send(self, status, payload, extra_headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _dispatch(self, method):
        split = urlsplit(self.path)
        path = split.path
        if path == "/api" or path == "/api/":
            path = "/"
        elif path.startswith("/api/"):
            path = path[len("/api"):]
        self.query = parse_qs(split.query, keep_blank_values=True)
        try:
            route = self.routes.get(path)
            if route is None:
                raise ApiError(404, "not_found")
            if route[0] != method:
                raise ApiError(
                    405, "method_not_allowed", {"Allow": route[0]}
                )
            payload = getattr(self, route[1])() or {}
            result = {"ok": True}
            result.update(payload)
            self._send(200, result)
        except ApiError as exc:
            self._send(
                exc.status,
                {"ok": False, "error": exc.code},
                exc.headers,
            )
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            traceback.print_exc()
            self._send(500, {"ok": False, "error": "internal"})

    # -- query helpers ----------------------------------------------------

    def _ip_hash(self):
        return hash_ip(self.salt, self.client_ip())

    # -- handlers ---------------------------------------------------------

    def handle_next(self):
        judge = self._arg("judge")
        if not isinstance(judge, str) or not JUDGE_RE.fullmatch(judge):
            raise ApiError(400, "bad_judge")
        raw_n = self._arg("n")
        try:
            n = int(raw_n) if raw_n is not None else 5
        except ValueError:
            n = 5
        n = max(1, min(10, n))
        from_tag = self._arg("from")
        ip_hash = self._ip_hash()

        conn = connect(self.db_path)
        try:
            known = conn.execute(
                "SELECT 1 FROM judges WHERE id = ?", (judge,)
            ).fetchone()
            if known is None:
                conn.execute("BEGIN IMMEDIATE")
                ensure_judge(conn, judge, from_tag, ip_hash)
                conn.commit()
            rows = conn.execute(
                """
                SELECT i.id, i.genre, i.source, i.a, i.b,
                       (SELECT COUNT(*) FROM votes v WHERE v.item_id = i.id)
                           AS vote_count
                FROM items i
                WHERE i.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM votes v
                      WHERE v.item_id = i.id AND v.judge_id = ?
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM skips s
                      WHERE s.item_id = i.id AND s.judge_id = ?
                  )
                ORDER BY vote_count ASC, RANDOM()
                """,
                (judge, judge),
            ).fetchall()
        finally:
            conn.close()

        capped = []
        overflow = []
        per_genre = {}
        for row in rows:
            genre = row["genre"] or ""
            used = per_genre.get(genre, 0)
            if used < 2:
                per_genre[genre] = used + 1
                capped.append(row)
            else:
                overflow.append(row)

        picked = capped[:n]
        if len(picked) < n:
            picked.extend(overflow[: n - len(picked)])

        items = [
            {
                "id": row["id"],
                "genre": row["genre"] or "",
                "source": row["source"] or "",
                "a": row["a"] or "",
                "b": row["b"] or "",
            }
            for row in picked
        ]
        return {"items": items}

    def handle_vote(self):
        data = self._json_body()
        judge = require_judge(data)
        item = _require_str(data, "item", "bad_item", max_len=200)
        choice = data.get("choice")
        if not isinstance(choice, str) or choice not in CHOICES:
            raise ApiError(400, "bad_choice")
        left = data.get("left")
        if not isinstance(left, str) or left not in SIDES:
            raise ApiError(400, "bad_left")
        ms = data.get("ms")
        if type(ms) is not int or ms < 0 or ms > MAX_MS:
            raise ApiError(400, "bad_ms")
        reasons = data.get("reasons", [])
        if reasons is None:
            reasons = []
        if not isinstance(reasons, list) or len(reasons) > MAX_REASONS:
            raise ApiError(400, "bad_reasons")
        for reason in reasons:
            if not isinstance(reason, str) or len(reason) > MAX_REASON_LEN:
                raise ApiError(400, "bad_reasons")
        note = data.get("note", "")
        if note is None:
            note = ""
        if not isinstance(note, str) or len(note) > MAX_NOTE_LEN:
            raise ApiError(400, "bad_note")

        ip_hash = self._ip_hash()
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM items WHERE id = ?", (item,)
            ).fetchone()
            if row is None:
                raise ApiError(404, "item_not_found")
            if row["status"] != "active":
                raise ApiError(400, "item_inactive")
            ensure_judge(conn, judge, None, ip_hash)
            stamp = now_iso()
            conn.execute(
                """
                INSERT INTO votes
                    (item_id, judge_id, choice, left_side, ms, reasons,
                     note, ip_hash, created, updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (item_id, judge_id) DO UPDATE SET
                    choice = excluded.choice,
                    left_side = excluded.left_side,
                    ms = excluded.ms,
                    reasons = excluded.reasons,
                    note = excluded.note,
                    ip_hash = excluded.ip_hash,
                    updated = excluded.updated
                """,
                (
                    item,
                    judge,
                    choice,
                    left,
                    ms,
                    json.dumps(reasons, ensure_ascii=False),
                    note,
                    ip_hash,
                    stamp,
                    stamp,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {}

    def handle_unvote(self):
        data = self._json_body()
        judge = require_judge(data)
        item = _require_str(data, "item", "bad_item", max_len=200)
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            ensure_judge(conn, judge, None, self._ip_hash())
            conn.execute(
                "DELETE FROM votes WHERE item_id = ? AND judge_id = ?",
                (item, judge),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {}

    def handle_skip(self):
        data = self._json_body()
        judge = require_judge(data)
        item = _require_str(data, "item", "bad_item", max_len=200)
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            ensure_judge(conn, judge, None, self._ip_hash())
            conn.execute(
                "INSERT OR IGNORE INTO skips (item_id, judge_id, created) "
                "VALUES (?, ?, ?)",
                (item, judge, now_iso()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {}

    def handle_submit(self):
        data = self._json_body()
        judge = require_judge(data)
        if data.get("consent") is not True:
            raise ApiError(400, "consent_required")
        text = data.get("text")
        if not isinstance(text, str):
            raise ApiError(400, "bad_text")
        text = text.strip()
        if len(text) > MAX_TEXT_LEN:
            raise ApiError(400, "too_long")
        if han_count(text) < MIN_HAN:
            raise ApiError(400, "too_short")
        model = data.get("model", "")
        if model is None:
            model = ""
        if not isinstance(model, str) or (model and model not in MODELS):
            raise ApiError(400, "bad_model")
        scene = data.get("scene", "")
        if scene is None:
            scene = ""
        if not isinstance(scene, str):
            raise ApiError(400, "bad_scene")
        scene = scene.strip()
        if len(scene) > 100:
            raise ApiError(400, "bad_scene")
        nickname = data.get("nickname", "")
        if nickname is None:
            nickname = ""
        if not isinstance(nickname, str):
            raise ApiError(400, "bad_nickname")
        nickname = nickname.strip()
        if len(nickname) > 30:
            raise ApiError(400, "bad_nickname")

        digest = text_hash(text)
        ip_hash = self._ip_hash()
        day = today_utc()
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            ensure_judge(conn, judge, None, ip_hash)
            dup = conn.execute(
                "SELECT id FROM submissions WHERE text_hash = ?", (digest,)
            ).fetchone()
            if dup is not None:
                raise ApiError(409, "duplicate")
            judge_count = conn.execute(
                "SELECT COUNT(*) FROM submissions "
                "WHERE judge_id = ? AND substr(created, 1, 10) = ?",
                (judge, day),
            ).fetchone()[0]
            if judge_count >= SUBMIT_PER_JUDGE_DAY:
                raise ApiError(429, "limit")
            ip_count = conn.execute(
                "SELECT COUNT(*) FROM submissions "
                "WHERE ip_hash = ? AND substr(created, 1, 10) = ?",
                (ip_hash, day),
            ).fetchone()[0]
            if ip_count >= SUBMIT_PER_IP_DAY:
                raise ApiError(429, "limit")
            cur = conn.execute(
                """
                INSERT INTO submissions
                    (judge_id, text, model, scene, nickname, ip_hash,
                     text_hash, status, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    judge,
                    text,
                    model,
                    scene,
                    nickname,
                    ip_hash,
                    digest,
                    now_iso(),
                ),
            )
            new_id = cur.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"id": new_id, "status": "pending"}

    def handle_stats(self):
        conn = connect(self.db_path)
        try:
            result = {}
            for table in ("judges", "votes", "items", "submissions"):
                result[table] = conn.execute(
                    "SELECT COUNT(*) FROM " + table
                ).fetchone()[0]
        finally:
            conn.close()
        return result


def make_server(db_path, port=8790, host="127.0.0.1", verbose=False):
    init_db(db_path)
    salt = load_salt(db_path)
    handler = type(
        "BoundApiHandler",
        (ApiHandler,),
        {"db_path": db_path, "salt": salt},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.verbose = verbose
    return httpd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verna vote API server")
    parser.add_argument("--db", required=True, help="path to the sqlite database")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    httpd = make_server(args.db, port=args.port, verbose=args.verbose)
    host, port = httpd.server_address[0], httpd.server_address[1]
    sys.stderr.write("verna vote API listening on http://%s:%d (db=%s)\n" % (host, port, args.db))
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
