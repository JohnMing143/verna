#!/usr/bin/env python3
"""Verna vote admin CLI.

Examples:
    python admin.py --db /var/lib/verna/verna.db import-items items.jsonl
    python admin.py --db /var/lib/verna/verna.db retire demo1 demo2
    python admin.py --db /var/lib/verna/verna.db export-votes > votes.jsonl
    python admin.py --db /var/lib/verna/verna.db export-submissions --status pending
    python admin.py --db /var/lib/verna/verna.db stats

--db may be given before or after the subcommand.
"""

import argparse
import json
import os
import sys

try:  # imported as part of a package (tests)
    from . import server as srv
except ImportError:  # executed as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import server as srv


REQUIRED_FIELDS = ("id", "genre", "source", "a", "b", "a_system", "b_system")


def import_items(db_path, jsonl_path):
    """Import items from JSONL. Existing ids are skipped.

    Returns (imported, skipped, bad).
    """
    srv.init_db(db_path)
    imported = 0
    skipped = 0
    bad = 0
    conn = srv.connect(db_path)
    try:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    sys.stderr.write(
                        "%s:%d: not valid JSON, skipped\n" % (jsonl_path, line_no)
                    )
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    sys.stderr.write(
                        "%s:%d: not a JSON object, skipped\n" % (jsonl_path, line_no)
                    )
                    bad += 1
                    continue
                item_id = obj.get("id")
                if not isinstance(item_id, str) or not item_id:
                    sys.stderr.write(
                        "%s:%d: missing id, skipped\n" % (jsonl_path, line_no)
                    )
                    bad += 1
                    continue
                missing = [
                    field
                    for field in REQUIRED_FIELDS
                    if not isinstance(obj.get(field), str)
                ]
                if missing:
                    sys.stderr.write(
                        "%s:%d: %s: missing text field(s): %s\n"
                        % (jsonl_path, line_no, item_id, ", ".join(missing))
                    )
                    bad += 1
                    continue
                kind = obj.get("kind") or "normal"
                if kind not in ("normal", "probe"):
                    sys.stderr.write(
                        "%s:%d: %s: bad kind, skipped\n"
                        % (jsonl_path, line_no, item_id)
                    )
                    bad += 1
                    continue
                probe_answer = obj.get("probe_answer")
                if probe_answer not in (None, "a", "b"):
                    sys.stderr.write(
                        "%s:%d: %s: bad probe_answer, skipped\n"
                        % (jsonl_path, line_no, item_id)
                    )
                    bad += 1
                    continue
                meta = obj.get("meta")
                if meta is not None and not isinstance(meta, str):
                    meta = json.dumps(meta, ensure_ascii=False)

                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                        (id, genre, source, a, b, a_system, b_system,
                         kind, probe_answer, status, meta, created)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        item_id,
                        obj["genre"],
                        obj["source"],
                        obj["a"],
                        obj["b"],
                        obj["a_system"],
                        obj["b_system"],
                        kind,
                        probe_answer,
                        meta,
                        srv.now_iso(),
                    ),
                )
                if cur.rowcount:
                    imported += 1
                else:
                    skipped += 1
        conn.commit()
    finally:
        conn.close()
    return imported, skipped, bad


def retire_items(db_path, item_ids):
    """Mark items retired. Returns (retired, missing)."""
    srv.init_db(db_path)
    retired = 0
    missing = []
    conn = srv.connect(db_path)
    try:
        for item_id in item_ids:
            cur = conn.execute(
                "UPDATE items SET status = 'retired' "
                "WHERE id = ? AND status != 'retired'",
                (item_id,),
            )
            if cur.rowcount:
                retired += 1
            else:
                exists = conn.execute(
                    "SELECT 1 FROM items WHERE id = ?", (item_id,)
                ).fetchone()
                if exists is None:
                    missing.append(item_id)
        conn.commit()
    finally:
        conn.close()
    return retired, missing


def export_votes(db_path):
    """Every vote as a dict, in creation order."""
    conn = srv.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT item_id, judge_id, choice, left_side, ms, reasons,
                   note, ip_hash, created, updated
            FROM votes
            ORDER BY created, item_id
            """
        ).fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        record = dict(row)
        try:
            record["reasons"] = json.loads(record["reasons"] or "[]")
        except ValueError:
            record["reasons"] = []
        out.append(record)
    return out


def export_submissions(db_path, status=None):
    """Submissions as dicts, in creation order, optionally filtered."""
    sql = (
        "SELECT id, judge_id, text, model, scene, nickname, ip_hash, "
        "text_hash, status, created FROM submissions"
    )
    params = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY id"
    conn = srv.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def stats(db_path):
    conn = srv.connect(db_path)
    try:
        result = {}
        for table in ("judges", "votes", "items", "submissions"):
            result[table] = conn.execute(
                "SELECT COUNT(*) FROM " + table
            ).fetchone()[0]
        result["active_items"] = conn.execute(
            "SELECT COUNT(*) FROM items WHERE status = 'active'"
        ).fetchone()[0]
        result["pending_submissions"] = conn.execute(
            "SELECT COUNT(*) FROM submissions WHERE status = 'pending'"
        ).fetchone()[0]
    finally:
        conn.close()
    return result


def _print_jsonl(records):
    for record in records:
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _add_db_argument(parser, suppress=False):
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--db", default=default, help="path to the sqlite database"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Verna vote admin")
    _add_db_argument(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    importer = sub.add_parser("import-items", help="import items from JSONL")
    importer.add_argument("path")
    _add_db_argument(importer, suppress=True)

    retire = sub.add_parser("retire", help="retire one or more item ids")
    retire.add_argument("ids", nargs="+")
    _add_db_argument(retire, suppress=True)

    exporter = sub.add_parser("export-votes", help="export votes as JSONL")
    _add_db_argument(exporter, suppress=True)

    submissions = sub.add_parser(
        "export-submissions", help="export submissions as JSONL"
    )
    submissions.add_argument("--status")
    _add_db_argument(submissions, suppress=True)

    stat_command = sub.add_parser("stats", help="print table counts as JSON")
    _add_db_argument(stat_command, suppress=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "db", None):
        parser.error("--db is required")

    if args.command == "import-items":
        imported, skipped, bad = import_items(args.db, args.path)
        sys.stderr.write(
            "imported %d, skipped existing %d, bad %d\n"
            % (imported, skipped, bad)
        )
        return 0 if bad == 0 else 1
    if args.command == "retire":
        retired, missing = retire_items(args.db, args.ids)
        sys.stderr.write("retired %d\n" % retired)
        if missing:
            sys.stderr.write("not found: %s\n" % ", ".join(missing))
            return 1
        return 0
    if args.command == "export-votes":
        _print_jsonl(export_votes(args.db))
        return 0
    if args.command == "export-submissions":
        _print_jsonl(export_submissions(args.db, args.status))
        return 0
    if args.command == "stats":
        sys.stdout.write(
            json.dumps(stats(args.db), ensure_ascii=False) + "\n"
        )
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
