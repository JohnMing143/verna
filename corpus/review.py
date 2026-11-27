#!/usr/bin/env python3
"""写评估结论到 corpus/reviews.json。
用法：.venv/bin/python corpus/review.py <id> <pending_review|clean|flagged|rejected> --summary "…" [--issue "…"]... [--cleaning "…"]... --checked "…"
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

C = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("id")
ap.add_argument("status", choices=["pending_review", "clean", "flagged", "rejected"])
ap.add_argument("--summary", required=True)
ap.add_argument("--issue", action="append", default=[])
ap.add_argument("--cleaning", action="append", default=[])
ap.add_argument("--checked", required=True)
a = ap.parse_args()
ids = {s["id"] for s in json.loads((C / "sources.json").read_text())["sources"]}
assert a.id in ids or a.id.endswith(".clean"), f"sources.json 里没有 {a.id}"
if a.status == "flagged":
    assert a.issue and a.cleaning, "flagged 必须写问题（--issue）和清洗要求（--cleaning）"
p = C / "reviews.json"
rv = json.loads(p.read_text()) if p.exists() else {}
rv[a.id] = {"status": a.status, "summary": a.summary, "issues": a.issue, "cleaning": a.cleaning, "checked": a.checked,
            "reviewed_at": datetime.now().isoformat(timespec="minutes"), "reviewer": "claude"}
p.write_text(json.dumps(rv, ensure_ascii=False, indent=1))
print(a.id, a.status)
