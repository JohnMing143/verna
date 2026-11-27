#!/usr/bin/env python3
"""给评估用的抽样统计：读 corpus/samples/<id>.jsonl，算长度、汉字/拉丁字母比例、HTML/维基标记/乱码/网址比例、重复率、日期范围、AI 腔指标，
再随机列几段摘录。只是辅助，结论要看原文后写进 reviews.json。

用法：.venv/bin/python corpus/profile.py zh.hc3 [--show 6] [--chars 300] [--field text] [--mid]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

C = Path(__file__).resolve().parent
sys.path.insert(0, str(C.parent))

HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")
HTML = re.compile(r"<(p|div|br|a|span|img|code|pre|table|li|ul|strong|em|blockquote|h\d)\b", re.I)
WIKI = re.compile(r"\{\{|\[\[|^==|'''", re.M)
URL = re.compile(r"https?://")
MOJI = re.compile(r"�|Ã.|â€|锟斤拷|烫烫|ï¿½")
EN_TELLS = re.compile(r"\b(delve|tapestry|testament to|in today's (fast-paced|digital)|it is important to note|as an ai language model|"
                      r"i hope this helps|navigat(e|ing) the (complex|landscape)|a rich tapestry|plays a (crucial|pivotal) role)\b", re.I)


def text_of(r: dict, field: str | None, mid: bool = False) -> str:
    if mid:
        mids = {k[5:]: v for k, v in r.items() if k.startswith("_mid_") and isinstance(v, str)}
        if field and field in mids:
            return mids[field]
        if not field and mids:
            return max(mids.values(), key=len)
    if field:
        v = r.get(field, "")
        return "\n".join(v) if isinstance(v, list) else str(v)
    best = ""
    for k, v in r.items():
        if k.startswith("_"):
            continue
        s = "\n".join(x for x in v if isinstance(x, str)) if isinstance(v, list) else (v if isinstance(v, str) else "")
        if len(s) > len(best):
            best = s
    return best


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--field", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mid", action="store_true", help="长文用中段（_mid_<字段>）算统计和摘录，避开版权页/声明")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (C / "samples" / f"{a.id}.jsonl").read_text().split("\n") if l.strip()]
    texts = [text_of(r, a.field, a.mid) for r in rows]
    n = len(texts)
    lens = [len(t) for t in texts]
    nonsp = [max(1, len(re.sub(r"\s", "", t))) for t in texts]
    han = [len(HAN.findall(t)) / m for t, m in zip(texts, nonsp)]
    lat = [len(LATIN.findall(t)) / m for t, m in zip(texts, nonsp)]
    frac = lambda rx: sum(1 for t in texts if rx.search(t)) / max(n, 1)
    dup = 1 - len(set(texts)) / max(n, 1)
    keys = sorted({k for r in rows for k in r})
    print(f"{a.id}：{n} 条；字段 {keys}")
    print(f"长度（字符，截断前最多 2500）p10/p50/p90 = {pct(lens, .1)}/{pct(lens, .5)}/{pct(lens, .9)}；空文本 {sum(1 for x in lens if x == 0)}")
    print(f"汉字占比中位数 {statistics.median(han):.2f}；拉丁字母占比中位数 {statistics.median(lat):.2f}")
    print(f"含 HTML {frac(HTML):.0%}，维基标记 {frac(WIKI):.0%}，网址 {frac(URL):.0%}，乱码 {frac(MOJI):.0%}，完全重复 {dup:.0%}")
    for k in keys:
        if re.search(r"date|time|created|year|timestamp", k, re.I):
            vals = sorted(str(r[k])[:10] for r in rows if r.get(k) not in (None, ""))
            if vals:
                print(f"{k}：{vals[0]} … {vals[len(vals) // 2]} … {vals[-1]}")
    zh = statistics.median(han) > 0.3
    if zh:
        from srh import tells
        dens = [tells.density(t) for t in texts if len(t) > 50]
        if dens:
            print(f"AI 腔密度（srh.tells，每千字加权命中）中位数 {statistics.median(dens):.2f}，p90 {pct(dens, .9):.2f}，>3 的占 {sum(d > 3 for d in dens) / len(dens):.0%}")
    else:
        print(f"英文 AI 腔词（delve/tapestry/as an AI language model 等）命中 {frac(EN_TELLS):.0%}")
    rnd = random.Random(a.seed)
    for i in rnd.sample(range(n), min(a.show, n)):
        meta = {k: v for k, v in rows[i].items() if not isinstance(v, (str, list)) or (isinstance(v, str) and len(v) < 60)}
        print(f"\n--- #{i} {json.dumps(meta, ensure_ascii=False)[:200]}")
        print(re.sub(r"\n+", " ⏎ ", texts[i][:a.chars]))


if __name__ == "__main__":
    main()
