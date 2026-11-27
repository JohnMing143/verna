#!/usr/bin/env python3
"""找老师模型的口头禅（Antislop 论文的做法）：比较老师写的文字和人写参照文字里各个短语的出现频率，
列出在老师那边多出几倍的短语。人工挑一遍，写进 data/teacher_tells.txt，distill.py 就会淘汰新出现这些说法的候选。

  python scripts/slop_profile.py --teacher data/distilled.jsonl --human data/human_raw.jsonl --top 80

老师文字取 distill 产出里 status=ok 的 target；人写参照取 human_raw 的 text。两边各至少几百段才有意义。

11-05 改：老师记录带 input 时，只数"老师新加的"短语（候选或答案里有、输入里没有）。
distill.py 的禁用表本来就只拦新出现的说法；不这么数的话，榜单会被输入里继承来的题材词占满
（300 条那次排前面的是 **、本研究、咖啡、电商，老师新加的次数全是 0）。
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh.runner import read_jsonl  # noqa: E402

PUNCT = re.compile(r"[\s，。！？；：、“”‘’（）《》【】—…,.!?;:()\"'\-]+")


def doc_freq(texts: list[str], ns=(2, 3, 4, 5)) -> collections.Counter:
    df = collections.Counter()
    for t in texts:
        grams = set()
        for piece in PUNCT.split(t):
            for n in ns:
                grams.update(piece[i:i + n] for i in range(len(piece) - n + 1))
        df.update(grams)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--min-docs", type=int, default=5, help="老师那边至少在这么多条不同的输入上新加了这个短语")
    a = ap.parse_args()

    recs = read_jsonl(a.teacher)
    ht = [r["text"] for r in read_jsonl(a.human) if r.get("text")]
    if any(r.get("input") for r in recs):
        # 每个候选（没有候选就用答案）对照它自己的输入，只记输入里没有的短语。
        # 11-06 改：同一条输入的几个候选算一条记录。原来按候选计数，一条输入出 3 个候选，
        # 题材词、专名（成都、洛川、"严格按以下步骤"）就数成 3 段、排进榜单前排；1500 那次前 120 名几乎全是这种。
        tt, n_rec = [], 0
        dt, drec, dtgt = collections.Counter(), collections.Counter(), collections.Counter()
        dgenre = collections.defaultdict(set)
        for r in recs:
            texts = [c for c in (r.get("candidates") or []) if isinstance(c, str) and c.strip()]
            if not texts and r.get("status") == "ok" and r.get("target"):
                texts = [r["target"]]
            if not texts:
                continue
            src = r.get("input") or ""
            n_rec += 1
            tt += texts
            union = set()
            for t in texts:
                new = {g for g in doc_freq([t]) if g not in src}
                dt.update(new)
                union |= new
            drec.update(union)
            for g in union:
                dgenre[g].add(r.get("genre"))
            if r.get("status") == "ok" and r.get("target"):
                dtgt.update(g for g in doc_freq([r["target"]]) if g not in src)
        mode = f"老师新加（候选里有、输入里没有），{n_rec} 条输入"
    else:
        tt = [r["target"] for r in recs if r.get("status") == "ok" and r.get("target")]
        dt = doc_freq(tt)
        drec, dtgt, dgenre = dt, dt, collections.defaultdict(set)
        mode = "老师答案"
    if len(tt) < 50 or len(ht) < 50:
        print(f"样本太少（老师 {len(tt)} 段，人写 {len(ht)} 段），结果参考价值有限", file=sys.stderr)
    dh = doc_freq(ht)
    rows = []
    for g, c in dt.items():
        # 门槛按不同输入数算；* # 是格式标记，srh/tells.py 已经管了
        if drec[g] < a.min_docs or re.fullmatch(r"[\d\w]{0,1}", g) or re.search(r"[\d*#]", g):
            continue
        pt = c / len(tt)
        ph = (dh.get(g, 0) + 0.5) / (len(ht) + 1)
        rows.append((pt / ph, g, c, dh.get(g, 0)))
    rows.sort(reverse=True)
    # 去掉被更长短语覆盖的子串，以及几乎出现在同一批输入里的加长版，避免一条口头禅占好几行
    picked = []
    for ratio, g, c, h in rows:
        if any(g in p or (p in g and drec[g] >= 0.8 * drec[p]) for _, p, _, _ in picked):
            continue
        picked.append((ratio, g, c, h))
        if len(picked) >= a.top:
            break
    print(f"老师 {len(tt)} 段（{mode}）/ 人写 {len(ht)} 段；倍数 = 老师段落占比 ÷ 人写段落占比")
    print("  列：倍数 短语 老师候选段数 不同输入数 进了最终答案的输入数 涉及文体数 人写段数")
    for ratio, g, c, h in picked:
        print(f"{ratio:7.1f}x  {g:10s}  老师 {c:4d} 段  输入 {drec[g]:4d}  答案 {dtgt[g]:4d}  文体 {len(dgenre[g]):2d}  人写 {h:4d} 段")

if __name__ == "__main__":
    main()
