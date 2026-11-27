#!/usr/bin/env python3
"""生成进度页 https://sga.143143.eu.org/reports/shuorenhua-progress/ 。只放统计，不放语料原文（原文有版权，也没必要公开）。
抽样给用户看的原文放在本地 data/v3/autopilot/samples.md。"""
from __future__ import annotations

import html
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "autopilot"))
import run as ap  # noqa: E402
from srh import genres  # noqa: E402

REP = Path("/root/reports/shuorenhua-progress")


def esc(x) -> str:
    return html.escape(str(x))


def main():
    s = ap.status()
    st = s.get("stats") or ap.stats()
    q = s.get("quota", {})
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "data").mkdir(exist_ok=True)
    (REP / "data" / "stats.json").write_text(json.dumps({"status": {k: v for k, v in s.items() if k != "stats"}, "stats": st}, ensure_ascii=False, indent=1))
    (REP / "meta.json").write_text(json.dumps({
        "title": "Verna：数据管线进度", "date": s.get("updated", "")[:10], "data_as_of": s.get("updated", "")[:16].replace("T", " "),
        "description": "无人值守数据管线的实时进度：收集、分级、范本、配对、抽检和额度。每两小时更新。",
        "datasets": {"stats": {"title": "进度统计", "file": "stats.json"}}}, ensure_ascii=False, indent=1))

    kinds = st.get("pairs_by_kind", {})
    total = st.get("pairs_total", 0)
    held = st.get("pairs_held") or {}  # 11-07：补查完之前暂扣的（run.all_pairs hold）
    audits = s.get("audits", [])
    last_a = audits[-1] if audits else None
    rows_src = []
    for k, v in sorted(st.get("by_source", {}).items()):
        g = v.get("graded", 0)
        rows_src.append(f"<tr><td>{esc(k)}</td><td class='r'>{st.get('raw_docs', {}).get(k, 0)}</td><td class='r'>{v.get('segments', 0)}</td>"
                        f"<td class='r'>{g}</td><td class='r'>{v.get('tier_exemplar', 0)}</td><td class='r'>{v.get('exemplar', 0)}</td>"
                        f"<td class='r'>{v.get('tier_improvable', 0)}</td><td class='r'>{v.get('tier_unusable', 0)}</td></tr>")
    rows_genre = "".join(f"<tr><td>{esc(genres.GENRES[g]['name'])}</td><td class='r'>{st.get('pairs_by_genre', {}).get(g, 0)}</td>"
                         f"<td class='r'>{round(genres.GENRES[g]['quota'] * 100)}%</td></tr>" for g in genres.GENRES)
    wakes = "".join(f"<li>{esc(w['t'])} {esc(w['kind'])}：{esc(w.get('summary', '')[:300])}</li>" for w in reversed(s.get("wakes", [])[-10:])) or "<li>还没有</li>"
    for_user = ap.F["for_user"].read_text() if ap.F["for_user"].exists() else ""
    body = f"""
<div class="qa">
  <header class="qa-head"><h1>Verna：数据管线进度</h1>
  <p class="qa-lede">状态：{esc(s.get('state', ''))}。更新于 {esc(s.get('updated', ''))}。首次启动 {esc(s.get('first_started', ''))}。方案见<a href="/reports/shuorenhua-v2/">第二版报告</a>。</p></header>
  <section class="qa-kpis">
    <div class="qa-kpi"><span class="qa-kpi-l">合格配对</span><span class="qa-kpi-v">{total:,}</span><span class="qa-kpi-s">{esc(json.dumps(kinds, ensure_ascii=False))}{esc(f"；另有 {sum(held.values())} 条暂扣待补查" if held else "")}</span></div>
    <div class="qa-kpi"><span class="qa-kpi-l">已分级段落</span><span class="qa-kpi-v">{st.get('graded', 0):,}</span><span class="qa-kpi-s">切段 {st.get('segments', 0):,} 段</span></div>
    <div class="qa-kpi"><span class="qa-kpi-l">抽检不合格</span><span class="qa-kpi-v">{st.get('audit_bad', 0)} / {st.get('audit_n', 0)}</span><span class="qa-kpi-s">最近一次：{esc(f"{last_a['bad']}/{last_a['n']}（{last_a['t']}）" if last_a else '还没有')}</span></div>
    <div class="qa-kpi"><span class="qa-kpi-l">agy Gemini 额度</span><span class="qa-kpi-v">{q.get('weekly', '?')}%</span><span class="qa-kpi-s">周额度剩余；5 小时窗口剩 {q.get('five', '?')}%</span></div>
  </section>
  {"<section class='qa-answer'><h2 style='margin:0'>需要你看的</h2><pre style='white-space:pre-wrap;margin:0'>" + esc(for_user) + "</pre></section>" if for_user.strip() else ""}
  <section class="qa-sec"><h2>各来源</h2><div class="qa-scroll"><table class="qa-table">
    <thead><tr><th>来源</th><th class="r">文档</th><th class="r">段落</th><th class="r">已分级</th><th class="r">初评范本</th><th class="r">复审通过</th><th class="r">值得改</th><th class="r">不能用</th></tr></thead>
    <tbody>{''.join(rows_src)}</tbody></table></div></section>
  <section class="qa-sec"><h2>配对按文体</h2><div class="qa-scroll"><table class="qa-table">
    <thead><tr><th>文体</th><th class="r">配对</th><th class="r">配额</th></tr></thead><tbody>{rows_genre}</tbody></table></div></section>
  <section class="qa-sec"><h2>最近的唤醒</h2><ul class="qa-note">{wakes}</ul></section>
  <p class="qa-note">原文抽样不放在公开页面上，在服务器的 /root/shuorenhua/data/v3/autopilot/samples.md。</p>
</div>"""
    style = (Path("/root/reports/shuorenhua-v2/page.html").read_text().split("</style>")[0] + "</style>")
    (REP / "page.html").write_text(style + body)
    subprocess.run([sys.executable, "/root/reports/publish.py", "shuorenhua-progress"], capture_output=True, text=True)

    # 本地抽样：给用户醒来时翻看
    ex = ap.exemplars(ap.cfg())
    pairs = ap.all_pairs()
    rng = random.Random()
    lines = [f"# 抽样（{s.get('updated', '')}）\n", "## 范本（复审通过的人写文字）\n"]
    for e in rng.sample(ex, min(8, len(ex))):
        lines.append(f"### {e['source']} · {genres.GENRES.get(e['genre'], {}).get('name', e['genre'])} · {e.get('title') or ''}\n\n{e['text']}\n")
    lines.append("## 配对（输入 → 答案）\n")
    for p in rng.sample(pairs, min(8, len(pairs))):
        lines.append(f"### {p['kind']} · {genres.GENRES.get(p['genre'], {}).get('name', p['genre'])} · {p['id']}\n\n**输入**\n\n{p['src']}\n\n**答案**\n\n{p['tgt']}\n")
    (ap.AP / "samples.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
