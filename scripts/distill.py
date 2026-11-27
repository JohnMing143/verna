#!/usr/bin/env python3
"""v2 第 3 步：老师模型逐条打磨，产出（输入，最好的改写）配对。

每条输入走四步：
  1. 分类：没有文体标签的先判文体和质量（excellent / improvable / unusable），unusable 直接丢
  2. 改写：老师模型先诊断问题、定改动幅度，再给 3 个候选
  3. 规则：数字/英文/书名/引语核对、长度比、禁用说法，不过的候选直接淘汰
  4. 裁判：剩下的候选打乱顺序交给裁判，逐个查事实、文体、新套话，再排序

输入 jsonl：{"id", "text", "kind": "ai_draft" | "human", "genre"(可选)}
  ai_draft  AI 写的初稿（gen_drafts.py 产出），一定要改
  human     人写的文字（任何年代），写得好的会变成"不用改"样本，教模型克制

  python scripts/distill.py --in data/inputs.jsonl --out data/distilled.jsonl \\
      --teacher agy:gemini-4-argon-medium --judge agy:gemini-4-argon-high
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srh import facts, genres, llm, prompts, tells  # noqa: E402
from srh.runner import read_jsonl, run  # noqa: E402

LEN_BOUNDS = {"none": (0.85, 1.15), "light": (0.6, 1.25), "heavy": (0.45, 1.45)}
# 11-10 加（主会话设计）：AI 初稿专用的规则。答案要比输入明显不同（相似度 < 0.92），AI 腔词表命中一个都不许留，
# 长度比上下限按 EDIT_AI_PROMPT 的 light/heavy 分（去掉空话本来就该变短）。
AI_LEN_BOUNDS = {"light": (0.55, 1.15), "heavy": (0.4, 1.15)}
AI_SIM_MAX = 0.92


def load_ban(path: str | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def local_check(src: str, cand: str, level: str, ban: list[str]) -> list[str]:
    issues = []
    c = facts.compare(src, cand)
    issues += c["hard"]
    lo, hi = LEN_BOUNDS.get(level, LEN_BOUNDS["heavy"])
    ratio = len(cand) / max(len(src), 1)
    if not lo <= ratio <= hi:
        issues.append(f"长度比 {ratio:.2f}")
    # 新出现的禁用说法（原文就有的不算）
    new_ban = set(tells.banned(cand)) - set(tells.banned(src))
    issues += [f"新套话 {b}" for b in sorted(new_ban)]
    issues += [f"老师腔 {w}" for w in ban if w in cand and w not in src]
    return issues


def local_check_ai(src: str, cand: str, level: str, ban: list[str]) -> list[str]:
    """AI 初稿版 local_check：核对前先去掉原文里的助手附言；禁用说法在候选里出现一个就算问题（原文就有的也要去干净）；
    和原文太像（相似度 ≥ 0.92）算“改动太少”，AI 初稿不能原样当答案。"""
    s = prompts.strip_assistant_meta(src)
    issues = facts.compare(s, cand)["hard"]
    lo, hi = AI_LEN_BOUNDS.get(level, AI_LEN_BOUNDS["heavy"])
    ratio = len(cand) / max(len(s), 1)
    if not lo <= ratio <= hi:
        issues.append(f"长度比 {ratio:.2f}")
    issues += [f"还留着禁用说法 {b}" for b in sorted(set(tells.banned(cand)))]
    if difflib.SequenceMatcher(None, s, cand, autojunk=False).ratio() >= AI_SIM_MAX:
        issues.append("改动太少")
    issues += [f"老师腔 {w}" for w in ban if w in cand and w not in src]
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="每条的完整记录（含被拒的），可断点续跑")
    ap.add_argument("--teacher", default="agy:gemini-4-argon-medium")
    ap.add_argument("--teacher-timeout", type=int, default=300,
                    help="老师单次调用的秒数上限。11-12 起 AI 线老师用 Carbon medium，改一段平均 141 秒、最长 249 秒（单版本），三个候选要放宽")
    ap.add_argument("--judge", default="agy:gemini-4-argon-medium",
                    help="试点里 medium 和 high 对 35 个候选的事实判断完全一致，medium 便宜约 8 倍")
    ap.add_argument("--no-repair", action="store_true", help="候选全被拒时不做返修")
    ap.add_argument("--final-gate", default="", help="终审模型（如 agy:gemini-4-argon-high），和抽检用同一套标准；不填就不终审")
    ap.add_argument("--ban-file", default=str(Path(__file__).resolve().parents[1] / "data" / "teacher_tells.txt"),
                    help="老师模型自己的口头禅（slop_profile.py 统计后人工挑出来的），候选里新出现就淘汰")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ai-mode", action="store_true",
                    help="AI 初稿专用（只对 kind=ai_draft 生效）：AI 专用的改写/裁判/终审提示词和规则，永不出 keep")
    a = ap.parse_args()
    ban = load_ban(a.ban_file)

    def final_gate(rec, src, target, genre, level, ai_mode=False):
        """终审：用抽检的同一把尺子（默认 Argon high）审答案；不过就带着意见返修一次再审，还不过就扔。
        AI 初稿（ai_mode）用 AUDIT_AI_PROMPT 和 AI 版规则。"""
        if not a.final_gate:
            return {**rec, "status": "ok", "target": target}
        check = local_check_ai if ai_mode else local_check
        for attempt in range(2):
            if ai_mode:
                j = llm.ask_json(a.final_gate, prompts.AUDIT_AI_PROMPT.format(
                    principles=prompts.principles(), genre_name=genres.GENRES[genre]["name"], src=src, tgt=target), tag="gate_ai")
            else:
                j = llm.ask_json(a.final_gate, prompts.AUDIT_PROMPT.format(
                    principles=prompts.principles(), kind="distill", genre_name=genres.GENRES[genre]["name"], src=src, tgt=target), tag="gate")
            rec.setdefault("gate", []).append(j)
            if all(j.get(k, True) for k in ("fact_ok", "quality_ok", "register_ok", "edit_ok")):
                return {**rec, "status": "ok", "target": target, "gate_rounds": attempt + 1}
            if attempt == 1:
                break
            issues = "\n".join(f"- {x}" for x in (j.get("problems") or [])) or "- 改得不到位"
            fixed = llm.chat(a.teacher, [{"role": "user", "content": prompts.REPAIR_PROMPT.format(
                genre_desc=genres.describe(genre), original=src, candidate=target, issues=issues)}], tag="repair",
                timeout=a.teacher_timeout).strip()
            if check(src, fixed, level, ban):
                break
            target = fixed
        return {**rec, "status": "reject", "reason": "终审未过"}

    def one(r):
        rec = {"id": r["id"], "kind": r.get("kind", "human"), "input": r["text"], "source": r.get("source"),
               "teacher": llm.label(a.teacher)}  # 11-12 起 AI 线和人写线的老师可能不同，逐条记下
        ai_mode = a.ai_mode and r.get("kind") == "ai_draft"
        if ai_mode:
            rec["ai_mode"] = 2
        genre = r.get("genre")
        quality = r.get("quality") or ("improvable" if r.get("kind") == "ai_draft" else None)
        if not genre or quality is None:
            c = llm.ask_json(a.judge, prompts.CLASSIFY_PROMPT.format(text=r["text"], genre_names=genres.names()), tag="classify")
            genre = genre or c.get("genre")
            quality = quality or c.get("quality")
            rec["classify"] = c
        if genre not in genres.GENRES:
            return {**rec, "status": "reject", "reason": f"文体不认识：{genre}"}
        rec["genre"] = genre
        rec["quality"] = quality
        if quality == "unusable":
            return {**rec, "status": "reject", "reason": "unusable"}

        g = genres.GENRES[genre]
        edit_prompt = prompts.EDIT_AI_PROMPT if ai_mode else prompts.EDIT_PROMPT
        e = llm.parse_json(llm.chat(a.teacher, [{"role": "user", "content": edit_prompt.format(
            genre_name=g["name"], genre_desc=genres.describe(genre), text=r["text"])}], tag="edit_ai" if ai_mode else "edit",
            timeout=a.teacher_timeout))
        level = e.get("edit_level", "heavy")
        cands = [c.strip() for c in e.get("candidates", []) if isinstance(c, str) and c.strip()]
        rec.update(diagnosis=e.get("diagnosis"), edit_level=level, candidates=cands)

        check = local_check_ai if ai_mode else local_check
        checked = [(c, check(r["text"], c, level, ban)) for c in cands]
        rec["local"] = [i for _, i in checked]
        survivors = [c for c, i in checked if not i]
        if not survivors:
            # 人写的好文字：候选都被规则拦下时，原文本身就是"不用改"的答案。AI 初稿永远不做 keep 样本
            if quality == "excellent" and not ai_mode:
                return {**rec, "status": "keep", "target": r["text"], "reason": "候选都没过规则，原文已经很好"}
            return {**rec, "status": "reject", "reason": "候选都没过规则"}

        rng = random.Random(zlib.crc32(r["id"].encode()))
        order = survivors[:]
        rng.shuffle(order)
        letters = "ABCDE"[: len(order)]
        block = "\n\n".join(f"版本 {L}：\n{c}" for L, c in zip(letters, order))
        judge_prompt = prompts.JUDGE_AI_PROMPT if ai_mode else prompts.JUDGE_PROMPT
        judge_tag = "judge_ai" if ai_mode else "judge"
        j = llm.parse_json(llm.chat(a.judge, [{"role": "user", "content": judge_prompt.format(
            genre_desc=genres.describe(genre), original=r["text"], candidates=block)}], tag=judge_tag))
        rec["judge"] = j
        reviews = j.get("reviews", {})
        best = None
        for L in j.get("ranking", []):
            rv = reviews.get(L, {})
            if (L in letters and not rv.get("fact_issues") and rv.get("register_kept", True) and not rv.get("slop")
                    and not (ai_mode and rv.get("ai_left"))):
                best = order[letters.index(L)]
                break
        verdict = j.get("best_vs_original", "better")
        if not best and not a.no_repair and reviews:
            # 返修：挑问题最少的版本，带着审稿意见让老师改一次，再单独审一次
            def n_issues(L):
                v = reviews.get(L, {})
                return (len(v.get("fact_issues") or []) + len(v.get("slop") or []) + (0 if v.get("register_kept", True) else 2)
                        + (len(v.get("ai_left") or []) if ai_mode else 0))
            L0 = min((L for L in letters if L in reviews), key=n_issues, default=None)
            if L0 is not None:
                v = reviews[L0]
                issue_list = (v.get("fact_issues") or []) + [f"新套话：{s}" for s in (v.get("slop") or [])]
                if ai_mode:
                    issue_list += [f"还留着的 AI 腔：{s}" for s in (v.get("ai_left") or [])]
                if not v.get("register_kept", True):
                    issue_list.append("文体或语气变了，要改回原文的文体")
                issues = "\n".join(f"- {x}" for x in issue_list)
                fixed = llm.chat(a.teacher, [{"role": "user", "content": prompts.REPAIR_PROMPT.format(
                    genre_desc=genres.describe(genre), original=r["text"], candidate=order[letters.index(L0)], issues=issues)}], tag="repair",
                    timeout=a.teacher_timeout).strip()
                li = check(r["text"], fixed, level, ban)
                rec["repair"] = {"from": L0, "text": fixed, "local": li}
                if not li:
                    j2 = llm.parse_json(llm.chat(a.judge, [{"role": "user", "content": judge_prompt.format(
                        genre_desc=genres.describe(genre), original=r["text"], candidates=f"版本 A：\n{fixed}")}], tag=judge_tag))
                    rec["repair"]["judge"] = j2
                    v2 = (j2.get("reviews") or {}).get("A", {})
                    if not v2.get("fact_issues") and v2.get("register_kept", True) and not v2.get("slop") \
                            and not (ai_mode and v2.get("ai_left")) \
                            and j2.get("best_vs_original") == "better":
                        return final_gate({**rec, "reason": "返修后通过"}, r["text"], fixed, genre, level, ai_mode)
        if best and verdict == "better":
            return final_gate(rec, r["text"], best, genre, level, ai_mode)
        if ai_mode:
            # AI 初稿不能当“不用改”的样本：裁判说 same/worse 一律拒
            return {**rec, "status": "reject", "reason": "没有合格候选" if not best else f"最好的候选也{verdict}于原文"}
        if quality == "excellent" and verdict in ("same", "worse"):
            return {**rec, "status": "keep", "target": r["text"], "reason": f"裁判认为改写{verdict}，保留原文"}
        return {**rec, "status": "reject", "reason": "没有合格候选" if not best else f"最好的候选也{verdict}于原文"}

    rows = read_jsonl(a.inp)
    run(rows, a.out, one, a.workers)

    stat = collections.defaultdict(collections.Counter)
    for r in read_jsonl(a.out):
        stat["总"][r["status"]] += 1
        stat[f"{r.get('kind')}"][r["status"]] += 1
        stat[f"文体 {r.get('genre')}"][r["status"]] += 1
        if r["status"] == "reject":
            stat["拒绝原因"][r.get("reason", "")[:20]] += 1
    for k, c in stat.items():
        print(f"{k:22s} {dict(c.most_common(6))}")


if __name__ == "__main__":
    main()
