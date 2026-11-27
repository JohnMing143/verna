"""评审结果汇总（vote/DESIGN.md 第 4.5、5、6 节）：评委权重、系统排名、单题结论、理由统计。

  python vote/report.py [--db /var/lib/verna/verna.db] [--batch s2q] [--json]

- 评委权重：只看保真题（以后加上语境对照题）答对的比例，Beta(4,1) 先验收缩，映射到 0.3–1.5；没做过保真题的按 1.0。
  不按“和多数人一致”给权重。
- 系统排名：Bradley-Terry，原文也是一个选手。三选一，选中的那份同时胜过另外两份：“A 更好”记 A 胜 B、A 胜原文；
  “原文更好”记原文同时胜 A 和 B。
  一位评委在某一对系统比较里的总权重不超过这一对总权重的 10%（票数少于 10 位评委时不封顶，否则一个人都投不满）。
- 单题结论：加权多数，胜方 ≥70% 才算有结论；不到 5 票的题标“票数不够”。
保真题不进排名，只用来算权重和报告评委读没读原文。
"""
import argparse, collections, json, math, sqlite3

CAP_SHARE, MIN_VOTES, DECISIVE = 0.10, 5, 0.70


def load(db, batch):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    items = {r["id"]: dict(r) for r in c.execute("select id, a_system, b_system, kind, probe_answer, meta, status from items")}
    votes = [dict(r) for r in c.execute("select item_id, judge_id, choice, ms, reasons, note from votes")]
    skips = collections.Counter(r[0] for r in c.execute("select item_id from skips"))
    if batch:
        keep = {i for i, it in items.items() if (json.loads(it["meta"] or "{}").get("batch") or "").startswith(batch) or it["kind"] == "probe"}
        votes = [v for v in votes if v["item_id"] in keep]
    return items, votes, skips


def judge_weights(items, votes):
    tally = collections.defaultdict(lambda: [0, 0])
    for v in votes:
        it = items.get(v["item_id"])
        if it and it["kind"] == "probe":
            tally[v["judge_id"]][1] += 1
            tally[v["judge_id"]][0] += v["choice"] == it["probe_answer"]
    w = {}
    for j, (ok, n) in tally.items():
        p = (ok + 4) / (n + 5)                       # Beta(4,1)：新评委默认 0.8 的正确率
        w[j] = round(min(1.5, max(0.3, 0.3 + (p - 0.5) / 0.5 * 1.2)), 3) if p > 0.5 else 0.3
    return w, tally


def bradley_terry(games, iters=500):
    """games: [(胜者, 负者, 权重)]。MM 算法；返回对数强度，原文固定为 0。"""
    players = sorted({p for g in games for p in g[:2]})
    s = {p: 1.0 for p in players}
    wins = collections.Counter()
    pair_w = collections.Counter()
    for a, b, w in games:
        wins[a] += w
        pair_w[frozenset((a, b))] += w
    for _ in range(iters):
        new = {}
        for p in players:
            den = sum(w / (s[p] + s[q]) for k, w in pair_w.items() if p in k for q in k - {p})
            new[p] = (wins[p] + 0.5) / (den + 1 / s[p])   # 加 0.5 场虚拟平局，免得全胜全败发散
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        s = {p: v / g for p, v in new.items()}
    base = math.log(s["原文"]) if "原文" in s else 0
    return {p: round(math.log(v) - base, 3) for p, v in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/var/lib/verna/verna.db")
    ap.add_argument("--batch", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    items, votes, skips = load(a.db, a.batch)
    w, tally = judge_weights(items, votes)
    normal = [v for v in votes if items.get(v["item_id"], {}).get("kind") == "normal"]

    # 每一对系统里，每位评委的权重封顶
    by_pair = collections.defaultdict(lambda: collections.defaultdict(float))
    for v in normal:
        it = items[v["item_id"]]
        by_pair[frozenset((it["a_system"], it["b_system"]))][v["judge_id"]] += w.get(v["judge_id"], 1.0)
    scale = {}
    for pair, per in by_pair.items():
        tot = sum(per.values())
        for j, x in per.items():
            cap = CAP_SHARE * tot if len(per) >= 10 else x
            scale[(pair, j)] = min(1.0, cap / x) if x else 1.0

    games, h2h = [], collections.defaultdict(lambda: collections.Counter())
    per_item = collections.defaultdict(collections.Counter)
    reasons = collections.defaultdict(collections.Counter)
    for v in normal:
        it = items[v["item_id"]]
        A, B = it["a_system"], it["b_system"]
        pair = frozenset((A, B))
        wt = w.get(v["judge_id"], 1.0) * scale[(pair, v["judge_id"])]
        key = tuple(sorted((A, B)))
        win = {"a": A, "b": B, "src": "原文"}[v["choice"]]
        h2h[key][win] += wt
        per_item[v["item_id"]][win] += wt
        per_item[v["item_id"]]["_n"] += 1
        # 三选一：选中的那份同时胜过另外两份（“A 更好”也说明 A 不比原文差，不然该选“原文更好”）
        games += [(win, lose, wt) for lose in (A, B, "原文") if lose != win]
        for r in json.loads(v["reasons"] or "[]"):
            reasons[win][r] += 1

    out = {"votes": len(votes), "normal_votes": len(normal), "judges": len({v["judge_id"] for v in votes}),
           "bt": bradley_terry(games) if games else {},
           "head_to_head": {f"{k[0]} vs {k[1]}": {s: round(x, 2) for s, x in c.items()} for k, c in h2h.items()},
           "probe": {"answered": sum(t[1] for t in tally.values()), "correct": sum(t[0] for t in tally.values())},
           "weights": w}
    concl = collections.Counter()
    for iid, c in per_item.items():
        n = c.pop("_n")
        tot = sum(c.values())
        top, x = c.most_common(1)[0]
        concl["票数不够" if n < MIN_VOTES else (f"{top} 胜" if x / tot >= DECISIVE else "有争议")] += 1
    out["items"] = dict(concl)
    out["skips"] = sum(skips.values())
    out["reasons"] = {k: dict(c.most_common(8)) for k, c in reasons.items()}
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    print(f"票 {out['votes']}（普通题 {out['normal_votes']}），评委 {out['judges']}，跳过 {out['skips']}")
    print(f"保真题：答了 {out['probe']['answered']}，答对 {out['probe']['correct']}")
    print("排名（Bradley-Terry 对数强度，原文 = 0，越大越好）：")
    for p, x in sorted(out["bt"].items(), key=lambda kv: -kv[1]):
        print(f"  {p:18} {x:+.2f}")
    print("两两对比（加权票）：")
    for k, c in out["head_to_head"].items():
        print(f"  {k}: {c}")
    print("单题结论：", out["items"])
    for k, c in out["reasons"].items():
        print(f"  选 {k} 的理由：{c}")


if __name__ == "__main__":
    main()
