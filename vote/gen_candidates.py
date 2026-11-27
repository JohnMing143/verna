"""评委站候选稿（11-27）：原文 × 系统（模型 × 提示词）。

原文按固定顺序排（native 测试集和 test 集在前，smoke2 会对它们出预测；其余随机），
跑的时候前面的原文先凑齐所有系统，额度或时间不够时，至少前面的题是完整的。

  python vote/gen_candidates.py sources          生成 vote/data/sources.jsonl
  python vote/gen_candidates.py run <组> [并发] [--limit N]
      组：gemini / carbon / deepseek / sonnet（见 GROUPS）；结果追加到 vote/data/cand_<组>.jsonl，可断点续跑
"""
import json, os, re, sys, random, time, threading, subprocess, collections
ROOT = "/root/shuorenhua"
sys.path.insert(0, ROOT)
from srh import llm, prompts, genres
from srh.runner import run, read_jsonl

D = f"{ROOT}/vote/data"
HAN = re.compile(r"[一-鿿]")
MIN, MAX, CUT_TO = 80, 450, 400
FLOOR = int(os.environ.get("VOTE_FLOOR_WEEKLY", "5"))   # Gemini 周额度剩到这个百分比就停


def han(s):
    return len(HAN.findall(s or ""))


def cut(text):
    """整段在长度范围内就整段用；太长就从开头按段落累加到 CUT_TO 字以内，不切句子。"""
    text = text.strip()
    if MIN <= han(text) <= MAX:
        return text
    if han(text) < MIN:
        return None
    sep = "\n\n" if "\n\n" in text else "\n"
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    out = []
    for p in paras:
        if han(sep.join(out + [p])) > CUT_TO:
            break
        out.append(p)
    while out and out[-1].rstrip("*").endswith(("：", ":")):   # 别停在“如下：”
        out.pop()
    t = sep.join(out)
    return t if han(t) >= 120 else None


def build_sources():
    rng = random.Random(1127)
    front, rest = [], []
    for r in read_jsonl(f"{ROOT}/data/native_test.jsonl"):
        if MIN <= han(r["src"]) <= MAX:
            front.append({"id": r["id"], "origin": "native", "genre": None, "task": r["prompt"], "writer": r["model"], "text": r["src"].strip()})
    test_ids = set()
    for r in read_jsonl(f"{ROOT}/data/v3/sft/test.jsonl"):
        test_ids.add(r["id"].split("#")[0])
        if r.get("recipe") in ("keep",) or r["src"].strip() == r["ref"].strip():
            continue
        if r.get("recipe") not in ("reverse", "reverse_ai", "revai", "ai", "ai2", "distill_ai", "ai_draft") and "#ai" not in r["id"] and "#rev" not in r["id"]:
            continue
        if MIN <= han(r["src"]) <= MAX:
            front.append({"id": "test:" + r["id"], "origin": "test", "genre": r.get("genre"), "task": None, "writer": None,
                          "text": r["src"].strip(), "human_ref": r["ref"].strip() if "#rev" in r["id"] and "#revai" not in r["id"] else None})
    by = collections.defaultdict(list)
    for r in read_jsonl(f"{ROOT}/data/v3/pairs/ai_drafts.jsonl"):
        t = cut(r.get("text") or "")
        if t and r["id"].split("#")[0] not in test_ids:
            by[r.get("genre")].append({"id": "ai:" + r["id"], "origin": "ai_draft", "genre": r.get("genre"), "task": r.get("task"),
                                       "writer": r.get("writer"), "text": t, "cut": t != r["text"].strip()})
    ai = []
    for g, xs in by.items():
        rng.shuffle(xs)
        ai += xs[:70]
    by = collections.defaultdict(list)
    for r in read_jsonl(f"{ROOT}/data/v3/pairs/reverse.jsonl"):
        d = (r.get("draft") or "").strip()
        if MIN <= han(d) <= MAX and r.get("original") and r["id"].split("#")[0] not in test_ids:
            by[r.get("genre")].append({"id": "rev:" + r["id"], "origin": "reverse", "genre": r.get("genre"), "task": None,
                                       "writer": r.get("degrader") or r.get("model"), "text": d, "human_ref": r["original"].strip()})
    rev = []
    for g, xs in by.items():
        rng.shuffle(xs)
        rev += xs[:40]
    rest = ai + rev
    rng.shuffle(rest)
    rng.shuffle(front)
    seen, rows = set(), []
    for i, r in enumerate(front + rest):
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        r["rank"] = len(rows)
        r["han"] = han(r["text"])
        rows.append(r)
    os.makedirs(D, exist_ok=True)
    with open(f"{D}/sources.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["origin"] for r in rows)
    print(len(rows), dict(c), "genres", dict(collections.Counter(r["genre"] for r in rows)))


# 提示词 full：现在管线老师用的 EDIT_AI_PROMPT（带全套规则），改成只要一个版本，同 data/neon_eval_1122/edit.py
P_FULL = prompts.EDIT_AI_PROMPT.replace("- candidates：3 个不同的改写版本，都要把诊断里的问题改掉", "- edit：改写后的全文，要把诊断里的问题改掉") \
    .replace('{{"diagnosis": ["..."], "edit_level": "...", "candidates": ["...", "...", "..."]}}', '{{"diagnosis": ["..."], "edit_level": "...", "edit": "..."}}')
assert "edit：改写后" in P_FULL and '"edit": "..."' in P_FULL
# 提示词 min：只说任务，不给规则，看各家模型自己的口味
P_MIN = """下面这段{genre_name}是 AI 写的，读起来有 AI 腔。请把它改写成好的中文：信息不增不减，文体和语气保持原样。

不要使用任何工具。只输出 JSON：{{"edit": "改写后的全文"}}

原文：
{text}"""

DS_MODEL = os.environ.get("VERNA_DS_MODEL", "deepseek-chat@https://api.deepseek.com/v1")  # <模型>@<OpenAI 兼容接口>
SYSTEMS = {
    "argon-m.full": ("agy:gemini-4-argon-medium", "full"),
    "argon-m.min": ("agy:gemini-4-argon-medium", "min"),
    "neon-m.full": ("agy:gemini-4-neon-medium", "full"),
    "neon-m.min": ("agy:gemini-4-neon-medium", "min"),
    "neon-h.min": ("agy:gemini-4-neon-high", "min"),
    "hydrogen-m.min": ("agy:gemini-4-hydrogen-medium", "min"),
    "flash-m.min": ("agy:gemini-3.8-flash-medium", "min"),
    "carbon-m.full": ("agy:gemini-4-carbon-medium", "full"),
    "carbon-m.min": ("agy:gemini-4-carbon-medium", "min"),
    "deepseek.full": (DS_MODEL, "full"),
    "deepseek.min": (DS_MODEL, "min"),
    "sonnet-l.min": ("agy:claude-sonnet-5-5-low", "min"),
}
GROUPS = {
    "gemini": ["argon-m.full", "neon-m.min", "argon-m.min", "neon-m.full", "hydrogen-m.min", "neon-h.min", "flash-m.min"],
    "gem1": ["argon-m.full", "neon-m.min", "argon-m.min", "neon-m.full"],   # 11-27 gemini 组太慢，拆成两个进程（各一个 agy acp）
    "gem2": ["hydrogen-m.min", "neon-h.min", "flash-m.min"],
    "carbon": ["carbon-m.full", "carbon-m.min"],
    "deepseek": ["deepseek.min", "deepseek.full"],
    "sonnet": ["sonnet-l.min"],
}

_q = {"t": 0, "weekly": 100}
_qlock = threading.Lock()
STOP = threading.Event()


def weekly_left():
    with _qlock:
        if time.time() - _q["t"] > 300:
            try:
                r = subprocess.run(["agy", "-p", "/quota"], capture_output=True, text=True, timeout=300, cwd="/tmp")
                for line in r.stdout.splitlines():
                    p = line.split("\t")
                    if len(p) >= 4 and p[0].startswith("Gemini") and "Weekly" in p[1]:
                        _q["weekly"] = int(p[2].rstrip("%"))
                _q["t"] = time.time()
                print(f"{time.strftime('%m-%d %H:%M')} Gemini 周额度剩 {_q['weekly']}%", file=sys.stderr, flush=True)
            except Exception as e:
                print("读额度失败", e, file=sys.stderr)
                _q["t"] = time.time() - 240
        return _q["weekly"]


def genre_name(s):
    g = genres.GENRES.get(s.get("genre") or "")
    name = g["name"] if g else "文字"
    return f"{name}（写作要求是：{s['task']}）" if s.get("task") else name


def make_fn(gemini_pool):
    def go(j):
        if STOP.is_set():
            return None
        if gemini_pool and weekly_left() <= FLOOR:
            STOP.set()
            print(f"周额度到 {FLOOR}%，停", file=sys.stderr, flush=True)
            return None
        s = j["s"]
        model, pv = SYSTEMS[j["system"]]
        if pv == "full":
            gd = genres.describe(s["genre"]) if s.get("genre") in genres.GENRES else ""
            p = P_FULL.format(genre_name=genre_name(s), genre_desc=gd, text=s["text"])
        else:
            p = P_MIN.format(genre_name=genre_name(s), text=s["text"])
        t = time.time()
        res = llm.ask_json(model, p, tag="vote-cand", timeout=900)
        out = (res.get("edit") or "").strip()
        if not out:
            raise RuntimeError("空输出")
        return {"id": j["id"], "src_id": s["id"], "rank": s["rank"], "system": j["system"], "model": llm.label(model), "prompt": pv,
                "out": out, "diagnosis": res.get("diagnosis"), "edit_level": res.get("edit_level"), "sec": round(time.time() - t, 1)}
    return go


def main():
    if sys.argv[1] == "sources":
        build_sources()
        return
    group = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 6
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    srcs = sorted(read_jsonl(f"{D}/sources.jsonl"), key=lambda r: r["rank"])
    if limit:
        srcs = srcs[:limit]
    jobs = [{"id": f"{s['id']}|{sy}", "system": sy, "s": s} for s in srcs for sy in GROUPS[group]]
    run(jobs, f"{D}/cand_{group}.jsonl", make_fn(group in ("gemini", "gem1", "gem2", "carbon")), workers)
    llm.shutdown_agy()


if __name__ == "__main__":
    main()
