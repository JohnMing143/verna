"""把候选稿配成评审题（11-27）：每段原文出一道题，两份改写来自指定的两个系统。

  python vote/build_pool.py pair <批次名> <系统A> <系统B> [--pred 名字=路径 ...] [--sources native,test] [--limit N]
  python vote/build_pool.py probes <批次名> <系统> [--n 20]      保真题：同一份改写，另一份悄悄改了一个数字或限定词

输出 vote/data/pool_<批次名>.jsonl，格式就是 admin.py import-items 要的格式。
过滤规则按 vote/DESIGN.md 第 3 节：两份改写字级相似度 < 0.9、差别不只在标点；哪一份和原文几乎一样（只差标点格式，或相似度 ≥ 0.97）
也不出题，因为三选一里它和“原文更好”分不开；两份都过 srh/facts.py
（对照的是去掉助手附言后的原文，删掉“约 330 字，如需调整……”不算丢信息）。没过的数量会打印出来，事实规则没过的
比例本身就是保真指标，要和投票结果一起报告。
--pred 用来接训练出来的模型：runs/<RUN>/<时间>/pred_native.jsonl 和 pred_test.jsonl，名字自己起（比如 qwen-s2）。
"""
import argparse, collections, difflib, glob, json, os, random, re, sys

ROOT = "/root/shuorenhua"
sys.path.insert(0, ROOT)
from srh import facts, prompts, tells  # noqa: E402

D = f"{ROOT}/vote/data"
HAN = re.compile(r"[一-鿿]")
PUNCT = re.compile(r"[\s，。、；：！？“”‘’（）《》…—\-,.;:!?()\"'*#>]+")

SHORT = {"academic": "学术", "news": "新闻", "official": "公文", "business": "工作文书", "tech": "技术文档",
         "popsci": "科普", "opinion": "评论", "forum": "网络帖子", "essay": "散文", "fiction": "小说",
         "marketing": "营销文案", "speech": "演讲稿"}
NATIVE = {"000": "公众号文章", "001": "工作邮件", "002": "科普", "003": "读书笔记", "004": "小红书笔记", "005": "周报",
          "006": "理财科普", "007": "产品介绍", "008": "科普", "009": "短视频旁白", "010": "道歉声明", "011": "演讲稿",
          "012": "知识总结", "013": "旅行攻略", "014": "影评", "015": "求职自我介绍", "016": "技术科普", "017": "育儿建议",
          "018": "年终总结", "019": "评论", "020": "朋友圈", "021": "技术入门", "022": "书评", "023": "健身建议",
          "024": "公司通知", "025": "科普", "026": "生活指南", "027": "评论", "028": "活动通知", "029": "历史介绍"}


def label(s):
    if s["origin"] == "native":
        return NATIVE.get(s["id"][6:9], "")
    return SHORT.get(s.get("genre") or "", "")


def han(t):
    return len(HAN.findall(t or ""))


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def load_sources():
    return {r["id"]: r for r in map(json.loads, open(f"{D}/sources.jsonl", encoding="utf-8"))}


def load_outputs(preds):
    outs = collections.defaultdict(dict)
    for p in sorted(glob.glob(f"{D}/cand_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("out"):
                outs[r["system"]][r["src_id"]] = r["out"].strip()
    for spec in preds:
        name, path = spec.split("=", 1)
        for f, pre in (("pred_native.jsonl", ""), ("pred_test.jsonl", "test:")):
            fp = os.path.join(path, f)
            if os.path.exists(fp):
                for line in open(fp, encoding="utf-8"):
                    r = json.loads(line)
                    if r.get("out"):
                        outs[name][pre + r["id"]] = r["out"].strip()
    return outs


def same_as_source(src, out):
    """只差标点、空白、加粗，或者字级相似度 ≥ 0.97。"""
    return PUNCT.sub("", src) == PUNCT.sub("", out) or sim(src, out) >= 0.97


def fact_ok(src, out):
    return facts.compare(prompts.strip_assistant_meta(src), out)["ok"]


def write(batch, rows):
    path = f"{D}/pool_{batch}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {path}：{len(rows)} 道")


def cmd_pair(a):
    S = load_sources()
    outs = load_outputs(a.pred)
    origins = set(a.sources.split(","))
    for sy in (a.sys_a, a.sys_b):
        if sy not in outs:
            sys.exit(f"没有系统 {sy}；有：{sorted(outs)}")
    rng = random.Random(f"{a.batch}")
    drop = collections.Counter()
    rows = []
    for sid, s in sorted(S.items(), key=lambda kv: kv[1]["rank"]):
        if s["origin"] not in origins:
            continue
        x, y = outs[a.sys_a].get(sid), outs[a.sys_b].get(sid)
        if not x or not y:
            drop["缺候选"] += 1
            continue
        bad = [n for n, t in ((a.sys_a, x), (a.sys_b, y)) if not fact_ok(s["text"], t)]
        for n in bad:
            drop[f"事实规则没过：{n}"] += 1
        if bad:
            continue
        same = [n for n, t in ((a.sys_a, x), (a.sys_b, y)) if same_as_source(s["text"], t)]
        for n in same:
            drop[f"改写和原文几乎一样：{n}"] += 1
        if same:      # 三选一里，这一份和“原文更好”就分不开了；这个比例单独报告
            continue
        if x == y or PUNCT.sub("", x) == PUNCT.sub("", y):
            drop["只差标点"] += 1
            continue
        if sim(x, y) >= 0.9:
            drop["两份太像（≥0.9）"] += 1
            continue
        if han(s["text"]) + han(x) + han(y) > 1300:
            drop["三段超过 1300 字"] += 1
            continue
        if not tells.hits(s["text"]) and s["origin"] != "native":
            drop["原文没测到 AI 腔"] += 1
            continue
        pair = [(a.sys_a, x), (a.sys_b, y)]
        rng.shuffle(pair)
        rows.append({"id": f"{a.batch}-{len(rows) + 1:03d}", "genre": label(s), "source": s["text"],
                     "a": pair[0][1], "b": pair[1][1], "a_system": pair[0][0], "b_system": pair[1][0],
                     "kind": "normal", "meta": {"src_id": sid, "origin": s["origin"], "batch": a.batch}})
        if a.limit and len(rows) >= a.limit:
            break
    print("没进题库：", dict(drop))
    write(a.batch, rows)


# 只删独立的限定词：“约”后面要跟数字，“左右”后面要是标点，“可能”不能是“可能性”（不然会删出“缔自由”“会、闺蜜局”）
HEDGES = [r"大约", r"约(?=\s?[\d一二三四五六七八九十两百千万几])", r"左右(?=[，。、；！？\s）)]|$)", r"可能(?!性)", r"初步", r"通常"]


def perturb(text, rng):
    """改一个数字，或删一个限定词。返回 (新文本, 说明)，改不了返回 None。"""
    nums = [m for m in re.finditer(r"(?<![\d.])(\d{1,4})(?![\d.])", text)
            if not re.match(r"\s*[.、)）]", text[m.end():m.end() + 2])          # 列表编号不改
            and not (m.start() == 0 or text[m.start() - 1] == "\n")
            and not (len(m.group(1)) > 1 and m.group(1).startswith("0"))           # 0084 这类编号
            and text[m.end():m.end() + 1] not in ("月", "日", "号", ":", "：", "点")   # 日期时刻容易改出不存在的值
            and text[m.end():m.end() + 2] != "年代"
            and not (m.start() > 0 and text[m.start() - 1].isascii() and text[m.start() - 1].isalpha())  # Q4、V8 这类
            and not (text[m.end():m.end() + 1].isascii() and text[m.end():m.end() + 1].isalpha())        # 5A 级、7B 模型
            and text[m.end():m.end() + 1] not in ("—", "-", "~", "至", "到")
            and text[max(0, m.start() - 1):m.start()] not in ("—", "-", "~", "至", "到")]                 # 区间两头改一头会改出倒序
    hedges = [(m.group(0), m.start()) for h in HEDGES for m in re.finditer(h, text)]
    opts = []
    if nums:
        m = rng.choice(nums)
        v = int(m.group(1))
        z = len(m.group(1)) - len(m.group(1).rstrip("0"))
        if 1900 <= v <= 2100:                     # 年份只挪一两年，改成 3033 一眼就看出来了
            nv = v + rng.choice([-2, -1, 1, 2])
        elif v < 10:
            nv = v + rng.choice([1, 2])
        else:                                     # 保持整数位数：40→30/50，2800→2700/2900，45→43/47
            nv = v + 10 ** z * rng.choice([-2, -1, 1, 2])
        if nv != v and nv > 0:
            opts.append((text[:m.start()] + str(nv) + text[m.end():], f"数字 {v}→{nv}"))
    if hedges:
        h, i = rng.choice(hedges)
        opts.append((text[:i] + text[i + len(h):], f"删掉“{h}”"))
    return rng.choice(opts) if opts else None


def cmd_probes(a):
    S = load_sources()
    outs = load_outputs([])[a.system]
    used = set()
    for p in glob.glob(f"{D}/pool_*.jsonl"):
        for line in open(p, encoding="utf-8"):
            used.add(json.loads(line)["meta"].get("src_id"))
    rng = random.Random(f"{a.batch}-probe")
    cands = [s for s in sorted(S.values(), key=lambda s: s["rank"]) if s["id"] not in used and s["id"] in outs
             and s["origin"] in ("ai_draft", "reverse") and fact_ok(s["text"], outs[s["id"]])]
    rng.shuffle(cands)
    rows = []
    for s in cands:
        good = outs[s["id"]]
        r = perturb(good, rng)
        if not r or han(s["text"]) + 2 * han(good) > 1300:
            continue
        bad, what = r
        right = rng.choice("ab")
        ab = (good, bad) if right == "a" else (bad, good)
        rows.append({"id": f"{a.batch}-p{len(rows) + 1:02d}", "genre": label(s), "source": s["text"], "a": ab[0], "b": ab[1],
                     "a_system": a.system if right == "a" else a.system + "+probe", "b_system": a.system if right == "b" else a.system + "+probe",
                     "kind": "probe", "probe_answer": right, "meta": {"src_id": s["id"], "origin": s["origin"], "batch": a.batch, "probe": what}})
        if len(rows) >= a.n:
            break
    write(a.batch, rows)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pair")
    p.add_argument("batch"); p.add_argument("sys_a"); p.add_argument("sys_b")
    p.add_argument("--pred", action="append", default=[])
    p.add_argument("--sources", default="native,test")
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("probes")
    p.add_argument("batch"); p.add_argument("system")
    p.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    {"pair": cmd_pair, "probes": cmd_probes}[a.cmd](a)


if __name__ == "__main__":
    main()
