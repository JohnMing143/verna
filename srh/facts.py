"""确定性的事实核对：数字（含中文数字、百分比、万/亿）、英文词和型号、书名号、引语。

compare(ref, cand) 比较两段文字，返回 hard（必须一致，不一致就判失败）和 soft（提醒）两类差异。
它抓不到"人名换了""因果反了""可能变成一定"——那些交给 judge 提示词（prompts.FACT_JUDGE）。
用在三处：造数据时过滤配对、评测、推理时的守卫（不一致就重采样，再不行就退回原文）。
"""
from __future__ import annotations

import re

_D = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_U = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10**4, "亿": 10**8}
_CN = "零〇一二两三四五六七八九十百千万亿"

# 中文数字后面跟这些字才算"数量"，避免把"一样""千万别""三心二意"当数字
_UNITS = "年月日号岁个人次倍元块角分万亿%米里斤吨克升度天周时秒页种项条家所位名台辆架场部篇本张件处只头匹座户亩成届期季代层级国例起批份集章节轮局"
# 这些量词前面的小数字多半是结构（"三点""两个方面"），算 soft
_STRUCT = ("点", "方面", "步", "招", "条", "类", "种", "项", "大", "个方面", "个原因", "个问题", "个建议")


def cn2num(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    if all(c in _D for c in s) and len(s) > 1:  # 二〇二三
        return float("".join(str(_D[c]) for c in s))
    total, section, num = 0, 0, 0
    for c in s:
        if c in _D:
            num = _D[c]
        elif c in _U:
            section += (num or 1) * _U[c]
            num = 0
        elif c in _BIG:
            section += num
            total += section * _BIG[c]
            section, num = 0, 0
        else:
            return None
    return float(total + section + num)


_RX_AR = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(万|亿|千)?\s*(%|％|‰)?")
_RX_PCT_CN = re.compile(r"(百|千)分之([" + _CN + r"点]+)")
_RX_FRAC_CN = re.compile(r"([" + _CN + r"]+)分之([" + _CN + r"]+)")
_RX_CN = re.compile(r"(第)?([" + _CN + r"]+(?:点[" + _CN[:-5] + r"]+)?)[多余来几]?(成|个?方面|点|[" + _UNITS + r"])?")


def _cn_decimal(raw: str) -> float | None:
    if "点" not in raw:
        return cn2num(raw)
    a, b = raw.split("点", 1)
    ia = cn2num(a) if a else 0
    if ia is None:
        return None
    return float(f"{int(ia)}." + "".join(str(_D[c]) for c in b if c in _D))
_RX_LISTMARK = re.compile(r"(?m)^\s*\d{1,2}[.、)）](?!\d)\s*")


def _num(v: float) -> str:
    return f"{v:.6g}"


def numbers(text: str) -> tuple[set[str], set[str]]:
    """返回 (实数集合, 结构性小数字集合)。值统一成字符串：'12000'、'pct:30'。"""
    hard: set[str] = set()
    soft: set[str] = set()
    t = _RX_LISTMARK.sub("", text)  # 列表序号不算
    for m in _RX_AR.finditer(t):
        intpart, dec, big, pct = m.groups()
        v = float(intpart.replace(",", "") + (dec or ""))
        if big:
            v *= {"万": 1e4, "亿": 1e8, "千": 1e3}[big]
        key = f"pct:{_num(v)}" if pct in ("%", "％") else (f"permil:{_num(v)}" if pct == "‰" else _num(v))
        before = t[max(0, m.start() - 1):m.start()]
        after = t[m.end():m.end() + 3]
        if before == "第" or (v <= 10 and not pct and after.startswith(_STRUCT)):
            soft.add(key)
        else:
            hard.add(key)
    taken: list[tuple[int, int]] = []  # 已按百分数/分数处理过的位置
    for m in _RX_PCT_CN.finditer(t):
        v = _cn_decimal(m.group(2))
        if v is not None:
            hard.add(f"pct:{_num(v)}" if m.group(1) == "百" else f"permil:{_num(v)}")
            taken.append(m.span())
    for m in _RX_FRAC_CN.finditer(t):
        if any(a <= m.start() < b for a, b in taken):
            continue
        d, n = cn2num(m.group(1)), cn2num(m.group(2))
        if d and n is not None:
            hard.add(f"frac:{_num(n)}/{_num(d)}")
            taken.append(m.span())
    for m in _RX_CN.finditer(t):
        ordinal, raw, unit = m.groups()
        if any(a <= m.start() < b for a, b in taken):
            continue
        if re.search(r"\d\s*$", t[max(0, m.start() - 3):m.start()]):  # "1,200 万"的"万"属于前面的阿拉伯数字
            continue
        v = _cn_decimal(raw)
        if v is None or v == 1 or (v == 0 and not any(c in "零〇" for c in raw)):
            continue
        if raw in ("百", "千", "万", "亿", "千万", "万万", "亿万"):  # 千头万绪、百年老店、万人空巷：成语里的虚数
            continue
        if raw == "十" and unit == "分" and t[m.end():m.end() + 1] != "钟":  # "十分严重"是副词；"十分钟"照算
            continue
        if unit is None:
            # 没跟量词：只认"三万""两亿"这种带具体数字的；"千万别""万万没想到"不算
            if not (raw[-1] in _BIG and any(c in "二两三四五六七八九" for c in raw) or (raw[-1] in _BIG and raw[0] == "一" and len(raw) > 1)):
                continue
        if unit == "成":
            hard.add(f"pct:{_num(v * 10)}")
            continue
        key = _num(v)
        if ordinal or (v <= 10 and unit in ("点", "方面", "个方面", "条", "类", "种", "项", "步", "招")):
            soft.add(key)
        else:
            hard.add(key)
    return hard, soft - hard


_RX_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.+/][A-Za-z0-9]+)*\+?")
_LATIN_IGNORE = {"ai"}  # 改写时常被加上/去掉的泛称，按需增减


def latin(text: str) -> set[str]:
    return {w.lower() for w in _RX_LATIN.findall(text) if w.lower() not in _LATIN_IGNORE}


_RX_TITLE = re.compile(r"《([^》\n]{1,40})》")
_RX_QUOTE = re.compile(r"[“\"「『]([^”\"」』\n]{2,80})[”\"」』]")


def titles(text: str) -> set[str]:
    return set(_RX_TITLE.findall(text))


def quotes(text: str) -> set[str]:
    return set(_RX_QUOTE.findall(text))


def _plain(s: str) -> str:
    return re.sub(r"[\s，,。.！!？?、；;：:“”\"「」『』‘’'（）()]", "", s)


def compare(ref: str, cand: str, long_quote: int = 8) -> dict:
    """ref 是信息来源，cand 是改写。hard 里有任何一项就判失败。"""
    hard, soft = [], []
    rn, rs = numbers(ref)
    cn, cs = numbers(cand)
    cand_all = cn | cs
    for k in sorted(rn - cand_all):
        hard.append(f"丢了数字 {k}")
    for k in sorted(cn - (rn | rs)):
        hard.append(f"多了数字 {k}")
    for k in sorted(rs - cand_all):
        soft.append(f"丢了结构性数字 {k}")
    for k in sorted(cs - (rn | rs)):
        soft.append(f"多了结构性数字 {k}")

    rl, cl = latin(ref), latin(cand)
    hard += [f"丢了英文/型号 {w}" for w in sorted(rl - cl)]
    hard += [f"多了英文/型号 {w}" for w in sorted(cl - rl)]

    # 书名比较书名本身在不在，不管有没有书名号（改写时给"辛丑条约"加上书名号不算改事实）
    pc0, pr0 = _plain(cand), _plain(ref)
    hard += [f"丢了书名 《{w}》" for w in sorted(titles(ref)) if _plain(w) not in pc0]
    hard += [f"多了书名 《{w}》" for w in sorted(titles(cand)) if _plain(w) not in pr0]

    pc, pr = _plain(cand), _plain(ref)
    for q in sorted(quotes(ref)):
        if _plain(q) in pc:
            continue
        (hard if len(q) >= long_quote else soft).append(f"引语变了 “{q}”")
    for q in sorted(quotes(cand)):
        if len(q) >= long_quote and _plain(q) not in pr:
            hard.append(f"多了引语 “{q}”")
    return {"ok": not hard, "hard": hard, "soft": soft}



def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\W", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def added_clauses(ref: str, cand: str, min_len: int = 12, thr: float = 0.5) -> list[str]:
    """cand 里和 ref 几乎对不上的分句（12 字以上、字二元组只有不到一半在 ref 里出现）：新加的句子或整句改写。
    只修小毛病的精修不该有。11-05 复查：906 条精修稿里 15 条，比如补了原文没有的定义"内卷化是指……的现象"、
    把引语"你尽可以攻击我，但是我还是站在这里"改成"不管怎么被打压，还是屹立不倒"；长度比和保留率都拦不住。"""
    rb = _bigrams(ref)
    out = []
    for c in re.split(r"[。！？!?；;：:\n]", cand):
        b = _bigrams(c)
        if len(c.strip()) >= min_len and b and len(b & rb) / len(b) < thr:
            out.append(c.strip())
    return out

if __name__ == "__main__":
    import json
    import sys

    a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
    print(json.dumps(compare(a, b), ensure_ascii=False, indent=2))
