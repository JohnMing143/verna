#!/usr/bin/env python3
"""把评估为 flagged 的来源按 reviews.json 里写的清洗要求清洗，产物登记成新来源 `<名字>.clean`（状态 pending_review），
复核通过（review.py 改成 clean）后才能用。原始文件不动。

流程：从 gcorpus 拉原文到 corpus/work/clean/<名字>/ → 逐条清洗、记每条规则去掉了多少 → 写 jsonl.gz
→ 上传 gcorpus:verna-corpus/<语言>/<名字>.clean/ 并核对 md5 → 登记 state/files.jsonl、sources.json、reviews.json
→ 抽 300 条到 samples/<名字>.clean.jsonl → 清洗报告写到 corpus/clean_reports/<名字>.clean.json。

用法：.venv/bin/python corpus/clean.py zh.zhihu_kol [--keep-work] [--dry]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

C = Path(__file__).resolve().parent
sys.path.insert(0, str(C))
import collect  # noqa: E402

WORK = C / "work" / "clean"
REPORTS = C / "clean_reports"
SHARD_ROWS = 200_000

# ---------------- 知乎 Zhihu-KOL ----------------

_ZH = r"[一-鿿]"
_ZHP = r"[一-鿿，。！？；：、“”‘’（）《》〈〉【】…—·]"
_END = "。！？…；：:”’）)!?"
_LIST = re.compile(r"^\s*(?:\d{1,2}\s*[、.．)）]|[（(]?[一二三四五六七八九十]{1,3}[）)、]|第[一二三四五六七八九十\d]+[章节条步点个]|[①-⑳]|[-*•·])")
_PROMO = re.compile(r"关注我|私信我|公众号[：:「“『]|扫码|加微信|微信号|[vV][xX][：:]|优惠券|领券|知乎好物|素质三连|一键三连|三连支持|点赞关注|求赞|求关注")
_CAPTION = re.compile(r"^(?:图\s*\d*\s*[：:]|图片?来[自源]|图源|图片出处|来源[：:]|配图|[（(]图片?来[自源]网络[）)])|[（(]图[）)]$|via\s")
_DROP_LINE = re.compile(r"编辑于|发布于|著作权归|未经授权|禁止转载|转载请|^谢邀[。，！]?$|^利益相关")
_LEAD = re.compile(r"^(?:谢邀[^。！？\n]{0,12}[。！，]\s*|泻药[。！，]?\s*)")


def _restore_paras(t: str) -> list[str]:
    """Zhihu-KOL 把换行都换成了空格。句末标点后的空白、两个汉字之间的空白还原成换行；
    英文单词之间、中英文之间的空白保留（多是正常的空格）。"""
    t = t.replace("　", " ").replace("\xa0", " ")
    t = re.sub(r"https?://[\w\-./?=&%#:~+@!,;]+", "", t)          # 网址去掉（11-06 复核：清洗版样本 5.7% 带网址）
    t = re.sub(r"[（(]\s*[）)]", "", t)                              # 去网址后留下的空括号
    t = re.sub(rf"(?<=[{re.escape(_END)}])\s+", "\n", t)
    t = re.sub(rf"(?<={_ZHP})\s+(?={_ZHP})", "\n", t)
    t = re.sub(r"\s{2,}(?=(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[、.．)）])", "\n", t)   # 双空格后跟序号：原来是换行
    t = re.sub(r"\s*[—\-=_*~]{5,}\s*", "\n", t)                      # 分隔线
    lines = [x.strip() for x in t.split("\n")]
    return [x for x in lines if x and not re.fullmatch(r"\W{0,3}", x)]


def _is_caption(line: str) -> bool:
    """图片说明/出处行。11-06 第一版把“≤20 字、没有标点”的行都当图片说明删，抽查发现大多是正文
    （知乎很多人分行写短句不加标点、小标题不带序号），改成只认明确的图片出处写法。"""
    return len(line) <= 40 and bool(_CAPTION.search(line))


def clean_zhihu(src_dir: Path, stats: Counter):
    import pyarrow.parquet as pq
    seen_ids, seen_text = set(), set()
    for f in sorted(src_dir.rglob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=5000, columns=["INSTRUCTION", "RESPONSE", "METADATA"]):
            for r in batch.to_pylist():
                stats["输入"] += 1
                try:
                    m = json.loads(r.get("METADATA") or "{}")
                except json.JSONDecodeError:
                    stats["去掉：元数据坏"] += 1
                    continue
                t = m.get("answer_creation_time") or ""
                if not t or t >= "2022-11-30":
                    stats["去掉：2022-11-30 及以后或没有时间"] += 1
                    continue
                up = _upvotes(m.get("upvotes"))
                if up < 100:
                    stats["去掉：赞同不到 100"] += 1
                    continue
                aid = str(int(m["answer_id"])) if m.get("answer_id") else None
                if not aid or aid in seen_ids:
                    stats["去掉：没有回答 id 或重复"] += 1
                    continue
                text = r.get("RESPONSE") or ""
                if _PROMO.search(text):
                    stats["去掉：引流/带货话术"] += 1
                    continue
                paras = _restore_paras(text)
                out, caps = [], 0
                for i, p in enumerate(paras):
                    if _DROP_LINE.search(p):
                        stats["行：删掉编辑时间/版权/谢邀行"] += 1
                        continue
                    if _is_caption(p):
                        caps += 1
                        continue
                    out.append(p)
                if out:
                    out[0] = _LEAD.sub("", out[0]).strip()
                    out = [p for p in out if p]
                stats["行：删掉图片说明"] += caps
                body = "\n".join(out)
                if len(body) < 300:
                    stats["去掉：清洗后不到 300 字"] += 1
                    continue
                h = hashlib.md5(body.encode()).hexdigest()
                if h in seen_text:
                    stats["去掉：正文重复"] += 1
                    continue
                seen_ids.add(aid)
                seen_text.add(h)
                stats["保留"] += 1
                yield {"id": f"zhihu:{aid}", "question_id": str(int(m["question_id"])) if m.get("question_id") else None,
                       "question": r.get("INSTRUCTION"), "text": body, "upvotes": int(up), "created": t[:19],
                       "url": m.get("url"), "n_paras": len(out), "captions_removed": caps}


def _upvotes(s) -> float:
    m = re.search(r"([\d.]+)\s*(万|[kK])?", str(s or "").replace(",", ""))
    return float(m.group(1)) * {"万": 1e4, "k": 1e3, "K": 1e3}.get(m.group(2) or "", 1) if m else 0.0


# ---------------- CSL 中文核心期刊摘要 ----------------

_CJK = "一-鿿"


def _csl_punct(t: str) -> tuple[str, int]:
    """半角标点转全角（只动挨着汉字的），修残缺括号“(TG-DTA(表明”→“(TG-DTA)表明”。返回 (文本, 改动数)。"""
    n0 = t
    t = re.sub(r"\(([A-Za-z0-9\-+/ ,.]{1,30})\((?=[" + _CJK + "，。])", r"(\1)", t)
    t = re.sub(r"(?<=[" + _CJK + r"\)）%])\s*,\s*", "，", t)
    t = re.sub(r",\s*(?=[" + _CJK + "])", "，", t)
    t = re.sub(r"(?<=[" + _CJK + r"\)）%])\s*;\s*", "；", t)
    t = re.sub(r";\s*(?=[" + _CJK + "])", "；", t)
    t = re.sub(r"(?<=[" + _CJK + r"])\s*:\s*", "：", t)
    t = re.sub(r"(?<=[" + _CJK + r"\)）%\"”])\.(?=\s*(?:[" + _CJK + r"]|$))", "。", t)
    return t, sum(1 for a, b in zip(n0, t) if a != b) + abs(len(n0) - len(t))


def clean_csl(src_dir: Path, stats: Counter):
    seen = set()
    f = next(src_dir.rglob("csl.jsonl.gz"))   # csl.gt.* 是英文机翻版，不用
    with gzip.open(f, "rt") as fh:
        for line in fh:
            stats["输入"] += 1
            r = json.loads(line)
            ab = (r.get("abstract") or "").strip()
            han = len(re.findall(_ZH, ab))
            if han < len(ab) * 0.3:
                stats["去掉：不是中文摘要"] += 1
                continue
            if len(ab) < 80:
                stats["去掉：不到 80 字"] += 1
                continue
            formula = len(re.findall(r"_\{?\d|\^|\$|\\[a-z]+", ab))
            latin = len(re.findall(r"[A-Za-z]", ab)) / max(len(ab), 1)
            if formula >= 3 or latin > 0.3:
                stats["去掉：公式/化学式/外文占比高"] += 1
                continue
            ab2, k = _csl_punct(ab)
            if k:
                stats["改：标点/括号"] += 1
            h = hashlib.md5(ab2.encode()).hexdigest()
            if h in seen:
                stats["去掉：重复"] += 1
                continue
            seen.add(h)
            stats["保留"] += 1
            yield {"id": f"csl:{r.get('doc_id')}", "title": _csl_punct(r.get("title") or "")[0], "abstract": ab2,
                   "keywords": r.get("keywords"), "category": r.get("category"), "discipline": r.get("discipline")}


# ---------------- THUCNews 新浪新闻 ----------------

_THU_DROP_CAT = {"股票", "彩票", "星座"}
_THU_PROMO_CAT = {"科技", "家居", "房产", "时尚", "游戏"}
_THU_PROMO = re.compile(r"促销|报价|售价|优惠|团购|热线|经销商|订购|咨询电话|电话[：:]|地址[：:]|[0-9]{3,5}元|特价|直降|礼包|免费领取|赠送")
_THU_BOILER = [
    (re.compile(r"^新浪[\u4e00-\u9fff]{0,4}讯\s*"), ""),
    (re.compile(r">>>[^\n。]*[？?]?"), ""),
    (re.compile(r"[（(]责任编辑[：:][^）)]*[）)]"), ""),
    (re.compile(r"【我要评论】|【进入.*?】|点击此处查看全部.*?图片|更多信息请访问[：:]\S*|\[点击.*?\]"), ""),
    # 11-06 复核补：文末评论入口、新浪房产的楼盘链接条、编辑署名、网址和去网址后的空括号
    (re.compile(r"(?:欢迎发表评论|已有_COUNT_条评论)?\s*我要评论"), ""),
    (re.compile(r"[（(]\s*论坛\s+相册[^）)]*[）)]"), ""),
    (re.compile(r"[（(](?:实习)?编辑\s*[\u4e00-\u9fff]{2,4}[）)]"), ""),
    (re.compile(r"(?:https?://|www\.)[\w\-./?=&%#:~+]+"), ""),
    (re.compile(r"[（(]\s*[）)]"), ""),
]
_THU_REVIEW_TITLE = re.compile(r"推荐|导购|评测|测评|试用|开箱|报价|热卖|仅需|仅售|降价|促销|特价|\d{2,6}元")   # 11-06 复核：科技类约 13% 是中关村在线式导购
_THU_LISTING = re.compile(r"在售|待售|开盘|认筹|样板间|户型|均价|总价|起价|售楼")


def clean_thucnews(src_dir: Path, stats: Counter):
    seen = set()
    f = next(src_dir.rglob("THUCNews.jsonl"))
    with open(f) as fh:
        for i, line in enumerate(fh):
            stats["输入"] += 1
            r = json.loads(line)
            cat = r.get("label") or ""
            if cat in _THU_DROP_CAT:
                stats[f"去掉：{cat}类"] += 1
                continue
            t = (r.get("text") or "").replace("\xa0", " ")
            if cat in _THU_PROMO_CAT and len(_THU_PROMO.findall(t)) >= 3:
                stats["去掉：软文（促销/报价词 ≥3）"] += 1
                continue
            if cat in _THU_PROMO_CAT and _THU_REVIEW_TITLE.search(r.get("title") or ""):
                stats["去掉：产品评测/导购（按标题）"] += 1
                continue
            if cat in ("房产", "家居") and len(_THU_LISTING.findall(t)) >= 3:
                stats["去掉：楼盘在售信息（户型/均价/在售等词 ≥3）"] += 1
                continue
            paras = [x.strip() for x in re.split(r"\u3000{2,}|\n+", t) if x.strip()]
            out = []
            for p in paras:
                for rx, rep in _THU_BOILER:
                    p = rx.sub(rep, p)
                p = p.strip()
                if p:
                    out.append(p)
            body = "\n".join(out)
            if len(body) < 200:
                stats["去掉：不到 200 字"] += 1
                continue
            h = hashlib.md5(body.encode()).hexdigest()
            if h in seen:
                stats["去掉：重复"] += 1
                continue
            seen.add(h)
            stats["保留"] += 1
            stats[f"保留：{cat}"] += 1
            yield {"id": f"thucnews:{i}", "category": cat, "title": r.get("title"), "text": body, "n_paras": len(out)}


# ---------------- COIG-CQIA ----------------

_ZHIHU_AID_CUT = 2_780_000_000   # 知乎回答 id 随时间递增：2,778,473,508＝2022-11-28，2,791,447,858＝2022-12-07（用 Zhihu-KOL 带时间的样本标定）
_DASH_FIX = [(re.compile(r"一-(?=[种个样定直些切般边起点下次位件条])"), "一"),
             (re.compile(r"(^|[。！？，；：、\n])-(?=[个种样定直些切般])"), r"\1一")]


def _cqia_minor(r: dict) -> str:
    t = r.get("task_type")
    try:
        t = json.loads(t) if isinstance(t, str) else t
    except ValueError:
        return ""
    return "/".join(t.get("minor") or []) if isinstance(t, dict) else ""


def _code_share(t: str) -> tuple[float, int]:
    code = sum(len(m) for m in re.findall(r"```.*?(?:```|$)", t, re.S))
    prose = re.sub(r"```.*?(?:```|$)", "", t, flags=re.S)
    return code / max(len(t), 1), len(re.sub(r"\s", "", prose))


def clean_cqia(src_dir: Path, stats: Counter):
    seen = set()
    f = next(src_dir.rglob("COIG-CQIA-full.jsonl"))
    with open(f) as fh:
        for i, line in enumerate(fh):
            stats["输入"] += 1
            r = json.loads(line)
            if r.get("answer_from") not in ("human", "expert"):
                stats["去掉：answer_from 不是 human/expert"] += 1
                continue
            m = _cqia_minor(r)
            out = (r.get("output") or "").strip()
            if m == "知乎问答":
                a = re.search(r"aid:(\d+)", str(r.get("metadata")))
                if not a or int(a.group(1)) >= _ZHIHU_AID_CUT:
                    stats["去掉：知乎回答 id 在 2022-11-29 之后（或没有 id）"] += 1
                    continue
                if _PROMO.search(out):
                    stats["去掉：知乎引流话术"] += 1
                    continue
                for rx, rep in _DASH_FIX:
                    out, k = rx.subn(rep, out)
                    stats["改：“一-种”“-个人”残留"] += k
                sub, min_len = "zhihu", 200
            elif m.startswith("故事概要/"):
                sub, min_len = "douban_synopsis", 100
            elif m == "代码问答":
                share, prose = _code_share(out)
                if share > 0.5 or prose < 100:
                    stats["去掉：segmentfault 代码占一半以上或文字不到 100 字"] += 1
                    continue
                sub, min_len = "segmentfault", 0
            else:
                stats["去掉：不要的子集"] += 1
                continue
            if len(out) < min_len:
                stats["去掉：太短"] += 1
                continue
            h = hashlib.md5(out.encode()).hexdigest()
            if h in seen:
                stats["去掉：重复"] += 1
                continue
            seen.add(h)
            stats["保留"] += 1
            stats[f"保留：{sub}"] += 1
            yield {"id": f"cqia:{i}", "subset": sub, "task_type": m, "question": r.get("instruction"), "text": out,
                   "metadata": r.get("metadata") if sub == "zhihu" else None}


# ---------------- MNBVC 外交部例行记者会 ----------------

_MFA_CUT = "20221201"
# 原数据没日期、标题里也没日期时，只认任期在 2022-11 前就结束的发言人（陆慷 2019 年、耿爽 2020 年离任）；
# 华春莹、赵立坚、汪文斌任期跨过 2022-11，补不到就去掉
_MFA_DONE_SPOKES = {"陆慷": "2019", "耿爽": "2020"}
_MFA_TITLE_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
# 问句：末尾 60 字里有问法，或以问号/问法收尾
_MFA_QW = re.compile(r"[？?]|有何评论|有何回应|有何看法|有何评价|有何反应|能否介绍|请介绍|请问|能否证实|请证实|是否属实|中方是否|中方对此|请谈谈")
_MFA_QEND = re.compile(r"(?:[？?]|评论|回应|看法|介绍|证实|评价|立场|反应|吗|呢|如何|怎样|什么|哪些|多少|何时)[。．\s]*$")
# 2008–2017 年的段落分隔被存成了“4”：“……的。4关于你提的第二个问题”。后面跟量词/日期的不动（可能是真数字）
_MFA_SEP4 = re.compile(r"(?<=[。！？；”])\s*4(?=[一-鿿“])(?![月日年个名点时次项亿万人天岁家位条国周架艘颗枚轮届分秒号层期批])")
# 数字被吃掉的痕迹：“去年月”“月日1200”“0月20日”“第届”“年月日”前没数字、百分号前没数字
_MFA_LOSS = re.compile(r"[今去明前]年[月日]|(?<![0-9０-９一二三四五六七八九十〇零两])月日|(?<![0-9０-９])0[月日]|(?<![0-9０-９])%|第届"
                       r"|(?<![0-9０-９一二三四五六七八九十〇零两])年月日|(?:^|[，。、；：“（\s])月[0-9０-９]{1,2}日")


def _mfa_q(t: str) -> bool:
    t = t.strip()
    return bool(_MFA_QW.search(t[-60:]) or _MFA_QEND.search(t[-20:]))


def _mfa_norm(t: str) -> str:
    t = re.sub(r"^\s*答[：:]?", "", t.strip())
    lines = [re.sub(r"(?<=[一-鿿，。！？；：、“”（）])[ 　]+(?=[一-鿿，。！？；：、“”（）])", "", x.strip()) for x in t.split("\n")]
    return "\n".join(x for x in lines if x)


def clean_mfa(src_dir: Path, stats: Counter):
    seen = set()
    f = next(src_dir.rglob("0.jsonl.gz"))
    for line in gzip.open(f, "rt"):
        stats["输入"] += 1
        r = json.loads(line)
        meta = r.get("元数据") or {}
        title = (meta.get("扩展字段") or "").strip().removesuffix("_中华人民共和国外交部") or None
        ct = (meta.get("create_time") or "")[:8]
        date, note = None, None
        if ct and not ct.startswith("0000"):
            date = ct
        elif title and (m := _MFA_TITLE_DATE.search(title)):
            date = f"{int(m[1]):04d}{int(m[2]):02d}{int(m[3]):02d}"
            note = "原数据无日期，按发布会标题补"
        else:
            sp = next((k for k in _MFA_DONE_SPOKES if title and k in title), None)
            if not sp:
                stats["去掉：无日期，标题也补不到"] += 1
                continue
            note = f"原数据无日期；标题里的发言人{sp} {_MFA_DONE_SPOKES[sp]} 年已离任，必在 2022-11 之前"
            stats["补：无日期，按发言人任期判定在截止前"] += 1
        if date and date >= _MFA_CUT:
            stats["去掉：2022-12-01 及以后"] += 1
            continue
        q_raw, a_raw = r.get("问") or "", r.get("答") or ""
        m = re.search(r"[？?。]\s*答[：:]?(?=\S)", q_raw)
        if m and _mfa_q(q_raw[:m.start() + 1]):
            # “问”里自带“问……答……”，这一条按“问”拆，“答”字段（多半是下一题）不用
            question, text, pairing = q_raw[:m.start() + 1], q_raw[m.end():], "inline"
            stats["改：问字段里自带回答，拆开"] += 1
        else:
            qq, aq = _mfa_q(q_raw), _mfa_q(a_raw)
            if qq and not aq:
                question, text, pairing = q_raw, a_raw, "ok"
            elif aq and not qq:
                # 2002–2008 年大量错位：“答”里是本题的问题，“问”里是上一题的回答；重新配对不可靠，只留回答
                question, text, pairing = None, q_raw, "answer_only"
                stats["改：问答错位，只留回答"] += 1
            else:
                stats["去掉：分不清哪个是问题"] += 1
                continue
        text = _mfa_norm(text)
        question = _mfa_norm(question) if question else None
        if question and "追问" in question:  # “问”里前半截是上一题的回答，只留“追问”后面的问题
            question = question.rsplit("追问", 1)[1]
        if question:
            question = question.lstrip("。，；：:、 ") or None
        if "追问" in text or re.search(r"问[：:]|(?<!申)请问", text):
            stats["去掉：回答里夹着记者提问或追问"] += 1
            continue
        text, k = _MFA_SEP4.subn("\n", text)
        stats["改：段落分隔“4”换成换行"] += k
        if _MFA_LOSS.search(text) or (question and _MFA_LOSS.search(question)):
            stats["去掉：有数字被吃掉的痕迹"] += 1
            continue
        if len(text) < 50:
            stats["去掉：回答不到 50 字"] += 1
            continue
        h = hashlib.md5(re.sub(r"\W", "", text).encode()).hexdigest()
        if h in seen:
            stats["去掉：回答重复"] += 1
            continue
        seen.add(h)
        stats["保留"] += 1
        stats[f"保留：{pairing}"] += 1
        yield {"id": f"mfa:{r.get('id')}", "date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else None, "date_note": note,
               "title": title, "question": question, "text": text, "pairing": pairing}


# ---------------- r/WritingPrompts ----------------

# 没有 sacremoses，按这份数据实际出现的分词痕迹写规则（11-06 在 validation 15,620 篇上统计）：
# 正文：“ n't” 71%，`` '' 引号 64%，带空格的弯引号/弯撇号 13–15%，段落是 “ \n \n ”；正文里 's 和标点前空格基本没有。
# 提示：完整分词（It 's、空格 + 标点、`` ''），开头有 [ WP ] 之类的标签
_WP_TAGS = re.compile(r"^\s*((?:\[\s*[A-Za-z]{1,4}\s*\]\s*)+)")
# Reddit 附言段。11-06 第一版不看位置、不看引号，把对白（"Thank you for coming," she said.）也删了：
# 强标记（Edit:、Update:、r/…、Part 2 here）哪里都删；感谢/第一次发帖/求反馈只删首段和末两段；引号开头的一律不删
_WP_META_STRONG = re.compile(r"(?i)^[\s*_#>(\[]*(?:edit\s*\d*\s*(?:[:;\-–—]|$)|update\s*\d*\s*(?:[:;\-–—]|$)|/?r/\w+|part \d+ (?:here|is up)|\[part)")
_WP_META_WEAK = re.compile(r"(?i)^[\s*_#>(\[]*(?:thanks? (?:you )?(?:for|so much|all|everyone)|this (?:is|was) my first|first (?:time|post|submission|attempt|story)"
                           r"|feedback (?:is )?(?:welcome|appreciated)|(?:cc|c&c|criticism|critique)s? (?:is |are )?(?:welcome|appreciated))")
_WP_META_CTX = re.compile(r"(?i)post|writ|submi|\bwp\b|feedback|critique|criticism|typo|format|mobile|reddit|prompt|story|attempt|reading|enjoy")
_WP_URL = re.compile(r"https?://|www\.\w")
_WP_PROFANE = re.compile(r"(?i)\b(?:fuck\w*|shit\w*|cunt\w*|bitch\w*|motherfuck\w*|asshole\w*|dick\w*|cock\w*|pussy\w*|faggot\w*|nigg\w*)\b")


def _wp_detok(t: str, full: bool) -> tuple[str, Counter]:
    """full=True 用于提示（完整分词）；正文只修实际出现的几类。返回 (文本, 各规则命中数)。"""
    k: Counter = Counter()

    def sub(name, rx, rep, s, flags=0):
        s2, n = re.subn(rx, rep, s, flags=flags)
        k[name] += n
        return s2
    t = t.replace("\r", "")
    t = sub("段落", r"[ \t]*\n[ \t]*\n[ \t\n]*", "\n\n", t)
    t = sub("段落", r"[ \t]*\n[ \t]*", "\n", t)
    t = sub("n't", r"(?<=\w) n't\b", "n't", t)
    t = sub("``''", r"``\s*", '"', t)
    t = sub("``''", r"\s*''", '"', t)
    t = sub("``''", r'(?<=[^\s"])""(?=[A-Za-z])', '" "', t)  # 原文 ''`` 紧挨着（上一句收引号、下一句开引号）
    t = sub("弯引号", r"“[ \t]+", "“", t)
    t = sub("弯引号", r"[ \t]+”", "”", t)
    t = sub("弯撇号", r"(?<=\w) ’ (?=(?:s|t|re|ve|ll|d|m)\b)", "’", t, re.I)
    t = sub("弯撇号", r"\b(ma) ’ (am)\b", r"\1’\2", t, re.I)
    t = sub("弯引号", r"‘[ \t]+(?=\w)", "‘", t)
    t = sub("弯引号", r"(?<=[\w.,!?])[ \t]+’(?=[\s.,!?;:]|$)", "’", t)
    if full:
        t = sub("'s 等", r"(?<=\w) '(s|re|ve|ll|d|m)\b", r"'\1", t, re.I)
        t = sub("标点前空格", r"[ \t]+([.,!?;:%)\]}])", r"\1", t)
        t = sub("括号后空格", r"([(\[{$])[ \t]+", r"\1", t)
    return t.strip(), k


def clean_writingprompts(src_dir: Path, stats: Counter):
    import pyarrow.parquet as pq
    per_prompt: Counter = Counter()
    seen = set()
    files = sorted(src_dir.rglob("*.parquet"), key=lambda p: ("train" not in p.name, p.name))
    for f in files:
        split = f.name.split("-")[0]
        i = -1
        for batch in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=["prompt", "story"]):
            for r in batch.to_pylist():
                i += 1
                stats["输入"] += 1
                p_raw, s_raw = r.get("prompt") or "", r.get("story") or ""
                m = _WP_TAGS.match(p_raw)
                tags = re.findall(r"[A-Za-z]+", m.group(1).upper()) if m else []
                prompt, _ = _wp_detok(p_raw[m.end():] if m else p_raw, True)
                story, k = _wp_detok(s_raw, False)
                for name, n in k.items():
                    stats[f"改：{name}"] += n
                if _WP_URL.search(story):
                    stats["去掉：含网址"] += 1
                    continue
                paras = story.split("\n\n")
                kept = [x for j, x in enumerate(paras) if not (len(x.split()) < 60 and (_WP_META_STRONG.match(x) or (
                    _WP_META_WEAK.match(x) and _WP_META_CTX.search(x) and (j == 0 or j >= len(paras) - 2))))]
                if len(kept) < len(paras):
                    stats["改：删掉 Reddit 附言段（edit/感谢/求反馈/第几部分）"] += len(paras) - len(kept)
                story = "\n\n".join(kept).strip()
                words = len(story.split())
                if words < 300:
                    stats["去掉：不到 300 词"] += 1
                    continue
                if len(_WP_PROFANE.findall(story)) / words > 0.02:
                    stats["去掉：脏话超过 2%"] += 1
                    continue
                if re.search(r"([A-Za-z!?.])\1{9,}", story) or len(set(story.split())) / words < 0.2:
                    stats["去掉：同一字符连写 10 次以上或用词重复太多"] += 1
                    continue
                pk = re.sub(r"\W+", " ", prompt.lower()).strip()
                if per_prompt[pk] >= 2:
                    stats["去掉：同一提示已留 2 篇"] += 1
                    continue
                h = hashlib.md5(re.sub(r"\W+", "", story.lower()).encode()).hexdigest()
                if h in seen:
                    stats["去掉：重复"] += 1
                    continue
                seen.add(h)
                per_prompt[pk] += 1
                stats["保留"] += 1
                yield {"id": f"wp:{split}:{i}", "split": split, "tags": tags, "prompt": prompt, "text": story, "words": words}


# ---------------- MNBVC 上市公司年报 ----------------
# 一条记录＝一份年报全文：text 是第二层 JSON（段落数组，每段 {行号, md5, 内容}），meta 也是 JSON 字符串
# （文件名 "2020/300515_2020.txt"＝年份/股票代码_年份）。评估见 reviews.json：绝大部分是表格、会计政策模板和
# 法定声明，只抽“经营情况讨论与分析/管理层讨论与分析”和“董事长致辞/致股东信”。

_ANN_SEC = re.compile(r"^(第[一二三四五六七八九十]{1,3}[节章])\s*(.{0,30})$")   # 章节标题；有的版式标题在下一段，只剩“第四节”
_ANN_BARE = re.compile(r"^第[一二三四五六七八九十]{1,3}[节章]$")
_ANN_MD = re.compile(r"经营情况讨论与分析|管理层讨论与分析|经营情况分析与讨论|经营讨论与分析|经营情况和管理层分析")
_ANN_GREET = re.compile(r"^(?:(?:董事长)?(?:致|给)(?:全体)?股东(?:的)?(?:一封信|信)?|董事长致辞|致股东信|致股东"
                        r"|——?\s*20\d\d\s*年致股东的信)$")
# 致辞后面紧跟这些就收尾：目录、重要提示、别的致辞、下一个章节标题
_ANN_STOP_GREET = re.compile(r"^(?:目\s*录|重要提示|行长致辞|监事会致辞|致股东|董事长致辞|第[一二三四五六七八九十]{1,3}[节章])")
_ANN_PAGENUM = re.compile(r"^(?:\d{1,3}\s*/\s*\d{1,3}|[-—]?\s*\d{1,3}\s*[-—]?|第\s*\d+\s*页(?:\s*共\s*\d+\s*页)?)$")
_ANN_WM = re.compile(r"taobao|hellohello|请勿下载|提取版本日期|淘汰版")   # 淘宝卖家的提取水印（每份第一行）
_ANN_BOX = re.compile(r"[□√✓■☑]\s*(?:适用|不适用|是|否)")
# “接待调研、沟通、采访等活动情况”整节是登记表，抽出来是一行行日期/机构，正文里不会出现这些说法
_ANN_TABLE_ROW = re.compile(r"实地调研|接待时间|接待地点|接待对象|谈论的主要内容|投资者关系活动记录表|调研的基本情况索引|公司一楼会议室")
# 压平的登记信息/表格：“企业类型：…注册地址：…法定代表人：…注册资本：…”这种一段里 ≥3 个字段标签的
_ANN_FIELD = re.compile(r"(?:企业类型|公司类型|企业性质|公司名称|中文名称|英文名称|注册地址|办公地址|公司注册地址|公司办公地址|"
                        r"法定代表人|注册资本|实收资本|成立日期|营业期限|经营范围|统一社会信用代码|注册号|组织机构代码|"
                        r"股东构成|股东名称|股东总数|持股比例|持股数量|主营业务|所属行业|上市地点|股票代码|股票简称|"
                        r"联系电话|传真|电子信箱|邮政编码|实际控制人|控股股东)\s*[：:]")
_CJK = re.compile(r"[一-鿿]")
_ANN_NUMTOK = re.compile(r"\d[\d,，.]*\s*%?")


def _ann_company(paras: list[str]) -> str | None:
    """公司名：Title 里的全称 → “公司简称：” → 封面“某某股份有限公司 20xx 年年度报告” → 单独一行的全称。
    有 1% 左右拿不到（老三板公司、封面被水印挤掉），就留空。"""
    title = next((s for s in paras[:5] if s.startswith("Title:")), "")
    if title:
        m = re.match(r"(.{4,30}?(?:股份有限公司|有限责任公司|集团有限公司))", title[6:].strip())
        if m:
            return m.group(1)
    for s in paras[:60]:
        m = re.search(r"公司简称[：:]\s*([^\s，。；]{2,10})", s)
        if m:
            return m.group(1)
    for s in paras[:60]:
        m = re.match(r"^(.{4,30}?(?:股份有限公司|有限责任公司|集团有限公司))\s*(?:19|20)\d\d\s*年年度报告", s)
        if m:
            return m.group(1)
    for s in paras[:60]:
        m = re.match(r"^(.{4,30}?(?:股份有限公司|有限责任公司|集团有限公司))$", s)
        if m:
            return m.group(1)
    return None


def _ann_strip_furn(s: str, company: str | None) -> str:
    """PDF 页眉/页码经常和正文粘在同一段开头（“32成都高新发展股份有限公司 2020 年年度报告全文 据国家统计局…”），
    整段删会丢正文，只能把前缀剥掉。"""
    s = s.lstrip("﻿").strip()
    changed = True
    while changed and s:
        changed = False
        for rx in (r"^\d{1,3}\s*/\s*\d{1,3}\s*", r"^第\s*\d+\s*页\s*(?:/|共)?\s*\d{0,3}\s*页?\s*"):
            s2 = re.sub(rx, "", s)
            if s2 != s:
                s, changed = s2.strip(), True
        if company and s.startswith(company):
            s, changed = s[len(company):].strip(), True
        s2 = re.sub(r"^\d{1,3}(?=[一-鿿]{2,}.{0,20}(?:股份有限公司|有限责任公司|年度报告))", "", s)
        if s2 != s:
            s, changed = s2, True
        s2 = re.sub(r"^\d{1,3}(?=[一-鿿])", "", s)   # 页码和正文粘连：“15随着机动车检测市场…”
        if s2 != s:
            s, changed = s2, True
        s2 = re.sub(r"^.{0,30}?(?:股份有限公司|有限责任公司|集团有限公司)\s*(?=(?:19|20)\d\d\s*年年度报告)", "", s)
        if s2 != s:
            s, changed = s2, True
        s2 = re.sub(r"^(?:19|20)\d\d\s*年?\s*年度报告(?:全文|（A\s*股）|\(A\s*股\))?\s*", "", s)
        if s2 != s:
            s, changed = s2, True
        # 页眉整行：“中国神华 2020 年度报告”（公司名 + 年份 + 年度报告，中间没有“年”）
        s2 = re.sub(r"^.{0,30}?(?:19|20)\d\d\s*年?\s*年度报告(?:全文|（A\s*股）|\(A\s*股\))?\s*$", "", s)
        if s2 != s:
            s, changed = s2.strip(), True
    return s.strip()


_ANN_MARK_S = (r"(?:第[一二三四五六七八九十]{1,3}[章节]|[（(]\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[)）]"
               r"|\d{1,2}\s*[、.．)）](?!\d)|[一二三四五六七八九十]{1,3}、)")
_ANN_MARK = re.compile(_ANN_MARK_S)
# 句末标点后紧跟小标题：把那个空白换成换行，先切成一行一个单元
_ANN_EMBED = re.compile(r"(?<=[。！？；：”\"])\s*(?=" + _ANN_MARK_S + r")")
# 断开处开头的连接词：说明这一段是半句的后半截
_ANN_HALF = re.compile(r"^(?:的|和|及|与|或|等|但|而|并(?![购表入行举])|同时|且|即|则|因此|此外|另|其中|从而|进而|以及)")
_ANN_END = tuple("。！？；：\"”")
# 表格拆出来以后从数字/单位开头的半句：“42.67%，主要原因为…”“亿元，主要原因是…”
_ANN_FRAG = re.compile(r"^[\d.,]+\s*(?:%|万元|亿元|元|吨|台|套|个|家|人|项|根|只|笔)?\s*[，；]"
                       r"|^(?:亿元|万元|元|万股|股|吨|台|套|个|家|人|项|根|只|笔)[，；]")
# 标题和正文之间没有空格时，在这些词处切
_ANN_BODY_OPEN = re.compile(r"(?:报告期内|本报告期|报告期|上半年|下半年|一季度|二季度|三季度|四季度|全年|本期|20\d\d\s*年)")


def _ann_title_end(s: str, start: int) -> int | None:
    """从 start 开始的小标题，返回正文起点；判断不出是标题就返回 None（当正文处理）。
    这些 PDF 的行内换行都被抽成了空格，所以小标题通常是一个空格隔开的短串；串里带句号/问号、
    以逗号顿号结尾、或长过 30 字的，多半是正文里的编号句，不算标题。"""
    while start < len(s) and s[start] == " ":
        start += 1
    sp = s.find(" ", start, start + 36)
    if sp > start:
        run = s[start:sp]
        if len(run) > 30 or run[-1] in "，、；:" or re.search(r"[。？！]", run):
            return None
        return sp + 1
    m = _ANN_BODY_OPEN.search(s, start, start + 30)
    if m and m.start() > start:
        return m.start()
    # “一、概述”“5、募集资金使用情况”这种整行就是标题、后面没有正文的
    rest = s[start:]
    if len(rest) <= 20 and not re.search(r"[。，；！？、：]", rest):
        return len(s)
    return None


def _ann_split_titles(line: str) -> tuple[str, int]:
    """把行首的小标题剥掉（可能有连续几个，如“2、收入与成本 （1）营业收入构成”），返回 (正文, 标题数)。"""
    n = 0
    while True:
        line = line.strip()
        # 章节名单独成段（“经营情况讨论与分析”），或和正文粘着（“…与分析2020 年，公司…”）
        m = _ANN_MD.match(line)
        if m:
            rest = line[m.end():].lstrip()
            if not rest or re.match(r"[\d本报告期公司我上]", rest):
                n += 1
                line = rest
                if not line:
                    return "", n
                continue
        m = _ANN_MARK.match(line)
        if not m:
            return line, n
        end = _ann_title_end(line, m.end())
        if end is None:
            return line, n
        n += 1
        line = line[end:]
        if not line.strip():
            return "", n


def _ann_pieces(s: str, company: str | None, stats: Counter, samples: dict):
    """一个原始 PDF 段 → 事件流。('furn', None) 是页眉页码这类版式垃圾，不打断跨页合并；
    ('drop', None) 是删掉的正文/表格/小标题，跨页合并到此为止；('text', s) 是一个自然段候选。"""
    s = _ann_strip_furn(s, company)
    if not s:
        stats["行：删页眉/页码（剥完为空）"] += 1
        yield ("furn", None)
        return
    if _ANN_WM.search(s):
        stats["行：删水印"] += 1
        samples.setdefault("水印", []).append(s[:100])
        yield ("furn", None)
        return
    if _ANN_PAGENUM.match(s):
        stats["行：删页码"] += 1
        samples.setdefault("页码", []).append(s[:40])
        yield ("furn", None)
        return
    s = _ANN_BOX.sub("", s).strip()
    s = re.sub(r"^[、，。:：\s]+", "", s)
    if not s:
        stats["行：删勾选框/表格符号行"] += 1
        yield ("drop", None)
        return
    if _ANN_TABLE_ROW.search(s):
        stats["行：删接待调研登记表行"] += 1
        samples.setdefault("接待调研表", []).append(s[:110])
        yield ("drop", None)
        return
    if len({m.group(0).rstrip("：:") for m in _ANN_FIELD.finditer(s)}) >= 3:
        stats["行：删字段标签 ≥3 的登记信息段"] += 1
        samples.setdefault("字段标签段", []).append(s[:130])
        yield ("drop", None)
        return
    n = len(s)
    dig = sum(c.isdigit() for c in s) / n
    cj = len(_CJK.findall(s)) / n
    if n <= 25 and (dig + sum(not c.isalnum() and not _CJK.match(c) for c in s) / n) >= 0.5 and not _ANN_MARK.match(s):
        stats["行：删数字碎行"] += 1
        samples.setdefault("数字碎行", []).append(s[:80])
        yield ("drop", None)
        return
    if dig >= 0.15 and not (n < 20 and _ANN_MARK.match(s)):
        stats["行：删数字占比高（≥15%）"] += 1
        samples.setdefault("数字占比高", []).append(s[:110])
        yield ("drop", None)
        return
    if len(_ANN_NUMTOK.findall(s)) >= 5 and cj < 0.5:
        stats["行：删数字密集（≥5 个数字且汉字<50%）"] += 1
        samples.setdefault("数字密集", []).append(s[:110])
        yield ("drop", None)
        return
    if n >= 15 and not re.search(r"[。，；！？]", s) and not _ANN_MARK.match(s):
        stats["行：删无标点表格行"] += 1
        samples.setdefault("无标点表格行", []).append(s[:110])
        yield ("drop", None)
        return
    # 小标题和正文之间靠 PDF 行尾的空格分开，得先切标题再抹汉字间空格
    for line in _ANN_EMBED.sub("\n", s).split("\n"):
        line = line.strip()
        if not line:
            continue
        body, ntitles = _ann_split_titles(line)
        if ntitles:
            stats["段：删小标题（当段落边界）"] += ntitles
            samples.setdefault("小标题", []).append(line[:80])
            yield ("drop", None)
        if not body:
            continue
        if _ANN_PAGENUM.match(body):   # 标题行尾跟着的页码
            stats["行：删页码"] += 1
            yield ("furn", None)
            continue
        body = re.sub(r"(?<=[一-鿿，。、；：、“”‘’（）《》])\s+(?=[一-鿿，。、；：、“”‘’（）《》])", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < 4:
            stats["行：删清理后太短"] += 1
            yield ("drop", None)
            continue
        yield ("text", body)


def _ann_assemble(events, stats: Counter) -> list[tuple[str, int]]:
    """把事件流拼成自然段：只在“上一段没句末标点 + 中间没删正文”时接续；接完再丢半句/断尾。"""
    cand: list[list] = []   # [text, 合并次数]
    gap = False
    for kind, s in events:
        if kind == "furn":
            continue
        if kind == "drop":
            gap = True
            continue
        if cand and not gap and not cand[-1][0].endswith(_ANN_END) and not _ANN_MARK.match(s):
            cand[-1][0] += s
            cand[-1][1] += 1
            stats["段：合并跨页断行"] += 1
        else:
            cand.append([s, 0])
        gap = False
    out, prev_broken = [], False
    for text, merges in cand:
        if len(text) < 10:
            stats["段：删不到 10 字的碎片"] += 1
            continue
        if prev_broken:
            stats["段：删断开的半句段（上一段没句末标点）"] += 1
            prev_broken = False
            continue
        if _ANN_HALF.match(text):
            stats["段：删半句开头（的/和/及/同时…）"] += 1
            continue
        if re.match(r"^[a-z]", text):
            stats["段：删英文残片开头"] += 1
            continue
        if _ANN_FRAG.match(text):
            stats["段：删数字/单位开头的半句"] += 1
            continue
        if text.endswith("；") and len(text) < 20:
            stats["段：删短分号碎片"] += 1
            continue
        if not text.endswith(_ANN_END):
            if len(text) < 50:
                stats["段：删无句末标点的短段（<50 字）"] += 1
            else:
                stats["段：删段尾无句末标点（截断）"] += 1
                prev_broken = True
            continue
        out.append((text, merges))
    return out


def _ann_sections(paras: list[str]) -> list[tuple[int, int, str]]:
    """找出目标章节 (起, 止, section)。止＝下一个同级章节标题，不会带进“重要事项”“公司治理”。"""
    heads = [j for j, s in enumerate(paras) if len(s) < 45 and _ANN_SEC.match(s) and "...." not in s and "。。。。" not in s]
    cands = []
    for i, j in enumerate(heads):
        end = heads[i + 1] if i + 1 < len(heads) else len(paras)
        title = _ANN_SEC.match(paras[j]).group(2)
        if _ANN_MD.search(title):
            cands.append((j, end, "md"))
        elif _ANN_BARE.match(paras[j]):
            # 有的版式“第四节”单独一段，标题和内容在随后几段里（标题甚至在正文之后）
            for x in paras[j + 1:j + 7]:
                if x and len(x) < 40 and "。" not in x and _ANN_MD.search(x) and not re.match(r"^(?:参见|详见|见第|根据|本报告)", x):
                    cands.append((j, end, "md"))
                    break
    # 目录页/页眉会重复出现同名标题（600036 每页页眉都是“第三章 经营情况讨论与分析”），取跨度最大的那个
    cands.sort(key=lambda x: x[1] - x[0], reverse=True)
    return cands[:1]


def clean_co_ann_report(src_dir: Path, stats: Counter):
    samples: dict = {}
    reports = []
    for f in sorted(src_dir.rglob("*.jsonl.gz"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        for line in gzip.open(f, "rt"):
            stats["输入"] += 1
            r = json.loads(line)
            try:
                meta = json.loads(r.get("meta") or "{}")
                raw = json.loads(r.get("text") or "[]")
            except json.JSONDecodeError:
                stats["去掉：两层 JSON 坏了"] += 1
                continue
            paras = []
            for x in raw:
                c = (x.get("内容") if isinstance(x, dict) else x) or ""
                c = c.strip()
                if c:
                    paras.append(c)
            if not paras:
                stats["去掉：没有正文"] += 1
                continue
            fn = meta.get("文件名") or ""
            m = re.search(r"(\d{6})_(\d{4})", fn)
            code, year = (m.group(1), m.group(2)) if m else (None, None)
            company = _ann_company(paras)
            if not year:
                m = re.search(r"(20\d\d)\s*年年度报告", " ".join(paras[:30]))
                year = m.group(1) if m else None
            secs = _ann_sections(paras)
            if not secs:
                stats["去掉：没有目标章节"] += 1
                continue
            a, b, _ = secs[0]
            md = paras[a:b]
            # 致辞标题在页眉里会重复出现（“董事长致辞”每页一行），连续几段里的取最后一个当真标题；
            # 正文里再遇到同名标题是页眉，不当收尾
            gj = [j for j, s in enumerate(paras[:a]) if len(s) < 30 and _ANN_GREET.match(s)]
            run = []
            for j in gj:
                if run and j - run[-1] <= 4:
                    run[-1] = j
                else:
                    run.append(j)
            greet = []
            if run:
                g0 = run[0]
                head = paras[g0]
                g1 = next((k for k in range(g0 + 1, min(len(paras), g0 + 400))
                           if paras[k] != head and _ANN_STOP_GREET.match(paras[k])), min(len(paras), g0 + 400))
                greet = paras[g0 + 1:g1]
            reports.append({"file": fn or f.name, "shard": f.name, "code": code, "year": year, "company": company,
                            "md": md, "greet": greet, "md_title": _ANN_MD.search(_ANN_SEC.match(paras[a]).group(2))})

    # 先按段 md5 统计跨公司出现次数：模板化的会计政策、风险提示在几十家公司里逐字相同
    seen_in: dict[str, set] = defaultdict(set)
    for i, rep in enumerate(reports):
        key = rep["code"] or f"#{i}"
        for s in rep["md"] + rep["greet"]:
            for kind, c in _ann_pieces(s, rep["company"], Counter(), {}):
                if kind == "text" and len(c) >= 20:
                    seen_in[hashlib.md5(c.encode()).hexdigest()].add(key)

    dup_cut = {h for h, keys in seen_in.items() if len(keys) >= 3}
    for rep in reports:
        secs = []
        if rep["greet"]:
            secs.append(("董事长致辞", rep["greet"]))
        if rep["md"]:
            title = "管理层讨论与分析" if "管理层讨论" in (rep["md_title"].group(0) if rep["md_title"] else "") else "经营情况讨论与分析"
            secs.append((title, rep["md"]))
        made = []
        for name, raw in secs:
            seen, events = set(), []
            for s in raw:
                for kind, c in _ann_pieces(s, rep["company"], stats, samples):
                    if kind == "text":
                        h = hashlib.md5(c.encode()).hexdigest()
                        if len(c) >= 20 and h in dup_cut:
                            stats["段：跨公司模板去重（≥3 家）"] += 1
                            samples.setdefault("跨公司去重", []).append(c[:110])
                            events.append(("drop", None))
                            continue
                        if h in seen:
                            stats["段：同一报告内重复"] += 1
                            events.append(("drop", None))
                            continue
                        seen.add(h)
                    events.append((kind, c))
            out = _ann_assemble(events, stats)
            text = "\n".join(t for t, _ in out)
            if len(text) < 300:
                stats["去掉：这一章清洗后不到 300 字"] += 1
                continue
            made.append((name, text, len(out)))
        total = sum(len(t) for _, t, _ in made)
        if total < 300:
            stats["去掉：抽出的正文不到 300 字"] += 1
            continue
        for name, text, nparas in made:
            stats["保留"] += 1
            stats[f"保留：{name}"] += 1
            stats["保留字数"] += len(text)
            yield {"id": f"co_ann:{rep['code'] or rep['file']}:{rep['year']}:"
                         f"{'greet' if name == '董事长致辞' else 'md'}",
                   "company": rep["company"], "year": rep["year"], "section": name,
                   "text": text, "source_file": rep["file"], "n_paras": nparas}

    if samples:   # 被删样本给 dry 复核用，放 work 里，正式跑不带走
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "zh.mnbvc_co_ann_report.deleted_samples.json").write_text(
            json.dumps({k: v[:5] for k, v in samples.items()}, ensure_ascii=False, indent=1))


# ---------------- 登记 ----------------

CLEANERS = {
    "en.writingprompts": {
        "fn": clean_writingprompts, "include": ["data/*.parquet"],
        "title": "r/WritingPrompts stories (detokenized, cleaned)",
        "time_range": "2018 及以前（Fan et al. 2018 发布）",
        "steps": ["还原分词痕迹（没有 sacremoses，按实际出现的写规则）：“ n't”→“n't”，`` 和 '' 换回双引号，弯引号/弯撇号两侧多余空格，“ \\n \\n ”段落；提示另外修 's、标点前空格，去掉 [ WP ] 等标签放进 tags",
                  "去掉含网址的；删掉 60 词以内的 Reddit 附言段：Edit:/Update:/r/…/Part N here 哪里都删，感谢、第一次发帖、求反馈只删首段和末两段；引号开头的（对白）不删",
                  "去掉不到 300 词的；脏话占词数 2% 以上的；同一字符连写 10 次以上或不同词占比 <20% 的",
                  "每个提示最多留 2 篇（按文件顺序先到先留，train 在前）；按正文去重"],
    },
    "zh.mnbvc_qa_mfa": {
        "fn": clean_mfa, "include": ["qa/20230196/mfa/*"],
        "title": "外交部例行记者会问答（清洗版）",
        "time_range": "2001 – 2022-11-30（按 create_time 截断）",
        "steps": ["按 create_time 截到 2022-12-01 之前；日期为 00000000 的用扩展字段里的发布会标题补日期，标题里没日期时只认任期在 2022-11 前结束的发言人（陆慷、耿爽），其余去掉",
                  "“问”里自带“……答……”的拆成问和答",
                  "问答错位（2002–2008 年约一半：“答”里是本题问题、“问”里是上一题回答）：重新配对不可靠，只留回答，question 为空、pairing=answer_only；问句判断不清的整条去掉",
                  "去掉开头的“答：”，去掉汉字间多余空格；2008–2017 年当段落分隔用的“4”换成换行（后面跟量词/日期的不动）",
                  "回答里夹着记者提问或追问（“追问”“问：”“请问”）的去掉", "有数字被吃掉痕迹的去掉（“去年月”“月日1200”“0月20日”“第届”、百分号前没数字）",
                  "回答不到 50 字的去掉；按回答（去标点空白后）去重"],
    },
    "zh.cqia": {
        "fn": clean_cqia, "include": ["COIG-CQIA-full.jsonl"],
        "title": "COIG-CQIA 真人社区子集（清洗版：知乎、豆瓣剧情简介、segmentfault）",
        "time_range": "知乎按回答 id 截到 2022-11-29 前；豆瓣简介、segmentfault 无日期",
        "steps": ["只留 answer_from 为 human/expert 的", "只留三个子集：知乎问答、豆瓣故事概要（剧情简介）、segmentfault 代码问答；"
                  "试题、百科/概念解析、医药问答、wikihow 译文、小红书风格、图书/作者介绍、人类价值观问答、NLP 任务、豆瓣影评/剧评/书评（营销号写法、没有日期）都不要",
                  "知乎按回答 id < 2,780,000,000（≈2022-11-29，用 Zhihu-KOL 标定）截断；去掉含引流话术的", "修“一-种”“-个人”这类抠图残留（只修“一”被识别成“-”的写法）",
                  "segmentfault 去掉代码占一半以上或代码外文字不到 100 字的", "知乎不到 200 字、简介不到 100 字的去掉；按正文去重"],
    },
    "zh.csl": {
        "fn": clean_csl, "include": ["data/csl.jsonl.gz"],
        "title": "CSL 中文核心期刊摘要（清洗版）", "time_range": "2010–2020",
        "steps": ["只用 csl.jsonl.gz（中文）；csl.gt 英文机翻版不用", "去掉不到 80 字的", "去掉公式/化学式标记 ≥3 处或拉丁字母占比 >30% 的",
                  "挨着汉字（或引号、括号后）的半角 , ; : . 转全角，修残缺括号“(TG-DTA(表明”", "按摘要去重"],
    },
    "zh.thucnews": {
        "fn": clean_thucnews, "include": ["THUCNews.jsonl"],
        "title": "THUCNews 新浪新闻（清洗版）", "time_range": "2005–2011",
        "steps": ["去掉股票、彩票、星座三类", "科技/家居/房产/时尚/游戏类里促销、报价、电话等词 ≥3 处的当软文去掉",
                  "房产/家居类里户型、均价、在售等词 ≥3 处的当楼盘在售信息去掉",
                  "科技/家居/房产/时尚/游戏类标题带推荐、导购、评测、开箱、报价、“××元”等的当产品评测/导购去掉",
                  "按全角空格恢复分段", "删掉“新浪×讯”、“>>>您如何评价”、责任编辑、“欢迎发表评论 我要评论”、“(论坛 相册 户型…)”、“(实习编辑 某某)”等固定格式，去掉网址",
                  "去掉不到 200 字的", "按正文去重"],
    },
    "zh.mnbvc_co_ann_report": {
        "fn": clean_co_ann_report,
        "include": ["co_ann_report/20230125/1.jsonl.gz", "co_ann_report/20230125/2.jsonl.gz",
                    "co_ann_report/20230125/3.jsonl.gz", "co_ann_report/20230125/4.jsonl.gz",
                    "co_ann_report/20230125/5.jsonl.gz", "co_ann_report/20230125/6.jsonl.gz",
                    "co_ann_report/20230125/7.jsonl.gz", "co_ann_report/20230125/8.jsonl.gz"],
        "title": "MNBVC 上市公司年报·经营讨论与致辞（清洗版）",
        "time_range": "2020–2021 报告期（MNBVC 2023-01 收集）",
        "steps": ["两层 JSON：meta 和 text（段落数组，每段 {行号, md5, 内容}）都解析；一条记录＝一份年报全文，文件名 “年份/股票代码_年份.txt” 取代码和年份",
                  "只抽两章：“经营情况讨论与分析/管理层讨论与分析”（含“经营情况分析与讨论”“经营讨论与分析”等写法；"
                  "有的版式“第四节”单独一段、标题在随后几段里，按后 6 段找标题）和“董事长致辞/致股东信”；找不到的整份跳过",
                  "章节止于下一个同级章节标题（如“重要事项”“公司治理”“股份变动及股东情况”），不会带进后面的章节；"
                  "目录页和页眉里重复出现的同名标题按“跨度最大”取真章节，避免一页就截断",
                  "删淘宝水印行（hellohellohello.taobao.com、提取版本日期、请勿下载）和页码行（“70 / 161”、单独的数字、“第 N 页”）；"
                  "PDF 页眉常和正文粘在同一段开头（“32某某股份有限公司 2020 年年度报告全文 据…”），剥掉前缀而不是整段删",
                  "表格残留：勾选框（□适用 √不适用）按 token 去掉；≤25 字且数字+符号 ≥50% 的碎行、数字字符占比 ≥15% 的段落（短编号标题豁免）、"
                  "≥5 个数字且汉字 <50% 的段落、≥15 字且没有任何句读（。，；！？）的表格行、"
                  "“接待调研/沟通/采访”登记表行、一段里 ≥3 个“企业类型：/注册地址：/法定代表人：”式字段标签的登记信息段，都去掉",
                  "小标题（第 X 节、一、、（一）、1、、(1)、1. 开头且不带句末标点的短行，以及单独的章节名）一律当段落边界并删掉；"
                  "标题和正文粘在一行（“（一）经营业绩情况 报告期内，…”）的先切掉标题、正文留下",
                  "汉字之间的空格（“减值准 备”“实 际利率”）去掉；PDF 断行只在“上一行不以句末标点结尾、中间没有删过正文、下一行不是标题”时接回一段"
                  "（页码/页眉属版式，不算删正文，跨页可以接上）",
                  "断开后的半句去掉：开头是“的/和/及/与/或/等/但/而/并/同时/因此…”的、上一段没句末标点就被断开的、段尾没有句末标点的整段；"
                  "没有句末标点且 <50 字的、<10 字的碎片、数字/单位开头的半句、短“；”碎片、英文残片开头也去掉",
                  "按段落 md5 跨公司去重：同一段在 ≥3 家公司的年报里出现就整段去掉（模板化会计政策、风险提示）",
                  "一章清洗后不到 300 字、或一份年报两章合计不到 300 字的跳过；每条记录是一份年报的一章（最多两章），id 带章节后缀"],
    },
    "zh.zhihu_kol": {
        "fn": clean_zhihu, "include": ["*.parquet"],
        "title": "知乎 KOL 回答（清洗版）",
        "time_range": "2011 – 2022-11-29（按回答时间截断）",
        "steps": ["按 answer_creation_time 截到 2022-11-30 之前", "赞同 ≥ 100", "按回答 id 和正文去重",
                  "整条去掉含引流话术的回答（关注我、私信我、公众号：、加微信、一键三连等）",
                  "按句末标点和汉字间空白恢复分段（英文单词间、中英文间空格保留）",
                  "删掉明确的图片出处/说明行（“图源”“图片来自网络”“图1：”等）；没有标点的短行保留（多是正文或小标题）",
                  "去掉网址和去网址后留下的空括号；分隔线（5 个以上 —-=_*~）换成换行；双空格后跟序号处还原换行", "删掉编辑时间/版权声明/谢邀行，去掉开头的“谢邀”",
                  "清洗后不到 300 字的去掉"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--keep-work", action="store_true", help="不删本地下载的原文")
    ap.add_argument("--dry", action="store_true", help="只清洗、出报告和样本，不上传不登记")
    a = ap.parse_args()
    spec = CLEANERS[a.id]
    src = next(s for s in collect.load_sources() if s["id"] == a.id)
    new_id = a.id + ".clean"
    raw_dir = WORK / a.id
    out_dir = WORK / new_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    inc = [x for p in spec["include"] for x in ("--include", p)]
    collect.log(f"清洗 {a.id}：下载原文到 {raw_dir}")
    r = collect.rclone("copy", collect.remote_dir(src), str(raw_dir), *inc, "--transfers", "4")
    if r.returncode != 0:
        raise SystemExit(f"下载失败：{r.stderr[-300:]}")

    stats: Counter = Counter()
    rnd = random.Random(0)
    sample: list[dict] = []
    shards: list[Path] = []
    fh, n_in_shard, n = None, 0, 0
    for rec in spec["fn"](raw_dir, stats):
        if fh is None or n_in_shard >= SHARD_ROWS:
            if fh:
                fh.close()
            shards.append(out_dir / f"{new_id.split('.', 1)[1]}-{len(shards):03d}.jsonl.gz")
            # mtime=0：同样的内容两次跑出字节一致的 gz（正式跑和 dry 的分片要能对上 md5）
            fh, n_in_shard = io.TextIOWrapper(gzip.GzipFile(str(shards[-1]), "wb", mtime=0), encoding="utf-8"), 0
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_in_shard += 1
        n += 1
        if len(sample) < 300:
            sample.append(rec)
        elif (j := rnd.randrange(n)) < 300:
            sample[j] = rec
    if fh:
        fh.close()

    REPORTS.mkdir(exist_ok=True)
    report = {"id": new_id, "derived_from": a.id, "made_at": datetime.now().isoformat(timespec="seconds"),
              "steps": spec["steps"], "stats": dict(stats), "rows": n, "files": [p.name for p in shards]}
    (REPORTS / f"{new_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (C / "samples" / f"{new_id}.jsonl").write_text("".join(json.dumps(collect._trim(x), ensure_ascii=False) + "\n" for x in sample))
    collect.log(f"清洗 {a.id}：保留 {n} 条；" + "，".join(f"{k} {v}" for k, v in stats.most_common()))
    if a.dry:
        return

    new_src = {"id": new_id, "lang": src["lang"], "title": spec["title"], "method": "derived", "derived_from": a.id,
               "license": src.get("license"), "redistributable": src.get("redistributable"), "time_range": spec["time_range"],
               "ai_risk": src.get("ai_risk"), "genres": src.get("genres"), "priority": src.get("priority", 9),
               "cleaning_steps": spec["steps"]}
    rdir = collect.remote_dir(new_src)
    for p in shards:
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        collect.upload(p, f"{rdir}/{p.name}", md5)
        collect.record({"id": new_id, "path": p.name, "remote": f"{rdir}/{p.name}", "bytes": p.stat().st_size, "md5": md5,
                        "source": f"derived:{a.id}", "fetched_at": datetime.now().isoformat(timespec="seconds")})
    rep = REPORTS / f"{new_id}.json"
    collect.upload(rep, f"{rdir}/clean_report.json", hashlib.md5(rep.read_bytes()).hexdigest())
    collect.rclone("copyto", str(C / "samples" / f"{new_id}.jsonl"), f"{collect.REMOTE}/_samples/{new_id}.jsonl")

    # sources.json 是手工排版的，整体 dump 会打乱排版：新条目按文本插到 sources 列表末尾（已有就不动），插完校验
    sp = C / "sources.json"
    txt = sp.read_text()
    if not any(s.get("id") == new_id for s in json.loads(txt)["sources"]):
        i = txt.rstrip().rfind("]")
        txt2 = txt[:i].rstrip() + ",\n    " + json.dumps(new_src, ensure_ascii=False) + "\n  " + txt[i:]
        assert json.loads(txt2)["sources"][-1]["id"] == new_id
        sp.write_text(txt2)
    rv = collect.load_reviews()
    rv[new_id] = {"status": "pending_review", "summary": f"由 {a.id} 按评估里的清洗要求生成，{n} 条，待复核",
                  "issues": [], "cleaning": [], "checked": "", "reviewed_at": datetime.now().isoformat(timespec="minutes"),
                  "reviewer": "clean.py"}
    (C / "reviews.json").write_text(json.dumps(rv, ensure_ascii=False, indent=1))
    collect.catalog()
    if not a.keep_work:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    collect.log(f"清洗 {a.id}：已上传 {len(shards)} 个文件到 {rdir}，登记为 {new_id}（pending_review）")


if __name__ == "__main__":
    main()
