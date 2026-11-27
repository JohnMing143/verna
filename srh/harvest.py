"""从网上收集人写文字。每种来源一个生成器，统一产出 {id, source, url, title, author, date, genre_hint, text}。

来源写在 autopilot/config.json 的 sources 里，类型：
  hf_zhihu     Zhihu-KOL，按回答时间和点赞过滤
  hf           任意 HuggingFace 数据集（指定文本字段、过滤条件）
  wikisource   维基文库：按作者分类取公有领域作品，繁体转简体
  site         网站：从列表页出发按链接规则抓文章页，trafilatura 提取正文和日期；遵守 robots.txt、限速

需要 .venv 里的 datasets、trafilatura、lxml_html_clean、opencc-python-reimplemented。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections.abc import Iterator

from .segment import _CODE_REF  # 引出代码的句子（"下面是一个例子。"），切段规则也用

UA = "srh-corpus/0.3 (research; low rate; https://github.com/JohnMing143/verna)"
_robots: dict[str, urllib.robotparser.RobotFileParser] = {}
_last_hit: dict[str, float] = {}


def _fetch(url: str, min_interval: float = 2.0, timeout: int = 30, api: bool = False) -> str | None:
    """api=True 用于官方开放的接口（如 MediaWiki API）：robots.txt 管的是网页爬虫，接口按其使用政策带 UA 访问即可。"""
    host = urllib.parse.urlparse(url).netloc
    if api:
        pass
    elif host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.parse(urllib.request.urlopen(urllib.request.Request(f"https://{host}/robots.txt", headers={"User-Agent": UA}), timeout=15).read().decode("utf-8", "ignore").splitlines())
        except Exception:
            rp.parse([])  # 没有 robots.txt 当作全部允许
        _robots[host] = rp
    if not api and not _robots[host].can_fetch(UA, url):
        return None
    wait = _last_hit.get(host, 0) + min_interval - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)
        raw = r.read()
        enc = r.headers.get_content_charset() or "utf-8"
        return raw.decode(enc, "ignore")
    except Exception:
        return None


def _id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


# ---------------- HuggingFace ----------------

def _upvotes(s) -> float:
    m = re.search(r"([\d.]+)\s*(万|[kK])?", str(s or ""))
    return float(m.group(1)) * {"万": 1e4, "k": 1e3, "K": 1e3}.get(m.group(2) or "", 1) if m else 0.0


def hf_zhihu(spec: dict) -> Iterator[dict]:
    from datasets import load_dataset
    before, min_up = spec.get("before", "2022-11-30"), spec.get("min_upvotes", 200)
    for i, r in enumerate(load_dataset("wangrui6/Zhihu-KOL", split="train", streaming=True)):
        try:
            m = json.loads(r.get("METADATA") or "{}")
        except json.JSONDecodeError:
            continue
        t = m.get("answer_creation_time") or ""
        if not t or (before and t >= before) or _upvotes(m.get("upvotes")) < min_up:
            continue
        yield {"id": f"zhihu:{int(m.get('answer_id') or i)}", "source": spec["name"], "url": m.get("url"), "title": r["INSTRUCTION"],
               "author": None, "date": t[:10], "genre_hint": spec.get("genre_hint"), "text": r["RESPONSE"] or ""}


def hf(spec: dict) -> Iterator[dict]:
    """data_files（可选）：按年份分文件的数据集，逐个文件取，每个文件最多 per_file 篇。
    11-05 复查：people_daily 不分文件时隔 20 取 1，3000 篇在 1995–1996 两年就取满了，date_min/date_max 等于没起作用。"""
    if spec.get("data_files"):
        for f in spec["data_files"]:
            sub = {k: v for k, v in spec.items() if k not in ("data_files", "per_file")}
            stem = f.split("/")[-1].split(".")[0]
            for k, doc in enumerate(_hf_one(sub, data_files=f)):
                if k >= spec.get("per_file", 200):
                    break
                yield {**doc, "id": f"{spec['name']}:{stem}:{doc['id'].rsplit(':', 1)[-1]}"}
        return
    yield from _hf_one(spec)


def _hf_one(spec: dict, data_files: str | None = None) -> Iterator[dict]:
    from datasets import load_dataset
    if data_files:
        ds = load_dataset(spec["dataset"], data_files=data_files, split="train", streaming=True)
    else:
        ds = load_dataset(spec["dataset"], spec.get("config"), split=spec.get("split", "train"), streaming=True)
    where = spec.get("where", {})  # {"字段": "值"}，全部相等才要
    every = spec.get("every", 1)    # 大数据集隔几条取一条，让样本分散到不同年份和栏目
    where_not = spec.get("where_not", {})  # {"字段": ["值", ...]}，等于其中任何一个就不要
    cap = spec.get("cap_by")  # {"field": "label", "max": 250}：每个类别最多取几篇（按类别排序的数据集，不设就只取到开头几类）
    taken = {}
    for i, r in enumerate(ds):
        if i % every or any(str(r.get(k)) != str(v) for k, v in where.items()):
            continue
        if any(str(r.get(k)) in {str(x) for x in v} for k, v in where_not.items()):
            continue
        if spec.get("title_re") and not re.search(spec["title_re"], str(r.get(spec.get("title_field", "title")) or "")):
            continue  # 11-05：人民日报里只取讲话、致辞（演讲稿文体没有别的人写来源）
        if cap:
            key = str(r.get(cap["field"]))
            if taken.get(key, 0) >= cap["max"]:
                continue
            taken[key] = taken.get(key, 0) + 1
        date = str(r.get(spec["date_field"]) or "")[:10].replace(".", "-") if spec.get("date_field") else None
        if date is not None and ((spec.get("date_min") and date < spec["date_min"]) or (spec.get("date_max") and date >= spec["date_max"])):
            continue
        text = (r.get(spec.get("text_field", "text")) or "").replace("　　", "\n").replace("　", "")
        yield {"id": f"{spec['name']}:{i}", "source": spec["name"], "url": None, "title": r.get(spec.get("title_field", "title")),
               "author": None, "date": date,
               "genre_hint": spec.get("genre_hint"), "text": text, "normalize_punct": spec.get("normalize_punct", False),
               "meta": {k: r.get(k) for k in spec.get("keep_fields", [])}}


# ---------------- 维基文库 ----------------

def _ws_api(**kw) -> dict:
    kw.update(format="json", formatversion=2)
    s = _fetch("https://zh.wikisource.org/w/api.php?" + urllib.parse.urlencode(kw), min_interval=1.0, api=True)
    return json.loads(s) if s else {}


def wikisource(spec: dict) -> Iterator[dict]:
    import opencc
    from lxml import html as lh
    t2s = opencc.OpenCC("t2s")
    for author in spec["authors"]:
        cont, titles = None, []
        while True:  # 作者分类下的作品页
            kw = dict(action="query", list="categorymembers", cmtitle=f"Category:{author}", cmlimit=200, cmnamespace=0)
            if cont:
                kw["cmcontinue"] = cont
            j = _ws_api(**kw)
            titles += [m["title"] for m in j.get("query", {}).get("categorymembers", [])]
            cont = j.get("continue", {}).get("cmcontinue")
            if not cont or len(titles) >= spec.get("max_per_author", 300):
                break
        for title in titles[: spec.get("max_per_author", 300)]:
            j = _ws_api(action="parse", page=title, prop="text|categories", redirects=1)
            p = j.get("parse")
            if not p:
                continue
            cats = [c["category"] for c in p.get("categories", [])]
            pd = [c for c in cats if "PD" in c]  # 维基文库只收可自由使用的作品，这里只记下版权标记
            tree = lh.fromstring(p["text"])
            for bad in tree.xpath('//table|//div[contains(@class,"header")]|//*[contains(@class,"ws-noexport")]|//sup|//style|//*[contains(@class,"mw-editsection")]'):
                bad.getparent().remove(bad)
            paras = [re.sub(r"\s+", "", x.text_content()) for x in tree.xpath("//p|//dd")]
            text = "\n".join(t2s.convert(x) for x in paras if len(x) > 1)
            years = [c for c in cats if re.fullmatch(r"\d{4}年", c)]
            yield {"id": f"ws:{_id(title)}", "source": spec["name"], "url": f"https://zh.wikisource.org/wiki/{urllib.parse.quote(title)}",
                   "title": t2s.convert(title), "author": author, "date": years[0][:4] if years else None,
                   "genre_hint": spec.get("genre_hint"), "text": text, "license": pd[0] if pd else None}


# ---------------- 网站 ----------------

def site(spec: dict) -> Iterator[dict]:
    """spec: start_urls（列表页）、article_re（文章页 URL 正则）、list_re（继续翻的列表页正则，可选）、
    max_articles、before（只要早于这个日期的，可选）、min_interval（每个域名两次请求的间隔秒数）"""
    import trafilatura
    art_re, list_re = re.compile(spec["article_re"]), re.compile(spec["list_re"]) if spec.get("list_re") else None
    queue, seen_list, seen_art, n = list(spec["start_urls"]), set(), set(), 0
    gap = spec.get("min_interval", 3.0)
    while queue and n < spec.get("max_articles", 500):
        u = queue.pop(0)
        if u in seen_list:
            continue
        seen_list.add(u)
        page = _fetch(u, gap)
        if not page:
            continue
        links = {urllib.parse.urljoin(u, h).split("#")[0] for h in re.findall(r'href=["\']([^"\']+)["\']', page)}
        for a in sorted(links):
            if art_re.search(a) and a not in seen_art:
                seen_art.add(a)
                doc = _fetch(a, gap)
                if not doc:
                    continue
                ex = trafilatura.bare_extraction(doc, url=a, with_metadata=True, include_comments=False, include_tables=False, favor_precision=True)
                if not ex:
                    continue
                ex = ex if isinstance(ex, dict) else ex.as_dict()
                date = (ex.get("date") or "")[:10] or None
                if spec.get("before") and (not date or date >= spec["before"]):
                    continue
                text = ex.get("text") or ""
                if len(text) < 200:
                    continue
                n += 1
                yield {"id": f"{spec['name']}:{_id(a)}", "source": spec["name"], "url": a, "title": ex.get("title"),
                       "author": ex.get("author"), "date": date, "genre_hint": spec.get("genre_hint"), "text": text}
                if n >= spec.get("max_articles", 500):
                    return
            elif list_re and list_re.search(a) and a not in seen_list:
                queue.append(a)


# ---------------- GitHub 上的 Markdown 文集 ----------------

# 收集代码的版本。run.py 的收集线程只收 needs_harvest_version 不超过它的来源：唤醒时改了收集代码、同时在 config 里加了用新功能的来源，
# 旧进程（内存里是旧代码）不会抢先按旧逻辑收掉（11-05 thucnews_mix 就是这样收错的）。
# 2：_md_to_text 处理句号结尾的引出句和引用块（11-05）
# 3：行内代码先藏起来再处理公式（之前 bash 教程的 `$1`、`$IFS` 被当成行内公式吃掉了 $）；<sup>53</sup> → ^53（之前变成"253"）；
#    spec 的 md_runs=true：代码块、引用块、标题处断开（RUN_SEP），紧挨代码块的前后两段整段去掉，切段不跨断点（11-05 13:43 复查 wangdoc2）
VERSION = 3
RUN_SEP = "\x1e"  # 文档里的硬断点：segment_doc 先按它分开再切段



def _md_to_text(md: str, runs: bool = False) -> str:
    """runs=True（第三版，spec 的 md_runs）：代码块、引用块、标题处插 RUN_SEP，紧挨代码块/引用块的前一段和后一段整段去掉。
    网道这类教程每隔一两段就有代码，"下面是一个例子。""Bash 函数定义的语法有两种。""上面代码中……"删句子删不干净
    （11-05 13:43 复查 wangdoc2：已分级 67 段 23 段 unusable，good 里也有"上面代码中的event.state"），只留不挨着代码的连续正文。"""
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)          # front matter
    md = re.sub(r"```.*?```", "\n\x01\n", md, flags=re.S)           # 代码块（先留记号，下面删掉指向它的句子）
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)                   # HTML 注释
    codes: list[str] = []  # 行内代码先藏起来：免得 `<img>` 被当成 HTML 标签删掉；第三版挪到公式前面，免得 `$1`~`$9` 被当成公式
    md = re.sub(r"``\s?([^`\n]+?)\s?``|`([^`\n]+)`", lambda m: codes.append(m.group(1) or m.group(2)) or f"\x02{len(codes) - 1}\x03", md)
    md = re.sub(r"\$\$.*?\$\$", "", md, flags=re.S)                     # 公式块
    md = re.sub(r"\$([^$\n]{1,40})\$", r"\1", md)                      # 行内公式留内容：$n$ → n
    md = re.sub(r"(?m)^\s*(?:!!!|\?\?\?\+?|===)\s.*$", "", md)          # mkdocs 提示框、代码分页的标记行（11-05 加，hello-algo）
    md = re.sub(r"(?m)^ {4}(?=\S)", "", md)                           # 提示框里缩进的正文
    md = re.sub(r"\[\^\w+\]:?", "", md)                                # 脚注记号
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                     # 图片
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)                 # 链接留文字
    md = re.sub(r"<sup>([^<]{1,12})</sup>", r"^\1", md)                 # 2<sup>53</sup> → 2^53（之前删标签变成"253"）
    md = re.sub(r"<sub>([^<]{1,12})</sub>", r"_\1", md)
    md = re.sub(r"<[^>]+>", "", md)                                  # HTML 标签
    md = re.sub(r"(?m)^#{1,6}\s.*$", "\n\x05\n" if runs else "", md)  # 标题行（runs：换节处断开）
    md = re.sub(r"(?m)^>.*(?:\n>.*)*$", "\n\x04\n", md)                # 引用块是别人的文字，整块去掉（留记号，下面处理引出它的句子）
    md = re.sub(r"(?m)^\s*[（(]图片说明[:：][^\n]*?[）)]\s*$", "", md)  # 独占一行的图注
    md = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m.group(1) or m.group(2), md)
    md = re.sub(r"\x02(\d+)\x03", lambda m: codes[int(m.group(1))], md)
    # 代码删掉后，"代码如下：""上面示例中……"这类句子就悬空了（11-05，网道教程、hello-algo），整段去掉。
    # 第二版（11-05 复查 wangdoc）：引出代码的句子用句号结尾的也算（"下面是一个例子。""上面的代码可以改写成下面这样。"），
    # 只去掉段末那一句；引用块同样处理（"网上找到的解释："去掉，紧跟着的"看了这个定义……"整段去掉）。
    paras = [x.strip() for x in re.split(r"\n\s*\n", md)]
    if runs:
        return _md_runs(paras)
    keep = []
    for i, x in enumerate(paras):
        nxt = next((y for y in paras[i + 1:] if y), "")
        prv = next((y for y in reversed(paras[:i]) if y), "")
        if x in ("\x01", "\x04") or not x:
            continue
        if nxt in ("\x01", "\x04"):
            sents = re.findall(r"[^。！？：:]*[。！？：:]|[^。！？：:]+$", x)
            last = sents[-1].strip() if sents else ""
            if last.endswith(("：", ":")) or (nxt == "\x01" and _CODE_REF.search(last)):
                x = "".join(sents[:-1]).strip()
                if not x:
                    continue
        if prv == "\x01" and re.match(r"(上面|上例|上述|以上|这段|该)(示例|代码|例子|命令|脚本|写法|程序)?", x):
            continue
        if prv == "\x04" and re.match(r"(看了这|这段话|这句话|这个定义|这就是说|他的意思|她的意思|我理解他|上面这|以上这)", x):
            continue
        keep.append(x.replace("\x01", "").replace("\x04", ""))
    return "\n\n".join(keep)


def _md_runs(paras: list[str]) -> str:
    """第三版：按代码块(\x01)、引用块(\x04)、标题(\x05)断开；紧挨代码块或引用块的前后一段去掉；各段连续正文用 RUN_SEP 隔开。"""
    marks = ("\x01", "\x04", "\x05")
    paras = [x for x in paras if x]
    out: list[str] = []
    for i, x in enumerate(paras):
        if x in marks:
            if not out or out[-1] != RUN_SEP:
                out.append(RUN_SEP)
            continue
        nxt = paras[i + 1] if i + 1 < len(paras) else ""
        prv = paras[i - 1] if i else ""
        if nxt in ("\x01", "\x04") or prv in ("\x01", "\x04") or "\x01" in x or "\x04" in x:
            continue
        out.append(x)
    return "\n\n".join(out).strip(RUN_SEP + "\n")


def git_md(spec: dict) -> Iterator[dict]:
    """spec: repo（https 地址）、glob（如 docs/*.md）、genre_hint、max_docs。浅克隆到 data/v3/git/<checkout 或 name>。
    section（可选）：二级标题的正则，只取匹配的那一节（到下一个二级标题为止），没有这一节的文件跳过。
    周刊这类文件整篇切段会把链接列表、图注、译文拼到一起，只有作者自己写的那一节能用。"""
    if spec.get("repos"):  # 11-05：同一作者的几个小仓库合成一个来源（分级按来源轮流，拆开会多占名额）
        for repo in spec["repos"]:
            sub = {k: v for k, v in spec.items() if k != "repos"}
            base = repo.rstrip("/").split("/")[-1]
            yield from git_md({**sub, "repo": repo, "checkout": spec.get("checkout", spec["name"]) + "/" + base, "_prefix": base + "/"})
        return
    import subprocess
    from pathlib import Path
    dest = Path(__file__).resolve().parents[1] / "data" / "v3" / "git" / spec.get("checkout", spec["name"])
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if spec.get("sparse"):  # 大仓库（图片、多语言）只取需要的 md 文件，不下载其他内容
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", spec["repo"], str(dest)],
                           check=True, capture_output=True, timeout=1800)
            subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *spec["sparse"]], check=True, capture_output=True, timeout=600)
            subprocess.run(["git", "-C", str(dest), "checkout"], check=True, capture_output=True, timeout=1800)
        else:
            subprocess.run(["git", "clone", "--depth", "1", spec["repo"], str(dest)], check=True, capture_output=True, timeout=1800)
    for p in sorted(dest.glob(spec["glob"])):
        md = p.read_text(errors="ignore")
        if spec.get("section"):
            m = re.search(r"(?m)" + spec["section"] + r".*$", md)
            if not m:
                continue
            nxt = re.search(r"(?m)^## ", md[m.end():])
            md = md[m.end(): m.end() + nxt.start()] if nxt else md[m.end():]
        text = _md_to_text(md, runs=bool(spec.get("md_runs")))
        if len(text.replace(RUN_SEP, "")) < 200:
            continue
        rel = str(p.relative_to(dest))
        yield {"id": f"{spec['name']}:{_id(spec.get('_prefix', '') + rel)}", "source": spec["name"], "url": f"{spec['repo']}/blob/HEAD/{rel}", "title": p.stem,
               "author": spec.get("author"), "date": None, "genre_hint": spec.get("genre_hint"), "text": text}


HARVESTERS = {"hf_zhihu": hf_zhihu, "hf": hf, "wikisource": wikisource, "site": site, "git_md": git_md}
