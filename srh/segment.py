"""把收集到的整篇文字切成适合训练的段落，并用规则粗筛。规则按文体区分：技术文档允许列表，口语允许语气词。"""
from __future__ import annotations

import hashlib
import re

from . import tells

NOISE = re.compile(
    r"https?://|www\.|编辑于|发布于|著作权归|谢邀|利益相关|如图|上图|下图|见图|\[图片\]|图源|图片来源|"
    r"关注我|点赞|私信|评论区|扫码|二维码|淘宝|拼多多|优惠券|转载请|责任编辑|原标题|本文来源|"
    r"~{2,}|！{3,}|\?{3,}|？{3,}|\.{6,}|哈{4,}|呜{3,}"
)
LIST_OK = {"tech", "business", "official"}  # 这些文体里分点是正常写法

# 切段之后补的规则（11-05 来源复查）：除了切段时用，分级、精修、复审、打磨挑候选时也再查一遍，
# 所以对已经切好的段落同样生效（不改已有数据，只是不再往下游送）。
_TRAD = set("這們個來為說時對會過還與麼後學經國實現發點開關無於從當動體裡長問題應認讓進種樣將軍書車門見電話頭東區處藝歲隨義爲盡傳寫聽變離歡難觀氣廣")
LATE_NOISE = re.compile(r"公有领域|版权期限|维基媒体基金会|本站作消极容忍|【参考资料】")  # 维基文库页尾的版权声明（123 段）；《经典常谈》段尾的书目行（11-05 复查，5 段）

# 《人民日报》版面尾部的东西，不是文章内容。11-08 复查 people_daily_opinion：作者署名"（作者为中国驻俄罗斯大使李辉）"
# 4 段（3 段 improvable 会进打磨），people_daily_span 2 段、people_daily 1 段（其中 1 段已进快照）；
# 书讯"有关购书事宜，请直接与人民日报出版社联系"2 段（已判 unusable）；投稿邮箱"（欢迎读者热忱投稿，来稿请投：……）"1 段（good）。
# "（作者为"锚定括号，正文里的"词作者为熊复"不受影响。
BYLINE_NOISE = re.compile(r"（作者为[^）\n]{2,80}）|购书事宜|人民日报出版社联系|来稿请投")


def head_residue(chunk: str) -> bool:
    """段首是标题残留：第一行没有句末标点，下面还有正文。《人民日报》语料的正文开头常带主副标题
    （"克林顿：伊应承担冲突责任⏎萨达姆：做好抗击侵略准备⏎新华社华盛顿2月5日电……"），旧规则只去掉不带任何标点的短行，
    带冒号、引号或超过 40 字的标题留了下来。11-05 复查：新闻来源里命中的 42 段 good/exemplar 全是标题、副题或电头；
    书信的"子恺兄："、诗行也会命中，所以只对新闻来源用。"""
    ls = chunk.split("\n")
    return len(ls) >= 2 and _is_head(ls[0])


def _is_head(line: str) -> bool:
    f = line.strip()
    if not f or len(f) > 80 or re.search(r"[。！？!?…]", f):
        return False
    return not (f[-1] in "：:—，,；;、“「（(" and len(f) > 15)  # 长的引导句（"……公告如下：""请看——"）是正文


# 新浪稿（THUCNews）人名、楼盘名后面跟的链接导航文字："首开国风上观 (论坛 相册 户型 样板间 地图搜索)""李晨( blog)""姚明(微博)"。
# 11-05 复查：thucnews_cat 2365 篇里 212 篇有，thucnews_mix 261 篇。括号里只有这些导航词才算，切段前删掉，已切好的段落不再往下游送。
LINK_RESIDUE = re.compile(r"[  ]*[(（][  ]*(?:(?:论坛|相册|户型|样板间|点评|地图搜索|视频|听歌|blog|博客|微博|企业专区|旗下楼盘)"
                          r"[  ,，、]*)+[)）]")

# 网道这类 markdown 教程，代码块删掉以后留下指向它的句子："下面是一个例子。""上面的代码可以改写成下面这样。"，
# 后面紧跟的却是正文。11-05 复查：wangdoc 361 段里 138 段有，已分级的 32 段里 28 段被判 unusable。
_CODE_REF = re.compile(r"(?:下面|下例|如下|以下)(?:是|的|就是|这|这个|这段|这些|这种|几个|一些|一个|两个|三个|几种|一段|其中的?)?"
                       r"[^。！？\n]{0,24}?(?:代码|例子|示例|写法|命令|脚本|程序|输出|这样|形式|样子)[^。！？\n]{0,6}[。：:]$"
                       r"|如下[。：:]$")  # "获取字符串长度的语法如下。"后面跟的不是列表就是被删的代码


def _prose(line: str) -> bool:
    """像正文的一行：中文占三成以上（代码行、命令行、列表项不算）。"""
    s = line.strip()
    return bool(s) and not re.match(r"[-*•>]|\d{1,2}[.、)）]|[（(]\d", s) and len(re.findall(r"[一-鿿]", s)) / len(s) >= 0.3


def dangling_ref(chunk: str, quotes_stripped: bool = False) -> bool:
    """指向代码/例子的句子后面没有代码；quotes_stripped=True 时，冒号结尾的引导句后面直接是正文也算（引用块被删了）。"""
    ls = chunk.split("\n")
    for i, line in enumerate(ls):
        nxt = ls[i + 1] if i + 1 < len(ls) else ""
        if not (nxt == "" or _prose(nxt)) or re.match(r"(?:比如|例如|譬如|假设|假如|举例|举个例子)", nxt):  # 例子用文字写在下一行，不算悬空
            continue
        if _CODE_REF.search(line.strip()) or (quotes_stripped and re.search(r"[：:]$", line.strip())):
            return True
    return False


# COIG-CQIA 把豆瓣用户的书评/影评套进问答模板，正文前加了一句助手腔开场白："当然可以！以下是一篇关于《X》的书评，字数约1900字："
# （11-05 复查：cqia_douban 1500 篇里 82 篇，切出 41 段，还没进配对）。正文是人写的，开场白不能进训练。
TEMPLATE_OPENER = re.compile(r"(?m)^(?:当然可以|好的|没问题)[！!，,。][^\n]{0,60}(?:以下是|下面是)|^以下是一篇关于")


# markdown 来源（代码块收集时删了）里明确指向代码的说法，出现在段中任何位置都算：dangling_ref 只查"下面是一个例子。"这类独占一行、
# 后面紧跟正文的引出句，查不到段中的"上面代码中的event.state""比如上例不写txt的类型""写法二有两个地方需要注意"。
# 11-05 13:43 复查 wangdoc2：已分级 67 段命中 26 段（good 3、improvable 9、unusable 14），逐条看过都是真指代；ruanyf_topic、linux101 为 0，
# hello_algo 1 段（"在本示例中"，也是指代码）。"这段代码""这个例子"在周刊和随笔里是正常说法，不收。
CODE_TRACE = re.compile(r"上面(?:的)?(?:代码|示例|例子|命令|脚本|写法|程序)|上例|本例|下例|下面(?:的)?(?:代码|示例|例子|命令|写法)"
                        r"|下面是[^。！？\n]{0,20}(?:例子|示例|代码|写法)|如下所示|写法[一二三四]|以下代码|示例中|本示例")

# 行内代码里的直引号被 normalize_text 换成了弯引号：class=“right”、scanf(“ %c”, &ch)、require(“./b”);、String a = “hi”;。
# 收集时行内代码的反引号去掉了，切段时分不出是代码还是正文。11-05 16:36 复查 wangdoc3：322 段里 7 段（good 3），
# wangdoc2 21 段、hello_algo 1 段。只对 markdown 来源查（新闻、知乎里半角括号接引号"(简称“乐居”)"是正常写法）。
# normalize_text 已改成 code=True 时不换代码样子的引号，这条拦的是改之前切好的段落。
MD_CODE_QUOTE = re.compile(r"=“|[A-Za-z_]\(“|”[;,]|“(?:\.{0,2}/|\*| ?%|\$)")

# mkdocs 扩展语法没转换：图注后面的属性行"{: .caption }"、按键"++“Shift”++ + ++“k”++"。11-05 16:36 复查：只有 linux101 有，8 段（good 1 复审没过、improvable 2），没进配对。
MD_MARKUP = re.compile(r"\{:\s*\.[\w-]+[^}\n]*\}|\+\+[\"“]?[A-Za-z][\w-]{0,10}[\"”]?\+\+")

# markdown 仓库里没发布的存档稿：网道 bash-tutorial 的 docs/archives/（含 commands/、deleted/）多是《The Linux Command Line》
# 的旧译稿（"wait 命令导致一个父脚本暂停运行，直到一个 特定的进程"），不是正式教程。11-05 16:36 复查 wangdoc3：15 段，已分级 11 段（unusable 4、improvable 6、good 1），
# 1 段进了打磨人写配对（已写进 exclude_ids.txt）。按网址拦，git_md 收集时也跳过。
MD_SKIP_PATH = re.compile(r"/docs/archives/")


# 《人民日报》讲话稿（people_daily_speech）正文前的署名、日期跟称呼粘成一行："胡锦涛同志们：""中华人民共和国国务院总理李克强尊敬的吴登盛总统，各位同事："
# "（一九九六年六月十四日）各位常委、各位同志："；文末还有"（新华社北京9月16日电）"。11-05 13:43 复查：906 段里命中 66 段（已分级的 5 段）。
_SPEECH_DATE = re.compile(r"^（[^）\n]{4,20}日[^）\n]{0,8}）")
_SPEECH_SAL = re.compile(r"(?<=[一-鿿])(?<![的们和及与体])(?:同志们|朋友们|各位|尊敬的|女士们)")  # 称呼前面紧挨着人名或职务
_XINHUA_TAIL = re.compile(r"（新华社[^（）\n]{2,24}电）")


def speech_residue(chunk: str) -> str | None:
    first = chunk.split("\n", 1)[0]
    if _SPEECH_DATE.match(first) or re.search(r"日[^）\n]{0,8}）\s*(?:同志们|朋友们|各位|尊敬的|女士们|主席先生)", first):
        return "署名残留"  # 日期括号在段首，或夹在署名和称呼中间（"国务院总理 李克强（二0一四年十月十六日，米兰）尊敬的主席先生"）
    m = _SPEECH_SAL.search(first) if "。" not in first and len(first) <= 150 else None
    if m and m.start() < (first.find("，") if "，" in first else len(first)) and not first.startswith(("尊敬的", "亲爱的", "各位")):
        return "署名残留"
    if _XINHUA_TAIL.search(chunk):
        return "电头残留"
    return None


def late_check(chunk: str, genre_hint: str | None = None, md: bool = False) -> str | None:
    """md=True：从 markdown 转来的文字（git_md 来源），代码块和引用块在收集时删掉了。"""
    if genre_hint == "news" and head_residue(chunk):
        return "标题残留"
    if genre_hint == "speech" and (r := speech_residue(chunk)):
        return r
    if LATE_NOISE.search(chunk):
        return "版权声明"
    if BYLINE_NOISE.search(chunk):
        return "署名书讯"
    if LINK_RESIDUE.search(chunk):
        return "网页残留"
    if TEMPLATE_OPENER.search(chunk):
        return "模板开场白"
    # 段尾是"他说：""要切实把握好以下几个重点："，下文被切到下一段了（11-05 复查：wikisource 11 段 good、知乎"相关回答："）
    if re.search(r"[：:]\s*$", chunk):
        return "冒号收尾"
    # 代码删掉后的"下面是一个例子。"（技术文档）；markdown 来源里冒号引导句后面的引用块也删了（阮一峰周刊"网上找到的解释：⏎看了这个定义……"）。
    # 新闻、维基文库、豆瓣访谈里冒号后面接正文是正常写法，所以冒号那条只查 markdown 来源。
    if (genre_hint == "tech" or md) and dangling_ref(chunk, quotes_stripped=md):
        return "引用悬空"
    if md and CODE_TRACE.search(chunk):
        return "代码指代"
    if md and MD_CODE_QUOTE.search(chunk):
        return "代码引号"
    if md and MD_MARKUP.search(chunk):
        return "标记残留"
    if sum(ch in _TRAD for ch in chunk) / max(len(chunk), 1) > 0.01:  # 繁体原文（豆瓣简介、知乎少数回答），会教模型输出繁体
        return "繁体"
    if re.search(r"[^。！？!?…\s’』][”」]\n", chunk):  # 句中引号后被断了行（旧切段规则的问题），读起来是半句
        return "引号处断句"
    return None


def late_check_seg(s: dict) -> str | None:
    """对切好的段落记录做 late_check（run.py 各环节用）。"""
    url = str(s.get("url") or "")
    if url.endswith(".md") and MD_SKIP_PATH.search(url):
        return "存档旧稿"
    return late_check(s["text"], s.get("genre_hint"), md=url.endswith(".md"))


_HALF = {",": "，", ".": "。", ":": "：", ";": "；", "?": "？", "!": "！", "(": "（", ")": "）"}


def normalize_punct(text: str) -> str:
    """CSL 这类语料在中文之间用半角标点，换成全角（数字和英文里的不动）。"""
    text = re.sub(r"(?<=[\u4e00-\u9fff”’）」%℃\]])([,.:;?!])(?=[\u4e00-\u9fff“‘（「]|$)|(?<=[\u4e00-\u9fff])([,.:;?!])",
                  lambda m: _HALF[m.group(1) or m.group(2)], text)
    return re.sub(r"(?<=[%℃])\.(?=\d)", "。", text)  # "+1.38℃.1998年" 这类：℃/% 后面紧跟的点是句号


_FW = {**{chr(0xFF10 + i): str(i) for i in range(10)}, **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
       **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)}, "％": "%", "＋": "+", "－": "-"}


def normalize_text(text: str, code: bool = False) -> str:
    """全角数字和字母转半角（《人民日报》语料里很多）；中文里的直引号换成弯引号。中文标点保持全角。
    code=True（技术文档、markdown 来源）：代码样子的引号不换——前面紧挨 = 或函数名加括号、后面紧跟 ; , 、引号里没有汉字
    （11-05 16:36 复查 wangdoc3：class="right"、scanf(" %c", &ch) 被换成了弯引号，代码被改了）。"""
    text = "".join(_FW.get(ch, ch) for ch in text)
    text = re.sub(r"(?<=\d)．(?=\d)", ".", text)

    def curl(m: re.Match) -> str:
        if code:
            before, after = text[m.start() - 1: m.start()], text[m.end(): m.end() + 1]
            if (not re.search(r"[一-鿿]", m.group(1)) or re.match(r"[=(\[{,+]", before)
                    or re.match(r"[;,)\]}]", after)):
                return m.group(0)
        return f"“{m.group(1)}”"
    return re.sub(r'"([^"\n]{1,200})"', curl, text)


def paragraphs(text: str, keep_list: bool = False) -> list[str]:
    text = text.replace("\r", "").replace("　", " ")
    # 换行被压成空格的语料。右引号后面断段，引号前得是句末标点：之前"“一个人多个银行账户” 的管理意见"会从引号后断开，
    # 切出"的管理意见."开头的残段（11-05 复查，约 650 段）
    text = re.sub(r"(?:(?<=[。！？!?])|(?<=[。！？!?…][”」]))[ ]{1,}(?=\S)", "\n", text)
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    # 去掉标题行：短、没有任何句读标点（新闻标题、小标题、栏目名）。
    # 技术文档的列表项（"- `<` 小于运算符"）也短、也没标点，之前一起被删，"主要有下面6个"后面只剩 1 个（11-05 复查 wangdoc），这类文体保留列表项
    return [p for p in paras if len(p) >= 40 or re.search(r"[。！？!?…，,：:；;]", p)
            or (keep_list and re.match(r"(?:[-*•]|\d{1,2}[.、)）])\s+\S", p))]


def chunks(paras: list[str], lo: int = 150, hi: int = 800) -> list[str]:
    out, cur = [], ""
    for p in paras:
        if len(p) > hi:
            for s in re.split(r"(?<=[。！？!?])", p):
                if cur and len(cur) + len(s) > hi:
                    out.append(cur)
                    cur = ""
                cur += s
            continue
        if cur and len(cur) + len(p) + 1 > hi:
            head, _, last = cur.rpartition("\n")
            if head and re.search(r"[：:]$", last):  # 冒号引导句跟着下文走，不留在上一段段尾（"他说：" + 下一段的引文）
                out.append(head)
                cur = last
            else:
                out.append(cur)
                cur = ""
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return [c.strip() for c in out if lo <= len(c.strip()) <= hi + 150]


def rule_check(chunk: str, genre_hint: str | None = None, max_density: float = 4.0, md: bool = False) -> str | None:
    """返回 None 表示通过，否则返回原因。这里只挡明显的噪声和残缺，好坏交给大模型分级。"""
    if NOISE.search(chunk):
        return "噪声词"
    han = len(re.findall(r"[一-鿿]", chunk))
    min_han = 0.4 if genre_hint == "tech" else 0.6
    if han / max(len(chunk), 1) < min_han:
        return "中文比例低"
    if len(re.findall(r"[。！？!?]", chunk)) < 2:
        return "句子太少"
    if re.search(r"\S {2,}\S", chunk):
        return "内容残缺"
    if not re.search(r"[。！？!?”」…）)】:：]$", chunk):
        return "没说完"
    if re.match(r"^(楼上|楼主|同上|补充|更新|追加|以上|另外|还有|所以|因此|但是|不过|而且|然后)", chunk):
        return "依赖上文"
    if genre_hint not in LIST_OK and re.search(r"(?m)^\s*(?:[-*•]|\d{1,2}[.、)）])\s+", chunk):
        return "列表"
    w = tells.weighted(chunk)
    if genre_hint in LIST_OK:  # 这些文体分点是正常写法，列表不算 AI 腔（11-05 复查：网道教程保留列表项后 358 块里大半只因列表被判 AI 腔偏高）
        w -= sum(n * wt for name, _, n, wt in tells.hits(chunk) if name == "列表")
    if w > max(3.0, max_density * len(chunk) / 1000):
        return "AI腔偏高"
    return late_check(chunk, genre_hint, md=md)


def drop_heads(paras: list[str]) -> list[str]:
    """新闻正文开头的主副标题、栏目名（没有句末标点的短段）去掉再切段，免得整段第 0 段因"标题残留"被扔掉。"""
    k = 0
    while k < min(3, len(paras) - 1) and _is_head(paras[k]):
        k += 1
    return paras[k:]


def dedupe_key(text: str) -> str:
    return hashlib.md5(re.sub(r"\W", "", text)[:80].encode()).hexdigest()


def segment_doc(doc: dict, per_doc: int = 3, seen: set | None = None, why: dict | None = None) -> list[dict]:
    seen = seen if seen is not None else set()
    out = []
    g = doc.get("genre_hint")
    hi = 1000 if g in ("fiction", "essay") else 800
    url = str(doc.get("url") or "")
    if url.endswith(".md") and MD_SKIP_PATH.search(url):  # 没发布的存档稿（见 MD_SKIP_PATH）
        if why is not None:
            why["存档旧稿"] = why.get("存档旧稿", 0) + 1
        return out
    # 先把直引号换成弯引号，normalize_punct 才认得出引号后的半角标点；技术文档、markdown 来源里代码样子的引号不换
    text = normalize_text(doc["text"], code=g == "tech" or url.endswith(".md"))
    text = normalize_punct(text) if doc.get("normalize_punct") else text
    text = LINK_RESIDUE.sub("", text)
    cands = []
    for run in text.split("\x1e"):  # harvest.RUN_SEP：markdown 第三版在代码块、引用块、标题处断开，切段不跨断点
        paras = paragraphs(run, keep_list=g in LIST_OK)
        paras = drop_heads(paras) if g == "news" else paras
        cands += chunks(paras, hi=hi)
    for k, c in enumerate(cands):
        r = rule_check(c, g, md=str(doc.get("url") or "").endswith(".md"))
        if r:
            if why is not None:
                why[r] = why.get(r, 0) + 1
            continue
        h = dedupe_key(c)
        if h in seen:
            if why is not None:
                why["重复"] = why.get("重复", 0) + 1
            continue
        seen.add(h)
        out.append({"id": f"{doc['id']}#{k}", "doc": doc["id"], "source": doc["source"], "genre_hint": g,
                    "date": doc.get("date"), "author": doc.get("author"), "title": doc.get("title"), "url": doc.get("url"), "text": c})
        if len(out) >= per_doc:
            break
    return out
