import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "autopilot"))
sys.path.insert(0, str(ROOT / "scripts"))

from srh import facts, harvest, prompts, segment, tells  # noqa: E402
import distill  # noqa: E402
import run as ap  # noqa: E402


class TestNumbers(unittest.TestCase):
    def n(self, s):
        return facts.numbers(s)[0]

    def test_arabic_and_units(self):
        self.assertEqual(self.n("营收 1,200 万元，增长 30%"), {"1.2e+07", "pct:30"})
        self.assertEqual(self.n("1.2万人"), {"12000"})
        self.assertEqual(self.n("12000人"), {"12000"})

    def test_chinese(self):
        self.assertEqual(self.n("两万多人参加"), {"20000"})
        self.assertEqual(self.n("百分之三十的人"), {"pct:30"})
        self.assertEqual(self.n("三成用户"), {"pct:30"})
        self.assertEqual(self.n("二〇二三年"), {"2023"})
        self.assertEqual(self.n("一百二十个"), {"120"})
        self.assertEqual(self.n("增长了三点五倍"), {"3.5"})

    def test_idioms_ignored(self):
        self.assertEqual(self.n("千万别这样，万万没想到，一样的"), set())
        self.assertEqual(self.n("三心二意"), set())

    def test_structural_soft(self):
        hard, soft = facts.numbers("主要有三点原因。第二，成本高。")
        self.assertEqual(hard, set())
        self.assertIn("3", soft)

    def test_list_marker_ignored(self):
        self.assertEqual(self.n("1. 先烧水\n2. 再下面"), set())


class TestCompare(unittest.TestCase):
    def test_equivalent(self):
        a = "2023年公司营收增长了30%，达到1.2亿元。张三说“这是十年来最好的一年”。"
        b = "公司 2023 年收入 1.2 亿，涨了百分之三十。张三的原话是：“这是十年来最好的一年”。"
        r = facts.compare(a, b)
        self.assertTrue(r["ok"], r)

    def test_number_changed(self):
        r = facts.compare("增长了30%", "增长了三成多，接近35%")
        self.assertFalse(r["ok"])
        self.assertTrue(any("35" in x for x in r["hard"]))

    def test_latin_and_title(self):
        r = facts.compare("用 GPT-4 读《三体》", "用大模型读《三体》")
        self.assertFalse(r["ok"])
        self.assertIn("丢了英文/型号 gpt-4", r["hard"])

    def test_title_marks_added_ok(self):
        self.assertTrue(facts.compare("辛丑条约第二条", "《辛丑条约》第二条")["ok"])

    def test_pilot_false_positives(self):
        self.assertEqual(facts.numbers("千头万绪终究化作沉默")[0], set())
        self.assertEqual(facts.numbers("和十一国宣战")[0], {"11"})
        self.assertTrue(facts.compare("别一口气十一个国家全宣了", "别十一国全宣啊")["ok"])

    def test_scare_quote_soft(self):
        r = facts.compare("所谓的“内卷”其实是", "所谓内卷其实是")
        self.assertTrue(r["ok"], r)
        r = facts.compare("所谓的“内卷”其实是", "其实是")
        self.assertTrue(r["ok"])
        self.assertTrue(r["soft"])


class TestTells(unittest.TestCase):
    AI = (
        "在当今快速发展的时代，人工智能正在深刻改变我们的生活。值得注意的是，这不仅仅是技术的进步，更是思维方式的革命。"
        "首先，它赋能了各行各业。其次，它打通了信息孤岛。最后，它构建了全新的生态闭环。"
        "总的来说，AI 不是洪水猛兽，而是我们的朋友。让我们一起拥抱这个充满无限可能的未来！"
    )
    HUMAN = (
        "我第一次用 ChatGPT 是 2022 年底，让它帮我写周报。写出来四平八稳，可老板看了一眼就问我是不是换了人。"
        "后来我只让它列提纲，句子自己写。省下的时间不多，大概每周半小时，但周报总算又像我写的了。"
    )

    def test_density_gap(self):
        self.assertGreater(tells.density(self.AI), 20)
        self.assertLess(tells.density(self.HUMAN), 3)

    def test_named_hits(self):
        h = tells.report(self.AI)["hits"]
        for k in ("在当今时代", "值得注意的是", "不仅仅是更是", "首先其次最后", "总的来说", "让我们", "不是而是", "黑话"):
            self.assertIn(k, h)


class TestLateRules(unittest.TestCase):
    """11-05 来源复查加的切段规则。"""

    def test_link_residue(self):
        self.assertEqual(segment.LINK_RESIDUE.sub("", "首开国风上观 (论坛 相册 户型 样板间 地图搜索)位于望京，李晨( blog)说"), "首开国风上观位于望京，李晨说")
        self.assertEqual(segment.late_check("姚明(微博)昨天表示，球队状态不错。他说还要努力。"), "网页残留")
        self.assertIsNone(segment.late_check("北京（中国首都）今天下雨。气温不高，路上人不多。"))

    def test_template_and_colon(self):
        self.assertEqual(segment.late_check("当然可以！以下是一篇关于《X》的书评：\n这本书写得好。读完很受启发。"), "模板开场白")
        self.assertEqual(segment.late_check("他想了很久。最后他说："), "冒号收尾")

    def test_dangling(self):
        tech = "Promise 有三种状态。下面是一个例子。\n如果服务器想改变一个 Cookie，必须同时匹配四个值。"
        self.assertEqual(segment.late_check(tech, "tech"), "引用悬空")
        self.assertIsNone(segment.late_check(tech, "news"))  # 非技术、非 markdown 来源不查
        ok = "直接看下面的一个小例子。\n比如，最简单的场景是连续调用两次。这样就清楚了。"
        self.assertIsNone(segment.late_check(ok, "tech"))
        quote = "网上找到的解释：\n看了这个定义，我还是不太明白。为什么叫这个名字？"
        self.assertEqual(segment.late_check(quote, None, md=True), "引用悬空")
        self.assertIsNone(segment.late_check("松居大悟：\n我很清楚当下的社会语境。所以一开始就讨论过。", None))

    def test_md_and_lists(self):
        md = "运算符有下面3个。\n\n- `>` 大于\n- `<` 小于\n- `==` 相等\n\n用法很简单。下面是一个例子。\n\n```js\n1 > 2\n```\n\n上面代码返回 false。\n\n结束了，这一节就讲到这里。"
        t = harvest._md_to_text(md)
        self.assertNotIn("下面是一个例子", t)
        self.assertNotIn("上面代码", t)
        self.assertEqual(len([p for p in segment.paragraphs(t, keep_list=True) if p.startswith("- ")]), 3)
        self.assertEqual(len([p for p in segment.paragraphs(t) if p.startswith("- ")]), 0)

    def test_added_clauses(self):
        orig = "网上找到的解释：\n看了这个定义，我还是不太明白。"
        new = "网上找到的解释：内卷化是指一种社会模式发展到一定阶段后停滞不前的现象。\n看了这个定义，我还是不太明白。"
        self.assertTrue(facts.added_clauses(orig, new))
        self.assertEqual(facts.added_clauses("他在这方面已经做了大量的工作。", "他在这方面已做了大量工作。"), [])

    def test_md_v3_runs(self):  # 11-05 13:43 复查 wangdoc2
        md = ("## 参数\n\n函数体内可以使用参数变量，获取函数参数。\n\n- `$1`~`$9`：第一个到第9个参数。\n- `$*`：全部参数，用`$IFS`分隔。\n\n"
              "范围是 -2<sup>53</sup> 到 2<sup>53</sup>，超出就不精确。\n\nBash 函数定义的语法有两种。\n\n```bash\nfn() {}\n```\n\n"
              "上面代码中，fn 是函数名。\n\n## 删除\n\n删除函数用unset命令，查看函数用declare命令。")
        t = harvest._md_to_text(md, runs=True)
        self.assertIn("$1~$9", t.replace("`", ""))
        self.assertIn("$IFS", t)
        self.assertIn("2^53", t)
        self.assertNotIn("语法有两种", t)  # 紧挨代码块的前一段
        self.assertNotIn("上面代码", t)    # 紧挨代码块的后一段
        self.assertEqual(t.count(harvest.RUN_SEP), 1)  # 代码块和后面的标题只算一个断点
        self.assertIn("$1", harvest._md_to_text(md))  # 不开 runs 也不再吃掉 $
        self.assertEqual(segment.late_check("如果不指定参数类型（比如上例不写txt的类型），就推断为any。\n返回值类型通常可以不写。", "tech", md=True), "代码指代")
        self.assertIsNone(segment.late_check("如果不指定参数类型，就推断为any。\n返回值类型通常可以不写。", "tech", md=True))

    def test_speech_residue(self):
        body = "\n今天，我们在这里举行座谈会，纪念他诞辰一百周年。他的一生是奋斗的一生。"
        for head in ("胡锦涛同志们：", "中华人民共和国国务院总理李克强尊敬的吴登盛总统，各位同事：", "（一九九六年六月十四日）各位常委、各位同志："):
            self.assertEqual(segment.late_check(head + body, "speech"), "署名残留", head)
        for head in ("同志们：", "马哈蒂尔总理阁下，各位同事：", "教师们、同学们、同志们：", "尊敬的各国工会朋友们，", "尊敬的主席先生，各位同事："):
            self.assertIsNone(segment.late_check(head + body, "speech"), head)
        self.assertEqual(segment.late_check("同志们：" + body + "\n谢谢大家。（新华社北京9月16日电）", "speech"), "电头残留")

    def test_code_quotes_and_archives(self):  # 11-05 16:36 复查 wangdoc3
        n = segment.normalize_text
        for code in ('比如class="right"属性', '写成scanf(" %c", &ch)，即', 'from "./b"，那么', 'String a = "hi"; String b'):
            self.assertEqual(n(code, code=True), code)
        self.assertEqual(n('称为"闭包"的写法', code=True), "称为“闭包”的写法")  # 正文引用照样换
        self.assertEqual(n('比如class="right"属性'), "比如class=“right”属性")  # 非技术来源不变
        body = "这样写可以排除文件。\n另一种写法也能达到同样的效果。"
        for bad in ("比如--exclude=“*.iso”。", "可以写成scanf(“ %c”, &ch)。", "一行代码import { b } from “./b”，就会查找b.ts。"):
            self.assertEqual(segment.late_check(bad + body, "tech", md=True), "代码引号", bad)
            self.assertIsNone(segment.late_check(bad + body, "news"), bad)
        for ok in ("在 VS Code 中安装插件“WSL”后可以用。", "沿着合生创展(00754.HK，下称“合生”)母公司。", "他说：“我需要钱。”"):
            self.assertIsNone(segment.late_check(ok + body, "tech", md=True), ok)
        seg = {"text": "Bash脚本有时候需要同时执行多个任务。通常这涉及到启动一个脚本。", "genre_hint": "tech",
               "url": "https://github.com/wangdoc/bash-tutorial/blob/HEAD/docs/archives/async.md"}
        self.assertEqual(segment.late_check_seg(seg), "存档旧稿")
        self.assertEqual(segment.segment_doc({**seg, "id": "x", "source": "x", "text": seg["text"] * 20}), [])
        self.assertNotEqual(segment.late_check_seg({**seg, "url": seg["url"].replace("archives/", "")}), "存档旧稿")
        self.assertEqual(segment.late_check("故需按大写 K（++“Shift”++ + ++“k”++）键显示。" + body, "tech", md=True), "标记残留")
        self.assertEqual(segment.late_check("各种常见操作。\n{: .caption }\n" + body, "tech", md=True), "标记残留")
        self.assertIsNone(segment.late_check("正如 C++ 和 Modern C++，C 语言也在变。" + body, "tech", md=True))


class TestStripAssistantMeta(unittest.TestCase):
    """11-10 加：AI 初稿里助手自己的开场白和结尾附言要去掉，正文里的“如需”不能误删。"""

    def test_opener_and_separator(self):
        raw = "好的，以下是改写后的内容：\n他走进屋子，四下看了看。\n\n---\n\n**字数**：约 320 字"
        self.assertEqual(prompts.strip_assistant_meta(raw), "他走进屋子，四下看了看。")

    def test_paren_tail(self):
        raw = "他走进屋子，四下看了看。\n\n(约 330 字,如需调整语气或篇幅,随时告诉我即可。)"
        self.assertEqual(prompts.strip_assistant_meta(raw), "他走进屋子，四下看了看。")

    def test_star_tail(self):
        raw = "他走进屋子。\n\n* **字数**：约 320 字"
        self.assertEqual(prompts.strip_assistant_meta(raw), "他走进屋子。")

    def test_body_ruxu_kept(self):
        raw = "办理流程如下。\n如需发票，请在付款时备注。"
        self.assertEqual(prompts.strip_assistant_meta(raw), raw)

    def test_body_zishu_kept(self):
        raw = "老师提醒我们，这篇论文的字数还不够。"
        self.assertEqual(prompts.strip_assistant_meta(raw), raw)

    def test_plain_text_unchanged(self):
        raw = "他走进屋子，四下看了看。\n桌上放着一只杯子。"
        self.assertEqual(prompts.strip_assistant_meta(raw), raw)


class TestAiLocalCheck(unittest.TestCase):
    """11-10 加：AI 初稿版的规则核对（相似度、长度、禁用说法）。"""

    SRC = "昨天下午三点，团队在会议室讨论了下季度的排期，最后决定先做客服工具，把报表往后放。"

    def test_same_text_too_similar(self):
        issues = distill.local_check_ai(self.SRC, self.SRC, "light", [])
        self.assertIn("改动太少", issues)

    def test_too_short(self):
        issues = distill.local_check_ai(self.SRC, "昨天下午开会，决定先做客服工具。", "heavy", [])
        self.assertTrue(any(x.startswith("长度比") for x in issues), issues)
        self.assertNotIn("改动太少", issues)

    def test_banned_left(self):
        cand = "团队昨天下午讨论了排期，先做客服工具。这不是效率问题，而是优先级问题。"
        issues = distill.local_check_ai(self.SRC, cand, "heavy", [])
        self.assertIn("还留着禁用说法 不是而是", issues)

    def test_strips_meta_before_compare(self):
        src = self.SRC + "\n\n(约 40 字，如需调整请告诉我。)"
        cand = "昨天下午的会上，团队排了下季度的事，决定先做客服工具，报表往后放。"
        issues = distill.local_check_ai(src, cand, "heavy", [])
        self.assertFalse(any(x.startswith("丢了数字") or x.startswith("多了数字") for x in issues), issues)


class TestAiFamily(unittest.TestCase):
    """11-10 加：AI 腔 family 按 id 的 crc32 选，第二次尝试换下一个。"""

    def test_deterministic(self):
        self.assertEqual(ap._ai_family("zhihu:1#0", "essay"), ap._ai_family("zhihu:1#0", "essay"))

    def test_genre_restricted(self):
        fams = prompts.AI_STYLE_BY_GENRE["fiction"]
        f0 = ap._ai_family("zhihu:1#0", "fiction", 0)
        f1 = ap._ai_family("zhihu:1#0", "fiction", 1)
        self.assertIn(f0, fams)
        self.assertIn(f1, fams)
        self.assertNotEqual(f0, f1)

    def test_other_genres_all_five(self):
        self.assertIn(ap._ai_family("csl:1#0", "tech"), prompts.AI_STYLE_FAMILIES)


class TestPlainReverse(unittest.TestCase):
    def test_excludes_revai(self):
        rows = [{"id": "a#rev", "genre": "news"}, {"id": "b#revai", "genre": "news"}, {"id": "c#rev2", "genre": "opinion"}]
        self.assertEqual([r["id"] for r in ap._plain_reverse(rows)], ["a#rev", "c#rev2"])


class TestAiPilotGating(unittest.TestCase):
    """11-10 加：pilot_ok=false 时 #revai 和 distill_ai2 不进快照；exclude_v1 时旧 AI 线不要。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "pairs").mkdir()
        (self.dir / "autopilot").mkdir()
        self.write("pairs/reverse.jsonl", [
            {"id": "doc1#0#rev", "doc": "doc1", "kind": "reverse", "genre": "news", "source": "s", "draft": "d", "original": "o"},
            {"id": "doc2#0#revai", "doc": "doc2", "kind": "reverse", "genre": "news", "source": "s", "draft": "d2", "original": "o2"},
        ])
        self.write("pairs/distill_ai.jsonl", [{"id": "a1", "status": "ok", "kind": "ai_draft", "genre": "news",
                                               "input": "i", "target": "t"}])
        self.write("pairs/distill_ai_v2.jsonl", [{"id": "b1#ai2", "status": "ok", "kind": "ai_draft", "genre": "news",
                                                  "input": "i2", "target": "t2"}])
        self.write("pairs/reverse_raw.jsonl", [{"id": "doc1#0#rev", "judge_v": ap.REVERSE_JUDGE_V, "ok": True},
                                               {"id": "doc2#0#revai", "judge_v": ap.REVERSE_JUDGE_V, "ok": True}])
        self.cfg = json.loads((ROOT / "autopilot" / "config.json").read_text())
        self.cfg["audit"]["recheck"] = []
        self.set_pilots(revai=False, ai2=False, exclude_v1=False)
        self.files = {k: v for k, v in ap.F.items()}
        for k, name in (("reverse", "reverse.jsonl"), ("keep", "keep.jsonl"), ("reverse_raw", "reverse_raw.jsonl"),
                        ("reverse_check", "reverse_check.jsonl"), ("audit", "audit.jsonl"),
                        ("distill_check", "distill_check.jsonl"), ("keep_edit", "keep_edit.jsonl"),
                        ("keep_fact", "keep_fact.jsonl"), ("distill_human", "distill_human.jsonl"),
                        ("distill_ai", "distill_ai.jsonl"), ("distill_ai2", "distill_ai_v2.jsonl")):
            self.files[k] = self.dir / "pairs" / name
        self.files["control"] = self.dir / "autopilot" / "control.json"
        self.patches = [mock.patch.dict(ap.F, self.files), mock.patch.object(ap, "CFG", self.dir / "config.json"),
                        mock.patch.object(ap, "AP", self.dir / "autopilot")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def write(self, rel: str, rows: list[dict]):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def set_pilots(self, revai: bool, ai2: bool, exclude_v1: bool):
        self.cfg["pairs"]["reverse_ai"]["pilot_ok"] = revai
        self.cfg["pairs"]["ai2"].update(pilot_ok=ai2, exclude_v1=exclude_v1)
        (self.dir / "config.json").write_text(json.dumps(self.cfg, ensure_ascii=False))

    def ids(self, hold: bool) -> set:
        return {p["id"] for p in ap.all_pairs(hold=hold)}

    def test_pilot_off_hides_new_lines(self):
        ids = self.ids(hold=True)
        self.assertIn("doc1#0#rev", ids)
        self.assertNotIn("doc2#0#revai", ids)
        self.assertNotIn("b1#ai2", ids)
        raw = self.ids(hold=False)  # 补查/生成用 hold=False，要能看见
        self.assertIn("doc2#0#revai", raw)
        self.assertIn("b1#ai2", raw)

    def test_pilot_on_includes(self):
        self.set_pilots(revai=True, ai2=True, exclude_v1=False)
        ids = self.ids(hold=True)
        self.assertIn("doc2#0#revai", ids)
        self.assertIn("b1#ai2", ids)
        self.assertIn("a1", ids)

    def test_exclude_v1(self):
        self.set_pilots(revai=True, ai2=True, exclude_v1=True)
        ids = self.ids(hold=True)
        self.assertNotIn("a1", ids)
        self.assertIn("b1#ai2", ids)


if __name__ == "__main__":
    unittest.main()
