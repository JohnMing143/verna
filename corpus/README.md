# Verna 语料库

和训练管线分开的原始语料归档。目标是给 Verna（中文、以后英文）和以后类似的项目留一份**来源清楚、问题写明**的人写语料。原则还是宁缺毋滥：收进来不等于能用，每个来源都要评估，结论写清楚。

## 放在哪

- 原文：Google Drive `verna-corpus/<语言>/<名字>/`（rclone remote `gcorpus:`），按来源原样存放（Hugging Face 的 parquet/jsonl、维基百科的 bz2、Stack Exchange 的 7z），不改动。
- 本目录（服务器 `/root/shuorenhua/corpus/`）只放元数据和脚本，不放原文：
  - `sources.json`：来源清单（地址、许可证、能否再分发、时间范围、AI 风险）。`enabled: false` 的是候选。
  - `state/files.jsonl`：每个已归档文件一行（Drive 路径、大小、md5、sha256、来源提交号、时间）。
  - `samples/<名字>.jsonl`：每个来源抽 300 条，评估用（Drive 上也有一份，在 `_samples/`）。
  - `reviews.json`：评估结论。
  - `CATALOG.md`、`manifest.json`：自动生成的总目录，同步到 Drive 根目录。
- 下载校验：Hugging Face 的文件和 HF 记录的 sha256 对；archive.org 的文件和它公布的 md5 对；传到 Drive 后再用 Drive 算的 md5 核一遍。

## 状态（reviews.json）

| 状态 | 含义 | 能不能用 |
|---|---|---|
| `pending_review` | 收了，还没评估 | 不能 |
| `clean` | 评估过：人写、质量够、没有系统性问题；写明常规处理步骤（比如按日期截断、去 HTML） | 按写明的步骤处理后能用 |
| `flagged` | 有价值，但有具体问题（AI 混入、机翻、乱码、模板化、版权不清……）。`issues` 列问题，`cleaning` 写清洗要求 | **不能直接用**。清洗产物登记成新条目（`<名字>.clean`），复核为 `clean` 后才能用 |
| `rejected` | 评估后不用，`summary` 写理由 | 不能 |

每条评估记录：`status`、`summary`（一两句结论）、`issues`（问题列表）、`cleaning`（清洗要求）、`checked`（看了什么：抽样条数、统计）、`reviewed_at`。

**不能因为来源有价值就略过它的问题。** 比如 Stack Exchange 质量高，但正文是 HTML、夹着代码、低分回答多，所以是 `flagged`：写明“去 HTML、按分数筛、代码块单独处理”，清洗后复核才能用。

## 怎么跑

```bash
bash corpus/ctl.sh start      # 后台收集（systemd 单元 srh-corpus，低优先级），做完自动退出；可重复执行，已完成的文件跳过
bash corpus/ctl.sh status     # 看进度
bash corpus/ctl.sh stop
.venv/bin/python corpus/collect.py --catalog   # 改了 reviews.json 后重新生成目录并上传
```

本地磁盘：下载的临时文件放 `corpus/work/`，单批不超过 4GB，剩余空间低于 20GB 就等待（训练管线低于 10GB 停收集）。`corpus/work` 和 `corpus/samples` 不进加密备份。

评估时看长文：书、年报这类开头全是版权页或法定声明，抽样时每个超过 7500 字的字段另存一段中间的 `_mid_<字段>`，`profile.py --mid` 用中段统计和摘录。

## 清洗（corpus/clean.py，11-06 起）

flagged 的来源按 reviews.json 里写的清洗要求，在 `clean.py` 里写一个清洗函数（加进 `CLEANERS`，`steps` 写清每一步），然后：

```bash
.venv/bin/python corpus/clean.py zh.thucnews --dry --keep-work   # 只出统计和样本，不上传；先抽查
.venv/bin/python corpus/clean.py zh.thucnews                     # 正式：上传、登记、删本地临时文件
```

产物登记成新来源 `<名字>.clean`：文件在 `verna-corpus/<语言>/<名字>.clean/`（含 `clean_report.json`），sources.json 里 `method: derived`，reviews.json 里先是 `pending_review`；`clean_reports/<名字>.clean.json` 记每条规则去掉了多少。复核（profile.py 看样本、逐条核对评估时列的问题）后用 review.py 定成 clean 或 flagged。原始文件不动。

sources.json 是手工排版的，脚本只按文本在列表末尾插条目，不整体重写。

经验：第一版规则常常删多了（知乎的“≤20 字无标点短行＝图片说明”误删了大量正文；WritingPrompts 的“Thanks for…/First time…＝Reddit 附言”把对白也删了），一定先 dry 跑、看被删掉的内容，再正式跑。字段名对不上的数据也别想当然：MNBVC 外交部问答所谓“问答对调”其实是整段错位（“问”里是上一题的回答），对调字段反而配错。

## Drive 限流和自建 client_id（11-05 14:30 已切换）

rclone 默认用全体 rclone 用户共享的 Google 项目（project 202264815644），每分钟请求数常被挤满：小文件还能挤进去，大文件上传到最后一步被 403 rateLimitExceeded 拒掉，然后整份重传（11-05 实测 arXiv 0.94GB 传了 3 遍）。备份和训练同步请求少，影响不大；语料库请求多，需要自己的项目配额。

计划：新建一个只给语料库用的 remote `gcorpus:`（自己的 OAuth client_id，仍是 drive.file 权限），归档目录改到它建的 `verna-corpus/`。drive.file 只能看到本应用建的文件，所以：
- 现有的 `gdrive:`（备份 shuorenhua-backup、训练 verna-train）保持不动；
- 旧 client 已经传到 `gdrive:verna-corpus/` 的 26 个文件已用 `gcorpus:` 重传，`rclone check` 26 个全部一致，旧目录已删（在 Drive 回收站）。
- collect.py 默认 `gcorpus:verna-corpus`，可用环境变量 VERNA_CORPUS_REMOTE 覆盖。
- 用户的 Google Cloud 项目里只用 drive.file 权限；实测 300MB 上传无限流。
