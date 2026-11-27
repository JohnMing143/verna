# 数据管线说明（v1–v3 的演进和目录）

对外名字 Verna（来自 vernacular，白话、口语）；内部代号“说人话”（shuorenhua），目录、systemd 单元、Python 包 `srh`、备份文件夹都沿用代号不改。目标是中英文都做，先做中文。

把中文改得更好读：信息不增不减，文体和作者的语气不变。

- 第一版方案（风险、评测方法、代码结构）：https://sga.143143.eu.org/reports/shuorenhua-plan/
- 第二版（底座三选一、用 Gemini 4 Argon 按文体打磨语料、agy 试点数据）：https://sga.143143.eu.org/reports/shuorenhua-v2/
- 第三版：无人值守数据管线，进度页 https://sga.143143.eu.org/reports/shuorenhua-progress/

## v3 无人值守管线（当前在跑的）

以优秀的人写文字为核心范本，宁缺毋滥。用户休息时自己跑，到需要判断的节点用 `claude -p` 唤醒 Claude。

```bash
bash autopilot/ctl.sh start          # systemd 托管（单元 srh-autopilot），崩溃自动重启
bash autopilot/ctl.sh status | log | pause | resume | stop
```

- 配置 `autopilot/config.json`：来源、分级阈值、配对目标、额度下限（周额度剩 30% 就停，留给日常使用）、每天最多唤醒 6 次
- 评判准则 `srh/principles.md`：综合经典文风批评（胡适、毛泽东、余光中、思果、吕叔湘朱德熙、王力、叶圣陶、老舍、汪曾祺）和网上关于 AI 味的讨论（只收多个独立来源提到的特征），按文体区分
- 流程：收集（知乎 2022 前高赞、CSL 期刊摘要、人民日报 1995–2022、THUCNews、维基文库公版白话、阮一峰周刊、COIG-CQIA）→ 切段规则 → Argon medium 批量分级 → Argon high 挑毛病复审（两轮都判范本才算）→ 反向样本 / 保留样本 / 打磨 → Argon high 抽检 → 老师口头禅统计 → 训练集快照 `data/v3/sft/`
- 产出都在 `data/v3/`，只追加；Claude 被唤醒时的记录在 `data/v3/autopilot/decisions.md`，需要你决定的事在 `for_user.md`，原文抽样在 `samples.md`
- 唤醒节点：新来源分级后复查、老师口头禅清单、抽检不合格率超 20%、同一环节连续报错 3 次、每日小结

## 目录

```
srh/            核心库（只用标准库）
  tells.py      AI 腔词表、禁用句式（HARD_BAN）、句长变化
  facts.py      数字/英文/书名/引语的规则核对
  prompts.py    所有提示词：小模型指令、造稿风格、裁判
  llm.py        OpenAI 兼容客户端
  runner.py     并发 + 断点续跑
scripts/        按编号顺序跑
notebooks/      train_sft.ipynb（由 build_notebook.py 生成，改 notebook 改那个文件）
data/           native_prompts.txt、baseline_instruction.txt、样本数据
tests/          python3 -m unittest tests.test_core
```

## v2 数据管线（当前方案）

答案端以质量为准，不再限定 ChatGPT 之前的人写文字。老师和裁判都走本机官方 agy CLI（`agy:<模型>`），不用 GPT 系模型。

```bash
export SRH_USAGE_LOG=data/usage.jsonl          # 每次调用的 token 用量，核算额度用
T=agy:gemini-4-argon-medium
# 1. 按 srh/genres.py 的配比出题，多家模型写 AI 初稿
python3 scripts/make_inputs.py tasks --total 300 --model $T --out data/tasks.jsonl
python3 scripts/make_inputs.py drafts --tasks data/tasks.jsonl --out data/ai_drafts.jsonl \
    --writers "agy:gemini-3.8-flash-low,<模型>@https://your-gateway.example.com/v1"
# 2. 人写文字（任何年代）转成输入
.venv/bin/python scripts/collect_human.py --source zhihu --limit 300 --out data/human_raw.jsonl
python3 scripts/make_inputs.py human --src data/human_raw.jsonl --out data/human_inputs.jsonl
# 3. 打磨：分类 → 3 个候选 → 规则 → 裁判 → 返修
cat data/ai_drafts.jsonl data/human_inputs.jsonl > data/inputs.jsonl
python3 scripts/distill.py --in data/inputs.jsonl --out data/distilled.jsonl
# 4. 反向样本：好的人写文字改坏，再核对
python3 scripts/make_inputs.py degrade --src data/distilled.jsonl --out data/degraded.jsonl --model $T
python3 scripts/verify_pairs.py --in data/degraded.jsonl --checked data/degraded_checked.jsonl --out data/reverse_pairs.jsonl --judge $T
# 5. 老师口头禅：统计后人工挑进 data/teacher_tells.txt，之后的 distill 会淘汰新出现这些说法的候选
.venv/bin/python scripts/slop_profile.py --teacher data/distilled.jsonl --human data/human_raw.jsonl
# 6. 训练集
python3 scripts/build_sft.py --pairs data/distilled.jsonl data/reverse_pairs.jsonl --outdir data/sft
```

`data/pilot2/` 是 11-04 的 12 条试点（含 4 条 GPT-OSS 写的初稿，之后不再用 GPT 系）。

## v1 数据管线（反向造数据，保留作参考）

模型参数统一写成 `模型名@base_url`，key 放环境变量 `SRH_API_KEY`（多家时用 `SRH_API_KEY_<主机名第一段大写>`，比如 `SRH_API_KEY_API` 对应 api.deepseek.com）。

```bash
python3 -m venv .venv && .venv/bin/pip install datasets      # 只有第 1 步需要
J=deepseek-chat@https://api.deepseek.com/v1                   # 裁判
G="$J,gemini-xxx@https://generativelanguage.googleapis.com/v1beta/openai,qwen-xxx@..."  # 造稿模型，至少三家

# 试点：先 300 段，看通过率再放大
.venv/bin/python scripts/collect_human.py --source zhihu --limit 1000 --out data/human_raw.jsonl
.venv/bin/python scripts/collect_human.py --source local:~/my_articles --out data/human_local.jsonl   # 可选
python3 scripts/select_human.py --in data/human_raw.jsonl --scored data/human_scored.jsonl --out data/human.jsonl --judge $J
python3 scripts/aiify.py --in data/human.jsonl --out data/drafts.jsonl --models "$G"
python3 scripts/verify_pairs.py --in data/drafts.jsonl --checked data/drafts_checked.jsonl --out data/pairs.jsonl --judge $J
python3 scripts/build_sft.py --pairs data/pairs.jsonl --outdir data/sft
python3 scripts/make_native_test.py --models "$G" --out data/native_test.jsonl
python3 scripts/pack_for_colab.py        # 生成 shuorenhua-colab.zip，解压到 Drive 的 MyDrive/shuorenhua/
```

（v1 旧做法。现在训练数据和 notebook 由 `autopilot/backup.py` 每 6 小时自动同步到 Drive 的 `MyDrive/verna-train/`，notebook 改完用 `.venv/bin/python autopilot/backup.py --notebook` 立即推送。）Colab 打开 `verna-train/train_sft.ipynb`，全部运行。训完 notebook 会把预测和运行信息盖写到 `verna-train/inbox/<底座>.zip`（服务器的 rclone 是 drive.file 权限，看不到 Colab 新建的文件，只能读自己建的占位文件），服务器用 `.venv/bin/python scripts/pull_results.py` 拉到 `runs/<RUN>/<完成时间>/`（备份每 6 小时也顺带拉一次；占位文件丢了用 `--init` 重建）。GGUF 在 Drive 的 `runs/<RUN>/gguf/`，自己下载。

## 用和评测

```bash
llama-server -m verna-qwen3.5-9b-v1-q8_0.gguf -c 8192 --port 8080 --jinja
python3 scripts/rewrite.py --model verna@http://127.0.0.1:8080/v1 < draft.md > out.md
# Gemma 4 底座用它的推荐采样：--temps 1.0 --top-p 0.95（llama-server 加 --top-k 64）

# 对照组：同一批输入
python3 scripts/rewrite.py --model verna@http://127.0.0.1:8080/v1 --raw --in data/sft/test.jsonl --out preds/verna-9b_raw.jsonl
python3 scripts/rewrite.py --model verna@http://127.0.0.1:8080/v1 --in data/sft/test.jsonl --out preds/verna-9b_guard.jsonl
python3 scripts/rewrite.py --model $J --instruction data/baseline_instruction.txt --raw --in data/sft/test.jsonl --out preds/bigmodel.jsonl
python3 scripts/eval.py preds/*.jsonl --judge $J --n 200
```

jialinyyzz/humanizer 做对照时加 `--instruction none --temps 1.0 --top-p 0.95`（它的模型卡要求温度 1.0、top-p 0.95；llama-server 的 top-k、min-p 默认值也要在启动参数里关掉：`--top-k 0 --min-p 0`）。

## 状态（v0.2.0，2026-11-04）

- v2：make_inputs / distill / slop_profile 已用 agy 真跑过 12 条试点（11/12 通过）
- 底座：H100 / A100 80GB 训练、24GB Mac 推理，Qwen3.5-9B 与 Gemma 4 12B 实测二选一（Qwen3.5-4B 作参照）；notebook 第二格改 `CHOICE` 切换，没在 GPU 上跑过

### v0.1.0

- tells / facts：单元测试通过；collect_human 在真实知乎数据上跑过 3,000 段
- 其余脚本用本地模拟接口跑通全流程，还没接真实模型
- notebook 语法检查通过，还没在 GPU 上跑过
