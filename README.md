# Verna

把带 AI 腔的中文改写成好中文，不增减信息，也不改变原来的文体和语气。

名字来自 vernacular（白话）。先做中文，以后做英文。

## 为什么做

大模型在训练时，自动打分程序会反复给它打分，写出来的中文也就带上了打分程序的偏好。最常见的毛病有两种：

- 动不动就用对比句。“不是 A，而是 B”本来用在听者真会以为 A 的时候，模型却到处用，前半句没有信息。五个商用模型使用这个句式的密度，是人类写作的 2.5–9.2 倍（[arXiv 2610.22249](https://arxiv.org/abs/2610.22249)）。
- 把词句压得太短。为了缩短输出，模型会把“已经核查”写成“已核”，“奖励”写成“奖”。训练时的长度惩罚会让模型一律选更短的句式（[arXiv 2610.29162](https://arxiv.org/abs/2610.29162)）。

用提示词让大模型“去 AI 味”，它往往换一套新的套话。Verna 的做法是训练专门的小模型，评判标准来自真人评审，规则跟着评审结果修订。

## 组成

| 部分 | 目录 | 做什么 |
|---|---|---|
| 评委站 | `vote/` | https://verna.143143.eu.org 。评委读一段 AI 写的原文和两份改写，选更愿意读的那份。设计见 [vote/DESIGN.md](vote/DESIGN.md) |
| 核心库 | `srh/` | 放评判准则、AI 腔规则、事实核对、提示词和模型调用，只用标准库 |
| 数据管线 | `autopilot/`、`scripts/` | 收集人写的好文字，构造“AI 腔 → 好中文”的配对，再经过规则和模型裁判的多道检查。说明见 [docs/pipeline.md](docs/pipeline.md) |
| 语料库 | `corpus/` | 中英文语料来源清单，以及质量评估和清洗脚本（语料本身不在仓库里） |
| 训练 | `notebooks/` | 在 Colab 上用 LoRA 微调 Qwen3.5-9B 和 Gemma 4 12B，目标是在 24GB 内存的 Mac 上能跑 |
| 测试 | `tests/` | `python3 -m unittest tests.test_core` |

内部代号是“说人话”（shuorenhua），所以 Python 包叫 `srh`，部分脚本和服务名也沿用这个代号。

## 现状（2026-11）

- 训练数据有 6,929 对，包括打磨人写的文字、把人写的文字改坏再改回、保留原文、打磨 AI 稿。第二轮冒烟训练（smoke2）的 Qwen3.5-9B 版已经训完，正在评委站上和 Gemini 4 Neon 对比；Gemma 4 12B 版正在训练。
- 评委站的前后端都已经上线。每道题有三个选项：A 更好、原文更好、B 更好，也可以投稿 AI 写的原文。候选稿现在有约 1,400 段原文、12 个系统，还在生成。
- 评判标准从“像人写的”改为“读者更愿意读的”，以评委的判断为准。

## 本地运行

```bash
python3 -m unittest tests.test_core                    # 核心库测试
python3 -m http.server -d vote/site 8000               # 评委站前端（没有后端时读 vote/site/items.json）
```

后端见 `vote/server/README.md`。运行数据管线需要 Gemini（[Antigravity CLI](https://antigravity.google)）和兼容 OpenAI 的 DeepSeek 接口。配置样例在 `autopilot/config.example.json`，复制成 `autopilot/config.json` 再改，密钥放在 `autopilot/env` 里（`SRH_API_KEY_<主机名第一段大写>=...`）。

## 数据

仓库里没有任何语料、训练数据、候选稿和评审数据。参与开发如果需要这些数据，请联系维护者。

## 参与

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

仓库里的代码和文档以 [Apache License 2.0](LICENSE) 发布。
