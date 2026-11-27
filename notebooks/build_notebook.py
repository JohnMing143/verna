#!/usr/bin/env python3
"""生成 train_sft.ipynb（改 notebook 就改这个文件再跑一遍，免得手改 JSON）。"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})


md("""
# Verna：SFT（LoRA）

中文改写模型（内部代号“说人话”）：把 AI 腔的中文改得像人写的，信息不增不减，文体和语气不变。

运行环境：Colab **H100 或 A100 80GB**（运行时 → 更改运行时类型），A100 40GB 也能跑，批量会自动减半。T4 不支持 bf16，Qwen3.5 在 T4 上会退回 float32 训练，慢几倍。

底座在第一个代码格的 `CHOICE` 里切换（换底座要重启运行时再从头跑，装包命令跟着变）：Qwen3.5-9B、Gemma 4 12B、Qwen3.5-4B。比较底座时用同一份数据、同样的超参数各跑一次，`RUN` 换个名字。

准备：不用手动传东西。服务器每 6 小时把最新的训练数据（`data/sft/`）、评测用的代码（`srh/`）、真实 AI 测试集（`data/native_test.jsonl`）和这个 notebook 同步到 Google Drive 的 `MyDrive/verna-train/`。在 Drive 里双击这个 notebook，用 Colaboratory 打开就行。

底座模型不放 Drive：第一次运行时从 Hugging Face 下载（9B 约 19GB，Colab 上几分钟），比从 Drive 读快。训练结果（checkpoint、LoRA、GGUF）写到 `MyDrive/verna-train/runs/<RUN>/`。

预计耗时（粗估，没实测，按 1 万条、平均 1k token）：A100 上 9B 两轮 2–3 小时；H100 会快不少。checkpoint 存在 Drive，断线后重跑全部单元格会接着训。
""")

code("""
# 先选底座，下一格按它装包（Qwen3.5 和 Gemma 4 的官方装包命令不一样）
CHOICE = "qwen3.5-9b"  # @param ["qwen3.5-9b", "gemma4-12b", "qwen3.5-4b"]
FAMILY = "gemma" if CHOICE.startswith("gemma") else "qwen"
# 训练或推理时报 torch 编译错误（inductor、UnboundLocalError: ... 'u0' 之类）就改成 False，再“重启运行时”从头跑
COMPILE = True  # @param {type:"boolean"}
import os
if not COMPILE:
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
""")

code("""
%%capture
# 装包命令分别抄自 Unsloth 官方 Qwen3_5_(4B) 和 Gemma4_(12B)_Text notebook（2026-11-05 核对）。
# 装不上时去 github.com/unslothai/notebooks 抄最新版本替换对应分支。
# 11-27：smoke2 用最新版 Unsloth 报 inductor 编译错误，版本全部钉在 smoke1 那天（11-10）PyPI 上的版本
import os, importlib.util
_UV = "2026.11.3"   # unsloth 和 unsloth_zoo
if FAMILY == "qwen":
    !pip install --upgrade -qqq uv
    if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
        try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
        except: _numpy = "numpy"; _pil = "pillow"
        !uv pip install -qqq \\
            "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision "torchaudio==2.8.0" bitsandbytes xformers==0.0.32.post2 \\
            "unsloth_zoo[base]=={_UV}" "unsloth[base]=={_UV}" "peft==0.21.2" "accelerate==1.15.0"
        !uv pip install -qqq --no-deps "torchcodec==0.7.0"
    elif importlib.util.find_spec("unsloth") is None:
        !uv pip install -qqq "unsloth=={_UV}"
    !uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 "unsloth=={_UV}" "unsloth_zoo=={_UV}" "peft==0.21.2" "accelerate==1.15.0"
    !uv pip install transformers==5.2.0
    !unsloth install-kernels
    !uv pip install --no-deps --upgrade "torchao>=0.16.0"
else:
    if "COLAB_" not in "".join(os.environ.keys()):
        !pip install "unsloth=={_UV}"
    else:
        !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
        !pip install --no-deps "unsloth_zoo=={_UV}" bitsandbytes "accelerate==1.15.0" "peft==0.21.2" "trl==1.17.1" triton "unsloth=={_UV}"
        !unsloth install-kernels
        !pip install --no-deps --upgrade "torchao>=0.16.0"
    !pip install --no-deps transformers==5.10.1 "tokenizers>=0.22.0,<=0.23.0"
    !pip install "huggingface_hub>=1.5.0,<2.0"
    import importlib.metadata as _md
    try: _tv = tuple(int(x) for x in _md.version("torch").split("+")[0].split(".")[:2])
    except Exception: _tv = (2, 11)
    _torchcodec = {(2, 11): "torchcodec>=0.11.0", (2, 10): "torchcodec==0.10.*", (2, 9): "torchcodec==0.9.*", (2, 8): "torchcodec==0.7.*", (2, 7): "torchcodec==0.5.*"}.get(_tv, "torchcodec>=0.12.0" if _tv >= (2, 12) else "torchcodec")
    !pip install "{_torchcodec}"
""")

code("""
from google.colab import drive
drive.mount("/content/drive")

WORK = "/content/drive/MyDrive/verna-train"   # 服务器自动同步的文件夹（rclone 建的）
MODELS = {  # 名字: (仓库, 家族, 80GB 卡上的批量, 梯度累积)；等效批量都是 16
    "qwen3.5-9b":  ("unsloth/Qwen3.5-9B",     "qwen",  8, 2),
    "gemma4-12b":  ("unsloth/gemma-4-12b-it", "gemma", 4, 4),
    "qwen3.5-4b":  ("unsloth/Qwen3.5-4B",     "qwen", 16, 1),
}
assert CHOICE in MODELS and MODELS[CHOICE][1] == FAMILY, "CHOICE 在第一个代码格改，改完要重启运行时、从头跑（装包命令跟着变）"
# LoRA 秩：显存充裕开到 64，想省就改成 32
LORA_R = 64  # @param {type:"integer"}
MODEL, FAMILY, BS, GA = MODELS[CHOICE]
# 每次实验换个名字，结果分开存在 runs/verna-<底座>-<名字>/
RUN_TAG = "v1"  # @param {type:"string"}
# 训练轮数：smoke1 在约 0.85 轮验证损失最低（当时排程按 2 轮），数据多了以后默认 1 轮；每 100 步验证、自动留最好的检查点
EPOCHS = 1  # @param {type:"number"}
RUN = f"verna-{CHOICE}-{RUN_TAG}"
OUT = f"{WORK}/runs/{RUN}"
MAX_LEN = 2048
# 推理采样：两家用同一组（Qwen3.5 非思考模式的推荐值）。改写要忠实原文，温度低些好；
# smoke1 时 Gemma 用官方的 1.0/0.95/64，保留样本 9 条只原样返回 1 条，还改错了一处事实，两家也没法直接比
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)

import sys, torch
if FAMILY == "gemma":
    torch._dynamo.config.recompile_limit = 64   # 官方 Gemma 4 notebook 的设置
sys.path.insert(0, WORK)
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9), "GB")
assert torch.cuda.is_bf16_supported(), "这张卡不支持 bf16（多半是 T4），换 H100 / A100 再跑"
if torch.cuda.get_device_properties(0).total_memory / 1e9 < 60:  # 40GB 卡：批量减半，梯度累积加倍，等效批量不变
    BS, GA = max(1, BS // 2), GA * 2
print(f"{CHOICE}：批量 {BS} × 梯度累积 {GA}，LoRA 秩 {LORA_R}")
""")

code("""
if FAMILY == "qwen":
    from unsloth import FastLanguageModel as Loader
    model, tokenizer = Loader.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_LEN,
        load_in_4bit=False, load_in_16bit=True,   # Unsloth 文档：Qwen3.5 不建议 4-bit QLoRA
        full_finetuning=False,
    )
    model = Loader.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_R, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=MAX_LEN,
    )
else:  # Gemma 4：写法照 Unsloth 官方 Gemma4_(12B)_Text notebook
    from unsloth import FastModel as Loader
    from unsloth.chat_templates import get_chat_template
    model, tokenizer = Loader.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_LEN, load_in_4bit=False, full_finetuning=False,
    )
    model = Loader.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=LORA_R, lora_alpha=LORA_R, lora_dropout=0, bias="none", random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
tok = getattr(tokenizer, "tokenizer", tokenizer)  # 多模态模型有时返回 processor，分词用里面的 tokenizer
""")

code("""
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": f"{WORK}/data/sft/train.jsonl", "val": f"{WORK}/data/sft/val.jsonl"})
def render(msgs, gen=False):
    kw = {"enable_thinking": False} if FAMILY == "qwen" else {}
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)

fmt = lambda b: {"text": [render(m) for m in b["messages"]]}
ds = ds.map(fmt, batched=True, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda x: len(tok(x["text"], add_special_tokens=False).input_ids) <= MAX_LEN)
print(len(ds["train"]), "train /", len(ds["val"]), "val")

sample = ds["train"][0]["text"]
print(sample)
if FAMILY == "qwen":
    # 模板里 assistant 开头如果带空的 <think></think>，推理时（enable_thinking=False）生成提示也会带，两边一致
    EMPTY_THINK = "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
    PARTS = dict(instruction_part="<|im_start|>user\\n",
                 response_part=EMPTY_THINK if EMPTY_THINK in sample else "<|im_start|>assistant\\n")
else:
    PARTS = {}  # Unsloth 会从 Gemma 4 模板里自动识别
print("回答段落标记：", PARTS or "自动识别")
""")

code("""
import glob
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_LEN,
        per_device_train_batch_size=BS, gradient_accumulation_steps=GA,   # 爆显存就把 BS 减半、GA 加倍
        num_train_epochs=EPOCHS, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, optim="adamw_8bit", bf16=True,
        logging_steps=10, eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",   # 两个已有项目都在 1 轮左右就到最低点
        output_dir=OUT, report_to="none", seed=3407,
    ),
)
trainer = train_on_responses_only(trainer, **PARTS)

# 检查：参与训练的只能是改写结果，不能有指令和原文
row = trainer.train_dataset[0]
print(tok.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != -100])[:300])
""")

code("""
resume = bool(glob.glob(f"{OUT}/checkpoint-*"))
print("接着上次训练" if resume else "从头训练")
stats = trainer.train(resume_from_checkpoint=resume or None)
print(round(stats.metrics["train_runtime"] / 60), "分钟；显存峰值", round(torch.cuda.max_memory_reserved() / 1e9, 1), "GB")
""")

md("""
## 快速自检

在测试集上生成，用和本机同一套规则算硬事实错误率和 AI 腔密度。结果存到 Drive，拿回本机用 `scripts/eval.py --judge` 做完整评测（含裁判和盲评）。
""")

code("""
import json, re, statistics as st
from srh import facts, tells, prompts

Loader.for_inference(model)
tok.padding_side = "left"

def gen(texts):
    ps = [render(prompts.rewrite_messages(t), gen=True) for t in texts]
    enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=1200, do_sample=True, **SAMPLING)
    res = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
    return [re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip() for r in res]

def run_file(path, out_path, n=200, bs=16):
    rows = [json.loads(l) for l in open(path)][:n]
    for i in range(0, len(rows), bs):
        for r, o in zip(rows[i:i + bs], gen([r["src"] for r in rows[i:i + bs]])):
            r["out"] = o
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    bad = [r for r in rows if not facts.compare(r["src"], r["out"])["ok"]]
    print(f"{path.split('/')[-1]}: {len(rows)} 条，硬事实错误 {len(bad) / len(rows):.1%}，"
          f"AI腔密度 {st.mean(tells.density(r['src']) for r in rows):.1f} → {st.mean(tells.density(r['out']) for r in rows):.1f}，"
          f"禁用句式 {sum(bool(tells.banned(r['out'])) for r in rows)} 条")
    return rows, bad

rows, bad = run_file(f"{WORK}/data/sft/test.jsonl", f"{OUT}/pred_test.jsonl")
for r in rows[:3]:
    print("【输入】", r["src"][:300], "\\n【输出】", r["out"][:300], "\\n【人写原文】", r["ref"][:300], "\\n")
for r in bad[:3]:
    print("【事实对不上】", facts.compare(r["src"], r["out"])["hard"])
""")

code("""
# 真实 AI 输出（没有人写参照）：30 道题 × Gemini 3.8 Flash / Gemini 4 Argon / Claude Sonnet 5.5 / DeepSeek V4.1 Flash，服务器上 scripts/make_native_test.py 生成
import os
if os.path.exists(f"{WORK}/data/native_test.jsonl"):
    nrows, nbad = run_file(f"{WORK}/data/native_test.jsonl", f"{OUT}/pred_native.jsonl")
    for r in nrows[:3]:
        print("【AI 原文】", r["src"][:300], "\\n【改写】", r["out"][:300], "\\n")
""")

md("""
## 回传结果给服务器

服务器连 Drive 用的是“只能看到自己建的文件”的权限，Colab 新建的文件（`runs/` 下面这些）它看不到。所以服务器事先在 `verna-train/inbox/` 给每个底座建了一个 `<底座>.zip` 占位文件，这一格把两份预测和运行信息（超参、训练曲线、数据版本、上面的自检指标）打成 zip 盖上去，服务器就能拉回去做完整评测。同一个底座再跑一次会盖掉上一次的，`runs/<RUN>/result.zip` 里另有一份。
""")

code("""
import hashlib, shutil, time, zipfile

def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def _ver(p):
    try:
        import importlib.metadata as _m
        return _m.version(p)
    except Exception:
        return None

def _summary(rs):
    return {"n": len(rs), "hard_fact_err": sum(not facts.compare(r["src"], r["out"])["ok"] for r in rs) / max(len(rs), 1),
            "tells_src": st.mean(tells.density(r["src"]) for r in rs), "tells_out": st.mean(tells.density(r["out"]) for r in rs),
            "banned": sum(bool(tells.banned(r["out"])) for r in rs)}

_stats = globals().get("stats")
meta = {
    "run": RUN, "choice": CHOICE, "model": MODEL, "lora_r": LORA_R, "batch": BS, "grad_accum": GA, "max_len": MAX_LEN, "sampling": SAMPLING,
    "train_args": {k: getattr(trainer.args, k) for k in ("num_train_epochs", "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay")},
    "gpu": torch.cuda.get_device_name(0), "n_train": len(ds["train"]), "n_val": len(ds["val"]),
    "data_md5": {n: _md5(f"{WORK}/data/sft/{n}.jsonl") for n in ("train", "val", "test")},
    "train_runtime_min": round(_stats.metrics["train_runtime"] / 60, 1) if _stats else None,
    "max_mem_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
    "versions": {_p: _ver(_p) for _p in ("torch", "triton", "transformers", "trl", "peft", "accelerate", "unsloth", "unsloth_zoo", "bitsandbytes")},
    "compile": COMPILE,
    "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
    "log_history": trainer.state.log_history,
    "summary": {"test": _summary(rows), **({"native": _summary(nrows)} if "nrows" in globals() else {})},
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
local = f"/content/{RUN}-result.zip"
with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    for n in ("pred_test.jsonl", "pred_native.jsonl"):
        if os.path.exists(f"{OUT}/{n}"):
            z.write(f"{OUT}/{n}", n)
shutil.copyfile(local, f"{OUT}/result.zip")
slot = f"{WORK}/inbox/{CHOICE}.zip"
if os.path.exists(slot):
    shutil.copyfile(local, slot)   # 盖写服务器建的文件：文件 ID 不变，服务器读得到（新建的读不到，所以不新建）
    print(f"已回传 {os.path.getsize(local) // 1024} KB 到 {slot}。Drive 同步要几分钟，之后告诉 Claude“跑完了”")
else:
    print(f"没找到占位文件 {slot}：让 Claude 在服务器上跑 scripts/pull_results.py --init，再重跑这一格")
""")

md("""
## 保存

LoRA 适配器很小（几百 MB），一定存。GGUF 给 llama.cpp / Ollama / LM Studio 用。24GB 的 Mac：9B 和 12B 的 q8_0（约 10GB / 13GB）都放得下，q8_0 质量损失最小；q4_k_m 更快、更省内存。
""")

code("""
model.save_pretrained(f"{OUT}/lora")
tok.save_pretrained(f"{OUT}/lora")

# GGUF 先写到本地盘再拷到 Drive，直接写 Drive 很慢
import glob, os, shutil
os.makedirs(f"{OUT}/gguf", exist_ok=True)
for q in ["q8_0", "q4_k_m"]:
    model.save_pretrained_gguf(f"/content/gguf_{q}", tokenizer, quantization_method=q)
    # 统一命名成 verna-<底座>-<名字>-<量化>.gguf；视觉部分（mmproj）一起拷，BF16 之类中间文件不拷
    fs = sorted(glob.glob(f"/content/gguf_{q}*/*.gguf"))
    keep = [f for f in fs if q in os.path.basename(f).lower() or "mmproj" in os.path.basename(f).lower()] or fs
    for f in keep:
        b = os.path.basename(f).lower()
        name = f"{RUN}-mmproj.gguf" if "mmproj" in b else (f"{RUN}-{q}.gguf" if q in b else f"{RUN}-{os.path.basename(f)}")
        shutil.copy(f, f"{OUT}/gguf/{name}")
!ls -lh "{OUT}/gguf"
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).with_name("train_sft.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
