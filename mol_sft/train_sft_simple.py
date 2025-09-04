# train_sft.py  —— 修正版（可直接用）
import os, json, random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from modules.mol_aware_lm_simple import MolAwareCausalLM   # ★ 使用新版
import yaml
from transformers import TrainerCallback
import swanlab

os.environ["TOKENIZERS_PARALLELISM"] = "false"

swanlab.init(
    project="mol-sft-simple",
    experiment_name="exp-001",
    description="SFT with <mol> embedding-append",
    mode="online",
)

class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            swanlab.log(logs, step=state.global_step)

def set_seed(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class CopyConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]; tok = kwargs.get("tokenizer", None)
        if getattr(model, "config", None) is not None:
            model.config.save_pretrained(args.output_dir)
        if tok is not None:
            tok.save_pretrained(args.output_dir)

def main(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    llm_name = cfg["paths"]["llm_name_or_path"]
    mol_token = cfg["tokens"]["mol_token"]

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 对 LLaMA 类必需
    role_specials = ["<|user|>", "<|assistant|>"]

    to_add = []
    if mol_token not in tokenizer.get_vocab():
        to_add.append(mol_token)
    for t in role_specials:
        if t not in tokenizer.get_vocab():
            to_add.append(t)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})  # ★ 扁平列表

    # 2) llm
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float32,
        device_map=None,
    )
    # ★ 注册新 special tokens 后，需要 resize
    llm.resize_token_embeddings(len(tokenizer))
    
    llm.config.vocab_size = len(tokenizer)
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.eos_token_id = tokenizer.eos_token_id
    llm.config.bos_token_id = tokenizer.bos_token_id
    
    # 3) 包装模型（embedding级扩展 + 实时mol插入支持）
    model = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=mol_token,
        debug=False,
    ).to(llm.device)

    # 4) 数据：把 {input, output} → messages
    raw = load_dataset("json", data_files=cfg["train"]["dataset_path"])["train"]
    max_len = int(cfg["train"]["max_seq_length"])

    def to_messages(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["input"].strip()},
                {"role": "assistant", "content": ex["output"].strip()},
            ]
        }
    raw = raw.map(to_messages, remove_columns=[c for c in raw.column_names if c != "messages"])

    # 切分
    split = raw.train_test_split(test_size=0.05, seed=cfg["seed"])
    print(f"[data] train: {len(split['train'])}, eval: {len(split['test'])}")

    # 5) 训练参数
    args = TrainingArguments(
        output_dir=cfg["paths"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    # 这里直接让 TRL 负责格式化；你的数据里 messages→字符串可用 formatting_func（略）
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        max_seq_length=cfg["train"]["max_seq_length"],
        packing=cfg["train"]["packing"],
        callbacks=[SwanLabCallback, CopyConfigCallback],
    )

    trainer.train()
    trainer.save_model(cfg["paths"]["output_dir"])
    tokenizer.save_pretrained(cfg["paths"]["output_dir"])

if __name__ == "__main__":
    main()
