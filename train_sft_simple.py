# train_sft_simple_mp.py
import os
import random
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from modules.mol_aware_lm_simple import MolAwareCausalLM
import swanlab
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- SwanLab ----------------
swanlab.init(project="mol-sft-simple", experiment_name="exp-001", description="SFT with <mol> embedding-append", mode="online")

import json

def safe_to_str(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        # 把多段输入/输出合成多行
        return "\n".join(safe_to_str(xx) for xx in x)
    if isinstance(x, dict):
        # 可读的 JSON
        return json.dumps(x, ensure_ascii=False)
    return str(x)

class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            if logs is not None:
                swanlab.log(logs, step=state.global_step)

class CopyConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            model = kwargs["model"]
            tok = kwargs.get("tokenizer", None)
            if getattr(model, "config", None) is not None:
                model.config.save_pretrained(args.output_dir)
            if tok is not None:
                tok.save_pretrained(args.output_dir)

# ---------------- Utils ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------- Main (DDP spawn) ----------------
def main_worker(world_size, cfg):
    # 进 main_worker 开头（即便交给 Trainer DDP，也可以放）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[rank {local_rank}] using cuda:{torch.cuda.current_device()}")
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=world_size)

    set_seed(cfg["seed"] + local_rank)

    # ---------------- Tokenizer ----------------
    llm_name = cfg["paths"]["llm_name_or_path"]
    mol_token = cfg["tokens"]["mol_token"]
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    role_specials = ["<|user|>", "<|assistant|>"]
    to_add = []
    current_vocab = tokenizer.get_vocab()
    if mol_token not in current_vocab: to_add.append(mol_token)
    for t in role_specials:
        if t not in current_vocab: to_add.append(t)
    if to_add: tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    # ---------------- LLM ----------------
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float32,
    ).to(local_rank)
    # .to(local_rank if local_rank == 0 else torch.device("cpu"))  # LLM 放 GPU0
    
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.vocab_size = len(tokenizer)
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.eos_token_id = tokenizer.eos_token_id
    llm.config.bos_token_id = tokenizer.bos_token_id

    # ---------------- MolAwareCausalLM ----------------
    model = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=mol_token,
        proxy=cfg["network"]["proxy"],
        debug=False,
    )

    # ---------------- GNN 权重加载 ----------------
    gnn_state_dict_path = cfg["paths"].get("gnn_state_dict_path")
    if gnn_state_dict_path and os.path.exists(gnn_state_dict_path):
        gnn_ckpt = torch.load(gnn_state_dict_path, map_location="cpu")
        gnn_state_dict = gnn_ckpt.get("model_state_dict", gnn_ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in gnn_state_dict.items())
        model.gvp_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"[{local_rank}] ✅ Loaded GVPEncoder")

    # ---------------- 模型并行 ----------------
    # 冻结参数可选
    freeze_llm = cfg["train"].get("freeze_llm", False)
    freeze_gnn = cfg["train"].get("freeze_gnn", False)
    freeze_mol_adapter = cfg["train"].get("freeze_mol_adapter", False)
    if freeze_llm:
        for n, p in model.llm.named_parameters():
            if 'embed_tokens' not in n: p.requires_grad = False
    if freeze_gnn:
        for p in model.gvp_encoder.parameters(): p.requires_grad = False
    if freeze_mol_adapter:
        for p in model.mol_adapter.parameters(): p.requires_grad = False

    # ---------------- DDP 包装 ----------------
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    # )
    
    # model = model.to(local_rank)
    
    if local_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{local_rank}] Trainable params: {trainable_params} / Total: {total_params}")
    # ---------------- 数据集 ----------------
    raw = load_dataset("json", data_files=cfg["train"]["dataset_path"])["train"]
    # 设定一些 tokenizer 细节（放在 tokenizer 创建后）
    tokenizer.model_max_length = int(cfg["train"]["max_seq_length"])
    tokenizer.padding_side = "right"  # 一般右 padding 更稳
    # 如果你的模型是 Llama/GLM 一类，这两句没副作用；pad_token 已在你的代码里处理

    def format_dataset(example):
        user = safe_to_str(example.get("input", "")).strip()
        assistant = safe_to_str(example.get("output", "")).strip()
        return {"text": f"<|user|>{user}\n<|assistant|>{assistant}"}

    processed_dataset = raw.map(format_dataset, remove_columns=raw.column_names)
    processed_dataset = processed_dataset.filter(lambda ex: isinstance(ex.get("text",""), str) 
                                            and len(ex["text"].strip())>0 
                                            and "<|assistant|>" in ex["text"])
    # 过滤掉异常/空文本与超短样本
    def is_valid(example):
        t = example.get("text", "")
        return isinstance(t, str) and len(t.strip()) > 0 and "<|assistant|>" in t

    processed_dataset = processed_dataset.filter(is_valid)

    # （可选）在小批样本上做一次“预 tokenization 检查”来提前暴露问题
    def quick_tokenize_check(batch):
        # 不写入数据集，仅用于 sanity check
        _ = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(cfg["train"]["max_seq_length"]),
            padding=False,  # 这里只是检查，不需要 pad
            return_attention_mask=True,
        )
        return batch
    processed_dataset = processed_dataset.map(quick_tokenize_check, batched=True, batch_size=64)

    split = processed_dataset.train_test_split(test_size=0.05, seed=cfg["seed"])

    # ---------------- DataCollator ----------------
    response_template = "<|assistant|>"
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    # ---------------- TrainingArguments ----------------
    args = TrainingArguments(
        output_dir=cfg["paths"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["train"]["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=True,
        ddp_find_unused_parameters=True,
        report_to="none",
        optim="paged_adamw_8bit"
    )

    if cfg["train"]["gradient_checkpointing"]:
        llm.config.use_cache = False
    # ---------------- SFTTrainer ----------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=int(cfg["train"]["max_seq_length"]),
        packing=cfg["train"]["packing"],
        data_collator=data_collator,
        callbacks=[SwanLabCallback(), CopyConfigCallback()],
    )

    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(cfg["paths"]["output_dir"])

def main(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    world_size = int(os.environ["WORLD_SIZE"])
    main_worker(world_size, cfg)

if __name__ == "__main__":
    main()
