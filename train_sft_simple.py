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
import re, time, glob
from typing import Optional
from datetime import timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- SwanLab ----------------
swanlab.init(project="mol-sft-simple", experiment_name="exp-001", description="SFT with <mol> embedding-append", mode="online")

import json

class BarrierCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        safe_barrier()  
    def on_evaluate(self, args, state, control, **kwargs):
        safe_barrier()
    def on_train_begin(self, args, state, control, **kwargs):
        safe_barrier()
    def on_train_end(self, args, state, control, **kwargs):
        safe_barrier()
        
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
        
def _is_complete_checkpoint(path: str) -> bool:
    """判断 checkpoint 是否完整（存在 state 与权重文件之一）"""
    if not os.path.isdir(path):
        return False
    has_state = os.path.exists(os.path.join(path, "trainer_state.json"))
    # 兼容 HF 多种保存格式
    has_model = (
        os.path.exists(os.path.join(path, "model.safetensors")) or
        os.path.exists(os.path.join(path, "pytorch_model.bin")) or
        os.path.exists(os.path.join(path, "pytorch_model.bin.index.json"))
    )
    return has_state and has_model

def _list_checkpoints(output_dir: str):
    cks = []
    for p in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m and _is_complete_checkpoint(p):
            step = int(m.group(1))
            cks.append((step, p))
    cks.sort(key=lambda x: x[0])
    return [p for _, p in cks]

def latest_checkpoint(output_dir: str) -> Optional[str]:
    cks = _list_checkpoints(output_dir)
    return cks[-1] if cks else None

def safe_barrier():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


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
    diffusion_conf = cfg.get("diffusion", {}) or {}
    diff_conf = diffusion_conf.get("diffusion", {}) or {}
    diff_adp_conf = diffusion_conf.get("adapter", {}) or {}

    model = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=mol_token,
        proxy=cfg.get("network", {}).get("proxy"),  # 避免 KeyError
        debug=False,
        diffusion_config=diff_conf,    
        diffusion_adapter_config=diff_adp_conf,     
    )

    # 若你要把 generation 参数塞进类里（供内部调用）
    gen_conf = diffusion_conf.get("generation", {}) or {}
    model.diffusion_gen_num_nodes_lig = gen_conf.get("num_nodes_lig", None)

    # ---------------- GNN 权重加载 ----------------
    gnn_state_dict_path = cfg["paths"].get("gnn_state_dict_path")
    if gnn_state_dict_path and os.path.exists(gnn_state_dict_path):
        gnn_ckpt = torch.load(gnn_state_dict_path, map_location="cpu")
        gnn_state_dict = gnn_ckpt.get("model_state_dict", gnn_ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in gnn_state_dict.items())
        model.gvp_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"[{local_rank}] ✅ Loaded GVPEncoder")

    # ---------------- 冻结参数（按 YAML 开关） ----------------
    freeze_llm = cfg["train"].get("freeze_llm", False)
    freeze_gnn = cfg["train"].get("freeze_gnn", False)
    freeze_diffusion = cfg["train"].get("freeze_diffusion", True)
    freeze_mol_adapter = cfg["train"].get("freeze_mol_adapter", False)
    freeze_diffusion_adapter = cfg["train"].get("freeze_diffusion_adapter", True)

    if freeze_llm:
        for n, p in model.llm.named_parameters():
            # 视情况放开 embedding / lm_head；下行仅示例保留输入嵌入可训
            if 'embed_tokens' not in n:
                p.requires_grad = False

    if freeze_gnn:
        for p in model.gvp_encoder.parameters():
            p.requires_grad = False

    if freeze_mol_adapter:
        for p in model.mol_adapter.parameters():
            p.requires_grad = False

    # 注意对象名：diffusion 是 DDPM 模型；diffusion_adapter 是你训练的 MLP
    if freeze_diffusion and getattr(model, "diffusion", None) is not None:
        for p in model.diffusion.parameters():
            p.requires_grad = False

    if freeze_diffusion_adapter and getattr(model, "diffusion_adapter", None) is not None:
        for p in model.diffusion_adapter.parameters():
            p.requires_grad = False

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
    # 过滤掉异常/空文本与超短样本
    def is_valid(example):
        t = example.get("text", "")
        return isinstance(t, str) and len(t.strip()) > 0 and "<|assistant|>" in t

    processed_dataset = processed_dataset.filter(is_valid)

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
        optim="paged_adamw_8bit",
        
        ddp_bucket_cap_mb = 25,
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
        callbacks=[BarrierCallback(), SwanLabCallback(), CopyConfigCallback()]
    )

    # --- 断点重试配置 ---
    max_retries = int(cfg["train"].get("max_retries", 3))
    backoff_base = int(cfg["train"].get("retry_backoff_sec", 30))  # 第一次 30s，随后 60s、90s...

    # 启动时若已有 checkpoint，则从最新点恢复
    resume_ckpt = latest_checkpoint(cfg["paths"]["output_dir"])

    if trainer.is_world_process_zero():
        print(f"[{local_rank}] Resume from checkpoint: {resume_ckpt}")

    for attempt in range(1, max_retries + 1):
        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
            # 成功则保存最终权重并跳出
            if trainer.is_world_process_zero():
                trainer.save_model(cfg["paths"]["output_dir"])
            break
        except Exception as e:
            # 打印错误并准备重试
            msg = repr(e)
            if trainer.is_world_process_zero():
                print(f"[{local_rank}] ❌ Train failed (attempt {attempt}/{max_retries}): {msg}")

            # 典型错误可提示一下（可选）
            if any(k in msg for k in ["CUDA out of memory", "CUBLAS", "CUDNN"]):
                if trainer.is_world_process_zero():
                    print(f"[{local_rank}] Hint: OOM/显存问题，可考虑减小 per_device_train_batch_size 或开启梯度累积。")

            # 清理 + 同步
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            safe_barrier()

            # 到此若达到最大重试次数，抛出错误
            if attempt == max_retries:
                raise

            # 指数回退等待
            wait_s = backoff_base * attempt
            if trainer.is_world_process_zero():
                print(f"[{local_rank}] ⏳ Waiting {wait_s}s before retry ...")
            time.sleep(wait_s)

            # 期间可能已经产生新的 ckpt，重设恢复点
            resume_ckpt = latest_checkpoint(cfg["paths"]["output_dir"])
            if trainer.is_world_process_zero():
                print(f"[{local_rank}] ↩️  Next resume checkpoint: {resume_ckpt}")


def main(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    world_size = int(os.environ["WORLD_SIZE"])
    main_worker(world_size, cfg)

if __name__ == "__main__":
    main()
