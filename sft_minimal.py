# train_sft_from_io_messages.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 防止 fork 前并行导致的警告/死锁
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ==== 改成你的路径 ====
MODEL_PATH = "/data1/opensource_models/llama3_2_3b_instruct"
DATA_PATH  = "/data1/lvchangwei/LLM/SFT_data/SFT_DATA.json"  # 你的 input/output/metadata 数据
OUTPUT_DIR = "outputs/sft_from_io_messages"

def build_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16

def main():
    # 1) tokenizer / model
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 可选：加入 <mol>，若不存在
    if "<mol>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<mol>"]})

    dtype = build_dtype()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    model.resize_token_embeddings(len(tok))

    # 2) 加载原始 json（含 input/output/metadata）
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # 3) 映射成 messages=[user, assistant]
    def to_messages(ex):
        # 保留你 output 中的原样内容（含 <mol>、the question is... the answer is...）
        return {
            "messages": [
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        }
    ds = ds.map(to_messages, remove_columns=[c for c in ds.column_names if c != "messages"])

    # 5) 训练参数（DDP 友好）
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        report_to="none",
    )

    # 6) 开训（不需要 dataset_text_field / 不需要 to_text）
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=args,
        train_dataset=ds,
        max_seq_length=2048,
        packing=False,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        print(f"Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
