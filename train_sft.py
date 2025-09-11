# train_sft.py
import os, json, random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from modules.mol_aware_lm import MolAwareCausalLM
import yaml

def set_seed(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    llm_name = cfg["paths"]["llm_name_or_path"]
    mol_token = cfg["tokens"]["mol_token"]
    freeze = cfg["freeze"]
    ckpts  = cfg["paths"]["checkpoints"]

    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    # 添加 special token
    if mol_token not in tokenizer.get_vocab():
        # tokenizer.add_special_tokens({"additional_special_tokens": [mol_token]})
        role_specials = ["<|user|>", "<|assistant|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": [
            [mol_token], *[t for t in role_specials if t not in tokenizer.get_vocab()]
        ]})
        llm.resize_token_embeddings(len(tokenizer))  # 记得扩容嵌入
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float32,
        device_map="auto",
    )
    # 由于增加了 token，需要 resize embedding
    llm.resize_token_embeddings(len(tokenizer))

    # 包装模型（在 <mol> 处注入）
    model = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=mol_token,
        dims={
            "diff_cond_dim": llm.get_input_embeddings().embedding_dim,
            "diff_latent_dim": llm.get_input_embeddings().embedding_dim,
            "gnn_node_dim": 8,
            "gnn_hidden": 256,
        },
        freeze=freeze,
        ckpts=ckpts,
    ).to(llm.device)

    # 数据集
    raw = load_dataset("json", data_files=cfg["train"]["dataset_path"])["train"]
    split = raw.train_test_split(test_size=0.05, seed=cfg["seed"])
    def format_example(ex):
        prompt = ex["input"].strip()
        answer = ex["output"].strip()
        text = f"<|user|>\n{prompt}\n<|assistant|>\n{answer}{tokenizer.eos_token}"
        return {"text": text}
    ds_train = split["train"].map(format_example, remove_columns=split["train"].column_names)
    ds_eval  = split["test"].map(format_example,  remove_columns=split["test"].column_names)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|assistant|>\n"
    )

    args = TrainingArguments(
        output_dir=cfg["paths"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=cfg["train"]["max_seq_length"],
        packing=cfg["train"]["packing"],
    )

    trainer.train()
    trainer.save_model(cfg["paths"]["output_dir"])

if __name__ == "__main__":
    main()
