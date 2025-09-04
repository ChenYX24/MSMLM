import os
import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
import swanlab
from transformers import TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer

def main():
    # 强制关闭 bitsandbytes 初始提示
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./llama3-chem-checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        bf16=True,
        report_to=[],  # 不用 wandb
        optim="adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_strategy="epoch",  # 每个epoch保存
    )

    # 初始化 SwanLab（可选）
    swanlab_config = {
        "output_dir": training_args.output_dir,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "save_steps": training_args.save_steps,
        "save_total_limit": training_args.save_total_limit,
        "logging_steps": training_args.logging_steps,
        "learning_rate": training_args.learning_rate,
        "warmup_steps": training_args.warmup_steps,
        "weight_decay": training_args.weight_decay,
        "bf16": training_args.bf16,
        "logging_dir": training_args.logging_dir,
        "current_time": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    swanlab.init(
        project="llama3-chem-pretrain",
        experiment_name="llama3.2b-chem",
        config=swanlab_config,
    )

    # 加载 tokenizer 和 model
    model_path = "/data1/lvchangwei/GNN/Project/LLM/llama3.2-chem-pretrain/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动放到单张GPU
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id  # 防止报错

    # 加载文本数据
    dataset = load_dataset("text", data_files={
        "train": "/data1/lvchangwei/GNN/orgin_data/chemical-data/mixed_corpus.txt"
    })["train"]

    # 分词 + 分块
    block_size = 128

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=False)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=4
    )

    # 打印样本查看格式
    print("Sample data:", lm_dataset[0])

    # SwanLab 日志回调
    class SwanlabCallback(TrainerCallback):
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs:
                log_data = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                if log_data:
                    swanlab.log(log_data)

    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SwanlabCallback()],
    )

    # 开始训练
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
