import json
from transformers import AutoTokenizer

# ====================== 配置 ======================
llm_name = "/data1/chenyuxuan/Project/MSMLM/code/mol_sft/outputs/pretrain"
data_path = "/data1/lvchangwei/LLM/SFT_data/SFT_DATA.json"
mol_token = "<mol>"

# ====================== 加载 tokenizer ======================
tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 特殊 token
role_specials = ["<|user|>", "<|assistant|>", mol_token]
to_add = []
current_vocab = tokenizer.get_vocab()
for t in role_specials:
    if t not in current_vocab:
        to_add.append(t)
if to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    print(f"✅ 添加了缺失的特殊 token: {to_add}")

# ====================== 检查训练数据 ======================
def check_special_tokens_in_sample(sample_text, required_tokens):
    missing = [t for t in required_tokens if t not in sample_text]
    return missing

missing_count = 0

# 解析整个 JSON 数组
with open(data_path, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 文件解析失败: {e}")
        exit(1)

for idx, sample in enumerate(data, 1):
    output_text = sample.get("output", "")
    missing_tokens = check_special_tokens_in_sample(output_text, ["<|user|>", "<|assistant|>"])
    if missing_tokens:
        missing_count += 1
        print(f"[样本 {idx}] 缺失特殊 token {missing_tokens}:\n{output_text[:500]}...\n")  # 截断显示前 500 字符

print(f"✅ 检查完成，总共缺失特殊 token 的样本数: {missing_count}")
