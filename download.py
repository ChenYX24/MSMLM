from transformers import AutoTokenizer, AutoModelForCausalLM
# 指定模型下载到 "model/" 目录
cache_dir = "/data1/chenyuxuan/Project/MSMLM/model/"

# 加载并缓存tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir=cache_dir)

# 加载并缓存模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir=cache_dir)

# 保存模型和tokenizer到指定目录
model.save_pretrained(cache_dir)
tokenizer.save_pretrained(cache_dir)