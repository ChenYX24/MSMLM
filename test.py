from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 改这里路径就能换模型
# model_path = "/data1/chenyuxuan/Project/MSMLM/llama3-chem-checkpoints"
# model_path = "/data1/opensource_models/llama3_2_3b_instruct"
model_path = "/data1/chenyuxuan/Project/MSMLM/code/mol_sft/outputs/sft"
# model_path = "/data1/lvchangwei/LLM/model/models--meta-llama--Llama-3.2-3B/snapshots/llama3.2"
# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  
    device_map="cuda:0"          
)

inputs = tokenizer("Ethanol is a common solvent with a density lower than water. Please explain the reason and continue.", return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# import json
# with open('/data1/lvchangwei/LLM/SFT_data/SFT_DATA.json', 'r') as f:
#     data = json.load(f)

# # 打印数据  
# keys = set()
# for i in range(len(data)):
#     # if data[i]['metadata']["task"] not in keys:
#     if data[i]['metadata']["task"] == "general":
#         print(data[i])
#         keys.add(data[i]['metadata']["task"])
    