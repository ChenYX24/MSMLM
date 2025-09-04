from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import json
from tqdm import tqdm

# 加载Llama模型和tokenizer
cache_dir = '/data1/chenyuxuan/Project/MSMLM/model'
model_path = '/data1/chenyuxuan/Project/MSMLM/model/llama3_2_3b_instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

def generate_3d_structure(smiles):
    # 创建分子对象
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        # 尝试生成三维构象
        success = AllChem.EmbedMolecule(molecule, randomSeed=42)  # 设置随机种子以提高可重复性
        if success != 0:  # 如果 EmbedMolecule 返回非零，表示失败
            print(f"警告：无法生成三维构象，SMILES: {smiles}")
            return None
        
        # 优化三维构象
        AllChem.UFFOptimizeMolecule(molecule)
        
        # 获取并返回构象坐标
        conformer = molecule.GetConformer()
        coordinates = conformer.GetPositions()
        return coordinates
    else:
        print(f"无法解析 SMILES: {smiles}")
        return None

def get_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # 获取模型输出，确保返回 hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # 获取所有层的隐藏状态
    hidden_states = outputs.hidden_states
    
    embedding = torch.stack(hidden_states)[-1]  # 获取最后一层的[batch, sentence_length, embedding_dim]
    
    return embedding


# 读取SMILES文件
cid_smiles_dict = {}
n = 0  # 从第n行开始处理
m = 5000  # 直到第m行结束
with open('/data1/chenyuxuan/Project/MSMLM/data/pubchem/CID-SMILES', 'r') as f:
    for line_count, line in enumerate(f):
        if line_count < n:
            continue  # 跳过前n行
        if line_count > m:
            break  # 停止处理超过第m行的数据
        cid, smiles = line.strip().split('\t')
        cid_smiles_dict[cid] = smiles

# 读取CID-IUPAC文件
cid_iupac_dict = {}
with open('/data1/chenyuxuan/Project/MSMLM/data/pubchem/CID-IUPAC', 'r') as f:
    for line_count, line in enumerate(f):
        if line_count < n:
            continue  # 跳过前n行
        if line_count > m:
            break  # 停止处理超过第m行的数据
        cid, iupac = line.strip().split('\t')
        cid_iupac_dict[cid] = iupac

# 获取CID-IUPAC的embedding并生成三维构象
output_data = []

for cid, smiles in tqdm(cid_smiles_dict.items(), desc="Processing SMILES", total=len(cid_smiles_dict)):
    # 从CID-IUPAC字典获取IUPAC名称
    iupac_name = cid_iupac_dict.get(cid, None)
    if not iupac_name:
        print(f"警告: 找不到CID {cid} 对应的IUPAC名称。跳过此CID。")
        continue

    # 获取CID-IUPAC的embedding
    embedding = get_embedding(iupac_name)
        
    # 使用RDKit生成三维构象
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        coordinates = generate_3d_structure(smiles)
        if coordinates is None:
            print(f"警告: 无法生成三维构象，SMILES: {smiles}，CID: {cid}")
            continue
        # 将结果存储
        output_data.append({
            "CID": cid,
            "IUPAC": iupac_name,
            "SMILES": smiles,
            "embedding": embedding.tolist(),  # 转为列表以便保存
            "coordinates": coordinates.tolist()  # 转为列表以便保存
        })

# 将结果保存为JSONL文件
with open('/data1/chenyuxuan/Project/MSMLM/data/output_data.jsonl', 'w') as outfile:
    for entry in output_data:
        json.dump(entry, outfile)
        outfile.write('\n') 

print("处理完成，结果已保存。")