import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import spacy
import scispacy
from scispacy.linking import EntityLinker
import re
import pickle

# 加载模型和NER工具
def setup_models():
    """初始化所需的模型和工具"""
    # 加载Llama模型
    cache_dir = '/data1/chenyuxuan/Project/MSMLM/model'
    model_path = '/data1/chenyuxuan/Project/MSMLM/model/llama3_2_3b_instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载scispacy NER模型
    nlp = spacy.load("en_core_sci_sm")
    # 可选择添加entity linker
    try:
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "chebi"})
    except:
        print("Warning: Could not load entity linker")
    
    return tokenizer, model, nlp

def generate_3d_structure(smiles):
    """生成分子的三维构象"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        success = AllChem.EmbedMolecule(molecule, randomSeed=42)
        if success != 0:
            return None
        AllChem.UFFOptimizeMolecule(molecule)
        conformer = molecule.GetConformer()
        coordinates = conformer.GetPositions()
        atom_types = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        bond_types = []
        for bond in molecule.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            btype = bond.GetBondTypeAsDouble()
            bond_types.append((start_atom, end_atom, btype))
        return {"coordinates": coordinates, "atom_types": atom_types, "bond_types": bond_types}
    
    return None

def get_embedding(text, tokenizer, model):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    attention_mask = inputs["attention_mask"]
    # 获取模型输出，确保返回 hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # 获取所有层的隐藏状态
    hidden_states = outputs.hidden_states
    
    embedding = torch.stack(hidden_states)[-1]  # 获取最后一层的[batch, sentence_length, embedding_dim]
    seq_lens = attention_mask.sum(dim=1) - 1  # 减1是因为索引从0开始
    last_token_embedding = embedding[0][seq_lens][0]

    return last_token_embedding


def extract_chemical_entities(text, nlp):
    """使用scispacy + 本体链接 + 名称规则提取化学实体（去重）"""
    doc = nlp(text)
    seen_texts = set()
    chemical_entities = []

    for ent in doc.ents:
        ent_text = ent.text.strip()
        lower_text = ent_text.lower()

        # 条件1：NER标签可信
        is_ner_label = ent.label_ in ["CHEMICAL", "DRUG", "MOLECULE"]

        # 条件2：链接到ChEBI等化学数据库
        is_linked_to_chebi = (
            hasattr(ent._, "kb_ents") and ent._.kb_ents and 
            any("CHEBI" in e[0] for e in ent._.kb_ents)
        )

        # 条件3：正则规则辅助（匹配一些命名习惯）
        chem_name_pattern = re.compile(r".*(acid|amine|benzene|phenol|ether|aldehyde|ketone|chloride|hydroxide).*", re.IGNORECASE)
        is_name_like = bool(chem_name_pattern.match(ent_text))

        if (is_ner_label or is_linked_to_chebi or is_name_like) and lower_text not in seen_texts:
            seen_texts.add(lower_text)
            chemical_entities.append({
                "text": ent_text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })

    return chemical_entities


def create_templates():
    templates = [
        "The molecule {name}",
        "Chemical compound {name}",
        "{name} is a molecule",
        "{description}. The molecule",
        "The chemical {name} has properties",
        "{description}. It",
        "The molecule {smiles}",
        "Chemical structure {smiles}",
        "{smiles} is the SMILES of",
        "{description}. The SMILES is {smiles}",
        "Compound {name} has SMILES {smiles}"
    ]
    return templates

def create_gnn_templates():
    """创建GNN模型的模板"""
    templates = [
        "The SMILES string for [GNN] is {smiles}.",
        "[GNN] is a molecule that {description}",
        "The chemical structure [GNN] has SMILES {smiles}.",
        "[GNN] exhibits {description}",
        "The compound [GNN] can be represented as {smiles}.",
        "Molecule [GNN] is characterized by {description}",
        "[GNN] has the molecular structure {smiles}.",
        "The substance [GNN] is described as {description}",
        "[GNN] corresponds to the SMILES notation {smiles}.",
        "Chemical [GNN] shows {description}"
    ]
    return templates

def truncate_description_at_chemical(description, nlp):
    """在化学实体处截断描述"""
    entities = extract_chemical_entities(description, nlp)
    
    if not entities:
        # 如果没有找到化学实体，按句子截断
        sentences = description.split('.')
        if len(sentences) > 1:
            return sentences[0] + '.'
        return description
    
    # 找到第一个化学实体的位置并在其前截断
    first_entity = min(entities, key=lambda x: x['start'])
    truncated = description[:first_entity['start']].strip()
    
    # 确保截断后的文本以句子结尾
    if truncated and not truncated.endswith('.'):
        # 找到最后一个完整句子
        last_period = truncated.rfind('.')
        if last_period > 0:
            truncated = truncated[:last_period + 1]
        else:
            truncated = truncated + '.'
    
    return truncated if truncated else description

def process_chatmol_data(file_path, tokenizer, model, nlp, output_file, data_type="train"):
    """处理ChatMol数据集"""
    templates = create_templates()
    gnn_templates = create_gnn_templates()
    
    diffusion_data = []
    gnn_data = []
    
    print(f"Processing {data_type} data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Processing {data_type}"):
            try:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                cid, smiles, description = parts[0], parts[1], parts[2]
                
                # 验证SMILES
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue
                
                # 生成3D构象
                structure = generate_3d_structure(smiles)
                if structure is None:
                    continue
                coordinates = structure['coordinates']
                atom_types = structure['atom_types']
                bond_types = structure['bond_types']
                
                # 提取化学实体用于命名
                entities = extract_chemical_entities(description, nlp)
                chemical_names = [ent['text'] for ent in entities if len(ent['text']) > 2]
                
                # 如果没有找到化学名称，使用CID
                if not chemical_names:
                    chemical_names = [f"compound {cid}"]
                
                # 截断描述
                truncated_desc = truncate_description_at_chemical(description, nlp)
                
                # 生成Diffusion模型数据
                if len(diffusion_data)<150000:
                    selected_templates = random.sample(templates, 3)    
                    for template in selected_templates:
                        for name in chemical_names[:2]:  # 最多使用前2个化学名称
                            try:
                                # 处理不同类型的模板
                                if "{name}" in template and "{description}" in template and "{smiles}" in template:
                                    input_text = template.format(name=name, description=description, smiles=smiles)
                                elif "{name}" in template and "{smiles}" in template:
                                    input_text = template.format(name=name, smiles=smiles)
                                elif "{description}" in template and "{smiles}" in template:
                                    input_text = template.format(description=description, smiles=smiles)
                                elif "{name}" in template and "{description}" in template:
                                    input_text = template.format(name=name, description=description)
                                elif "{name}" in template:
                                    input_text = template.format(name=name)
                                elif "{description}" in template:
                                    input_text = template.format(description=truncated_desc)
                                elif "{smiles}" in template:
                                    input_text = template.format(smiles=smiles)
                                else:
                                    input_text = template
                                
                                # 获取输入embedding
                                input_embedding = get_embedding(input_text, tokenizer, model)
                                
                                diffusion_data.append({
                                    "cid": cid,
                                    "smiles": smiles,
                                    "description": description,
                                    "name": name,
                                    "input_text": input_text,
                                    "embedding": input_embedding.tolist(),
                                    "coordinates": coordinates.tolist(),
                                    "atom_types": atom_types,
                                    "bond_types": bond_types,
                                })
                            except Exception as e:
                                print(f"Error processing diffusion template: {e}")
                                continue
            except Exception as e:
                print(f"Error processing diffusion template: {e}")
                continue
        
    # 保存数据
    diffusion_file = output_file
    
    with open(diffusion_file, 'w') as f:
        json.dump(diffusion_data, f, indent=2)

    diffusion_file = diffusion_file.replace(".json", ".pkl")
    with open(diffusion_file, "wb") as f:
        pickle.dump(diffusion_data, f)
        
    print(f"Saved {len(diffusion_data)} diffusion samples to {diffusion_file}")


def main():
    """主函数"""
    # 初始化模型
    print("Initializing models...")
    tokenizer, model, nlp = setup_models()
    
    # 数据文件路径
    data_files = {
        "train": "/data1/chenyuxuan/Project/MSMLM/data/chatmol/train.txt"
    }
    # "validation": "/data1/chenyuxuan/Project/MSMLM/data/chatmol/validation.txt",
    # "test": "/data1/chenyuxuan/Project/MSMLM/data/chatmol/test.txt"
    # 处理每个数据集
    for data_type, file_path in data_files.items():
        output_file = f"/data1/chenyuxuan/Project/MSMLM/data/traindata/chatmol_data.json"
        try:
            process_chatmol_data(file_path, tokenizer, model, nlp, output_file, data_type)
        except Exception as e:
            print(f"Error processing {data_type} data: {e}")
    
    print("Data processing completed!")

if __name__ == "__main__":
    main()