import pickle
import random
from rdkit import Chem
from tqdm import tqdm

def extract_atom_bond_types(smiles):
    """根据 SMILES 生成 atom_types 和 bond_types"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_types = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        bond_types.append([start, end, btype])
    return atom_types, bond_types

def update_and_save(input_pkl_path, output_pkl_path, sample_size=100_000):
    # 读取原始数据
    with open(input_pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} samples from {input_pkl_path}")

    # 随机抽样
    sampled_data = random.sample(data, min(sample_size, len(data)))
    updated_data = []

    for item in tqdm(sampled_data, desc="Updating atom/bond types"):
        smiles = item["smiles"]
        atom_types, bond_types = extract_atom_bond_types(smiles)
        if atom_types is None or bond_types is None:
            continue
        item["atom_types"] = atom_types
        item["bond_types"] = bond_types
        updated_data.append(item)

    print(f"{len(updated_data)} samples updated with atom/bond types")

    # 保存更新后的数据
    with open(output_pkl_path, "wb") as f:
        pickle.dump(updated_data, f)

    print(f"Saved to {output_pkl_path}")


# 更新 diffusion 数据
update_and_save(
    input_pkl_path="/data1/chenyuxuan/Project/MSMLM/data/processed_train_data_diffusion.pkl",
    output_pkl_path="/data1/chenyuxuan/Project/MSMLM/data/traindata/processed_train_data_diffusion.pkl"
)

# 更新 GNN 数据
update_and_save(
    input_pkl_path="/data1/chenyuxuan/Project/MSMLM/data/processed_train_data_gnn.pkl",
    output_pkl_path="/data1/chenyuxuan/Project/MSMLM/data/traindata/processed_train_data_gnn.pkl"
)
