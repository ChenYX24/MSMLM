import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from gvp import GVP, GVPConvLayer
from gvp import LayerNorm
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import logging
from typing import Optional
# ============================ 依赖和配置 ============================
# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
warnings.filterwarnings("ignore", message="Pandas requires version")

# 定义原子类型（根据你提供的代码）
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

# 辅助函数：计算边的几何特征
def _get_edge_features_from_coords(coords: torch.Tensor, edge_index: torch.Tensor):
    if edge_index is None or edge_index.numel() == 0:
        device = coords.device if coords is not None else 'cpu'
        return torch.empty(0, 1, 3, dtype=torch.float, device=device), \
               torch.empty(0, 1, dtype=torch.float, device=device)
    
    row, col = edge_index
    edge_vectors = coords[col] - coords[row]
    edge_vectors_gvp_format = edge_vectors.unsqueeze(1)
    edge_length_scalar = edge_vectors.norm(dim=-1, keepdim=True)
    return edge_vectors_gvp_format, edge_length_scalar

# ============================ GVP Encoder 类 ============================
class GVPEncoder(nn.Module):
    def __init__(self,
                 node_dims=(5, 3),
                 edge_dims=(5, 3),
                 hidden_scalar_dim=256,
                 hidden_vector_dim=1,
                 output_dim=128,
                 num_layers=3):
        super().__init__()

        self.node_input = nn.Sequential(
            GVP(node_dims, (hidden_scalar_dim, hidden_vector_dim), activations=(F.relu, None)),
            LayerNorm((hidden_scalar_dim, hidden_vector_dim))
        )
        
        self.edge_input = GVP(edge_dims, (hidden_scalar_dim, hidden_vector_dim), h_dim=hidden_scalar_dim)

        self.convs = nn.ModuleList([
            GVPConvLayer((hidden_scalar_dim, hidden_vector_dim), (hidden_scalar_dim, hidden_vector_dim), activations=(F.relu, None))
            for _ in range(num_layers)
        ])

        self.project = nn.Sequential(
            GVP((hidden_scalar_dim, hidden_vector_dim), (output_dim, 1), activations=(None, None))
        )
        
        self.output_dim = output_dim

    # ★ 新增方法：SMILES 到 PyG Data 对象转换
    def _smiles_to_data(self, smiles: str) -> Optional[Data]:
        """
        将 SMILES 字符串转换为 PyG Data 对象。
        此方法封装了离线预处理脚本中的分子解析逻辑。
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            mol = Chem.RemoveHs(mol)
            if mol.GetNumAtoms() == 0:
                return None

            # --- 1. 原子特征 (x) ---
            x_list = []
            for atom in mol.GetAtoms():
                onehot = [0] * len(ATOM_TYPES)
                if atom.GetSymbol() in ATOM_TYPES:
                    onehot[ATOM_TYPES.index(atom.GetSymbol())] = 1
                x_list.append(onehot)
            x = torch.tensor(np.array(x_list, dtype=np.float32), dtype=torch.float)
            
            # 增加原子杂化特征以匹配 GVP 节点维度
            # 这是一个简化的假设，更完整的特征需要更多的RDKit调用
            # 这里我只使用你原始脚本中的onehot，所以节点的标量维度是9
            # 你的GVPEncoder init中的node_dims[0]应该是9
            if self.node_input.gvp_layer.V_in != len(ATOM_TYPES):
                 logging.warning(f"GVP node_dims[0] ({self.node_input.gvp_layer.V_in}) does not match atom features ({len(ATOM_TYPES)}).")
            
            # --- 2. 边索引 (edge_index) 和 边特征 (edge_attr) ---
            edge_index, edge_attr = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                btype = bond.GetBondType()
                bt_onehot = [
                    float(btype == Chem.rdchem.BondType.SINGLE),
                    float(btype == Chem.rdchem.BondType.DOUBLE),
                    float(btype == Chem.rdchem.BondType.TRIPLE),
                    float(btype == Chem.rdchem.BondType.AROMATIC),
                    float(btype == Chem.rdchem.BondType.AROMATIC)
                ]
                edge_index += [[i, j], [j, i]]
                edge_attr += [bt_onehot, bt_onehot]
            edge_index = torch.tensor(np.array(edge_index, dtype=np.int64).T, dtype=torch.long)
            # 你原始脚本的edge_attr是np.float32，这里我转为tensor
            edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32), dtype=torch.float)

            # --- 3. 3D 坐标 (pos) ---
            coords = self._embed_3d_coords_rdkit(mol)
            if coords is None:
                # 警告：这里没有 OpenBabel 备份，因为它需要额外的库，
                # 你需要自行决定是否添加
                return None
            pos = torch.tensor(coords, dtype=torch.float)
            x_vector = pos.clone().unsqueeze(1) # [N, 1, 3]

            # --- 4. 边的矢量特征 ---
            edge_attr_vector, edge_length_scalar = _get_edge_features_from_coords(pos, edge_index)

            # --- 5. 组装 PyG Data 对象 ---
            data = Data(
                x=x,
                x_vector=x_vector,
                edge_index=edge_index,
                edge_attr=edge_attr, # 你原代码中的 edge_attr 是键特征，但GVP的edge_attr是标量
                edge_attr_vector=edge_attr_vector,
                pos=pos,
                smiles=smiles,
                edge_scalar=edge_length_scalar
            )
            return data

        except Exception as e:
            logging.error(f"Failed to convert smiles to Data object for {smiles}: {e}")
            return None
    
    # ★ 新增方法：封装RDKit 3D嵌入逻辑
    def _embed_3d_coords_rdkit(self, mol, max_iters=200):
        try:
            n_atoms = mol.GetNumAtoms()
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                for seed in [0, 1, 42, 123]:
                    params.randomSeed = seed
                    if AllChem.EmbedMolecule(mol, params) == 0:
                        break
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
                except Exception:
                    pass
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=np.float32)
            return coords
        except Exception:
            return None
    
    def forward(self, data):
        node_scalar, node_vector = data.x, data.x_vector 
        
        # GVP的edge_attr是标量，edge_attr_vector是矢量
        # 我们在这里使用 `data.edge_scalar` 作为 GVP 的边标量特征
        # 并且使用 `data.edge_attr_vector` 作为矢量特征
        edge_scalar = data.edge_scalar if hasattr(data, 'edge_scalar') else data.edge_attr
        edge_vector = data.edge_attr_vector

        node_feats = (node_scalar, node_vector)
        edge_feats = (edge_scalar, edge_vector)

        h_V = self.node_input(node_feats)
        h_E = self.edge_input(edge_feats)

        for conv in self.convs:
            h_V = conv(h_V, data.edge_index, h_E)

        out_scalar, out_vector = self.project(h_V)
        
        # GVP层输出的标量维度是 [num_nodes, output_dim]，batch是一个单独的tensor
        graph_embeddings = global_mean_pool(out_scalar, data.batch)
        
        return graph_embeddings

    # ★ 新增方法：在线处理 SMILES 并返回图嵌入
    def forward_from_smiles(self, smiles: str):
        # 1. 将 SMILES 转换为 PyG Data 对象
        data = self._smiles_to_data(smiles)
        
        if data is None:
            # 如果 SMILES 无效，返回一个零向量
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_dim, device=device)

        # 2. 将单个 Data 对象转换为 PyG 支持的批次格式
        # PyG 的 `global_mean_pool` 需要一个 `batch` 属性来区分图
        # 对于单个图，我们只需创建一个全零的批次张量
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.project[0].weight.device)
        
        # 3. 将数据移动到正确的设备
        data = data.to(self.project[0].weight.device)

        # 4. 运行前向传播
        graph_embeddings = self.forward(data)

        return graph_embeddings