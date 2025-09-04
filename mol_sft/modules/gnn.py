# modules/gnn_encoder.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List

class SimpleSmilesGNN(nn.Module):
    """
    一个无依赖的占位“GNN”：字符嵌入 + GRU -> 向量
    你可以把它替换为真正的图神经网络（例如使用你已有的分子图特征、PyG 等）。
    """
    def __init__(self, hidden_dim: int = 256, vocab: str = None):
        super().__init__()
        if vocab is None:
            vocab = "CNOFPSI[]()=#+-\\/1234567890@"
        self.vocab = vocab
        self.stoi = {ch: i+1 for i, ch in enumerate(vocab)}  # 0留给UNK
        self.emb = nn.Embedding(len(self.stoi) + 1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, smiles: List[str]) -> torch.Tensor:
        """
        输入：list[str] 的 SMILES
        输出：[B, hidden_dim] 的分子向量
        """
        max_len = max(len(s) for s in smiles)
        idx = torch.zeros(len(smiles), max_len, dtype=torch.long, device=self.emb.weight.device)
        for i, s in enumerate(smiles):
            ids = [self.stoi.get(ch, 0) for ch in s[:max_len]]
            idx[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=idx.device)
        x = self.emb(idx)                             # [B, L, H]
        _, h = self.gru(x)                            # h: [1, B, H]
        return h.squeeze(0)                           # [B, H]

    @torch.no_grad()
    def encode_smiles(self, smiles: List[str]) -> torch.Tensor:
        self.eval()
        return self.forward(smiles)
