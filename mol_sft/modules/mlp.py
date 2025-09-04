# modules/mlp_heads.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ===== 版本 B：更贴近 diffusion 训练常用的 MLP（稳定性更强） =====
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class DiffusionStyleMLP(nn.Module):
    """
    结构：x -> RMSNorm -> Linear -> SiLU -> Dropout -> Linear(zero-init) -> +residual
    - 末层权重/偏置零初始化（或很小的尺度），便于稳定收敛（常见于 EDM/DiT/ControlNet 的 head）
    - 自带 residual，有利于保留原始上下文信息
    """
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0, zero_init=True, residual=True):
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(d_hidden, d_out)
        self.residual = residual and (d_in == d_out)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        if zero_init:
            nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        if self.residual:
            y = x + y
        return y

# === 对外导出两个头（和之前名字保持一致） ===
class DiffusionMLP(nn.Module):
    """把 LLM d_model -> diffusion 条件维度"""
    def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
        super().__init__()
        if variant == "simple":
            self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
        else:
            self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=True, residual=False)
    def forward(self, x): return self.proj(x)

class GNNMLP(nn.Module):
    """把 GNN 向量 -> LLM d_model"""
    def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
        super().__init__()
        if variant == "simple":
            self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
        else:
            self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=False, residual=False)
    def forward(self, x): return self.proj(x)
