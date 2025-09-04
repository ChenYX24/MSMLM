# modules/diffusion.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class ToyDiffusion(nn.Module):
    """
    一个占位 Diffusion：给定条件向量 cond -> 线性投影 -> 高斯采样叠加
    用于在训练时跑通前向，不追求物化正确性。
    """
    def __init__(self, cond_dim: int, latent_dim: int):
        super().__init__()
        self.cond_to_latent = nn.Linear(cond_dim, latent_dim)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 10) -> torch.Tensor:
        # 简单可重复的“采样”
        base = self.cond_to_latent(cond)             # [B, D]
        noise = torch.randn_like(base) / (steps ** 0.5)
        return base + noise
