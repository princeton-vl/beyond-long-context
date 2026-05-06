from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dimension must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        norm = hidden.pow(2).mean(dim=-1, keepdim=True)
        scaled = hidden * torch.rsqrt(norm + self.eps)
        return scaled * self.weight
