from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalMode(str, Enum):
    NONE = "none"
    ROPE = "rope"
    POPE = "pope"


def _build_position_index(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionEncoding(nn.Module):
    """Classic rotary embeddings shared across attention heads."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, positions: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.to(dtype=self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_q: torch.Tensor,
        positions_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_q, sin_q = self._cos_sin(positions_q, dtype=q.dtype)
        cos_k, sin_k = self._cos_sin(positions_k, dtype=k.dtype)
        cos_q = cos_q[None, None, :, :]
        sin_q = sin_q[None, None, :, :]
        cos_k = cos_k[None, None, :, :]
        sin_k = sin_k[None, None, :, :]
        q_rot = (q * cos_q) + (_rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (_rotate_half(k) * sin_k)
        return q_rot, k_rot


class PoPEPositionEncoding(nn.Module):
    """Polar-coordinate positional embedding (PoPE)."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        theta_base: float = 10000.0,
        bias_init: str = "uniform",
        bias_min: float = -2 * math.pi,
        bias_max: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("PoPE dimension must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if theta_base <= 0:
            raise ValueError("theta_base must be positive")
        self.num_heads = int(num_heads)
        span = torch.arange(dim, dtype=torch.float32)
        theta = theta_base ** (span / dim)
        self.register_buffer("theta", theta, persistent=False)
        self.bias_min = bias_min
        self.bias_max = bias_max
        if bias_init == "zero":
            bias = torch.zeros(self.num_heads, dim)
        elif bias_init == "uniform":
            bias = torch.empty(self.num_heads, dim).uniform_(bias_min, bias_max)
        else:
            raise ValueError("bias_init must be either 'zero' or 'uniform'")
        self.bias = nn.Parameter(bias)

    def _bounded_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bias = self.bias.to(device=device, dtype=dtype)
        return torch.clamp(bias, self.bias_min, self.bias_max)

    def _polar_components(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
        *,
        add_bias: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            raise ValueError("PoPE requires explicit positions")
        device = tensor.device
        out_dtype = tensor.dtype
        phase_dtype = torch.float32 if out_dtype in {torch.float16, torch.bfloat16} else out_dtype
        theta = self.theta.to(device=device, dtype=phase_dtype)
        pos = positions.to(device=device, dtype=phase_dtype).view(1, 1, -1, 1)
        phases = pos * theta.view(1, 1, 1, -1)
        if add_bias:
            bias = self._bounded_bias(device, phase_dtype)
            phases = phases + bias.view(1, self.num_heads, 1, -1)
        magnitude = F.softplus(tensor.to(torch.float32)).to(out_dtype)
        cos_phase = torch.cos(phases).to(out_dtype)
        sin_phase = torch.sin(phases).to(out_dtype)
        return magnitude * cos_phase, magnitude * sin_phase

    def project_real_components(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        positions_q: torch.Tensor,
        positions_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_x, q_y = self._polar_components(q, positions_q, add_bias=False)
        k_x, k_y = self._polar_components(k, positions_k, add_bias=True)
        q_real = torch.cat((q_x, q_y), dim=-1)
        k_real = torch.cat((k_x, k_y), dim=-1)
        return q_real, k_real

    def compute_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_q: torch.Tensor,
        positions_k: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        q_x, q_y = self._polar_components(q, positions_q, add_bias=False)
        k_x, k_y = self._polar_components(k, positions_k, add_bias=True)
        logits = torch.matmul(q_x, k_x.transpose(-1, -2)) + torch.matmul(q_y, k_y.transpose(-1, -2))
        return logits * scale
