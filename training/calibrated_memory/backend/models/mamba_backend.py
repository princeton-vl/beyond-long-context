from __future__ import annotations

import math

import torch
import torch.nn as nn

from .backend_base import MemoryBackend, SequenceInputs
from .modules import RMSNorm
from mamba_ssm import Mamba2


def _enumerate_divisors(value: int) -> list[int]:
    if value <= 0:
        raise ValueError("Divisor enumeration requires a positive value")
    divisors: set[int] = set()
    limit = int(math.sqrt(value))
    for candidate in range(1, limit + 1):
        if value % candidate != 0:
            continue
        divisors.add(candidate)
        divisors.add(value // candidate)
    return sorted(divisors)


def _resolve_headdim(
    embed_dim: int,
    expand: int,
    requested: int | None,
    *,
    strict: bool,
) -> int:
    if embed_dim <= 0 or expand <= 0:
        raise ValueError("embed_dim and expand must be positive")
    total = embed_dim * expand
    if requested is not None and requested <= 0:
        raise ValueError("headdim must be positive when provided")
    if requested is not None and total % requested == 0:
        return requested
    divisors = _enumerate_divisors(total)
    if not divisors:
        raise ValueError(f"Unable to factor expand * embed_dim (value={total})")
    if requested is not None and strict:
        formatted = ", ".join(str(div) for div in divisors)
        raise ValueError(
            "MambaBackend received headdim="
            f"{requested}, but expand * embed_dim = {total} requires one of: {formatted}."
        )
    target = requested if requested is not None else min(64, total)
    target = max(1, min(target, total))
    best = min(
        divisors,
        key=lambda div: (abs(div - target), -div),
    )
    return best


class MambaBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        layer_idx: int,
        headdim: int,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.mamba = Mamba2(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mem_eff_path=False,
            layer_idx=layer_idx,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)
        self.headdim = headdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x).contiguous()
        # Mamba's Triton kernels only work on CUDA, but they avoid stride requirements that
        # the causal_conv1d CUDA path enforces. Flip the flag dynamically based on device.
        use_mem_eff = normed.is_cuda
        if getattr(self.mamba, "use_mem_eff_path", False) != use_mem_eff:
            self.mamba.use_mem_eff_path = use_mem_eff
        y = self.mamba(normed)
        return x + self.dropout(y)


class MambaBackend(MemoryBackend):
    """Backend that stacks lightweight Mamba2 layers over the token embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int | None = None,
        *,
        headdim_was_overridden: bool = False,
    ) -> None:
        super().__init__(
            embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        resolved_headdim = _resolve_headdim(
            embed_dim,
            expand,
            headdim,
            strict=headdim_was_overridden,
        )
        self.headdim = resolved_headdim
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    embed_dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    layer_idx=i,
                    headdim=resolved_headdim,
                )
                for i in range(max(1, num_layers))
            ]
        )
        self.final_norm = RMSNorm(embed_dim)
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("MambaBackend requires token embeddings.")
        hidden = token_embeddings
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        padding_mask = self._resolve_padding_mask(sequence)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
