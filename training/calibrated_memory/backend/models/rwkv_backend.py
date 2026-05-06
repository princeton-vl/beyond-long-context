"""Flash-linear RWKV backend built on top of RWKV7 kernels."""

from __future__ import annotations

import torch

from fla.models.rwkv7 import RWKV7Config, RWKV7Model

from .fla_backend import FlaSequenceBackend, _build_fla_config


class RWKVBackend(FlaSequenceBackend):
    """Flash-linear RWKV7 backend emitting decoder-aligned slots."""

    _DEFAULT_AUTODTYPE = torch.bfloat16

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int = 4,
        ffn_mult: int = 4,
        ctx_len: int = 256,
        num_heads: int | None = None,
        head_dim: int | None = None,
        autocast_dtype: torch.dtype | str | None = None,
    ) -> None:
        resolved_heads = self._resolve_head_count(num_heads, embed_dim)
        overrides = {
            "num_heads": resolved_heads,
            "head_dim": self._resolve_head_dim(head_dim, embed_dim, resolved_heads),
            "attn_mode": "chunk",
            "use_cache": False,
        }
        config = _build_fla_config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            ctx_len=ctx_len,
            overrides=overrides,
        )
        self.config_cls = RWKV7Config
        self.model_cls = RWKV7Model
        resolved_autocast = self._resolve_autocast_dtype(autocast_dtype)
        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=config,
            ctx_len=ctx_len,
            autocast_dtype=resolved_autocast,
        )
        self.ffn_mult = ffn_mult
        self.supports_direct_logits = True

    @staticmethod
    def _resolve_head_count(requested: int | None, embed_dim: int) -> int:
        if requested is not None and requested > 0:
            return requested
        inferred = max(1, embed_dim // 64)
        return inferred

    @staticmethod
    def _resolve_head_dim(head_dim: int | None, embed_dim: int, num_heads: int | None) -> int:
        if head_dim is not None and head_dim > 0:
            return head_dim
        heads = num_heads or max(1, embed_dim // 64)
        base = embed_dim // heads
        return max(16, base)

    @classmethod
    def _resolve_autocast_dtype(cls, value: torch.dtype | str | None) -> torch.dtype | None:
        if value is None:
            return cls._default_autocast()
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16"}:
            return torch.float16
        if normalized in {"none", "fp32", "float32"}:
            return None
        raise ValueError(f"Unknown autocast dtype '{value}' for RWKVBackend")

    @classmethod
    def _default_autocast(cls) -> torch.dtype | None:
        if torch.cuda.is_available():
            return cls._DEFAULT_AUTODTYPE
        return None


__all__ = ["RWKVBackend"]
