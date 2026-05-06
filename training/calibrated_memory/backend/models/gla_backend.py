"""Backend wrapping the Gated Linear Attention model from flash-linear-attention."""

from __future__ import annotations

from typing import Any

import torch
from fla.models.gla import GLAConfig, GLAModel

from .fla_backend import FlaSequenceBackend, _build_fla_config


class GLABackend(FlaSequenceBackend):
    """Expose the flash-linear-attention GLA stack through the MemoryBackend interface."""

    config_cls = GLAConfig
    model_cls = GLAModel

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int = 6,
        ctx_len: int = 2048,
        **config_overrides: Any,
    ) -> None:
        overrides = dict(config_overrides)
        autocast_dtype = overrides.pop("autocast_dtype", torch.bfloat16)
        base_config = _build_fla_config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            ctx_len=ctx_len,
            overrides=overrides,
        )
        # Ensure required attention layout knobs are set when omitted.
        base_config.setdefault("num_heads", 4)
        num_kv_heads = base_config.get("num_kv_heads")
        if num_kv_heads in {None, "None"}:
            base_config["num_kv_heads"] = base_config["num_heads"]
        else:
            base_config["num_kv_heads"] = int(num_kv_heads)
        base_config.setdefault("hidden_ratio", 4)
        base_config.setdefault("feature_map", None)
        base_config.setdefault("attn_mode", "chunk")

        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=base_config,
            ctx_len=ctx_len,
            autocast_dtype=autocast_dtype,
        )


__all__ = ["GLABackend"]
