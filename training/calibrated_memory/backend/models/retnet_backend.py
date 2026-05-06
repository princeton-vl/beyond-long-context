"""Backend that wraps the RetNet model from flash-linear-attention."""

from __future__ import annotations

from typing import Any

import torch
from fla.models.retnet import RetNetConfig, RetNetModel

from .fla_backend import FlaSequenceBackend, _build_fla_config


class RetNetBackend(FlaSequenceBackend):
    """Expose the RetNet retention stack as a memory backend."""

    config_cls = RetNetConfig
    model_cls = RetNetModel

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int = 4,
        ctx_len: int = 512,
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
        base_config.setdefault("num_heads", 1)
        num_heads = int(base_config["num_heads"])
        num_kv_heads = base_config.get("num_kv_heads")
        if num_kv_heads in {None, "None"}:
            base_config["num_kv_heads"] = num_heads
        else:
            base_config["num_kv_heads"] = int(num_kv_heads)

        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=base_config,
            ctx_len=ctx_len,
            autocast_dtype=autocast_dtype,
        )


__all__ = ["RetNetBackend"]
