"""Backend for the DeltaNet architecture from flash-linear-attention."""

from __future__ import annotations

from typing import Any

from fla.models.delta_net import DeltaNetConfig, DeltaNetModel

from .fla_backend import FlaSequenceBackend, _build_fla_config

import torch


class DeltaNetBackend(FlaSequenceBackend):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        ctx_len: int = 256,
        **config_overrides: Any,
    ) -> None:
        self.config_cls = DeltaNetConfig
        self.model_cls = DeltaNetModel
        autocast_dtype = config_overrides.pop("autocast_dtype", torch.bfloat16)
        base_config = _build_fla_config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            ctx_len=ctx_len,
            overrides=config_overrides,
        )
        base_config.setdefault("num_heads", 1)
        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=base_config,
            ctx_len=ctx_len,
            autocast_dtype=autocast_dtype,
        )


__all__ = ["DeltaNetBackend"]
