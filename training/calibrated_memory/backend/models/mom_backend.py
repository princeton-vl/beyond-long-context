"""Backend for the MoM (Mixture-of-Memory) architecture."""

from __future__ import annotations

from typing import Any

from fla.models.mom import MomConfig, MomModel

from .fla_backend import FlaSequenceBackend, _build_fla_config

import torch


class MoMBackend(FlaSequenceBackend):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        ctx_len: int = 256,
        **config_overrides: Any,
    ) -> None:
        self.config_cls = MomConfig
        self.model_cls = MomModel
        autocast_dtype = config_overrides.pop("autocast_dtype", torch.bfloat16)
        mode = str(config_overrides.pop("mode", config_overrides.pop("attn_mode", "chunk")))
        config_overrides["mode"] = mode
        config_overrides["attn_mode"] = mode
        base_config = _build_fla_config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            ctx_len=ctx_len,
            overrides=config_overrides,
        )
        base_config.setdefault("num_heads", 1)
        base_config.setdefault("head_dim", int(embed_dim // base_config["num_heads"]) or int(embed_dim))
        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=base_config,
            ctx_len=ctx_len,
            autocast_dtype=autocast_dtype,
            forward_kwargs={"mode": mode},
        )


__all__ = ["MoMBackend"]
