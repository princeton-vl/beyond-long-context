from __future__ import annotations

from collections import defaultdict
from typing import Literal

import torch
import torch.nn as nn

try:
    from torch import amp as _amp_mod

    def _autocast(**kwargs):
        return _amp_mod.autocast(device_type="cuda", **kwargs)

except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _cuda_autocast

    def _autocast(**kwargs):  # type: ignore[no-redef]
        return _cuda_autocast(**kwargs)

from .external.ttt_fast.configuration_ttt import TTTConfig as FastTTTConfig
from .external.ttt_fast.modeling_ttt import (
    TTTLinearFast,
    TTTMLPFast,
    tk_ttt_linear_prefill,
    tk_ttt_mlp_prefill,
)


class _FastTTTCache:
    """Lightweight cache object matching the fast kernel API."""

    def __init__(
        self,
        layer: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.seqlen_offset = 0
        self.mini_batch_size = layer.mini_batch_size
        self.params_dict = defaultdict(dict)
        self._init_params(layer, batch_size, dtype, device)

    def _init_params(
        self,
        layer: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        layer_idx = layer.layer_idx
        tile_shape = (batch_size,) + (1,) * (layer.W1.dim() - 1)
        w1_state = torch.tile(layer.W1.detach(), tile_shape).to(device=device, dtype=dtype)
        self.params_dict["W1_init"][layer_idx] = w1_state.contiguous()
        self.params_dict["W1_grad"][layer_idx] = torch.zeros_like(w1_state)

        b1_tile_shape = (batch_size,) + (1,) * (layer.b1.dim() - 1)
        b1_state = torch.tile(layer.b1.detach(), b1_tile_shape).to(device=device, dtype=dtype)
        self.params_dict["b1_init"][layer_idx] = b1_state.contiguous()
        self.params_dict["b1_grad"][layer_idx] = torch.zeros_like(b1_state)

        if hasattr(layer, "W2") and hasattr(layer, "b2"):
            w2_tile_shape = (batch_size,) + (1,) * (layer.W2.dim() - 1)
            w2_state = torch.tile(layer.W2.detach(), w2_tile_shape).to(device=device, dtype=dtype)
            self.params_dict["W2_init"][layer_idx] = w2_state.contiguous()
            self.params_dict["W2_grad"][layer_idx] = torch.zeros_like(w2_state)

            b2_tile_shape = (batch_size,) + (1,) * (layer.b2.dim() - 1)
            b2_state = torch.tile(layer.b2.detach(), b2_tile_shape).to(device=device, dtype=dtype)
            self.params_dict["b2_init"][layer_idx] = b2_state.contiguous()
            self.params_dict["b2_grad"][layer_idx] = torch.zeros_like(b2_state)

        conv_cache = torch.zeros(
            batch_size,
            layer.hidden_size,
            layer.conv_kernel,
            dtype=dtype,
            device=device,
        )
        self.params_dict["conv_cache"][layer_idx] = conv_cache


class FastTTTLayerWrapper(nn.Module):
    """Adapts the fast Triton/ThunderKittens TTT kernels to the backend API."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mini_batch_size: int,
        mlp_ratio: int,
        layer_idx: int,
        variant: Literal["linear_fast", "mlp_fast"],
    ) -> None:
        super().__init__()
        if variant not in {"linear_fast", "mlp_fast"}:
            raise ValueError(f"Unsupported fast TTT variant: {variant}")
        if variant == "linear_fast" and tk_ttt_linear_prefill is None:
            raise RuntimeError(
                "Fast linear TTT kernels require the tk_ttt_linear_prefill extension."
            )
        if variant == "mlp_fast" and tk_ttt_mlp_prefill is None:
            raise RuntimeError(
                "Fast MLP TTT kernels require the tk_ttt_mlp_prefill extension."
            )

        config = FastTTTConfig(
            hidden_size=embed_dim,
            intermediate_size=embed_dim * mlp_ratio,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            mini_batch_size=mini_batch_size,
            seq_modeling_block="ttt-linear-fast" if variant == "linear_fast" else "ttt-mlp-fast",
            ttt_base_lr=1.0 if variant == "linear_fast" else 0.1,
            conv_before_ttt=False,
            conv_kernel=4,
        )
        config.use_compile = False
        config.fused_add_norm = False
        config.residual_in_fp32 = False
        config.dtype = torch.float32

        if variant == "linear_fast":
            self.layer = TTTLinearFast(config=config, layer_idx=layer_idx)
        else:
            self.layer = TTTMLPFast(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_params=None,
    ) -> torch.Tensor:
        del attention_mask, cache_params
        with _autocast(enabled=False):
            batch, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            orig_dtype = hidden_states.dtype
            compute_dtype = torch.float16
            if orig_dtype != compute_dtype:
                hidden_states = hidden_states.to(compute_dtype)
            self.layer = self.layer.to(device=device, dtype=compute_dtype)
            self.layer.config.dtype = compute_dtype

            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

            hidden_states, position_ids, pad_len = self._pad_sequence(hidden_states, position_ids)
            cache = _FastTTTCache(self.layer, batch, compute_dtype, device)
            outputs = self.layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                cache_params=cache,
                is_prefill=True,
                is_last_in_mini_batch=True,
            )
            if pad_len:
                outputs = outputs[:, :-pad_len, :]
            outputs = outputs.to(orig_dtype)
        return outputs

    def _pad_sequence(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        mini_batch = self.layer.mini_batch_size
        seq_len = hidden_states.size(1)
        remainder = seq_len % mini_batch
        if remainder == 0:
            return hidden_states, position_ids, 0
        pad_len = mini_batch - remainder
        batch = hidden_states.size(0)
        dim = hidden_states.size(2)
        pad_states = hidden_states.new_zeros(batch, pad_len, dim)
        hidden_states = torch.cat([hidden_states, pad_states], dim=1)

        last_positions = position_ids[:, -1:]
        offset = torch.arange(1, pad_len + 1, device=hidden_states.device)
        pad_positions = last_positions + offset
        position_ids = torch.cat([position_ids, pad_positions], dim=1)
        return hidden_states, position_ids, pad_len
