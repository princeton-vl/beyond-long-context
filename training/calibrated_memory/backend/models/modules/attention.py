from __future__ import annotations

import math
from contextlib import contextmanager
import os
from typing import Iterator, Optional


_FLASH_ATTENTION_MAX_HEAD_DIM = 256

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import PoPEPositionEncoding, PositionalMode, RotaryPositionEncoding


@contextmanager
def _sdp_kernel(use_flash: bool) -> Iterator[None]:
    if torch.cuda.is_available():
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
            if not use_flash:
                backends = backends[1:]
            ctx = torch.nn.attention.sdpa_kernel(backends, set_priority=True)
        else:  # pragma: no cover - legacy fallback
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=use_flash,
                enable_math=not use_flash,
                enable_mem_efficient=not use_flash,
            )
        with ctx:
            yield
    else:
        yield


class AttentionCore(nn.Module):
    """Multi-head attention with optional RoPE/PoPE and QK normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        positional_mode: PositionalMode = PositionalMode.NONE,
        rotary_base: float = 10000.0,
        pope_theta_base: float = 10000.0,
        pope_bias_init: str = "zero",
        use_flash_attention: bool = True,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        sliding_window: int = -1 # if above 0, implements sliding window attention
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_dropout = attn_dropout
        self.positional_mode = positional_mode
        self.use_flash_attention = use_flash_attention
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(attn_dropout)
        self.sliding_window = sliding_window
        self._debug_enabled = os.environ.get("ATTN_DEBUG", "0") not in {"0", ""}
        if self._debug_enabled:
            self._install_grad_hooks()
        if positional_mode is PositionalMode.ROPE:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires an even head dimension")
            self.positional = RotaryPositionEncoding(self.head_dim, base=rotary_base)
        elif positional_mode is PositionalMode.POPE:
            self.positional = PoPEPositionEncoding(
                self.head_dim,
                num_heads=self.num_heads,          # <-- add this
                theta_base=pope_theta_base,
                bias_init=pope_bias_init,
            )
        else:
            self.positional = None

    @property
    def requires_explicit_causal_mask(self) -> bool:
        return self.positional_mode is PositionalMode.POPE

    def _install_grad_hooks(self) -> None:
        def make_hook(name: str):
            def _hook(grad: torch.Tensor | None) -> torch.Tensor | None:
                if grad is None:
                    print(f"[attention-grad] {name}: None")
                    return None
                norm = grad.detach().float().norm().item()
                print(f"[attention-grad] {name}: shape={tuple(grad.shape)} norm={norm:.4e}")
                return grad

            return _hook

        for param_name, param in self.named_parameters():
            param.register_hook(make_hook(param_name))

    def _describe_mask(self, tensor: Optional[torch.Tensor]) -> str:
        if tensor is None:
            return "None"
        if tensor.dtype == torch.bool:
            true_count = int(tensor.to(dtype=torch.int64).sum().item())
            return f"bool(true={true_count})"
        return f"float(shape={tuple(tensor.shape)})"

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        query_positions: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = query if key is None else key
        value = key if value is None else value
        batch_q, seq_q, _ = query.shape
        batch_k, seq_k, _ = key.shape
        if batch_q != batch_k:
            raise ValueError("Query and key batches must match")
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_q, seq_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_k, seq_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_k, seq_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self._debug_enabled:
            print(
                f"[attention] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} attn_mask={self._describe_mask(attn_mask)} pad_mask={self._describe_mask(key_padding_mask)} is_causal={is_causal}"
            )
        if self.sliding_window > 0:
            # we assume sliding window is always causal
            W = self.sliding_window
            if query_positions is not None and key_positions is not None:
                pass
            else:
                query_positions = torch.arange(seq_q, device=query.device, dtype=torch.long)
                key_positions   = torch.arange(seq_k, device=key.device,   dtype=torch.long)
            query_positions = query_positions.to(device=query.device)
            key_positions   = key_positions.to(device=key.device)
            qpos = query_positions[:, None]
            kpos = key_positions[None, :]

            blocked = (qpos < kpos) | (kpos < qpos - (W-1))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    blocked = blocked.to(attn_mask.device)
                    attn_mask = attn_mask | blocked
                else:
                    if not attn_mask.is_floating_point():
                        raise TypeError("attn_mask must be bool or floating additive bias")
                    # its a float, convert blocked to 0 inf mask
                    blocked = blocked.to(attn_mask.device)
                    blocked_bias = torch.zeros_like(
                        blocked, dtype=attn_mask.dtype, device=attn_mask.device
                    )
                    blocked_bias = blocked_bias.masked_fill(
                        blocked, torch.finfo(attn_mask.dtype).min
                    )
                    attn_mask = attn_mask + blocked_bias
            else:
                attn_mask = blocked
            

        if self.positional_mode is PositionalMode.ROPE:
            if query_positions is None:
                query_positions = torch.arange(seq_q, device=query.device, dtype=torch.long)
            if key_positions is None:
                key_positions   = torch.arange(seq_k, device=key.device,   dtype=torch.long)
            q, k = self.positional.apply(q, k, query_positions, key_positions)
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        if self.positional_mode is PositionalMode.POPE:
            if query_positions is None or key_positions is None:
                raise ValueError("PoPE attention requires explicit query/key positions")
            use_pope_flash = self.use_flash_attention and (self.head_dim * 2) <= _FLASH_ATTENTION_MAX_HEAD_DIM
            if use_pope_flash:
                pope_q, pope_k = self.positional.project_real_components(
                    q,
                    k,
                    positions_q=query_positions,
                    positions_k=key_positions,
                )
                context = self._flash_attention(
                    pope_q * math.sqrt(2.0),
                    pope_k,
                    v,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                )
            else:
                context = self._pope_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
        else:
            context = self._flash_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_q, seq_q, self.embed_dim)
        return self.out_proj(context)

    def _build_attention_bias(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        *,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        bias: Optional[torch.Tensor] = None
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(device=device)
            if mask.dtype == torch.bool:
                float_mask = torch.zeros_like(mask, dtype=dtype)
                float_mask = float_mask.masked_fill(mask, torch.finfo(dtype).min)
                mask = float_mask
            bias = mask
        if key_padding_mask is not None:
            pad = key_padding_mask.to(device=device, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            pad_bias = torch.zeros_like(pad, dtype=dtype).masked_fill(
                pad, torch.finfo(dtype).min
            )
            bias = pad_bias if bias is None else bias + pad_bias
        return bias

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        batch, _, query_len, _ = q.shape
        _, _, key_len, _ = k.shape
        attn_bias = self._build_attention_bias(
            attn_mask,
            key_padding_mask,
            query_len=query_len,
            key_len=key_len,
            device=q.device,
            dtype=q.dtype,
        )
        dropout_p = self.attn_dropout if self.training else 0.0
        context: torch.Tensor
        use_causal = is_causal and attn_bias is None
        with _sdp_kernel(self.use_flash_attention and use_causal):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=use_causal,
            )
        return context

    def _pope_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
        key_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = self.positional.compute_logits(
            q,
            k,
            positions_q=query_positions,
            positions_k=key_positions,
            scale=self.scale,
        )
        attn_bias = self._build_attention_bias(
            attn_mask,
            key_padding_mask,
            query_len=q.size(2),
            key_len=k.size(2),
            device=logits.device,
            dtype=logits.dtype,
        )
        if attn_bias is not None:
            logits = logits + attn_bias
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)
