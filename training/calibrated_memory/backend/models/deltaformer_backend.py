"""Direct-mode backend for HuggingFace DeltaFormer models."""

from __future__ import annotations

from typing import Any

import torch

from fla.models.deltaformer.configuration_deltaformer import DeltaFormerConfig
from fla.models.deltaformer.modeling_deltaformer import DeltaFormerModel

from .backend_base import MemoryBackend, SequenceInputs


_CHUNK_KERNELS_INSTALLED = False


def _ensure_chunk_kernels() -> None:
    """Swap in the local Triton kernels when chunk mode is requested."""

    global _CHUNK_KERNELS_INSTALLED
    if _CHUNK_KERNELS_INSTALLED:
        return
    from .external import deltaformer_parallel_debug as chunk_impl
    import fla.ops.deltaformer as fla_deltaformer_ops
    import fla.layers.deltaformer as fla_deltaformer_layer

    fla_deltaformer_ops.deltaformer_attn = chunk_impl.deltaformer_attn
    fla_deltaformer_layer.deltaformer_attn = chunk_impl.deltaformer_attn
    _CHUNK_KERNELS_INSTALLED = True


class DeltaFormerBackend(MemoryBackend):
    """Runs DeltaFormer directly on concatenated stream/query embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        ctx_len: int = 512,
        hidden_ratio: float = 4.0,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        attn_mode: str = "parallel",
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float = 10000.0,
        vocab_size: int = 32000,
        autocast_dtype: torch.dtype | None = None,
        **config_overrides: Any,
    ) -> None:
        super().__init__(
            output_dim=embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        self.supports_direct_logits = True
        self._validate_layout(embed_dim, num_heads)
        self.ctx_len = int(ctx_len)
        if num_kv_heads not in (None, "None") and int(num_kv_heads) != int(num_heads):
            raise ValueError("DeltaFormerBackend currently requires num_kv_heads to match num_heads")
        normalized_attn_mode = str(attn_mode)
        if normalized_attn_mode.lower() == "chunk":
            _ensure_chunk_kernels()
        config_kwargs = dict(config_overrides)
        config_kwargs.update(
            {
                "hidden_size": int(embed_dim),
                "num_hidden_layers": int(num_layers),
                "max_position_embeddings": int(ctx_len),
                "num_heads": int(num_heads),
                "num_kv_heads": None if num_kv_heads in {None, "None"} else int(num_kv_heads),
                "hidden_ratio": float(hidden_ratio),
                "attn_mode": normalized_attn_mode,
                "qkv_bias": bool(qkv_bias),
                "qk_norm": bool(qk_norm),
                "rope_theta": float(rope_theta),
                "vocab_size": int(vocab_size),
            }
        )
        self.config = DeltaFormerConfig(**config_kwargs)
        self.encoder = DeltaFormerModel(self.config)
        if autocast_dtype is not None:
            self.encoder = self.encoder.to(dtype=autocast_dtype)

    def _validate_layout(self, embed_dim: int, num_heads: int) -> None:
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("DeltaFormer requires embed_dim divisible by num_heads")
        head_dim = embed_dim // num_heads
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(
                "DeltaFormer requires embed_dim/num_heads to be a power of two so the kernels remain aligned."
            )
        if head_dim < 16:
            raise ValueError("DeltaFormer requires embed_dim/num_heads >= 16")

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("DeltaFormerBackend requires decoder token embeddings")
        seq_len = token_embeddings.size(1)
        if seq_len > self.ctx_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds ctx_len={self.ctx_len}."
                " Increase --backend-option ctx_len to continue."
            )
        padding_mask = self._resolve_padding_mask(sequence)
        attention_mask = (~padding_mask).to(dtype=torch.long)
        target_dtype = next(self.encoder.parameters()).dtype
        embeddings = token_embeddings
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.to(target_dtype)
        outputs = self.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if hidden.dtype != token_embeddings.dtype:
            hidden = hidden.to(token_embeddings.dtype)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


__all__ = ["DeltaFormerBackend"]
