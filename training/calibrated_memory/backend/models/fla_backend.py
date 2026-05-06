"""Common utilities for wrapping flash-linear-attention HuggingFace models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Type

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .backend_base import MemoryBackend, SequenceInputs

import torch


def _build_fla_config(
    *,
    embed_dim: int,
    num_layers: int,
    ctx_len: int,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a normalized HuggingFace config dict for flash-linear models."""

    config: Dict[str, Any] = {
        "hidden_size": int(embed_dim),
        "num_hidden_layers": int(max(1, num_layers)),
        "max_position_embeddings": int(max(1, ctx_len)),
    }
    config.update(overrides)
    return config


class FlaSequenceBackend(MemoryBackend):
    """Base class for backends implemented via flash-linear-attention models."""

    config_cls: Type[PretrainedConfig]
    model_cls: Type[PreTrainedModel]

    _MIN_HEAD_DIM = 16

    def __init__(
        self,
        *,
        embed_dim: int,
        config_kwargs: Dict[str, Any],
        ctx_len: int,
        autocast_dtype: torch.dtype | None = None,
        forward_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            output_dim=embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        if self.config_cls is None or self.model_cls is None:
            raise ValueError("FlaSequenceBackend subclasses must define config_cls/model_cls")
        normalized_kwargs = dict(config_kwargs)
        normalized_kwargs.setdefault("hidden_size", embed_dim)
        normalized_kwargs.setdefault("num_hidden_layers", 1)
        normalized_kwargs.setdefault("max_position_embeddings", ctx_len)
        self.config = self.config_cls(**normalized_kwargs)
        self.ctx_len = int(getattr(self.config, "max_position_embeddings", ctx_len))
        self._validate_attention_layout(embed_dim)
        self.encoder = self.model_cls(self.config)
        if autocast_dtype is not None:
            self.encoder = self.encoder.to(dtype=autocast_dtype)
        self._autocast_dtype = autocast_dtype
        self._forward_kwargs = dict(forward_kwargs or {})
        self.supports_direct_logits = True

    def _validate_attention_layout(self, embed_dim: int) -> None:
        num_heads = getattr(self.config, "num_heads", None)
        if num_heads is None:
            return
        try:
            parsed_heads = int(num_heads)
        except (TypeError, ValueError) as exc:  # pragma: no cover - config should guard this
            raise ValueError("FlashLinearAttention backends require integer num_heads") from exc
        if parsed_heads <= 0:
            raise ValueError("FlashLinearAttention backends require num_heads >= 1")
        hidden_size = int(getattr(self.config, "hidden_size", embed_dim))
        if hidden_size % parsed_heads != 0:
            raise ValueError(
                "FlashLinearAttention backends expect embed_dim to be divisible by num_heads so"
                f" the head dimension stays integral (got embed_dim={hidden_size}, num_heads={parsed_heads})."
            )
        head_dim_attr = getattr(self.config, "head_dim", None)
        if head_dim_attr is not None:
            head_dim = int(head_dim_attr)
        else:
            head_dim = hidden_size // parsed_heads
        if head_dim < self._MIN_HEAD_DIM:
            raise ValueError(
                "FlashLinearAttention backends require embed_dim/num_heads >= 16 so the Triton"
                " kernels in flash-linear-attention see K>=16. Increase --backend-option embed_dim"
                " or reduce num_heads to continue."
            )

    def _autocast_context(self, embeddings: torch.Tensor):
        if (
            self._autocast_dtype is not None
            and embeddings.is_cuda
            and torch.cuda.is_available()
        ):
            if hasattr(torch, "autocast"):
                return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
            return torch.cuda.amp.autocast(dtype=self._autocast_dtype)
        if embeddings.is_cuda and torch.cuda.is_available() and torch.is_autocast_enabled():
            if hasattr(torch, "autocast"):
                return torch.autocast(device_type="cuda", enabled=False)
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _resolve_padding(self, lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        if lengths is None:
            raise ValueError("Sequence lengths must be provided for FLA backends")
        arange = torch.arange(seq_len, device=device)
        padding = arange.unsqueeze(0) >= lengths.unsqueeze(1)
        return padding

    def _encode_embeddings(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = embeddings.shape
        if seq_len > self.ctx_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured ctx_len={self.ctx_len}."
                " Increase --backend-option ctx_len to continue."
            )
        original_dtype = embeddings.dtype
        first_param = next(self.encoder.parameters(), None)
        if first_param is not None:
            target_dtype = first_param.dtype
            if embeddings.dtype != target_dtype:
                embeddings = embeddings.to(dtype=target_dtype)
        padding_mask = self._resolve_padding(lengths, seq_len, embeddings.device)
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        with self._autocast_context(embeddings):
            outputs = self.encoder(
                inputs_embeds=embeddings,
                attention_mask=None,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                **self._forward_kwargs,
            )
        hidden = outputs.last_hidden_state
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if hidden.dtype != original_dtype:
            hidden = hidden.to(original_dtype)
        return hidden

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("FLA backends require decoder token embeddings")
        lengths = sequence.lengths.to(device=token_embeddings.device)
        encoded = self._encode_embeddings(token_embeddings, lengths)
        padding_mask = self._resolve_padding_mask(sequence)
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)


__all__ = ["FlaSequenceBackend"]
