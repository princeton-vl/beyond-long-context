"""Wrappers around the upstream Titans (lucidrains) reference implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from external.titans_pytorch.titans_pytorch.mac_transformer import MemoryAsContextTransformer
from external.titans_pytorch.titans_pytorch.memory_models import MemoryMLP

from .backend_base import MemoryBackend, SequenceInputs


@dataclass
class _HookCache:
    capture_fn: Callable[[torch.Tensor], None]
    handle: Any | None = None

    def install(self, module: nn.Module) -> None:
        self.handle = module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        if inputs:
            self.capture_fn(inputs[0])


class TitansExternalMAC(MemoryBackend):
    """Direct-mode backend that delegates to lucidrains' Memory-as-Context transformer."""

    def __init__(
        self,
        *,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int,
        num_slots: int = 4,
        vocab_size: int,
        pad_id: int,
        longterm_mem_tokens: int,
        chunk_size: int,
        local_window_heads: int,
        neural_memory_chunk_size: int,
        neural_memory_model_depth: int,
        neural_memory_model_expansion: float,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
        dim_head: int | None = None,
        persist_mem_tokens: int | None = None,
        use_flex_attention: bool = False,
        sliding_window_attn: bool = False,
    ) -> None:
        hidden_dim = hidden_dim or embed_dim
        super().__init__(
            hidden_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=False,
        )
        if vocab_size <= pad_id:
            raise ValueError("vocab_size must exceed pad_id so padding stays within range")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if neural_memory_chunk_size <= 0:
            raise ValueError("neural_memory_chunk_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.hidden_dim = hidden_dim
        self.num_slots = int(num_slots)
        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)
        self.chunk_size = int(chunk_size)
        self.longterm_mem_tokens = int(longterm_mem_tokens)
        self.neural_memory_chunk_size = int(neural_memory_chunk_size)
        self.dropout = float(dropout)
        self._dim_head = dim_head or max(16, hidden_dim // max(1, local_window_heads))
        self._ff_mult = max(1.0, float(ff_mult))
        self._persist_tokens = (
            int(persist_mem_tokens)
            if persist_mem_tokens is not None
            else int(longterm_mem_tokens)
        )

        mem_model = MemoryMLP(
            dim=hidden_dim,
            depth=int(max(1, neural_memory_model_depth)),
            expansion_factor=float(max(1.0, neural_memory_model_expansion)),
        )
        self.transformer = MemoryAsContextTransformer(
            num_tokens=self.vocab_size,
            dim=hidden_dim,
            depth=num_layers,
            segment_len=self.chunk_size,
            neural_memory_segment_len=self.neural_memory_chunk_size,
            num_longterm_mem_tokens=self.longterm_mem_tokens,
            num_persist_mem_tokens=self._persist_tokens,
            heads=local_window_heads,
            dim_head=self._dim_head,
            ff_mult=self._ff_mult,
            neural_memory_model=mem_model,
            use_flex_attn=use_flex_attention,
            sliding_window_attn=sliding_window_attn,
        )
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del label_mask
        padding_mask = self._resolve_padding_mask(sequence)
        trimmed_ids = self._trim_and_pad_tokens(
            sequence.input_ids,
            padding_mask,
            sequence.lengths,
        )
        self._ensure_token_capacity(trimmed_ids, sequence.lengths)
        hidden = self._encode_ids(trimmed_ids)
        hidden = self._project_hidden(hidden)
        batch, seq_len = sequence.input_ids.shape
        full_hidden = hidden.new_zeros(batch, seq_len, hidden.size(-1))
        for idx, length in enumerate(sequence.lengths.tolist()):
            length = int(length)
            if length <= 0:
                continue
            full_hidden[idx, :length, :] = hidden[idx, :length, :]
        return self._mask_embeddings(full_hidden, padding_mask)

    def _trim_and_pad_tokens(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(dtype=torch.long)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        trimmed = token_ids.new_full((token_ids.size(0), max_len), self.pad_id)
        for idx, length in enumerate(lengths.tolist()):
            length = int(length)
            if length <= 0:
                continue
            trimmed[idx, :length] = token_ids[idx, :length]
        return trimmed

    def _ensure_token_capacity(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> None:
        valid = token_ids
        if lengths is not None:
            max_len = token_ids.size(1)
            mask = torch.arange(max_len, device=token_ids.device).unsqueeze(0) >= lengths.unsqueeze(1)
            if mask.any():
                valid = token_ids.masked_fill(mask, self.pad_id)
        max_id = int(valid.max().item())
        required = max(max_id + 1, self.pad_id + 1)
        if required <= self.vocab_size:
            return
        self._expand_vocab(required)

    def _expand_vocab(self, required: int) -> None:
        new_size = max(required, self.vocab_size * 2)
        old_emb = self.transformer.token_emb
        new_emb = nn.Embedding(new_size, old_emb.embedding_dim, device=old_emb.weight.device, dtype=old_emb.weight.dtype)
        with torch.no_grad():
            new_emb.weight.zero_()
            new_emb.weight[: old_emb.num_embeddings] = old_emb.weight
        self.transformer.token_emb = new_emb

        old_out = self.transformer.to_logits
        new_out = nn.Linear(old_out.in_features, new_size, bias=False, device=old_out.weight.device, dtype=old_out.weight.dtype)
        with torch.no_grad():
            new_out.weight.zero_()
            new_out.weight[: old_out.out_features] = old_out.weight
        self.transformer.to_logits = new_out
        self.vocab_size = new_size

    def _encode_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def _capture_hidden(states: torch.Tensor) -> None:
            captured["hidden"] = states

        hook = _HookCache(capture_fn=_capture_hidden)
        hook.install(self.transformer.to_logits)
        try:
            _ = self.transformer(token_ids)
        finally:
            hook.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture Titans hidden states during forward pass")
        return hidden

__all__ = ["TitansExternalMAC"]
