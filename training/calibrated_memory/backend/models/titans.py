"""Titans-inspired memory backends (MAC + gated memory variants)."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans_pytorch import NeuralMemory

from .backend_base import MemoryBackend, SequenceInputs
from .modules import AttentionCore, PositionalMode, RMSNorm


_VALID_INCORPORATION = {"mal"}


class TitansEncoder(MemoryBackend):
    """Backend that mirrors the Titans test-time memory designs."""

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        memory_incorporation: Literal["mac", "gated", "mal", "lmm"] = "mac",
        local_window_size: int = 64,
        local_window_heads: int = 4,
        longterm_mem_tokens: int = 2,
        chunk_size: int = 64,
        memory_chunk_size: int | None = None,
    ) -> None:
        hidden_dim = hidden_dim or embed_dim
        if memory_incorporation not in _VALID_INCORPORATION:
            raise ValueError("TitansEncoder now only supports memory_incorporation='mal'.")
        super().__init__(
            hidden_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if local_window_size <= 0:
            raise ValueError("local_window_size must be positive")
        if local_window_heads <= 0:
            raise ValueError("local_window_heads must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.memory_incorporation = memory_incorporation
        self.local_window_size = local_window_size
        self.local_window_heads = local_window_heads
        self.longterm_mem_tokens = longterm_mem_tokens
        self.chunk_size = chunk_size
        resolved_mem_chunk = memory_chunk_size if memory_chunk_size is not None else chunk_size
        self.memory_chunk_size = max(2, int(resolved_mem_chunk))

        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()

        prefix = longterm_mem_tokens
        self.prefix_tokens = (
            nn.Parameter(torch.randn(prefix, hidden_dim) * 0.02) if prefix > 0 else None
        )

        if memory_incorporation != "mal":
            raise ValueError(
                "TitansEncoder now only supports memory_incorporation='mal'. "
                "Use the 'titans_external' backend for MAC / gated / LMM variants."
            )

        self.encoder = TitansMemoryAsLayer(
            hidden_dim=hidden_dim,
            heads=local_window_heads,
            window=local_window_size,
            dropout=dropout,
            prefix_tokens=self.prefix_tokens,
            memory_chunk_size=self.memory_chunk_size,
            chunk_size=self.chunk_size,
        )

        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del label_mask  # TitansEncoder processes all tokens regardless of supervision mask.
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("TitansEncoder requires token embeddings for every token in the sequence.")
        padding_mask = self._resolve_padding_mask(sequence)
        masked_inputs = self._mask_embeddings(token_embeddings, padding_mask)
        hidden = self._encode_sequence(masked_inputs, padding_mask)
        hidden = self._project_hidden(hidden)
        return self._mask_embeddings(hidden, padding_mask)

    def _encode_sequence(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self.input_proj(tokens)
        return self.encoder(hidden, padding_mask=padding_mask)


class TitansMACStack(nn.Module):
    """Memory-as-context stack that mirrors the Titans MAC block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        prefix_tokens: nn.Parameter | None,
        chunk_size: int,
        memory_chunk_size: int,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive for Titans MAC mode")
        self.prefix_tokens = prefix_tokens
        self.chunk_size = chunk_size
        self.memory_chunk_size = max(2, int(memory_chunk_size))
        # MAC uses full causal attention inside each chunk so sliding_window is disabled.
        self.blocks = nn.ModuleList(
            [
                TitansBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    window=-1,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.chunk_norm = RMSNorm(hidden_dim)
        self.mem_norm = RMSNorm(hidden_dim)
        self.out_norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mem_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neural_memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=self.memory_chunk_size,
            num_kv_per_token=1,
            pre_rmsnorm=True,
            per_head_learned_parameters=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if tokens.size(1) == 0:
            return tokens
        outputs: list[torch.Tensor] = []
        mem_state: Any | None = None
        prev_context: torch.Tensor | None = None
        start = 0
        seq_len = tokens.size(1)
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk = tokens[:, start:end, :]
            chunk_mask = None if padding_mask is None else padding_mask[:, start:end]
            mem_context = self._resolve_memory_context(chunk, prev_context)
            fused_chunk, mem_state, prev_context = self._attend_and_update(
                chunk,
                mem_context,
                chunk_mask,
                mem_state,
            )
            outputs.append(fused_chunk)
            start = end
        return torch.cat(outputs, dim=1)

    def _attend_and_update(
        self,
        chunk: torch.Tensor,
        mem_context: torch.Tensor,
        chunk_mask: torch.Tensor | None,
        state: Any | None,
    ) -> tuple[torch.Tensor, Any | None, torch.Tensor]:
        combined, attn_mask = self._build_attention_inputs(chunk, mem_context, chunk_mask)
        hidden = combined
        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=attn_mask)
        hidden = self.final_norm(hidden)
        prefix_len = 0 if self.prefix_tokens is None else self.prefix_tokens.size(0)
        context_len = mem_context.size(1)
        chunk_out = hidden[:, prefix_len + context_len :, :]
        mem_values, next_state = self._run_neural_memory(chunk_out, state=state)
        mem_values = mem_values[:, : chunk_out.size(1), :]
        chunk_norm = self.chunk_norm(chunk_out)
        mem_norm = self.mem_norm(mem_values)
        gate = torch.sigmoid(self.gate_proj(torch.cat([chunk_norm, mem_norm], dim=-1)))
        fused = gate * chunk_norm + (1.0 - gate) * mem_norm
        return self.out_norm(self.dropout(fused)), next_state, mem_values

    @staticmethod
    def _resolve_memory_context(
        chunk: torch.Tensor,
        prev_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Use the previous chunk's retrievals (or zeros) as the current context."""

        batch, chunk_len, dim = chunk.shape
        if prev_context is None:
            return chunk.new_zeros(batch, chunk_len, dim)
        if prev_context.size(1) == chunk_len:
            return prev_context
        if prev_context.size(1) > chunk_len:
            return prev_context[:, :chunk_len, :]
        pad_len = chunk_len - prev_context.size(1)
        pad = prev_context.new_zeros(batch, pad_len, dim)
        return torch.cat([prev_context, pad], dim=1)

    def _build_attention_inputs(
        self,
        chunk: torch.Tensor,
        mem_context: torch.Tensor,
        chunk_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = chunk.size(0)
        segments = []
        masks = []
        if self.prefix_tokens is not None:
            prefix = self.prefix_tokens.unsqueeze(0).expand(batch, -1, -1)
            segments.append(prefix)
            if chunk_mask is not None:
                masks.append(chunk_mask.new_zeros(batch, prefix.size(1), dtype=torch.bool))
        segments.append(mem_context)
        segments.append(chunk)
        if chunk_mask is not None:
            masks.append(chunk_mask)
            masks.append(chunk_mask)
        combined = torch.cat(segments, dim=1)
        attn_mask = None
        if chunk_mask is not None:
            attn_mask = torch.cat(masks, dim=1)
        return combined, attn_mask

    def _run_neural_memory(self, inputs: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """Execute NeuralMemory in float32 and cast back to the original dtype."""

        aligned, orig_len = _align_memory_inputs(inputs, self.memory_chunk_size)
        orig_dtype = aligned.dtype
        inputs32 = aligned.to(torch.float32)
        converted_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                converted_kwargs[key] = value.to(torch.float32)
            else:
                converted_kwargs[key] = value
        with torch.cuda.amp.autocast(enabled=False):
            outputs, state = self.neural_memory(inputs32, **converted_kwargs)
        return outputs[:, :orig_len, :].to(orig_dtype), state


class TitansGatedMemory(nn.Module):
    """Implements the gated short + long-term memory fusion."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        window: int,
        dropout: float,
        prefix_tokens: nn.Parameter | None,
        longterm_mem_tokens: int,
        memory_chunk_size: int,
    ) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.sliding_attn = AttentionCore(
            hidden_dim,
            heads,
            attn_dropout=dropout,
            positional_mode=PositionalMode.NONE,
            sliding_window=window,
        )
        self.short_norm = RMSNorm(hidden_dim)
        self.mem_norm = RMSNorm(hidden_dim)
        self.out_norm = RMSNorm(hidden_dim)
        gate_dim = hidden_dim * 2
        self.gate_proj = nn.Linear(gate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.memory_chunk_size = max(2, int(memory_chunk_size))
        self.neural_memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=self.memory_chunk_size,
            num_kv_per_token=1,
            pre_rmsnorm=True,
            per_head_learned_parameters=False,
        )
        self.longterm_mem_tokens = longterm_mem_tokens

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden, mask = _prepend_prefix(tokens, padding_mask, self.prefix_tokens)
        short_term = self.sliding_attn(
            self.short_norm(hidden),
            attn_mask=None,
            key_padding_mask=mask,
            is_causal=True,
        )
        mem_out, _state = self._run_neural_memory(hidden)
        mem_out = mem_out[:, : hidden.size(1), :]
        short_norm = self.short_norm(short_term)
        mem_norm = self.mem_norm(mem_out)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([short_norm, mem_norm], dim=-1))
        )
        fused = gate * short_norm + (1.0 - gate) * mem_norm
        fused = self.out_norm(self.dropout(fused))
        prefix_len = 0 if self.prefix_tokens is None else self.prefix_tokens.size(0)
        return fused[:, prefix_len:, :]

    def _run_neural_memory(self, inputs: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        aligned, orig_len = _align_memory_inputs(inputs, self.memory_chunk_size)
        dtype = aligned.dtype
        inputs32 = aligned.to(torch.float32)
        converted_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                converted_kwargs[key] = value.to(torch.float32)
            else:
                converted_kwargs[key] = value
        with torch.cuda.amp.autocast(enabled=False):
            outputs, state = self.neural_memory(inputs32, **converted_kwargs)
        return outputs[:, :orig_len, :].to(dtype), state


class TitansBlock(nn.Module):
    """Sliding-window transformer block used in MAC mode."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dropout: float,
        window: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = AttentionCore(
            dim,
            heads,
            attn_dropout=dropout,
            positional_mode=PositionalMode.NONE,
            sliding_window=window,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm1(hidden),
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        hidden = hidden + self.dropout1(attn_out)
        mlp_out = self.mlp(self.norm2(hidden))
        hidden = hidden + self.dropout2(mlp_out)
        return hidden


class TitansMemoryAsLayer(nn.Module):
    """Memory-as-a-layer: NeuralMemory precedes sliding-window attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        window: int,
        dropout: float,
        prefix_tokens: nn.Parameter | None,
        memory_chunk_size: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.memory_chunk_size = max(2, int(memory_chunk_size))
        self.chunk_size = max(1, int(chunk_size))
        self.window = max(0, int(window))
        self.neural_memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=self.memory_chunk_size,
            num_kv_per_token=1,
            pre_rmsnorm=True,
            per_head_learned_parameters=False,
        )
        self.mem_norm = RMSNorm(hidden_dim)
        self.chunk_norm = RMSNorm(hidden_dim)
        self.attn_norm = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = AttentionCore(
            hidden_dim,
            heads,
            attn_dropout=dropout,
            positional_mode=PositionalMode.NONE,
            sliding_window=window,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        state: Any | None = None
        seq_len = tokens.size(1)
        start = 0
        context_tokens: torch.Tensor | None = None
        context_mask: torch.Tensor | None = None
        prefix_len = 0 if self.prefix_tokens is None else self.prefix_tokens.size(0)

        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk = tokens[:, start:end, :]
            chunk_mask = None if padding_mask is None else padding_mask[:, start:end]
            chunk_with_prefix, _chunk_mask_with_prefix = _prepend_prefix(
                chunk, chunk_mask, self.prefix_tokens
            )
            # NeuralMemory enforces a fixed chunk size; including the prefix tokens
            # expands each chunk by `prefix_len` and leads to dynamic shapes that the
            # underlying lucidrains implementation cannot handle (it materializes
            # weight tensors with varying leading dimensions). Feed only the actual
            # chunk into the neural memory so every update uses the configured
            # chunk_size, then re-apply the prefix downstream for attention.
            mem_inputs = chunk if self.prefix_tokens is not None else chunk_with_prefix
            mem_out, state = self._run_neural_memory(mem_inputs, state=state)
            mem_out = mem_out[:, : chunk.size(1), :]
            chunk_norm = self.chunk_norm(chunk)
            mem_norm = self.mem_norm(mem_out)
            gate = torch.sigmoid(self.gate_proj(torch.cat([chunk_norm, mem_norm], dim=-1)))
            fused_chunk = gate * chunk_norm + (1.0 - gate) * mem_norm

            attn_inputs: list[torch.Tensor] = []
            attn_masks: list[torch.Tensor] = []
            context_len = 0
            if context_tokens is not None:
                attn_inputs.append(context_tokens)
                context_len = context_tokens.size(1)
                if context_mask is not None:
                    attn_masks.append(context_mask)
            attn_inputs.append(fused_chunk)
            if chunk_mask is not None:
                attn_masks.append(chunk_mask)
            attn_input = torch.cat(attn_inputs, dim=1) if len(attn_inputs) > 1 else fused_chunk
            attn_mask_tensor = None
            if attn_masks:
                attn_mask_tensor = (
                    torch.cat(attn_masks, dim=1)
                    if len(attn_masks) > 1
                    else attn_masks[0]
                )
            attn_out = self.attn(
                self.attn_norm(attn_input),
                attn_mask=None,
                key_padding_mask=attn_mask_tensor,
                is_causal=True,
            )
            attn_chunk = attn_out[:, context_len:, :]
            chunk_result = fused_chunk + self.dropout(attn_chunk)
            chunk_result = self.out_norm(chunk_result)
            if chunk_mask is not None:
                chunk_result = chunk_result.masked_fill(chunk_mask.unsqueeze(-1), 0.0)
            outputs.append(chunk_result)
            context_tokens, context_mask = self._update_context(chunk_result, chunk_mask)
            start = end

        return torch.cat(outputs, dim=1)

    def _update_context(
        self,
        chunk_result: torch.Tensor,
        chunk_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.window <= 0:
            return None, None
        keep = min(self.window, chunk_result.size(1))
        context = chunk_result[:, -keep:, :]
        mask = None
        if chunk_mask is not None:
            mask = chunk_mask[:, -keep:]
        return context, mask

    def _run_neural_memory(
        self,
        inputs: torch.Tensor,
        *,
        state: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        aligned, orig_len = _align_memory_inputs(inputs, self.memory_chunk_size)
        dtype = aligned.dtype
        inputs32 = aligned.to(torch.float32)
        converted_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                converted_kwargs[key] = value.to(torch.float32)
            else:
                converted_kwargs[key] = value
        with torch.cuda.amp.autocast(enabled=False):
            outputs, next_state = self.neural_memory(
                inputs32,
                state=state,
                **converted_kwargs,
            )
        return outputs[:, :orig_len, :].to(dtype), next_state


class TitansMemoryOnly(nn.Module):
    """Memory-only variant (Titans LMM) without attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        prefix_tokens: nn.Parameter | None,
        memory_chunk_size: int,
    ) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.memory_chunk_size = max(2, int(memory_chunk_size))
        self.neural_memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=self.memory_chunk_size,
            num_kv_per_token=1,
            pre_rmsnorm=True,
            per_head_learned_parameters=False,
        )
        self.out_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden, _ = _prepend_prefix(tokens, padding_mask, self.prefix_tokens)
        mem_out, _ = self._run_neural_memory(hidden)
        mem_out = mem_out[:, : hidden.size(1), :]
        prefix_len = 0 if self.prefix_tokens is None else self.prefix_tokens.size(0)
        return self.out_norm(mem_out)[:, prefix_len:, :]

    def _run_neural_memory(self, inputs: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        aligned, orig_len = _align_memory_inputs(inputs, self.memory_chunk_size)
        dtype = aligned.dtype
        inputs32 = aligned.to(torch.float32)
        converted_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                converted_kwargs[key] = value.to(torch.float32)
            else:
                converted_kwargs[key] = value
        with torch.cuda.amp.autocast(enabled=False):
            outputs, state = self.neural_memory(inputs32, **converted_kwargs)
        return outputs[:, :orig_len, :].to(dtype), state


def _prepend_prefix(
    tokens: torch.Tensor,
    padding_mask: torch.Tensor | None,
    prefix_tokens: nn.Parameter | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append learned prefix tokens and extend the padding mask."""

    if prefix_tokens is None or prefix_tokens.size(0) == 0:
        return tokens, padding_mask
    batch = tokens.size(0)
    prefix = prefix_tokens.unsqueeze(0).expand(batch, -1, -1)
    augmented = torch.cat([prefix, tokens], dim=1)
    if padding_mask is None:
        mask = torch.zeros(batch, augmented.size(1), dtype=torch.bool, device=tokens.device)
        return augmented, mask
    prefix_mask = padding_mask.new_zeros(batch, prefix.size(1))
    mask = torch.cat([prefix_mask, padding_mask], dim=1)
    return augmented, mask


def _align_memory_inputs(inputs: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, int]:
    """Pad or trim neural-memory inputs so every update sees a fixed chunk length."""

    original_len = inputs.size(1)
    if original_len == chunk_size:
        return inputs, original_len
    if original_len < chunk_size:
        pad = chunk_size - original_len
        padded = F.pad(inputs, (0, 0, 0, pad))
        return padded, original_len
    return inputs[:, :chunk_size, :], original_len


__all__ = ["TitansEncoder"]
