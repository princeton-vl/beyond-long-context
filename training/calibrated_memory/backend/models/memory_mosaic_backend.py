from __future__ import annotations

import torch
import torch.nn as nn

from .backend_base import MemoryBackend, SequenceInputs
from .external.memory_mosaics_blocks import Block, MemoryMosaicConfig


class _MemoryMosaicEncoder(nn.Module):
    """Thin wrapper that reuses Memory Mosaic blocks on precomputed embeddings."""

    def __init__(self, config: MemoryMosaicConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


class MemoryMosaicBackend(MemoryBackend):
    """Memory backend that encodes streams using Memory Mosaic blocks."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_layer: int = 6,
        n_head: int = 8,
        pmem_size: int = 2048,
        pmem_count: int = 2,
        dropout: float = 0.1,
        block_size: int = 512,
        leaky_cuda: bool = False,
    ) -> None:
        if embed_dim % n_head != 0:
            raise ValueError("embed_dim must be divisible by n_head for Memory Mosaic blocks")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        super().__init__(
            embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        self.block_size = int(block_size)
        config = MemoryMosaicConfig(
            block_size=self.block_size,
            vocab_size=max(1, embed_dim),
            n_layer=max(1, int(n_layer)),
            n_head=max(1, int(n_head)),
            n_embd=int(embed_dim),
            dropout=float(dropout),
            pmem_size=max(1, int(pmem_size)),
            pmem_count=max(1, int(pmem_count)),
            bias=True,
            leaky_cuda=bool(leaky_cuda),
        )
        self.encoder = _MemoryMosaicEncoder(config)
        self.supports_direct_logits = True

    def _resolve_padding_mask(self, sequence: SequenceInputs) -> torch.Tensor:
        mask = sequence.padding_mask
        if mask is not None:
            return mask
        seq_len = sequence.input_ids.size(1)
        device = sequence.input_ids.device
        indices = torch.arange(seq_len, device=device).unsqueeze(0)
        return indices >= sequence.lengths.unsqueeze(1)

    def _validate_length(self, seq_len: int) -> None:
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured block_size={self.block_size}."
                " Increase --backend-option block_size to cover the dataset prefix."
            )

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("MemoryMosaicBackend requires token embeddings from the decoder")
        batch, seq_len, _ = token_embeddings.shape
        self._validate_length(seq_len)
        padding_mask = self._resolve_padding_mask(sequence)
        masked_inputs = token_embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(masked_inputs)
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        projected = self._project_hidden(encoded)
        return projected.masked_fill(padding_mask.unsqueeze(-1), 0.0)
