from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .backend_base import MemoryBackend, SequenceInputs
from .external.compressive_transformer import CompressiveTransformer


class CompressiveTransformerBackend(MemoryBackend):
    """Backend that summarizes streams with a Compressive Transformer."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        heads: int = 8,
        block_length: int = 64,
        mem_length: int = 128,
        compression_factors: int | List[int] = 4,
        compression_lengths: int | List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            hidden_dim,
            projects_to_decoder_dim=False,
            requires_token_embeddings=True,
        )
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by the number of heads")
        self.block_length = block_length
        self.mem_transformer = CompressiveTransformer(
            num_tokens=1,
            emb_dim=embed_dim,
            dim=hidden_dim,
            depth=num_layers,
            heads=heads,
            seq_len=block_length,
            mem_len=mem_length,
            cmem_ratios=compression_factors,
            cmem_lengths=compression_lengths,
            attn_layer_dropout=dropout,
            ff_dropout=dropout,
            attn_dropout=dropout,
            reconstruction_attn_dropout=dropout,
        )
        # We pass pre-computed embeddings so disable the internal embedding + LM head.
        self.mem_transformer.token_emb = nn.Identity()
        self.mem_transformer.to_logits = nn.Identity()
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("CompressiveTransformerBackend requires token embeddings.")

        seq_len = token_embeddings.size(1)
        padding_mask = sequence.padding_mask
        if padding_mask is None:
            lengths = sequence.lengths
            if lengths is None:
                lengths = torch.full(
                    (token_embeddings.size(0),),
                    seq_len,
                    dtype=torch.long,
                    device=token_embeddings.device,
                )
            padding_mask = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0) >= lengths.unsqueeze(1)
        valid_mask = ~padding_mask

        memories = (None, None, None)
        hidden = token_embeddings.new_zeros(token_embeddings.size(0), seq_len, self.output_dim)
        for start in range(0, seq_len, self.block_length):
            end = min(start + self.block_length, seq_len)
            block = token_embeddings[:, start:end, :]
            block_mask = valid_mask[:, start:end]
            if block.size(1) == 0:
                continue
            out, memories, _ = self.mem_transformer(block, memories=memories, mask=block_mask)
            hidden[:, start:end, :] = out
        projected = self._project_hidden(hidden)
        return projected.masked_fill(padding_mask.unsqueeze(-1), 0.0)
