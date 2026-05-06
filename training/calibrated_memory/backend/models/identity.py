from __future__ import annotations

import torch

from .backend_base import MemoryBackend, SequenceInputs


class IdentityBackend(MemoryBackend):
    """Backend that simply returns the decoder-provided token embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__(
            output_dim=embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("IdentityBackend requires token embeddings.")
        padding_mask = self._resolve_padding_mask(sequence)
        return self._mask_embeddings(token_embeddings, padding_mask)
