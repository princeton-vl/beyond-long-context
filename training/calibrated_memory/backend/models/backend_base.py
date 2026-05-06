from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def _ensure_2d_mask(mask: torch.Tensor, *, batch: int, length: int, name: str) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(1)
    if mask.dim() != 2:
        raise ValueError(f"{name} must be rank-2 (batch, length)")
    if mask.size(0) != batch or mask.size(1) != length:
        raise ValueError(
            f"{name} shape mismatch: expected ({batch}, {length}) got {tuple(mask.shape)}"
        )
    return mask.to(dtype=torch.bool)


@dataclass(frozen=True)
class SequenceInputs:
    """Unified token batch handed to every backend."""

    input_ids: torch.Tensor
    token_embeddings: torch.Tensor | None
    lengths: torch.Tensor
    padding_mask: torch.Tensor | None
    stream_lengths: torch.Tensor | None = None


class MemoryBackend(nn.Module):
    """Common utilities for direct-mode backends."""

    def __init__(
        self,
        output_dim: int,
        *,
        projects_to_decoder_dim: bool = False,
        requires_token_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(output_dim)
        self.projects_to_decoder_dim = bool(projects_to_decoder_dim)
        self.requires_token_embeddings = bool(requires_token_embeddings)
        self._decoder_dim: int | None = None
        self._projection_layers: nn.ModuleDict = nn.ModuleDict()

    def register_decoder_dim(self, decoder_dim: int) -> None:
        if decoder_dim <= 0:
            raise ValueError("decoder_dim must be positive")
        self._decoder_dim = int(decoder_dim)

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _resolve_padding_mask(self, sequence: SequenceInputs) -> torch.Tensor:
        mask = sequence.padding_mask
        if mask is None:
            if sequence.lengths.dim() != 1:
                raise ValueError("Sequence lengths must be 1D")
            device = sequence.input_ids.device
            arange = torch.arange(sequence.input_ids.size(1), device=device).unsqueeze(0)
            mask = arange >= sequence.lengths.unsqueeze(1)
        return _ensure_2d_mask(
            mask,
            batch=sequence.input_ids.size(0),
            length=sequence.input_ids.size(1),
            name="padding_mask",
        )

    def _mask_embeddings(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        target_dim = self._decoder_dim
        if not target_dim or hidden.size(-1) == target_dim:
            return hidden
        key = f"{hidden.size(-1)}->{target_dim}"
        if key not in self._projection_layers:
            projector = nn.Linear(hidden.size(-1), target_dim)
            projector = projector.to(device=hidden.device, dtype=hidden.dtype)
            self._projection_layers[key] = projector
        projector = self._projection_layers[key]
        weight = projector.weight
        if weight.device != hidden.device or weight.dtype != hidden.dtype:
            projector = projector.to(device=hidden.device, dtype=hidden.dtype)
        return projector(hidden)

    @staticmethod
    def _ensure_stream_lengths(sequence: SequenceInputs) -> torch.Tensor:
        stream_lengths = sequence.stream_lengths
        if stream_lengths is None:
            raise ValueError(
                "Backends that depend on stream/query boundaries require metadata['stream_length']."
            )
        return stream_lengths


__all__ = ["MemoryBackend", "SequenceInputs"]
