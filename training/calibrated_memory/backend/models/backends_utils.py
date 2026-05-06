from __future__ import annotations

import torch

from calibrated_memory.data.sequences.common import IGNORE_INDEX


def build_concatenated_tokens(
    stream_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.cat([stream_tokens, query_tokens], dim=1)
    mask = torch.cat([padding_mask, query_mask], dim=1)
    return hidden, mask


def build_label_indices(query_labels: torch.Tensor) -> torch.Tensor:
    return (query_labels != IGNORE_INDEX).nonzero(as_tuple=False)[:, 1]
