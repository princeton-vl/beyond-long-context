from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .common import IGNORE_INDEX, QUERY_END_SEPARATOR, STREAM_QUERY_SEPARATOR


def _extract_query_intervals(seq: torch.Tensor, stream_len: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start_idx: int | None = None
    for pos in range(stream_len, seq.size(0)):
        token = int(seq[pos].item())
        if token == STREAM_QUERY_SEPARATOR and start_idx is None:
            start_idx = pos
            continue
        if token == QUERY_END_SEPARATOR and start_idx is not None:
            intervals.append((start_idx, pos + 1))
            start_idx = None
    return intervals


def _shuffle_query_embeddings(
    extras: dict | None,
    lengths: list[int],
    permutation: list[int],
) -> dict | None:
    if not extras:
        return extras
    updated = dict(extras)
    query_embeddings = extras.get("query_embeddings")
    query_embedding_mask = extras.get("query_embedding_mask")
    if query_embeddings is not None and lengths:
        emb_chunks = list(torch.split(query_embeddings, lengths, dim=0))
        reordered = [emb_chunks[idx] for idx in permutation]
        updated["query_embeddings"] = torch.cat(reordered, dim=0)
    if query_embedding_mask is not None and lengths:
        mask_chunks = list(torch.split(query_embedding_mask, lengths, dim=0))
        reordered = [mask_chunks[idx] for idx in permutation]
        updated["query_embedding_mask"] = torch.cat(reordered, dim=0)
    metadata = extras.get("metadata")
    if isinstance(metadata, dict):
        queries_meta = metadata.get("queries")
        if isinstance(queries_meta, list) and queries_meta:
            reordered_meta = [queries_meta[idx] for idx in permutation if idx < len(queries_meta)]
            new_meta = dict(metadata)
            new_meta["queries"] = reordered_meta
            updated["metadata"] = new_meta
    return updated


def _maybe_shuffle_queries(
    seq: torch.Tensor,
    labels: torch.Tensor,
    stream_len_tensor: torch.Tensor,
    extras: dict | None,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    stream_len = int(stream_len_tensor.item())
    total_len = seq.size(0)
    if total_len <= stream_len:
        return seq, labels, extras
    intervals = _extract_query_intervals(seq, stream_len)
    if len(intervals) <= 1:
        return seq, labels, extras
    permutation = torch.randperm(len(intervals)).tolist()
    seq_chunks = [seq[:stream_len]]
    label_chunks = [labels[:stream_len]]
    lengths: list[int] = []
    for idx in permutation:
        start, end = intervals[idx]
        seq_chunks.append(seq[start:end])
        label_chunks.append(labels[start:end])
        lengths.append(end - start)
    new_seq = torch.cat(seq_chunks, dim=0)
    new_labels = torch.cat(label_chunks, dim=0)
    updated_extras = _shuffle_query_embeddings(extras, lengths, permutation)
    return new_seq, new_labels, updated_extras


def _pad_1d(
    tensors: Sequence[torch.Tensor], pad_value: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of 1D tensors and produce (padded, padding_mask, lengths)."""

    batch_size = len(tensors)
    max_len = max((t.numel() for t in tensors), default=0)

    if max_len == 0:
        shape = (batch_size, 0)
        return (
            torch.empty(shape, dtype=dtype, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    output = torch.full((batch_size, max_len), pad_value, dtype=dtype, device=device)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, tensor in enumerate(tensors):
        length = tensor.numel()
        if length == 0:
            continue
        output[i, :length] = tensor
        padding_mask[i, :length] = False
        lengths[i] = length

    return output, padding_mask, lengths


def build_collate(pad_id: int):
    """Collate function that keeps stream/query pairs concatenated as a sequence."""

    def collate(batch: Sequence[Tuple]):
        if not batch:
            raise ValueError("Collator received an empty batch.")
        device = batch[0][0].device

        sequence_tokens: list[torch.Tensor] = []
        sequence_labels: list[torch.Tensor] = []
        sequence_embeddings: list[torch.Tensor | None] = []
        sequence_embedding_masks: list[torch.Tensor | None] = []
        metadata_entries: list[dict] = []

        for sample in batch:
            if len(sample) == 4:
                seq, labels, stream_len_tensor, extras = sample
            else:
                seq, labels, stream_len_tensor = sample
                extras = None
            stream_len = int(stream_len_tensor.item())
            if stream_len < 0 or stream_len > seq.numel():
                raise ValueError(
                    f"Invalid stream length {stream_len} for sample with {seq.numel()} tokens."
                )
            if seq.numel() != labels.numel():
                raise ValueError("Sequence tokens and labels must share the same length.")

            seq, labels, extras = _maybe_shuffle_queries(seq, labels, stream_len_tensor, extras)

            metadata_source = extras.get("metadata") if extras else None
            if metadata_source is None:
                metadata_entry = {}
            elif isinstance(metadata_source, dict):
                metadata_entry = dict(metadata_source)
            else:
                raise ValueError("Sample metadata payload must be a mapping when provided.")
            if "stream_length" in metadata_entry and metadata_entry["stream_length"] != stream_len:
                raise ValueError(
                    "Metadata-provided stream_length does not match the dataset sample."
                )
            metadata_entry["stream_length"] = stream_len
            metadata_entries.append(metadata_entry)

            sequence_tokens.append(seq)
            sequence_labels.append(labels)

            stream_embeddings = extras.get("stream_embeddings") if extras else None
            query_embeddings = extras.get("query_embeddings") if extras else None
            query_embedding_mask = extras.get("query_embedding_mask") if extras else None
            embeddings, embedding_mask = _combine_embeddings(
                stream_embeddings,
                query_embeddings,
                query_embedding_mask,
                stream_len,
                seq.numel(),
            )
            sequence_embeddings.append(embeddings)
            sequence_embedding_masks.append(embedding_mask)

        sequence_ids, padding_mask, lengths = _pad_1d(
            sequence_tokens, pad_value=pad_id, dtype=torch.long, device=device
        )
        labels_padded, _, _ = _pad_1d(
            sequence_labels, pad_value=IGNORE_INDEX, dtype=torch.long, device=device
        )

        sequence_embeddings_padded = _pad_embeddings(
            sequence_embeddings, sequence_ids.shape, device=device
        )
        embedding_masks_padded = _pad_embedding_mask(
            sequence_embedding_masks, sequence_ids.shape, device
        )

        batch_dict = {
            "sequence": {
                "input_ids": sequence_ids,
                "padding_mask": padding_mask,
                "lengths": lengths,
                "embeddings": sequence_embeddings_padded,
                "embedding_mask": embedding_masks_padded,
            },
            "labels": labels_padded,
            "metadata": metadata_entries,
        }
        return batch_dict

    return collate


def _pad_embeddings(raw_list, target_shape, device):
    if not any(tensor is not None for tensor in raw_list):
        return None
    batch_size, seq_len = target_shape
    embed_dim = next(t.size(-1) for t in raw_list if t is not None)
    output = torch.zeros((batch_size, seq_len, embed_dim), dtype=torch.float32, device=device)
    for idx, tensor in enumerate(raw_list):
        if tensor is None:
            continue
        length = min(tensor.size(0), seq_len)
        output[idx, :length] = tensor[:length].to(dtype=torch.float32, device=device)
    return output


def _pad_embedding_mask(raw_masks, target_shape, device):
    if not any(mask is not None for mask in raw_masks):
        return None
    batch_size, seq_len = target_shape
    output = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for idx, mask in enumerate(raw_masks):
        if mask is None:
            continue
        mask_tensor = mask.to(device=device, dtype=torch.bool)
        length = min(mask_tensor.size(0), seq_len)
        output[idx, :length] = mask_tensor[:length]
    return output


def _combine_embeddings(
    stream_embeddings: torch.Tensor | None,
    query_embeddings: torch.Tensor | None,
    query_embedding_mask: torch.Tensor | None,
    stream_len: int,
    total_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    query_len = total_len - stream_len
    if query_len < 0:
        raise ValueError("Total length cannot be smaller than stream length.")
    if stream_embeddings is None and query_embeddings is None:
        if query_embedding_mask is not None:
            raise ValueError(
                "query_embedding_mask provided without corresponding query embeddings."
            )
        return None, None

    embed_dim = None
    base_device = None
    if stream_embeddings is not None:
        if stream_embeddings.dim() != 2 or stream_embeddings.size(0) != stream_len:
            raise ValueError("stream_embeddings shape does not match stream length.")
        embed_dim = stream_embeddings.size(-1)
        base_device = stream_embeddings.device
    if query_embeddings is not None:
        if query_embeddings.dim() != 2 or query_embeddings.size(0) != query_len:
            raise ValueError("query_embeddings shape does not match query length.")
        embed_dim = embed_dim or query_embeddings.size(-1)
        if query_embeddings.size(-1) != embed_dim:
            raise ValueError("Query embeddings must share the same feature dimension as stream embeddings.")
        base_device = base_device or query_embeddings.device
    if embed_dim is None:
        raise ValueError("Failed to resolve embedding dimension from provided tensors.")

    device = base_device or torch.device("cpu")
    stream_tensor: torch.Tensor
    stream_mask: torch.Tensor
    if stream_embeddings is None:
        stream_tensor = torch.zeros((stream_len, embed_dim), dtype=torch.float32, device=device)
        stream_mask = torch.zeros((stream_len,), dtype=torch.bool, device=device)
    else:
        stream_tensor = stream_embeddings.to(device=device, dtype=torch.float32)
        stream_mask = torch.ones((stream_len,), dtype=torch.bool, device=device)

    query_tensor: torch.Tensor
    if query_embeddings is None:
        query_tensor = torch.zeros((query_len, embed_dim), dtype=torch.float32, device=device)
    else:
        query_tensor = query_embeddings.to(device=device, dtype=torch.float32)
    if query_embedding_mask is None:
        query_mask = torch.zeros((query_len,), dtype=torch.bool, device=device)
        if query_embeddings is not None:
            query_mask[:] = True
    else:
        if query_embedding_mask.dim() != 1 or query_embedding_mask.size(0) != query_len:
            raise ValueError("query_embedding_mask must match the query length.")
        query_mask = query_embedding_mask.to(device=device, dtype=torch.bool)

    combined = torch.cat([stream_tensor, query_tensor], dim=0)
    combined_mask = torch.cat([stream_mask, query_mask], dim=0)
    return combined, combined_mask
