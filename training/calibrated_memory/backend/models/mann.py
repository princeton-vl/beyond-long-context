"""Differentiable controller backends (DNC + STM) for sequence inputs."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from dnc import DNC

from .backend_base import MemoryBackend, SequenceInputs
from .external.stm import STM


def _segment_stream(
    token_embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    projector: nn.Module,
    segment_length: int,
    method: Literal["flat", "avg"],
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    projected = projector(token_embeddings)
    batch_size, seq_len, hidden = projected.shape
    valid_mask = ~padding_mask
    pad_len = (-seq_len) % segment_length
    if pad_len:
        pad_embed = torch.zeros(batch_size, pad_len, hidden, device=projected.device, dtype=projected.dtype)
        pad_mask = torch.zeros(batch_size, pad_len, dtype=torch.bool, device=projected.device)
        projected = torch.cat([projected, pad_embed], dim=1)
        valid_mask = torch.cat([valid_mask, pad_mask], dim=1)

    segments = projected.view(batch_size, -1, segment_length, hidden)
    seg_mask = valid_mask.view(batch_size, -1, segment_length)
    if method == "flat":
        seg_repr = segments.reshape(batch_size, -1, segment_length * hidden)
    elif method == "avg":
        weights = seg_mask.unsqueeze(-1).type_as(projected)
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        summed = (segments * weights).sum(dim=2)
        seg_repr = summed / denom.squeeze(2)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    seg_mask = seg_mask.any(dim=2)
    return seg_repr, seg_mask


def _slice_stream_region(
    token_embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    stream_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stream_lengths.numel() == 0:
        empty_embeds = token_embeddings.new_zeros(0, 0, token_embeddings.size(-1))
        empty_mask = padding_mask.new_zeros(0, 0, dtype=torch.bool)
        return empty_embeds, empty_mask
    max_stream = int(stream_lengths.max().item()) if stream_lengths.numel() > 0 else 0
    if max_stream == 0:
        empty_embeds = token_embeddings.new_zeros(token_embeddings.size(0), 0, token_embeddings.size(-1))
        empty_mask = padding_mask.new_zeros(token_embeddings.size(0), 0, dtype=torch.bool)
        return empty_embeds, empty_mask
    stream_tokens = token_embeddings[:, :max_stream, :].clone()
    stream_mask = padding_mask[:, :max_stream].clone()
    positions = torch.arange(max_stream, device=token_embeddings.device).unsqueeze(0)
    stream_mask = stream_mask | (positions >= stream_lengths.unsqueeze(1))
    stream_tokens = stream_tokens.masked_fill(stream_mask.unsqueeze(-1), 0.0)
    return stream_tokens, stream_mask


def _extract_query_segments(
    token_embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    stream_lengths: torch.Tensor,
    label_mask: torch.Tensor,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    batch, seq_len, _ = token_embeddings.shape
    label_mask = label_mask.to(dtype=torch.bool)
    segments_per_batch: list[list[torch.Tensor]] = []
    label_positions_per_batch: list[torch.Tensor] = []
    for b in range(batch):
        start = int(stream_lengths[b].item())
        segments: list[torch.Tensor] = []
        positions: list[int] = []
        current: list[torch.Tensor] = []
        for pos in range(start, seq_len):
            if padding_mask[b, pos]:
                continue
            current.append(token_embeddings[b, pos])
            if label_mask[b, pos]:
                segments.append(torch.stack(current, dim=0))
                positions.append(pos)
                current = []
        segments_per_batch.append(segments)
        if positions:
            label_positions_per_batch.append(
                torch.tensor(positions, dtype=torch.long, device=token_embeddings.device)
            )
        else:
            label_positions_per_batch.append(
                torch.empty(0, dtype=torch.long, device=token_embeddings.device)
            )
    return segments_per_batch, label_positions_per_batch


class DNCBackend(MemoryBackend):
    def __init__(
        self,
        embed_dim: int,
        segment_length: int,
        mem_input_dim: int,
        segmentation_method: Literal["flat", "avg"] = "flat",
        hidden_size: int = 128,
        num_layers: int = 1,
        nr_cells: int = 16,
        read_heads: int = 4,
        cell_size: int = 32,
    ) -> None:
        self.segment_length = segment_length
        self.segmentation_method = segmentation_method
        self.per_token_dim = mem_input_dim
        self.segment_dim = (
            mem_input_dim * segment_length if segmentation_method == "flat" else mem_input_dim
        )
        super().__init__(
            self.segment_dim,
            projects_to_decoder_dim=False,
            requires_token_embeddings=True,
        )
        if self.segment_dim <= 0:
            raise ValueError("Segment dimension must be positive.")
        self.input_proj = nn.Linear(embed_dim, mem_input_dim)
        self.input_norm = nn.LayerNorm(self.segment_dim)
        self.dnc = DNC(
            input_size=self.segment_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nr_cells=nr_cells,
            read_heads=read_heads,
            cell_size=cell_size,
            batch_first=True,
        )
        self.sequence_res_proj = nn.Linear(self.segment_dim, self.segment_dim)
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError("DNCBackend requires label_mask to align outputs with labeled tokens.")
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("DNCBackend requires token embeddings.")
        padding_mask = self._resolve_padding_mask(sequence)
        stream_lengths = self._ensure_stream_lengths(sequence)
        stream_tokens, stream_mask = _slice_stream_region(token_embeddings, padding_mask, stream_lengths)
        if stream_tokens.size(1) > 0:
            stream_segments, _ = _segment_stream(
                stream_tokens,
                stream_mask,
                self.input_proj,
                self.segment_length,
                self.segmentation_method,
            )
        else:
            stream_segments = token_embeddings.new_zeros(
                token_embeddings.size(0), 0, self.segment_dim
            )
        query_segment_lists, label_positions = _extract_query_segments(
            token_embeddings,
            padding_mask,
            stream_lengths,
            label_mask,
        )
        chunk_counts = [
            sum(len(self._chunk_query_tokens(seg)) for seg in segments)
            for segments in query_segment_lists
        ]
        max_chunks = max(chunk_counts, default=0)
        device = token_embeddings.device
        query_segment_tensor = token_embeddings.new_zeros(
            token_embeddings.size(0), max_chunks, self.segment_dim
        )
        label_index_tensor = torch.full(
            (token_embeddings.size(0), max_chunks),
            -1,
            dtype=torch.long,
            device=device,
        )
        for b, segments in enumerate(query_segment_lists):
            positions = label_positions[b]
            slot = 0
            for query_idx, token_chunk in enumerate(segments):
                chunks = self._chunk_query_tokens(token_chunk)
                for chunk_idx, chunk in enumerate(chunks):
                    if slot >= max_chunks:
                        break
                    query_segment_tensor[b, slot] = self._encode_query_tokens(chunk)
                    if chunk_idx == len(chunks) - 1 and query_idx < positions.size(0):
                        label_index_tensor[b, slot] = positions[query_idx]
                    slot += 1
        combined = torch.cat([stream_segments, query_segment_tensor], dim=1)
        if combined.size(1) == 0:
            empty_hidden = token_embeddings.new_zeros(
                token_embeddings.size(0), token_embeddings.size(1), self.segment_dim
            )
            return self._mask_embeddings(self._project_hidden(empty_hidden), padding_mask)
        self._sync_dnc_device(combined.device)
        normed = self.input_norm(combined)
        residual_inputs = self.sequence_res_proj(combined)
        outputs, _state = self.dnc(normed, reset_experience=True)
        outputs = outputs + residual_inputs
        if max_chunks > 0:
            query_hidden = outputs[:, -max_chunks:, :]
        else:
            query_hidden = outputs.new_zeros(outputs.size(0), 0, outputs.size(2))
        hidden_full = token_embeddings.new_zeros(
            token_embeddings.size(0), token_embeddings.size(1), query_hidden.size(-1)
        )
        for b in range(hidden_full.size(0)):
            valid = label_index_tensor[b] >= 0
            if not valid.any():
                continue
            positions = label_index_tensor[b, valid]
            values = query_hidden[b, : positions.size(0), :]
            if values.dtype != hidden_full.dtype:
                values = values.to(dtype=hidden_full.dtype)
            hidden_full[b].index_copy_(0, positions, values)
        projected = self._project_hidden(hidden_full)
        return self._mask_embeddings(projected, padding_mask)

    def _encode_query_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(tokens)
        if self.segmentation_method == "flat":
            length = min(projected.size(0), self.segment_length)
            buf = projected.new_zeros(self.segment_length, projected.size(-1))
            buf[:length] = projected[:length]
            return buf.reshape(-1)
        return projected.mean(dim=0)

    def _chunk_query_tokens(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        length = tokens.size(0)
        if length <= self.segment_length:
            chunks.append(tokens)
            return chunks
        for start in range(0, length, self.segment_length):
            chunks.append(tokens[start : start + self.segment_length])
        return chunks

    def _sync_dnc_device(self, device: torch.device) -> None:
        gpu_id = -1 if device.type == "cpu" else device.index
        if self.dnc.gpu_id == gpu_id:
            return
        self.dnc.gpu_id = gpu_id
        for mem in self.dnc.memories:
            mem.gpu_id = gpu_id
            mem.I = mem.I.to(device=device)


class STMBackend(MemoryBackend):
    def __init__(
        self,
        embed_dim: int,
        segment_length: int,
        stm_input_dim: int,
        segmentation_method: Literal["flat", "avg"] = "avg",
        stm_kwargs: dict | None = None,
    ) -> None:
        self.segment_length = segment_length
        self.segmentation_method = segmentation_method
        self.per_token_dim = stm_input_dim
        self.segment_dim = (
            stm_input_dim * segment_length if segmentation_method == "flat" else stm_input_dim
        )
        super().__init__(
            self.segment_dim,
            projects_to_decoder_dim=False,
            requires_token_embeddings=True,
        )
        self.input_proj = nn.Linear(embed_dim, stm_input_dim)
        kwargs = dict(stm_kwargs or {})
        self.stm = STM(self.segment_dim, self.segment_dim, **kwargs)
        self.step_norm = nn.LayerNorm(self.segment_dim)
        self.step_res_proj = nn.Linear(self.segment_dim, self.segment_dim)
        self.supports_direct_logits = True

    def _encode_query_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(tokens)
        if self.segmentation_method == "flat":
            length = min(projected.size(0), self.segment_length)
            buf = projected.new_zeros(self.segment_length, projected.size(-1))
            buf[:length] = projected[:length]
            return buf.reshape(-1)
        return projected.mean(dim=0)

    def _chunk_query_tokens(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        length = tokens.size(0)
        if length <= self.segment_length:
            chunks.append(tokens)
            return chunks
        for start in range(0, length, self.segment_length):
            chunks.append(tokens[start : start + self.segment_length])
        return chunks

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError("STMBackend requires label_mask to align outputs with labeled tokens.")
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("STMBackend requires token embeddings.")
        padding_mask = self._resolve_padding_mask(sequence)
        stream_lengths = self._ensure_stream_lengths(sequence)
        stream_tokens, stream_mask = _slice_stream_region(token_embeddings, padding_mask, stream_lengths)
        if stream_tokens.size(1) > 0:
            stream_segments, _ = _segment_stream(
                stream_tokens,
                stream_mask,
                self.input_proj,
                self.segment_length,
                self.segmentation_method,
            )
        else:
            stream_segments = token_embeddings.new_zeros(
                token_embeddings.size(0), 0, self.segment_dim
            )
        query_segment_lists, label_positions = _extract_query_segments(
            token_embeddings,
            padding_mask,
            stream_lengths,
            label_mask,
        )
        chunk_counts = [
            sum(len(self._chunk_query_tokens(seg)) for seg in segments)
            for segments in query_segment_lists
        ]
        max_chunks = max(chunk_counts, default=0)
        device = token_embeddings.device
        query_segment_tensor = token_embeddings.new_zeros(
            token_embeddings.size(0), max_chunks, self.segment_dim
        )
        label_index_tensor = torch.full(
            (token_embeddings.size(0), max_chunks),
            -1,
            dtype=torch.long,
            device=device,
        )
        for b, segments in enumerate(query_segment_lists):
            positions = label_positions[b]
            slot = 0
            for query_idx, token_chunk in enumerate(segments):
                chunks = self._chunk_query_tokens(token_chunk)
                for chunk_idx, chunk in enumerate(chunks):
                    if slot >= max_chunks:
                        break
                    query_segment_tensor[b, slot] = self._encode_query_tokens(chunk)
                    if chunk_idx == len(chunks) - 1 and query_idx < positions.size(0):
                        label_index_tensor[b, slot] = positions[query_idx]
                    slot += 1
        combined = torch.cat([stream_segments, query_segment_tensor], dim=1)
        if combined.size(1) == 0:
            empty_hidden = token_embeddings.new_zeros(
                token_embeddings.size(0), token_embeddings.size(1), self.segment_dim
            )
            return self._mask_embeddings(self._project_hidden(empty_hidden), padding_mask)
        state = self.stm.create_new_state(combined.size(0))
        state = tuple(part.to(device=combined.device, dtype=combined.dtype) for part in state)
        stream_steps = stream_segments.size(1)
        target_dtype = token_embeddings.dtype
        collected: list[torch.Tensor] = []
        for t in range(combined.size(1)):
            step = combined[:, t, :]
            normed = self.step_norm(step)
            rel_out, state = self.stm.compute(normed, state)
            logits = self.stm.decode_output(rel_out).to(dtype=target_dtype)
            logits = logits + self.step_res_proj(step)
            if t >= stream_steps:
                collected.append(logits.unsqueeze(1))
        if max_chunks > 0 and collected:
            query_hidden = torch.cat(collected, dim=1)
        else:
            query_hidden = token_embeddings.new_zeros(token_embeddings.size(0), 0, self.segment_dim)
        hidden_full = token_embeddings.new_zeros(
            token_embeddings.size(0), token_embeddings.size(1), query_hidden.size(-1)
        )
        for b in range(hidden_full.size(0)):
            valid = label_index_tensor[b] >= 0
            if not valid.any():
                continue
            positions = label_index_tensor[b, valid]
            values = query_hidden[b, : positions.size(0), :]
            if values.dtype != hidden_full.dtype:
                values = values.to(dtype=hidden_full.dtype)
            hidden_full[b].index_copy_(0, positions, values)
        projected = self._project_hidden(hidden_full)
        return self._mask_embeddings(projected, padding_mask)
