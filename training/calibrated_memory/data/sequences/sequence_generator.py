"""Datasets for manifest-driven and synthetic binary questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from .common import NO_TOKEN, TOKEN_OFFSET, YES_TOKEN, validate_stream_token_offset
from .metadata_utils import build_video_metadata
from .question_generator import (
    build_samples,
    build_continuation_samples,
    build_membership_samples,
)
from .question_importer import (
    _extract_sequence_bundle,
    _resolve_sequence_keys,
    _resolve_sequence_offsets,
    build_question_metadata,
    is_positive_answer,
    question_mode,
    question_type,
)
from .sources import BucketManifestSource, SequenceRecord, _compute_tertiles


def _resolve_video_entropy(entropy_map: dict[str, Any] | None) -> float | None:
    if not isinstance(entropy_map, dict):
        return None
    for key in ("empirical_bits", "analytic_bits"):
        value = entropy_map.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _resolve_builder(task: str) -> Callable:
    mapping = {
        "membership": build_membership_samples,
        "continuation": build_continuation_samples,
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'")
    return mapping[task]


class SequenceDataset(Dataset):
    """Creates query/respond datasets from token streams plus metadata."""

    def __init__(
        self,
        json_path: Path | None = None,
        num_videos: int = -1,
        unique_sequences: int = 50,
        token_offset: int = TOKEN_OFFSET,
        task: str = "membership",
        cont_len: int = 3,
        vocab_size: int | None = None,
        manifest_root: str | Path | None = None,
        *,
        records: Sequence[SequenceRecord] | None = None,
        metadata_summary: dict[str, Any] | None = None,
        sequence_keys: Sequence[str] | None = None,
        max_stream_len: int | None = None,
    ) -> None:
        if task not in {"membership", "continuation"}:
            raise ValueError(f"Unsupported dataset task '{task}'")
        token_offset = validate_stream_token_offset(token_offset)
        record_list: list[SequenceRecord]
        summary: dict[str, Any] | None = metadata_summary
        stream_limit = int(max_stream_len) if max_stream_len and max_stream_len > 0 else 0
        if records is not None:
            record_list = list(records)
        else:
            if json_path is None:
                raise ValueError("json_path is required when records are not provided")
            source = BucketManifestSource(
                json_path,
                token_offset=token_offset,
                num_videos=num_videos,
                truncate_len=0,
                sequence_keys=sequence_keys,
                root_path=Path(manifest_root) if manifest_root else None,
                include_questions=True,
                max_stream_len=stream_limit,
            )
            record_list = source.records
            summary = summary or source.summary
        if stream_limit > 0:
            filtered: list[SequenceRecord] = []
            skipped = 0
            for rec in record_list:
                if len(rec.tokens) > stream_limit:
                    skipped += 1
                    continue
                filtered.append(rec)
            record_list = filtered
            if summary is None:
                summary = {}
            summary = dict(summary)
            summary["skipped_streams_over_limit"] = (
                int(summary.get("skipped_streams_over_limit", 0)) + skipped
            )
        if not record_list:
            raise ValueError("SequenceDataset requires at least one stream")
        raw_streams: list[list[int]] = []
        raw_metadata: list[dict[str, Any]] = []
        raw_video_metadata: list[dict[str, Any] | None] = []
        stream_lengths: list[int] = []
        stream_queries: list[list[dict[str, Any]]] = []
        type_whitelist = {"sequential"} if task == "continuation" else None
        for rec in record_list:
            tokens = [int(tok) for tok in rec.tokens]
            raw_streams.append(tokens)
            meta = dict(rec.metadata)
            stream_length = len(tokens)
            stream_lengths.append(stream_length)
            meta["stream_length"] = stream_length
            meta.setdefault("length_value", float(stream_length))
            if "entropy_value" not in meta:
                entropy_map = meta.get("entropy_overall") or {}
                entropy_value = _resolve_video_entropy(entropy_map)
                if entropy_value is not None:
                    meta["entropy_value"] = entropy_value
            video_meta = build_video_metadata(meta, stream_length=stream_length)
            raw_metadata.append(meta)
            raw_video_metadata.append(video_meta)
            queries = _collect_manifest_queries(
                meta,
                video_meta=video_meta,
                task=task,
                cont_len=cont_len,
                token_offset=token_offset,
                question_type_whitelist=type_whitelist,
            )
            stream_queries.append(queries)

        use_manifest_questions = any(stream_queries)
        if use_manifest_questions:
            filtered_streams: list[list[int]] = []
            filtered_queries: list[list[dict[str, Any]]] = []
            filtered_metadata: list[dict[str, Any]] = []
            filtered_video_meta: list[dict[str, Any] | None] = []
            for tokens, queries, meta, video_meta in zip(
                raw_streams, stream_queries, raw_metadata, raw_video_metadata
            ):
                if not queries:
                    continue
                filtered_streams.append(tokens)
                filtered_queries.append(queries)
                filtered_metadata.append(meta)
                filtered_video_meta.append(video_meta)
            if filtered_streams:
                (
                    self.samples,
                    self.pad_id,
                    self.vocab_size,
                    self.max_input_len,
                    manifest_metadata,
                ) = _build_manifest_samples(
                    filtered_streams,
                    filtered_queries,
                    task=task,
                    cont_len=cont_len,
                    vocab_size=vocab_size,
                )
                self.sample_metadata = filtered_metadata
                self._video_metadata = filtered_video_meta
                self._query_metadata = manifest_metadata
                stream_lengths = [len(tokens) for tokens in filtered_streams]
            else:
                use_manifest_questions = False

        if not use_manifest_questions:
            builder = _resolve_builder(task)
            builder_kwargs = {
                "streams": raw_streams,
                "unique_sequences": unique_sequences,
                "vocab_size": vocab_size,
                "cont_len": cont_len,
                "token_offset": token_offset,
            }
            self.samples, self.pad_id, self.vocab_size, self.max_input_len = builder(
                **builder_kwargs
            )
            self.sample_metadata = raw_metadata
            self._video_metadata = raw_video_metadata
            self._query_metadata = [None] * len(self.samples)
            stream_lengths = [len(tokens) for tokens in raw_streams]

        summary = summary or {}
        if stream_lengths:
            summary = dict(summary)
            summary.setdefault("stream_length_tertiles", _compute_tertiles(stream_lengths))
        self.metadata_summary = summary
        if len(self._query_metadata) < len(self.samples):
            padding = len(self.samples) - len(self._query_metadata)
            self._query_metadata.extend(None for _ in range(padding))
        if not self._video_metadata:
            self._video_metadata = [
                {
                    "stream_length": meta.get("stream_length"),
                    "length_value": meta.get("length_value"),
                }
                for meta in self.sample_metadata
            ]
        if len(self._video_metadata) < len(self.samples):
            padding = len(self.samples) - len(self._video_metadata)
            self._video_metadata.extend({} for _ in range(padding))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query_metadata = None
        if self._query_metadata and idx < len(self._query_metadata):
            query_metadata = self._query_metadata[idx]
        video_meta = None
        if self._video_metadata and idx < len(self._video_metadata):
            video_meta = self._video_metadata[idx]
        if query_metadata or video_meta:
            seq, labels, stream_len = sample
            payload: dict[str, Any] = {}
            if video_meta:
                payload["video"] = dict(video_meta)
            if query_metadata:
                payload["queries"] = [dict(entry) for entry in query_metadata]
            extras = {"metadata": payload}
            return seq, labels, stream_len, extras
        return sample

    def metadata_for_index(self, idx: int) -> dict[str, Any]:
        """Return stored metadata for a stream aligned with __getitem__."""

        return dict(self.sample_metadata[idx])


def _collect_manifest_queries(
    metadata: dict[str, Any],
    *,
    video_meta: dict[str, Any] | None,
    task: str,
    cont_len: int,
    token_offset: int,
    question_type_whitelist: set[str] | None,
) -> list[dict[str, Any]]:
    sequence_keys = _resolve_sequence_keys(metadata)
    if not sequence_keys:
        return []
    offsets = _resolve_sequence_offsets(metadata, token_offset)
    questions = metadata.get("questions") or []
    collected: list[dict[str, Any]] = []
    for question in questions:
        q_type = question_type(question)
        if q_type == "spatial":
            # Spatial questions rely on auxiliary sequence channels (e.g., S_lanes)
            # that are not available in membership/continuation training runs.
            continue
        if question_type_whitelist and q_type not in question_type_whitelist:
            continue
        fmt = str(question.get("question_format") or "").lower()
        mode = question_mode(question)
        if task == "continuation":
            if "continuation" not in fmt and mode != "continuation":
                continue
            prefix_bundle = _extract_sequence_bundle(
                question.get("prefix", {}),
                sequence_keys,
                offsets,
            )
            if prefix_bundle[0] is None:
                continue
            candidate_bundle = _extract_sequence_bundle(
                question.get("candidate", {}),
                sequence_keys,
                offsets,
            )
            if candidate_bundle[0] is None:
                continue
            prefix_tokens, _ = prefix_bundle
            candidate_tokens, _ = candidate_bundle
            label_token = YES_TOKEN if is_positive_answer(question) else NO_TOKEN
            metadata_entry = build_question_metadata(
                video_meta,
                question,
                prefix_length=len(prefix_tokens),
                candidate_length=len(candidate_tokens),
            )
            collected.append(
                {
                    "query": (list(prefix_tokens), list(candidate_tokens), label_token),
                    "metadata": metadata_entry,
                }
            )
        else:
            if "binary" not in fmt and mode != "exists":
                continue
            candidate_bundle = _extract_sequence_bundle(
                question.get("candidate", {}),
                sequence_keys,
                offsets,
            )
            if candidate_bundle[0] is None:
                continue
            tokens, _ = candidate_bundle
            metadata_entry = build_question_metadata(
                video_meta,
                question,
                prefix_length=len(tokens),
                candidate_length=len(tokens),
            )
            collected.append(
                {
                    "query": (list(tokens), is_positive_answer(question)),
                    "metadata": metadata_entry,
                }
            )
    return collected


def _build_manifest_samples(
    streams: Sequence[Sequence[int]],
    query_entries: Sequence[list[dict[str, Any]]],
    *,
    task: str,
    cont_len: int,
    vocab_size: int | None,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    int,
    int,
    int,
    list[list[dict[str, Any]] | None],
]:
    iterator = iter(query_entries)

    def query_fn(_: Sequence[int]) -> list:
        entries = next(iterator, [])
        return [entry["query"] for entry in entries]

    samples, pad_id, vocab_total, max_input_len = build_samples(
        streams,
        query_fn,
        task=task,
        cont_len=cont_len,
        vocab_size=vocab_size,
    )
    metadata_entries: list[list[dict[str, Any]] | None] = []
    for entries in query_entries:
        payload = [entry["metadata"] for entry in entries if entry.get("metadata")]
        metadata_entries.append(payload or None)
    return samples, pad_id, vocab_total, max_input_len, metadata_entries


class SyntheticSampleFactory:
    """Reusable synthetic sample builder that mirrors SyntheticSequenceDataset."""

    def __init__(
        self,
        *,
        seq_len: int | None = None,
        unique_sequences: int,
        vocab_size: int,
        seed: int = 0,
        task: str = "membership",
        cont_len: int = 3,
        seq_len_range: tuple[int, int] | None = None,
        token_offset: int = TOKEN_OFFSET,
    ) -> None:
        if seq_len is None and seq_len_range is None:
            raise ValueError("SyntheticSampleFactory requires seq_len or seq_len_range.")
        if seq_len_range is None:
            seq_len_range = (int(seq_len), int(seq_len))
        min_len, max_len = seq_len_range
        if min_len <= 0 or max_len <= 0:
            raise ValueError("Sequence lengths must be positive.")
        if min_len > max_len:
            raise ValueError("seq_len_range must be (min_len, max_len).")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        self.generator = torch.Generator().manual_seed(int(seed))
        self.unique_sequences = int(unique_sequences)
        self.vocab_size = int(vocab_size)
        self.task = str(task)
        if self.task not in {"membership", "continuation"}:
            raise ValueError(f"Unsupported synthetic task '{self.task}'")
        self.cont_len = int(cont_len)
        self._stream_offset = max(int(token_offset), TOKEN_OFFSET)
        self._seq_len_bounds = (int(min_len), int(max_len))
        self._builder = _resolve_builder(self.task)

    def _sample_length(self) -> int:
        low, high = self._seq_len_bounds
        if low == high:
            return low
        return int(torch.randint(low, high + 1, (1,), generator=self.generator).item())

    def _sample_stream(self) -> list[int]:
        length = self._sample_length()
        high = self._stream_offset + self.vocab_size
        if high <= self._stream_offset:
            raise RuntimeError("Invalid token range for synthetic stream generation.")
        return torch.randint(
            self._stream_offset,
            high,
            (length,),
            generator=self.generator,
        ).tolist()

    def _builder_kwargs(self, streams: list[list[int]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "streams": streams,
            "unique_sequences": self.unique_sequences,
            "generator": self.generator,
            "vocab_size": self.vocab_size,
            "token_offset": self._stream_offset,
        }
        kwargs["cont_len"] = self.cont_len
        return kwargs

    def build_batch(
        self,
        count: int,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int, int, int]:
        if count <= 0:
            raise ValueError("SyntheticSampleFactory requires count > 0.")
        streams = [self._sample_stream() for _ in range(count)]
        return self._builder(**self._builder_kwargs(streams))

    def build_samples(self, count: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        samples, _, _, _ = self.build_batch(count)
        return samples


class SyntheticSampleDataset(Dataset):
    """Dataset wrapper that serves pre-generated synthetic samples."""

    def __init__(self, samples: Sequence[tuple[Any, Any, Any]]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


class SyntheticSequenceDataset(Dataset):
    """Generates random streams and applies the same query logic."""

    def __init__(
        self,
        num_sequences: int,
        unique_sequences: int,
        seq_len: int | None = None,
        vocab_size: int = 16,
        seed: int = 0,
        task: str = "membership",
        cont_len: int = 3,
        *,
        seq_len_min: int | None = None,
        seq_len_max: int | None = None,
    ):
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        length_value: int | None = None
        length_range: tuple[int, int] | None = None
        if seq_len is not None:
            length_value = int(seq_len)
            if length_value <= 0:
                raise ValueError("seq_len must be positive")
            self.seq_len_bounds = (length_value, length_value)
        else:
            if seq_len_min is None or seq_len_max is None:
                raise ValueError(
                    "SyntheticSequenceDataset requires seq_len or both seq_len_min and seq_len_max."
                )
            low = int(seq_len_min)
            high = int(seq_len_max)
            if low <= 0 or high <= 0:
                raise ValueError("seq_len_min and seq_len_max must be positive")
            if low > high:
                raise ValueError("seq_len_min cannot exceed seq_len_max")
            length_range = (low, high)
            self.seq_len_bounds = length_range
        self._factory = SyntheticSampleFactory(
            seq_len=length_value,
            seq_len_range=length_range,
            unique_sequences=unique_sequences,
            vocab_size=vocab_size,
            seed=seed,
            task=task,
            cont_len=cont_len,
        )
        samples, pad_id, vocab_total, max_input = self._factory.build_batch(num_sequences)
        self.samples = samples
        self.pad_id = pad_id
        self.vocab_size = vocab_total
        self.max_input_len = max_input
        self.metadata_summary = {
            "source": "synthetic",
            "num_sequences": num_sequences,
            "seq_len_min": self.seq_len_bounds[0],
            "seq_len_max": self.seq_len_bounds[1],
            "vocab_size": vocab_size,
            "task": task,
            "seed": seed,
        }
        if self.seq_len_bounds[0] == self.seq_len_bounds[1]:
            self.metadata_summary["seq_len"] = self.seq_len_bounds[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
