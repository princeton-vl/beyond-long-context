"""Dataset builder for binary membership/continuation manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset

from calibrated_memory.data.sequences.common import (
    IGNORE_INDEX,
    TOKEN_OFFSET,
    YES_TOKEN,
    NO_TOKEN,
)
from calibrated_memory.data.sequences.metadata_utils import build_video_metadata
from calibrated_memory.data.sequences.question_generator import build_samples
from calibrated_memory.data.sequences.question_importer import (
    _resolve_sequence_keys,
    _resolve_sequence_offsets,
    _extract_sequence_bundle,
    build_question_metadata,
    is_positive_answer,
    question_mode,
)
from calibrated_memory.data.sequences.sources import BucketManifestSource, SequenceRecord


@dataclass(frozen=True)
class EvaluationConfig:
    manifest_path: Path
    sequence_key: str | None = None
    num_videos: int = -1
    truncate_len: int = 0
    task: str = "membership"
    cont_len: int = 3
    token_offset: int = TOKEN_OFFSET
    manifest_root: Path | None = None


class EvaluationDataset(Dataset):
    """Wraps manifest-defined binary questions for evaluation."""

    def __init__(
        self,
        samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]],
        *,
        pad_id: int,
        vocab_size: int,
        max_input_len: int,
        metadata_summary: dict[str, Any],
    ) -> None:
        if not samples:
            raise ValueError("EvaluationDataset requires at least one sample")
        self._samples = samples
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.metadata_summary = metadata_summary

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


def build_evaluation_dataset(config: EvaluationConfig) -> EvaluationDataset:
    records, summary = _load_records(config)
    if config.task not in {"membership", "continuation"}:
        raise NotImplementedError(
            f"Evaluation currently supports membership/continuation; got {config.task!r}."
        )
    builder = (
        _iter_membership_questions if config.task == "membership" else _iter_continuation_questions
    )
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    vocab_max = config.token_offset
    max_len = 0
    for sample in builder(records, config):
        seq, labels, stream_len, extras = sample
        samples.append(sample)
        if seq.numel():
            vocab_max = max(vocab_max, int(seq.max().item()))
            max_len = max(max_len, seq.numel())
    if not samples:
        raise RuntimeError("Manifest did not yield any evaluable questions.")
    pad_id = vocab_max + 1
    vocab_size = pad_id + 1
    metadata_summary = dict(summary)
    metadata_summary.update(
        {
            "task": config.task,
            "question_count": len(samples),
            "manifest_path": str(config.manifest_path),
        }
    )
    return EvaluationDataset(
        samples,
        pad_id=pad_id,
        vocab_size=vocab_size,
        max_input_len=max_len,
        metadata_summary=metadata_summary,
    )


def _load_records(config: EvaluationConfig) -> tuple[list[SequenceRecord], dict[str, Any]]:
    manifest_path = Path(config.manifest_path).expanduser()
    root_path = config.manifest_root
    if root_path is not None and not isinstance(root_path, Path):
        root_path = Path(root_path)
    sequence_keys = _parse_sequence_key_arg(config.sequence_key)
    source = BucketManifestSource(
        manifest_path,
        token_offset=config.token_offset,
        num_videos=config.num_videos,
        truncate_len=config.truncate_len,
        sequence_keys=sequence_keys,
        root_path=root_path,
        include_questions=True,
    )
    return source.records, source.summary


def _parse_sequence_key_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    trimmed = raw.strip()
    if not trimmed or trimmed.lower() == "auto":
        return None
    if "," in trimmed:
        values = [chunk.strip() for chunk in trimmed.split(",") if chunk.strip()]
        return values or None
    return [trimmed]


def _iter_membership_questions(
    records: Iterable[SequenceRecord],
    config: EvaluationConfig,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]:
    for record in records:
        metadata = record.metadata or {}
        stream_tokens = [int(tok) for tok in record.tokens]
        video_meta = build_video_metadata(metadata, stream_length=len(stream_tokens))
        sequence_keys = _resolve_sequence_keys(metadata)
        if not sequence_keys:
            continue
        offsets = _resolve_sequence_offsets(metadata, config.token_offset)
        questions = metadata.get("questions") or []
        for question in questions:
            q_type = str(question.get("question_type") or "").lower()
            if q_type == "spatial":
                continue
            fmt = str(question.get("question_format") or "").lower()
            if "binary" not in fmt and question_mode(question) != "exists":
                continue
            candidate = question.get("candidate") or {}
            tokens, _ = _extract_sequence_bundle(candidate, sequence_keys, offsets)
            if tokens is None:
                continue
            label_token = YES_TOKEN if is_positive_answer(question) else NO_TOKEN
            seq, labels, stream_len = _format_membership_sample(stream_tokens, tokens, label_token)
            extras = {
                "metadata": build_question_metadata(
                    video_meta,
                    question,
                    prefix_length=len(tokens),
                    candidate_length=len(tokens),
                )
            }
            yield seq, labels, stream_len, extras


def _iter_continuation_questions(
    records: Iterable[SequenceRecord],
    config: EvaluationConfig,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]:
    for record in records:
        metadata = record.metadata or {}
        stream_tokens = [int(tok) for tok in record.tokens]
        video_meta = build_video_metadata(metadata, stream_length=len(stream_tokens))
        sequence_keys = _resolve_sequence_keys(metadata)
        if not sequence_keys:
            continue
        offsets = _resolve_sequence_offsets(metadata, config.token_offset)
        questions = metadata.get("questions") or []
        for question in questions:
            q_type = str(question.get("question_type") or "").lower()
            if q_type == "spatial":
                continue
            fmt = str(question.get("question_format") or "").lower()
            if "continuation" not in fmt and question_mode(question) != "continuation":
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
            seq, labels, stream_len = _format_continuation_sample(
                stream_tokens,
                prefix_tokens,
                candidate_tokens,
                label_token,
            )
            extras = {
                "metadata": build_question_metadata(
                    video_meta,
                    question,
                    prefix_length=len(prefix_tokens),
                    candidate_length=len(candidate_tokens),
                )
            }
            yield seq, labels, stream_len, extras

def _format_membership_sample(
    stream_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    label_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _query_fn(_: Sequence[int]) -> list[tuple[list[int], bool]]:
        return [(list(candidate_tokens), label_token == YES_TOKEN)]

    samples, _, _, _ = build_samples([stream_tokens], _query_fn, task="membership")
    return samples[0]


def _format_continuation_sample(
    stream_tokens: Sequence[int],
    prefix_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    label_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _query_fn(_: Sequence[int]) -> list[tuple[list[int], list[int], int]]:
        return [(list(prefix_tokens), list(candidate_tokens), label_token)]

    samples, _, _, _ = build_samples([stream_tokens], _query_fn, task="continuation")
    return samples[0]
