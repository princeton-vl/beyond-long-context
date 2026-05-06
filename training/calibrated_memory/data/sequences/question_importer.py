"""Helpers for ingesting manifest-defined sequences and prefixes."""

from __future__ import annotations

from typing import Any, Sequence


def _coerce_tokens(raw: Sequence[str | int], token_offset: int) -> list[int]:
    return [int(tok) + token_offset for tok in raw]


def question_mode(question: dict[str, Any]) -> str:
    return str(question.get("question_mode") or "exists").lower()


def question_type(question: dict[str, Any]) -> str:
    return str(question.get("question_type") or "").lower()


def is_positive_answer(question: dict[str, Any]) -> bool:
    answer = str(question.get("answer") or "").lower()
    if answer in {"yes", "true", "present"}:
        return True
    if answer in {"no", "false", "absent"}:
        return False
    present = question.get("present")
    return bool(present)


def build_question_metadata(
    video_meta: dict[str, Any] | None,
    question: dict[str, Any],
    *,
    prefix_length: int,
    candidate_length: int,
) -> dict[str, Any]:
    payload = dict(video_meta or {})
    metadata: dict[str, Any] = {
        "video": payload,
        "question_index": question.get("question_index"),
        "stream_prefix_length": float(prefix_length),
        "target_length": float(candidate_length),
        "entropy_prefix": question.get("entropy_prefix"),
        "concerned_ranges": question.get("concerned_ranges"),
        "scenario": question.get("scenario"),
        "question_time": question.get("question_time"),
        "bucket_id": question.get("bucket_id") or payload.get("bucket_id"),
    }
    stream_length = payload.get("length_value") or payload.get("stream_length")
    if stream_length is not None:
        try:
            metadata["stream_total_length"] = float(stream_length)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            pass
    return metadata


def _resolve_sequence_keys(metadata: dict[str, Any]) -> list[str]:
    keys = metadata.get("sequence_keys")
    if keys:
        return [str(key) for key in keys if key]
    key = metadata.get("sequence_key")
    return [key] if key else []


def _resolve_sequence_offsets(metadata: dict[str, Any], fallback: int) -> dict[str, int]:
    offsets = metadata.get("sequence_offsets")
    if offsets:
        return {str(key): int(value) for key, value in offsets.items()}
    keys = _resolve_sequence_keys(metadata)
    if not keys:
        return {}
    return {keys[0]: fallback}


def _extract_sequence_bundle(
    option: dict[str, Any] | Sequence[int] | None,
    sequence_keys: Sequence[str],
    offsets: dict[str, int],
    *,
    cont_len: int | None = None,
) -> tuple[list[int], dict[str, int]] | tuple[None, None]:
    if not option:
        return None, None
    if isinstance(option, (list, tuple)):
        option = {"sequence": option}
    sequences = option.get("sequences") or {}
    combined: list[int] = []
    lengths: dict[str, int] = {}
    for key in sequence_keys:
        payload = sequences.get(key)
        if payload is None and isinstance(option, dict):
            payload = option.get(key)
        if payload is None and len(sequence_keys) == 1:
            payload = option.get("sequence")
        if not payload:
            continue
        tokens = _coerce_tokens(payload, offsets.get(key, 0))
        if cont_len is not None and len(tokens) < cont_len:
            return None, None
        truncated = tokens[:cont_len] if cont_len is not None else tokens
        combined.extend(truncated)
        lengths[key] = len(truncated)
    if not combined:
        return None, None
    return combined, lengths


def _extract_prefix_bundle(
    question: dict[str, Any] | None,
    sequence_keys: Sequence[str],
    offsets: dict[str, int],
) -> list[int] | None:
    if not question:
        return None
    prefixes = question.get("prefix")
    if prefixes is None:
        return None
    if isinstance(prefixes, (list, tuple)):
        key = sequence_keys[0] if sequence_keys else None
        offset = offsets.get(key, 0) if key is not None else 0
        return _coerce_tokens(prefixes, offset)
    combined: list[int] = []
    for key in sequence_keys:
        payload = prefixes.get(key)
        if not payload:
            continue
        combined.extend(_coerce_tokens(payload, offsets.get(key, 0)))
    return combined if combined else None
