from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import warnings


@dataclass(frozen=True)
class SequenceRecord:
    """Container for a tokenized stream and associated metadata."""

    tokens: list[int]
    metadata: dict[str, Any]


def _stat_summary(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return {}
    total = sum(data)
    count = len(data)
    return {
        "count": count,
        "min": min(data),
        "max": max(data),
        "mean": total / float(count),
    }


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    lower_val = float(sorted_values[lower])
    upper_val = float(sorted_values[upper])
    weight = position - lower
    return lower_val + (upper_val - lower_val) * weight


def _compute_tertiles(values: Iterable[float]) -> list[float] | None:
    data = sorted(float(v) for v in values if v is not None)
    if len(data) < 3:
        return None
    first = _quantile(data, 1.0 / 3.0)
    second = _quantile(data, 2.0 / 3.0)
    return [first, second]


def _load_manifest_videos(path: Path, num_videos: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    videos = list(manifest.get("videos", []))
    if num_videos > 0:
        videos = videos[:num_videos]
    if not videos:
        raise ValueError(f"No videos found in manifest: {path}")
    return manifest, videos


def _select_sequence_key(
    seq_dict: Dict[str, Sequence[str | int]],
    requested: str | None,
) -> str:
    if not seq_dict:
        raise ValueError("Expected at least one sequence in sequences_used")
    if requested:
        if requested not in seq_dict:
            raise KeyError(f"Sequence '{requested}' not present in manifest")
        return requested
    return max(seq_dict.items(), key=lambda kv: max(int(x) for x in kv[1]))[0]


def _resolve_sequence_key_list(
    videos: Sequence[dict[str, Any]],
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        return [key for key in requested if key]
    for video in videos:
        sequences = video.get("sequences_used") or {}
        if sequences:
            return [_select_sequence_key(sequences, None)]
    raise ValueError("Unable to infer sequence keys from manifest")


def _estimate_sequence_vocab(
    videos: Sequence[dict[str, Any]],
    sequence_key: str,
) -> int:
    max_token = 0
    for video in videos:
        sequences = video.get("sequences_used") or {}
        tokens = sequences.get(sequence_key)
        if not tokens:
            continue
        try:
            local_max = max(int(tok) for tok in tokens)
        except (TypeError, ValueError):
            continue
        max_token = max(max_token, local_max)
    return max_token + 1 if max_token >= 0 else 1


def _load_sequence_vocab(
    manifest: dict[str, Any],
    manifest_path: Path,
    sequence_keys: Sequence[str],
    videos: Sequence[dict[str, Any]],
) -> dict[str, int]:
    sources = manifest.get("sequence_sources") or {}
    vocab_map: dict[str, int] = {}
    for key in sequence_keys:
        source_rel = sources.get(key)
        vocab_value: int | None = None
        if source_rel:
            resolved_path = Path(_resolve_path(source_rel, _resolve_root(manifest_path, None)))
            if resolved_path.exists():
                try:
                    payload = json.loads(resolved_path.read_text())
                    sequences = payload.get("sequences") or []
                    if sequences:
                        vocab_entry = sequences[0].get("vocab_size")
                        if vocab_entry is not None:
                            vocab_value = int(vocab_entry)
                except Exception:
                    vocab_value = None
        if vocab_value is None:
            vocab_value = _estimate_sequence_vocab(videos, key)
        vocab_map[key] = max(1, vocab_value)
    return vocab_map


def _tokens_with_offset(
    tokens: Sequence[str | int],
    *,
    token_offset: int,
    truncate_len: int,
) -> list[int]:
    if truncate_len > 0:
        tokens = tokens[:truncate_len]
    return [int(tok) + token_offset for tok in tokens]


def _resolve_root(path: Path, root_override: Path | None) -> Path:
    if root_override is None:
        return path.parent.resolve()
    candidate = Path(root_override).expanduser()
    if not candidate.is_absolute():
        candidate = (path.parent / candidate).resolve()
    return candidate


def _resolve_path(value: str | None, root: Path) -> str:
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return str(candidate)


def _extract_entropy_map(payload: Any, sequence_key: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    resolved: dict[str, float] = {}
    for metric, mapping in payload.items():
        if not isinstance(mapping, dict):
            continue
        value = mapping.get(sequence_key)
        if value is None:
            continue
        try:
            resolved[metric] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def _extract_scalar(payload: Any, sequence_key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(sequence_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_scalar_for_keys(payload: Any, sequence_keys: Sequence[str]) -> float | None:
    for key in sequence_keys:
        value = _extract_scalar(payload, key)
        if value is not None:
            return value
    return None


def _candidate_length_for_keys(candidate: Any, sequence_keys: Sequence[str]) -> float | None:
    if not candidate:
        return None
    sequences = candidate.get("sequences") if isinstance(candidate, dict) else None
    if sequences:
        for key in sequence_keys:
            payload = sequences.get(key)
            if payload:
                return float(len(payload))
    seq = candidate.get("sequence") if isinstance(candidate, dict) else None
    if isinstance(seq, (list, tuple)):
        return float(len(seq))
    return None


def _normalize_question(question: dict[str, Any], root: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(question))
    options = []
    for option in payload.get("options", []) or []:
        opt = dict(option)
        clip_path = opt.get("clip_path")
        if clip_path:
            opt["clip_path"] = _resolve_path(clip_path, root)
        options.append(opt)
    payload["options"] = options
    clip_path = payload.get("clip_path")
    if clip_path:
        payload["clip_path"] = _resolve_path(clip_path, root)
    return payload


class BucketManifestSource:
    """Parser for the canonical PatternVideos bucket manifest schema."""

    def __init__(
        self,
        path: Path,
        *,
        token_offset: int,
        num_videos: int,
        truncate_len: int,
        sequence_keys: Sequence[str] | None,
        root_path: Path | None = None,
        include_questions: bool = False,
        max_stream_len: int = 0,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        manifest, videos = _load_manifest_videos(path, num_videos)
        root = _resolve_root(path, root_path)
        resolved_keys = _resolve_sequence_key_list(videos, sequence_keys)
        sequence_label = "+".join(resolved_keys)
        sequence_vocab = _load_sequence_vocab(manifest, path, resolved_keys, videos)
        # All manifest-provided streams already share the same base vocabulary; reapply the
        # dataset token offset uniformly so imported data lines up with the synthetic builder.
        sequence_offsets: dict[str, int] = {key: token_offset for key in resolved_keys}
        entropy_empirical: list[float] = []
        entropy_analytic: list[float] = []
        entropy_primary: list[float] = []
        entropy_prefix_values: list[float] = []
        prefix_lengths: list[float] = []
        target_lengths: list[float] = []
        stream_lengths: list[float] = []
        question_times: list[float] = []
        clip_durations: list[float] = []
        question_counts: list[int] = []
        scenario_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        records: list[SequenceRecord] = []
        skipped_over_limit = 0
        for video in videos:
            sequences = video.get("sequences_used") or {}
            if not sequences:
                continue
            primary_key = resolved_keys[0]
            missing = [key for key in resolved_keys if key not in sequences]
            if missing:
                warnings.warn(
                    f"Skipping video_index={video.get('video_index')} because sequences missing: {missing}",
                    UserWarning,
                )
                continue
            seq_tokens: list[int] = []
            per_key_lengths: dict[str, int] = {}
            for key in resolved_keys:
                payload = sequences.get(key)
                if payload is None:
                    continue
                chunk = _tokens_with_offset(
                    payload,
                    token_offset=sequence_offsets[key],
                    truncate_len=truncate_len,
                )
                seq_tokens.extend(chunk)
                per_key_lengths[key] = len(chunk)
            if not seq_tokens:
                continue
            if max_stream_len > 0 and len(seq_tokens) > max_stream_len:
                skipped_over_limit += 1
                continue
            stream_lengths.append(float(len(seq_tokens)))
            entropy_payload = video.get("entropy_overall")
            entropy_overall = _extract_entropy_map(entropy_payload, primary_key)
            empirical_bits = entropy_overall.get("empirical_bits")
            analytic_bits = entropy_overall.get("analytic_bits")
            if empirical_bits is not None:
                entropy_empirical.append(empirical_bits)
            if analytic_bits is not None:
                entropy_analytic.append(analytic_bits)
            primary_entropy = analytic_bits if analytic_bits is not None else empirical_bits
            if primary_entropy is not None:
                entropy_primary.append(primary_entropy)
            questions = list(video.get("questions", []))
            question_counts.append(len(questions))
            metadata = {
                "video_index": video.get("video_index"),
                "variant": video.get("variant"),
                "video_path": _resolve_path(video.get("video_path"), root),
                "sequence_key": resolved_keys[0] if resolved_keys else None,
                "sequence_keys": list(resolved_keys),
                "sequence_offsets": dict(sequence_offsets),
                "sequence_vocab": dict(sequence_vocab),
                "sequence_token_lengths": dict(per_key_lengths) or None,
                "entropy_overall": entropy_payload or entropy_overall or None,
                "questions_at_end": bool(video.get("questions_at_end")),
            }
            bucket_id = video.get("bucket_id")
            if bucket_id is not None:
                metadata["bucket_id"] = bucket_id
            bucket_from = video.get("bucket_from")
            if bucket_from:
                metadata["bucket_from"] = bucket_from
            prepared_questions = []
            for question in questions:
                scenario = question.get("scenario")
                if scenario:
                    scenario_counts[str(scenario)] += 1
                entropy_prefix = _extract_scalar_for_keys(question.get("entropy_prefix"), resolved_keys)
                if entropy_prefix is not None:
                    entropy_prefix_values.append(entropy_prefix)
                candidate_len = _candidate_length_for_keys(question.get("candidate"), resolved_keys)
                if candidate_len is not None:
                    target_lengths.append(candidate_len)
                q_time = question.get("question_time")
                if q_time is not None:
                    question_times.append(float(q_time))
                clip_start = question.get("clip_start_time")
                clip_end = question.get("clip_end_time")
                if clip_start is not None and clip_end is not None:
                    try:
                        clip_durations.append(float(clip_end) - float(clip_start))
                    except (TypeError, ValueError):
                        pass
                mode = str(question.get("question_mode") or "exists").lower()
                mode_counts[mode] += 1
                prefix = question.get("prefix") or {}
                prefix_total = 0
                has_prefix = False
                for key in resolved_keys:
                    tokens = prefix.get(key)
                    if isinstance(tokens, (list, tuple)):
                        prefix_total += len(tokens)
                        has_prefix = True
                if has_prefix:
                    prefix_lengths.append(float(prefix_total))
                if include_questions:
                    prepared_questions.append(_normalize_question(question, root))
            if include_questions:
                metadata["questions"] = prepared_questions
            records.append(SequenceRecord(tokens=seq_tokens, metadata=metadata))
        resolved_path = str(path.resolve())
        self._records = records
        self._summary = {
            "source": "bucket_manifest",
            "manifest_path": resolved_path,
            "root_path": str(root),
            "video_count": len(records),
            "sequence_key": resolved_keys[0] if resolved_keys else None,
            "sequence_keys": list(resolved_keys),
            "sequence_vocab": dict(sequence_vocab),
            "sequence_offsets": dict(sequence_offsets),
            "sequence_label": sequence_label,
            "question_count": sum(question_counts),
            "questions_per_video": _stat_summary(question_counts),
            "entropy_overall": {
                "selected_bits": _stat_summary(entropy_primary),
                "empirical_bits": _stat_summary(entropy_empirical),
                "analytic_bits": _stat_summary(entropy_analytic),
            },
            "entropy_prefix": _stat_summary(entropy_prefix_values),
            "target_length": _stat_summary(target_lengths),
            "question_time": _stat_summary(question_times),
            "clip_duration": _stat_summary(clip_durations),
            "stream_length": _stat_summary(stream_lengths),
            "prefix_length": _stat_summary(prefix_lengths),
            "scenario_counts": dict(scenario_counts),
            "question_mode_counts": dict(mode_counts),
            "entropy_prefix_tertiles": _compute_tertiles(entropy_prefix_values),
            "target_length_tertiles": _compute_tertiles(target_lengths),
            "prefix_length_tertiles": _compute_tertiles(prefix_lengths),
            "stream_length_tertiles": _compute_tertiles(stream_lengths),
            "video_entropy_tertiles": _compute_tertiles(entropy_primary),
            "skipped_streams_over_limit": skipped_over_limit,
        }

    @property
    def records(self) -> list[SequenceRecord]:
        return list(self._records)

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._summary)
