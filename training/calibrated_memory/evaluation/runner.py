"""Evaluation runner and statistics aggregator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader

from calibrated_memory.data.sequences.common import (
    IGNORE_INDEX,
    YES_TOKEN,
    NO_TOKEN,
    UNCERTAIN_TOKEN,
    LABEL_TOKENS,
)
from calibrated_memory.metrics.video import VideoBucketTracker, sanitize_boundaries


@dataclass
class QuestionResult:
    metadata: dict[str, Any]
    truth_token: int
    pred_token: int
    truth_prob: float
    pred_prob: float
    logits: List[float]
    correct: bool


class EvaluationRunner:
    """Feeds evaluation batches through the decoder and records predictions."""

    def __init__(
        self,
        model: Any,
        *,
        device: torch.device,
        task: str,
    ) -> None:
        self.model = model
        self.device = device
        self.task = task

    def run(self, dataloader: DataLoader) -> list[QuestionResult]:
        results: list[QuestionResult] = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                metadata = batch.get("metadata") or [{} for _ in range(batch["sequence"]["input_ids"].size(0))]
                _move_batch_to_device(batch, self.device)
                logits, labels = self._forward(batch)
                preds = logits.argmax(dim=-1)
                mask = labels != IGNORE_INDEX
                for idx in range(labels.size(0)):
                    entry_metadata = dict(metadata[idx] or {})
                    for slot in range(labels.size(1)):
                        truth_index = int(labels[idx, slot].item())
                        if truth_index == IGNORE_INDEX:
                            continue
                        pred_index = int(preds[idx, slot].item())
                        logits_slice = logits[idx, slot]
                        probs = torch.softmax(logits_slice, dim=-1)
                        truth_prob = (
                            float(probs[truth_index].item()) if truth_index < probs.size(0) else 0.0
                        )
                        pred_prob = float(probs[pred_index].item())
                        results.append(
                            QuestionResult(
                                metadata=entry_metadata,
                                truth_token=_class_index_to_token(truth_index),
                                pred_token=_class_index_to_token(pred_index),
                                truth_prob=truth_prob,
                                pred_prob=pred_prob,
                                logits=logits_slice.tolist(),
                                correct=truth_index == pred_index,
                            )
                        )
        return results

    def _forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = batch.get("metadata") or [{} for _ in range(batch["sequence"]["input_ids"].size(0))]
        logits, labels = self.model.compute_sequence_logits(  # type: ignore[attr-defined]
            batch["sequence"],
            batch["labels"],
            metadata,
        )
        return logits, labels


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> None:
    sequence = batch.get("sequence")
    if isinstance(sequence, dict):
        for key, tensor in sequence.items():
            if isinstance(tensor, torch.Tensor):
                sequence[key] = tensor.to(device)
    labels = batch.get("labels")
    if isinstance(labels, torch.Tensor):
        batch["labels"] = labels.to(device)


def describe_label(token: int) -> str:
    if token == YES_TOKEN:
        return "yes"
    if token == NO_TOKEN:
        return "no"
    if token == UNCERTAIN_TOKEN:
        return "uncertain"
    return str(token)


def _class_index_to_token(index: int) -> int:
    if 0 <= index < len(LABEL_TOKENS):
        return LABEL_TOKENS[index]
    return index


@dataclass(frozen=True)
class _QuestionStats:
    length: float | None
    entropy: float | None
    bucket_id: str | None
    correct: bool
    answered: bool
    answered_correct: bool
    pred_abstain: bool
    truth_abstain: bool


class StatsAggregator:
    """Aggregates correctness, coverage, and uncertainty metrics."""

    _CONSISTENCY_EPS = 1e-6

    def __init__(
        self,
        *,
        task: str,
        entropy_boundaries: list[float] | None = None,
        length_boundaries: list[float] | None = None,
        compression_thresholds: Sequence[float] | None = None,
    ) -> None:
        self.task = task
        self.records: list[QuestionResult] = []
        self._video_totals: dict[tuple[str, Any], dict[str, Any]] = {}
        self._tracker = VideoBucketTracker()
        self._entropy_boundaries = sanitize_boundaries(entropy_boundaries)
        self._length_boundaries = sanitize_boundaries(length_boundaries)
        if compression_thresholds:
            self._compression_thresholds = sorted({float(x) for x in compression_thresholds})
        else:
            self._compression_thresholds = [0.5, 0.7, 0.8, 0.9]
        self._prefix_consistency: dict[tuple[str, Any], tuple[float | None, float | None]] = {}
        self._prefix_mismatch_logged: set[tuple[str, Any]] = set()
        self._prefix_mismatch_budget = 10

    def add(self, result: QuestionResult) -> None:
        self.records.append(result)
        metadata = result.metadata or {}
        video_meta = metadata.get("video")
        key = _video_identifier(video_meta)
        entry = self._video_totals.get(key)
        if entry is None:
            entry = {
                "length": _video_length_value(video_meta, metadata.get("stream_total_length")),
                "entropy": _video_entropy_value(video_meta),
                "correct": 0,
                "total": 0,
                "uncertain": 0,
            }
            self._video_totals[key] = entry
        if result.correct:
            entry["correct"] += 1
        entry["total"] += 1
        if describe_label(result.pred_token) == "uncertain":
            entry["uncertain"] += 1
        prefix_length = _coerce_float(metadata.get("stream_prefix_length"))
        prefix_entropy = _coerce_float(metadata.get("entropy_prefix"))
        self._verify_prefix_consistency(key, prefix_length, prefix_entropy)

    def finalize(self) -> dict[str, Any]:
        question_stats = self._build_question_stats()
        total = len(question_stats)
        correct = sum(1 for stats in question_stats if stats.correct)
        answered = sum(1 for stats in question_stats if stats.answered)
        answered_correct = sum(1 for stats in question_stats if stats.answered_correct)
        truth_uncertain = sum(1 for stats in question_stats if stats.truth_abstain)
        pred_uncertain = sum(1 for stats in question_stats if stats.pred_abstain)
        uncertain_miss = sum(
            1 for stats in question_stats if stats.truth_abstain and stats.answered
        )
        false_uncertain = sum(
            1 for stats in question_stats if (not stats.truth_abstain) and stats.pred_abstain
        )
        for entry in self._video_totals.values():
            self._tracker.add_record(
                length_value=entry["length"],
                entropy_value=entry["entropy"],
                correct=entry["correct"],
                total=entry["total"],
                uncertain=entry["uncertain"],
            )
        entropy_summary = []
        length_summary = []
        if self._entropy_boundaries:
            entropy_summary = self._tracker.summary("entropy", self._entropy_boundaries)
        else:
            bounds = self._tracker.compute_tertiles("entropy")
            if bounds:
                self._entropy_boundaries = bounds
                entropy_summary = self._tracker.summary("entropy", bounds)
        if self._length_boundaries:
            length_summary = self._tracker.summary("length", self._length_boundaries)
        else:
            bounds = self._tracker.compute_tertiles("length")
            if bounds:
                self._length_boundaries = bounds
                length_summary = self._tracker.summary("length", bounds)
        overall_cov = _safe_ratio(answered, total)
        accuracy_when_answering = _safe_ratio(answered_correct, answered)
        ua_rate = _safe_ratio(answered_correct, total)
        tar = _safe_ratio(
            sum(1 for stats in question_stats if stats.truth_abstain and stats.pred_abstain),
            truth_uncertain,
        )
        answerable_total = total - truth_uncertain
        far = _safe_ratio(
            sum(1 for stats in question_stats if (not stats.truth_abstain) and stats.pred_abstain),
            answerable_total,
        )
        (
            length_bucket_metrics,
            entropy_bucket_metrics,
            joint_bucket_metrics,
            joint_matrix,
            length_edges,
            entropy_edges,
            missing_lengths,
            missing_entropies,
        ) = self._question_bucket_metrics(question_stats)
        compression_summary = self._compression_summary(
            entropy_edges,
            length_edges,
            joint_matrix,
        )
        return {
            "total_questions": total,
            "accuracy": correct / total if total else 0.0,
            "coverage": overall_cov,
            "accuracy_when_answering": accuracy_when_answering,
            "useful_answer_rate": ua_rate,
            "tar": tar,
            "far": far,
            "truth_uncertain": truth_uncertain,
            "pred_uncertain": pred_uncertain,
            "uncertain_miss_rate": (uncertain_miss / truth_uncertain) if truth_uncertain else None,
            "false_uncertain_rate": (
                false_uncertain / max(1, total - truth_uncertain)
                if total - truth_uncertain > 0
                else None
            ),
            "entropy_buckets": entropy_summary,
            "length_buckets": length_summary,
            "question_bucket_metrics": {
                "entropy": entropy_bucket_metrics,
                "length": length_bucket_metrics,
                "joint": joint_bucket_metrics,
                "missing_length_count": missing_lengths,
                "missing_entropy_count": missing_entropies,
            },
            "compression_summary": compression_summary,
        }
    def _build_question_stats(self) -> list[_QuestionStats]:
        stats: list[_QuestionStats] = []
        for record in self.records:
            metadata = record.metadata or {}
            video_meta = metadata.get("video")
            length_value = _resolve_prefix_length(metadata)
            if length_value is None:
                length_value = _video_length_value(video_meta, metadata.get("stream_total_length"))
            entropy_value = _resolve_prefix_entropy(metadata)
            if entropy_value is None:
                entropy_value = _video_entropy_value(video_meta)
            bucket_id = metadata.get("bucket_id")
            if bucket_id is None and isinstance(video_meta, dict):
                bucket_id = video_meta.get("bucket_id")
            truth_kind = describe_label(record.truth_token)
            pred_kind = describe_label(record.pred_token)
            pred_abstain = pred_kind == "uncertain"
            truth_abstain = truth_kind == "uncertain"
            answered = not pred_abstain
            stats.append(
                _QuestionStats(
                    length=length_value,
                    entropy=entropy_value,
                    bucket_id=str(bucket_id) if bucket_id is not None else None,
                    correct=record.correct,
                    answered=answered,
                    answered_correct=bool(record.correct and answered),
                    pred_abstain=pred_abstain,
                    truth_abstain=truth_abstain,
                )
            )
        return stats

    def _verify_prefix_consistency(
        self,
        key: tuple[str, Any],
        prefix_length: float | None,
        prefix_entropy: float | None,
    ) -> None:
        if prefix_length is None and prefix_entropy is None:
            return
        stored = self._prefix_consistency.get(key)
        if stored is None:
            self._prefix_consistency[key] = (prefix_length, prefix_entropy)
            return
        expected_length, expected_entropy = stored
        inconsistent_length = (
            expected_length is not None
            and prefix_length is not None
            and not _floats_close(prefix_length, expected_length, self._CONSISTENCY_EPS)
        )
        inconsistent_entropy = (
            expected_entropy is not None
            and prefix_entropy is not None
            and not _floats_close(prefix_entropy, expected_entropy, self._CONSISTENCY_EPS)
        )
        if inconsistent_length or inconsistent_entropy:
            if (
                key not in self._prefix_mismatch_logged
                and self._prefix_mismatch_budget > 0
            ):
                print(
                    "[eval] Warning: prefix metadata varied within video bucket",
                    key,
                    "(length delta="
                    + ("yes" if inconsistent_length else "no")
                    + ", entropy delta="
                    + ("yes" if inconsistent_entropy else "no")
                    + "); treating prefix stats as unknown.",
                    flush=True,
                )
                self._prefix_mismatch_budget -= 1
                if self._prefix_mismatch_budget == 0:
                    print(
                        "[eval] Additional prefix mismatch warnings suppressed.",
                        flush=True,
                    )
            self._prefix_mismatch_logged.add(key)
            self._prefix_consistency[key] = (
                None if inconsistent_length else expected_length,
                None if inconsistent_entropy else expected_entropy,
            )
            return
        if expected_length is None and prefix_length is not None:
            self._prefix_consistency[key] = (prefix_length, expected_entropy)
        if expected_entropy is None and prefix_entropy is not None:
            self._prefix_consistency[key] = (expected_length, prefix_entropy)

    def _question_bucket_metrics(
        self,
        question_stats: Sequence[_QuestionStats],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[list[dict[str, Any]]] | None,
        list[float],
        list[float],
        int,
        int,
    ]:
        length_values = [stats.length for stats in question_stats if stats.length is not None]
        entropy_values = [stats.entropy for stats in question_stats if stats.entropy is not None]
        length_edges = self._resolve_boundaries(self._length_boundaries, length_values)
        entropy_edges = self._resolve_boundaries(self._entropy_boundaries, entropy_values)
        length_bucket_metrics: list[dict[str, Any]] = []
        entropy_bucket_metrics: list[dict[str, Any]] = []
        joint_bucket_metrics: list[dict[str, Any]] = []
        missing_length = len(question_stats) - len(length_values)
        missing_entropy = len(question_stats) - len(entropy_values)
        length_bucket_count = len(length_edges) + 1 if length_values else 0
        entropy_bucket_count = len(entropy_edges) + 1 if entropy_values else 0
        length_tables = [_empty_bucket_state() for _ in range(length_bucket_count)]
        entropy_tables = [_empty_bucket_state() for _ in range(entropy_bucket_count)]
        joint_matrix = (
            [
                [_empty_bucket_state(include_bucket_ids=True) for _ in range(length_bucket_count)]
                for _ in range(entropy_bucket_count)
            ]
            if length_bucket_count and entropy_bucket_count
            else None
        )
        for stats in question_stats:
            if stats.length is not None and length_tables:
                idx = _bucket_index(stats.length, length_edges)
                _accumulate_bucket_state(length_tables[idx], stats, stats.length)
            if stats.entropy is not None and entropy_tables:
                idx = _bucket_index(stats.entropy, entropy_edges)
                _accumulate_bucket_state(entropy_tables[idx], stats)
            if (
                stats.length is not None
                and stats.entropy is not None
                and joint_matrix is not None
            ):
                e_idx = _bucket_index(stats.entropy, entropy_edges)
                l_idx = _bucket_index(stats.length, length_edges)
                joint_state = joint_matrix[e_idx][l_idx]
                _accumulate_bucket_state(joint_state, stats, stats.length)
                if stats.bucket_id is not None:
                    joint_state.setdefault("bucket_ids", set()).add(stats.bucket_id)
        if length_tables:
            length_bucket_metrics = _serialize_axis_buckets(length_tables, length_edges, axis="length")
        if entropy_tables:
            entropy_bucket_metrics = _serialize_axis_buckets(
                entropy_tables,
                entropy_edges,
                axis="entropy",
            )
        if joint_matrix is not None:
            joint_bucket_metrics = _serialize_joint_buckets(joint_matrix, length_edges, entropy_edges)
        return (
            length_bucket_metrics,
            entropy_bucket_metrics,
            joint_bucket_metrics,
            joint_matrix,
            length_edges,
            entropy_edges,
            missing_length,
            missing_entropy,
        )

    def _compression_summary(
        self,
        entropy_edges: list[float],
        length_edges: list[float],
        joint_matrix: list[list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        summary = {
            "alphas": self._compression_thresholds,
            "entropy_buckets": [],
        }
        if joint_matrix is None or not joint_matrix:
            return summary
        length_bucket_count = len(joint_matrix[0])
        entropy_bucket_count = len(joint_matrix)
        for entropy_idx in range(entropy_bucket_count):
            lower, upper = _bucket_bounds(entropy_idx, entropy_edges)
            bucket_label = _format_bucket_label(lower, upper)
            entries: list[tuple[float | None, dict[str, Any]]] = []
            for length_idx in range(length_bucket_count):
                state = joint_matrix[entropy_idx][length_idx]
                if not state["question_total"]:
                    continue
                length_value = state.get("max_length")
                if length_value is None:
                    bounds = _bucket_bounds(length_idx, length_edges)
                    length_value = bounds[1]
                clean_state = dict(state)
                bucket_ids = clean_state.get("bucket_ids")
                if isinstance(bucket_ids, set):
                    clean_state["bucket_ids"] = sorted(bucket_ids)
                entries.append((length_value, clean_state))
            summary["entropy_buckets"].append(
                {
                    "label": bucket_label,
                    "length_entries": entries,
                }
            )
        return summary

    def _resolve_boundaries(
        self,
        cached: list[float] | None,
        values: Sequence[float],
    ) -> list[float]:
        if cached:
            return list(cached)
        if not values:
            return []
        tracker = VideoBucketTracker()
        for value in values:
            tracker.add_record(length_value=value, entropy_value=None, correct=0, total=0, uncertain=0)
        tertiles = tracker.compute_tertiles("length")
        return tertiles or []


def _video_identifier(video_meta: dict[str, Any] | None) -> tuple[str, Any]:
    if not isinstance(video_meta, dict):
        return ("unknown", id(video_meta))
    for key in ("video_index", "video_id", "video_path"):
        value = video_meta.get(key)
        if value is not None:
            return (key, value)
    return ("unknown", id(video_meta))


def _video_length_value(meta: dict[str, Any] | None, fallback: Any) -> float | None:
    if isinstance(meta, dict):
        value = meta.get("length_value")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        stream_len = meta.get("stream_length")
        if stream_len is not None:
            try:
                return float(stream_len)
            except (TypeError, ValueError):
                pass
    if fallback is not None:
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return None
    return None


def _video_entropy_value(meta: dict[str, Any] | None) -> float | None:
    if not isinstance(meta, dict):
        return None
    value = meta.get("entropy_value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_prefix_length(metadata: dict[str, Any]) -> float | None:
    if not metadata:
        return None
    if "stream_prefix_length" in metadata:
        return _coerce_float(metadata.get("stream_prefix_length"))
    return _coerce_float(metadata.get("prefix_length"))


def _resolve_prefix_entropy(metadata: dict[str, Any]) -> float | None:
    if not metadata:
        return None
    if "entropy_prefix" in metadata:
        return _coerce_float(metadata.get("entropy_prefix"))
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _floats_close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def _bucket_index(value: float, boundaries: Sequence[float]) -> int:
    idx = 0
    limit = len(boundaries)
    while idx < limit and value > boundaries[idx]:
        idx += 1
    return idx


def _bucket_bounds(index: int, boundaries: list[float] | None) -> tuple[float, float]:
    edges = list(boundaries or [])
    lower = edges[index - 1] if index > 0 and index - 1 < len(edges) else float("-inf")
    upper = edges[index] if index < len(edges) else float("inf")
    return lower, upper


def _format_bucket_label(lower: float, upper: float) -> str:
    def _fmt(value: float) -> str:
        if math.isinf(value):
            return "-inf" if value < 0 else "+inf"
        return f"{value:.2f}"

    return f"[{_fmt(lower)}, {_fmt(upper)})"


def _empty_bucket_state(include_bucket_ids: bool = False) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question_total": 0,
        "correct": 0,
        "answered": 0,
        "answered_correct": 0,
        "pred_abstain": 0,
        "truth_abstain": 0,
        "max_length": None,
    }
    if include_bucket_ids:
        state["bucket_ids"] = set()
    return state


def _accumulate_bucket_state(state: dict[str, Any], stats: _QuestionStats, length: float | None = None) -> None:
    state["question_total"] += 1
    if stats.correct:
        state["correct"] += 1
    if stats.answered:
        state["answered"] += 1
    if stats.answered_correct:
        state["answered_correct"] += 1
    if stats.pred_abstain:
        state["pred_abstain"] += 1
    if stats.truth_abstain:
        state["truth_abstain"] += 1
    if length is not None:
        current = state.get("max_length")
        if current is None or length > current:
            state["max_length"] = float(length)


def _serialize_axis_buckets(tables: list[dict[str, Any]], edges: list[float], *, axis: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for idx, state in enumerate(tables):
        lower, upper = _bucket_bounds(idx, edges)
        metrics.append(
            {
                "axis": axis,
                "bucket": _format_bucket_label(lower, upper),
                "question_total": state["question_total"],
                "accuracy": _safe_ratio(state["correct"], state["question_total"]),
                "coverage": _safe_ratio(state["answered"], state["question_total"]),
                "abstain_rate": _safe_ratio(state["pred_abstain"], state["question_total"]),
            }
        )
    return metrics


def _serialize_joint_buckets(
    matrix: list[list[dict[str, Any]]],
    length_edges: list[float],
    entropy_edges: list[float],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for entropy_idx, row in enumerate(matrix):
        entropy_bounds = _bucket_bounds(entropy_idx, entropy_edges)
        entropy_label = _format_bucket_label(*entropy_bounds)
        for length_idx, state in enumerate(row):
            if not state["question_total"]:
                continue
            length_bounds = _bucket_bounds(length_idx, length_edges)
            length_label = _format_bucket_label(*length_bounds)
            metrics.append(
                {
                    "entropy_bucket": entropy_label,
                    "length_bucket": length_label,
                    "question_total": state["question_total"],
                    "accuracy": _safe_ratio(state["correct"], state["question_total"]),
                    "coverage": _safe_ratio(state["answered"], state["question_total"]),
                    "abstain_rate": _safe_ratio(state["pred_abstain"], state["question_total"]),
                    "bucket_ids": sorted(state.get("bucket_ids", set())),
                }
            )
    return metrics


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
