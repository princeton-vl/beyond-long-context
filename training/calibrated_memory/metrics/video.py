"""Utilities for aggregating per-video validation statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class _VideoRecord:
    length_value: float | None
    entropy_value: float | None
    correct: int
    total: int
    uncertain: int
    uncertain_truth_total: int
    uncertain_truth_errors: int
    option_truth_total: int
    option_truth_uncertain: int


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    if lower == upper:
        return float(sorted_values[lower])
    lower_val = float(sorted_values[lower])
    upper_val = float(sorted_values[upper])
    weight = position - lower
    return lower_val + (upper_val - lower_val) * weight


def sanitize_boundaries(bounds: Iterable[float] | None) -> list[float] | None:
    if not bounds:
        return None
    items = [float(value) for value in bounds if value is not None]
    if len(items) < 2:
        return None
    items.sort()
    return items[:2]


class VideoBucketTracker:
    """Collects per-video accuracy stats and projects them into tertile buckets."""

    def __init__(self) -> None:
        self._records: list[_VideoRecord] = []

    def reset(self) -> None:
        self._records.clear()

    def has_data(self) -> bool:
        return bool(self._records)

    def add_record(
        self,
        *,
        length_value: float | None,
        entropy_value: float | None,
        correct: int,
        total: int,
        uncertain: int,
        uncertain_truth_total: int = 0,
        uncertain_truth_errors: int = 0,
        option_truth_total: int = 0,
        option_truth_uncertain: int = 0,
    ) -> None:
        if total <= 0:
            return
        self._records.append(
            _VideoRecord(
                length_value=float(length_value) if length_value is not None else None,
                entropy_value=float(entropy_value) if entropy_value is not None else None,
                correct=int(correct),
                total=int(total),
                uncertain=int(uncertain),
                uncertain_truth_total=int(max(0, uncertain_truth_total)),
                uncertain_truth_errors=int(max(0, uncertain_truth_errors)),
                option_truth_total=int(max(0, option_truth_total)),
                option_truth_uncertain=int(max(0, option_truth_uncertain)),
            )
        )

    def compute_tertiles(self, metric: str) -> list[float] | None:
        values = [self._metric_value(record, metric) for record in self._records]
        filtered = [value for value in values if value is not None]
        if len(filtered) < 3:
            return None
        filtered.sort()
        return [
            _quantile(filtered, 1.0 / 3.0),
            _quantile(filtered, 2.0 / 3.0),
        ]

    def summary(self, metric: str, boundaries: Iterable[float]) -> list[dict[str, float]]:
        edges = list(boundaries)
        if not edges:
            return []
        bucket_count = len(edges) + 1
        question_totals = [0] * bucket_count
        correct_totals = [0] * bucket_count
        uncertain_totals = [0] * bucket_count
        video_totals = [0] * bucket_count
        uncertain_truth_totals = [0] * bucket_count
        uncertain_truth_errors = [0] * bucket_count
        option_truth_totals = [0] * bucket_count
        option_truth_uncertain = [0] * bucket_count
        for record in self._records:
            value = self._metric_value(record, metric)
            if value is None:
                continue
            bucket_idx = self._bucket_index(value, edges)
            question_totals[bucket_idx] += record.total
            correct_totals[bucket_idx] += record.correct
            uncertain_totals[bucket_idx] += record.uncertain
            video_totals[bucket_idx] += 1
            uncertain_truth_totals[bucket_idx] += record.uncertain_truth_total
            uncertain_truth_errors[bucket_idx] += record.uncertain_truth_errors
            option_truth_totals[bucket_idx] += record.option_truth_total
            option_truth_uncertain[bucket_idx] += record.option_truth_uncertain
        rows: list[dict[str, float]] = []
        lower = float("-inf")
        for idx in range(bucket_count):
            upper = edges[idx] if idx < len(edges) else float("inf")
            total = question_totals[idx]
            accuracy = correct_totals[idx] / total if total else 0.0
            uncertain_pct = uncertain_totals[idx] / total if total else 0.0
            uncertain_truth_total = uncertain_truth_totals[idx]
            uncertain_truth_error_pct = (
                uncertain_truth_errors[idx] / uncertain_truth_total if uncertain_truth_total else 0.0
            )
            option_truth_total = option_truth_totals[idx]
            option_truth_uncertain_pct = (
                option_truth_uncertain[idx] / option_truth_total if option_truth_total else 0.0
            )
            rows.append(
                {
                    "bucket": f"({lower:.2f},{upper:.2f}]",
                    "accuracy": accuracy,
                    "question_count": float(total),
                    "video_count": float(video_totals[idx]),
                    "uncertain_pct": uncertain_pct,
                    "uncertain_truth_error_pct": uncertain_truth_error_pct,
                    "option_truth_uncertain_pct": option_truth_uncertain_pct,
                }
            )
            lower = upper
        return rows

    @staticmethod
    def _bucket_index(value: float, boundaries: List[float]) -> int:
        idx = 0
        while idx < len(boundaries) and value > boundaries[idx]:
            idx += 1
        return idx

    @staticmethod
    def _metric_value(record: _VideoRecord, metric: str) -> float | None:
        if metric == "length":
            return record.length_value
        if metric == "entropy":
            return record.entropy_value
        raise ValueError(f"Unknown metric '{metric}'")
