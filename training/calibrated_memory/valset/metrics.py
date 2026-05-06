"""Metrics helpers for validation analysis."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class QuestionEval:
    """Flattened evaluation result for a single question."""

    bucket_id: str
    stream_length: int
    concerned_ranges: Sequence[tuple[int, int]]
    truth_kind: str
    pred_kind: str
    correct: bool


@dataclass(frozen=True)
class BucketMetric:
    bucket_id: str
    question_count: int
    accuracy: float
    coverage: float
    abstention_rate: float
    uncertain_truth_error_pct: float
    option_abstain_pct: float


@dataclass(frozen=True)
class BucketEighthStats:
    bucket_id: str
    eighth_index: int
    question_count: int
    accuracy: float
    coverage: float
    abstention_rate: float
    uncertain_truth_error_pct: float
    option_abstain_pct: float


@dataclass(frozen=True)
class EvaluationMetrics:
    bucket_metrics: list[BucketMetric]
    bucket_eighth_metrics: list[BucketEighthStats]


def build_bucket_metrics(records: Iterable[QuestionEval]) -> list[BucketMetric]:
    buckets: dict[str, list[QuestionEval]] = {}
    for record in records:
        buckets.setdefault(record.bucket_id, []).append(record)
    metrics: list[BucketMetric] = []
    for bucket_id, bucket_records in sorted(buckets.items()):
        metrics.append(_aggregate_records(bucket_id, bucket_records))
    return metrics


def build_bucket_eighth_metrics(records: Iterable[QuestionEval]) -> list[BucketEighthStats]:
    aggregate: dict[tuple[str, int], list[QuestionEval]] = {}
    for record in records:
        indices = _eighth_indices(record.stream_length, record.concerned_ranges)
        if not indices:
            continue
        for idx in indices:
            aggregate.setdefault((record.bucket_id, idx), []).append(record)
    stats: list[BucketEighthStats] = []
    for (bucket_id, eighth), rows in sorted(aggregate.items()):
        summary = _aggregate_records(bucket_id, rows)
        stats.append(
            BucketEighthStats(
                bucket_id=bucket_id,
                eighth_index=eighth,
                question_count=summary.question_count,
                accuracy=summary.accuracy,
                coverage=summary.coverage,
                abstention_rate=summary.abstention_rate,
                uncertain_truth_error_pct=summary.uncertain_truth_error_pct,
                option_abstain_pct=summary.option_abstain_pct,
            )
        )
    return stats


def _aggregate_records(bucket_id: str, records: Sequence[QuestionEval]) -> BucketMetric:
    total = len(records)
    if total == 0:
        return BucketMetric(bucket_id, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    correct = sum(1 for record in records if record.correct)
    pred_uncertain = sum(1 for record in records if record.pred_kind == "uncertain")
    answered = total - pred_uncertain
    truth_uncertain = sum(1 for record in records if record.truth_kind == "uncertain")
    answerable = total - truth_uncertain
    uncertain_errors = sum(
        1
        for record in records
        if record.truth_kind == "uncertain" and record.pred_kind != "uncertain"
    )
    option_abstain = sum(
        1
        for record in records
        if record.truth_kind != "uncertain" and record.pred_kind == "uncertain"
    )
    accuracy = correct / total if total else 0.0
    coverage = answered / total if total else 0.0
    abstention_rate = pred_uncertain / total if total else 0.0
    uncertain_pct = (uncertain_errors / truth_uncertain) if truth_uncertain else 0.0
    option_abstain_pct = (option_abstain / answerable) if answerable else 0.0
    return BucketMetric(
        bucket_id=bucket_id,
        question_count=total,
        accuracy=accuracy,
        coverage=coverage,
        abstention_rate=abstention_rate,
        uncertain_truth_error_pct=uncertain_pct,
        option_abstain_pct=option_abstain_pct,
    )


def _eighth_indices(
    stream_length: int,
    ranges: Sequence[tuple[int, int]],
) -> list[int]:
    if stream_length <= 0 or not ranges:
        return []
    segments: list[tuple[int, int]] = []
    for idx in range(8):
        start = math.floor(idx * stream_length / 8)
        end = math.floor((idx + 1) * stream_length / 8)
        if idx == 7:
            end = stream_length
        segments.append((start, max(end, start + 1)))
    indices: set[int] = set()
    for start, end in ranges:
        for idx, (seg_start, seg_end) in enumerate(segments):
            if _intervals_overlap(start, end, seg_start, seg_end):
                indices.add(idx)
    return sorted(indices)


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def write_bucket_csv(path: Path, rows: Sequence[BucketMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bucket",
                "question_count",
                "accuracy",
                "coverage",
                "abstention_rate",
                "uncertain_truth_error_pct",
                "option_abstain_pct",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.bucket_id,
                    row.question_count,
                    f"{row.accuracy:.4f}",
                    f"{row.coverage:.4f}",
                    f"{row.abstention_rate:.4f}",
                    f"{row.uncertain_truth_error_pct:.4f}",
                    f"{row.option_abstain_pct:.4f}",
                ]
            )


def write_bucket_eighth_csv(path: Path, rows: Sequence[BucketEighthStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bucket",
                "eighth",
                "question_count",
                "accuracy",
                "coverage",
                "abstention_rate",
                "uncertain_truth_error_pct",
                "option_abstain_pct",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.bucket_id,
                    row.eighth_index,
                    row.question_count,
                    f"{row.accuracy:.4f}",
                    f"{row.coverage:.4f}",
                    f"{row.abstention_rate:.4f}",
                    f"{row.uncertain_truth_error_pct:.4f}",
                    f"{row.option_abstain_pct:.4f}",
                ]
            )


class SvgPlotter:
    """Lightweight SVG plotter for bar charts and grid heatmaps."""

    def __init__(self, width: int = 900, height: int = 520) -> None:
        self.width = width
        self.height = height
        self.margin = 60

    def bar_chart(
        self,
        path: Path,
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        y_label: str,
    ) -> None:
        if not labels:
            return
        max_value = max(values) if values else 1.0
        max_value = max(max_value, 1e-6)
        bar_width = (self.width - 2 * self.margin) / len(labels)
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.width}' height='{self.height}'>",
            f"<text x='{self.width / 2}' y='30' text-anchor='middle' font-size='20'>{title}</text>",
            f"<text x='{self.margin / 2}' y='{self.height / 2}' transform='rotate(-90 {self.margin / 2},{self.height / 2})' font-size='14'>{y_label}</text>",
            f"<line x1='{self.margin}' y1='{self.height - self.margin}' x2='{self.width - self.margin}' y2='{self.height - self.margin}' stroke='black' stroke-width='2' />",
            f"<line x1='{self.margin}' y1='{self.margin}' x2='{self.margin}' y2='{self.height - self.margin}' stroke='black' stroke-width='2' />",
        ]
        for idx, (label, value) in enumerate(zip(labels, values)):
            bar_height = (self.height - 2 * self.margin) * (value / max_value)
            x = self.margin + idx * bar_width + bar_width * 0.1
            y = self.height - self.margin - bar_height
            svg.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.8:.1f}' height='{bar_height:.1f}' fill='#4F6BED' />"
            )
            svg.append(
                f"<text x='{x + bar_width * 0.4:.1f}' y='{self.height - self.margin + 15}' text-anchor='middle' font-size='12'>{label}</text>"
            )
            svg.append(
                f"<text x='{x + bar_width * 0.4:.1f}' y='{y - 5:.1f}' text-anchor='middle' font-size='11'>{value:.2f}</text>"
            )
        svg.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(svg), encoding="utf-8")

    def accuracy_heatmap(
        self,
        path: Path,
        metrics: Sequence[BucketEighthStats],
        title: str,
    ) -> None:
        if not metrics:
            return
        buckets = sorted({row.bucket_id for row in metrics})
        grid = {row.bucket_id: {} for row in metrics}
        for row in metrics:
            grid.setdefault(row.bucket_id, {})[row.eighth_index] = row.accuracy
        cell_width = (self.width - 2 * self.margin) / max(1, len(buckets))
        cell_height = (self.height - 2 * self.margin) / 8
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.width}' height='{self.height}'>",
            f"<text x='{self.width / 2}' y='30' text-anchor='middle' font-size='20'>{title}</text>",
        ]
        for row_idx in range(8):
            svg.append(
                f"<text x='{self.margin - 10}' y='{self.margin + (row_idx + 0.5) * cell_height:.1f}' text-anchor='end' font-size='12'>E{row_idx}</text>"
            )
        for col_idx, bucket in enumerate(buckets):
            svg.append(
                f"<text x='{self.margin + (col_idx + 0.5) * cell_width:.1f}' y='{self.height - self.margin + 20}' text-anchor='middle' font-size='12'>{bucket}</text>"
            )
            for row_idx in range(8):
                value = grid.get(bucket, {}).get(row_idx, 0.0)
                color = _heatmap_color(value)
                x = self.margin + col_idx * cell_width
                y = self.margin + row_idx * cell_height
                svg.append(
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_width:.1f}' height='{cell_height:.1f}' fill='{color}' stroke='#333' stroke-width='1' />"
                )
                svg.append(
                    f"<text x='{x + cell_width / 2:.1f}' y='{y + cell_height / 2 + 4:.1f}' text-anchor='middle' font-size='11'>{value * 100:.1f}%</text>"
                )
        svg.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(svg), encoding="utf-8")


def _heatmap_color(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    red = int(255 * clamped)
    blue = int(255 * (1.0 - clamped))
    return f"rgb({red},{50},{blue})"
