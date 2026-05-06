"""Utilities for emitting per-question CSV logs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

QUESTION_LOG_HEADERS = [
    "video_id",
    "variant",
    "bucket",
    "question_id",
    "question_order",
    "video_entropy",
    "correct_answer",
    "model_answer",
    "is_correct",
    "is_dont_know",
    "response_full",
    "response_truncated",
    "num_options",
    "question_type",
    "question_variant",
    "question_time",
    "clip_start_time",
    "clip_end_time",
    "candidate_present",
    "candidate_clip_start",
    "candidate_clip_end",
    "has_unique_answer",
    "scenario",
    "eval_mode",
    "input_mode",
    "is_native_binary",
    "response_token_count",
    "output_was_truncated",
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _truncate_response(response: str, max_length: int = 1024) -> str:
    """Truncate response to first 512 + last 512 chars (up to max_length total)."""
    if not response or len(response) <= max_length:
        return response
    # Account for the 3-char ellipsis separator
    half = (max_length - 3) // 2
    return response[:half] + "..." + response[-half:]


def _normalize_row(row: Mapping[str, Any]) -> Dict[str, str]:
    # Handle both 'correct_answer' and 'correct' keys (from different sources)
    correct_answer = row.get("correct_answer") or row.get("correct")
    model_answer = row.get("model_answer") or row.get("predicted")

    normalized = {
        "video_id": _as_text(row.get("video_id")),
        "variant": _as_text(row.get("variant") if row.get("variant") is not None else ""),
        "bucket": _as_text(row.get("bucket")),
        "question_id": _as_text(row.get("question_id")),
        "question_order": _as_text(row.get("question_order")),
        "video_entropy": _as_text(row.get("video_entropy")),
        "correct_answer": _as_text(correct_answer),
        "model_answer": _as_text(model_answer),
        "is_correct": _as_text(row.get("is_correct")),
        "is_dont_know": _as_text(row.get("is_dont_know")),
        "response_full": _as_text(row.get("response")),
        "response_truncated": _truncate_response(_as_text(row.get("response")), max_length=1024),
        "num_options": _as_text(row.get("num_options")),
        # New fields from eval_membership dataset
        "question_type": _as_text(row.get("question_type")),
        "question_variant": _as_text(row.get("question_variant")),
        "question_time": _as_text(row.get("question_time")),
        "clip_start_time": _as_text(row.get("clip_start_time")),
        "clip_end_time": _as_text(row.get("clip_end_time")),
        "candidate_present": _as_text(row.get("candidate_present")),
        "candidate_clip_start": _as_text(row.get("candidate_clip_start")),
        "candidate_clip_end": _as_text(row.get("candidate_clip_end")),
        "has_unique_answer": _as_text(row.get("has_unique_answer")),
        "scenario": _as_text(row.get("scenario")),
        "eval_mode": _as_text(row.get("eval_mode")),
        "input_mode": _as_text(row.get("input_mode")),
        "is_native_binary": _as_text(row.get("is_native_binary")),
        "response_token_count": _as_text(row.get("response_token_count")),
        "output_was_truncated": _as_text(row.get("output_was_truncated")),
    }
    return normalized


def write_question_log_csv(path: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write per-question results to CSV and return the destination path."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_LOG_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))

    return destination
