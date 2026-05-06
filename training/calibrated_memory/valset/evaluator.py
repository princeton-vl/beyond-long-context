"""Evaluation helper that ties checkpoints, manifests, and metrics together."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.common import TOKEN_OFFSET
from calibrated_memory.evaluation.checkpoint import instantiate_model_from_run
from calibrated_memory.evaluation.dataset import EvaluationConfig, build_evaluation_dataset
from calibrated_memory.evaluation.runner import EvaluationRunner, describe_label
from calibrated_memory.valset.metrics import (
    EvaluationMetrics,
    QuestionEval,
    SvgPlotter,
    build_bucket_eighth_metrics,
    build_bucket_metrics,
    write_bucket_csv,
    write_bucket_eighth_csv,
)


@dataclass
class ValidationConfig:
    """Inputs controlling a validation run."""

    run_dir: Path
    manifest_path: Path
    task: str
    checkpoint_name: str = "best.ckpt"
    token_offset: int | None = None
    cont_len: int | None = None
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"
    output_dir: Path | None = None
    save_plots: bool = True


def run_validation(config: ValidationConfig) -> EvaluationMetrics:
    """Run inference on the manifest and write metrics/artifacts."""

    resolved_device = _resolve_device(config.device)
    model, dataset_overrides, dataset_artifacts, _ = instantiate_model_from_run(
        config.run_dir,
        config.checkpoint_name,
        resolved_device,
    )
    token_offset = config.token_offset or int(dataset_overrides.get("token_offset", TOKEN_OFFSET))
    task = str(config.task)
    cont_len = config.cont_len
    if cont_len is None:
        cont_override = dataset_overrides.get("cont_len")
        if cont_override is not None:
            cont_len = int(cont_override)
        else:
            derived_cont = getattr(dataset_artifacts.dataset, "cont_len", None)
            if derived_cont is not None:
                cont_len = int(derived_cont)
    if task == "continuation" and (cont_len is None or cont_len <= 0):
        raise ValueError("Continuation validation requires a positive cont_len.")
    eval_dataset = build_evaluation_dataset(
        EvaluationConfig(
            manifest_path=config.manifest_path,
            task=task,
            cont_len=cont_len or 0,
            token_offset=token_offset,
        )
    )
    collate_fn = build_collate(eval_dataset.pad_id)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    runner = EvaluationRunner(
        model,
        device=resolved_device,
        task=task,
    )
    question_results = runner.run(dataloader)
    eval_records = _to_question_evals(
        question_results,
        task,
    )
    bucket_metrics = build_bucket_metrics(eval_records)
    eighth_metrics = build_bucket_eighth_metrics(eval_records)
    output_dir = config.output_dir or _default_output_dir(config.manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_bucket_csv(output_dir / "bucket_metrics.csv", bucket_metrics)
    write_bucket_eighth_csv(output_dir / "bucket_eighth_metrics.csv", eighth_metrics)
    _write_question_csv(output_dir / "per_question.csv", eval_records)
    _write_summary(
        output_dir / "summary.json",
        eval_records,
        bucket_metrics,
    )
    if config.save_plots:
        plotter = SvgPlotter()
        plotter.bar_chart(
            output_dir / "accuracy_per_bucket.svg",
            [row.bucket_id for row in bucket_metrics],
            [row.accuracy * 100 for row in bucket_metrics],
            "Accuracy per Bucket",
            "Accuracy (%)",
        )
        plotter.bar_chart(
            output_dir / "abstention_per_bucket.svg",
            [row.bucket_id for row in bucket_metrics],
            [row.abstention_rate * 100 for row in bucket_metrics],
            "Abstention Rate per Bucket",
            "Abstention (%)",
        )
        plotter.accuracy_heatmap(
            output_dir / "accuracy_heatmap.svg",
            eighth_metrics,
            "Accuracy by Eighth",
        )
    return EvaluationMetrics(
        bucket_metrics=bucket_metrics,
        bucket_eighth_metrics=eighth_metrics,
    )


def _resolve_device(spec: str) -> torch.device:
    normalized = (spec or "auto").lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("GPU is required for evaluation but no CUDA device is visible.")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable on this machine.")
    return device


def _to_question_evals(
    records: Sequence[Any],
    task: str,
) -> list[QuestionEval]:
    eval_rows: list[QuestionEval] = []
    for record in records:
        metadata = record.metadata or {}
        bucket_id = str(metadata.get("bucket_id") or "unknown")
        stream_length = int(metadata.get("stream_total_length") or metadata.get("video", {}).get("length_value", 0) or 0)
        concerned = _parse_ranges(metadata.get("concerned_ranges"))
        truth_desc = describe_label(record.truth_token)
        pred_desc = describe_label(record.pred_token)
        truth_kind = "uncertain" if truth_desc == "uncertain" else "option"
        pred_kind = "uncertain" if pred_desc == "uncertain" else "option"
        eval_rows.append(
            QuestionEval(
                bucket_id=bucket_id,
                stream_length=stream_length,
                concerned_ranges=concerned,
                truth_kind=truth_kind,
                pred_kind=pred_kind,
                correct=record.correct,
            )
        )
    return eval_rows


def _parse_ranges(raw: Any) -> list[tuple[int, int]]:
    if not isinstance(raw, Iterable):
        return []
    ranges: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start = int(entry.get("start", 0))
        end = int(entry.get("end", start))
        if end <= start:
            continue
        ranges.append((start, end))
    return ranges


def _default_output_dir(manifest_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return manifest_path.parent / f"val-{timestamp}"


def _write_summary(
    path: Path,
    records: Sequence[QuestionEval],
    bucket_metrics: Sequence[Any],
) -> None:
    total = len(records)
    if total == 0:
        summary = {"question_count": 0}
    else:
        correct = sum(1 for row in records if row.correct)
        pred_uncertain = sum(1 for row in records if row.pred_kind == "uncertain")
        truth_uncertain = sum(1 for row in records if row.truth_kind == "uncertain")
        answerable = total - truth_uncertain
        summary = {
            "question_count": total,
            "accuracy": correct / total,
            "abstention_rate": pred_uncertain / total,
            "truth_uncertain": truth_uncertain,
            "option_total": answerable,
        }
    summary["buckets"] = [
        {
            "bucket": row.bucket_id,
            "question_count": row.question_count,
            "accuracy": row.accuracy,
            "coverage": row.coverage,
            "abstention_rate": row.abstention_rate,
            "uncertain_truth_error_pct": row.uncertain_truth_error_pct,
            "option_abstain_pct": row.option_abstain_pct,
        }
        for row in bucket_metrics
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_question_csv(path: Path, rows: Sequence[QuestionEval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("bucket,stream_length,truth_kind,pred_kind,correct,concerned_ranges\n")
        for row in rows:
            handle.write(
                f"{row.bucket_id},{row.stream_length},{row.truth_kind},{row.pred_kind},{int(row.correct)},\"{row.concerned_ranges}\"\n"
            )
