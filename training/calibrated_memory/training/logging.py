from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase


@dataclass(frozen=True)
class LoggingArtifacts:
    loggers: List[LightningLoggerBase]
    callbacks: List[pl.Callback]


class MetricsProgressBar(TQDMProgressBar):
    """TQDM progress bar that highlights key decoder metrics."""

    def __init__(self, refresh_rate: int, metric_keys: Sequence[str]):
        super().__init__(refresh_rate=refresh_rate)
        self.metric_keys = list(metric_keys)

    def get_metrics(self, trainer, pl_module):  # type: ignore[override]
        metrics = super().get_metrics(trainer, pl_module)
        if not self.metric_keys:
            return metrics
        prioritized: Dict[str, float] = {}
        for key in self.metric_keys:
            if key in metrics:
                prioritized[key] = metrics[key]
        # Append the remaining metrics to keep Lightning internals intact.
        for key, value in metrics.items():
            if key not in prioritized:
                prioritized[key] = value
        return prioritized


WANDB_RUN_NOTE_KEY = "wandb_log_note"
_NOTE_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def format_wandb_run_name(base_name: str, note: str | None) -> str:
    """Append a sanitized log note slug to the WandB run name when available."""

    normalized_note = note.strip() if note else ""
    if not normalized_note:
        return base_name
    slug = _slugify_note(normalized_note)
    if not slug:
        return base_name
    return f"{base_name}-{slug}"


def _slugify_note(note: str) -> str:
    slug = _NOTE_SANITIZE_PATTERN.sub("-", note)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-_.")


def build_logging(
    *,
    log_dir: Path,
    experiment_name: str,
    enable_wandb: bool,
    wandb_project: str | None,
    wandb_run_name: str | None,
    wandb_tags: list[str] | None,
    wandb_dir: Path | None,
    wandb_mode: str | None,
    wandb_log_note: str | None,
    progress_refresh_rate: int,
    run_config: Dict[str, object],
    metric_keys: Sequence[str],
    dataset_metadata: dict[str, Any] | None,
) -> LoggingArtifacts:
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers: list[LightningLoggerBase] = [
        CSVLogger(save_dir=str(log_dir), name=experiment_name)
    ]
    callbacks: list[pl.Callback] = [
        MetricsProgressBar(
            refresh_rate=max(1, progress_refresh_rate),
            metric_keys=metric_keys,
        )
    ]

    metadata_payload = dataset_metadata or {}
    if metadata_payload:
        run_log_dir = log_dir / experiment_name
        run_log_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = run_log_dir / "dataset_metadata.json"
        metadata_file.write_text(json.dumps(metadata_payload, indent=2))

    note_value = wandb_log_note.strip() if wandb_log_note else ""

    if enable_wandb:
        wandb_save_dir = wandb_dir or log_dir
        wandb_save_dir.mkdir(parents=True, exist_ok=True)
        tags = list(wandb_tags or [])
        timestamp_tag = datetime.now().strftime("ts-%Y%m%d-%H%M%S")
        tags.append(timestamp_tag)
        run_name = format_wandb_run_name(wandb_run_name or experiment_name, note_value)
        wandb_logger = WandbLogger(
            project=wandb_project or "qa-ego-memory",
            name=run_name,
            tags=tags,
            save_dir=str(wandb_save_dir),
            mode=wandb_mode or "online",
            log_model=False,
        )
        experiment = wandb_logger.experiment
        if experiment is None:
            raise RuntimeError(
                "WandB failed to initialize; re-run with --disable-wandb to skip remote logging."
            )
        experiment.config.update(run_config, allow_val_change=True)
        if note_value:
            _record_wandb_note(experiment, note_value)
        if metadata_payload:
            summary = experiment.summary
            try:
                existing = summary.get("dataset_metadata")
            except AttributeError:
                existing = None
            if not existing:
                summary["dataset_metadata"] = metadata_payload
            experiment.log({"dataset_metadata": metadata_payload}, step=0, commit=True)
        loggers.append(wandb_logger)

    return LoggingArtifacts(loggers=loggers, callbacks=callbacks)


def _record_wandb_note(experiment: Any, note: str) -> None:
    payload = {WANDB_RUN_NOTE_KEY: note}
    experiment.summary[WANDB_RUN_NOTE_KEY] = note
    experiment.log(payload, step=0, commit=True)
