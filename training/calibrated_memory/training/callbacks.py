"""Custom Lightning callbacks used by the training CLI."""

from __future__ import annotations

from typing import Sequence

import torch
import pytorch_lightning as pl


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


class DualValidationAverager(pl.Callback):
    """Logs a weighted average metric when two validation loaders are active."""

    def __init__(self, metric_name: str = "val_overall_acc") -> None:
        super().__init__()
        self.metric_name = metric_name

    def _val_batch_weights(self, batches: Sequence[int | float] | int | float) -> tuple[float, float] | None:
        if isinstance(batches, (int, float)):
            return None
        if len(batches) < 2:
            return None
        primary = float(batches[0])
        secondary = float(batches[1])
        if primary <= 0.0 or secondary <= 0.0:
            return None
        return primary, secondary

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        weights = self._val_batch_weights(getattr(trainer, "num_val_batches", None))
        if weights is None:
            return
        metrics = trainer.callback_metrics
        val_metric = _to_float(metrics.get("val_acc"))
        synthetic_metric = _to_float(metrics.get("synthetic_val_acc"))
        if val_metric is None or synthetic_metric is None:
            return
        primary_weight, secondary_weight = weights
        total = primary_weight + secondary_weight
        if total <= 0.0:
            return
        overall = (val_metric * primary_weight + synthetic_metric * secondary_weight) / total
        device = getattr(pl_module, "device", None)
        tensor = torch.tensor(overall, device=device if device is not None else None)
        pl_module.log(
            self.metric_name,
            tensor,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            add_dataloader_idx=False,
        )

