from __future__ import annotations

from typing import Any, Iterable


class MetricRegistry:
    """Tracks all metric names that should be logged across training."""

    def __init__(self) -> None:
        self._metrics: list[str] = []

    def register(self, *names: str) -> None:
        for name in names:
            if name not in self._metrics:
                self._metrics.append(name)

    def extend(self, names: Iterable[str]) -> None:
        for name in names:
            self.register(name)

    @property
    def metrics(self) -> list[str]:
        return list(self._metrics)


default_registry = MetricRegistry()

DEFAULT_REGISTRY = default_registry

DEFAULT_REGISTRY.extend(
    [
        "train_loss",
        "train_acc",
        "train_pred_option_pct",
        "train_pred_uncertain_pct",
        "train_uncertain_truth_error_pct",
        "train_option_truth_uncertain_pct",
        "val_loss",
        "val_acc",
        "val_pred_option_pct",
        "val_pred_uncertain_pct",
        "val_uncertain_truth_error_pct",
        "val_option_truth_uncertain_pct",
        "synthetic_val_loss",
        "synthetic_val_acc",
        "synthetic_val_pred_option_pct",
        "synthetic_val_pred_uncertain_pct",
        "synthetic_val_pred_yes_pct",
        "synthetic_val_pred_no_pct",
        "synthetic_val_uncertain_truth_error_pct",
        "synthetic_val_option_truth_uncertain_pct",
    ]
)

DEFAULT_REGISTRY.register("gradients/global_norm")

for idx in range(1, 4):
    DEFAULT_REGISTRY.register(
        f"val_entropy_bucket_{idx}_acc",
        f"val_entropy_bucket_{idx}_uncertain_pct",
        f"val_entropy_bucket_{idx}_video_count",
        f"val_entropy_bucket_{idx}_uncertain_truth_error_pct",
        f"val_entropy_bucket_{idx}_option_truth_uncertain_pct",
        f"val_length_bucket_{idx}_acc",
        f"val_length_bucket_{idx}_uncertain_pct",
        f"val_length_bucket_{idx}_video_count",
        f"val_length_bucket_{idx}_uncertain_truth_error_pct",
        f"val_length_bucket_{idx}_option_truth_uncertain_pct",
    )

DEFAULT_REGISTRY.register("val_entropy_tertile_1", "val_entropy_tertile_2")
DEFAULT_REGISTRY.register("val_length_tertile_1", "val_length_tertile_2")


def initialize_csv_logger(logger: Any | None, registry: MetricRegistry = DEFAULT_REGISTRY) -> None:
    """Ensure the CSV logger knows about every metric key before training starts."""

    if logger is None:
        return
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    metrics = getattr(experiment, "metrics_keys", None)
    if metrics is None:
        return
    for key in registry.metrics:
        if key not in metrics:
            metrics.append(key)
    save = getattr(experiment, "save", None)
    if callable(save):
        save()
