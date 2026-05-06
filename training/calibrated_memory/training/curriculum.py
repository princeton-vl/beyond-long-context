from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .data import DatasetArtifacts
from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.sequence_generator import SyntheticSampleDataset


def _compute_tertiles_from_lengths(values: Sequence[int]) -> list[float] | None:
    data = sorted(float(v) for v in values if v is not None)
    if len(data) < 3:
        return None
    def _quantile(fraction: float) -> float:
        if len(data) == 1:
            return data[0]
        position = fraction * (len(data) - 1)
        lower = int(position)
        upper = min(len(data) - 1, lower + 1)
        if lower == upper:
            return data[lower]
        weight = position - lower
        return data[lower] + (data[upper] - data[lower]) * weight

    first = _quantile(1.0 / 3.0)
    second = _quantile(2.0 / 3.0)
    return [first, second]


def _subset_tertiles(lengths: Sequence[int], indices: Sequence[int]) -> list[float] | None:
    values = [lengths[idx] for idx in indices if 0 <= idx < len(lengths)]
    if not values:
        return None
    return _compute_tertiles_from_lengths(values)


@dataclass
class CurriculumStage:
    index: int
    max_stream_len: int
    train_indices: list[int]
    val_indices: list[int]


class CurriculumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_artifacts: DatasetArtifacts,
        stages: list[CurriculumStage],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        synthetic_train_samples: Sequence[tuple[Any, Any, Any]] | None,
        synthetic_val_builder: Callable[..., DataLoader | None] | None,
        synthetic_val_factory_kwargs: dict[str, Any] | None,
        synthetic_val_seed: int,
        lengths: list[int],
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("Curriculum requires at least one stage with training samples.")
        self.dataset = dataset_artifacts.dataset
        self.collate_fn = dataset_artifacts.collate_fn or build_collate(dataset_artifacts.pad_id)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.stages = stages
        self.stage_index = 0
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._synthetic_val_loader: DataLoader | None = None
        self._needs_rebuild = True
        self.synthetic_val_builder = synthetic_val_builder
        self.synthetic_val_factory_kwargs = synthetic_val_factory_kwargs
        self.synthetic_val_seed = synthetic_val_seed
        self.synthetic_train_dataset = (
            SyntheticSampleDataset(synthetic_train_samples)
            if synthetic_train_samples
            else None
        )
        self._has_primary_val = any(stage.val_indices for stage in stages)
        self._expects_synthetic_val = bool(
            synthetic_val_builder is not None
            and synthetic_val_factory_kwargs
            and self._has_primary_val
        )
        self.lengths = lengths

    def _stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    def stage_count(self) -> int:
        return len(self.stages)

    def stage_summary(self) -> dict[str, Any]:
        stage = self._stage()
        metadata = self.stage_metadata_overrides()
        return {
            "stage": stage.index,
            "max_stream_len": stage.max_stream_len,
            "train_size": len(stage.train_indices),
            "val_size": len(stage.val_indices),
            "length_tertiles": metadata.get("stream_length_tertiles"),
        }

    def has_primary_val(self) -> bool:
        return self._has_primary_val

    def expects_synthetic_val(self) -> bool:
        return self._expects_synthetic_val

    def advance_stage(self) -> bool:
        if self.stage_index + 1 >= len(self.stages):
            return False
        self.stage_index += 1
        self._needs_rebuild = True
        return True

    def _build_loaders(self) -> None:
        stage = self._stage()
        if not stage.train_indices:
            raise RuntimeError(
                f"Curriculum stage {stage.index} does not contain any training samples."
            )
        train_dataset: Dataset = Subset(self.dataset, stage.train_indices)
        if self.synthetic_train_dataset is not None:
            train_dataset = ConcatDataset([train_dataset, self.synthetic_train_dataset])
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
        if stage.val_indices:
            val_dataset: Dataset = Subset(self.dataset, stage.val_indices)
            self._val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                drop_last=False,
            )
        else:
            self._val_loader = None
        if (
            self.synthetic_val_builder is not None
            and self.synthetic_val_factory_kwargs
            and stage.val_indices
        ):
            self._synthetic_val_loader = self.synthetic_val_builder(
                factory_kwargs=self.synthetic_val_factory_kwargs or {},
                dataset_size=len(stage.val_indices),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.synthetic_val_seed,
            )
        else:
            self._synthetic_val_loader = None
        self._needs_rebuild = False

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is None or self._needs_rebuild:
            self._build_loaders()
        return self._train_loader  # type: ignore[return-value]

    def val_dataloader(self):
        if self._needs_rebuild:
            self._build_loaders()
        loaders: list[DataLoader] = []
        if self._val_loader is not None:
            loaders.append(self._val_loader)
        if self._synthetic_val_loader is not None:
            loaders.append(self._synthetic_val_loader)
        if not loaders:
            return None
        if len(loaders) == 1:
            return loaders[0]
        return loaders

    def stage_metadata_overrides(self) -> dict[str, Any]:
        stage = self._stage()
        overrides: dict[str, Any] = {}
        indices = stage.train_indices + stage.val_indices
        length_bounds = _subset_tertiles(self.lengths, indices)
        if length_bounds:
            overrides["stream_length_tertiles"] = length_bounds
        return overrides


class CurriculumCallback(pl.Callback):
    def __init__(self, data_module: CurriculumDataModule, target_acc: float) -> None:
        super().__init__()
        self.data_module = data_module
        self.target_acc = target_acc

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        self._log_stage(trainer, pl_module, note="initialized curriculum stage")

    def on_train_epoch_end(self, trainer, pl_module, unused: Any = None) -> None:  # type: ignore[override]
        train_acc = trainer.callback_metrics.get("train_acc")
        if train_acc is None:
            return
        try:
            value = float(train_acc)
        except (TypeError, ValueError):
            return
        if value < self.target_acc:
            return
        advanced = self.data_module.advance_stage()
        if not advanced:
            return
        trainer.reset_train_dataloader()
        trainer.reset_val_dataloader()
        self._log_stage(trainer, pl_module, note="advanced curriculum stage")

    def _log_stage(self, trainer, pl_module, note: str) -> None:
        summary = self.data_module.stage_summary()
        message = (
            f"[{note}] stage={summary['stage']} max_len={summary['max_stream_len']} "
            f"train={summary['train_size']} val={summary['val_size']}"
        )
        rank = getattr(trainer, "global_rank", 0)
        if rank == 0:
            print(message)
        metrics = {
            "curriculum_stage": summary["stage"],
            "curriculum_max_len": summary["max_stream_len"],
            "curriculum_train_samples": summary["train_size"],
            "curriculum_val_samples": summary["val_size"],
        }
        logger = trainer.logger
        if isinstance(logger, Sequence):
            loggers = [entry for entry in logger if entry is not None]
        else:
            loggers = [logger] if logger is not None else []
        for entry in loggers:
            try:
                entry.log_metrics(metrics, step=trainer.global_step)
            except Exception:  # noqa: BLE001
                continue
        self._update_model_metadata(pl_module)

    def _update_model_metadata(self, pl_module) -> None:
        overrides = self.data_module.stage_metadata_overrides()
        if not overrides:
            return
        metadata = dict(getattr(pl_module, "dataset_metadata", {}) or {})
        metadata.update(overrides)
        pl_module.set_dataset_metadata(metadata)


@dataclass
class CurriculumComponents:
    data_module: CurriculumDataModule
    callback: CurriculumCallback
    summary: dict[str, Any]


def _extract_stream_lengths(dataset: Dataset) -> list[int] | None:
    metadata = getattr(dataset, "sample_metadata", None)
    if metadata is None:
        return None
    lengths: list[int] = []
    for entry in metadata:
        if not entry:
            lengths.append(0)
            continue
        length = entry.get("stream_length")
        if length is None:
            length = entry.get("length_value")
        if length is None:
            return None
        lengths.append(int(length))
    return lengths if len(lengths) == len(dataset) else None


def build_curriculum_components(
    *,
    args: Any,
    dataset_artifacts: DatasetArtifacts,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    synthetic_train_samples: Sequence[tuple[Any, Any, Any]] | None,
    synthetic_val_builder: Callable[..., DataLoader | None] | None,
    synthetic_val_factory_kwargs: dict[str, Any] | None,
    length_overrides: list[int] | None = None,
) -> CurriculumComponents | None:
    start_len = args.curriculum_start
    target_acc = args.curriculum_target_acc
    if start_len is None or target_acc is None:
        return None
    if start_len <= 0 or target_acc <= 0:
        raise ValueError("Curriculum flags must be positive values.")
    lengths = _extract_stream_lengths(dataset_artifacts.dataset)
    if not lengths:
        raise ValueError(
            "Curriculum mode requires datasets that expose per-sample stream lengths."
        )
    combined_lengths = length_overrides if length_overrides else lengths
    max_length = max(lengths)
    current_limit = min(max_length, max(start_len, 1))
    stage_limits: List[int] = []
    while True:
        stage_limits.append(current_limit)
        if current_limit >= max_length:
            break
        current_limit = min(max_length, max(current_limit * 2, current_limit + 1))
    generator = torch.Generator().manual_seed(seed)
    total = len(lengths)
    all_indices = torch.randperm(total, generator=generator).tolist()
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must lie in [0,1) for curriculum mode.")
    val_size = 0
    if val_fraction > 0.0:
        val_size = max(1, int(total * val_fraction))
        val_size = min(val_size, total - 1)
    train_indices = all_indices[val_size:]
    val_indices = all_indices[:val_size]
    stages: list[CurriculumStage] = []
    next_index = 0
    for limit in stage_limits:
        train_subset = [idx for idx in train_indices if lengths[idx] <= limit]
        if not train_subset:
            continue
        val_subset = [idx for idx in val_indices if lengths[idx] <= limit]
        stage = CurriculumStage(
            index=next_index,
            max_stream_len=limit,
            train_indices=train_subset,
            val_indices=val_subset,
        )
        stages.append(stage)
        next_index += 1
    if len(stages) <= 1:
        return None
    data_module = CurriculumDataModule(
        dataset_artifacts=dataset_artifacts,
        stages=stages,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        synthetic_train_samples=synthetic_train_samples,
        synthetic_val_builder=synthetic_val_builder,
        synthetic_val_factory_kwargs=synthetic_val_factory_kwargs,
        synthetic_val_seed=seed + 1000,
        lengths=lengths,
    )
    callback = CurriculumCallback(data_module, target_acc=target_acc)
    summary = {
        "start_len": start_len,
        "target_train_acc": target_acc,
        "combined_length_tertiles": _compute_tertiles_from_lengths(combined_lengths),
        "stages": [
            {
                "index": stage.index,
                "max_stream_len": stage.max_stream_len,
                "train_size": len(stage.train_indices),
                "val_size": len(stage.val_indices),
                "length_tertiles": _subset_tertiles(lengths, stage.train_indices + stage.val_indices),
            }
            for stage in stages
        ],
    }
    return CurriculumComponents(data_module=data_module, callback=callback, summary=summary)
