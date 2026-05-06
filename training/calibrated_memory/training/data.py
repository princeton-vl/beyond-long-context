from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.sequence_generator import SyntheticSampleDataset


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset: Dataset
    pad_id: int
    vocab_size: int
    max_seq_len: int
    metadata: dict[str, Any] | None = None
    collate_fn: Callable | None = None


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: Optional[DataLoader]


def _maybe_split_dataset(
    dataset: Dataset,
    val_fraction: float,
    *,
    seed: int,
) -> tuple[Dataset, Optional[Dataset]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must lie in [0.0, 1.0)")
    if val_fraction == 0.0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    splits = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits[0], splits[1]


def _build_loader(
    dataset: Dataset,
    collate_fn: Callable,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )


def create_dataloaders(
    artifacts: DatasetArtifacts,
    *,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    synthetic_train_samples: Sequence[tuple[Any, Any, Any]] | None = None,
) -> DataLoaders:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    collate_fn = artifacts.collate_fn or build_collate(artifacts.pad_id)
    train_dataset, val_dataset = _maybe_split_dataset(artifacts.dataset, val_fraction, seed=seed)
    if synthetic_train_samples:
        synthetic_dataset = SyntheticSampleDataset(synthetic_train_samples)
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

    train_loader = _build_loader(
        train_dataset,
        collate_fn,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _build_loader(
            val_dataset,
            collate_fn,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return DataLoaders(train=train_loader, val=val_loader)
