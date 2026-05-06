from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.utils.data import DataLoader

from calibrated_memory.utils.paths import (
    REPO_ROOT,
    cleanup_stray_tempdirs,
    configure_temp_directory,
)

from calibrated_memory.training.data import create_dataloaders, DatasetArtifacts, DataLoaders
from calibrated_memory.training.curriculum import build_curriculum_components
from calibrated_memory.training.logging import build_logging
from calibrated_memory.benchmark.runner import BenchmarkConfig, BenchmarkRunner, StreamingProfile
from calibrated_memory.utils.logging import DEFAULT_REGISTRY, initialize_csv_logger
from calibrated_memory.training.registries import (
    BACKEND_SPECS,
    backend_choices,
    build_backend,
    build_dataset,
    ensure_backend_context_capacity,
    dataset_choices,
)
from calibrated_memory.backend.models.backend_base import MemoryBackend
from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.common import (
    IGNORE_INDEX,
    TOKEN_OFFSET,
    UNCERTAIN_TOKEN,
    LABEL_TOKENS,
)
from calibrated_memory.data.sequences.sequence_generator import (
    SequenceDataset,
    SyntheticSampleDataset,
    SyntheticSampleFactory,
)
from calibrated_memory.evaluation.checkpoint import instantiate_model_from_run, load_run_metadata
from calibrated_memory.evaluation.dataset import EvaluationConfig, EvaluationDataset, build_evaluation_dataset
from calibrated_memory.evaluation.runner import EvaluationRunner, StatsAggregator, describe_label
from calibrated_memory.training.callbacks import DualValidationAverager

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


DECODER_D_MODEL_DEFAULT = 64
_FLA_BACKENDS = {
    "deltaformer",
    "deltanet",
    "gated_deltanet",
    "log_linear_mamba",
    "mom",
    "retnet",
    "rwkv",
}


def _enforce_fla_device_policy(args: argparse.Namespace) -> None:
    """Force CUDA execution for flash-linear backends and fail fast otherwise."""

    backend_name = getattr(args, "backend", None)
    if backend_name not in _FLA_BACKENDS:
        return
    accelerator = str(getattr(args, "accelerator", "auto") or "auto").lower()
    if accelerator == "auto":
        args.accelerator = "gpu"
        accelerator = "gpu"
    if accelerator not in {"gpu", "cuda"}:
        raise SystemExit(
            f"Backend '{backend_name}' requires accelerator='gpu' (or 'cuda'); got {args.accelerator!r}."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            f"Backend '{backend_name}' relies on Triton/FlashAttention kernels but no CUDA devices are visible. "
            "Double-check the SLURM submission (e.g., --gres/--constraint) so the job lands on a GPU node."
        )
DECODER_TRANSFORMER_DEFAULTS = {
    "decoder_num_layers": 3,
    "decoder_nhead": 8,
    "decoder_mlp_ratio": 1,
    "decoder_attn_dropout": 0.1,
    "decoder_resid_dropout": 0.1,
    "decoder_rotary_base": 10000.0,
}

STRUCTURAL_ARG_FLAGS: dict[str, list[str]] = {
    "backend": ["--backend"],
    "dataset": ["--dataset"],
    "decoder_d_model": ["--decoder-d-model"],
    "decoder_num_layers": ["--decoder-num-layers"],
    "decoder_nhead": ["--decoder-nhead"],
    "decoder_mlp_ratio": ["--decoder-mlp-ratio"],
    "feature_input_dim": ["--feature-input-dim"],
}


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        literal = ast.literal_eval(raw)
        return literal
    except (ValueError, SyntaxError):
        pass
    return raw


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Overrides must follow key=value format, got: {value}")
        key, raw_val = value.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Invalid override key in '{value}'")
        overrides[key] = _coerce_value(raw_val.strip())
    return overrides


@dataclass(frozen=True)
class TrainingInitContext:
    run_dir: Path
    checkpoint_path: Path
    backend_overrides: dict[str, Any]
    dataset_overrides: dict[str, Any]
    parent_run_name: str
    metadata: dict[str, Any]
    load_optimizer_state: bool
    run_suffix: str


_GRAD_COMPONENT_SANITIZE = re.compile(r"[^0-9A-Za-z._-]+")


def _sanitize_grad_component_label(raw: str) -> str:
    slug = _GRAD_COMPONENT_SANITIZE.sub("-", raw.strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-_.")


def _parse_grad_component_specs(values: Sequence[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for entry in values:
        normalized = entry.strip()
        if not normalized:
            continue
        if "=" in normalized:
            label, prefix = normalized.split("=", 1)
        else:
            label, prefix = normalized, normalized
        label = label.strip()
        prefix = prefix.strip()
        if not prefix:
            raise ValueError(
                f"Gradient component specification '{entry}' is missing a valid module prefix."
            )
        slug_source = label or prefix
        slug = _sanitize_grad_component_label(slug_source) or _sanitize_grad_component_label(prefix)
        if not slug:
            raise ValueError(
                f"Failed to derive a logging label from gradient component '{entry}'."
            )
        specs.append((slug, prefix))
    return specs


def _flag_was_provided(argv: Sequence[str], options: Sequence[str]) -> bool:
    for token in argv:
        for option in options:
            if token == option or token.startswith(option + "="):
                return True
    return False


def _coerce_structural_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in {"decoder_d_model", "decoder_num_layers", "decoder_nhead", "decoder_mlp_ratio"}:
        return int(value)
    if name == "feature_input_dim":
        return None if value in {"", None} else int(value)
    return str(value)


def _assign_structural_args(args: argparse.Namespace, stored_args: dict[str, Any]) -> None:
    for name in STRUCTURAL_ARG_FLAGS:
        stored_value = _coerce_structural_value(name, stored_args.get(name))
        if stored_value is None:
            continue
        setattr(args, name, stored_value)


def _validate_structural_conflicts(
    args: argparse.Namespace,
    argv: Sequence[str],
    stored_args: dict[str, Any],
) -> None:
    conflicts: list[str] = []
    for name, flags in STRUCTURAL_ARG_FLAGS.items():
        stored_value = _coerce_structural_value(name, stored_args.get(name))
        if stored_value is None:
            continue
        if not _flag_was_provided(argv, flags):
            continue
        current_value = getattr(args, name, None)
        if current_value == stored_value:
            continue
        option_label = ", ".join(flags)
        conflicts.append(f"{option_label}={current_value!r} (expected {stored_value!r})")
    if conflicts:
        raise SystemExit(
            "--init-from detected conflicting structural overrides: " + "; ".join(conflicts)
        )
    _assign_structural_args(args, stored_args)


def _validate_override_conflicts(
    cli_overrides: dict[str, Any],
    stored_overrides: dict[str, Any] | None,
    kind: str,
) -> dict[str, Any]:
    stored = dict(stored_overrides or {})
    if not stored:
        return dict(cli_overrides)
    if not cli_overrides:
        return stored
    conflicts: list[str] = []
    for key, value in cli_overrides.items():
        if key not in stored:
            conflicts.append(f"{key} (not present in saved {kind} overrides)")
            continue
        if stored[key] != value:
            conflicts.append(f"{key}={value!r} (expected {stored[key]!r})")
    if conflicts:
        raise SystemExit(
            "--init-from requires {kind} overrides to match the saved run; "
            + "; ".join(conflicts)
        )
    return stored


def _apply_training_initialization(
    args: argparse.Namespace,
    argv: Sequence[str],
    cli_dataset_overrides: dict[str, Any],
    cli_backend_overrides: dict[str, Any],
) -> TrainingInitContext | None:
    run_dir = args.init_from
    if run_dir is None:
        return None
    if args.mode != "train":
        raise SystemExit("--init-from is only valid in training mode.")
    if args.resume_from:
        raise SystemExit("--resume-from cannot be combined with --init-from; choose one.")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    metadata = load_run_metadata(run_dir)
    stored_args = metadata.get("args", {})
    _validate_structural_conflicts(args, argv, stored_args)
    backend_overrides = _validate_override_conflicts(
        cli_backend_overrides, metadata.get("backend_overrides"), "backend"
    )
    dataset_overrides = _validate_override_conflicts(
        cli_dataset_overrides, metadata.get("dataset_overrides"), "dataset"
    )
    checkpoint_name = args.init_checkpoint or "best.ckpt"
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found under {run_dir}"
        )
    suffix = datetime.now().strftime("init-%Y%m%d-%H%M%S%f")
    parent_run_name = metadata.get("run_name") or run_dir.name
    load_optimizer_state = bool(args.init_load_optimizer)
    return TrainingInitContext(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        backend_overrides=backend_overrides,
        dataset_overrides=dataset_overrides,
        parent_run_name=parent_run_name,
        metadata=metadata,
        load_optimizer_state=load_optimizer_state,
        run_suffix=suffix,
    )


def _load_checkpoint_weights(model: MemoryBankDecoder, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)


def _parse_devices(raw: str) -> Any:
    normalized = raw.strip()
    if normalized == "auto":
        return normalized
    if "," in normalized:
        entries = [entry for entry in normalized.split(",") if entry]
        return [int(entry) for entry in entries]
    try:
        return int(normalized)
    except ValueError:
        return normalized


def _resolve_dataset_stat(
    dataset_overrides: dict[str, Any],
    key: str,
    fallback: int,
) -> int:
    value = dataset_overrides.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _describe_sequence_length(
    dataset_overrides: dict[str, Any],
    fallback: int,
) -> str:
    seq_value = dataset_overrides.get("seq_len")
    if seq_value is not None:
        try:
            return str(int(seq_value))
        except (TypeError, ValueError):
            pass
    seq_min = dataset_overrides.get("seq_len_min")
    seq_max = dataset_overrides.get("seq_len_max")
    if seq_min is not None or seq_max is not None:
        try:
            min_value = int(seq_min if seq_min is not None else seq_max)
            max_value = int(seq_max if seq_max is not None else seq_min)
        except (TypeError, ValueError):
            min_value = max_value = fallback
        if min_value == max_value:
            return str(min_value)
        return f"{min_value}-{max_value}"
    return str(int(fallback))


def _parse_sequence_lengths(raw: str, fallback: int) -> list[int]:
    if not raw:
        return [fallback]
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(max(1, int(chunk)))
        except ValueError as exc:  # noqa: PERF203
            raise ValueError(f"Invalid sequence length '{chunk}' in --benchmark-seq-lens") from exc
    return values or [fallback]


def _resolve_lr_warmup(raw_value: float, max_epochs: int) -> tuple[int, float | None]:
    warmup = float(raw_value or 0.0)
    if warmup <= 0.0:
        return 0, None
    if max_epochs <= 1:
        raise SystemExit("--lr-warmup requires --max-epochs greater than 1.")
    if warmup < 1.0:
        fraction = max(0.0, min(warmup, 0.999999))
        return 0, fraction
    warmup_epochs = int(round(warmup))
    warmup_epochs = min(warmup_epochs, max_epochs - 1)
    return warmup_epochs, None


def _validate_augmentation_args(args: argparse.Namespace) -> float:
    ratio = float(getattr(args, "augment_with_synthetic", 0.0) or 0.0)
    if ratio < 0.0:
        raise SystemExit("--augment-with-synthetic must be non-negative.")
    if ratio > 0.0:
        if args.dataset != "file":
            raise SystemExit("--augment-with-synthetic only supports --dataset=file.")
        if args.mode != "train":
            raise SystemExit(
                "--augment-with-synthetic is only available in training mode; "
                "set --mode=train to enable it."
            )
    return ratio


def _infer_question_count(
    sample: Sequence[Any],
    *,
    task: str,
    cont_len: int,
) -> int:
    if len(sample) < 3:
        raise ValueError("Dataset sample is missing the (seq, labels, stream_len) structure.")
    _, labels, stream_len_tensor = sample[:3]
    stream_len = int(stream_len_tensor.item())
    query_labels = labels[stream_len:]
    valid_positions = query_labels != IGNORE_INDEX
    labeled_tokens = int(valid_positions.sum().item())
    if labeled_tokens <= 0:
        raise ValueError("Unable to infer question count; the first sample lacked labeled tokens.")
    return labeled_tokens


def _dataset_length_bounds(dataset: Any) -> tuple[int, int]:
    metadata = getattr(dataset, "sample_metadata", None)
    lengths: list[int] = []
    if isinstance(metadata, list) and metadata:
        for entry in metadata:
            value = entry.get("stream_length") if isinstance(entry, dict) else None
            if value is None:
                continue
            lengths.append(int(value))
    if not lengths:
        total = len(dataset)
        for idx in range(total):
            sample = dataset[idx]
            if len(sample) < 3:
                continue
            stream_len_tensor = sample[2]
            lengths.append(int(stream_len_tensor.item()))
    if not lengths:
        raise ValueError("Failed to infer stream length bounds from dataset metadata.")
    return min(lengths), max(lengths)


def _prepare_synthetic_augmentation(
    *,
    dataset_artifacts: DatasetArtifacts,
    dataset_overrides: dict[str, Any],
    task: str,
    cont_len: int,
    ratio: float,
    seed: int,
) -> tuple[dict[str, Any], list[tuple[Any, Any, Any]]] | tuple[None, None]:
    dataset = dataset_artifacts.dataset
    total_samples = len(dataset)
    if total_samples == 0:
        raise SystemExit("Synthetic augmentation requires a non-empty dataset.")
    synthetic_total = int(math.ceil(total_samples * ratio))
    if synthetic_total <= 0:
        return None, None
    try:
        first_sample = dataset[0]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Failed to inspect the dataset for augmentation planning.") from exc
    try:
        question_count = _infer_question_count(first_sample, task=task, cont_len=cont_len)
    except ValueError as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc
    try:
        min_len, max_len = _dataset_length_bounds(dataset)
    except ValueError as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc
    if min_len <= 0 or max_len <= 0:
        raise SystemExit("Synthetic augmentation requires positive stream lengths.")
    token_offset = int(dataset_overrides.get("token_offset", TOKEN_OFFSET))
    vocab_span = dataset_artifacts.pad_id - token_offset
    if vocab_span <= 0:
        raise SystemExit(
            "Derived vocab span for synthetic augmentation is non-positive; check token_offset overrides."
        )
    factory_config = {
        "seq_len": None,
        "seq_len_range": (min_len, max_len),
        "unique_sequences": question_count,
        "vocab_size": max(1, int(vocab_span)),
        "task": task,
        "cont_len": cont_len,
        "token_offset": token_offset,
    }
    factory = SyntheticSampleFactory(seed=seed, **factory_config)
    synthetic_samples = factory.build_samples(synthetic_total)
    if synthetic_samples:
        avg_len = sum(int(sample[2].item()) for sample in synthetic_samples) / len(synthetic_samples)
        print(f"[synthetic] augmented stream length avg={avg_len:.2f}")
    return factory_config, synthetic_samples


def _build_synthetic_val_loader(
    *,
    factory_kwargs: dict[str, Any],
    dataset_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader | None:
    if dataset_size <= 0:
        return None
    factory = SyntheticSampleFactory(seed=seed, **factory_kwargs)
    samples, derived_pad_id, _, _ = factory.build_batch(dataset_size)
    synthetic_ds = SyntheticSampleDataset(samples)
    collate_fn = build_collate(derived_pad_id)
    return DataLoader(
        synthetic_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _metadata_bounds(summary: dict[str, Any], keys: list[str]) -> list[float] | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            items = [float(v) for v in value if v is not None]
            if len(items) >= 2:
                items.sort()
                return items[:2]
    return None


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _infer_stream_token_bounds(
    artifacts: DatasetArtifacts,
    *,
    sample_limit: int = 128,
) -> tuple[int, int] | None:
    dataset = artifacts.dataset
    total = len(dataset)
    if total == 0:
        return None
    min_token: int | None = None
    max_token: int | None = None
    limit = min(total, sample_limit)
    for idx in range(limit):
        sample = dataset[idx]
        if len(sample) < 3:
            continue
        seq, _, stream_len_tensor = sample[:3]
        try:
            stream_len = int(stream_len_tensor.item())
        except Exception:  # noqa: BLE001
            stream_len = int(stream_len_tensor)
        if stream_len <= 0:
            continue
        stream_tokens = seq[:stream_len]
        if stream_tokens.numel() == 0:
            continue
        stream_min = int(stream_tokens.min().item())
        stream_max = int(stream_tokens.max().item())
        min_token = stream_min if min_token is None else min(min_token, stream_min)
        max_token = stream_max if max_token is None else max(max_token, stream_max)
    if min_token is None or max_token is None:
        return None
    return min_token, max_token


def _build_random_eval_dataset(
    *,
    count: int,
    dataset_overrides: dict[str, Any],
    dataset_artifacts: DatasetArtifacts,
    task: str,
    cont_len: int | None,
    seed: int,
    single_query: bool = False,
) -> EvaluationDataset:
    if count <= 0:
        raise SystemExit("--eval-random-synthetic requires a positive question count.")
    seq_len_min = int(dataset_overrides.get("seq_len_min") or dataset_overrides.get("seq_len") or 16)
    seq_len_max = int(dataset_overrides.get("seq_len_max") or dataset_overrides.get("seq_len") or seq_len_min)
    if seq_len_min > seq_len_max:
        seq_len_min, seq_len_max = seq_len_max, seq_len_min
    if single_query:
        unique_sequences = 1
    else:
        unique_sequences = int(dataset_overrides.get("unique_sequences") or dataset_overrides.get("num_sequences") or 8)
    bounds = _infer_stream_token_bounds(dataset_artifacts)
    if bounds is None:
        max_stream_token = max(1, dataset_artifacts.pad_id - 1)
        token_offset = max(0, int(dataset_overrides.get("token_offset", 0)))
    else:
        token_offset, max_stream_token = bounds
    token_offset = max(0, token_offset)
    max_stream_token = max(token_offset, max_stream_token)
    vocab_span = max_stream_token - token_offset + 1
    factory = SyntheticSampleFactory(
        seq_len_range=(max(1, seq_len_min), max(1, seq_len_max)),
        unique_sequences=max(1, unique_sequences),
        vocab_size=max(2, vocab_span),
        seed=seed,
        task=task,
        cont_len=int(cont_len or dataset_overrides.get("cont_len", 0) or 0),
        token_offset=token_offset,
    )
    samples, pad_id, vocab_size, max_input_len = factory.build_batch(count)
    eval_samples = []
    for idx, (seq, labels, stream_len_tensor) in enumerate(samples):
        metadata = {
            "video_index": idx,
            "video_uid": f"synthetic-{idx}",
            "length_value": float(stream_len_tensor.item()),
            "bucket_id": f"synthetic-{idx}",
        }
        extras = {"metadata": metadata}
        eval_samples.append((seq, labels, stream_len_tensor, extras))
    summary = {
        "task": task,
        "question_count": len(eval_samples),
        "source": "synthetic",
        "single_query_mode": bool(single_query),
    }
    return EvaluationDataset(
        eval_samples,
        pad_id=pad_id,
        vocab_size=vocab_size,
        max_input_len=max_input_len,
        metadata_summary=summary,
    )


def _write_benchmark_profiles(
    args: argparse.Namespace,
    *,
    backend: str,
    profiles: list[StreamingProfile],
) -> None:
    if not profiles:
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"benchmark-{backend}"
    output_dir = args.benchmark_profile_output / run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [profile.__dict__ for profile in profiles]
    json_path = output_dir / "streaming_profile.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = output_dir / "streaming_profile.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[benchmark:{backend}] wrote streaming profile JSON to {json_path}")
    print(f"[benchmark:{backend}] wrote streaming profile CSV to {csv_path}")
    for row in rows:
        length = row["sequence_length"]
        percep = row["perception_flops"] / 1e6 if row["perception_flops"] else 0.0
        query = row["query_flops"] / 1e6 if row["query_flops"] else 0.0
        total = row["total_flops"] / 1e6 if row["total_flops"] else 0.0
        percep_avg = row["perception_flops_per_token"] / 1e6 if row["perception_flops_per_token"] else 0.0
        percep_lat = row["perception_latency_ms"]
        percep_lat_avg = row["perception_latency_per_token_ms"]
        print(
            f"  L={length:<5} Σc_U={percep:.3f} MFLOPs (avg {percep_avg:.3f} MFLOPs/token) "
            f"c_R={query:.3f} MFLOPs total={total:.3f} MFLOPs ℓ_U={percep_lat:.2f}ms "
            f"(avg {percep_lat_avg:.2f}ms/token) ℓ_R={row['query_latency_ms']:.2f}ms"
        )


def _build_run_name(
    args: argparse.Namespace,
    *,
    dataset_artifacts: DatasetArtifacts,
    dataset_overrides: dict[str, Any],
    task: str,
) -> str:
    seq_len_display = _describe_sequence_length(
        dataset_overrides,
        dataset_artifacts.max_seq_len,
    )
    vocab_size = _resolve_dataset_stat(
        dataset_overrides,
        "vocab_size",
        dataset_artifacts.vocab_size,
    )
    parts = [
        args.backend,
        args.dataset,
        task,
        f"seq{seq_len_display}",
        f"v{vocab_size}",
        f"d{args.decoder_d_model}",
        f"seed{args.seed}",
    ]
    return "-".join(parts)


def _serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        config[key] = _serialize_value(value)
    return config


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, argparse.Namespace):
        return _serialize_config(value)
    return value


def _serialize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in overrides.items()}


def _format_override_entries(overrides: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for key in sorted(overrides):
        value = overrides[key]
        entries.append(f"{key}={value}")
    return entries


def _write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _resolve_eval_device(spec: str) -> torch.device:
    normalized = spec.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _result_row(
    result,
    *,
    task: str,
) -> dict[str, Any]:
    metadata = dict(result.metadata)
    uncertain_prob = _compute_uncertain_probability(result)
    return {
        "metadata": metadata,
        "truth_token": result.truth_token,
        "pred_token": result.pred_token,
        "truth_label": describe_label(result.truth_token),
        "pred_label": describe_label(result.pred_token),
        "truth_prob": result.truth_prob,
        "pred_prob": result.pred_prob,
        "correct": result.correct,
        "uncertain_prob": uncertain_prob,
    }


def _compute_uncertain_probability(
    result,
) -> float | None:
    index = _label_index(UNCERTAIN_TOKEN)
    if index is None:
        return None
    return _token_probability(result.logits, index)


def _token_probability(logits: Sequence[float], class_index: int | None) -> float | None:
    if class_index is None or class_index < 0:
        return None
    if not logits or class_index >= len(logits):
        return None
    max_logit = max(logits)
    if math.isinf(max_logit) and max_logit > 0:
        # fallback to zero when overflowing
        return None
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_values)
    if total == 0.0:
        return None
    return exp_values[class_index] / total


def _label_index(token: int) -> int | None:
    try:
        return LABEL_TOKENS.index(token)
    except ValueError:
        return None


def _write_per_question_csv(
    results,
    path: Path,
    *,
    task: str,
) -> None:
    fieldnames = [
        "video_index",
        "question_index",
        "bucket_id",
        "video_bucket_id",
        "stream_prefix_length",
        "stream_total_length",
        "entropy_prefix",
        "video_entropy_value",
        "video_length_value",
        "truth_token",
        "truth_label",
        "pred_token",
        "pred_label",
        "correct",
        "truth_prob",
        "pred_prob",
        "uncertain_prob",
        "question_time",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metadata = dict(result.metadata or {})
            video_meta = metadata.get("video") or {}
            row = {
                "video_index": metadata.get("video_index", video_meta.get("video_index")),
                "question_index": metadata.get("question_index"),
                "bucket_id": metadata.get("bucket_id"),
                "video_bucket_id": video_meta.get("bucket_id"),
                "stream_prefix_length": metadata.get("stream_prefix_length"),
                "stream_total_length": metadata.get("stream_total_length"),
                "entropy_prefix": metadata.get("entropy_prefix"),
                "video_entropy_value": video_meta.get("entropy_value"),
                "video_length_value": video_meta.get("length_value"),
                "truth_token": result.truth_token,
                "truth_label": describe_label(result.truth_token),
                "pred_token": result.pred_token,
                "pred_label": describe_label(result.pred_token),
                "correct": result.correct,
                "truth_prob": result.truth_prob,
                "pred_prob": result.pred_prob,
                "uncertain_prob": _compute_uncertain_probability(result),
                "question_time": metadata.get("question_time"),
            }
            writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the MemoryBankDecoder with configurable backends and datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval", "benchmark"],
        default="train",
        help="Select 'train', 'eval', or 'benchmark' to profile an architecture without training.",
    )

    backend_group = parser.add_argument_group("Backend Selection")
    backend_group.add_argument(
        "--backend",
        choices=list(backend_choices()),
        required=False,
        help="Registered memory backend to attach to the decoder.",
    )
    backend_group.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override backend constructor kwargs (repeat to supply multiple overrides).",
    )

    dataset_group = parser.add_argument_group("Dataset Selection")
    dataset_group.add_argument(
        "--dataset",
        choices=list(dataset_choices()),
        required=False,
        help="Data source to build continuation queries from (synthetic/file/video_features).",
    )
    dataset_group.add_argument(
        "--dataset-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override dataset-specific knobs such as seq_len or manifest path.",
    )
    dataset_group.add_argument(
        "--feature-input-dim",
        type=int,
        default=None,
        help=(
            "Dimension of precomputed video embeddings; when set, the decoder adds a trainable "
            "projection into decoder space so video features can differ from decoder_d_model."
        ),
    )
    dataset_group.add_argument(
        "--augment-with-synthetic",
        type=float,
        default=0.0,
        metavar="RATIO",
        help=(
            "Inject synthetic streams into the training batches when --dataset=file. "
            "RATIO expresses synthetic_per_real (0 disables augmentation)."
        ),
    )

    curriculum_group = parser.add_argument_group("Curriculum Learning")
    curriculum_group.add_argument(
        "--curriculum-start",
        type=int,
        default=None,
        help=(
            "Enable curriculum training by seeding the first stage with streams whose lengths are "
            "<= this value. Length bounds double every time the target accuracy is met."
        ),
    )
    curriculum_group.add_argument(
        "--curriculum-target-acc",
        type=float,
        default=None,
        help=(
            "Training accuracy threshold that triggers an expansion to the next curriculum stage. "
            "Only applied when --curriculum-start is also set."
        ),
    )

    benchmark_group = parser.add_argument_group("Benchmarking")
    benchmark_group.add_argument(
        "--benchmark-seq-lens",
        default="",
        help="Comma-separated sequence lengths to profile when --mode=benchmark (default: dataset max).",
    )
    benchmark_group.add_argument(
        "--benchmark-batch-size",
        type=int,
        default=1,
        help="Batch size used when measuring latency/FLOPs in benchmark mode.",
    )
    benchmark_group.add_argument(
        "--benchmark-repeat",
        type=int,
        default=10,
        help="Number of timed iterations per sequence length when benchmarking.",
    )
    benchmark_group.add_argument(
        "--benchmark-warmup",
        type=int,
        default=3,
        help="Warmup runs discarded before collecting latency measurements (default: 3).",
    )
    benchmark_group.add_argument(
        "--benchmark-device",
        default="auto",
        help="Device spec for benchmarking (auto/cpu/cuda).",
    )
    benchmark_group.add_argument(
        "--benchmark-query-length",
        type=int,
        default=None,
        help="Override synthetic query length during benchmarking (defaults to 16 tokens).",
    )
    benchmark_group.add_argument(
        "--benchmark-flops-only",
        action="store_true",
        help="Skip latency measurements and emit a single FLOPs sample per sequence length.",
    )
    benchmark_group.add_argument(
        "--benchmark-latency-only",
        action="store_true",
        help="Skip FLOP profiling and only collect latency metrics in benchmark mode.",
    )
    benchmark_group.add_argument(
        "--benchmark-profile",
        action="store_true",
        help="Collect streaming perception/query metrics for selected lengths in benchmark mode.",
    )
    benchmark_group.add_argument(
        "--benchmark-profile-lengths",
        default="256,1024,4096",
        help="Comma-separated lengths used for streaming profiling (ignored unless --benchmark-profile is set).",
    )
    benchmark_group.add_argument(
        "--benchmark-profile-max-repeat",
        type=int,
        default=10,
        help="Maximum per-metric samples captured during streaming profiling windows (first half FLOPs, second half latency).",
    )
    benchmark_group.add_argument(
        "--benchmark-profile-output",
        type=Path,
        default=Path("artifacts/logs/benchmark_profiles"),
        help="Directory where profiling JSON/CSV logs are written.",
    )

    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument("--batch-size", type=int, default=8, help="Batch size per optimization step.")
    data_group.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for validation (0 disables splitting).",
    )
    data_group.add_argument(
        "--val-set-percent",
        type=float,
        default=None,
        help="Percent of samples reserved for validation (mirrors --val-fraction; overrides it when set).",
    )
    data_group.add_argument("--num-workers", type=int, default=0, help="Dataloader worker processes.")
    data_group.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned-memory loading for faster host-to-device transfers.",
    )

    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument("--max-epochs", type=int, default=1000, help="Training epochs for Lightning Trainer.")
    optim_group.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    optim_group.add_argument("--weight-decay", type=float, default=0.15, help="AdamW weight decay.")
    optim_group.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value.")
    optim_group.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Gradient accumulation steps for large effective batch sizes.",
    )
    optim_group.add_argument(
        "--lr-scheduler",
        choices=("cosine_restart", "cosine_epoch", "constant"),
        default="cosine_restart",
        help=(
            "Learning-rate scheduler to pair with AdamW. 'cosine_restart' matches the existing "
            "per-epoch warm restarts, 'cosine_epoch' runs a single cosine decay across the run, "
            "and 'constant' holds the LR fixed aside from optional warmup."
        ),
    )
    optim_group.add_argument(
        "--lr-warmup",
        type=float,
        default=0.0,
        help=(
            "Optional learning-rate warmup. Values between 0 and 1 express the fraction of the"
            " first epoch used for a smooth ramp (default: 0 disables warmup). Values >=1 keep"
            " the legacy behavior of warming up for that many full epochs."
        ),
    )
    optim_group.add_argument(
        "--warmup-first-epoch",
        action="store_true",
        help=(
            "Linearly ramp the learning rate from zero to the target value over the first"
            " full training epoch (overrides --lr-warmup when set)."
        ),
    )
    optim_group.add_argument("--seed", type=int, default=0, help="Master seed used for Lightning + PyTorch.")

    logging_group = parser.add_argument_group("Logging & Progress")
    logging_group.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/logs/runs"),
        help="Directory where CSV logs (and optional WandB files) are stored.",
    )
    logging_group.add_argument(
        "--progress-refresh-rate",
        type=int,
        default=1,
        help="Number of steps between TQDM updates (higher = calmer progress bar).",
    )
    logging_group.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Opt out of the default WandB logger (CSV logging remains enabled).",
    )
    logging_group.add_argument(
        "--wandb-project",
        default="memory-streaming-backends",
        help="WandB project name.",
    )
    logging_group.add_argument(
        "--run-name",
        default=None,
        help=(
            "Override the auto-generated run name used for checkpoint/log directories; "
            "also becomes the default WandB run name unless --wandb-run-name is provided."
        ),
    )
    logging_group.add_argument("--wandb-run-name", default=None, help="Custom run name (default derived from config).")
    logging_group.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        help="Tag(s) applied to the WandB run; pass multiple times for several tags.",
    )
    logging_group.add_argument("--wandb-mode", default="online", help="WandB mode: online/offline/disabled.")
    logging_group.add_argument("--wandb-dir", type=Path, default=None, help="Explicit directory for WandB artifacts.")
    logging_group.add_argument(
        "--wandb-log-note",
        default=None,
        metavar="TEXT",
        help="Optional string appended to the WandB summary/log feed and run name at startup.",
    )
    logging_group.add_argument(
        "--log-sample-queries",
        type=int,
        default=0,
        help="Number of validation queries to pretty-print for debugging (0 disables).",
    )
    logging_group.add_argument(
        "--log-grad-component",
        action="append",
        default=[],
        metavar="[LABEL=]PREFIX",
        help=(
            "Track gradient L2 norms for module name prefixes (repeatable). "
            "Use label=prefix to override the WandB metric suffix; entries default to their prefix."
        ),
    )

    early_group = parser.add_argument_group("Early Stopping")
    early_group.add_argument(
        "--early-stop-acc",
        type=float,
        default=0.0,
        help="Stop once validation accuracy reaches this threshold (<=0 disables).",
    )
    early_group.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Patience in epochs once the threshold is reached (>0). Set <=0 to disable patience so training only stops when the accuracy threshold is met or max epochs are exhausted.",
    )

    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Persist best and last checkpoints under the checkpoint directory.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Root directory where run configs and checkpoints are stored.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-monitor",
        default="val_loss",
        help="Metric name to monitor when selecting the best checkpoint.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-mode",
        choices=["min", "max"],
        default="min",
        help="Whether lower (min) or higher (max) values win for the monitor metric.",
    )

    runtime_group = parser.add_argument_group("Runtime Controls")
    runtime_group.add_argument("--accelerator", default="auto", help="Lightning accelerator string (cpu/gpu/auto).")
    runtime_group.add_argument("--devices", default="auto", help="Device spec (auto, integer, or comma list).")
    runtime_group.add_argument("--precision", default="32-true", help="Mixed-precision policy string.")
    runtime_group.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic operations when supported by the backend.",
    )
    runtime_group.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction or count of training batches to run each epoch.",
    )
    runtime_group.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction or count of validation batches to run each epoch.",
    )
    runtime_group.add_argument(
        "--val-check-interval",
        type=float,
        default=1.0,
        help="Interval (epochs or fraction of epoch) between validation runs during training.",
    )
    runtime_group.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Frequency (steps) for Lightning's internal logging hooks.",
    )
    runtime_group.add_argument(
        "--temp-root",
        type=Path,
        default=Path(os.environ.get("QA_TEMP_ROOT", tempfile.gettempdir())) / "calibrated-temp",
        help=(
            "Base directory where per-run temp folders (TMPDIR) are created. "
            "Defaults to $QA_TEMP_ROOT/calibrated-temp if set, else "
            "<system-tmp>/calibrated-temp (e.g. /tmp/calibrated-temp)."
        ),
    )
    runtime_group.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=0,
        help="Number of validation sanity batches to run before the first epoch.",
    )
    runtime_group.add_argument("--resume-from", type=Path, default=None, help="Checkpoint file to resume from (Lightning format).")

    init_group = parser.add_argument_group("Training Initialization")
    init_group.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Existing run directory whose config + checkpoint should seed a new training run.",
    )
    init_group.add_argument(
        "--init-checkpoint",
        default="best.ckpt",
        help="Checkpoint filename under --init-from to load before training begins.",
    )
    init_group.add_argument(
        "--init-load-optimizer",
        action="store_true",
        help="When set, resume optimizer/scheduler state from the checkpoint instead of reinitializing them.",
    )

    eval_group = parser.add_argument_group("Evaluation Mode")
    eval_group.add_argument("--eval-run-dir", type=Path, help="Checkpoint run directory containing config.json.")
    eval_group.add_argument(
        "--eval-checkpoint-name",
        default="best.ckpt",
        help="Checkpoint filename inside --eval-run-dir to load before evaluation.",
    )
    eval_group.add_argument("--eval-device", default="auto", help="Device spec for evaluation (cpu, cuda, auto).")
    eval_group.add_argument("--eval-manifest", type=Path, help="Path to the questions manifest JSON for evaluation.")
    eval_group.add_argument(
        "--eval-manifest-root",
        type=Path,
        default=None,
        help="Optional base directory for resolving relative video/clip paths in the eval manifest.",
    )
    eval_group.add_argument(
        "--eval-synthetic-single-query",
        action="store_true",
        help="When using --eval-random-synthetic, force each sampled stream to contain exactly one query.",
    )
    eval_group.add_argument(
        "--eval-random-synthetic",
        type=int,
        default=0,
        help=(
            "If > 0, bypass the manifest and evaluate on this many randomly generated synthetic questions "
            "based on the training dataset configuration."
        ),
    )
    eval_group.add_argument(
        "--eval-synth-seed",
        type=int,
        default=0,
        help="RNG seed used when --eval-random-synthetic is enabled (defaults to --seed).",
    )
    eval_group.add_argument("--eval-sequence-key", default="", help="Optional manifest sequence key override.")
    eval_group.add_argument("--eval-num-videos", type=int, default=-1, help="Limit videos loaded from the manifest (<=0 keeps all).")
    eval_group.add_argument("--eval-truncate-len", type=int, default=0, help="Truncate each stream to this many tokens (<=0 keeps full length).")
    eval_group.add_argument(
        "--eval-token-offset",
        type=int,
        default=None,
        help="Explicit token offset overriding the manifest default (falls back to training config).",
    )
    eval_group.add_argument("--eval-task", default=None, help="Task name to use for evaluation (defaults to the trained dataset task).")
    eval_group.add_argument(
        "--eval-cont-len",
        type=int,
        default=None,
        help="Continuation length for continuation manifests (fallback: trained dataset override).",
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size used when running evaluation forward passes.",
    )
    eval_group.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("artifacts/logs/eval"),
        help="Directory where evaluation summaries and per-question logs are stored.",
    )
    eval_group.add_argument(
        "--eval-write-csv",
        action="store_true",
        help="Emit a per-question CSV alongside per_question.jsonl for downstream analysis.",
    )

    decoder_group = parser.add_argument_group("Decoder Architecture")
    decoder_group.add_argument(
        "--decoder-d-model",
        type=int,
        default=DECODER_D_MODEL_DEFAULT,
        help="Decoder embedding dimension.",
    )
    decoder_group.add_argument(
        "--decoder-nhead", type=int, default=DECODER_TRANSFORMER_DEFAULTS["decoder_nhead"], help="Number of attention heads per block."
    )
    decoder_group.add_argument(
        "--decoder-num-layers", type=int, default=DECODER_TRANSFORMER_DEFAULTS["decoder_num_layers"], help="Number of decoder transformer layers."
    )
    decoder_group.add_argument(
        "--decoder-mlp-ratio", type=int, default=DECODER_TRANSFORMER_DEFAULTS["decoder_mlp_ratio"], help="FFN hidden size multiplier."
    )
    decoder_group.add_argument(
        "--decoder-attn-dropout", type=float, default=DECODER_TRANSFORMER_DEFAULTS["decoder_attn_dropout"], help="Attention dropout probability."
    )
    decoder_group.add_argument("--decoder-embed-dropout", type=float, default=0.1, help="Token embedding dropout probability.")
    decoder_group.add_argument(
        "--decoder-resid-dropout", type=float, default=DECODER_TRANSFORMER_DEFAULTS["decoder_resid_dropout"], help="Residual dropout probability."
    )
    decoder_group.add_argument(
        "--decoder-rotary-base", type=float, default=DECODER_TRANSFORMER_DEFAULTS["decoder_rotary_base"], help="Base used for rotary embeddings."
    )
    decoder_group.add_argument(
        "--decoder-context-cap",
        type=int,
        default=0,
        help="Max decoder context length after concatenating backend memory + query tokens (0 uses dataset maximum).",
    )
    decoder_group.add_argument(
        "--loss-type",
        choices=("cross_entropy", "deep_gambler"),
        default="cross_entropy",
        help="Primary supervision objective for query tokens.",
    )
    decoder_group.add_argument(
        "--deep-gambler-mode",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="Use a fixed wager o or recompute it per batch when Deep Gambler loss is enabled.",
    )
    decoder_group.add_argument(
        "--deep-gambler-o",
        type=float,
        default=1.5,
        help="Baseline wager multiplier o for Deep Gambler loss (used directly when mode=fixed).",
    )
    decoder_group.add_argument(
        "--deep-gambler-eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon applied inside log() for Deep Gambler loss.",
    )
    decoder_group.add_argument(
        "--deep-gambler-activation-acc",
        type=float,
        default=0.33,
        help="Minimum batch accuracy (non-UNK) required before adaptive wagers override the base value.",
    )
    return parser


def _sanitize_backend_overrides(
    overrides: dict[str, Any], *, backend_name: str
) -> dict[str, Any]:
    """Force every backend into the unified direct-mode configuration."""

    sanitized = dict(overrides)
    spec = BACKEND_SPECS.get(backend_name)
    supports_memory_mode = bool(spec and ("memory_mode" in spec.defaults or "memory_mode" in spec.required_keys))
    if backend_name == "stm":
        if "memory_mode" in sanitized:
            raise ValueError("STM backend does not accept memory_mode overrides.")
        if "num_slots" in sanitized:
            raise ValueError("STM backend ignores num_slots; drop the flag.")
        return sanitized
    if supports_memory_mode:
        memory_mode = sanitized.get("memory_mode")
        if memory_mode is not None and memory_mode != "hidden_state":
            raise ValueError(
                f"Backend '{backend_name}' cannot run with memory_mode={memory_mode!r}; use the implicit hidden-state path instead."
            )
        sanitized["memory_mode"] = "hidden_state"
    if "num_slots" in sanitized:
        raise ValueError(
            f"Backend '{backend_name}' ignores --backend-option num_slots; drop the flag to continue."
        )
    return sanitized


def _align_decoder_dim_with_backend(
    args: argparse.Namespace,
    backend: MemoryBackend,
    backend_config: dict[str, Any],
) -> None:
    requires_embeddings = getattr(backend, "requires_token_embeddings", True)
    projects_to_decoder = getattr(backend, "projects_to_decoder_dim", False)
    if not requires_embeddings and not projects_to_decoder:
        return
    embed_value = backend_config.get("embed_dim")
    if embed_value is None:
        return
    try:
        embed_dim = int(embed_value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return
    if args.decoder_d_model != embed_dim:
        args.decoder_d_model = embed_dim


def _maybe_assign_backend_vocab(
    backend_name: str,
    overrides: dict[str, Any],
    dataset_vocab_size: int,
) -> None:
    if backend_name not in _FLA_BACKENDS:
        return
    overrides["vocab_size"] = int(dataset_vocab_size)


def _validate_backend_embedding(
    backend: MemoryBackend,
    backend_config: dict[str, Any],
    decoder_dim: int,
) -> None:
    if not getattr(backend, "requires_token_embeddings", True):
        return
    requested = backend_config.get("embed_dim")
    if requested is None:
        return
    if int(requested) != decoder_dim:
        raise ValueError(
            "Backends that consume token embeddings must share decoder_d_model; "
            f"got embed_dim={requested} vs decoder_d_model={decoder_dim}."
        )


def _build_memory_backend(
    args: argparse.Namespace,
    overrides: dict[str, Any],
    *,
    min_context: int | None = None,
) -> tuple[MemoryBackend, dict[str, Any]]:
    overrides = dict(overrides)
    fla_backends = {"deltanet", "gated_deltanet", "deltaformer", "mom"}
    if args.backend in fla_backends and "autocast_dtype" not in overrides:
        precision = str(getattr(args, "precision", "")).lower()
        if "bf16" in precision:
            overrides["autocast_dtype"] = torch.bfloat16
        elif "16" in precision:
            overrides["autocast_dtype"] = torch.float16
        elif "32" in precision:
            overrides["autocast_dtype"] = None
        elif args.backend in {"deltanet", "gated_deltanet", "mom"}:
            overrides["autocast_dtype"] = torch.bfloat16
    overrides = ensure_backend_context_capacity(args.backend, overrides, min_context)
    try:
        return build_backend(args.backend, overrides)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to build backend: {exc}") from exc


def _build_dataset(args: argparse.Namespace, overrides: dict[str, Any]) -> DatasetArtifacts:
    try:
        return build_dataset(args.dataset, overrides)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to build dataset: {exc}") from exc


def _extract_dataset_lengths(dataset: Any) -> list[int]:
    metadata = getattr(dataset, "sample_metadata", None)
    if not metadata:
        return []
    lengths: list[int] = []
    for entry in metadata:
        if not entry:
            continue
        value = entry.get("stream_length")
        if value is None:
            value = entry.get("length_value")
        if value is None:
            continue
        try:
            lengths.append(int(value))
        except (TypeError, ValueError):
            continue
    return lengths


def _extract_sample_lengths(samples: Sequence[tuple[Any, Any, Any]]) -> list[int]:
    lengths: list[int] = []
    for sample in samples:
        if len(sample) < 3:
            continue
        tensor = sample[2]
        try:
            lengths.append(int(tensor.item()))
        except AttributeError:
            try:
                lengths.append(int(tensor))
            except (TypeError, ValueError):
                continue
    return lengths


def _synthetic_max_input_len(samples: Sequence[tuple[Any, Any, Any]]) -> int:
    max_len = 0
    for sample in samples or []:
        if not sample:
            continue
        seq = sample[0]
        length = 0
        if hasattr(seq, "numel"):
            try:
                length = int(seq.numel())
            except (TypeError, ValueError):
                length = 0
        if length == 0:
            try:
                length = len(seq)
            except TypeError:
                length = 0
        max_len = max(max_len, length)
    return max_len


def _compute_length_tertiles(lengths: Sequence[int]) -> list[float] | None:
    data = sorted(float(v) for v in lengths if v is not None)
    if len(data) < 3:
        return None
    def _quantile(fraction: float) -> float:
        if len(data) == 1:
            return data[0]
        position = fraction * (len(data) - 1)
        lower = int(math.floor(position))
        upper = min(len(data) - 1, lower + 1)
        if lower == upper:
            return data[lower]
        weight = position - lower
        return data[lower] + (data[upper] - data[lower]) * weight

    return [_quantile(1.0 / 3.0), _quantile(2.0 / 3.0)]


def _build_decoder(
    args: argparse.Namespace,
    dataset_artifacts: DatasetArtifacts,
    backend: MemoryBackend,
    max_seq_len: int,
    *,
    task: str,
    feature_input_dim: int | None,
    lr_warmup_epochs: int,
    lr_scheduler_mode: str,
    warmup_first_epoch_fraction: float | None,
) -> MemoryBankDecoder:
    grad_component_specs = _parse_grad_component_specs(getattr(args, "log_grad_component", []))
    for label, _ in grad_component_specs:
        DEFAULT_REGISTRY.register(f"gradients/{label}")
    decoder = MemoryBankDecoder(
        vocab_size=dataset_artifacts.vocab_size,
        pad_id=dataset_artifacts.pad_id,
        max_seq_len=max_seq_len,
        d_model=args.decoder_d_model,
        nhead=args.decoder_nhead,
        num_layers=args.decoder_num_layers,
        mlp_ratio=args.decoder_mlp_ratio,
        lr=args.learning_rate,
        attn_dropout=args.decoder_attn_dropout,
        embed_dropout=args.decoder_embed_dropout,
        resid_dropout=args.decoder_resid_dropout,
        rotary_base=args.decoder_rotary_base,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        memory_backend=backend,
        task=task,
        log_sample_queries=args.log_sample_queries,
        loss_type=args.loss_type,
        deep_gambler_mode=args.deep_gambler_mode,
        deep_gambler_o=args.deep_gambler_o,
        deep_gambler_epsilon=args.deep_gambler_eps,
        deep_gambler_activation_acc=args.deep_gambler_activation_acc,
        grad_component_specs=grad_component_specs,
        feature_input_dim=feature_input_dim,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_scheduler_mode=lr_scheduler_mode,
        warmup_first_epoch_fraction=warmup_first_epoch_fraction,
    )
    decoder.set_dataset_metadata(dataset_artifacts.metadata)
    return decoder


def _run_eval_mode(args: argparse.Namespace) -> None:
    if args.eval_run_dir is None:
        raise SystemExit("--eval-run-dir must be provided when --mode=eval")
    synthetic_eval = int(args.eval_random_synthetic or 0)
    use_synthetic_eval = synthetic_eval > 0
    if not use_synthetic_eval and args.eval_manifest is None:
        raise SystemExit(
            "--eval-manifest must be provided when --mode=eval unless --eval-random-synthetic is set"
        )

    metadata = load_run_metadata(args.eval_run_dir)
    run_name = metadata.get("run_name") or args.eval_run_dir.name
    device = _resolve_eval_device(args.eval_device)
    model, dataset_overrides, dataset_artifacts, _ = instantiate_model_from_run(
        args.eval_run_dir,
        args.eval_checkpoint_name,
        device,
        metadata=metadata,
    )

    base_task = str(dataset_overrides.get("task", "membership"))
    task = args.eval_task or base_task

    cont_len = args.eval_cont_len
    if cont_len is None:
        cont_override = dataset_overrides.get("cont_len")
        if cont_override is not None:
            cont_len = int(cont_override)
        else:
            derived_cont = getattr(dataset_artifacts.dataset, "cont_len", None)
            if derived_cont is not None:
                cont_len = int(derived_cont)
    if task == "continuation" and (cont_len is None or cont_len <= 0):
        raise SystemExit("Continuation evaluation requires cont_len > 0; set --eval-cont-len explicitly.")
    token_offset = args.eval_token_offset
    if token_offset is None:
        token_offset = int(dataset_overrides.get("token_offset", TOKEN_OFFSET))
    sequence_key = args.eval_sequence_key.strip() or dataset_overrides.get("sequence_key")
    sequence_key = sequence_key or None

    if use_synthetic_eval:
        dataset = _build_random_eval_dataset(
            count=synthetic_eval,
            dataset_overrides=dataset_overrides,
            dataset_artifacts=dataset_artifacts,
            task=task,
            cont_len=cont_len,
            seed=args.eval_synth_seed if args.eval_synth_seed else args.seed,
            single_query=args.eval_synthetic_single_query,
        )
    else:
        eval_config = EvaluationConfig(
            manifest_path=args.eval_manifest,
            sequence_key=sequence_key,
            num_videos=args.eval_num_videos,
            truncate_len=args.eval_truncate_len,
            task=task,
            cont_len=cont_len or 0,
            token_offset=token_offset,
            manifest_root=args.eval_manifest_root,
        )
        dataset = build_evaluation_dataset(eval_config)
    metadata_summary = dataset.metadata_summary
    collate_fn = build_collate(dataset.pad_id)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.eval_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )

    runner = EvaluationRunner(
        model,
        device=device,
        task=task,
    )
    results = runner.run(loader)
    entropy_bounds = _metadata_bounds(
        metadata_summary,
        ["entropy_prefix_tertiles", "video_entropy_tertiles"],
    )
    length_bounds = _metadata_bounds(
        metadata_summary,
        ["prefix_length_tertiles", "stream_length_tertiles", "target_length_tertiles"],
    )
    aggregator = StatsAggregator(
        task=task,
        entropy_boundaries=entropy_bounds,
        length_boundaries=length_bounds,
    )
    for result in results:
        aggregator.add(result)
    summary = aggregator.finalize()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.eval_output_dir / run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    per_question_path = output_dir / "per_question.jsonl"
    with per_question_path.open("w", encoding="utf-8") as handle:
        for result in results:
            row = _result_row(result, task=task)
            json.dump(row, handle)
            handle.write("\n")
    csv_path = None
    if args.eval_write_csv:
        csv_path = output_dir / "per_question.csv"
        _write_per_question_csv(
            results,
            csv_path,
            task=task,
        )

    summary_payload = {
        "run_name": run_name,
        "checkpoint": str((args.eval_run_dir / args.eval_checkpoint_name).resolve()),
        "dataset_metadata": dataset.metadata_summary,
        "evaluation_config": {
            "task": task,
            "cont_len": cont_len,
            "device": str(device),
            "manifest": str(args.eval_manifest) if args.eval_manifest else None,
            "manifest_root": str(args.eval_manifest_root) if args.eval_manifest_root else None,
            "synthetic_questions": synthetic_eval if use_synthetic_eval else 0,
            "synthetic_seed": args.eval_synth_seed if use_synthetic_eval else None,
        },
        "stats": summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    cov_display = _format_percentage(summary.get("coverage"))
    acc_ans_display = _format_percentage(summary.get("accuracy_when_answering"))
    ua_display = _format_percentage(summary.get("useful_answer_rate"))
    tar_display = _format_percentage(summary.get("tar"))
    far_display = _format_percentage(summary.get("far"))
    print(
        f"[eval:{run_name}] questions={summary['total_questions']} accuracy={summary['accuracy']*100:.2f}% "
        f"coverage={cov_display} acc_when_answering={acc_ans_display} ua={ua_display} tar={tar_display} far={far_display}"
    )
    print(
        f"[eval:{run_name}] uncertain_miss={summary['uncertain_miss_rate']} "
        f"false_uncertain={summary['false_uncertain_rate']}"
    )
    if entropy_bounds:
        print(f"[eval:{run_name}] entropy tertiles={entropy_bounds}")
    if length_bounds:
        print(f"[eval:{run_name}] length tertiles={length_bounds}")
    print(f"[eval:{run_name}] wrote per-question rows to {per_question_path}")
    if csv_path is not None:
        print(f"[eval:{run_name}] wrote per-question CSV to {csv_path}")


def _run_benchmark_mode(
    args: argparse.Namespace,
    *,
    dataset_artifacts: DatasetArtifacts,
    model: MemoryBankDecoder,
    task: str,
) -> None:
    if args.benchmark_flops_only and args.benchmark_latency_only:
        raise SystemExit("--benchmark-flops-only and --benchmark-latency-only cannot be combined")
    seq_lens = _parse_sequence_lengths(args.benchmark_seq_lens, dataset_artifacts.max_seq_len)
    config = BenchmarkConfig(
        sequence_lengths=seq_lens,
        batch_size=max(1, args.benchmark_batch_size),
        repeat=max(1, args.benchmark_repeat),
        warmup=max(0, args.benchmark_warmup),
        device=args.benchmark_device,
        query_length=args.benchmark_query_length,
        flops_only=bool(args.benchmark_flops_only),
        latency_only=bool(args.benchmark_latency_only),
    )
    runner = BenchmarkRunner(
        model,
        pad_id=dataset_artifacts.pad_id,
        vocab_size=dataset_artifacts.vocab_size,
        task=task,
        config=config,
    )
    results = runner.run()
    print("Sequence Length | MFLOPs | Latency Mean (ms) | Latency P90 (ms)")
    for result in results:
        if result.flops:
            flops_display = f"{result.flops / 1e6:.8g} M"
        else:
            flops_display = "n/a"
        if math.isnan(result.latency_mean_ms):
            latency_mean = "n/a"
        else:
            latency_mean = f"{result.latency_mean_ms:.2f}"
        if math.isnan(result.latency_p90_ms):
            latency_p90 = "n/a"
        else:
            latency_p90 = f"{result.latency_p90_ms:.2f}"
        print(
            f"{result.sequence_length:>15} | {flops_display:>12} | "
            f"{latency_mean:>16} | {latency_p90:>15}"
        )
    if args.benchmark_profile:
        profile_lengths = _parse_sequence_lengths(
            args.benchmark_profile_lengths,
            dataset_artifacts.max_seq_len,
        )
        if not profile_lengths:
            profile_lengths = [dataset_artifacts.max_seq_len]
        profiles = runner.profile(
            profile_lengths,
            bulk_repeat_limit=max(1, args.benchmark_profile_max_repeat),
        )
        if profiles:
            _write_benchmark_profiles(
                args,
                backend=args.backend,
                profiles=profiles,
            )

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)
    cli_dataset_overrides = _parse_overrides(args.dataset_option)
    cli_backend_overrides = _parse_overrides(args.backend_option)
    init_context = _apply_training_initialization(
        args,
        argv_list,
        cli_dataset_overrides,
        cli_backend_overrides,
    )
    args.augment_with_synthetic = _validate_augmentation_args(args)

    if args.mode == "eval":
        _run_eval_mode(args)
        return

    if args.mode in {"train", "benchmark"} and args.backend is None and init_context is None:
        raise SystemExit("--backend is required unless --init-from is specified.")
    if args.mode in {"train", "benchmark"} and args.dataset is None and init_context is None:
        raise SystemExit("--dataset is required unless --init-from is specified.")
    if args.mode in {"train", "benchmark"}:
        _enforce_fla_device_policy(args)

    if args.val_set_percent is not None:
        if not 0.0 <= args.val_set_percent < 100.0:
            raise SystemExit("--val-set-percent must lie in [0, 100).")
        args.val_fraction = args.val_set_percent / 100.0
    else:
        args.val_set_percent = args.val_fraction * 100.0

    cleanup_stray_tempdirs(REPO_ROOT)
    warmup_epochs, warmup_first_epoch_fraction = _resolve_lr_warmup(
        args.lr_warmup, args.max_epochs
    )
    if args.warmup_first_epoch:
        if args.max_epochs <= 1:
            raise SystemExit("--warmup-first-epoch requires --max-epochs greater than 1.")
        warmup_epochs = 0
        warmup_first_epoch_fraction = 0.25
    args.lr_warmup_epochs = warmup_epochs
    args.warmup_first_epoch_fraction = warmup_first_epoch_fraction

    if init_context is not None:
        dataset_overrides = dict(init_context.dataset_overrides)
        backend_overrides = dict(init_context.backend_overrides)
        args.dataset_option = _format_override_entries(dataset_overrides)
        args.backend_option = _format_override_entries(backend_overrides)
    else:
        dataset_overrides = dict(cli_dataset_overrides)
        backend_overrides = dict(cli_backend_overrides)

    dataset_overrides.setdefault("task", "membership")
    dataset_artifacts = _build_dataset(args, dataset_overrides)
    task = str(dataset_overrides.get("task", "membership"))
    cont_len = int(dataset_overrides.get("cont_len", 3))
    feature_input_dim = getattr(args, "feature_input_dim", None)
    if feature_input_dim is not None and feature_input_dim <= 0:
        feature_input_dim = None
    metadata_embed_dim = dataset_artifacts.metadata.get("embed_dim") if dataset_artifacts.metadata else None
    if feature_input_dim is None and metadata_embed_dim is not None and args.dataset == "video_features":
        feature_input_dim = int(metadata_embed_dim)
    augmentation_factory_kwargs: dict[str, Any] | None = None
    synthetic_train_samples: list[tuple[Any, Any, Any]] | None = None
    synthetic_max_input = 0
    if args.augment_with_synthetic > 0.0:
        (
            augmentation_factory_kwargs,
            synthetic_train_samples,
        ) = _prepare_synthetic_augmentation(
            dataset_artifacts=dataset_artifacts,
            dataset_overrides=dataset_overrides,
            task=task,
            cont_len=cont_len,
            ratio=args.augment_with_synthetic,
            seed=args.seed,
        )
        synthetic_max_input = _synthetic_max_input_len(synthetic_train_samples)

    base_lengths = _extract_dataset_lengths(dataset_artifacts.dataset)
    synthetic_lengths = _extract_sample_lengths(synthetic_train_samples or [])
    combined_lengths = base_lengths + synthetic_lengths
    if combined_lengths:
        updated_metadata = dict(dataset_artifacts.metadata or {})
        length_tertiles = _compute_length_tertiles(combined_lengths)
        if length_tertiles is not None:
            updated_metadata["stream_length_tertiles"] = length_tertiles
        dataset_artifacts = replace(
            dataset_artifacts,
            metadata=updated_metadata,
        )
    if synthetic_max_input:
        combined_max = max(dataset_artifacts.max_seq_len, synthetic_max_input)
        if combined_max != dataset_artifacts.max_seq_len:
            dataset_artifacts = replace(
                dataset_artifacts,
                max_seq_len=combined_max,
            )

    curriculum_requested = (
        args.curriculum_start is not None or args.curriculum_target_acc is not None
    )
    if curriculum_requested and (
        args.curriculum_start is None or args.curriculum_target_acc is None
    ):
        raise SystemExit(
            "--curriculum-start and --curriculum-target-acc must be provided together."
        )

    curriculum_components = None
    if curriculum_requested:
        try:
            curriculum_components = build_curriculum_components(
                args=args,
                dataset_artifacts=dataset_artifacts,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                seed=args.seed,
                synthetic_train_samples=synthetic_train_samples or [],
                synthetic_val_builder=_build_synthetic_val_loader,
                synthetic_val_factory_kwargs=augmentation_factory_kwargs,
                length_overrides=combined_lengths,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if curriculum_components is None:
            print(
                "[curriculum] configuration produced a single stage; continuing without curriculum."
            )

    backend_overrides.setdefault("embed_dim", args.decoder_d_model)
    backend_overrides = _sanitize_backend_overrides(
        backend_overrides, backend_name=args.backend
    )
    _maybe_assign_backend_vocab(args.backend, backend_overrides, dataset_artifacts.vocab_size)

    backend, backend_config = _build_memory_backend(
        args,
        backend_overrides,
        min_context=dataset_artifacts.max_seq_len,
    )
    _align_decoder_dim_with_backend(args, backend, backend_config)
    _validate_backend_embedding(
        backend, backend_config, args.decoder_d_model
    )

    resume_checkpoint: str | None = None
    manual_checkpoint_path: Path | None = None
    if args.resume_from:
        resume_checkpoint = str(args.resume_from)
    elif init_context is not None:
        if init_context.load_optimizer_state:
            resume_checkpoint = str(init_context.checkpoint_path)
        else:
            manual_checkpoint_path = init_context.checkpoint_path

    auto_run_name = _build_run_name(
        args,
        dataset_artifacts=dataset_artifacts,
        dataset_overrides=dataset_overrides,
        task=task,
    )
    if init_context is not None and args.run_name is None:
        auto_run_name = f"{auto_run_name}-{init_context.run_suffix}"
    run_name = args.run_name or auto_run_name
    config_payload = _serialize_config(args)
    run_dir = args.checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "args": config_payload,
        "backend": args.backend,
        "dataset": args.dataset,
        "backend_overrides": _serialize_overrides(backend_overrides),
        "dataset_overrides": _serialize_overrides(dataset_overrides),
    }
    if curriculum_components is not None:
        metadata["curriculum"] = curriculum_components.summary
    if init_context is not None:
        metadata["provenance"] = {
            "parent_run_dir": str(init_context.run_dir.resolve()),
            "parent_run_name": init_context.parent_run_name,
            "init_checkpoint": init_context.checkpoint_path.name,
            "init_checkpoint_path": str(init_context.checkpoint_path.resolve()),
            "init_load_optimizer_state": init_context.load_optimizer_state,
        }
    _write_run_metadata(run_dir, metadata)

    configure_temp_directory(args.temp_root, run_name)

    max_seq_len = dataset_artifacts.max_seq_len
    if args.decoder_context_cap > 0:
        max_seq_len = max(max_seq_len, args.decoder_context_cap)

    pl.seed_everything(args.seed, workers=True)

    dataloaders: DataLoaders | None = None
    synthetic_val_loader: DataLoader | None = None
    if curriculum_components is None:
        dataloaders = create_dataloaders(
            dataset_artifacts,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            seed=args.seed,
            synthetic_train_samples=synthetic_train_samples,
        )
        if augmentation_factory_kwargs is not None and dataloaders.val is not None:
            val_dataset = getattr(dataloaders.val, "dataset", None)
            reference_size = len(val_dataset) if val_dataset is not None else 0
            synthetic_val_loader = _build_synthetic_val_loader(
                factory_kwargs=augmentation_factory_kwargs,
                dataset_size=reference_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                seed=args.seed + 1000,
            )

    model = _build_decoder(
        args,
        dataset_artifacts,
        backend,
        max_seq_len,
        task=task,
        feature_input_dim=feature_input_dim,
        lr_warmup_epochs=warmup_epochs,
        lr_scheduler_mode=args.lr_scheduler,
        warmup_first_epoch_fraction=args.warmup_first_epoch_fraction,
    )
    if manual_checkpoint_path is not None:
        _load_checkpoint_weights(model, manual_checkpoint_path)
    if curriculum_components is not None:
        dm = curriculum_components.data_module
        model.configure_synthetic_val(
            enabled=dm.expects_synthetic_val(),
            has_primary_val=dm.has_primary_val(),
        )
    else:
        assert dataloaders is not None
        model.configure_synthetic_val(
            enabled=synthetic_val_loader is not None,
            has_primary_val=dataloaders.val is not None,
        )

    if args.mode == "benchmark":
        _run_benchmark_mode(
            args,
            dataset_artifacts=dataset_artifacts,
            model=model,
            task=task,
        )
        return

    enable_wandb = not args.disable_wandb
    logging_artifacts = build_logging(
        log_dir=args.log_dir,
        experiment_name=run_name,
        enable_wandb=enable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or run_name,
        wandb_tags=args.wandb_tag,
        wandb_dir=args.wandb_dir,
        wandb_mode=args.wandb_mode,
        wandb_log_note=args.wandb_log_note,
        progress_refresh_rate=args.progress_refresh_rate,
        run_config=config_payload,
        metric_keys=DEFAULT_REGISTRY.metrics,
        dataset_metadata=dataset_artifacts.metadata,
    )
    for logger in logging_artifacts.loggers:
        try:
            from pytorch_lightning.loggers import CSVLogger as _CSV

            if isinstance(logger, _CSV):
                initialize_csv_logger(logger, DEFAULT_REGISTRY)
        except ImportError:
            continue

    callbacks = list(logging_artifacts.callbacks)

    has_primary_val = False
    dual_val_enabled = False
    if curriculum_components is not None:
        dm = curriculum_components.data_module
        has_primary_val = dm.has_primary_val()
        dual_val_enabled = has_primary_val and dm.expects_synthetic_val()
    else:
        has_primary_val = dataloaders is not None and dataloaders.val is not None
        dual_val_enabled = has_primary_val and synthetic_val_loader is not None
    if dual_val_enabled:
        callbacks.append(DualValidationAverager())
    enable_checkpointing = args.enable_checkpoints
    if enable_checkpointing:
        if not has_primary_val:
            raise ValueError("Checkpointing requires a validation split; set --val-fraction > 0.")
        if dual_val_enabled:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(run_dir),
                    filename="best-val",
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    auto_insert_metric_name=False,
                )
            )
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(run_dir),
                    filename="best-synthetic",
                    monitor="synthetic_val_acc",
                    mode="max",
                    save_top_k=1,
                    auto_insert_metric_name=False,
                )
            )
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(run_dir),
                    filename="best-overall",
                    monitor="val_overall_acc",
                    mode="max",
                    save_top_k=1,
                    auto_insert_metric_name=False,
                )
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(run_dir),
                filename="best",
                monitor=args.checkpoint_monitor,
                mode=args.checkpoint_mode,
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,
            )
            callbacks.append(checkpoint_callback)

    if args.early_stop_acc > 0:
        has_validation = False
        if curriculum_components is not None:
            has_validation = curriculum_components.data_module.has_primary_val()
        elif dataloaders is not None:
            has_validation = dataloaders.val is not None
        if not has_validation:
            raise ValueError(
                "Early stopping requires a validation split; set --val-fraction > 0 and ensure the first curriculum stage includes validation samples."
            )
        patience = args.early_stop_patience if args.early_stop_patience > 0 else max(1, args.max_epochs)
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=patience,
                stopping_threshold=args.early_stop_acc,
            )
        )

    if curriculum_components is not None:
        callbacks.append(curriculum_components.callback)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=_parse_devices(str(args.devices)),
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_checkpointing=enable_checkpointing,
        logger=logging_artifacts.loggers,
        callbacks=callbacks,
    )

    ckpt = resume_checkpoint
    if curriculum_components is not None:
        trainer.fit(model, datamodule=curriculum_components.data_module, ckpt_path=ckpt)
    else:
        assert dataloaders is not None
        val_loaders: Any = dataloaders.val
        if synthetic_val_loader is not None and dataloaders.val is not None:
            val_loaders = [dataloaders.val, synthetic_val_loader]
        if val_loaders is not None:
            trainer.fit(model, dataloaders.train, val_loaders, ckpt_path=ckpt)
        else:
            trainer.fit(model, dataloaders.train, ckpt_path=ckpt)


if __name__ == "__main__":
    main(sys.argv[1:])
