#!/usr/bin/env python
"""Tune backend configs so FLOPs roughly align at a fixed context length."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from calibrated_memory.training.registries import build_backend, build_dataset

TARGET_SEQ_LEN = 256
DEFAULT_TARGET_SMALL = 5e7  # 0.05 GFLOPs
DEFAULT_TARGET_RATIO_MEDIUM = 2.5
DEFAULT_TARGET_RATIO_LARGE = 6.0

SIZE_PRESETS = {
    "small": {"hidden_dim": 64, "layers": 3},
    "medium": {"hidden_dim": 96, "layers": 4},
    "large": {"hidden_dim": 128, "layers": 6},
}

DIM_KEYS = (
    "embed_dim",
    "hidden_dim",
    "model_dim",
    "d_model",
    "memory_dim",
    "slot_dim",
)
LAYER_KEYS = ("num_layers", "depth", "n_layers", "layers")
HEAD_KEYS = ("num_heads", "n_heads", "n_head", "heads", "local_window_heads")

CONTEXT_KEYS = {
    "deltaformer": "ctx_len",
    "deltanet": "ctx_len",
    "gated_deltanet": "ctx_len",
    "mom": "ctx_len",
    "log_linear_mamba": "ctx_len",
    "retnet": "ctx_len",
    "rwkv": "ctx_len",
}

TUNABLE_VALUES: dict[str, Sequence[Any]] = {
    "n_heads": [1, 2, 3, 4, 6, 8, 10, 12, 16],
    "num_heads": [1, 2, 3, 4, 6, 8, 10, 12, 16],
    "heads": [1, 2, 3, 4, 6, 8, 10, 12, 16],
    "head_dim": [16, 24, 32, 48, 64],
    "headdim": [16, 24, 32, 48, 64],
    "ff_mult": [1, 2, 3, 4],
    "mlp_ratio": [1, 2, 3, 4],
    "hidden_ratio": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
    "d_state": [64, 96, 128, 160, 192, 224, 256],
    "expand": [1, 2, 3, 4],
    "compress_ratio": [1, 2, 3, 4],
    "num_slots": [4, 8, 12, 16, 24],
    "num_memory_cells": [32, 48, 64, 96, 128],
    "chunk_size": [2, 4, 8, 16],
}


@dataclass(frozen=True)
class HeadConstraint:
    min_head_dim: int = 1
    require_power_of_two: bool = False


_FLA_MIN_HEAD_BACKENDS = {
    "deltaformer",
    "deltanet",
    "gated_deltanet",
    "log_linear_mamba",
    "mom",
    "retnet",
    "rwkv",
}

HEAD_CONSTRAINTS: dict[str, HeadConstraint] = {
    backend: HeadConstraint(min_head_dim=16) for backend in _FLA_MIN_HEAD_BACKENDS
}
HEAD_CONSTRAINTS["deltaformer"] = HeadConstraint(min_head_dim=16, require_power_of_two=True)

HEAD_DIM_SOURCES: dict[str, tuple[str, ...]] = {
    "heads": ("hidden_dim", "embed_dim", "model_dim"),
    "local_window_heads": ("hidden_dim", "embed_dim"),
}

HALF_PRECISION_BACKENDS = {"deltaformer"}
DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
    "f16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "none": None,
    "": None,
}

SCRIPT_ROOT = Path(__file__).resolve().parent.parent / "slurm_scripts" / "sweep_sequences"
SCRIPT_TEMPLATE = "{name}_bucket.sh"

OPTION_PATTERN = re.compile(r"--backend-option\s+([^=\s]+)=([^\s\\]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Equalize backend FLOPs by tweaking hyperparameters.")
    parser.add_argument("--backends", nargs="*", default=[], help="Backend names to tune (default: all baseline scripts).")
    parser.add_argument("--sizes", nargs="*", choices=list(SIZE_PRESETS.keys()), default=list(SIZE_PRESETS.keys()))
    parser.add_argument("--target-small", type=float, default=DEFAULT_TARGET_SMALL, help="FLOP target for the 'small' tier.")
    parser.add_argument("--target-medium", type=float, default=None, help="FLOP target for the 'medium' tier (defaults to --target-small * ratio).")
    parser.add_argument("--target-large", type=float, default=None, help="FLOP target for the 'large' tier (defaults to --target-small * ratio^2).")
    parser.add_argument(
        "--ratio-medium",
        type=float,
        default=DEFAULT_TARGET_RATIO_MEDIUM,
        help="Multiplier applied to --target-small when deriving the default medium target.",
    )
    parser.add_argument(
        "--ratio-large",
        type=float,
        default=DEFAULT_TARGET_RATIO_LARGE,
        help="Multiplier applied to the medium target when deriving the default large target.",
    )
    parser.add_argument("--tolerance", type=float, default=5e6, help="Stop once abs(delta) <= tolerance.")
    parser.add_argument("--max-trials", type=int, default=10, help="Max tuning steps per backend/size.")
    parser.add_argument(
        "--dataset-option",
        action="append",
        default=[],
        help="Synthetic dataset override (repeatable). Use seq_len, num_sequences, etc.",
    )
    parser.add_argument("--output", type=Path, default=Path("equalized_backends.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking; torch.cuda.is_available() returned False.")

    size_targets = _resolve_size_targets(args)

    dataset_overrides = {
        "task": "membership",
        "seq_len": TARGET_SEQ_LEN,
        "num_sequences": 8,
        "unique_sequences": 8,
        "vocab_size": 256,
    }
    dataset_overrides.update(_parse_cli_overrides(args.dataset_option))
    task_name = str(dataset_overrides.get("task", "membership")).lower()
    if task_name == "continuation" and "cont_len" not in dataset_overrides:
        dataset_overrides["cont_len"] = 4
    dataset = build_dataset("synthetic", dataset_overrides)
    dataset_context = int(getattr(dataset, "max_seq_len", TARGET_SEQ_LEN))

    target_backends = args.backends or _discover_backends()
    results: list[dict[str, Any]] = []

    for backend_name in sorted(target_backends):
        base_overrides = _load_baseline_overrides(backend_name)
        if base_overrides is None:
            print(f"[skip] missing baseline overrides for {backend_name}")
            continue
        backend_rows: list[dict[str, Any]] = []
        for size in args.sizes:
            tuned, flops = _tune_backend(
                backend_name,
                base_overrides,
                size,
                dataset,
                context_len=dataset_context,
                target=size_targets[size],
                tolerance=args.tolerance,
                max_trials=args.max_trials,
            )
            if flops is None:
                continue
            delta = flops - size_targets[size]
            row = {
                "backend": backend_name,
                "size": size,
                "flops": flops,
                "delta": delta,
                "target": size_targets[size],
                "overrides": tuned,
            }
            results.append(row)
            backend_rows.append(row)
        if backend_rows:
            print(f"\nBackend: {backend_name}")
            for size in args.sizes:
                row = next((entry for entry in backend_rows if entry["size"] == size), None)
                if not row:
                    print(f"  {size:<6}: unable to benchmark")
                    continue
                preset = SIZE_PRESETS[size]
                target_mflops = row["target"] / 1e6
                print(
                    f"  {size:<6} | hidden={preset['hidden_dim']:>3} | layers={preset['layers']:>2} | "
                    f"{row['flops']/1e9:.6f} GFLOPs @ T={TARGET_SEQ_LEN} (target {target_mflops:.3f} MFLOPs) "
                    f"| overrides={row['overrides']}"
                )
    args.output.write_text(json.dumps(results, indent=2))


def _resolve_size_targets(args: argparse.Namespace) -> dict[str, float]:
    targets = {
        "small": float(args.target_small),
        "medium": float(args.target_medium)
        if args.target_medium is not None
        else float(args.target_small) * float(args.ratio_medium),
    }
    base_large = targets["medium"] * float(args.ratio_large)
    targets["large"] = float(args.target_large) if args.target_large is not None else base_large
    return targets


def _discover_backends() -> list[str]:
    names: list[str] = []
    for script in SCRIPT_ROOT.glob("*_bucket.sh"):
        parts = script.name.split("_bucket.sh", 1)
        if parts and parts[0]:
            names.append(parts[0])
    return sorted(names)


def _load_baseline_overrides(backend: str) -> dict[str, Any] | None:
    script = SCRIPT_ROOT / SCRIPT_TEMPLATE.format(name=backend)
    if not script.exists():
        return None
    text = script.read_text()
    overrides: dict[str, Any] = {}
    for match in OPTION_PATTERN.finditer(text):
        key, raw = match.group(1), match.group(2)
        overrides[key.replace("-", "_")] = _coerce_value(raw)
    return overrides


def _ensure_context_capacity(
    backend: str,
    overrides: dict[str, Any],
    context_len: int,
) -> None:
    if context_len <= 0:
        return
    key = CONTEXT_KEYS.get(backend)
    if key is not None:
        overrides[key] = int(context_len)
        return
    if backend == "memory_mosaic":
        overrides["block_size"] = int(context_len)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resolve_head_constraint(backend: str) -> HeadConstraint:
    return HEAD_CONSTRAINTS.get(backend, HeadConstraint())


def _resolve_head_key(overrides: dict[str, Any]) -> tuple[str | None, int | None]:
    for key in HEAD_KEYS:
        if key not in overrides:
            continue
        try:
            return key, int(overrides[key])
        except (TypeError, ValueError):
            return key, None
    return None, None


def _resolve_dim_for_head(overrides: dict[str, Any], head_key: str | None) -> int | None:
    if head_key is None:
        return None
    sources = HEAD_DIM_SOURCES.get(head_key, DIM_KEYS)
    for key in sources:
        if key in overrides and overrides[key] is not None:
            try:
                return int(overrides[key])
            except (TypeError, ValueError):
                continue
    return None


def _divisors(value: int) -> list[int]:
    if value <= 0:
        return []
    divisors: set[int] = set()
    limit = int(math.sqrt(value))
    for candidate in range(1, limit + 1):
        if value % candidate != 0:
            continue
        divisors.add(candidate)
        divisors.add(value // candidate)
    return sorted(divisors)


def _head_dim_allowed(dim_value: int, head_count: int, constraint: HeadConstraint) -> bool:
    if head_count <= 0 or dim_value % head_count != 0:
        return False
    head_dim = dim_value // head_count
    if head_dim < constraint.min_head_dim:
        return False
    if constraint.require_power_of_two and not _is_power_of_two(head_dim):
        return False
    return True


def _enumerate_valid_head_counts(dim_value: int, constraint: HeadConstraint) -> list[int]:
    valid: list[int] = []
    for candidate in _divisors(dim_value):
        if _head_dim_allowed(dim_value, candidate, constraint):
            valid.append(candidate)
    return sorted(valid)


def _select_head_value(valid: list[int], preferred: int | None) -> int:
    if not valid:
        raise ValueError("No valid head counts available for the requested dimension")
    if preferred is None:
        return min(valid)
    return min(valid, key=lambda choice: (abs(choice - preferred), choice))


def _harmonize_backend_overrides(backend: str, overrides: dict[str, Any]) -> None:
    _ensure_autocast_policy(backend, overrides)
    head_key, head_value = _resolve_head_key(overrides)
    dim_value = _resolve_dim_for_head(overrides, head_key)
    if backend == "deltaformer" and head_key is not None and "num_kv_heads" in overrides:
        kv_heads = overrides.get("num_kv_heads")
        if kv_heads not in {None, "None"}:
            overrides["num_kv_heads"] = overrides[head_key]
    if head_key is None or dim_value is None:
        return
    constraint = _resolve_head_constraint(backend)
    current_value = head_value if isinstance(head_value, int) else None
    if current_value is not None and _head_dim_allowed(dim_value, current_value, constraint):
        return
    try:
        overrides[head_key] = _select_head_value(
            _enumerate_valid_head_counts(dim_value, constraint),
            current_value,
        )
    except ValueError:
        pass


def _ensure_autocast_policy(backend: str, overrides: dict[str, Any]) -> None:
    if backend not in HALF_PRECISION_BACKENDS:
        return
    dtype_value = overrides.get("autocast_dtype")
    if dtype_value in {None, "", "None", "fp32", "float32", "torch.float32"}:
        overrides["autocast_dtype"] = "bf16"


def _prepare_runtime_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(overrides)
    dtype_value = runtime.get("autocast_dtype")
    if isinstance(dtype_value, str):
        runtime["autocast_dtype"] = DTYPE_ALIASES.get(dtype_value.strip().lower(), dtype_value)
    return runtime


def _violates_constraints(backend: str, overrides: dict[str, Any]) -> bool:
    head_key, head_value = _resolve_head_key(overrides)
    if head_key is None or head_value is None:
        return False
    dim_value = _resolve_dim_for_head(overrides, head_key)
    if dim_value is None:
        return False
    return not _head_dim_allowed(dim_value, head_value, _resolve_head_constraint(backend))


def _tune_backend(
    backend_name: str,
    base_overrides: dict[str, Any],
    size: str,
    dataset,
    *,
    context_len: int,
    target: float,
    tolerance: float,
    max_trials: int,
) -> tuple[dict[str, Any], float | None]:
    size_cfg = SIZE_PRESETS[size]
    current = _apply_size(
        deepcopy(base_overrides),
        size_cfg["hidden_dim"],
        size_cfg["layers"],
        backend_name,
        context_len,
    )
    best_flops = _evaluate(backend_name, current, dataset, size_cfg)
    if best_flops is None:
        return current, None
    trials = 0
    keys = [key for key in TUNABLE_VALUES if key in current and key not in DIM_KEYS and key not in LAYER_KEYS]
    while trials < max_trials and abs(best_flops - target) > tolerance and keys:
        improved = False
        for key in keys:
            values = TUNABLE_VALUES[key]
            base_value = current.get(key)
            local_best = best_flops
            local_choice = base_value
            for candidate in values:
                if candidate == base_value:
                    continue
                trial = dict(current)
                trial[key] = candidate
                if _violates_constraints(backend_name, trial):
                    continue
                flops = _evaluate(backend_name, trial, dataset, size_cfg)
                trials += 1
                if flops is None:
                    continue
                if abs(flops - target) < abs(local_best - target):
                    local_best = flops
                    local_choice = candidate
                    improved = True
                if trials >= max_trials:
                    break
            if local_choice is not None:
                current[key] = local_choice
                best_flops = local_best
            if trials >= max_trials:
                break
        if not improved:
            break
    return current, best_flops

def _apply_size(
    overrides: dict[str, Any],
    hidden_dim: int,
    layers: int,
    backend_name: str,
    context_len: int,
) -> dict[str, Any]:
    for key in DIM_KEYS:
        if key in overrides:
            overrides[key] = hidden_dim
    for key in LAYER_KEYS:
        if key in overrides:
            overrides[key] = layers
    _ensure_context_capacity(backend_name, overrides, context_len)
    _harmonize_backend_overrides(backend_name, overrides)
    return overrides


def _evaluate(
    backend_name: str,
    overrides: dict[str, Any],
    dataset,
    size_cfg: dict[str, int],
) -> float | None:
    try:
        runtime_overrides = _prepare_runtime_overrides(overrides)
        backend, _ = build_backend(backend_name, runtime_overrides)
        model = MemoryBankDecoder(
            vocab_size=dataset.vocab_size,
            pad_id=dataset.pad_id,
            max_seq_len=max(dataset.max_seq_len, TARGET_SEQ_LEN),
            d_model=size_cfg["hidden_dim"],
            nhead=1,
            num_layers=1,
            mlp_ratio=1,
            lr=1e-3,
            attn_dropout=0.0,
            embed_dropout=0.0,
            resid_dropout=0.0,
            rotary_base=10000.0,
            weight_decay=0.0,
            max_epochs=1,
            memory_backend=backend,
            task="membership",
            execution_mode="direct",
            loss_type="cross_entropy",
            deep_gambler_mode="fixed",
            deep_gambler_o=1.5,
            deep_gambler_epsilon=1e-12,
            deep_gambler_activation_acc=0.5,
        )
        runner = BenchmarkRunner(
            model,
            pad_id=dataset.pad_id,
            vocab_size=dataset.vocab_size,
            task="membership",
            config=BenchmarkConfig(
                sequence_lengths=[TARGET_SEQ_LEN],
                batch_size=1,
                warmup=1,
                repeat=1,
                flops_only=True,
                device="cuda",
            ),
        )
        flops = runner.run()[0].flops
        return flops
    except Exception as exc:  # noqa: BLE001
        print(f"[error] backend={backend_name} overrides={overrides}: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()


def _iter_overrides(hyper: Dict[str, Iterable[Any]]) -> Iterator[dict[str, Any]]:
    keys = sorted(hyper)
    if not keys:
        yield {}
        return

    def _recurse(idx: int, current: Dict[str, Any]):
        if idx >= len(keys):
            yield dict(current)
            return
        key = keys[idx]
        values = hyper.get(key, [])
        if not values:
            yield from _recurse(idx + 1, current)
            return
        for value in values:
            current[key] = value
            yield from _recurse(idx + 1, current)
        current.pop(key, None)

    yield from _recurse(0, {})


def _parse_cli_overrides(entries: Sequence[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for entry in entries:
        if not entry or "=" not in entry:
            continue
        key, raw = entry.split("=", 1)
        payload[key.strip().replace("-", "_")] = _coerce_value(raw.strip())
    return payload


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


if __name__ == "__main__":
    main()
