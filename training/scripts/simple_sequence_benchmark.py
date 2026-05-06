#!/usr/bin/env python
"""Lightweight sequence benchmark runner for MemoryBankDecoder backends."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from calibrated_memory.training.registries import BACKEND_SPECS, build_backend

DEFAULT_LENGTHS = [1024, 2048, 4096, 8192]
LENGTH_TIME_LIMIT_SEC = int(os.environ.get("BENCHMARK_LENGTH_LIMIT_SEC", "300"))
QUERY_LENGTH = 1
VOCAB_SIZE = 16
PAD_ID = 0
DECODER_NUM_LAYERS = 3
DECODER_NHEAD = 8
DECODER_MLP_RATIO = 1
PROFILE_PAD_TOKENS = 16  # matches BenchmarkRunner._profile_samples_per_metric * 2
TIER_MAP = {
    "small": "layers3",
    "medium": "layers3",
    "large": "layers5",
}
BACKEND_SEQUENCE = [
    "deltaformer",
    "deltanet",
    "gated_deltanet",
    "mamba",
    "mom",
    "gla",
    "retnet",
    "memory_mosaic",
    "titans_external",
    "transformer_pp",
    "simple_rnn",
    "rwkv",
    "ttt",
    "ttt_fast",
]
BACKEND_LABELS = {
    "gated_deltanet": "gated-deltanet",
    "memory_mosaic": "memory-mosaic",
    "transformer_pp": "transformer-pp",
    "simple_rnn": "simple-rnn",
    "titans_external": "titans-external",
    "ttt_fast": "ttt-fast",
}
BACKEND_DYNAMIC_OVERRIDES = {
    "deltaformer": "ctx_len",
    "deltanet": "ctx_len",
    "gated_deltanet": "ctx_len",
    "mom": "ctx_len",
    "retnet": "ctx_len",
    "rwkv": "ctx_len",
    "memory_mosaic": "block_size",
    "gla": "ctx_len",
}
LENGTH_KEYS = (
    "ctx_len",
    "max_position_embeddings",
    "max_seq_len",
    "seq_len_max",
    "block_size",
)


def _require_cuda(device_name: str) -> str:
    """Ensure benchmarks only run against CUDA backends."""

    normalized = device_name.lower()
    if not normalized.startswith("cuda"):
        raise SystemExit(
            f"sequence benchmark currently requires CUDA; set --device cuda (got '{device_name}')."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() returned False")
    return device_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLOPs/latency/profile sweeps without loading datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", type=Path, default=Path("configs.json"), help="Path to configs.json overrides.")
    parser.add_argument("--embed-dim", type=int, choices=(64, 128, 256), default=64, help="Backend embedding dimension selector.")
    parser.add_argument("--size", choices=("medium",), default="medium", help="Model scale used to pick override tiers.")
    parser.add_argument(
        "--backends",
        default="",
        help="Comma-separated backend filter; leave empty to benchmark the full suite.",
    )
    parser.add_argument("--device", default="cuda", help="Device used by BenchmarkRunner (cuda/cpu).")
    parser.add_argument("--repeat", type=int, default=5, help="Timed iterations per latency measurement.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs discarded before timing.")
    parser.add_argument(
        "--profile-lengths",
        default="",
        help="Optional comma-separated subset for perception profiling (default picks first/mid/last).",
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
        help="Comma-separated sequence lengths to benchmark (defaults to the fixed long-context sweep).",
    )
    parser.add_argument(
        "--run-mode",
        choices=("full", "metrics", "profiles"),
        default="full",
        help="full runs every stage, metrics only collects FLOPs/latency, and profiles only runs the streaming profiler.",
    )
    return parser.parse_args()


def _load_configs(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file '{path}' is missing")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_int(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        text = str(value)
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(math.floor(float(text)))
    except (TypeError, ValueError):
        return None


def _parse_lengths(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(max(1, int(chunk)))
        except ValueError as exc:
            raise SystemExit(f"Invalid sequence length '{chunk}'") from exc
    if not values:
        raise SystemExit("At least one sequence length must be provided")
    return sorted(set(values))


def _select_profile_lengths(seq_lengths: Iterable[int], override: str | None) -> list[int]:
    if override:
        picked = []
        for chunk in override.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                picked.append(max(1, int(chunk)))
            except ValueError as exc:
                raise SystemExit(f"Invalid profile length '{chunk}'") from exc
        return sorted(set(picked))
    ordered = sorted(set(seq_lengths))
    if not ordered:
        return []
    picks = {ordered[0]}
    if len(ordered) > 1:
        picks.add(ordered[-1])
    if 256 in ordered:
        picks.add(256)
    return sorted(picks)


def _build_model(backend_name: str, overrides: dict, embed_dim: int) -> MemoryBankDecoder:
    backend, config = build_backend(backend_name, overrides)
    decoder = MemoryBankDecoder(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        max_seq_len=QUERY_LENGTH + 4,
        d_model=embed_dim,
        nhead=DECODER_NHEAD,
        num_layers=DECODER_NUM_LAYERS,
        mlp_ratio=DECODER_MLP_RATIO,
        memory_backend=backend,
        task="membership",
        execution_mode="direct",
    )
    decoder.eval()
    return decoder


def _run_benchmark(
    model: MemoryBankDecoder,
    *,
    device: str,
    lengths: list[int],
    repeat: int,
    warmup: int,
    flops_only: bool,
    latency_only: bool,
) -> list:
    config = BenchmarkConfig(
        sequence_lengths=lengths,
        batch_size=1,
        repeat=max(1, repeat),
        warmup=max(0, warmup),
        device=device,
        query_length=QUERY_LENGTH,
        flops_only=flops_only,
        latency_only=latency_only,
    )
    runner = BenchmarkRunner(
        model,
        pad_id=PAD_ID,
        vocab_size=VOCAB_SIZE,
        task="membership",
        config=config,
    )
    return runner.run()


def _run_profile(
    model: MemoryBankDecoder,
    *,
    device: str,
    lengths: list[int],
) -> list:
    config = BenchmarkConfig(
        sequence_lengths=lengths,
        batch_size=1,
        repeat=1,
        warmup=0,
        device=device,
        query_length=QUERY_LENGTH,
    )
    runner = BenchmarkRunner(
        model,
        pad_id=PAD_ID,
        vocab_size=VOCAB_SIZE,
        task="membership",
        config=config,
    )
    return runner.profile(lengths)


def _run_with_limit(
    *,
    label: str,
    stage: str,
    length: int,
    limit_sec: int,
    runner: Callable[[], list],
) -> tuple[list, bool]:
    start = time.perf_counter()
    results = runner()
    elapsed = time.perf_counter() - start
    timed_out = limit_sec > 0 and elapsed > limit_sec
    if timed_out:
        print(
            f"[{_format_timestamp()}] Length {length} {stage} for {label} took {elapsed:.1f}s (> {limit_sec}s); "
            "skipping remaining lengths for this backend.",
            flush=True,
        )
    return results, timed_out


def _format_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _print_flops(results) -> None:
    print("Sequence Length |   MFLOPs")
    for item in results:
        flops = "n/a" if math.isnan(item.flops) else f"{item.flops / 1e6:.6g}"
        print(f"{item.sequence_length:>15} | {flops:>8}")


def _print_latency(results) -> None:
    print("Sequence Length | Latency Mean (ms) | Latency P90 (ms)")
    for item in results:
        mean = "n/a" if math.isnan(item.latency_mean_ms) else f"{item.latency_mean_ms:.2f}"
        p90 = "n/a" if math.isnan(item.latency_p90_ms) else f"{item.latency_p90_ms:.2f}"
        print(f"{item.sequence_length:>15} | {mean:>18} | {p90:>15}")


def _print_profiles(profiles) -> None:
    for row in profiles:
        total = row.total_flops / 1e6 if row.total_flops else 0.0
        percep = row.perception_flops / 1e6 if row.perception_flops else 0.0
        query = row.query_flops / 1e6 if row.query_flops else 0.0
        percep_avg = row.perception_flops_per_token / 1e6 if row.perception_flops_per_token else 0.0
        print(
            f"  L={row.sequence_length:<6} Σc≈{percep:.3f} MFLOPs (avg {percep_avg:.3f}/token; "
            f"{row.bulk_repeat} flop samples) query={query:.3f} MFLOPs total={total:.3f} MFLOPs "
            f"ℓ_total={row.total_latency_mean_ms:.2f}ms ℓ_p90={row.total_latency_p90_ms:.2f}ms"
        )
        print(
            f"      perception_latency≈{row.perception_latency_ms:.2f}ms "
            f"per-token={row.perception_latency_per_token_ms:.4f}ms ("
            f"{row.latency_repeat} latency samples) query_latency={row.query_latency_ms:.2f}ms"
        )


def _sanitize_backends(raw: str) -> list[str] | None:
    if not raw:
        return None
    results = []
    for chunk in raw.split(","):
        name = chunk.strip()
        if name:
            results.append(name)
    return results or None


def main() -> None:
    args = _parse_args()
    args.device = _require_cuda(args.device)
    data = _load_configs(args.configs)
    tier = TIER_MAP[args.size]
    requested = _sanitize_backends(args.backends)
    backends = BACKEND_SEQUENCE if requested is None else requested
    torch.set_grad_enabled(False)
    run_metrics = args.run_mode in {"full", "metrics"}
    run_profiles = args.run_mode in {"full", "profiles"}
    for backend_name in backends:
        backend_data = data.get(backend_name)
        if backend_data is None:
            raise SystemExit(f"No config block for backend '{backend_name}'")
        width_data = backend_data.get(str(args.embed_dim))
        if width_data is None:
            raise SystemExit(f"No config for {backend_name}/{args.embed_dim}")
        tier_data = width_data.get(tier)
        if tier_data is None:
            raise SystemExit(f"No tier '{tier}' config for {backend_name}/{args.embed_dim}")
        overrides = dict(tier_data)
        overrides.setdefault("embed_dim", args.embed_dim)
        if backend_name == "transformer_pp":
            overrides["positional_mode"] = "rope"
        spec = BACKEND_SPECS.get(backend_name)
        if spec and "vocab_size" in spec.defaults:
            overrides["vocab_size"] = VOCAB_SIZE
        seq_lengths = _parse_lengths(args.lengths)
        limit = None
        for key in LENGTH_KEYS:
            if key in overrides:
                candidate = _coerce_int(overrides[key])
                if candidate and candidate > 0:
                    limit = candidate if limit is None else max(limit, candidate)
        max_length = seq_lengths[-1]
        combined_length = max_length + QUERY_LENGTH
        if run_profiles:
            combined_length += PROFILE_PAD_TOKENS
        if limit is not None and limit < combined_length:
            print(
                f"[{_format_timestamp()}] WARNING: {backend_name} config caps context at {limit} tokens "
                f"but the benchmark will request {combined_length}; override your configs if this fails.",
                flush=True,
            )
        if backend_name in BACKEND_DYNAMIC_OVERRIDES:
            overrides[BACKEND_DYNAMIC_OVERRIDES[backend_name]] = combined_length
        label = BACKEND_LABELS.get(backend_name, backend_name)
        timestamp = _format_timestamp()
        print(f"[{timestamp}] Benchmarking {label} ({args.embed_dim}-{args.size}) with lengths {seq_lengths}")
        print(f"[{timestamp}] Backend config: {json.dumps(overrides, sort_keys=True)}")
        model = _build_model(backend_name, overrides, args.embed_dim)
        abort_backend = False
        if run_metrics:
            flops_results: list = []
            for length in seq_lengths:
                result, timed_out = _run_with_limit(
                    label=label,
                    stage="MFLOPs",
                    length=length,
                    limit_sec=LENGTH_TIME_LIMIT_SEC,
                    runner=lambda length=length: _run_benchmark(
                        model,
                        device=args.device,
                        lengths=[length],
                        repeat=args.repeat,
                        warmup=args.warmup,
                        flops_only=True,
                        latency_only=False,
                    ),
                )
                flops_results.extend(result)
                if timed_out:
                    abort_backend = True
                    break
            _print_flops(flops_results)
        if run_metrics and not abort_backend:
            latency_results: list = []
            for length in seq_lengths:
                result, timed_out = _run_with_limit(
                    label=label,
                    stage="latency",
                    length=length,
                    limit_sec=LENGTH_TIME_LIMIT_SEC,
                    runner=lambda length=length: _run_benchmark(
                        model,
                        device=args.device,
                        lengths=[length],
                        repeat=args.repeat,
                        warmup=args.warmup,
                        flops_only=False,
                        latency_only=True,
                    ),
                )
                latency_results.extend(result)
                if timed_out:
                    abort_backend = True
                    break
            _print_latency(latency_results)
        if run_profiles and not abort_backend:
            profile_lengths = _select_profile_lengths(seq_lengths, args.profile_lengths)
            print(f"[{_format_timestamp()}] Streaming perception/profile lengths {profile_lengths}")
            profiles: list = []
            for length in profile_lengths:
                result, timed_out = _run_with_limit(
                    label=label,
                    stage="profile",
                    length=length,
                    limit_sec=LENGTH_TIME_LIMIT_SEC,
                    runner=lambda length=length: _run_profile(
                        model,
                        device=args.device,
                        lengths=[length],
                    ),
                )
                profiles.extend(result)
                if timed_out:
                    abort_backend = True
                    break
            _print_profiles(profiles)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
