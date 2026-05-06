from __future__ import annotations

import time
import math
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Any, List, Sequence

import torch
from torch.profiler import ProfilerActivity, profile

from calibrated_memory.data.sequences.common import IGNORE_INDEX


@dataclass
class BenchmarkConfig:
    sequence_lengths: list[int]
    batch_size: int = 1
    repeat: int = 10
    warmup: int = 3
    device: str = "auto"
    query_length: int | None = None
    flops_only: bool = False
    latency_only: bool = False


@dataclass
class BenchmarkResult:
    sequence_length: int
    flops: float
    latency_mean_ms: float
    latency_p90_ms: float


@dataclass
class StreamingProfile:
    sequence_length: int
    perception_flops: float
    perception_latency_ms: float
    perception_flops_per_token: float
    perception_latency_per_token_ms: float
    query_flops: float
    query_latency_ms: float
    total_flops: float
    total_latency_mean_ms: float
    total_latency_p90_ms: float
    bulk_repeat: int
    latency_repeat: int


class BenchmarkRunner:
    def __init__(
        self,
        model,
        *,
        pad_id: int,
        vocab_size: int,
        task: str,
        config: BenchmarkConfig,
    ) -> None:
        self.model = model.eval()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.task = task
        self.config = config
        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        self.device = device
        self.model.to(device)
        self._long_seq_threshold = 4096
        lengths = sorted(config.sequence_lengths)
        self._ultra_seq_threshold = lengths[-1] if lengths else None
        self._ultra_repeat_runs = 3
        self._max_latency_repeat = 5
        self._profile_samples_per_metric = 8

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for seq_len in self.config.sequence_lengths:
            results.append(self._measure_sequence(seq_len, flops_only=self.config.flops_only))
        return results

    def _measure_sequence(self, seq_len: int, *, flops_only: bool) -> BenchmarkResult:
        batch = self._build_batch(seq_len)
        self._warmup(batch)
        if self.config.latency_only and flops_only:
            raise ValueError("BenchmarkRunner cannot be latency-only and flops-only simultaneously")
        collect_flops = not self.config.latency_only
        flops = float("nan")
        if collect_flops:
            flops = self._measure_flops(batch)
        if flops_only:
            latency_mean = float("nan")
            latency_p90 = float("nan")
        else:
            repeat = self._repeat_for_length(seq_len)
            latency_mean, latency_p90 = self._measure_latency(batch, repeat)
        return BenchmarkResult(
            sequence_length=seq_len,
            flops=flops,
            latency_mean_ms=latency_mean,
            latency_p90_ms=latency_p90,
        )

    def profile(
        self,
        sequence_lengths: Sequence[int],
        *,
        bulk_repeat_limit: int = 10,
    ) -> list[StreamingProfile]:
        profiles: list[StreamingProfile] = []
        per_metric_samples = max(1, min(self._profile_samples_per_metric, bulk_repeat_limit))
        for seq_len in sequence_lengths:
            if seq_len <= 0:
                continue
            profile = self._profile_sequence(seq_len, per_metric_samples)
            profiles.append(profile)
        return profiles

    def _profile_sequence(self, seq_len: int, samples_per_metric: int) -> StreamingProfile:
        measurement_window = samples_per_metric * 2
        total_stream_len = seq_len + measurement_window
        perception_batch = self._build_batch(total_stream_len)
        flop_samples, prefix_len = self._profile_flop_window(perception_batch, seq_len, samples_per_metric)
        latency_samples = self._profile_latency_window(perception_batch, prefix_len, samples_per_metric)
        perception_flops_per_token = (
            sum(flop_samples) / float(len(flop_samples)) if flop_samples else 0.0
        )
        perception_latency_per_token = (
            sum(latency_samples) / float(len(latency_samples)) if latency_samples else 0.0
        )
        perception_flops = perception_flops_per_token * float(seq_len)
        perception_latency = perception_latency_per_token * float(seq_len)
        full_batch = self._build_batch(seq_len)
        total_flops = self._measure_flops(full_batch)
        total_latency_mean, total_latency_p90 = self._measure_latency(full_batch, 1)
        query_flops = max(total_flops - perception_flops, 0.0)
        query_latency = max(total_latency_mean - perception_latency, 0.0)
        return StreamingProfile(
            sequence_length=seq_len,
            perception_flops=perception_flops,
            perception_latency_ms=perception_latency,
            perception_flops_per_token=perception_flops_per_token,
            perception_latency_per_token_ms=perception_latency_per_token,
            query_flops=query_flops,
            query_latency_ms=query_latency,
            total_flops=total_flops,
            total_latency_mean_ms=total_latency_mean,
            total_latency_p90_ms=total_latency_p90,
            bulk_repeat=len(flop_samples),
            latency_repeat=len(latency_samples),
        )

    def _profile_flop_window(
        self,
        base_batch: dict[str, Any],
        start_length: int,
        steps: int,
    ) -> tuple[list[float], int]:
        raw_stream = base_batch.get("_raw_stream")
        if raw_stream is None:
            raise ValueError("Base batch missing raw stream for profiling")
        max_len = raw_stream.size(1)
        if start_length <= 0:
            start_length = 1
        prefix_len = min(start_length, max_len)
        prev_total = self._profile_stream_total(base_batch, prefix_len)
        samples: list[float] = []
        available = max(0, max_len - prefix_len)
        actual_steps = min(steps, available)
        for _ in range(actual_steps):
            prefix_len = min(prefix_len + 1, max_len)
            total = self._profile_stream_total(base_batch, prefix_len)
            samples.append(max(total - prev_total, 0.0))
            prev_total = total
        return samples, prefix_len

    def _profile_latency_window(
        self,
        base_batch: dict[str, Any],
        start_length: int,
        steps: int,
    ) -> list[float]:
        raw_stream = base_batch.get("_raw_stream")
        if raw_stream is None:
            raise ValueError("Base batch missing raw stream for profiling")
        max_len = raw_stream.size(1)
        if start_length <= 0:
            start_length = 1
        prefix_len = min(start_length, max_len)
        prev_total = self._time_profile_prefix(base_batch, prefix_len)
        samples: list[float] = []
        available = max(0, max_len - prefix_len)
        actual_steps = min(steps, available)
        for _ in range(actual_steps):
            prefix_len = min(prefix_len + 1, max_len)
            elapsed = self._time_profile_prefix(base_batch, prefix_len)
            samples.append(max(elapsed - prev_total, 0.0))
            prev_total = elapsed
        return samples

    def _build_profile_batch(
        self,
        base_batch: dict[str, Any],
        prefix_len: int,
    ) -> dict[str, Any]:
        raw_stream = base_batch.get("_raw_stream")
        raw_query = base_batch.get("_raw_query")
        if raw_stream is None or raw_query is None:
            raise ValueError("Base batch is missing raw stream/query tensors for profiling.")
        prefix_len = max(0, min(prefix_len, raw_stream.size(1)))
        stream_prefix = raw_stream[:, :prefix_len].contiguous()
        return self._assemble_sequence_batch(stream_prefix, raw_query)

    def _profile_stream_total(
        self,
        base_batch: dict[str, Any],
        prefix_len: int,
    ) -> float:
        batch = self._build_profile_batch(base_batch, prefix_len)
        return self._measure_flops(batch)

    def _repeat_for_length(self, seq_len: int) -> int:
        base = max(1, self.config.repeat)
        if self._ultra_seq_threshold is not None and seq_len >= self._ultra_seq_threshold:
            repeat = min(base, self._ultra_repeat_runs)
        elif seq_len >= self._long_seq_threshold:
            repeat = max(1, math.ceil(base / 2))
        else:
            repeat = base
        return min(repeat, self._max_latency_repeat)

    # ------------------------------------------------------------------
    def _build_batch(
        self,
        seq_len: int,
        *,
        query_len: int | None = None,
    ) -> dict[str, Any]:
        batch_size = self.config.batch_size
        device = self.device
        stream_ids = torch.randint(
            low=1,
            high=max(2, self.vocab_size - 1),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        query_len = self._resolve_query_length(query_len)
        query_ids = torch.randint(
            low=1,
            high=max(2, self.vocab_size - 1),
            size=(batch_size, query_len),
            device=device,
            dtype=torch.long,
        )
        batch = self._assemble_sequence_batch(stream_ids, query_ids)
        batch["_raw_stream"] = stream_ids
        batch["_raw_query"] = query_ids
        return batch

    def _assemble_sequence_batch(
        self,
        stream_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> dict[str, Any]:
        batch_size = stream_ids.size(0)
        stream_len = stream_ids.size(1)
        query_len = query_ids.size(1)
        device = stream_ids.device
        total_len = stream_len + query_len
        if query_len > 0:
            sequence_ids = torch.cat([stream_ids, query_ids], dim=1)
        else:
            sequence_ids = stream_ids.clone()
        padding_mask = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), total_len, dtype=torch.long, device=device)
        labels = torch.full((batch_size, total_len), IGNORE_INDEX, dtype=torch.long, device=device)
        metadata = [{"stream_length": stream_len} for _ in range(batch_size)]
        sequence = {
            "input_ids": sequence_ids,
            "padding_mask": padding_mask,
            "lengths": lengths,
            "embeddings": None,
            "embedding_mask": None,
        }
        return {
            "sequence": sequence,
            "labels": labels,
            "metadata": metadata,
        }

    def _resolve_query_length(self, override: int | None) -> int:
        if override is not None and override >= 0:
            return int(override)
        query_len = self.config.query_length
        if query_len is None or query_len <= 0:
            query_len = 16
        return int(query_len)

    def _forward_once(self, batch: dict[str, Any]) -> None:
        with torch.inference_mode():
            metadata = batch.get("metadata") or [
                {"stream_length": int(batch["sequence"]["lengths"][0].item())}
            ]
            _ = self.model.compute_sequence_logits(
                batch["sequence"],
                batch["labels"],
                metadata,
            )

    def _warmup(self, batch: dict[str, Any]) -> None:
        for _ in range(max(0, self.config.warmup)):
            self._forward_once(batch)
            self._sync_device()

    def _measure_latency(
        self,
        batch: dict[str, Any],
        repeat: int,
    ) -> tuple[float, float]:
        durations: list[float] = []
        loops = max(1, repeat)
        for _ in range(loops):
            start = time.perf_counter()
            self._forward_once(batch)
            self._sync_device()
            end = time.perf_counter()
            durations.append((end - start) * 1000.0)
        durations.sort()
        mean = sum(durations) / len(durations)
        idx = max(0, min(len(durations) - 1, int(math.ceil(len(durations) * 0.9)) - 1))
        p90 = durations[idx]
        return mean, p90

    def _time_profile_prefix(
        self,
        base_batch: dict[str, Any],
        prefix_len: int,
    ) -> float:
        batch = self._build_profile_batch(base_batch, prefix_len)
        start = time.perf_counter()
        self._forward_once(batch)
        self._sync_device()
        return (time.perf_counter() - start) * 1000.0

    def _measure_flops(self, batch: dict[str, Any]) -> float:
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        context = self._flop_profile_context()
        with context, profile(
            activities=activities,
            with_flops=True,
            record_shapes=True,
        ) as prof:
            self._forward_once(batch)
            self._sync_device()
        flops = 0.0
        for item in prof.key_averages():
            if hasattr(item, "self_flops") and item.self_flops is not None:
                flops += float(item.self_flops)
        if flops <= 0.0:
            flops = self._fallback_flop_estimate(batch)
        return flops

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _fallback_flop_estimate(self, batch: dict[str, Any]) -> float:
        """Crude FLOP proxy when the profiler backend cannot emit counts (e.g., CPU-only builds)."""

        sequence_ids = batch["sequence"]["input_ids"]
        total_tokens = sequence_ids.numel()
        param_total = sum(param.numel() for param in self.model.parameters())
        return float(total_tokens * max(param_total, 1))

    def _flop_profile_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        backends = torch.backends.cuda
        if hasattr(backends, "flash_sdp_enabled"):
            @contextmanager
            def guard():
                flash = backends.flash_sdp_enabled()
                mem = backends.mem_efficient_sdp_enabled()
                math = backends.math_sdp_enabled()
                backends.enable_flash_sdp(False)
                backends.enable_mem_efficient_sdp(False)
                backends.enable_math_sdp(True)
                try:
                    yield
                finally:
                    backends.enable_flash_sdp(flash)
                    backends.enable_mem_efficient_sdp(mem)
                    backends.enable_math_sdp(math)

            return guard()
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            sdpa_backends = [torch.nn.attention.SDPBackend.MATH]
            if hasattr(torch.nn.attention.SDPBackend, "EFFICIENT_ATTENTION"):
                sdpa_backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
            return torch.nn.attention.sdpa_kernel(sdpa_backends, set_priority=True)
        if hasattr(backends, "sdp_kernel"):
            return backends.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=True,
            )
        return nullcontext()
