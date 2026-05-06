"""Utilities for building device placement constraints for multi-GPU models."""

from __future__ import annotations

from typing import Dict, Union

import torch


def _ensure_cuda_available() -> int:
    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("CUDA is required to build a GPU memory map")
    return count


def build_max_memory_map(
    max_gpu_gb: float,
    *,
    first_gpu_ratio: float = 2.0 / 3.0,
    clamp_to_free: bool = False,
    reserve_gb: float = 0.0,
    format_strings: bool = True,
) -> Dict[Union[int, str], Union[str, float]]:
    """Return a Hugging Face ``max_memory`` dict for the current host."""

    if max_gpu_gb <= 0:
        raise ValueError("max_gpu_gb must be positive")

    device_count = _ensure_cuda_available()
    mapping: Dict[Union[int, str], Union[str, float]] = {}
    first_cap = max_gpu_gb * first_gpu_ratio

    for idx in range(device_count):
        cap = first_cap if idx == 0 else max_gpu_gb
        if clamp_to_free:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_gb = free_bytes / (1024 ** 3)
            cap = min(cap, max(free_gb - reserve_gb, 1.0))
        mapping[idx] = f"{cap:.2f}GiB" if format_strings else cap

    mapping["cpu"] = "0GiB" if format_strings else 0.0
    return mapping
