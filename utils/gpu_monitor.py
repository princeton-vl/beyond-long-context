"""GPU monitoring utilities for tracking usage during batch processing."""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import torch


def get_gpu_usage() -> Dict[str, any]:
    """
    Get current GPU usage statistics for all visible GPUs.

    Returns a dict with:
    - gpu_count: Number of visible GPUs
    - gpus: List of dicts with per-GPU stats (memory, utilization if available)
    - process_info: Current process memory usage per GPU
    """
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpus": [], "process_info": {}}

    gpu_count = torch.cuda.device_count()
    gpus = []
    process_info = {}

    # Get SLURM job info for context
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    slurm_node = os.environ.get("SLURM_NODELIST", os.environ.get("SLURMD_NODENAME", "N/A"))

    for gpu_id in range(gpu_count):
        try:
            # Get memory info
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            allocated_mem = torch.cuda.memory_allocated(gpu_id)
            reserved_mem = torch.cuda.memory_reserved(gpu_id)

            # Get device properties
            props = torch.cuda.get_device_properties(gpu_id)

            gpu_info = {
                "id": gpu_id,
                "name": props.name,
                "total_mem_gb": total_mem / (1024 ** 3),
                "free_mem_gb": free_mem / (1024 ** 3),
                "used_mem_gb": (total_mem - free_mem) / (1024 ** 3),
                "allocated_mem_gb": allocated_mem / (1024 ** 3),
                "reserved_mem_gb": reserved_mem / (1024 ** 3),
                "memory_utilization_pct": ((total_mem - free_mem) / total_mem) * 100,
            }

            gpus.append(gpu_info)

            # Process-specific memory
            if allocated_mem > 0:
                process_info[gpu_id] = {
                    "allocated_gb": allocated_mem / (1024 ** 3),
                    "reserved_gb": reserved_mem / (1024 ** 3),
                }

        except Exception as e:
            gpus.append({
                "id": gpu_id,
                "error": str(e),
            })

    return {
        "gpu_count": gpu_count,
        "gpus": gpus,
        "process_info": process_info,
        "slurm_job_id": slurm_job_id,
        "slurm_node": slurm_node,
    }


def print_gpu_usage(batch_idx: Optional[int] = None, prefix: str = "") -> None:
    """
    Print GPU usage in a human-readable format.

    Args:
        batch_idx: Optional batch index to include in output
        prefix: Optional prefix for the output (e.g., "  " for indentation)
    """
    usage = get_gpu_usage()

    if usage["gpu_count"] == 0:
        print(f"{prefix}GPU Usage: No GPUs available")
        return

    # Header
    header_parts = ["GPU Usage"]
    if batch_idx is not None:
        header_parts.append(f"(after batch {batch_idx})")
    if usage["slurm_job_id"] != "N/A":
        header_parts.append(f"[Job {usage['slurm_job_id']}, Node {usage['slurm_node']}]")

    print(f"{prefix}{'─' * 80}")
    print(f"{prefix}{' '.join(header_parts)}")
    print(f"{prefix}{'─' * 80}")

    # Per-GPU stats
    for gpu in usage["gpus"]:
        if "error" in gpu:
            print(f"{prefix}  GPU {gpu['id']}: ERROR - {gpu['error']}")
            continue

        print(f"{prefix}  GPU {gpu['id']}: {gpu['name']}")
        print(f"{prefix}    Memory: {gpu['used_mem_gb']:.2f} / {gpu['total_mem_gb']:.2f} GB ({gpu['memory_utilization_pct']:.1f}% used)")
        print(f"{prefix}    Free: {gpu['free_mem_gb']:.2f} GB")

        # Process-specific memory
        if gpu['id'] in usage['process_info']:
            proc_info = usage['process_info'][gpu['id']]
            print(f"{prefix}    This process: {proc_info['allocated_gb']:.2f} GB allocated, {proc_info['reserved_gb']:.2f} GB reserved")

    print(f"{prefix}{'─' * 80}")


def get_compact_gpu_summary() -> str:
    """
    Get a compact one-line GPU usage summary.

    Returns:
        String like "GPUs: 0: 23.4/80GB (29%), 1: 45.2/80GB (57%)"
    """
    usage = get_gpu_usage()

    if usage["gpu_count"] == 0:
        return "GPUs: None"

    gpu_strs = []
    for gpu in usage["gpus"]:
        if "error" in gpu:
            gpu_strs.append(f"{gpu['id']}: ERROR")
        else:
            gpu_strs.append(
                f"{gpu['id']}: {gpu['used_mem_gb']:.1f}/{gpu['total_mem_gb']:.0f}GB ({gpu['memory_utilization_pct']:.0f}%)"
            )

    return f"GPUs: {', '.join(gpu_strs)}"


class GPUMonitor:
    """
    Context manager for monitoring GPU usage at intervals during batch processing.

    Usage:
        monitor = GPUMonitor(print_every_n_batches=5)
        for batch_idx, batch in enumerate(batches):
            process_batch(batch)
            monitor.record_batch(batch_idx)
    """

    def __init__(self, print_every_n_batches: int = 5, prefix: str = "  "):
        """
        Args:
            print_every_n_batches: Print GPU stats every N batches
            prefix: Prefix for printed output (for indentation)
        """
        self.print_every_n = print_every_n_batches
        self.prefix = prefix
        self.batch_count = 0
        self.enabled = torch.cuda.is_available()

    def record_batch(self, batch_idx: Optional[int] = None) -> None:
        """
        Record completion of a batch and optionally print GPU usage.

        Args:
            batch_idx: Optional batch index (will use internal counter if not provided)
        """
        if not self.enabled:
            return

        self.batch_count += 1
        effective_idx = batch_idx if batch_idx is not None else self.batch_count

        if self.batch_count % self.print_every_n == 0:
            print_gpu_usage(batch_idx=effective_idx, prefix=self.prefix)

    def print_final(self) -> None:
        """Print final GPU usage summary."""
        if self.enabled:
            print_gpu_usage(prefix=self.prefix)
