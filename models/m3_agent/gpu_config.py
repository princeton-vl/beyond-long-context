"""GPU allocation helper used by the M3-Agent integration."""

from __future__ import annotations

import itertools
import logging
from typing import Dict, Iterable, List

import torch


LOGGER = logging.getLogger(__name__)
_GIB = 1024 ** 3


class GPUConfig:
    """Track GPU availability and coordinate component placements.

    The class intentionally performs no silent fallbacks: all resource
    violations raise ``RuntimeError`` so callers can surface actionable
    errors to operators.  Memory figures are reported in GiB to match SLURM
    reservations and project documentation.
    """

    def __init__(self) -> None:
        # Get device count without initializing CUDA to avoid conflicts with vLLM multiprocessing
        # Try to get from environment first (SLURM or CUDA_VISIBLE_DEVICES)
        import os

        # Check SLURM or CUDA_VISIBLE_DEVICES to count GPUs without initializing CUDA
        if "SLURM_GPUS_ON_NODE" in os.environ:
            self.device_count = int(os.environ["SLURM_GPUS_ON_NODE"])
        elif "SLURM_GPUS" in os.environ:
            self.device_count = int(os.environ["SLURM_GPUS"])
        elif "SLURM_STEP_GPUS" in os.environ:
            # SLURM_STEP_GPUS is like "0,1,2" or "0-2"
            step_gpus = os.environ["SLURM_STEP_GPUS"]
            if "-" in step_gpus:
                # Range format like "0-2"
                parts = step_gpus.split("-")
                self.device_count = int(parts[1]) - int(parts[0]) + 1
            else:
                # Comma-separated format like "0,1,2"
                self.device_count = len(step_gpus.split(","))
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            if visible:
                self.device_count = len(visible.split(","))
            else:
                # If CUDA_VISIBLE_DEVICES is empty, no GPUs
                self.device_count = 0
        else:
            # Fallback: use torch.cuda but this may initialize CUDA
            if not torch.cuda.is_available():
                raise RuntimeError("M3-Agent requires at least one CUDA-capable GPU.")
            self.device_count = torch.cuda.device_count()

        if self.device_count == 0:
            raise RuntimeError("No GPUs detected; cannot continue.")

        self.allocations: Dict[str, Dict[str, float | int | List[int]]] = {}
        # IMPORTANT: Don't collect GPU inventory yet - wait until after vLLM initializes
        # to avoid CUDA initialization conflicts with vLLM multiprocessing
        self.gpu_info: Dict[int, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # GPU inventory helpers
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Refresh cached GPU memory information.

        This will initialize CUDA if not already initialized.
        Only call this AFTER vLLM has been initialized to avoid multiprocessing conflicts.
        """

        self.gpu_info = self._collect_gpu_inventory()

    def _collect_gpu_inventory(self) -> Dict[int, Dict[str, float]]:
        """Snapshot total/free memory for every visible GPU."""

        inventory: Dict[int, Dict[str, float]] = {}
        for idx in range(self.device_count):
            props = torch.cuda.get_device_properties(idx)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            except RuntimeError:
                total_bytes = float(props.total_memory)
                allocated_bytes = float(torch.cuda.memory_allocated(idx))
                free_bytes = max(total_bytes - allocated_bytes, 0.0)
            inventory[idx] = {
                "name": props.name,
                "total_gb": total_bytes / _GIB,
                "free_gb": free_bytes / _GIB,
            }
        return inventory

    # ------------------------------------------------------------------
    # Allocation routines
    # ------------------------------------------------------------------
    def allocate_vllm(self, model_size_gb: float = 72.0) -> Dict[str, object]:
        """Allocate GPUs for the control LLM.

        Strategy: Use 1, 2, or 4 GPUs with tensor parallelism.
        Leave at least 1 GPU free for InsightFace + Qwen VLM when possible.

        This method does NOT initialize CUDA to avoid conflicts with vLLM multiprocessing.
        """

        # Prefer 2 or 4 GPUs for vLLM, but allow single GPU if needed
        # IMPORTANT: Never use TP=3 - M3-Agent-Control has 64 attention heads (not divisible by 3)
        # With 4+ GPUs: use 2 or 4 for vLLM (TP=2 or TP=4), leave rest for InsightFace+Qwen
        # With 3 GPUs: use 2 for vLLM (TP=2), 1 for InsightFace+Qwen
        # With 2 GPUs: use 2 for vLLM (TP=2), share 1 for InsightFace+Qwen
        # With 1 GPU: use 1 for vLLM (TP=1), share for InsightFace+Qwen
        if self.device_count >= 4:
            preferred_tp = [2, 4]  # Prefer TP=2, fallback to TP=4 (both divisible by 64 heads)
        elif self.device_count >= 2:
            preferred_tp = [2]  # Always use TP=2 with 2-3 GPUs
        elif self.device_count == 1:
            preferred_tp = [1]  # Single GPU mode
        else:
            raise RuntimeError(f"M3-Agent requires at least 1 GPU, got {self.device_count}")

        for tp_size in preferred_tp:
            if tp_size > self.device_count:
                continue

            # Use simple allocation: first N GPUs for vLLM
            gpu_ids = list(range(tp_size))

            allocation = {
                "gpus": gpu_ids,
                "tensor_parallel_size": tp_size,
                "total_free_gb": model_size_gb,  # Estimated
                "efficiency_score": 1.0,
            }

            self.allocations["vllm"] = allocation
            LOGGER.info(
                "Allocated vLLM across GPUs %s (tensor parallel = %d)",
                allocation["gpus"],
                tp_size,
            )
            return allocation

        raise RuntimeError(
            f"Cannot allocate vLLM with {self.device_count} GPUs. "
            f"M3-Agent requires at least 2 GPUs for vLLM."
        )

    def allocate_insightface(self, min_memory_gb: float = 2.0) -> Dict[str, object]:
        """Reserve a GPU for InsightFace, preferring devices not used by vLLM.

        Strategy: Prefer the last available GPU to keep GPU 0 clear.

        This method does NOT initialize CUDA.
        """

        occupied = set(self.allocations.get("vllm", {}).get("gpus", []))

        # Prefer higher GPU IDs (last GPUs) to avoid GPU 0
        # This helps prevent CUDA from defaulting to GPU 0
        available_gpus = [gpu_id for gpu_id in range(self.device_count) if gpu_id not in occupied]

        if available_gpus:
            # Pick the highest GPU ID
            gpu_id = max(available_gpus)
            allocation = {"gpu": gpu_id}
            self.allocations["insightface"] = allocation
            LOGGER.info("InsightFace scheduled on GPU %d", gpu_id)
            return allocation

        # Fall back to sharing with vLLM if needed (use highest GPU ID)
        if self.device_count > 0:
            gpu_id = self.device_count - 1
            allocation = {"gpu": gpu_id}
            self.allocations["insightface"] = allocation
            LOGGER.warning("InsightFace sharing GPU %d with vLLM", gpu_id)
            return allocation

        raise RuntimeError(
            "Unable to reserve a GPU for InsightFace; no GPUs available."
        )

    def allocate_qwen(self, min_memory_gb: float = 16.0, max_video_frames: int = 300) -> Dict[str, object]:
        """Reserve a GPU for the memorisation model including video buffers.

        Strategy: Can share with InsightFace (small footprint), but prefer to avoid vLLM GPUs.
        When limited GPUs (2-3), can share with vLLM as fallback.

        This method does NOT initialize CUDA.
        """

        # Estimate video memory needs (simplified without GPU info)
        video_budget = 12.0  # Conservative estimate for most scenarios

        # Avoid vLLM GPUs, but can share with InsightFace
        vllm_gpus: set[int] = set()
        vllm_alloc = self.allocations.get("vllm", {})
        vllm_gpus.update(vllm_alloc.get("gpus", []))

        # Try to share GPU with InsightFace first (both are small, can coexist)
        insightface_alloc = self.allocations.get("insightface", {})
        insightface_gpu = insightface_alloc.get("gpu")
        if isinstance(insightface_gpu, int) and insightface_gpu not in vllm_gpus:
            allocation = {
                "gpu": insightface_gpu,
                "video_memory_reserved_gb": video_budget,
            }
            self.allocations["qwen"] = allocation
            LOGGER.info(
                "Qwen sharing GPU %d with InsightFace (%.2f GiB video reserve)",
                insightface_gpu,
                video_budget,
            )
            return allocation

        # Otherwise find a non-vLLM GPU
        for gpu_id in range(self.device_count):
            if gpu_id in vllm_gpus:
                continue
            allocation = {
                "gpu": gpu_id,
                "video_memory_reserved_gb": video_budget,
            }
            self.allocations["qwen"] = allocation
            LOGGER.info(
                "Qwen on dedicated GPU %d with %.2f GiB video reserve",
                gpu_id,
                video_budget,
            )
            return allocation

        # FALLBACK: With limited GPUs (1-3), allow sharing with vLLM
        # H200s have 140GB memory - vLLM uses ~70GB, leaving room for Qwen (~30GB)
        if self.device_count <= 3 and vllm_gpus:
            # Share with the last vLLM GPU (highest GPU ID)
            gpu_id = max(vllm_gpus)
            allocation = {
                "gpu": gpu_id,
                "video_memory_reserved_gb": video_budget,
            }
            self.allocations["qwen"] = allocation
            LOGGER.warning(
                "Qwen sharing GPU %d with vLLM and InsightFace (limited GPU setup, %.2f GiB video reserve)",
                gpu_id,
                video_budget,
            )
            return allocation

        raise RuntimeError(
            "Unable to reserve capacity for Qwen memorisation. "
            f"All {self.device_count} GPUs are allocated to vLLM."
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_allocation(self, component: str) -> Dict[str, object]:
        """Return the allocation dictionary for ``component``."""

        if component not in self.allocations:
            raise KeyError(f"GPU allocation for '{component}' has not been created yet.")
        return self.allocations[component]

    def print_summary(self) -> None:
        """Emit a summary of current allocations to stdout and the logger."""

        summary = {
            component: allocation
            for component, allocation in self.allocations.items()
        }
        LOGGER.info("GPU allocation summary: %s", summary)
        print(f"GPU allocation summary: {summary}")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _iter_gpu_combinations(self, tp_size: int) -> Iterable[tuple[float, List[int]]]:
        """Yield GPU combinations sorted by total free memory."""

        combinations = (
            (sum(self.gpu_info[gpu_id]["free_gb"] for gpu_id in combo), list(combo))
            for combo in itertools.combinations(range(self.device_count), tp_size)
        )
        return sorted(combinations, key=lambda item: item[0], reverse=True)

    def _calculate_video_memory_needs(self, max_frames: int) -> float:
        """Estimate the video buffer required for memorisation."""

        average_total = sum(info["total_gb"] for info in self.gpu_info.values()) / len(self.gpu_info)

        if average_total >= 40:
            base_memory = 6.0
            frame_factor = 0.02
        elif average_total >= 24:
            base_memory = 4.0
            frame_factor = 0.015
        else:
            base_memory = 2.5
            frame_factor = 0.01

        estimated = base_memory + (max_frames * frame_factor)
        return min(estimated, average_total * 0.3)
