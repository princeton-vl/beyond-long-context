"""Shared memory calculation utilities."""

from __future__ import annotations

import torch
from typing import Optional


# Model size estimates in GB (approximate memory footprint at FP16/BF16)
# Increased by 25% for better OOM handling
MODEL_SIZES_GB = {
    "m3_agent": 57.75,
    "minicpm": 10.725,
    "glm45v": 248,
    "timechat": 28.875,
    "qwen3_full": 28.875,
    "qwen3_omni": 90.75,
    "internvl-3-5": 26.5,
    "internvl-3-5-thinking": 26.5,
    "internvl-3-5-38b": 96.75,
    "internvl-3-5-38b-thinking": 96.75,
    "internvl-3-5-30b-a3b": 80.375,
    "internvl-3-5-30b-a3b-thinking": 80.375,
    "minicpm-4-5": 18.48,
    "mimo-vl": 29.875,
    "Phi-4-multimodal": 18.75,
    "longvila": 28.875,
    "default": 28.875,
}


def calculate_max_gpu_mem(model_name: str, override: Optional[float] = None) -> float:
    """Calculate max_gpu_mem based on model size and available GPUs.

    Args:
        model_name: Model identifier (e.g., "internvl-3-5-38b")
        override: Optional manual override value

    Returns:
        Memory limit in GB per GPU

    Formula: (MODEL_SIZE_IN_GB / NUM_GPUS) * 1.05
    1.05x multiplier provides small overhead for loading and activations.
    """
    if override is not None:
        return override

    model_size_gb = MODEL_SIZES_GB.get(model_name, MODEL_SIZES_GB["default"])
    num_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    return (model_size_gb / num_gpus) * 1.05
