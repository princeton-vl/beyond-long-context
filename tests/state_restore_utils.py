import random

import numpy as np
import pytest

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("State-restore tests require PyTorch to be installed") from exc


def require_cuda(model_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{model_name} state-restore test requires PyTorch with CUDA support.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_video_tensor(num_frames: int, seed: int, height: int = 48, width: int = 48) -> np.ndarray:
    """Return channels-first uint8 frames shaped (T, 3, H, W)."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)
    return np.transpose(frames, (0, 3, 1, 2))
