from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image


def _to_pil_list(frames: torch.Tensor) -> list[Image.Image]:
    """Convert a (N,H,W,C) float tensor in [0,1] to a list of PIL images."""

    arr = (frames.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return [Image.fromarray(sample) for sample in arr]


class FeatureBackbone(abc.ABC):
    """Abstract base for frame/clip embedding backbones."""

    name: str
    embed_dim: int

    def __init__(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype

    @abc.abstractmethod
    def embed_frames(self, frames: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Return embeddings shaped (N, D) for the provided frames."""


class DummyBackbone(FeatureBackbone):
    """Lightweight backbone for tests that just averages RGB values."""

    name = "dummy"

    def __init__(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        super().__init__(device=device, dtype=dtype)
        self.embed_dim = 3

    def embed_frames(self, frames: torch.Tensor, batch_size: int) -> torch.Tensor:  # noqa: D401
        if frames.numel() == 0:
            return torch.zeros(0, self.embed_dim)
        flat = frames.view(frames.size(0), -1, 3)
        mean = flat.mean(dim=1)
        return mean.to(torch.float32)


class DinoVisionBackbone(FeatureBackbone):
    """HuggingFace DINO vision transformer applied frame-by-frame."""

    def __init__(
        self,
        *,
        repo_id: str,
        device: torch.device,
        dtype: torch.dtype,
        use_fast_processor: bool,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        from transformers import AutoImageProcessor, AutoModel  # Lazy import

        self.processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=use_fast_processor)
        self.model = AutoModel.from_pretrained(repo_id).to(device=device, dtype=dtype)
        self.model.eval()
        self.embed_dim = int(self.model.config.hidden_size)
        self.repo_id = repo_id

    @torch.inference_mode()
    def embed_frames(self, frames: torch.Tensor, batch_size: int) -> torch.Tensor:
        images = _to_pil_list(frames)
        if not images:
            return torch.zeros(0, self.embed_dim)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            result = self.model(**inputs)
            hidden = result.last_hidden_state[:, 0, :].to(torch.float32)
            outputs.append(hidden.cpu())
        return torch.cat(outputs, dim=0)


class VideoMAEBackbone(FeatureBackbone):
    """HuggingFace VideoMAE model that embeds clips of consecutive frames."""

    def __init__(
        self,
        *,
        repo_id: str,
        device: torch.device,
        dtype: torch.dtype,
        clip_frames: int | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.processor = VideoMAEImageProcessor.from_pretrained(repo_id)
        self.model = VideoMAEModel.from_pretrained(repo_id).to(device=device, dtype=dtype)
        self.model.eval()
        self.clip_frames = clip_frames or int(self.model.config.num_frames)
        self.embed_dim = int(self.model.config.hidden_size)
        self.repo_id = repo_id

    def _prepare_clip(self, frames: torch.Tensor) -> list[Image.Image]:
        if frames.size(0) < self.clip_frames:
            pad_count = self.clip_frames - frames.size(0)
            pad = frames[-1:].repeat(pad_count, 1, 1, 1)
            frames = torch.cat([frames, pad], dim=0)
        elif frames.size(0) > self.clip_frames:
            frames = frames[: self.clip_frames]
        return _to_pil_list(frames)

    @torch.inference_mode()
    def embed_frames(self, frames: torch.Tensor, batch_size: int) -> torch.Tensor:
        if frames.numel() == 0:
            return torch.zeros(0, self.embed_dim)
        clips: list[list[Image.Image]] = []
        for start in range(0, frames.size(0), self.clip_frames):
            clip = frames[start : start + self.clip_frames]
            if clip.size(0) == 0:
                continue
            clips.append(self._prepare_clip(clip))
        outputs: list[torch.Tensor] = []
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            result = self.model(**inputs)
            pooled = getattr(result, "pooler_output", None)
            if pooled is None:
                pooled = result.last_hidden_state[:, 0, :]
            outputs.append(pooled.to(torch.float32).cpu())
        return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class BackboneConfig:
    builder: Callable[..., FeatureBackbone]
    description: str


def _bool_override(overrides: dict[str, str], key: str, default: bool) -> bool:
    raw = overrides.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean override for {key}: {raw}")


def build_backbone(
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    overrides: dict[str, str],
) -> FeatureBackbone:
    normalized = name.lower()
    if normalized == "dummy":
        return DummyBackbone(device=device, dtype=dtype)
    if normalized.startswith("dinov2"):
        repo = overrides.get("hf_repo", f"facebook/{normalized}")
        use_fast = _bool_override(overrides, "use_fast_processor", True)
        return DinoVisionBackbone(
            repo_id=repo,
            device=device,
            dtype=dtype,
            use_fast_processor=use_fast,
        )
    if normalized in {"videomae-base", "videomae-large"}:
        repo = overrides.get("hf_repo", f"MCG-NJU/{normalized}")
        clip_frames = overrides.get("clip_frames")
        return VideoMAEBackbone(
            repo_id=repo,
            device=device,
            dtype=dtype,
            clip_frames=int(clip_frames) if clip_frames else None,
        )
    raise ValueError(
        f"Unknown backbone '{name}'. Supported options: dinov2-small, dinov2-base, videomae-base, videomae-large, dummy"
    )
