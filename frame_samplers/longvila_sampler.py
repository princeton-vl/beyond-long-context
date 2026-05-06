"""LongVILA-style frame sampling with PIL image outputs."""

from __future__ import annotations

import logging
import math
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    VideoReader = None
    cpu = None

from .base_sampler import FrameSamplerInterface


logger = logging.getLogger(__name__)


class LongVILAFrameSampler(FrameSamplerInterface):
    """Frame sampler that mimics LongVILA's media extraction helpers."""

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample frames as PIL images from a video file or directory."""

        if not video_path:
            raise ValueError("video_path must be provided for frame sampling")

        resolved_path = Path(video_path).expanduser()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Video source does not exist: {video_path}")

        frame_cap = self._resolve_frame_cap(max_frames, kwargs)

        if resolved_path.is_dir():
            frames = self._load_from_directory(resolved_path, frame_cap)
        else:
            frames = self._load_from_video_file(resolved_path, frame_cap)

        if not frames:
            raise RuntimeError(f"No frames decoded from video source: {video_path}")

        return self._stack_frames(frames)

    def make_video_chunks(
        self,
        video_frames: torch.Tensor,
        fps: float,
        chunk_size: float,
        segment_start_time: float,
    ):
        """Split a list of PIL frames into timeline-aligned chunks."""

        frame_tensor = self._ensure_tensor(video_frames)
        total_frames = int(frame_tensor.shape[0])
        if total_frames == 0:
            return []

        total_duration = total_frames / fps if fps > 0 else float(total_frames)

        if not chunk_size or chunk_size <= 0:
            return [(
                frame_tensor,
                segment_start_time,
                segment_start_time + total_duration,
            )]

        frames_per_chunk = max(1, int(chunk_size * fps)) if fps > 0 else max(1, int(chunk_size))
        chunks = []
        for idx in range(0, total_frames, frames_per_chunk):
            chunk_frames = frame_tensor[idx : idx + frames_per_chunk]
            chunk_start = segment_start_time + (idx / fps if fps > 0 else idx)
            chunk_end = segment_start_time + (
                (idx + len(chunk_frames)) / fps if fps > 0 else idx + len(chunk_frames)
            )
            chunks.append((chunk_frames, chunk_start, chunk_end))

        return chunks

    def get_frame_count(self, video_frames) -> int:
        """Return the number of frames contained in a list of PIL images."""

        if isinstance(video_frames, torch.Tensor) and video_frames.ndim >= 1:
            return int(video_frames.shape[0])
        return 0

    def slice_frames(
        self,
        video_frames: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        tensor = self._ensure_tensor(video_frames)
        return tensor[start_idx:end_idx].contiguous()

    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 2.0,
        frame_width: int = 448,
        frame_height: int = 448,
    ) -> torch.Tensor:
        """Create list of PIL frames with the provided text rendered on them."""

        num_frames = max(1, int(math.ceil(duration_seconds * fps)))
        text = text or ""

        base_frame = Image.new("RGB", (frame_width, frame_height), color="black")
        draw = ImageDraw.Draw(base_frame)
        font = ImageFont.load_default()

        max_chars_per_line = max(1, frame_width // 16)
        wrapped_lines = textwrap.wrap(text, width=max_chars_per_line) or [""]
        max_lines = frame_height // 24
        wrapped_lines = wrapped_lines[:max_lines]

        line_height = max(font.size + 4, 20)
        total_text_height = len(wrapped_lines) * line_height
        start_y = max(0, (frame_height - total_text_height) // 2)

        for idx, line in enumerate(wrapped_lines):
            text_width = self._measure_text_width(draw, line, font)
            pos_x = max(0, (frame_width - text_width) // 2)
            pos_y = start_y + idx * line_height
            draw.text((pos_x, pos_y), line, fill="white", font=font)

        frames = [base_frame.copy() for _ in range(num_frames)]
        return self._stack_frames(frames)

    @staticmethod
    def _resolve_frame_cap(max_frames: Optional[int], kwargs: dict) -> Optional[int]:
        """Resolve preferred frame cap allowing overrides from kwargs."""

        frame_cap = max_frames if max_frames is not None else kwargs.get("max_frames")
        if frame_cap is None and "num_frames" in kwargs:
            frame_cap = kwargs["num_frames"]
        if frame_cap is not None and frame_cap <= 0:
            frame_cap = None
        return int(frame_cap) if frame_cap is not None else None

    def _load_from_directory(self, directory: Path, frame_cap: Optional[int]) -> List[Image.Image]:
        """Load uniformly sampled frames from a directory of pre-extracted images."""

        image_files = sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in self._IMAGE_EXTENSIONS
            ]
        )

        if not image_files:
            raise RuntimeError(f"No image frames found in directory: {directory}")

        target_count = frame_cap if frame_cap is not None else len(image_files)
        target_count = min(len(image_files), max(1, target_count))
        indices = self._select_indices(len(image_files), target_count)

        frames: List[Image.Image] = []
        for idx in indices:
            try:
                with Image.open(image_files[idx]) as img:
                    frames.append(img.convert("RGB"))
            except OSError as exc:
                logger.warning("Failed to load frame %s: %s", image_files[idx], exc)
        return frames

    def _load_from_video_file(
        self,
        video_path: Path,
        frame_cap: Optional[int],
    ) -> List[Image.Image]:
        """Decode ALL frames from video using decord."""

        if not DECORD_AVAILABLE:
            raise RuntimeError(
                "decord is required for LongVILA frame sampling. "
                "Install with: pip install decord"
            )

        # Read ALL frames
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        n = len(vr)
        logger.info(f"[LongVILA] {video_path} -> {n} frames (reading ALL)")

        frame_idx = list(range(n))
        all_frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3), uint8

        # Downsample AFTER reading if needed
        if frame_cap and n > frame_cap:
            logger.info(f"[LongVILA] Downsampling from {n} to {frame_cap} frames")
            indices = np.linspace(0, n - 1, frame_cap, dtype=int)
            all_frames = all_frames[indices]

        # Convert to List[PIL.Image]
        frames = []
        for frame in all_frames:
            frames.append(Image.fromarray(frame).convert("RGB"))

        return frames

    @staticmethod
    def _select_indices(source_length: int, target_count: int) -> np.ndarray:
        """Compute evenly spaced frame indices."""

        target_count = max(1, min(source_length, target_count))
        if source_length == 1:
            return np.array([0], dtype=np.int64)
        return np.linspace(0, source_length - 1, target_count, dtype=np.int64)

    def _ensure_tensor(self, video_frames) -> torch.Tensor:
        """Normalize incoming frame containers to a torch tensor."""

        if isinstance(video_frames, torch.Tensor):
            return video_frames
        if isinstance(video_frames, (list, tuple)):
            return self._stack_frames(list(video_frames))
        raise TypeError(f"Unsupported frame container type: {type(video_frames)}")

    @staticmethod
    def _measure_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
        """Measure text width in a Pillow-version-agnostic manner."""

        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        if hasattr(draw, "textsize"):
            width, _ = draw.textsize(text, font=font)
            return width
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        if hasattr(font, "getsize"):
            width, _ = font.getsize(text)
            return width
        return len(text) * max(1, font.size)

    @staticmethod
    def _stack_frames(frames: List[Image.Image]) -> torch.Tensor:
        """Stack a list of PIL frames into a torch tensor."""

        if not frames:
            raise ValueError("No frames available to stack")

        tensor_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                arr = np.array(frame.convert("RGB"), copy=True)
            elif isinstance(frame, np.ndarray):
                arr = np.array(frame, copy=True)
            elif torch.is_tensor(frame):
                tensor = frame
                if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
                    tensor_frames.append(tensor[:3].to(dtype=torch.uint8).cpu())
                    continue
                arr = tensor.cpu().numpy()
            else:
                raise TypeError(f"Unsupported frame type: {type(frame)}")

            tensor_frames.append(torch.from_numpy(arr).to(dtype=torch.uint8).permute(2, 0, 1))

        return torch.stack(tensor_frames, dim=0)
