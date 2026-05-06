"""Qwen-style frame sampling for tensor-based models."""

import logging
import math
import os
import textwrap
from pathlib import Path
from typing import Optional

# CRITICAL: pin the qwen_vl_utils backend BEFORE any import of that package
# so the cached choice (torchvision) sticks even though we no longer call
# process_vision_info ourselves. Removing this line will let downstream code
# fall back to torchcodec, which has the memory blowup we're avoiding.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_sampler import FrameSamplerInterface


logger = logging.getLogger(__name__)


class QwenFrameSampler(FrameSamplerInterface):
    """Frame sampler for Qwen and other tensor-based models.

    Decodes ALL frames via decord and downsamples afterwards (np.linspace).
    The decord backend is forced to a single thread (num_threads=1) to avoid
    a memory blowup we previously hit on long videos with multi-threaded
    decoding workers.
    """

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode every frame with decord, then downsample to max_frames.

        Args:
            video_path: Path to video file.
            fps: IGNORED. We always read every frame and use np.linspace to
                hit max_frames; downstream models do their own time alignment
                via VideoMetadata. Kept in the signature for interface parity
                with samplers that honor it.
            min_pixels: Minimum pixel count for resizing (resize up if smaller).
            max_pixels: Maximum pixel count for resizing (resize down if larger).
            max_frames: Optional hard cap on frames after decode.

        Returns:
            torch.Tensor: Shape (N, 3, H, W), dtype=uint8 (0-255), device=CPU.
        """
        resolved_path = self._resolve_video_path(video_path)
        return self._decode_with_decord_all_frames(
            resolved_path,
            max_frames=max_frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def _resolve_video_path(self, video_path: str) -> str:
        """Resolve missing or renamed video assets in synthetic datasets."""

        if not video_path:
            raise ValueError("video_path must be provided for frame sampling")

        if os.path.exists(video_path):
            return video_path

        candidate = Path(video_path)
        parent_dir = candidate.parent
        if not parent_dir.exists():
            raise FileNotFoundError(
                f"Video directory does not exist for requested path: {video_path}"
            )

        suffix = candidate.suffix or ".mp4"
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        pattern = f"{candidate.stem}*{suffix}"
        matches = [path for path in parent_dir.glob(pattern) if path.is_file()]

        if not matches:
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            ranked_candidates = [
                (path.stat(), path)
                for path in matches
            ]
            chosen = max(
                ranked_candidates,
                key=lambda item: (item[0].st_size, item[0].st_mtime),
            )[1]
        except OSError as exc:
            raise FileNotFoundError(
                f"Unable to stat candidate video files for {video_path}: {exc}"
            ) from exc

        logger.warning("Resolved %s to candidate video %s", video_path, chosen)
        return str(chosen)

    def _decode_with_decord_all_frames(
        self,
        video_path: str,
        max_frames: Optional[int],
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> torch.Tensor:
        """
        Decode ALL frames using decord.

        Returns:
            torch.Tensor: Shape (N, 3, H, W), dtype=uint8 (0-255), device=CPU
        """
        from decord import VideoReader, cpu as decord_cpu

        # num_threads=1: avoid worker-thread memory blowup on long videos.
        # Multi-threaded decord decoding has shown OOMs on the long-context
        # eval clips even when the underlying frame budget is small.
        vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=1)
        n = len(vr)
        logger.info(f"[Qwen] {video_path} -> {n} frames (reading ALL)")

        frame_idx = list(range(n))
        all_frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3), uint8

        # Downsample AFTER reading if needed
        if max_frames and n > max_frames:
            logger.info(f"[Qwen] Downsampling from {n} to {max_frames} frames")
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
            all_frames = all_frames[indices]

        # Convert to torch tensor: (N, H, W, 3) -> (N, 3, H, W)
        frame_tensor = torch.from_numpy(all_frames)
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).contiguous()
        frame_tensor = frame_tensor.to(dtype=torch.uint8)

        # Apply pixel constraints if specified
        if min_pixels is not None or max_pixels is not None:
            frame_tensor = self._apply_pixel_constraints(frame_tensor, min_pixels, max_pixels)

        return frame_tensor

    @staticmethod
    def _apply_pixel_constraints(
        frames: torch.Tensor,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> torch.Tensor:
        """Resize frames to honor pixel-count constraints if provided."""

        if not torch.is_tensor(frames) or frames.ndim != 4:
            return frames

        height = frames.shape[2]
        width = frames.shape[3]
        if height <= 0 or width <= 0:
            return frames

        area = height * width
        target_area: Optional[int] = None

        if max_pixels and area > max_pixels:
            target_area = max_pixels
        elif min_pixels and area < min_pixels:
            target_area = min_pixels

        if not target_area or target_area <= 0 or target_area == area:
            return frames

        scale = math.sqrt(target_area / float(area))
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))

        if new_height == height and new_width == width:
            return frames

        # F.interpolate works directly on float tensors; the previous
        # div_(255)/mul_(255) round-trip was a no-op for bilinear sampling.
        resized = F.interpolate(
            frames.float(),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.to(dtype=torch.uint8)

    def make_video_chunks(self, video_frames, fps: float, chunk_size: float, segment_start_time: float):
        """
        Build a list of (frames_chunk, time_start, time_end) tuples for the given video.

        If chunk_size is None or <= 0, return a single chunk representing the whole video.
        """
        total_frames = video_frames.shape[0]
        total_duration = total_frames / fps

        # No chunking: single chunk covering the entire video
        if not chunk_size or chunk_size <= 0:
            return [(video_frames, segment_start_time, segment_start_time + total_duration)]

        # Chunking
        frames_per_chunk = max(1, int(chunk_size * fps))
        chunks = []
        for i in range(0, total_frames, frames_per_chunk):
            chunk_of_frames = video_frames[i : i + frames_per_chunk]
            chunk_start_on_timeline = segment_start_time + (i / fps)
            chunk_end_on_timeline = segment_start_time + ((i + len(chunk_of_frames)) / fps)
            chunks.append((chunk_of_frames, chunk_start_on_timeline, chunk_end_on_timeline))

        return chunks

    def get_frame_count(self, video_frames) -> int:
        """Get frame count for np.ndarray format."""
        if hasattr(video_frames, 'shape'):
            return video_frames.shape[0]
        return 0

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 2.0,
                          frame_width: int = 640, frame_height: int = 480) -> np.ndarray:
        """
        Create text frames in numpy array format for Qwen and tensor-based models.

        Args:
            text: Text to render
            fps: Frames per second for timing calculations
            duration_seconds: Duration of text video in seconds
            frame_width: Width of generated frames
            frame_height: Height of generated frames

        Returns:
            Stacked frames as numpy array with shape (num_frames, height, width, channels)
        """
        # Calculate number of frames needed
        num_frames = max(1, int(duration_seconds * fps))

        # Create black frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Text rendering settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)  # White text
        font_thickness = 2
        max_chars_per_line = 40
        max_lines = 8

        # Wrap text to fit within frame
        wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)[:max_lines]

        # Calculate text positioning
        line_height = 30
        total_text_height = len(wrapped_lines) * line_height
        start_y = (frame_height - total_text_height) // 2 + line_height

        # Render each line
        for i, line in enumerate(wrapped_lines):
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            x = (frame_width - text_width) // 2
            y = start_y + i * line_height

            # Render text
            cv2.putText(frame, line, (x, y), font, font_scale,
                       font_color, font_thickness, cv2.LINE_AA)

        # Duplicate frame for duration and stack
        frames = np.stack([frame for _ in range(num_frames)], axis=0)
        # Convert from (N, H, W, C) to (N, C, H, W) to match video frame format
        frame_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return frame_tensor.to(dtype=torch.uint8)

    def slice_frames(
        self,
        video_frames,
        start_idx: int,
        end_idx: int,
    ):
        if torch.is_tensor(video_frames):
            return video_frames[start_idx:end_idx].contiguous()
        raise TypeError("Qwen sampler expects torch.Tensor outputs")
