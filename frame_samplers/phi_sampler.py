"""Phi frame sampling using decord for reliable extraction."""

import logging
import os
import textwrap
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    VideoReader = None
    cpu = None

from .base_sampler import FrameSamplerInterface


logger = logging.getLogger(__name__)


class PhiFrameSampler(FrameSamplerInterface):
    """Frame sampler for Phi using decord to read all frames."""

    def __init__(self) -> None:
        pass

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """
        Sample ALL frames using decord for reliable extraction.

        Args:
            video_path: Path to video file
            fps: Target frames per second (ignored - reads all frames)
            min_pixels: Minimum pixel count for downsampling
            max_pixels: Maximum pixel count for downsampling
            max_frames: Optional hard cap on frames

        Returns:
            List[PIL.Image]: RGB images, uint8 (0-255)
        """
        if not DECORD_AVAILABLE:
            raise RuntimeError(
                "decord is required for Phi frame sampling. "
                "Install with: pip install decord"
            )

        resolved_path = self._resolve_video_path(video_path)

        # Read ALL frames
        vr = VideoReader(resolved_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        logger.info(f"[Phi] {resolved_path} -> {n} frames (reading ALL)")

        frame_idx = list(range(n))
        all_frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3), uint8

        # Downsample AFTER reading if needed
        if max_frames and n > max_frames:
            logger.info(f"[Phi] Downsampling from {n} to {max_frames} frames")
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
            all_frames = all_frames[indices]

        # Convert to List[PIL.Image]
        images = []
        for frame in all_frames:
            # frame is (H, W, 3), uint8
            img = Image.fromarray(frame).convert("RGB")
            images.append(img)

        return images

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

    def slice_frames(
        self,
        video_frames,
        start_idx: int,
        end_idx: int,
    ):
        if isinstance(video_frames, list):
            return video_frames[start_idx:end_idx]
        raise TypeError("Phi sampler expects a list of PIL images")

    def make_video_chunks(self, video_frames, fps: float, chunk_size: float, segment_start_time: float):
        """
        Build a list of (frames_chunk, time_start, time_end) tuples for the given video.

        If chunk_size is None or <= 0, return a single chunk representing the whole video.
        """
        total_frames = len(video_frames)
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
        return len(video_frames)

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

        # Duplicate frame for duration and stack, then convert to PIL.Image
        # to match sample_frames() output format.
        frames = np.stack([frame for _ in range(num_frames)], axis=0)
        formatted_frames = [Image.fromarray(f).convert("RGB") for f in frames]

        return formatted_frames
