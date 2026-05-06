"""Frame sampler for MiniCPM-V 4.5 built on the official 3D-resampler pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

try:  # pragma: no cover - optional runtime dependency
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None

from .base_sampler import FrameSamplerInterface

logger = logging.getLogger(__name__)

MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 6
TIME_SCALE = 0.1


def map_to_nearest_scale(values: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    """Map each value to the nearest entry in the provided scale grid."""

    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr: Sequence[int], size: int) -> List[List[int]]:
    """Group array into fixed-size chunks preserving order."""

    return [list(arr[i : i + size]) for i in range(0, len(arr), size)]


def encode_video(
    video_path: str,
    choose_fps: int = 3,
    force_packing: Optional[int] = None,
    frame_budget: Optional[int] = None,
):
    """Reference implementation from OpenBMB MiniCPM-V-4.5 demo."""

    if VideoReader is None or cpu is None:
        raise RuntimeError("MiniCPM 4.5 sampler requires decord. Please install it.")

    def uniform_sample(indices: Sequence[int], n: int) -> List[int]:
        if n <= 0:
            return []
        n = min(len(indices), n)
        if n == 0:
            return []
        gap = len(indices) / n
        return [int(i * gap + gap / 2) for i in range(n)]

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)

    # Calculate duration from actual frame count
    # NOTE: Removed unique frame detection - it was incorrectly treating
    # the list of unique frame INDICES as a frame COUNT, causing videos
    # to report 1-frame duration when they had 8+ frames.
    # The downsampling logic below will handle any padding correctly.

    video_duration = total_frames / fps if fps > 0 else 0.0

    capped_frames = MAX_NUM_FRAMES * MAX_NUM_PACKING
    desired_frames = choose_fps * video_duration if video_duration > 0 else total_frames
    if frame_budget is not None and frame_budget > 0:
        desired_frames = min(desired_frames, float(frame_budget))
    if desired_frames > capped_frames + 1e-6:
        raise ValueError(
            f"MiniCPM-V-4.5 sampler received fps={choose_fps} over {video_duration:.2f}s, "
            f"which exceeds the maximum budget of {capped_frames} frames. Reduce fps or shorten the clip."
        )

    packing_nums = max(1, math.ceil(desired_frames / MAX_NUM_FRAMES))
    choose_frames = max(1, min(int(round(desired_frames)), total_frames))

    # Sample uniformly from all frames
    frame_idx = list(range(total_frames))
    frame_idx = np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(max(1, int(force_packing)), MAX_NUM_PACKING)

    logger.debug(
        "MiniCPM45 encode_video: %s duration=%.2fs frames=%d packing=%d",
        video_path,
        video_duration,
        len(frame_idx),
        packing_nums,
    )

    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps if fps > 0 else frame_idx
    scale = np.arange(0, max(video_duration, TIME_SCALE), TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    pil_frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)

    return pil_frames, frame_ts_id_group


@dataclass
class MiniCPM45Sample:
    frames: List[Image.Image]
    temporal_ids: List[List[int]]
    fps: float


class MiniCPM45FrameSampler(FrameSamplerInterface):
    """Sampler producing MiniCPM 4.5 ready frame/temporal-id pairs."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 3,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> MiniCPM45Sample:
        resolved = self._resolve_video_path(video_path)
        force_packing = kwargs.get("force_packing")
        # Align fallback fps with encode_video's default (3, the official
        # MiniCPM-V-4.5 reference rate). Previously a caller passing fps=0
        # would sample at 3 but tag the Sample with fps=1.0, breaking
        # downstream timing math.
        chosen_fps = int(fps) if fps and fps > 0 else 3
        chosen_fps = max(1, chosen_fps)
        frames, temporal_ids = encode_video(
            resolved,
            choose_fps=chosen_fps,
            force_packing=force_packing,
            frame_budget=max_frames,
        )
        return MiniCPM45Sample(
            frames=frames,
            temporal_ids=temporal_ids,
            fps=float(chosen_fps),
        )

    def make_video_chunks(
        self,
        video_frames: MiniCPM45Sample,
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0,
    ) -> List[tuple]:
        """Return the sample as a single (frames, start, end) chunk.

        NOTE: ``chunk_size`` is intentionally ignored. MiniCPM-V-4.5's 3D
        resampler couples ``frames`` and ``temporal_ids`` and won't
        re-tile cleanly across arbitrary chunk boundaries, so callers that
        need finer chunking must re-sample the source video instead.
        """
        if not isinstance(video_frames, MiniCPM45Sample):
            return [(video_frames, segment_start_time, segment_start_time)]

        frame_rate = fps if fps and fps > 0 else video_frames.fps
        total_duration = len(video_frames.frames) / frame_rate if frame_rate > 0 else 0.0
        return [
            (
                video_frames,
                segment_start_time,
                segment_start_time + total_duration,
            )
        ]

    def get_frame_count(self, video_frames: MiniCPM45Sample) -> int:
        if isinstance(video_frames, MiniCPM45Sample):
            return len(video_frames.frames)
        return 0

    def slice_frames(
        self,
        video_frames: MiniCPM45Sample,
        start_idx: int,
        end_idx: int,
    ) -> MiniCPM45Sample:
        if isinstance(video_frames, MiniCPM45Sample):
            sliced_frames = video_frames.frames[start_idx:end_idx]
            temporal_ids: List[List[int]] = []
            if video_frames.temporal_ids:
                flat_ids: List[int] = []
                for group in video_frames.temporal_ids:
                    flat_ids.extend(group)
                sliced_ids = flat_ids[start_idx:end_idx]
                temporal_ids = [[value] for value in sliced_ids]
            return MiniCPM45Sample(
                frames=sliced_frames,
                temporal_ids=temporal_ids,
                fps=video_frames.fps,
            )
        if isinstance(video_frames, list):
            return video_frames[start_idx:end_idx]
        raise TypeError("MiniCPM45 sampler expected MiniCPM45Sample for slicing")

    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 1.0,
        frame_width: int = 448,
        frame_height: int = 448,
    ) -> List[Image.Image]:
        frame_count = max(1, int(duration_seconds * fps))
        blank = Image.new("RGB", (frame_width, frame_height), color=(0, 0, 0))
        return [blank.copy() for _ in range(frame_count)]

    def _resolve_video_path(self, video_path: str) -> str:
        path = Path(video_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Video file not found: {video_path}")
