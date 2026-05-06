"""InternVL 3.5 frame sampler built directly on the base sampler interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import textwrap
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

try:  # pragma: no cover - decord is optional at runtime
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover
    VideoReader = None
    cpu = None

from .base_sampler import FrameSamplerInterface

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class InternVLSampledVideo:
    """Container for InternVL-ready video data."""

    pixel_values: torch.Tensor
    num_patches_list: List[int]
    fps: float


class InternVLFrameSampler(FrameSamplerInterface):
    """Sampler that mirrors the official InternVL preprocessing pipeline."""

    # Per-frame square crop size used by the InternVL preprocessor.
    DEFAULT_INPUT_SIZE = 448
    # Disable aspect-ratio tiling for video frames; we ship one thumbnail
    # patch per frame. Raising this re-enables dynamic_preprocess() splits
    # and changes the patch count each frame contributes.
    DEFAULT_MAX_TILES = 1
    # Frame budget if the caller doesn't pass max_frames or fps.
    DEFAULT_SEGMENTS = 32

    def __init__(self) -> None:
        # No instance state required; load_video rebuilds its transform
        # locally so it can honor a per-call input_size if it ever differs.
        pass

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,  # Unused, kept for interface parity
        max_pixels: Optional[int] = None,  # Unused, kept for interface parity
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> InternVLSampledVideo:
        if VideoReader is None or cpu is None:
            raise RuntimeError(
                "InternVLFrameSampler requires decord. Please install it to continue."
            )

        resolved_path = self._resolve_video_path(video_path)
        num_segments = self._segments_from_request(
            resolved_path,
            requested_fps=fps,
            max_frames=max_frames,
        )
        pixel_values, num_patches_list = load_video(
            resolved_path,
            input_size=self.DEFAULT_INPUT_SIZE,
            max_num=self.DEFAULT_MAX_TILES,
            num_segments=num_segments,
        )
        fps_value = float(fps) if fps and fps > 0 else 1.0
        return InternVLSampledVideo(pixel_values=pixel_values, num_patches_list=num_patches_list, fps=fps_value)

    def make_video_chunks(
        self,
        video_frames: InternVLSampledVideo,
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0,
    ) -> List[tuple]:
        if not isinstance(video_frames, InternVLSampledVideo):
            duration = 0.0
            return [(video_frames, segment_start_time, segment_start_time + duration)]

        frame_count = len(video_frames.num_patches_list)
        if frame_count == 0:
            return []

        frame_rate = fps if fps and fps > 0 else video_frames.fps
        total_duration = frame_count / frame_rate

        if not chunk_size or chunk_size <= 0:
            return [
                (
                    video_frames,
                    segment_start_time,
                    segment_start_time + total_duration,
                )
            ]

        frames_per_chunk = max(1, int(round(chunk_size * frame_rate)))
        chunks = []
        start_frame = 0
        while start_frame < frame_count:
            end_frame = min(frame_count, start_frame + frames_per_chunk)
            chunk_frames = self._slice_clip(video_frames, start_frame, end_frame)
            chunk_start = segment_start_time + (start_frame / frame_rate)
            chunk_end = segment_start_time + (end_frame / frame_rate)
            chunks.append((chunk_frames, chunk_start, chunk_end))
            start_frame = end_frame
        return chunks

    def get_frame_count(self, video_frames: InternVLSampledVideo) -> int:
        if isinstance(video_frames, InternVLSampledVideo):
            return len(video_frames.num_patches_list)
        if isinstance(video_frames, torch.Tensor):
            return int(video_frames.shape[0])
        return 0

    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 2.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> torch.Tensor:
        num_frames = max(1, int(duration_seconds * fps))
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)
        font_thickness = 2
        max_chars_per_line = 40
        max_lines = 8
        wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)[:max_lines]
        line_height = 30
        total_text_height = len(wrapped_lines) * line_height
        start_y = (frame_height - total_text_height) // 2 + line_height
        for i, line in enumerate(wrapped_lines):
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            x = (frame_width - text_width) // 2
            y = start_y + i * line_height
            cv2.putText(frame, line, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        frames = np.stack([frame for _ in range(num_frames)], axis=0)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return tensor.to(dtype=torch.uint8)

    def slice_frames(
        self,
        video_frames: InternVLSampledVideo,
        start_idx: int,
        end_idx: int,
    ) -> InternVLSampledVideo:
        if not isinstance(video_frames, InternVLSampledVideo):
            raise TypeError("InternVL sampler expects InternVLSampledVideo containers")
        return self._slice_clip(video_frames, start_idx, end_idx)

    def _slice_clip(
        self,
        clip: InternVLSampledVideo,
        start_frame: int,
        end_frame: int,
    ) -> InternVLSampledVideo:
        prefix = self._frame_offsets(clip.num_patches_list)
        start_idx = prefix[start_frame]
        end_idx = prefix[end_frame]
        pixel_values = clip.pixel_values[start_idx:end_idx]
        num_patches = clip.num_patches_list[start_frame:end_frame]
        return InternVLSampledVideo(pixel_values=pixel_values, num_patches_list=num_patches, fps=clip.fps)

    @staticmethod
    def _frame_offsets(counts: Sequence[int]) -> List[int]:
        offsets = [0]
        total = 0
        for count in counts:
            total += count
            offsets.append(total)
        return offsets

    def _segments_from_request(
        self,
        video_path: str,
        *,
        requested_fps: int,
        max_frames: Optional[int],
    ) -> int:
        target_fps = requested_fps if requested_fps and requested_fps > 0 else None
        if target_fps is not None:
            duration = self._infer_duration(video_path)
            if duration is not None:
                segments = max(1, int(round(duration * target_fps)))
                if max_frames is not None and max_frames > 0:
                    segments = min(segments, max_frames)
                return segments
        if max_frames is not None and max_frames > 0:
            return max(1, int(max_frames))
        return self.DEFAULT_SEGMENTS

    @staticmethod
    def _infer_duration(video_path: str) -> Optional[float]:
        if VideoReader is None or cpu is None:
            return None
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps_value = float(vr.get_avg_fps())

            # Duration is total_frames / fps. (We previously also tried unique
            # frame detection here; it was wrong on padded videos.)
            if total_frames > 0 and fps_value > 0:
                return total_frames / fps_value
        except Exception:
            return None
        return None

    def _resolve_video_path(self, video_path: str) -> str:
        if not video_path:
            raise ValueError("video_path must be provided for frame sampling")
        candidate = Path(video_path)
        if candidate.exists():
            return str(candidate)
        parent_dir = candidate.parent
        if not parent_dir.exists():
            raise FileNotFoundError(f"Video directory does not exist: {video_path}")
        suffix = candidate.suffix or ".mp4"
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        pattern = f"{candidate.stem}*{suffix}"
        matches = [path for path in parent_dir.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"Video file not found: {video_path}")
        chosen = max(matches, key=lambda path: (path.stat().st_size, path.stat().st_mtime))
        return str(chosen)


# ==== Helper functions copied from the official InternVL example ====


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, input_size=448, max_num=1, num_segments=32):
    """
    Load video and preprocess for InternVL.

    We deliberately read ALL frames with decord and then downsample with
    np.linspace, rather than computing a strided frame index up front. The
    strided approach (previously the get_index() helper, since deleted) gave
    inconsistent counts on padded videos and on clips whose duration didn't
    cleanly divide by num_segments. Reading-all-then-downsampling costs more
    memory but is robust to those edge cases and matches what the model side
    expects.

    Returns:
        pixel_values: torch.Tensor, shape (total_patches, 3, 448, 448), dtype=float32, normalized
        num_patches_list: List[int], number of patches per frame
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n = len(vr)

    # Read ALL frames first
    frame_idx = list(range(n))
    all_frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3), uint8

    # Downsample AFTER reading if num_segments < n
    if num_segments < n:
        indices = np.linspace(0, n - 1, num_segments, dtype=int)
        all_frames = all_frames[indices]

    # Process each frame with InternVL preprocessing
    pixel_values_list, num_patches_list = [], []
    transform = _build_transform(input_size=input_size)

    for frame in all_frames:
        # Convert numpy to PIL (still uint8, 0-255)
        img = Image.fromarray(frame).convert('RGB')

        # Dynamic preprocess: split into tiles based on aspect ratio
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)

        # Transform each tile: ToTensor (0-1) + ImageNet normalize
        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)  # Shape: (num_tiles, 3, 448, 448)

        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    # Concatenate all patches from all frames
    pixel_values = torch.cat(pixel_values_list)  # Shape: (total_patches, 3, 448, 448)
    return pixel_values, num_patches_list
