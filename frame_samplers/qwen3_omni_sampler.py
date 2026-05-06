"""Qwen-style frame sampling for tensor-based models."""

import os
# CRITICAL: Set backend BEFORE importing qwen_omni_utils to avoid torchcodec memory issues
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

import numpy as np
import torch
import textwrap
from typing import Optional, Union, Iterable
from .base_sampler import FrameSamplerInterface
import cv2
from qwen_omni_utils import process_mm_info


class Qwen3oFrameSampler(FrameSamplerInterface):
    """Frame sampler for Qwen and other tensor-based models."""
    
    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample frames using Qwen's processor to return tensor format.

        Args:
            video_path: Path to video file
            fps: Target frames per second to sample
            min_pixels: Minimum pixel count for downsampling
            max_pixels: Maximum pixel count for downsampling
            max_frames: Optional hard cap on frames (will clamp computed value)
        Returns:
            Video tensor ready for model input
        """
        # Compute desired max_frames from video duration
        computed_max_frames = None
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_native_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps and native_fps > 0 and total_native_frames and total_native_frames > 0:
                duration_sec = total_native_frames / native_fps
                computed_max_frames = max(1, int(round(duration_sec * fps)))
        cap.release()
        # Determine final max frames without heuristic fallback
        if computed_max_frames is None:
            final_max_frames = max_frames
        else:
            final_max_frames = min(max_frames, computed_max_frames) if max_frames is not None else computed_max_frames

        content_entry = {
            "type": "video",
            "video": video_path,
        }
        if final_max_frames is not None:
            content_entry["max_frames"] = final_max_frames

        messages = [
            {
                "role": "user",
                "content": [content_entry],
            }
        ]
        _, _, video_inputs = process_mm_info(messages, use_audio_in_video=False)
        if isinstance(video_inputs, list) and len(video_inputs) > 0:
            return video_inputs[0]
        return video_inputs
    def _as_tensor(self, video_frames: Union[torch.Tensor, np.ndarray, Iterable]) -> torch.Tensor:
        """Normalize incoming frame containers into a single tensor."""
        if isinstance(video_frames, torch.Tensor):
            tensor = video_frames
        elif isinstance(video_frames, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(video_frames))
        elif isinstance(video_frames, (list, tuple)):
            if not video_frames:
                raise ValueError("No frame data provided for chunking.")
            parts = [self._as_tensor(part) for part in video_frames]
            tensor = torch.cat(parts, dim=0)
        else:
            raise TypeError(f"Unsupported video frame container: {type(video_frames)}")

        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor of shape (frames, channels, height, width); got {tensor.shape}")

        return tensor.contiguous()

    def make_video_chunks(self, video_frames, fps: float, chunk_size: float, segment_start_time: float):
        """
        Build a list of (frames_chunk, time_start, time_end) tuples for the given video.

        If chunk_size is None or <= 0, return a single chunk representing the whole video.
        """
        frames_tensor = self._as_tensor(video_frames)
        total_frames = frames_tensor.shape[0]
        total_duration = total_frames / fps

        # No chunking: single chunk covering the entire video
        if not chunk_size or chunk_size <= 0:
            return [(frames_tensor, segment_start_time, segment_start_time + total_duration)]

        # Chunking
        frames_per_chunk = max(1, int(chunk_size * fps))
        chunks = []
        for i in range(0, total_frames, frames_per_chunk):
            chunk_of_frames = frames_tensor[i : i + frames_per_chunk].contiguous()
            chunk_start_on_timeline = segment_start_time + (i / fps)
            chunk_end_on_timeline = segment_start_time + ((i + len(chunk_of_frames)) / fps)
            chunks.append((chunk_of_frames, chunk_start_on_timeline, chunk_end_on_timeline))

        return chunks

    def get_frame_count(self, video_frames) -> int:
        """Get frame count for qwen3omni processed format."""
        if video_frames is None:
            return 0

        def _count_frames(container) -> int:
            if isinstance(container, torch.Tensor) or isinstance(container, np.ndarray):
                if container.ndim == 0:
                    return 0
                return int(container.shape[0])
            if isinstance(container, (list, tuple)):
                return sum(_count_frames(item) for item in container)
            raise TypeError(f"Unsupported frame container: {type(container)}")

        return _count_frames(video_frames)

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                          frame_width: int = 640, frame_height: int = 480) -> np.ndarray:
        """
        Create text frames in numpy array format for Qwen3Omni.

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
        tensor = self._as_tensor(video_frames)
        return tensor[start_idx:end_idx].contiguous()
