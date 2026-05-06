"""TimeChat-specific frame sampling that handles the unique requirements."""

import os
# CRITICAL: Set backend BEFORE importing qwen_vl_utils to avoid torchcodec memory issues
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

import numpy as np
import torch
import cv2
import textwrap
from typing import Optional
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_sampler import FrameSamplerInterface


class TimeChatFrameSampler(FrameSamplerInterface):
    """Frame sampler for TimeChat with special text frame handling."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Sample frames using Qwen's processor to return tensor format.

        Args:
            video_path: Path to video file
            fps: Target frames per second to sample
            min_pixels: Minimum pixel count for downsampling
            max_pixels: Maximum pixel count for downsampling

        Returns:
            Video tensor ready for model input
        """
        max_frames = kwargs.get("max_frames")

        computed_max_frames = None
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_native_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                native_fps = cap.get(cv2.CAP_PROP_FPS)
                if native_fps and native_fps > 0 and total_native_frames and total_native_frames > 0:
                    duration_sec = total_native_frames / native_fps
                    computed_max_frames = max(1, int(round(duration_sec * fps)))
            cap.release()
        except Exception:
            pass

        if computed_max_frames is None:
            final_max_frames = max_frames
        else:
            final_max_frames = min(max_frames, computed_max_frames) if max_frames is not None else computed_max_frames

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        **({"max_frames": int(final_max_frames)} if final_max_frames is not None else {}),
                    }
                ],
            }
        ]

        # Create processor with optional pixel constraints
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        _, video_inputs = process_vision_info(messages)

        if isinstance(video_inputs, list) and len(video_inputs) > 0:
            video_tensor = video_inputs[0]
        else:
            video_tensor = video_inputs

        return video_tensor

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

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                          frame_width: int = 560, frame_height: int = 420) -> np.ndarray:
        """
        Create text frames in the format expected by TimeChat (normalized, TCHW format).

        Args:
            text: Text to render
            fps: Frames per second for timing calculations
            duration_seconds: Duration of text video in seconds
            frame_width: Width of generated frames
            frame_height: Height of generated frames

        Returns:
            Stacked frames as numpy array with shape (num_frames, channels, height, width), normalized to [0, 1]
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

        # Convert from (T, H, W, C) to (T, C, H, W) to match TimeChat processor format
        frames = frames.transpose(0, 3, 1, 2)

        return frames

    def slice_frames(
        self,
        video_frames: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        if not hasattr(video_frames, "shape"):
            raise TypeError("TimeChat sampler expects numpy array outputs")
        return video_frames[start_idx:end_idx]
