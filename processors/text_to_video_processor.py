"""
Text-to-video processing utilities for converting option labels to video frames.
Handles rendering text onto black frames for video stream integration.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import textwrap
import base64
from PIL import Image


class TextToVideoProcessor:
    """Handles conversion of text labels to video frames using frame samplers."""

    def __init__(self, frame_width: int = 448, frame_height: int = 448, fps: float = 1.0):
        """
        Initialize text-to-video processor.

        Args:
            frame_width: Width of generated frames
            frame_height: Height of generated frames
            fps: Frames per second for timing calculations
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def create_option_label_frames_for_model(self, option_index: int, frame_sampler, duration_seconds: float = 2.0) -> Union[np.ndarray, List]:
        """
        Create video frames for option label using the frame sampler's create_text_frames method.

        Args:
            option_index: Index of the option (0, 1, 2, etc.)
            frame_sampler: Frame sampler instance with create_text_frames method
            duration_seconds: Duration of label video in seconds

        Returns:
            Frames in the format expected by the model (determined by frame sampler)
        """
        option_text = f"Option {option_index}:"
        return frame_sampler.create_text_frames(
            text=option_text,
            fps=self.fps,
            duration_seconds=duration_seconds,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )

    def get_frame_count_for_model(self, frames, frame_sampler) -> int:
        """Get frame count using the frame sampler's method."""
        return frame_sampler.get_frame_count(frames)

    def calculate_duration(self, frame_count: int) -> float:
        """Calculate duration in seconds for a given frame count."""
        return frame_count / self.fps if self.fps > 0 else 0.0


def create_option_text_video(option_index: int, processor: TextToVideoProcessor,
                           frame_sampler, duration_seconds: float = 2.0):
    """
    Convenience function to create option label video frames.

    Args:
        option_index: Index of the option
        processor: TextToVideoProcessor instance
        frame_sampler: Frame sampler instance
        duration_seconds: Duration of the text video

    Returns:
        Video frames in model-appropriate format
    """
    return processor.create_option_label_frames_for_model(option_index, frame_sampler, duration_seconds)