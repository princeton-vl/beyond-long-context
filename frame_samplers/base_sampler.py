"""Base interface for frame sampling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class FrameSamplerInterface(ABC):
    """Base interface for video frame sampling strategies."""

    @abstractmethod
    def sample_frames(self, video_path: str, fps: int = 2, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None, **kwargs) -> Any:
        """
        Sample frames from video according to model requirements.

        Args:
            video_path: Path to video file
            fps: Target frames per second to sample
            min_pixels: Minimum pixel count for downsampling
            max_pixels: Maximum pixel count for downsampling
            **kwargs: Additional model-specific parameters (e.g., max_frames for VideoLLM)

        Returns:
            Sampled frames in model-appropriate format
        """
        pass

    @abstractmethod
    def get_frame_count(self, video_frames: Any) -> int:
        """
        Get the number of frames in the video data.

        Args:
            video_frames: Video frames in model-specific format

        Returns:
            Number of frames
        """
        pass

    @abstractmethod
    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 2.0,
                          frame_width: int = 448, frame_height: int = 448) -> Any:
        """
        Create text frames in the model's expected format.

        Args:
            text: Text to render
            fps: Frames per second for timing calculations
            duration_seconds: Duration of text video in seconds
            frame_width: Width of generated frames
            frame_height: Height of generated frames

        Returns:
            Text frames in model-appropriate format
        """
        pass

    @abstractmethod
    def slice_frames(self, video_frames: Any, start_idx: int, end_idx: int) -> Any:
        """Return a subset of sampled frames between the provided indices."""

        pass

    def slice_frames(self, video_frames: Any, start_idx: int, end_idx: int) -> Any:
        """Return a subset of frames between start_idx and end_idx."""

        if hasattr(video_frames, "__getitem__"):
            return video_frames[start_idx:end_idx]
        raise TypeError(
            "Frame representation does not support slicing; override slice_frames in a derived sampler."
        )
