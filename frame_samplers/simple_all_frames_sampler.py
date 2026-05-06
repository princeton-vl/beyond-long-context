"""
Simple frame sampler for 1 FPS videos - reads ALL frames without sampling.

This approach is optimal when:
1. Videos are already at target FPS (1 FPS)
2. Goal is to never miss a single frame
3. No need for complex FPS calculations or frame budgeting

Usage:
    sampler = SimpleAllFramesSampler()
    frames = sampler.sample_frames(video_path)
    # Returns ALL frames in the video as numpy array or PIL Images
"""

from typing import Any, List, Optional, Union
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


class SimpleAllFramesSampler(FrameSamplerInterface):
    """
    Simple sampler that reads ALL frames from video using decord.

    No FPS sampling, no frame budget calculations - just reads every frame.
    Perfect for 1 FPS videos where you cannot afford to lose any frames.
    """

    def __init__(self, output_format: str = "numpy"):
        """
        Args:
            output_format: "numpy" (array) or "pil" (list of PIL Images)
        """
        if not DECORD_AVAILABLE:
            raise RuntimeError(
                "decord is required for SimpleAllFramesSampler. "
                "Install with: pip install decord"
            )
        self.output_format = output_format

    def sample_frames(
        self,
        video_path: str,
        fps: int = 1,  # Ignored - reads all frames
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, List[Image.Image]]:
        """
        Read ALL frames from video in presentation order.

        Args:
            video_path: Path to video file
            fps: IGNORED - this sampler reads all frames regardless
            min_pixels: Optional minimum pixel count for resizing
            max_pixels: Optional maximum pixel count for resizing
            max_frames: Optional max frame limit - if set, downsamples AFTER reading all
            **kwargs: Additional args (ignored)

        Returns:
            - If output_format="numpy": np.ndarray of shape (N, H, W, 3)
            - If output_format="pil": List[PIL.Image] of length N
        """
        # Open video with decord
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)

        print(f"[SimpleAllFramesSampler] {video_path} -> {n} frames (reading ALL)")

        # Read ALL frames
        frame_idx = list(range(n))
        frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3)

        # Apply max_frames limit AFTER reading (downsample if needed)
        if max_frames is not None and n > max_frames:
            print(f"[SimpleAllFramesSampler] Downsampling from {n} to {max_frames} frames")
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
            frames = frames[indices]

        # Apply pixel constraints if specified
        if min_pixels is not None or max_pixels is not None:
            frames = self._apply_pixel_constraints(frames, min_pixels, max_pixels)

        # Convert to requested format
        if self.output_format == "pil":
            return [Image.fromarray(frame.astype("uint8")) for frame in frames]
        else:
            return frames

    def _apply_pixel_constraints(
        self,
        frames: np.ndarray,
        min_pixels: Optional[int],
        max_pixels: Optional[int]
    ) -> np.ndarray:
        """Resize frames to satisfy pixel constraints."""
        if frames.shape[0] == 0:
            return frames

        h, w = frames.shape[1], frames.shape[2]
        current_pixels = h * w

        target_pixels = current_pixels
        if max_pixels is not None and current_pixels > max_pixels:
            target_pixels = max_pixels
        elif min_pixels is not None and current_pixels < min_pixels:
            target_pixels = min_pixels
        else:
            return frames

        # Calculate new dimensions maintaining aspect ratio
        scale = (target_pixels / current_pixels) ** 0.5
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # Resize all frames
        resized_frames = []
        for frame in frames:
            img = Image.fromarray(frame.astype("uint8"))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_frames.append(np.array(img))

        return np.stack(resized_frames)

    def get_frame_count(self, video_frames: Union[np.ndarray, List]) -> int:
        """Get frame count."""
        if isinstance(video_frames, np.ndarray):
            return video_frames.shape[0]
        elif isinstance(video_frames, list):
            return len(video_frames)
        return 0

    def slice_frames(
        self,
        video_frames: Union[np.ndarray, List],
        start_idx: int,
        end_idx: int
    ) -> Union[np.ndarray, List]:
        """Slice frames."""
        if isinstance(video_frames, np.ndarray):
            return video_frames[start_idx:end_idx]
        elif isinstance(video_frames, list):
            return video_frames[start_idx:end_idx]
        raise TypeError(f"Unsupported frame type: {type(video_frames)}")

    def make_video_chunks(
        self,
        video_frames: Union[np.ndarray, List],
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0
    ) -> List[tuple]:
        """Chunk video frames by time."""
        frame_count = self.get_frame_count(video_frames)
        if frame_count == 0:
            return []

        total_duration = frame_count / fps if fps > 0 else frame_count

        # No chunking
        if not chunk_size or chunk_size <= 0:
            return [(video_frames, segment_start_time, segment_start_time + total_duration)]

        # Chunk by frame count
        frames_per_chunk = max(1, int(chunk_size * fps))
        chunks = []

        for i in range(0, frame_count, frames_per_chunk):
            end_i = min(frame_count, i + frames_per_chunk)
            chunk_frames = self.slice_frames(video_frames, i, end_i)
            chunk_start = segment_start_time + (i / fps)
            chunk_end = segment_start_time + (end_i / fps)
            chunks.append((chunk_frames, chunk_start, chunk_end))

        return chunks

    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 2.0,
        frame_width: int = 640,
        frame_height: int = 480
    ) -> Union[np.ndarray, List[Image.Image]]:
        """Create text frames (simple black frame with white text)."""
        import cv2
        import textwrap

        num_frames = max(1, int(duration_seconds * fps))

        # Create black frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Render text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)
        font_thickness = 2

        wrapped_lines = textwrap.wrap(text, width=40)[:8]
        line_height = 30
        start_y = (frame_height - len(wrapped_lines) * line_height) // 2 + line_height

        for i, line in enumerate(wrapped_lines):
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            x = (frame_width - text_width) // 2
            y = start_y + i * line_height
            cv2.putText(frame, line, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Duplicate frame
        frames = np.stack([frame for _ in range(num_frames)])

        if self.output_format == "pil":
            return [Image.fromarray(f) for f in frames]
        return frames
