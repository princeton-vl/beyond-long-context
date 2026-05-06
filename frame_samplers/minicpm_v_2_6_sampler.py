"""MiniCPM-style frame sampling with chunking and downsampling."""

import math
from typing import List, Optional

import numpy as np
import cv2
import textwrap
from PIL import Image

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    VideoReader = None
    cpu = None

from .base_sampler import FrameSamplerInterface


class MiniCPMFrameSampler(FrameSamplerInterface):
    """Frame sampler for MiniCPM with chunking and proper downsampling."""
    
    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> List:
        """
        Sample ALL frames using decord for reliable frame extraction.

        Args:
            video_path: Path to video file
            fps: Target frames per second (ignored - reads all frames)
            min_pixels: Minimum pixel count for downsampling
            max_pixels: Maximum pixel count for downsampling

        Returns:
            Flattened list of ["<unit>", image] pairs
        """
        if not DECORD_AVAILABLE:
            raise RuntimeError(
                "decord is required for MiniCPM frame sampling. "
                "Install with: pip install decord"
            )

        # Read ALL frames using decord
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        print(f'[MiniCPM] video: {video_path} -> {n} frames (reading ALL)')

        frame_idx = list(range(n))
        all_frames = vr.get_batch(frame_idx).asnumpy()  # Shape: (N, H, W, 3), dtype=uint8

        # Apply max_frames limit AFTER reading if specified
        max_frames = kwargs.get("max_frames")
        if max_frames and n > max_frames:
            print(f'[MiniCPM] Downsampling from {n} to {max_frames} frames')
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
            all_frames = all_frames[indices]

        # Convert to MiniCPM format: ["<unit>", PIL.Image, "<unit>", PIL.Image, ...]
        contents = []
        for frame in all_frames:
            # frame is (H, W, 3), uint8 already
            image = Image.fromarray(frame)

            # Apply pixel constraints if specified
            if min_pixels is not None or max_pixels is not None:
                image = self._downsample_image(image, min_pixels, max_pixels)

            contents.extend(["<unit>", image])

        return contents

    def make_video_chunks(
        self,
        video_frames: List,
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0,
    ) -> List[tuple]:
        """
        Build a list of (frames_chunk, time_start, time_end) tuples for the given video.

        If chunk_size is None or <= 0, return a single chunk representing the whole video.
        Expects `video_frames` to be the flattened ["<unit>", image, ...] list from sample_frames.
        """
        # Extract images in order from the flattened ["<unit>", image, ...] list
        frames = []
        if isinstance(video_frames, list):
            i = 0
            n = len(video_frames)
            while i < n:
                if video_frames[i] == "<unit>":
                    if i + 1 < n:
                        frames.append(video_frames[i + 1])
                    i += 2
                else:
                    # Skip any unexpected tokens to be robust
                    i += 1

        total_frames = len(frames)
        if total_frames == 0:
            return []

        if fps is None or fps <= 0:
            fps = 1.0
        total_duration = total_frames / fps

        # If no chunking requested, return entire content as a single chunk
        if not chunk_size or chunk_size <= 0:
            time_start = segment_start_time
            time_end = segment_start_time + total_duration
            # Use the original flattened content so add_video() hits the "<unit>" branch
            return [(video_frames, time_start, time_end)]

        # Compute how many sampled frames per chunk
        frames_per_chunk = max(1, int(math.ceil(chunk_size * fps)))

        chunks = []
        for start_idx in range(0, total_frames, frames_per_chunk):
            end_idx = min(total_frames, start_idx + frames_per_chunk)
            chunk_images = frames[start_idx:end_idx]

            # Rebuild flattened content for this chunk so it starts with "<unit>"
            chunk_content = []
            for img in chunk_images:
                chunk_content.extend(["<unit>", img])

            time_start = segment_start_time + (start_idx / fps)
            time_end = segment_start_time + (end_idx / fps)
            chunks.append((chunk_content, time_start, time_end))

        return chunks
    
    def _downsample_image(self, image: Image.Image, min_pixels: Optional[int], max_pixels: Optional[int]) -> Image.Image:
        """
        Downsample image based on pixel constraints.
        
        Args:
            image: PIL Image to downsample
            min_pixels: Minimum pixel count
            max_pixels: Maximum pixel count
            
        Returns:
            Downsampled PIL Image
        """
        width, height = image.size
        current_pixels = width * height
        
        # Calculate target pixels
        target_pixels = current_pixels
        
        if max_pixels is not None and current_pixels > max_pixels:
            target_pixels = max_pixels
        elif min_pixels is not None and current_pixels < min_pixels:
            target_pixels = min_pixels
        else:
            return image  # No resizing needed
        
        # Calculate scale factor maintaining aspect ratio
        scale_factor = (target_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def get_frame_count(self, video_frames) -> int:
        """Get frame count for MiniCPM format: ["<unit>", image, "<unit>", image, ...]."""
        if not isinstance(video_frames, list):
            return 0

        # Count the number of actual frames (images)
        frame_count = 0
        i = 0
        while i < len(video_frames):
            if video_frames[i] == "<unit>" and i + 1 < len(video_frames):
                frame_count += 1
                i += 2
            else:
                i += 1
        return frame_count

    def slice_frames(
        self,
        video_frames,
        start_idx: int,
        end_idx: int,
    ):
        if not isinstance(video_frames, list):
            raise TypeError("MiniCPM sampler expects flattened frame lists for slicing")

        frames: List = []
        i = 0
        while i < len(video_frames):
            if video_frames[i] == "<unit>" and i + 1 < len(video_frames):
                frames.append(video_frames[i + 1])
                i += 2
            else:
                i += 1

        sliced = frames[start_idx:end_idx]
        flattened: List = []
        for frame in sliced:
            flattened.extend(["<unit>", frame])
        return flattened

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                          frame_width: int = 640, frame_height: int = 480) -> List:
        """
        Create text frames in MiniCPM format: ["<unit>", image, "<unit>", image, ...]

        Args:
            text: Text to render
            fps: Frames per second for timing calculations
            duration_seconds: Duration of text video in seconds
            frame_width: Width of generated frames
            frame_height: Height of generated frames

        Returns:
            Flattened list of ["<unit>", image] pairs
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

        # Convert to PIL Image and create MiniCPM format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb.astype(np.uint8))

        # Create flattened content format expected by MiniCPM
        content = []
        for _ in range(num_frames):
            content.extend(["<unit>", image])

        return content
