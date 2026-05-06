"""M3-Agent frame sampling that matches original implementation exactly."""

import base64
import cv2
import numpy as np
import textwrap
from typing import Optional, List
from moviepy import VideoFileClip
from .base_sampler import FrameSamplerInterface


class M3AgentFrameSampler(FrameSamplerInterface):
    """Frame sampler that matches original M3-Agent video processing exactly."""
    
    def sample_frames(self, video_path: str, fps: int = 5, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None, **kwargs) -> List[str]:
        """
        Sample frames using exact original M3-Agent method.
        
        Args:
            video_path: Path to video file
            fps: Sample rate (frames per second) - original default is 5
            min_pixels: Unused (for compatibility)
            max_pixels: Unused (for compatibility)
            
        Returns:
            List of base64-encoded frame strings (not tensor!)
        """
        print(f"M3-Agent frame extraction: {video_path} at {fps} fps")
        
        # Use MoviePy exactly like original
        video = VideoFileClip(video_path)
        frames = []
        max_frames = kwargs.get("max_frames")
        frame_interval = 1.0 / fps
        
        print(f"Video duration: {video.duration}s, extracting every {frame_interval}s")
        
        sample_times = self._plan_sample_times(
            duration=video.duration,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )

        # Extract frames at specified intervals (matching original extract_frames)
        for t in sample_times:
            try:
                frame = video.get_frame(t)  # RGB format from MoviePy

                # Convert RGB to BGR exactly like original
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Encode to JPEG exactly like original
                _, buffer = cv2.imencode(".jpg", frame_bgr)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                frames.append(frame_b64)

            except Exception as e:
                print(f"Warning: Failed to extract frame at {t}s: {e}")
                continue
        
        video.close()
        print(f"Extracted {len(frames)} frames as base64 strings")
        return frames

    @staticmethod
    def _plan_sample_times(
        *,
        duration: float,
        frame_interval: float,
        max_frames: Optional[int],
    ) -> List[float]:
        """Return timestamps that cover the clip as evenly as possible."""

        effective_interval = frame_interval if frame_interval > 1e-9 else 1.0

        # Calculate natural frame count based on video duration and fps
        natural_frame_count = int(duration / effective_interval)

        if max_frames is None or max_frames <= 0:
            return [
                float(min(t, max(duration - 1e-6, 0.0)))
                for t in np.arange(0.0, max(duration, 0.0), effective_interval)
            ]

        # Don't load more frames than the video naturally has
        # e.g., 20 second video at 1 fps = 20 frames max, not 5000
        target = max(1, int(max_frames))
        if target > natural_frame_count:
            target = natural_frame_count

        safe_duration = max(duration, 1e-6)
        spread = np.linspace(0.0, safe_duration, target, endpoint=False)
        upper = max(duration - 1e-6, 0.0)
        return [float(min(t, upper)) for t in spread]
    
    def make_video_chunks(self, video_frames: List[str], fps: float, chunk_size: float, segment_start_time: float):
        """
        Build chunks from base64 frame list - for M3-Agent, video_frames is already base64 strings.
        
        Args:
            video_frames: List of base64 strings (not tensor!)
            fps: Original sampling fps
            chunk_size: Seconds per chunk
            segment_start_time: Start time offset
            
        Returns:
            List of (frames_chunk, time_start, time_end) tuples
        """
        total_frames = len(video_frames)
        total_duration = total_frames / fps
        
        print(f"Making chunks: {total_frames} frames, {total_duration}s duration, {chunk_size}s chunks")
        
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

        print(f"Created {len(chunks)} chunks")
        return chunks

    def get_frame_count(self, video_frames) -> int:
        """Get frame count for M3Agent format: list of base64 strings."""
        if isinstance(video_frames, list):
            return len(video_frames)
        return 0

    def slice_frames(
        self,
        video_frames: List[str],
        start_idx: int,
        end_idx: int,
    ) -> List[str]:
        if not isinstance(video_frames, list):
            raise TypeError("M3-Agent sampler expects frame lists for slicing")

        frames: List[str] = []
        i = 0
        n = len(video_frames)
        while i < n:
            token = video_frames[i]
            if token == "<unit>":
                if i + 1 < n:
                    frames.append(video_frames[i + 1])
                i += 2
            else:
                i += 1

        sliced = frames[start_idx:end_idx]
        result: List[str] = []
        for frame in sliced:
            result.extend(["<unit>", frame])
        return result

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                          frame_width: int = 640, frame_height: int = 480) -> List[str]:
        """
        Create text frames in base64 string format for M3Agent.

        Args:
            text: Text to render
            fps: Frames per second for timing calculations
            duration_seconds: Duration of text video in seconds
            frame_width: Width of generated frames
            frame_height: Height of generated frames

        Returns:
            List of base64-encoded frame strings
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

        # Convert to base64 strings like M3Agent expects
        base64_frames = []
        for _ in range(num_frames):
            # Convert RGB to BGR for consistency with M3Agent's expected format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Encode to JPEG and then to base64
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(frame_b64)

        return base64_frames
