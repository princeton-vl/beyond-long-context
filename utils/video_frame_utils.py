"""
Utilities for handling videos with incorrect metadata and duplicated frames.

Some videos have metadata claiming many frames but only contain a few unique frames
with padding/duplication. This module provides utilities to detect and extract
the actual unique frames.
"""

import numpy as np
from typing import List, Tuple, Optional


def detect_unique_frame_indices(vr, max_check: Optional[int] = None, threshold: float = 0.01) -> List[int]:
    """
    Detect indices of unique frames in a VideoReader by comparing frame differences.

    This handles videos where metadata claims many frames but most are duplicates.
    For example, a video with 5 real frames padded to 125 frames via duplication.

    Args:
        vr: decord.VideoReader instance
        max_check: Maximum number of frames to check (None = check all)
        threshold: Threshold for considering frames different (0-1, fraction of pixels)

    Returns:
        List of frame indices that are unique/real frames
    """
    total_frames = len(vr)
    check_frames = min(total_frames, max_check) if max_check else total_frames

    if check_frames == 0:
        return []

    unique_indices = [0]  # First frame is always included
    prev_frame = vr[0].asnumpy()

    for i in range(1, check_frames):
        curr_frame = vr[i].asnumpy()

        # Calculate difference between frames
        diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
        diff_ratio = np.mean(diff) / 255.0  # Normalize to 0-1

        # If frames are sufficiently different, this is a unique frame
        if diff_ratio > threshold:
            unique_indices.append(i)
            prev_frame = curr_frame

    return unique_indices


def get_real_frame_count(video_path) -> Tuple[int, int]:
    """
    Get both metadata frame count and estimated real frame count.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (metadata_frames, estimated_real_frames)
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return (0, 0)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    metadata_frames = len(vr)

    # Sample frames to estimate real count
    # Check first 10, middle 10, and last 10 frames to find unique ones
    sample_indices = []
    if metadata_frames <= 30:
        sample_indices = list(range(metadata_frames))
    else:
        sample_indices = (
            list(range(10)) +
            list(range(metadata_frames // 2 - 5, metadata_frames // 2 + 5)) +
            list(range(metadata_frames - 10, metadata_frames))
        )

    unique_count = 1  # First frame
    if sample_indices:
        prev_frame = vr[sample_indices[0]].asnumpy()
        for idx in sample_indices[1:]:
            curr_frame = vr[idx].asnumpy()
            diff_ratio = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))) / 255.0
            if diff_ratio > 0.01:
                unique_count += 1
                prev_frame = curr_frame

    return (metadata_frames, unique_count)


def get_all_unique_frames(vr, threshold: float = 0.01) -> Tuple[np.ndarray, List[int]]:
    """
    Extract all unique frames from a VideoReader.

    Args:
        vr: decord.VideoReader instance
        threshold: Threshold for considering frames different

    Returns:
        Tuple of (frames_array, source_indices) where:
        - frames_array: numpy array of shape (num_unique_frames, H, W, C)
        - source_indices: list of indices in original video where each unique frame came from
    """
    unique_indices = detect_unique_frame_indices(vr, threshold=threshold)

    if not unique_indices:
        return (np.array([]), [])

    # Extract frames
    frames = []
    for idx in unique_indices:
        frames.append(vr[idx].asnumpy())

    frames_array = np.stack(frames, axis=0)
    return (frames_array, unique_indices)


def ensure_minimum_frames(
    unique_frames: np.ndarray,
    unique_indices: List[int],
    target_count: int,
    interpolate: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Ensure we have at least target_count frames by interpolating/repeating if needed.

    Args:
        unique_frames: Array of unique frames (N, H, W, C)
        unique_indices: Source indices of unique frames
        target_count: Desired number of frames
        interpolate: If True, interpolate between frames; if False, repeat last frame

    Returns:
        Tuple of (frames_array, frame_positions) where:
        - frames_array: numpy array with target_count frames
        - frame_positions: fractional positions in original video (for timestamps)
    """
    num_unique = len(unique_frames)

    if num_unique >= target_count:
        # Already have enough frames, uniformly sample
        indices = np.linspace(0, num_unique - 1, target_count).astype(int)
        return (unique_frames[indices], [unique_indices[i] for i in indices])

    if not interpolate or num_unique == 1:
        # Repeat frames to reach target
        repeat_factor = (target_count + num_unique - 1) // num_unique
        repeated = np.tile(unique_frames, (repeat_factor, 1, 1, 1))[:target_count]
        positions = [unique_indices[i % num_unique] for i in range(target_count)]
        return (repeated, positions)

    # Interpolate between unique frames
    result_frames = []
    result_positions = []

    for i in range(target_count):
        # Position in the unique frames sequence
        pos = i * (num_unique - 1) / (target_count - 1) if target_count > 1 else 0
        idx_low = int(pos)
        idx_high = min(idx_low + 1, num_unique - 1)
        alpha = pos - idx_low

        if alpha < 1e-6 or idx_low == idx_high:
            result_frames.append(unique_frames[idx_low])
            result_positions.append(unique_indices[idx_low])
        else:
            # Linear interpolation
            frame_low = unique_frames[idx_low].astype(np.float32)
            frame_high = unique_frames[idx_high].astype(np.float32)
            interpolated = ((1 - alpha) * frame_low + alpha * frame_high).astype(np.uint8)
            result_frames.append(interpolated)
            result_positions.append(unique_indices[idx_low] * (1 - alpha) + unique_indices[idx_high] * alpha)

    return (np.stack(result_frames, axis=0), result_positions)
