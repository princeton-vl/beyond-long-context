"""Frame sampling helpers optimized for readability and minimal I/O."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import torch


def _ensure_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required to sample frames but was not found in PATH")


def _require_media_tools() -> None:
    _ensure_binary("ffmpeg")
    _ensure_binary("ffprobe")


def _probe_dimensions(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width_str, height_str = result.stdout.strip().split(",")
    return int(height_str), int(width_str)


def _build_ffmpeg_cmd(path: Path, fps: float, start: float | None, end: float | None) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{float(start):.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        if start is not None:
            duration = max(0.01, float(end) - float(start))
            cmd += ["-t", f"{duration:.3f}"]
        else:
            cmd += ["-to", f"{float(end):.3f}"]
    cmd += [
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "image2pipe",
        "-vcodec",
        "rawvideo",
        "pipe:1",
    ]
    return cmd


def _read_frames(pipe, frame_size: int, height: int, width: int) -> List[torch.Tensor]:
    frames: list[torch.Tensor] = []
    while True:
        chunk = pipe.read(frame_size)
        if len(chunk) < frame_size:
            break
        tensor = torch.frombuffer(memoryview(chunk), dtype=torch.uint8)
        frames.append(tensor.reshape(height, width, 3).clone())
    return frames


def sample_video_frames(
    path: Path,
    *,
    fps: float,
    start: float | None = None,
    end: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return uint8 frames sampled at `fps` plus synthetic timestamps."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    _require_media_tools()
    height, width = _probe_dimensions(path)
    frame_size = height * width * 3
    cmd = _build_ffmpeg_cmd(path, fps, start, end)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Unable to capture ffmpeg output stream")
    frames = _read_frames(process.stdout, frame_size, height, width)
    process.stdout.close()
    process.wait()

    if not frames:
        return torch.zeros(0, 1, 1, 3, dtype=torch.uint8), torch.zeros(0)

    stacked = torch.stack(frames, dim=0).contiguous()
    timestamps = torch.arange(stacked.size(0), dtype=torch.float32) / float(fps)
    return stacked, timestamps
