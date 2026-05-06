from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import os
import shutil
import subprocess
import threading
from collections import deque
import numpy as np

from .engine import VideoInstance, frame_generator

def render_video_to_tensors(instance: VideoInstance, batch_size: int = 32, channels_first: bool = True,
                            device: str = "cpu", dtype: "torch.dtype" = None) -> Iterator[Tuple["torch.Tensor", Dict[str, Any]]]:
    """Stream frames as torch tensors in batches.

    Yields (batch_tensor, meta), where:
      - batch_tensor: [B,3,H,W] uint8 (default) or [B,H,W,3] if channels_first=False
      - meta includes start_t, fps, seed, variant, job_id, etc.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("render_video_to_tensors requires torch; install torch or use render_video_to_mp4") from exc

    H, W = instance.height, instance.width
    buf = np.empty((batch_size, H, W, 3), dtype=np.uint8)
    times: List[float] = []
    b = 0
    meta0 = dict(job_id=instance.job.id, variant=instance.variant_idx, seed=instance.seed, fps=instance.fps, width=W, height=H)
    for t, frame in frame_generator(instance):
        buf[b] = frame
        times.append(t)
        b += 1
        if b == batch_size:
            batch = torch.from_numpy(buf.copy())  # copy to detach from reuse buffer
            if channels_first:
                batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = batch.to(device=device, dtype=dtype)
            yield batch, {**meta0, "times": times}
            b = 0
            times = []
    if b > 0:
        batch = torch.from_numpy(buf[:b].copy())
        if channels_first:
            batch = batch.permute(0, 3, 1, 2).contiguous()
        batch = batch.to(device=device, dtype=dtype)
        yield batch, {**meta0, "times": times}

def render_video_to_mp4(
    instance: VideoInstance,
    out_path: str,
    crf: int = 23,
    preset: str = "veryfast",
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    frame_observer: Optional[Callable[[int, float, Optional[List[dict]]], None]] = None,
) -> str:
    """Render and write an MP4 using ffmpeg by streaming raw RGB frames.

    Requires ffmpeg available on PATH.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg or use tensor output.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    W, H = instance.width, instance.height
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-nostats",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(instance.fps),
        "-i", "-",
        "-an",
        "-vcodec", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err_buf = deque(maxlen=20000)
    err_lock = threading.Lock()

    def _drain_err() -> None:
        if proc.stderr is None:
            return
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            with err_lock:
                err_buf.extend(chunk)

    err_thread = threading.Thread(target=_drain_err, daemon=True)
    err_thread.start()
    try:
        for fi, (t, frame) in enumerate(frame_generator(instance, frame_observer=frame_observer)):
            proc.stdin.write(frame.tobytes(order="C"))
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        err_thread.join(timeout=1.0)
        if proc.returncode != 0:
            with err_lock:
                err = bytes(err_buf).decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}\n{err[:2000]}")
    return out_path
