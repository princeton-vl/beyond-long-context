"""
Claude API adapter for the streaming-mem evaluation harness.

Implements VideoLanguageModelInterface using the Anthropic SDK. Video frames
arrive as torch tensors already sampled at the harness's target fps (default 1
fps via QwenFrameSampler); each frame is converted to a base64-encoded image
block and sent as part of the user message. Text-mode (sequence) runs need no
frame handling at all.

Notes
-----
- Default model is ``claude-opus-4-7`` (per ``claude-api`` skill cache). The
  older ``claude-opus-4-6`` alias is also accepted.
- On Opus 4.7, sampling parameters (``temperature``, ``top_p``, ``top_k``) are
  removed and will 400 — we never send them.
- Extended thinking is disabled by default (`thinking={"type": "disabled"}`)
  to match a single-pass eval. Pass ``thinking_mode="adaptive"`` via kwargs to
  enable adaptive thinking.
- FLOPs are not reported for API models (cost proxy only); the per-request
  ``usage`` dict is surfaced via ``get_last_response_token_stats``.
- Prompt caching: the user message is annotated with a ``cache_control``
  breakpoint so repeated questions over the same video can hit the cache.
"""

from __future__ import annotations

import base64
import copy
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.base_interface import VideoLanguageModelInterface


logger = logging.getLogger(__name__)


# Anthropic's documented per-request image cap is ~100 at time of writing.
# Longer video buckets (L=512/1024) will be uniformly subsampled down to this
# many frames. L=8/16/64/128/256 all fit under this cap without truncation.
DEFAULT_MAX_IMAGES_PER_REQUEST = 256

# Default model choice — skill doc says Opus 4.7 is the current recommended
# model; Opus 4.6 is still active and can be opted into via model_id kwarg.
DEFAULT_MODEL_ID = "claude-opus-4-7"

# Official friendly names (per CLAUDE.md) for display in logs / tags.
_MODEL_FRIENDLY_NAMES = {
    "claude-opus-4-7": "Claude-Opus-4-7",
    "claude-opus-4-6": "Claude-Opus-4-6",
    "claude-sonnet-4-6": "Claude-Sonnet-4-6",
    "claude-haiku-4-5": "Claude-Haiku-4-5",
}


class ClaudeAPIModel(VideoLanguageModelInterface):
    """Video-language model adapter backed by the Anthropic Messages API."""

    # Models that require adaptive thinking (no ``type: disabled``, no sampling
    # params). Opus 4.7 is the only such model in the current catalog; Opus 4.6
    # and earlier accept ``type: disabled``.
    _ADAPTIVE_ONLY_MODELS = frozenset({"claude-opus-4-7"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(
        self,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the Anthropic client and per-run state.

        Recognized kwargs:
            max_images_per_request (int): override the default image cap.
            thinking_mode (str): "disabled" (default) or "adaptive".
            image_format (str): "png" (default) or "jpeg".
            image_max_edge (Optional[int]): if set, resize frames so the long
                edge is at most this many pixels. Reduces token cost.
        """
        del max_gpu_mem  # no GPU

        try:
            import anthropic  # noqa: WPS433 — deferred import, API-only dep.
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for ClaudeAPIModel. "
                "Install via `uv pip install anthropic` in the full-stack env."
            ) from exc

        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before launching the "
                "eval (it will not be read from any file)."
            )

        # SDK retries internally for 429/5xx. Give each call a generous
        # timeout; the eval harness has its own per-question timeout handling.
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=kwargs.get("request_timeout", 120.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        self._anthropic = anthropic  # keep module handle for exception types

        self.max_images_per_request = int(
            kwargs.get("max_images_per_request", DEFAULT_MAX_IMAGES_PER_REQUEST)
        )
        self.thinking_mode = str(kwargs.get("thinking_mode", "disabled")).lower()
        if self.thinking_mode not in {"disabled", "adaptive"}:
            raise ValueError(
                f"thinking_mode must be 'disabled' or 'adaptive', got {self.thinking_mode!r}"
            )
        self.image_format = str(kwargs.get("image_format", "png")).lower()
        if self.image_format not in {"png", "jpeg"}:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        self._image_media_type = (
            "image/png" if self.image_format == "png" else "image/jpeg"
        )
        self.image_max_edge: Optional[int] = kwargs.get("image_max_edge")

        # Documented but informational — the caller already samples at this fps.
        self.fps: float = float(kwargs.get("fps", 1.0))

        # Context state. Mirrors MiMoVLVideo's schema for cross-compatibility.
        self.latest_time: float = 0.0
        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[tuple] = []  # (text, timestamp)

        # Last-response usage for logging / downstream inspection.
        self._last_usage: Optional[Dict[str, Any]] = None
        self._video_was_truncated: Optional[bool] = None

    # ------------------------------------------------------------------
    # Context mutation (mirrors MiMoVLVideo)
    # ------------------------------------------------------------------

    def add_video(
        self,
        video_frames: Union[np.ndarray, torch.Tensor, List[Any]],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError(
                "time_start must be after the last added video segment"
            )
        self.latest_time = time_end

        # Normalize to torch tensor on CPU. Frames arrive from QwenFrameSampler
        # as (N, 3, H, W) uint8 tensors. Keep on CPU — we only base64 them.
        if isinstance(video_frames, np.ndarray):
            frames_tensor = torch.from_numpy(video_frames)
        elif isinstance(video_frames, torch.Tensor):
            frames_tensor = video_frames.detach().cpu()
        else:
            # Already in some pre-processed list form (e.g. from sequence mode);
            # stash opaquely. Not expected in video mode.
            frames_tensor = video_frames  # type: ignore[assignment]

        if video_id is None:
            video_id = 0
        self.video_segments.setdefault(video_id, []).append(
            {
                "frames": frames_tensor,
                "time_start": float(time_start),
                "time_end": float(time_end),
                "duration": float(time_end) - float(time_start),
                "num_frames": int(getattr(frames_tensor, "shape", [0])[0])
                if hasattr(frames_tensor, "shape")
                else 0,
            }
        )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        self.text_entries.append((text, self.latest_time))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        start = time.perf_counter()
        content_blocks = self._build_content_blocks(
            question=question,
            max_frames_in_video=max_frames_in_video,
            sample_method=sample_method,
        )

        model_id = self._resolve_api_model_id()
        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max(1, int(max_tokens)),
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
        }

        # Thinking. Opus 4.7 only accepts ``adaptive`` (or omit → off by default).
        if model_id in self._ADAPTIVE_ONLY_MODELS:
            if self.thinking_mode == "adaptive":
                request_kwargs["thinking"] = {"type": "adaptive"}
            # Otherwise: omit the field. On Opus 4.7, omit == off.
        else:
            # Opus 4.6 / Sonnet 4.6: explicit disable is accepted.
            if self.thinking_mode == "adaptive":
                request_kwargs["thinking"] = {"type": "adaptive"}
            else:
                request_kwargs["thinking"] = {"type": "disabled"}

        try:
            response = self._client.messages.create(**request_kwargs)
        except self._anthropic.APIError as exc:
            logger.exception("[ClaudeAPI] request failed: %s", exc)
            return ""

        # Extract text. API guarantees response.content is a list of blocks.
        output_text = ""
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                output_text = getattr(block, "text", "") or ""
                break

        # Stash usage for downstream logging.
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                self._last_usage = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cache_creation_input_tokens": getattr(
                        usage, "cache_creation_input_tokens", None
                    ),
                    "cache_read_input_tokens": getattr(
                        usage, "cache_read_input_tokens", None
                    ),
                    "stop_reason": getattr(response, "stop_reason", None),
                    "model": getattr(response, "model", None),
                }
            except Exception:  # pragma: no cover
                self._last_usage = None

        # Record metrics (flops=None; we can't measure them for API calls).
        if self.enable_metrics and self._metrics:
            latency = time.perf_counter() - start
            latest_context_time = max(
                (seg["time_end"] for segs in self.video_segments.values() for seg in segs),
                default=0.0,
            )
            question_timestamp = float(current_video_time or 0.0)
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time
            self._record_ask_question_metrics(
                latency=latency,
                flops=0.0,  # base interface expects a numeric; 0 == unknown
                peak_gpu_mem_increase=0.0,
                peak_gpu_mem_absolute=0.0,
                video_time=question_timestamp,
                state_memory_total=0.0,
            )

        return output_text

    def get_last_response_token_stats(self) -> Optional[Dict[str, Any]]:
        return copy.copy(self._last_usage)

    def was_video_truncated(self) -> Optional[bool]:
        return self._video_was_truncated

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "latest_time": self.latest_time,
            "text_entries": list(self.text_entries),
            # Shallow video_segments — tensors are referenced, not deep-copied.
            "video_segments": {
                vid: [dict(seg) for seg in segs]
                for vid, segs in self.video_segments.items()
            },
        }

    def clear_context(self) -> None:
        self.latest_time = 0.0
        self.video_segments.clear()
        self.text_entries.clear()
        self._last_usage = None
        self._video_was_truncated = None

    def save_state(self) -> Any:
        return self.get_state()

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("ClaudeAPIModel.load_state expects a dict state")
        self.latest_time = float(state.get("latest_time", 0.0))
        self.text_entries = list(state.get("text_entries", []))
        self.video_segments = {
            vid: [dict(seg) for seg in segs]
            for vid, segs in state.get("video_segments", {}).items()
        }
        self._last_usage = None
        self._video_was_truncated = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_api_model_id(self) -> str:
        """Return the model_id string to pass to the Anthropic SDK.

        We accept friendly names too; strip known suffixes / normalize.
        """
        mid = (self.model_id or DEFAULT_MODEL_ID).strip()
        if mid in _MODEL_FRIENDLY_NAMES:
            return mid
        lowered = mid.lower()
        if lowered in _MODEL_FRIENDLY_NAMES:
            return lowered
        # Accept 4.6-dot / 4-6-dash / "claude-opus-4.6" style inputs.
        for slug in _MODEL_FRIENDLY_NAMES:
            if lowered.replace(".", "-") == slug:
                return slug
        # Fallthrough: pass whatever we got; Anthropic will 404 if wrong.
        return mid

    def _build_content_blocks(
        self,
        question: str,
        max_frames_in_video: int,
        sample_method: str,
    ) -> List[Dict[str, Any]]:
        """Build the user-message ``content`` array for the API call.

        Structure:
            [image] * N  (interleaved by video_id, in timeline order)
            + [text] * M (text_entries in timeline order)
            + [text]     (the question, with cache_control breakpoint)

        We interleave videos and text by timeline order so Claude sees the
        same ordering other VLMs do.
        """
        # Build timeline (matches MiMoVLVideo behaviour).
        timeline: List[Dict[str, Any]] = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline.append(
                    {
                        "type": "video",
                        "time": segment["time_start"],
                        "video_id": video_id,
                        "segment": segment,
                    }
                )
        for text, timestamp in self.text_entries:
            timeline.append(
                {
                    "type": "text",
                    "time": timestamp,
                    "text": text,
                }
            )
        timeline.sort(key=lambda ev: (ev["time"], ev["type"] == "video"))

        # Aggregate frames per video_id in first-pass, then truncate/sample.
        video_data: Dict[int, Dict[str, Any]] = {}
        for event in timeline:
            if event["type"] != "video":
                continue
            vid = event["video_id"]
            seg = event["segment"]
            frames = seg["frames"]
            bucket = video_data.setdefault(
                vid, {"frames": None, "total_frames": 0, "duration": 0.0}
            )
            if bucket["frames"] is None:
                bucket["frames"] = frames
            else:
                bucket["frames"] = torch.cat([bucket["frames"], frames], dim=0)
            bucket["total_frames"] += int(seg.get("num_frames", frames.shape[0]))
            bucket["duration"] += float(seg["duration"])

        # Apply the user's max_frames_in_video cap first, then the API image
        # cap. The per-request image cap is a hard API constraint; the
        # max_frames_in_video flag matches the harness's other models.
        effective_cap = min(
            int(max_frames_in_video) if max_frames_in_video else self.max_images_per_request,
            self.max_images_per_request,
        )

        self._video_was_truncated = False
        for vid, data in video_data.items():
            frames = data["frames"]
            if frames is None:
                continue
            n = int(frames.shape[0])
            if n <= effective_cap:
                continue
            self._video_was_truncated = True
            if sample_method == "RANDOM":
                idx = np.random.choice(n, effective_cap, replace=False)
                idx.sort()
            else:
                idx = np.linspace(0, n - 1, effective_cap, dtype=int)
            data["frames"] = frames[idx]
            logger.warning(
                "[ClaudeAPI] video_id=%s truncated from %d -> %d frames (cap=%d)",
                vid,
                n,
                effective_cap,
                effective_cap,
            )

        # Emit content in timeline order; emit a video's frames at the first
        # timeline event that references it, then skip subsequent events for
        # the same video.
        processed_video_ids: set = set()
        content: List[Dict[str, Any]] = []
        for event in timeline:
            if event["type"] == "video":
                vid = event["video_id"]
                if vid in processed_video_ids:
                    continue
                processed_video_ids.add(vid)
                frames = video_data[vid]["frames"]
                if frames is None or frames.shape[0] == 0:
                    continue
                for image_block in self._frames_to_image_blocks(frames):
                    content.append(image_block)
            else:  # text
                content.append({"type": "text", "text": event["text"]})

        # Final block: the question. Attach a cache_control breakpoint here so
        # the image + context prefix can be cached across repeated questions
        # for the same video.
        content.append(
            {
                "type": "text",
                "text": question,
                "cache_control": {"type": "ephemeral"},
            }
        )
        return content

    def _frames_to_image_blocks(
        self, frames: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Convert a (N, 3, H, W) uint8 tensor into Anthropic image blocks."""
        if frames is None or frames.shape[0] == 0:
            return []

        # Expect (N, 3, H, W) uint8 (QwenFrameSampler output). Handle (N, H, W, 3)
        # just in case another sampler ever feeds us that layout.
        if frames.dim() != 4:
            raise ValueError(
                f"Expected 4D frame tensor; got shape {tuple(frames.shape)}"
            )
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
            # (N, C, H, W)
            frames_hwc = frames.permute(0, 2, 3, 1).contiguous()
        elif frames.shape[-1] == 3:
            frames_hwc = frames
        else:
            raise ValueError(
                f"Cannot infer channel axis for frame tensor of shape "
                f"{tuple(frames.shape)}; expected 3 channels."
            )

        if frames_hwc.dtype != torch.uint8:
            frames_hwc = frames_hwc.clamp(0, 255).to(torch.uint8)

        frames_np = frames_hwc.cpu().numpy()
        blocks: List[Dict[str, Any]] = []
        for frame in frames_np:
            img = Image.fromarray(frame, mode="RGB")
            if self.image_max_edge is not None:
                img = _resize_max_edge(img, int(self.image_max_edge))
            buf = io.BytesIO()
            if self.image_format == "jpeg":
                img.save(buf, format="JPEG", quality=85, optimize=True)
            else:
                img.save(buf, format="PNG", optimize=True)
            data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._image_media_type,
                        "data": data,
                    },
                }
            )
        return blocks


def _resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    """Resize so the longer edge is ``max_edge`` pixels, preserving aspect."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


__all__ = ["ClaudeAPIModel", "DEFAULT_MODEL_ID"]
