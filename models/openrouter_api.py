"""
OpenRouter API adapter for the streaming-mem evaluation harness.

Implements ``VideoLanguageModelInterface`` using the OpenAI-compatible
OpenRouter endpoint (``https://openrouter.ai/api/v1``). Lets us point the
harness at any OpenRouter-hosted model (e.g. Gemini 3.1, GPT 5.5, Grok-5,
Llama 4, etc.) with a single slug.

Video frames arrive as torch tensors already sampled at the harness's target
fps (default 1 fps via ``QwenFrameSampler``). Each frame is converted to a
base64-encoded ``data:image/...`` URL and wrapped in a standard OpenAI
``image_url`` content block. Text-mode (sequence) runs skip the frame path
entirely.

Notes
-----
- Auth: reads ``OPENROUTER_API_KEY`` from the environment at setup time.
  Setup raises ``RuntimeError`` with a clear message if unset.
- Retries: the adapter handles 429/503 with exponential backoff (up to 3
  retries, base=2s). Other errors propagate immediately as an empty string
  response (matches the Claude adapter's behaviour) and are logged.
- Frame cap: default 256 images per request (matches the bumped Claude cap).
  Long buckets (L=512/1024) are uniformly subsampled down to this many
  frames and a WARNING is logged.
- FLOPs are not reported (0.0). Per-request ``usage`` (prompt_tokens,
  completion_tokens) is stashed on ``self._last_usage`` and surfaced via
  ``get_last_response_token_stats``.
- No GPU use (``max_gpu_mem`` ignored).
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


# OpenRouter accepts whatever the upstream model accepts; 256 is a safe
# shared ceiling that matches the other API adapter in this repo. Longer
# video buckets will be uniformly subsampled down to this many frames.
DEFAULT_MAX_IMAGES_PER_REQUEST = 256  # With JPEG default (~25 KB/frame at 448x448), 256 frames ~= 6 MB — fits Gemini's 20 MB inline ceiling.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional convenience slugs — verify against OpenRouter's /api/v1/models
# catalog before trusting these in production. If a slug is wrong the API
# returns a 404 which surfaces as an empty response (logged).
# TODO(user): confirm the exact slugs for Gemini 3.1 and GPT 5.5 against
# https://openrouter.ai/models once OPENROUTER_API_KEY is set.
# Verified 2026-04-24 via OpenRouter model listing + vendor docs.
# Image limits: Gemini ~3600 count, 20 MB inline (fits 256 JPEG frames at 448×448);
# GPT 1500 count / 512 MB. Defaults below use the BEST available slug per vendor.
# GPT 5.5 and Grok 5 do not currently exist — using top-tier current instead.
KNOWN_SLUGS = {
    "gemini-3.1": "google/gemini-3.1-pro-preview",
    "gpt-5.5":    "openai/gpt-5.4-pro",       # GPT 5.5 not live; 5.4-pro is the top-tier 5.4 variant
    "grok-5":     "x-ai/grok-4",              # Grok 5 not live yet — latest is grok-4
}


class OpenRouterAPIModel(VideoLanguageModelInterface):
    """Video-language model adapter backed by the OpenRouter chat API."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not model_id or not isinstance(model_id, str):
            raise ValueError(
                "OpenRouterAPIModel requires a non-empty model_id (the OpenRouter slug, e.g. 'google/gemini-3.1-pro')."
            )
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
        """Initialize the OpenAI-compat OpenRouter client and per-run state.

        Recognized kwargs:
            max_images_per_request (int): override the default image cap.
            image_format (str): "png" (default) or "jpeg".
            image_max_edge (Optional[int]): resize frames so the longer edge
                is at most this many pixels. Reduces token cost.
            request_timeout (float): per-request timeout in seconds.
            max_retries (int): max retries on 429/503 (default 3).
            http_referer (str): optional ``HTTP-Referer`` header (OpenRouter
                uses this for leaderboard attribution). Defaults to a local
                marker; harmless to leave as-is.
            app_title (str): optional ``X-Title`` header for the same.
            api_key (str): override the ``OPENROUTER_API_KEY`` env var.
        """
        del max_gpu_mem  # no GPU

        try:
            import openai  # noqa: WPS433 — deferred import.
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenRouterAPIModel. "
                "Install via `pip install openai` in the full-stack env."
            ) from exc

        api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before launching the "
                "eval (it will not be read from any file)."
            )

        # OpenRouter is OpenAI-protocol-compatible; just override base_url.
        # We deliberately don't enable the SDK's built-in retry because we
        # do our own (with per-error logging + 429/503 discrimination).
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=float(kwargs.get("request_timeout", 120.0)),
            max_retries=0,
        )
        self._openai = openai  # keep module handle for exception types

        self.max_images_per_request = int(
            kwargs.get("max_images_per_request", DEFAULT_MAX_IMAGES_PER_REQUEST)
        )
        self.image_format = str(kwargs.get("image_format", "jpeg")).lower()
        if self.image_format not in {"png", "jpeg"}:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        self._image_mime = (
            "image/png" if self.image_format == "png" else "image/jpeg"
        )
        self.image_max_edge: Optional[int] = kwargs.get("image_max_edge")
        self.max_retries = int(kwargs.get("max_retries", 3))
        self._retry_base_delay = float(kwargs.get("retry_base_delay", 2.0))

        # OpenRouter attribution headers (optional; the service recommends
        # setting them but does not require them).
        self._extra_headers: Dict[str, str] = {
            "HTTP-Referer": str(
                kwargs.get("http_referer", "https://github.com/streaming-mem")
            ),
            "X-Title": str(kwargs.get("app_title", "streaming-mem eval")),
        }

        # Documented but informational — the caller already samples at this fps.
        self.fps: float = float(kwargs.get("fps", 1.0))

        # Optional reasoning effort (OpenRouter unified field). Accepts
        # "low" / "medium" / "high"; if unset, no reasoning field is sent
        # (provider default applies). Roughly matches Claude adaptive thinking
        # when set to "medium" or "high".
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = os.environ.get("OPENROUTER_REASONING_EFFORT")
        if reasoning_effort:
            re_str = str(reasoning_effort).lower().strip()
            if re_str not in {"low", "medium", "high"}:
                raise ValueError(
                    f"reasoning_effort must be low/medium/high, got {re_str!r}"
                )
            self.reasoning_effort: Optional[str] = re_str
        else:
            self.reasoning_effort = None

        # Context state. Mirrors the Claude adapter.
        self.latest_time: float = 0.0
        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[tuple] = []  # (text, timestamp)

        # Last-response usage for logging / downstream inspection.
        self._last_usage: Optional[Dict[str, Any]] = None
        self._video_was_truncated: Optional[bool] = None

    # ------------------------------------------------------------------
    # Context mutation
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

        # Normalize to torch tensor on CPU.
        if isinstance(video_frames, np.ndarray):
            frames_tensor = torch.from_numpy(video_frames)
        elif isinstance(video_frames, torch.Tensor):
            frames_tensor = video_frames.detach().cpu()
        else:
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
        user_content = self._build_user_content(
            question=question,
            max_frames_in_video=max_frames_in_video,
            sample_method=sample_method,
        )
        messages = [{"role": "user", "content": user_content}]

        request_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "extra_headers": self._extra_headers,
        }
        if self.reasoning_effort:
            request_kwargs["extra_body"] = {
                "reasoning": {"effort": self.reasoning_effort}
            }

        response = self._call_with_retry(request_kwargs)
        if response is None:
            return ""

        # Parse OpenAI-style response.
        output_text = ""
        try:
            choices = getattr(response, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        output_text = content
                    elif isinstance(content, list):
                        # Some providers return a list of content parts.
                        parts: List[str] = []
                        for part in content:
                            text_val = None
                            if isinstance(part, dict):
                                text_val = part.get("text")
                            else:
                                text_val = getattr(part, "text", None)
                            if text_val:
                                parts.append(str(text_val))
                        output_text = "".join(parts)
        except Exception:  # pragma: no cover
            logger.exception("[OpenRouter] failed to parse response")

        # Stash usage + stop reason for downstream logging.
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                self._last_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "stop_reason": (
                        getattr(response.choices[0], "finish_reason", None)
                        if getattr(response, "choices", None)
                        else None
                    ),
                    "model": getattr(response, "model", None),
                }
            except Exception:  # pragma: no cover
                self._last_usage = None

        # Record metrics (flops=0.0; we can't measure them for API calls).
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
                flops=0.0,
                peak_gpu_mem_increase=0.0,
                peak_gpu_mem_absolute=0.0,
                video_time=question_timestamp,
                state_memory_total=0.0,
            )

        return output_text

    def _call_with_retry(self, request_kwargs: Dict[str, Any]):
        """Call chat.completions.create with 429/503 exponential backoff."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:  # broad: openai SDK wraps many HTTPX errors
                last_exc = exc
                status = _extract_status_code(exc)
                retriable = status in (408, 429, 500, 502, 503, 504)
                if not retriable or attempt >= self.max_retries:
                    logger.error(
                        "[OpenRouter] request failed (status=%s, attempt=%d/%d): %s",
                        status,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    return None
                delay = self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    "[OpenRouter] retriable error (status=%s, attempt=%d/%d); "
                    "sleeping %.1fs: %s",
                    status,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
        # Unreachable, but keep the linter happy.
        logger.error("[OpenRouter] exhausted retries: %s", last_exc)
        return None

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
            raise TypeError("OpenRouterAPIModel.load_state expects a dict state")
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

    def _build_user_content(
        self,
        question: str,
        max_frames_in_video: int,
        sample_method: str,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Build the ``content`` field for the single user message.

        Sequence/text-only runs (no video_segments, no text_entries) collapse
        to a plain string so providers that don't accept content-part arrays
        still work. Any richer timeline uses the standard OpenAI content-part
        schema:

            [
              {"type": "image_url", "image_url": {"url": "data:image/..."}},
              ...
              {"type": "text", "text": "<context>"},
              {"type": "text", "text": "<question>"},
            ]
        """
        has_video = any(
            seg.get("frames") is not None
            and getattr(seg["frames"], "shape", [0])[0] > 0
            for segs in self.video_segments.values()
            for seg in segs
        )
        has_text_ctx = bool(self.text_entries)

        if not has_video and not has_text_ctx:
            # Pure Q&A — no prior context.
            return question

        # Build timeline (videos first at their time_start, then interleaved
        # text entries). Matches the Claude adapter.
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

        # Aggregate per video_id, then apply the frame cap.
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
                "[OpenRouter] video_id=%s truncated from %d -> %d frames (cap=%d)",
                vid,
                n,
                effective_cap,
                effective_cap,
            )

        # Emit content parts in timeline order.
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
            else:
                content.append({"type": "text", "text": event["text"]})

        content.append({"type": "text", "text": question})
        return content

    def _frames_to_image_blocks(
        self, frames: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Convert a (N, 3, H, W) uint8 tensor into OpenAI image_url blocks."""
        if frames is None or frames.shape[0] == 0:
            return []

        if frames.dim() != 4:
            raise ValueError(
                f"Expected 4D frame tensor; got shape {tuple(frames.shape)}"
            )
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
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
            b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self._image_mime};base64,{b64}",
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


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Pull the HTTP status code off an openai SDK exception, if present."""
    # openai>=1.x exceptions expose .status_code on APIStatusError subclasses.
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "OpenRouterAPIModel",
    "DEFAULT_MAX_IMAGES_PER_REQUEST",
    "OPENROUTER_BASE_URL",
    "KNOWN_SLUGS",
]
