"""Qwen3 VL 8B model adapter for the streaming evaluation harness."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

from metrics.flops_calc import qwen3_vl_8b_thinking_flops
from models.qwen2_5_vl import CACHE_ROOT, QwenFullVideo


class Qwen3Dense(QwenFullVideo):
    """Drop-in replacement that upgrades QwenFullVideo to the Qwen3 VL stack.

    The shared ask_question / ask_question_batch logic lives on the parent.
    This subclass only overrides the hooks where Qwen3's processor disagrees
    with Qwen2.5: explicit VideoMetadata, CPU-side video tensors, nested-list
    batched payloads, and a higher per-video pixel cap.
    """

    INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    THINKING_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

    # Cap on the processor's longest-edge pixel total per video. The default
    # (~25M = 1024*448*448) silently downsamples large videos; we keep the same
    # ceiling explicit so every frame stays at 448x448 even at N=1024 frames.
    MAX_VIDEO_LONGEST_EDGE_PIXELS = 1024 * 448 * 448

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        thinking: bool = False,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the Qwen3 model, defaulting to the instruct variant."""

        resolved_model_id = model_id or (
            self.THINKING_MODEL_ID if thinking else self.INSTRUCT_MODEL_ID
        )

        super().__init__(
            resolved_model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """Instantiate Qwen3 VL assets while keeping the shared context plumbing.

        max_gpu_mem is accepted to match the base signature but unused: Accelerate
        handles placement via device_map="auto".
        """
        del max_gpu_mem  # silence unused-arg linters; keep for signature parity

        min_pixels: Optional[int] = kwargs.get("min_pixels")
        max_pixels: Optional[int] = kwargs.get("max_pixels")

        model_kwargs: Dict[str, object] = {
            "dtype": torch.bfloat16,  # Qwen3 from_pretrained uses dtype=, not torch_dtype=
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        processor_kwargs: Dict[str, object] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs["cache_dir"] = CACHE_ROOT
        processor_kwargs["padding_side"] = "left"  # Required for decoder-only models

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            **processor_kwargs,
        )

        # Disable silent video downsampling: ensure each frame stays at 448x448
        # even at N=1024 frames.
        self.processor.video_processor.size["longest_edge"] = self.MAX_VIDEO_LONGEST_EDGE_PIXELS

        self.latest_time = 0.0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, "_reset_state_memory_tracking"):
            self._reset_state_memory_tracking()

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _build_video_payload(
        self,
        video_id,
        data: Dict[str, object],
    ) -> Tuple[Optional[torch.Tensor], Optional[VideoMetadata], Optional[float]]:
        """Build (frames_on_cpu, VideoMetadata, fps) for a single video.

        Qwen3 VL requires an explicit VideoMetadata per video and rejects
        tensors that arrive on a non-CPU device.
        """
        frames = data.get("frames")
        if frames is None:
            return None, None, None
        if torch.is_tensor(frames):
            frames = frames.cpu()
            frame_count = int(frames.shape[0])
            if frame_count == 0:
                return frames, None, None
            duration = float(data.get("total_duration", 0.0) or 0.0)
            safe_duration = duration if duration > 0 else max(frame_count / 24.0, 1e-3)
            fps_value = frame_count / safe_duration if safe_duration > 0 else None
            height = int(frames.shape[2]) if frames.ndim >= 3 else None
            width = int(frames.shape[3]) if frames.ndim >= 4 else None
            frame_indices = data.get("frame_indices") or list(range(frame_count))
            metadata = VideoMetadata(
                total_num_frames=frame_count,
                fps=fps_value,
                width=width,
                height=height,
                duration=safe_duration,
                video_backend="tensor",
                frames_indices=[int(idx) for idx in frame_indices],
            )
            return frames, metadata, fps_value
        # Non-tensor frame container: punt to base behaviour.
        return frames, None, None

    def _apply_processor_video_kwargs(
        self,
        processor_kwargs,
        video_inputs,
        video_metadata,
        video_fps,
        *,
        batched: bool,
        batch_size: int = 1,
    ) -> None:
        """Wire videos + video_metadata into Qwen3's nested-list processor API."""
        if not video_inputs:
            return

        if not batched:
            processor_kwargs["videos"] = video_inputs
            if video_metadata:
                processor_kwargs["video_metadata"] = video_metadata
        else:
            # Qwen3 VL processor expects nested lists for batched videos:
            # videos = [[v1, v2], [v1, v2], ...]  # one inner list per text
            # video_metadata = [[m1, m2], [m1, m2], ...]  # parallel structure
            processor_kwargs["videos"] = [video_inputs for _ in range(batch_size)]
            if video_metadata:
                processor_kwargs["video_metadata"] = [video_metadata for _ in range(batch_size)]

        coalesced_fps = self._coalesce_video_fps(video_fps)
        if coalesced_fps is not None:
            processor_kwargs["fps"] = coalesced_fps

    def _estimate_ask_question_flops(
        self,
        *,
        vision_frames: int,
        vision_height: int,
        vision_width: int,
        lang_prompt_len: int,
        num_generated: int,
        do_backward: bool,
    ) -> float:
        """Return the FLOPs estimate for Qwen3 question turns.

        We currently route both Instruct and Thinking variants through the
        thinking predictor — the per-token compute is identical (same backbone)
        and the predictor doesn't bake in any thinking-specific overhead.
        """

        flops_breakdown = qwen3_vl_8b_thinking_flops(
            vision_frames=vision_frames,
            vision_height=vision_height,
            vision_width=vision_width,
            lang_prompt_len=lang_prompt_len,
            num_generated=num_generated,
            do_backward=do_backward,
        )
        if isinstance(flops_breakdown, dict):
            total = flops_breakdown.get("total_flops", 0.0)
        else:
            total = flops_breakdown
        return float(total)
