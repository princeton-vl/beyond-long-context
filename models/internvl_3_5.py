"""InternVL 3.5 model implementations."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from models.device_map_utils import build_max_memory_map

from frame_samplers.internvl_sampler import InternVLSampledVideo
from .base_interface import VideoLanguageModelInterface
from metrics.flops_calc import (
    INTERNVL35_NUM_IMAGE_TOKENS_PER_PATCH,
    internvl3_5_8b_flops,
    internvl3_5_30b_a3b_flops,
    internvl3_5_38b_flops,
)

# System prompt for thinking mode (per HuggingFace docs)
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


@dataclass
class _VideoSegment:
    clip: InternVLSampledVideo
    video_id: str
    frame_labels: List[str]
    time_start: float
    time_end: float


class InternVL35Model(VideoLanguageModelInterface):
    """InternVL 3.5 8B integration built on top of AutoModel.chat."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        *,
        is_thinking: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.is_thinking = is_thinking
        generation_override = kwargs.pop("generation_max_tokens", None)
        if generation_override is None:
            generation_override = kwargs.pop("max_tokens", None)
        self._generation_override = (
            int(generation_override) if generation_override is not None else None
        )
        self._model_kwargs = dict(kwargs)
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **self._model_kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        torch.set_float32_matmul_precision("high")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if self._has_flash_attention():
            load_kwargs["use_flash_attn"] = True
        load_kwargs.update(self._model_kwargs)
        if max_gpu_mem is not None and torch.cuda.is_available():
            load_kwargs["max_memory"] = build_max_memory_map(max_gpu_mem)
        # NOTE: Thinking mode is NOT a separate revision - it's the same model
        # with different system prompt and sampling parameters

        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False,
            padding_side="left",  # Required for decoder-only batched generation
        )
        # Set thinking mode system prompt if enabled
        if self.is_thinking:
            self.model.system_message = R1_SYSTEM_PROMPT
        self._text_token_cache: Dict[str, int] = {}
        self._primary_device = next(self.model.parameters()).device
        self._timeline: List[Dict[str, Any]] = []
        self._video_frame_counters: Dict[str, int] = {}
        self._video_was_truncated: Optional[bool] = None
        default_max = self._generation_override or 512
        # Thinking mode uses sampling with temperature 0.6 per HuggingFace docs
        if self.is_thinking:
            self._default_generation_config = {
                "do_sample": True,
                "temperature": 0.6,
                "max_new_tokens": max(1, default_max),
            }
        else:
            self._default_generation_config = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max(1, default_max),
            }

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def add_video(
        self,
        video_frames: Union[InternVLSampledVideo, torch.Tensor, np.ndarray],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())
        clip = self._coerce_clip(video_frames)
        vid_key = str(video_id if video_id is not None else 0)
        start_counter = self._video_frame_counters.get(vid_key, 0)
        frame_labels = [
            f"Video{vid_key} Frame{start_counter + idx + 1}: <image>"
            for idx in range(len(clip.num_patches_list))
        ]
        self._video_frame_counters[vid_key] = start_counter + len(clip.num_patches_list)
        segment = _VideoSegment(
            clip=clip,
            video_id=vid_key,
            frame_labels=frame_labels,
            time_start=time_start,
            time_end=time_end,
        )
        self._timeline.append({"type": "video", "segment": segment})

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())

        timestamp = float(current_video_time)
        self._timeline.append({"type": "text", "text": text, "time": timestamp})

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                timestamp,
                state_mem,
            )

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        # Frame-budget truncation happens upstream in InternVLFrameSampler (it caps
        # to max_frames during sampling), so the model side never trims here.
        # We still report a value so callers polling was_video_truncated() get
        # a defined answer.
        self._video_was_truncated = False
        clip = self._build_combined_clip()
        prompt = self._build_prompt(question)

        # Print raw prompt for debugging
        print("\n[model-input] ===== RAW CHAT PROMPT START (InternVL) =====")
        print(prompt)
        print("[model-input] ===== RAW CHAT PROMPT END (InternVL) =====", flush=True)

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))
        vision_patch_count = int(clip.pixel_values.shape[0]) if clip is not None else 0
        vision_height = int(clip.pixel_values.shape[2]) if clip is not None else 0
        vision_width = int(clip.pixel_values.shape[3]) if clip is not None else 0

        pixel_values = None
        num_patches_list: Optional[List[int]] = None
        if clip is not None:
            pixel_values = clip.pixel_values.to(
                dtype=torch.bfloat16,
                device=self._primary_device,
                non_blocking=True,
            )
            num_patches_list = clip.num_patches_list

        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
        )

        # Print raw response for debugging
        print("\n[model-output] ===== RAW MODEL RESPONSE START (InternVL) =====")
        print(response)
        print("[model-output] ===== RAW MODEL RESPONSE END (InternVL) =====", flush=True)

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            prompt_text_tokens = int(prompt_ids.shape[-1]) if prompt_ids is not None else 0
            response_ids = self.tokenizer(
                response,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            num_generated = int(response_ids.shape[-1]) if response_ids is not None else 0

            vision_tokens = vision_patch_count * INTERNVL35_NUM_IMAGE_TOKENS_PER_PATCH
            lang_prompt_len = prompt_text_tokens + vision_tokens

            flops_breakdown = self._flops_predictor()(
                vision_frames=vision_patch_count,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=lang_prompt_len,
                num_generated=num_generated,
                do_backward=False,
            )
            flops_value = float(flops_breakdown.get("total_flops", 0.0))

            latest_context_time = self._latest_context_timestamp()
            question_timestamp = float(current_video_time) if current_video_time > 0 else latest_context_time
            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                flops_value,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
            )
        return response

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> List[str]:
        """
        Ask multiple questions in parallel using batch_chat.
        All questions share the same video/text context.
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False
        clip = self._build_combined_clip()

        # Build prompts for all questions
        prompts = [self._build_prompt(q) for q in questions]

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))

        # Batched inference with OOM retry
        batch_size = len(questions)
        all_responses: List[Optional[str]] = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_prompts = prompts[chunk_start:chunk_end]
                    chunk_size = len(chunk_prompts)

                    # Prepare pixel values for this chunk
                    # batch_chat expects: len(pixel_values) == sum(num_patches_list)
                    # and len(num_patches_list) == len(questions)
                    chunk_pixel_values = None
                    chunk_num_patches = None

                    if clip is not None:
                        single_pixel_values = clip.pixel_values.to(
                            dtype=torch.bfloat16,
                            device=self._primary_device,
                            non_blocking=True,
                        )
                        # Replicate pixel_values for each question in chunk
                        chunk_pixel_values = torch.cat([single_pixel_values] * chunk_size, dim=0)
                        # Each question gets the full video: sum of original num_patches_list
                        num_patches_per_question = sum(clip.num_patches_list)
                        chunk_num_patches = [num_patches_per_question] * chunk_size
                    else:
                        # Text-only: no images
                        chunk_num_patches = [0] * chunk_size

                    responses = self.model.batch_chat(
                        self.tokenizer,
                        chunk_pixel_values,
                        num_patches_list=chunk_num_patches,
                        questions=chunk_prompts,
                        generation_config=generation_config,
                    )

                    for i, resp in enumerate(responses):
                        all_responses[chunk_start + i] = resp

                # Success - break out of retry loop
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[InternVL] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Keep partial responses across OOM retries — mirrors the
                    # isolated-batch path. Successful chunks stay populated; the
                    # smaller-batch retry re-runs from chunk_start=0 and just
                    # overwrites them with equivalent values.
                else:
                    raise

        # Debug output (gated)
        if os.environ.get("INTERNVL_DEBUG"):
            print(f"\n[model-output-batch] ===== BATCHED RESPONSE InternVL ({len(all_responses)} questions, batch_size={batch_size}) =====")
            for i, resp in enumerate(all_responses[:3]):
                print(f"Q{i+1}: {resp[:100] if resp else '(empty)'}...")
            if len(all_responses) > 3:
                print(f"... and {len(all_responses)-3} more")
            print("[model-output-batch] ===== END =====", flush=True)

        return all_responses

    def ask_question_batch_isolated(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
    ) -> List[str]:
        """
        TRUE parallel batching with isolated contexts per question.

        Each question gets its own complete context (main + ONE candidate).

        Args:
            contexts: List of context dicts, one per question.
                     For sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                        'question_text': str, 'mode': 'sequence'}
                     For video mode: {'main_video_frames': InternVLSampledVideo,
                                     'candidate_video_frames': InternVLSampledVideo,
                                     'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate
            max_frames_in_video: Max frames per video

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # InternVL: Keep batch_size=1 shortcut because batch_chat() has a prompt
        # structure mismatch (1 <image> token total vs 1 per frame in chat()).
        # The batch path needs prompt restructuring to fix properly.
        if len(contexts) == 1:
            return [self._process_single_context_isolated(contexts[0], max_tokens, max_frames_in_video)]

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        # Build separate prompts and pixel_values for each question
        prompts = []
        all_pixel_values = []
        all_num_patches = []

        for ctx in contexts:
            if is_sequence_mode:
                # Sequence mode: pure text
                prompt = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                prompts.append(prompt)
                all_num_patches.append(0)  # No images

            else:
                # Video mode: build separate clip for this question
                main_clip = ctx.get('main_video_frames')  # InternVLSampledVideo
                candidate_clip = ctx.get('candidate_video_frames')  # InternVLSampledVideo

                # Build prompt with ONE <image> per frame (matching single path)
                prompt_parts = []
                current_patches_for_question = []

                # Add main video frames with per-frame <image> tokens
                if main_clip is not None:
                    prompt_parts.append("Here is a main video to remember:")
                    for idx in range(len(main_clip.num_patches_list)):
                        prompt_parts.append(f"Main Frame{idx + 1}: <image>")
                    all_pixel_values.append(main_clip.pixel_values)
                    current_patches_for_question.extend(main_clip.num_patches_list)

                # Add candidate clip frames with per-frame <image> tokens
                if candidate_clip is not None:
                    prompt_parts.append("Here is a candidate clip:")
                    for idx in range(len(candidate_clip.num_patches_list)):
                        prompt_parts.append(f"Candidate Frame{idx + 1}: <image>")
                    all_pixel_values.append(candidate_clip.pixel_values)
                    current_patches_for_question.extend(candidate_clip.num_patches_list)

                # Add question text
                prompt_parts.append(f"\n{ctx['question_text']}")

                prompt = "\n".join(prompt_parts)
                prompts.append(prompt)

                # batch_chat needs per-QUESTION total (it splits internally across <image> tokens)
                total_patches_for_question = sum(current_patches_for_question)
                all_num_patches.append(total_patches_for_question)

        # Prepare for batch_chat
        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))

        # Combine pixel values if in video mode
        combined_pixel_values = None
        if not is_sequence_mode and all_pixel_values:
            combined_pixel_values = torch.cat(all_pixel_values, dim=0).to(
                dtype=torch.bfloat16,
                device=self._primary_device,
                non_blocking=True,
            )

        # Batch process with OOM retry
        batch_size = len(contexts)
        all_responses: List[Optional[str]] = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_prompts = prompts[chunk_start:chunk_end]
                    chunk_size = len(chunk_prompts)

                    # Prepare chunk data
                    chunk_pixel_values = None
                    chunk_num_patches = None

                    if is_sequence_mode:
                        # Text-only mode
                        chunk_num_patches = [0] * chunk_size
                    else:
                        # Video mode: slice pixel_values for this chunk
                        patches_before_chunk = sum(all_num_patches[:chunk_start])
                        patches_in_chunk = sum(all_num_patches[chunk_start:chunk_end])

                        if combined_pixel_values is not None:
                            chunk_pixel_values = combined_pixel_values[patches_before_chunk:patches_before_chunk + patches_in_chunk]
                        chunk_num_patches = all_num_patches[chunk_start:chunk_end]

                    # Call batch_chat
                    if os.environ.get("INTERNVL_DEBUG"):
                        print(f"[DEBUG batch_chat] chunk_size={len(chunk_prompts)}, "
                              f"pixel_values_shape={chunk_pixel_values.shape if chunk_pixel_values is not None else None}, "
                              f"num_patches_flat={chunk_num_patches} (length={len(chunk_num_patches) if chunk_num_patches else 0})")
                    responses = self.model.batch_chat(
                        self.tokenizer,
                        chunk_pixel_values,
                        num_patches_list=chunk_num_patches,
                        questions=chunk_prompts,
                        generation_config=generation_config,
                    )

                    for i, resp in enumerate(responses):
                        all_responses[chunk_start + i] = resp

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[InternVL-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                    # all_responses already has successful chunks, just retry failed ones
                else:
                    raise

        # Debug output (gated)
        if os.environ.get("INTERNVL_DEBUG"):
            print(f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE InternVL "
                  f"({len(all_responses)} questions, batch_size={batch_size}, mode={mode}) =====")
            if prompts and len(prompts) > 0:
                print(f"\n[DEBUG] First batched prompt structure:")
                print(prompts[0][:500] if len(prompts[0]) > 500 else prompts[0])
                print("...")
            for i, resp in enumerate(all_responses[:3]):
                preview = resp[:100] if resp else "(empty)"
                print(f"Q{i+1}: {preview}...")
            if len(all_responses) > 3:
                print(f"... and {len(all_responses)-3} more")
            print("[model-output-batch-isolated] ===== END =====", flush=True)

        return all_responses

    def _process_single_context_isolated(self, ctx: Dict[str, Any], max_tokens: int, max_frames_in_video: int) -> str:
        """Process a single isolated context (fallback for batch size 1)."""
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            # Text-only mode
            prompt = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            return self.ask_question(prompt, current_video_time=0.0, max_tokens=max_tokens)
        else:
            # Video mode - need to build timeline
            self.clear_context()

            main_clip = ctx.get('main_video_frames')
            candidate_clip = ctx.get('candidate_video_frames')

            # Add main video — duration = num_frames at 1fps
            main_dur = 0.0
            if main_clip is not None:
                num_main = len(main_clip.num_patches_list)
                main_dur = float(num_main)
                main_frame_labels = [
                    f"Main Frame{idx + 1}: <image>"
                    for idx in range(num_main)
                ]
                self._timeline.append({
                    "type": "video",
                    "segment": _VideoSegment(
                        clip=main_clip,
                        video_id="main_video",
                        frame_labels=main_frame_labels,
                        time_start=0.0,
                        time_end=main_dur
                    )
                })

            # Add candidate clip after main
            cand_start = main_dur + 1.0
            cand_dur = 0.0
            if candidate_clip is not None:
                num_cand = len(candidate_clip.num_patches_list)
                cand_dur = float(num_cand)
                candidate_frame_labels = [
                    f"Candidate Frame{idx + 1}: <image>"
                    for idx in range(num_cand)
                ]
                self._timeline.append({
                    "type": "video",
                    "segment": _VideoSegment(
                        clip=candidate_clip,
                        video_id="candidate_clip",
                        frame_labels=candidate_frame_labels,
                        time_start=cand_start,
                        time_end=cand_start + cand_dur
                    )
                })

            return self.ask_question(ctx['question_text'], current_video_time=cand_start + cand_dur, max_tokens=max_tokens)

    def get_state(self) -> Dict[str, Any]:
        return {
            "timeline": copy.deepcopy(self._timeline),
            "counters": dict(self._video_frame_counters),
        }

    def clear_context(self) -> None:
        self._timeline.clear()
        self._video_frame_counters.clear()
        self._video_was_truncated = None

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def save_state(self) -> Any:
        return copy.deepcopy(self.get_state())

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Invalid InternVL state")
        self._timeline = copy.deepcopy(state.get("timeline", []))
        self._video_frame_counters = dict(state.get("counters", {}))

    def _flops_predictor(self):
        """Return the FLOPs predictor matching this checkpoint's variant.

        Routes by self.model_id:
          - "...-38B"     -> internvl3_5_38b_flops
          - "...-30B-A3B" -> internvl3_5_30b_a3b_flops (MoE)
          - everything else (8B and unknown variants) -> internvl3_5_8b_flops
        """
        model_id = (self.model_id or "").upper()
        if "38B" in model_id:
            return internvl3_5_38b_flops
        if "30B-A3B" in model_id or "30B_A3B" in model_id:
            return internvl3_5_30b_a3b_flops
        return internvl3_5_8b_flops

    def _latest_context_timestamp(self) -> float:
        latest = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                latest = max(latest, float(entry["segment"].time_end))
            elif entry.get("type") == "text":
                latest = max(latest, float(entry.get("time", 0.0)))
        return latest

    def _get_state_memory_floats(self) -> float:
        total = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                clip = entry["segment"].clip
                total += float(clip.pixel_values.numel())
            elif entry.get("type") == "text":
                total += float(self._count_text_tokens(entry.get("text", "")))
        return total

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0

        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached

        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoding.get("input_ids") if isinstance(encoding, dict) else encoding.input_ids
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                count = len(input_ids[0])
            else:
                count = len(input_ids)
        elif hasattr(input_ids, "__len__"):
            count = len(input_ids)
        else:
            shape = getattr(input_ids, "shape", None)
            count = int(shape[0]) if shape else 0

        self._text_token_cache[text] = count
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_clip(
        self,
        video_frames: Union[InternVLSampledVideo, torch.Tensor, np.ndarray],
    ) -> InternVLSampledVideo:
        if isinstance(video_frames, InternVLSampledVideo):
            return InternVLSampledVideo(
                pixel_values=video_frames.pixel_values.detach().cpu(),
                num_patches_list=list(video_frames.num_patches_list),
                fps=video_frames.fps,
            )
        tensor = self._ensure_tensor(video_frames)
        if tensor.ndim != 4:
            raise ValueError("InternVL requires 4D tensors (frames, C, H, W)")
        if tensor.shape[1] not in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        num_frames = tensor.shape[0]
        counts = [1] * num_frames
        return InternVLSampledVideo(pixel_values=tensor, num_patches_list=counts, fps=1.0)

    @staticmethod
    def _ensure_tensor(data: Union[torch.Tensor, np.ndarray, InternVLSampledVideo]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            return tensor
        raise TypeError("Unsupported video frame type for InternVL model")

    def _build_combined_clip(self) -> Optional[InternVLSampledVideo]:
        if not self._timeline:
            return None
        pixel_values: List[torch.Tensor] = []
        counts: List[int] = []
        for entry in self._timeline:
            if entry.get("type") != "video":
                continue
            segment = entry["segment"]
            pixel_values.append(segment.clip.pixel_values)
            counts.extend(segment.clip.num_patches_list)
        if not pixel_values:
            return None

        # Safety check: under fixed INPUT_SIZE=448 + MAX_TILES=1 every clip
        # should land at the same (C, H, W). Previously we resized mismatched
        # tensors via bilinear interpolate; that path is unreachable today but
        # we keep the assert as a guard rail in case MAX_TILES is ever raised
        # and frames pick up tile-count or aspect-ratio variation again.
        if len(pixel_values) > 1:
            reference_shape = pixel_values[0].shape[1:]
            for pv in pixel_values[1:]:
                assert pv.shape[1:] == reference_shape, (
                    f"InternVL pixel_values shape mismatch: {pv.shape[1:]} vs "
                    f"{reference_shape}. If MAX_TILES > 1 was reintroduced, restore "
                    f"the bilinear-resize path that previously lived here."
                )

        combined = torch.cat(pixel_values, dim=0)
        return InternVLSampledVideo(pixel_values=combined, num_patches_list=counts, fps=1.0)

    def _build_prompt(self, question: str) -> str:
        lines: List[str] = []
        for entry in self._timeline:
            if entry.get("type") == "video":
                lines.extend(entry["segment"].frame_labels)
            elif entry.get("type") == "text":
                lines.append(entry["text"])
        if lines:
            lines.append(question)
            return "\n".join(lines)
        return question

    @staticmethod
    def _has_flash_attention() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("flash_attn") is not None
        except Exception:
            return False


class InternVL35ThinkingModel(InternVL35Model):
    """Thin wrapper that enables thinking mode."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            is_thinking=True,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )
