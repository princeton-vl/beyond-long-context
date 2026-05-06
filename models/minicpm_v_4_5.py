"""MiniCPM-V 4.5 model integration."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from models.device_map_utils import build_max_memory_map

from frame_samplers.minicpm_v_4_5_sampler import MiniCPM45Sample
from .base_interface import PerformanceMetrics, VideoLanguageModelInterface
# Reuses minicpm_v_2_6_flops; verify dim parity in metrics/flops_calc.py before changing
# model versions. (No minicpm_v_4_5_flops exists — same Qwen2-7B-derived backbone.)
from metrics.flops_calc import minicpm_v_2_6_flops


@dataclass
class _MiniCPMSegment:
    frames: List[Image.Image]
    temporal_ids: List[List[int]]
    video_id: str
    time_start: float
    time_end: float


class MiniCPM45Model(VideoLanguageModelInterface):
    """MiniCPM-V 4.5 wrapper around the HF chat interface."""

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        attn_impl = kwargs.pop("attn_implementation", "sdpa")
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        device_map = kwargs.pop("device_map", "auto")

        # CRITICAL FOR BATCHING: Avoid offloading to disk/CPU (meta device)
        # The issue: device_map="auto" with max_memory constraints that are TOO SMALL
        # causes Accelerate to offload parts to disk, breaking batching with "Cannot copy
        # out of meta tensor" errors.
        #
        # Solutions:
        # 1. Multi-GPU: Use device_map="auto" WITHOUT max_memory (natural GPU split, no offloading)
        # 2. Single GPU: Either use device_map="auto" without max_memory, or "cuda:0"
        #
        # Model size: ~17GB actual usage, fits on any 25GB+ GPU
        load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }

        # Strategy: Only use max_memory if explicitly requested AND sufficient to avoid offloading
        # Otherwise let Accelerate naturally distribute across available GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_mem_gb = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count))

            if max_gpu_mem is not None and device_map == "auto":
                # User requested max_gpu_mem - check if it would cause offloading
                # Model needs ~17GB minimum, so each GPU should have at least 20GB to be safe
                estimated_total = max_gpu_mem * gpu_count

                if estimated_total >= 20.0:
                    # Sufficient memory - use max_memory
                    load_kwargs["max_memory"] = build_max_memory_map(max_gpu_mem)
                    print(f"[MiniCPM-4.5] Using {gpu_count} GPU(s) with max_memory={max_gpu_mem:.1f}GB each")
                else:
                    # Insufficient - would cause offloading, skip max_memory
                    print(f"[MiniCPM-4.5] WARNING: max_gpu_mem={max_gpu_mem:.1f}GB too small, ignoring to avoid offloading")
                    print(f"[MiniCPM-4.5] Using device_map='auto' with full GPU memory ({total_mem_gb:.1f}GB total)")
            else:
                print(f"[MiniCPM-4.5] Using device_map='{device_map}' across {gpu_count} GPU(s) ({total_mem_gb:.1f}GB total)")

        load_kwargs.update(kwargs)
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self._timeline: List[Dict[str, Any]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._history: Optional[Any] = None
        self._video_was_truncated: Optional[bool] = None
        self._default_generation_config = {
            "use_image_id": False,
            "max_slice_nums": 1,
            "max_new_tokens": 512,
            "eos_token_id": self.tokenizer.eos_token_id,  # Stop at <|im_end|>
        }

    def add_video(
        self,
        video_frames: Union[MiniCPM45Sample, Sequence[Image.Image]],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        metrics_start, baseline_mem = self._metrics_begin()
        segment = self._coerce_segment(video_frames, time_start, time_end, video_id)
        self._timeline.append({"type": "video", "segment": segment})
        if self.enable_metrics:
            latency, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "add_video",
                latency=latency,
                flops=0.0,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=time_end,
                state_memory_total=state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        metrics_start, baseline_mem = self._metrics_begin()
        self._timeline.append({"type": "text", "text": text, "time": current_video_time})
        if self.enable_metrics:
            latency, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "add_text",
                latency=latency,
                flops=0.0,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=current_video_time,
                state_memory_total=state_mem,
            )

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        self._video_was_truncated = False  # MiniCPM 4.5 doesn't truncate frames
        metrics_start, baseline_mem = self._metrics_begin()
        frames, temporal_ids, content = self._build_chat_inputs(question)

        msgs = [{"role": "user", "content": content}]
        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        start = time.perf_counter()
        response = None
        last_error = None

        # Calculate appropriate max_inp_length based on accumulated context
        # Each video segment adds ~500 tokens, so scale based on number of segments
        video_segments = len([e for e in self._timeline if e.get("type") == "video"])
        # Default is 16384, increase if we have many videos to prevent truncation
        max_inp_length = max(16384, video_segments * 1000)

        # Try with temporal_ids first if available
        if temporal_ids is not None:
            try:
                response = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    temporal_ids=temporal_ids,
                    max_inp_length=max_inp_length,
                    **generation_config,
                )
            except RuntimeError as e:
                if "Sizes of tensors must match" in str(e):
                    print(f"Warning: Tensor shape mismatch with temporal_ids, trying without: {e}")
                    last_error = e
                else:
                    raise

        # If temporal_ids failed or weren't provided, try without them
        if response is None:
            try:
                response = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    temporal_ids=None,
                    max_inp_length=max_inp_length,
                    **generation_config,
                )
            except RuntimeError as e:
                print(f"Error: Failed even without temporal_ids")
                print(f"  Current error: {e}")
                if last_error:
                    print(f"  First error (with temporal_ids): {last_error}")
                print(f"  Video segments in context: {len([e for e in self._timeline if e.get('type') == 'video'])}")
                print(f"  Used max_inp_length: {max_inp_length}")
                raise

        latency = time.perf_counter() - start
        if self.enable_metrics:
            gpu_latency, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            # gpu_latency is the measured latency after optional torch operations; prefer actual latency
            _ = gpu_latency
            generated_tokens = self._count_text_tokens(response)
            prompt_tokens = self._context_text_token_count() + self._count_text_tokens(question)
            flops = self._estimate_question_flops(prompt_tokens, generated_tokens)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "ask_question",
                latency=latency,
                flops=flops,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=current_video_time,
                state_memory_total=state_mem,
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
        Ask multiple questions sharing the same video/text context.
        Uses model.chat with batched msgs format.
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False

        # Build shared content (frames + text) for all questions
        frames, temporal_ids, base_content = self._build_chat_inputs("")
        # Remove empty string we just added
        if base_content and base_content[-1] == "":
            base_content = base_content[:-1]

        # Build batch of message sequences, one per question
        batch_msgs = []
        for question in questions:
            content = list(base_content)
            content.append(question)
            batch_msgs.append([{"role": "user", "content": content}])

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        video_segments = len([e for e in self._timeline if e.get("type") == "video"])
        max_inp_length = max(16384, video_segments * 1000)

        all_responses: List[str] = []
        batch_succeeded = False

        # Try batched inference once. The previous control flow used a
        # `while batch_size >= 1` retry loop, but every path inside it either
        # broke on success or broke on OOM (no continue), so the loop never
        # iterated — replaced with a flat try/except that falls through to the
        # sequential path below on any failure.
        try:
            responses = None
            if temporal_ids is not None:
                try:
                    responses = self.model.chat(
                        msgs=batch_msgs,
                        tokenizer=self.tokenizer,
                        temporal_ids=temporal_ids,
                        max_inp_length=max_inp_length,
                        **generation_config,
                    )
                    if isinstance(responses, str):
                        raise ValueError("Batch inference not supported, falling back")
                except (RuntimeError, ValueError) as e:
                    msg = str(e)
                    if "out of memory" in msg.lower():
                        raise
                    if (
                        "Sizes of tensors must match" not in msg
                        and "Batch inference not supported" not in msg
                    ):
                        raise
                    # Fall through to retry without temporal_ids
                    responses = None

            if responses is None:
                responses = self.model.chat(
                    msgs=batch_msgs,
                    tokenizer=self.tokenizer,
                    temporal_ids=None,
                    max_inp_length=max_inp_length,
                    **generation_config,
                )
                if isinstance(responses, str):
                    raise ValueError("Batch inference not supported")

            all_responses = responses if isinstance(responses, list) else [responses]
            batch_succeeded = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[MiniCPM] OOM during batched inference; falling back to sequential")
                all_responses = []
            else:
                raise
        except ValueError:
            # Batch path not supported by this model build — sequential fallback.
            all_responses = []

        if not batch_succeeded:
            # Batch didn't work - fall back to sequential
            print(f"[MiniCPM] Batch inference not supported, using sequential processing")
            all_responses = []
            for question in questions:
                try:
                    result = self.ask_question(question, current_video_time, max_tokens, max_frames_in_video, sample_method)
                    all_responses.append(result)
                except RuntimeError as e2:
                    if "out of memory" in str(e2).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"[MiniCPM] OOM, appending empty response")
                        all_responses.append("")
                    else:
                        raise

        # Debug output
        mode = "batched" if len(all_responses) == len(questions) and "not supported" not in str(all_responses) else "sequential"
        print(f"\n[model-output-batch] ===== RESPONSE MiniCPM ({len(all_responses)} questions, {mode}) =====")
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

        Each question gets its own complete context (main + candidate).

        Args:
            contexts: List of context dicts, one per question.
                     For sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                        'question_text': str, 'mode': 'sequence'}
                     For video mode: {'main_video_frames': List[Image], 'candidate_video_frames': List[Image],
                                     'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate per response
            max_frames_in_video: Max frames per video

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # NOTE: Do NOT shortcut to _process_single_context_isolated for batch_size=1.
        # That path uses different temporal_ids/sampling/max_inp_length logic.
        # Always use the batch code path for consistency.

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        # Build batch of messages, one per question
        batch_messages = []

        for ctx in contexts:
            content = []

            if is_sequence_mode:
                # Sequence mode: pure text
                full_text = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                content.append(full_text)
            else:
                # Video mode: main video + candidate clip as frames
                main_frames_obj = ctx.get('main_video_frames')
                candidate_frames_obj = ctx.get('candidate_video_frames')

                # Extract frame lists from MiniCPM45Sample objects or use directly
                from frame_samplers.minicpm_v_4_5_sampler import MiniCPM45Sample

                if isinstance(main_frames_obj, MiniCPM45Sample):
                    main_frames = main_frames_obj.frames
                    main_temporal = main_frames_obj.temporal_ids
                elif isinstance(main_frames_obj, (list, tuple)):
                    main_frames = main_frames_obj
                    main_temporal = None
                else:
                    main_frames = []
                    main_temporal = None

                if isinstance(candidate_frames_obj, MiniCPM45Sample):
                    candidate_frames = candidate_frames_obj.frames
                    candidate_temporal = candidate_frames_obj.temporal_ids
                elif isinstance(candidate_frames_obj, (list, tuple)):
                    candidate_frames = candidate_frames_obj
                    candidate_temporal = None
                else:
                    candidate_frames = []
                    candidate_temporal = None

                # Add main video frames - MODIFIED TO MATCH NON-BATCHED MODE
                if main_frames:
                    # Add text anchor BEFORE frames (matches non-batched line 2529)
                    content.append("Here is a main video to remember:")
                    for frame in main_frames[:max_frames_in_video // 2]:
                        content.append(frame)

                # Add candidate clip frames - MODIFIED TO MATCH NON-BATCHED MODE
                if candidate_frames:
                    # Add text anchor BEFORE frames (matches non-batched line 2561)
                    content.append("\nHere is a candidate clip:\n")
                    for frame in candidate_frames[:max_frames_in_video // 2]:
                        content.append(frame)

                # Add question text
                content.append(ctx['question_text'])

            batch_messages.append([{"role": "user", "content": content}])

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        # Batched inference with OOM retry (halve batch size on OOM)
        batch_size = len(contexts)
        all_responses = []

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_msgs = batch_messages[chunk_start:chunk_end]

                    # Call model.chat() with batch of messages
                    # Match single-path behavior: no forced sampling, dynamic max_inp_length
                    responses = self.model.chat(
                        msgs=chunk_msgs,
                        tokenizer=self.tokenizer,
                        temporal_ids=None,
                        max_inp_length=max(16384, len(chunk_msgs) * 1000),
                        **generation_config,
                    )

                    # Handle response
                    if isinstance(responses, str):
                        # Single question case - wrap in list
                        all_responses.append(responses)
                    elif isinstance(responses, list):
                        all_responses.extend(responses)
                    else:
                        raise ValueError(f"Unexpected response type: {type(responses)}")

                # Success
                break

            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg and batch_size > 1:
                    # OOM - try smaller batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[MiniCPM45-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    all_responses = []
                else:
                    # Other error - re-raise
                    raise

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== RESPONSE MiniCPM-4.5 ({len(all_responses)} questions, mode={mode}) =====")
        for i, resp in enumerate(all_responses[:3]):
            print(f"Q{i+1}: {resp[:100] if resp else '(empty)'}...")
        if len(all_responses) > 3:
            print(f"... and {len(all_responses)-3} more")
        print("[model-output-batch-isolated] ===== END =====", flush=True)

        return all_responses

    # _process_single_context_isolated was removed: ask_question_batch_isolated
    # always uses the batch code path (see comment at the top of that method),
    # so the helper had no callers.

    def get_state(self) -> Dict[str, Any]:
        return {
            "timeline": copy.deepcopy(self._timeline),
        }

    def clear_context(self) -> None:
        self._timeline.clear()
        self._video_was_truncated = None

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def save_state(self) -> Any:
        return copy.deepcopy(self.get_state())

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Invalid MiniCPM-V state payload")
        self._timeline = copy.deepcopy(state.get("timeline", []))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coerce_segment(
        self,
        video_frames: Union[MiniCPM45Sample, Sequence[Image.Image]],
        time_start: float,
        time_end: float,
        video_id: Optional[int],
    ) -> _MiniCPMSegment:
        vid_key = str(video_id if video_id is not None else 0)
        if isinstance(video_frames, MiniCPM45Sample):
            frames = [self._ensure_image(frame) for frame in video_frames.frames]
            temporal = [list(ids) for ids in video_frames.temporal_ids]
        else:
            frames = [self._ensure_image(frame) for frame in video_frames]
            temporal = []
        if temporal and sum(len(ids) for ids in temporal) != len(frames):
            raise ValueError("MiniCPM temporal ids must align with frames")
        return _MiniCPMSegment(
            frames=frames,
            temporal_ids=temporal,
            video_id=vid_key,
            time_start=time_start,
            time_end=time_end,
        )

    @staticmethod
    def _ensure_image(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame)
        raise TypeError("MiniCPM frames must be PIL.Image or numpy arrays")

    def _build_chat_inputs(
        self, question: str
    ) -> tuple[List[Image.Image], Optional[List[List[int]]], List[Any]]:
        frames: List[Image.Image] = []
        temporal: List[List[int]] = []
        temporal_available = True
        content: List[Any] = []

        for entry in self._timeline:
            if entry.get("type") == "video":
                segment = entry["segment"]
                frames.extend(segment.frames)
                if segment.temporal_ids:
                    temporal.extend(segment.temporal_ids)
                else:
                    temporal_available = False
                content.extend(segment.frames)
            elif entry.get("type") == "text":
                content.append(entry["text"])

        content.append(question)
        temporal_ids = temporal if temporal and temporal_available else None
        return frames, temporal_ids, content

    def _record_metrics_event(
        self,
        event: str,
        *,
        latency: float,
        flops: float,
        peak_increase_mb: float,
        peak_absolute_mb: float,
        video_time: float,
        state_memory_total: float,
    ) -> None:
        if not self.enable_metrics or not isinstance(self._metrics, PerformanceMetrics):
            return
        if event == "add_video":
            self._record_add_video_metrics(
                latency,
                flops,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )
        elif event == "add_text":
            self._record_add_text_metrics(
                latency,
                flops,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )
        elif event == "ask_question":
            self._record_ask_question_metrics(
                latency,
                flops,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )

    def _metrics_begin(self) -> Tuple[Optional[float], float]:
        if not self.enable_metrics:
            return None, 0.0
        baseline_mem = 0.0
        if torch.cuda.is_available():
            baseline_mem = float(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), baseline_mem

    def _metrics_end(self, start_time: Optional[float], baseline_mem: float) -> Tuple[float, float, float]:
        if not self.enable_metrics or start_time is None:
            return 0.0, 0.0, 0.0
        latency = time.perf_counter() - start_time
        peak_bytes = baseline_mem
        if torch.cuda.is_available():
            peak_bytes = float(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        peak_increase_mb = max(0.0, peak_bytes - baseline_mem) / (1024 * 1024)
        peak_absolute_mb = (
            peak_bytes / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        return latency, peak_increase_mb, peak_absolute_mb

    def _estimate_state_memory_floats(self) -> float:
        total = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                segment: _MiniCPMSegment = entry["segment"]
                for frame in segment.frames:
                    total += self._frame_float_count(frame)
            elif entry.get("type") == "text":
                total += self._count_text_tokens(entry.get("text", ""))
        return total

    @staticmethod
    def _frame_float_count(frame: Any) -> float:
        if isinstance(frame, Image.Image):
            array_view = np.asarray(frame)
            return float(array_view.size)
        return 0.0

    def _context_text_token_count(self) -> int:
        total = 0
        for entry in self._timeline:
            if entry.get("type") == "text":
                total += self._count_text_tokens(entry.get("text", ""))
        return total

    def _vision_statistics(self) -> Tuple[int, int, int]:
        total_frames = 0
        height = 0
        width = 0
        for entry in self._timeline:
            if entry.get("type") != "video":
                continue
            segment: _MiniCPMSegment = entry["segment"]
            total_frames += len(segment.frames)
            if segment.frames and (height == 0 or width == 0):
                frame = segment.frames[0]
                width, height = frame.size
        return total_frames, height, width

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

    def _estimate_question_flops(self, prompt_tokens: int, generated_tokens: int) -> float:
        vision_frames, height, width = self._vision_statistics()
        flops_info = minicpm_v_2_6_flops(
            vision_frames=vision_frames,
            vision_height=height,
            vision_width=width,
            lang_prompt_len=prompt_tokens,
            num_generated=generated_tokens,
            do_backward=False,
        )
        return float(flops_info.get("total_flops", 0.0))
