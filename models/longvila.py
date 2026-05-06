"""LongVILA-R1-7B integration with the streaming benchmark interface."""

from __future__ import annotations

import copy
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from models.device_map_utils import build_max_memory_map
from utils.paths import get_model_cache_dir

from models.base_interface import VideoLanguageModelInterface
from metrics.flops_calc import longvila_r1_7b_flops

EXTERNAL_LONGVILA_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "LongVILA")
if EXTERNAL_LONGVILA_DIR not in sys.path:
    sys.path.append(EXTERNAL_LONGVILA_DIR)

from tokenizer_utils import tokenize_conversation, auto_set_conversation_mode  # type: ignore
from mm_utils import process_images  # type: ignore


CACHE_ROOT = get_model_cache_dir()
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)


DEFAULT_MODEL_ID = "Efficient-Large-Model/LongVILA-R1-7B"
VIDEO_TOKEN = "<vila/video>"
# LongVILA-R1 model card ships a thinking system prompt. Without it, the model
# emits a single-token answer ({0} / {uncertain}) and ignores the video content.
# See HF card "Efficient-Large-Model/LongVILA-R1-7B" — reasoning example.
SYSTEM_PROMPT = (
    "You are a helpful assistant. The user asks a question about a video, and "
    "you answer it.\n\n"
    "Please first think deeply about the question based on the given video, "
    "and then provide your final answer. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively. For example: "
    "<think> reasoning here </think> <answer> answer here </answer>.\n\n"
    "When the user's prompt specifies an answer convention (e.g. \"write {0} "
    "for yes and {1} for no\"), follow that convention exactly inside the "
    "<answer> tags. Do not invert it: if the prompt says write 0 for yes, "
    "then write <answer>0</answer> for yes and <answer>1</answer> for no."
)


class LongVILAModel(VideoLanguageModelInterface):
    """Thin wrapper that reproduces LongVILA's media + prompt pipeline."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id or DEFAULT_MODEL_ID,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", self.model_id or DEFAULT_MODEL_ID)
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "cache_dir": CACHE_ROOT,
        }
        if max_gpu_mem is not None and torch.cuda.is_available():
            load_kwargs["max_memory"] = build_max_memory_map(max_gpu_mem)
        load_kwargs.update(kwargs.get("model_load_kwargs", {}))

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)

        provided_tokenizer = getattr(self.model, "tokenizer", None)
        if provided_tokenizer is not None:
            self.tokenizer = provided_tokenizer
        else:
            tokenizer_id = kwargs.get("tokenizer_id") or kwargs.get("model_name") or DEFAULT_MODEL_ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=CACHE_ROOT,
            )
            resize_fn = getattr(self.model, "resize_token_embeddings", None)
            if callable(resize_fn):
                try:
                    resize_fn(len(self.tokenizer))
                except Exception:
                    pass

        # Set left padding for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name, cache_dir=CACHE_ROOT)
        except Exception:
            self.generation_config = GenerationConfig()

        if not hasattr(self.model, "vision_tower") or not hasattr(self.model.vision_tower, "image_processor"):
            raise RuntimeError("LongVILA model does not expose a vision image processor")

        self.image_processor = self.model.vision_tower.image_processor
        # Use input embeddings device for proper multi-GPU support
        try:
            input_embeddings = self.model.get_input_embeddings()
            self.device = input_embeddings.weight.device if input_embeddings is not None else next(self.model.parameters()).device
        except (NotImplementedError, AttributeError):
            # Fallback for custom models (like VILAForCausalLM) that don't implement get_input_embeddings()
            self.device = next(self.model.parameters()).device
        self.system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        auto_set_conversation_mode(model_name)

        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[Tuple[str, float]] = []
        self.latest_time: float = 0.0
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = self.tokenizer
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, "_reset_state_memory_tracking"):
            self._reset_state_memory_tracking()

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def add_video(self, video_frames: torch.Tensor, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")

        frames_tensor = self._ensure_tensor(video_frames)
        if frames_tensor.shape[0] == 0:
            raise ValueError("Video segment must contain at least one frame")

        self.latest_time = max(self.latest_time, time_end)
        target_id = 0 if video_id is None else video_id

        if target_id not in self.video_segments:
            self.video_segments[target_id] = []

        segment = {
            "frames": frames_tensor.contiguous(),
            "time_start": float(time_start),
            "time_end": float(time_end),
            "duration": float(time_end - time_start),
            "num_frames": int(frames_tensor.shape[0]),
        }
        self.video_segments[target_id].append(segment)

        if self.enable_metrics:
            # add_video only mutates Python-side bookkeeping (no CUDA work),
            # so latency is effectively zero; peak-memory deltas are still
            # reported in case a caller pre-loaded the frames onto device.
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_increase = max(0, peak_mem - baseline_mem) / (1024 * 1024)
            peak_absolute = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                0.0,
                0.0,
                peak_increase,
                peak_absolute,
                time_end,
                state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        if text is None:
            return

        timestamp = float(current_video_time) if current_video_time is not None else self.latest_time
        self.text_entries.append((text, timestamp))

        if self.enable_metrics:
            # add_text only appends to a Python list; latency is effectively zero.
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_increase = max(0, peak_mem - baseline_mem) / (1024 * 1024)
            peak_absolute = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                0.0,
                0.0,
                peak_increase,
                peak_absolute,
                timestamp,
                state_mem,
            )

    def reset_context(self) -> None:
        self.video_segments.clear()
        self.text_entries.clear()
        self.latest_time = 0.0
        self._video_was_truncated = None

    def clear_context(self) -> None:
        self.reset_context()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def get_state(self) -> Dict[str, Any]:
        return {
            "video_segments": self._clone_segments(self.video_segments),
            "text_entries": list(self.text_entries),
            "latest_time": self.latest_time,
        }

    def save_state(self) -> Dict[str, Any]:
        return self.get_state()

    def load_state(self, state: Dict[str, Any]) -> None:
        cloned_segments: Dict[int, List[Dict[str, Any]]] = {}
        for video_id, segments in state.get("video_segments", {}).items():
            cloned_segments[int(video_id)] = [self._clone_segment(segment) for segment in segments]
        self.video_segments = cloned_segments
        self.text_entries = [tuple(entry) for entry in state.get("text_entries", [])]
        self.latest_time = float(state.get("latest_time", 0.0))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 512,
        sample_method: str = "TIME",
        **_: Any,
    ) -> str:
        if not question:
            raise ValueError("Question text must be provided")

        timeline = self._build_timeline()
        prompt_text, media_videos = self._build_prompt_and_media(timeline, question, max_frames_in_video)

        if not prompt_text:
            prompt_text = question

        conversation = [
            {"from": "human", "value": prompt_text},
        ]
        input_ids = tokenize_conversation(
            conversation,
            self.tokenizer,
            add_generation_prompt=True,
        ).unsqueeze(0).to(self.device)

        media_payload = None
        media_config = None
        if media_videos:
            media_payload = {"video": [video.to(self.device) for video in media_videos]}
            media_config = defaultdict(dict)

        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = max(1, int(max_tokens))

        if self.enable_metrics:
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            start_time = time.perf_counter()
        generation_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "media_config": media_config if media_config is not None else defaultdict(dict),
            # LongVILA's generate() concatenates input_ids with output_ids by default,
            # which would cause batch_decode below to re-emit the prompt as part of
            # the response. The isolated batch path already passes this flag.
            "return_output_ids_only": True,
        }
        if media_payload is not None:
            generation_kwargs["media"] = media_payload
        output_ids = self.model.generate(**generation_kwargs)

        response = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_increase = max(0, peak_mem - baseline_mem) / (1024 * 1024)
            peak_absolute = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            vision_frames = sum(video.shape[0] for video in media_videos) if media_videos else 0
            lang_prompt_len = int(input_ids.shape[1])
            num_generated = int(output_ids.shape[1]) if output_ids.numel() > 0 else 0
            flops_value = self._estimate_ask_question_flops(
                vision_frames=vision_frames,
                lang_prompt_len=lang_prompt_len,
                num_generated=num_generated,
                do_backward=False,
            )

            self._record_ask_question_metrics(
                latency,
                flops_value,
                peak_increase,
                peak_absolute,
                current_video_time,
                self._get_state_memory_floats(),
            )

        return response

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 512,
        sample_method: str = "TIME",
        **_: Any,
    ) -> List[str]:
        """
        Ask multiple questions sharing the same video/text context.
        Uses sequential processing (LongVILA's complex media/prompt handling makes batching difficult).
        """
        if not questions:
            return []

        # LongVILA's system prompt injection and media handling don't batch well
        # Use sequential processing
        all_responses: List[str] = []
        for question in questions:
            try:
                response = self.ask_question(question, current_video_time, max_tokens, max_frames_in_video, sample_method)
                all_responses.append(response)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[LongVILA] OOM, appending empty response")
                    all_responses.append("")
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch] ===== RESPONSE LongVILA ({len(all_responses)} questions, sequential) =====")
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
        max_frames_in_video: int = 512,
    ) -> List[str]:
        """
        TRUE parallel batching with isolated contexts per question.

        Currently supports SEQUENCE MODE ONLY (text-only batching).
        Video mode falls back to sequential due to LongVILA architectural limitations.

        Args:
            contexts: List of context dicts, one per question.
                     Sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                    'question_text': str, 'mode': 'sequence'}
                     Video mode: Not supported, will use sequential fallback
            max_tokens: Maximum tokens to generate per response
            max_frames_in_video: Max frames per video (unused in sequence mode)

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        if len(contexts) == 1:
            return [self._process_single_isolated_context(contexts[0], max_tokens)]

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if not is_sequence_mode:
            # Video mode: batching doesn't work due to media embedding issues
            # Fall back to sequential processing
            print(f"[LongVILA-Isolated] Video mode batching not supported, using sequential")
            return [self._process_single_isolated_context(ctx, max_tokens) for ctx in contexts]

        # Sequence mode: text-only batching (proven to work)
        # TODO: extract this prompt-build/tokenize/system-prompt block — duplicated in
        # _process_single_isolated_context below.
        batch_texts = []
        for ctx in contexts:
            # Build prompt with main sequence, candidate, and question
            prompt_text = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )

            # Add system prompt
            if self.system_prompt:
                prompt_text = f"{self.system_prompt}\n\n{prompt_text}"

            conversation = [{"from": "human", "value": prompt_text}]
            input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True)
            batch_texts.append(input_ids)

        # Pad to same length (left padding)
        max_len = max(ids.shape[0] for ids in batch_texts)
        padded_ids = []
        for ids in batch_texts:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                padded = torch.cat([padding, ids])
            else:
                padded = ids
            padded_ids.append(padded)

        input_ids_batch = torch.stack(padded_ids).to(self.device)

        # Create attention mask
        attention_mask = torch.zeros_like(input_ids_batch)
        for i, ids in enumerate(batch_texts):
            pad_len = max_len - ids.shape[0]
            attention_mask[i, pad_len:] = 1

        # Batched inference with OOM retry
        batch_size = len(contexts)
        all_responses: List[Optional[str]] = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_ids = input_ids_batch[chunk_start:chunk_end]
                    chunk_mask = attention_mask[chunk_start:chunk_end]

                    # Generate (text-only, no media)
                    # NOTE: LongVILA's generate() returns input_ids + output_ids concatenated (line 1162)
                    # unless return_output_ids_only=True is set
                    output_ids = self.model.generate(
                        input_ids=chunk_ids,
                        attention_mask=chunk_mask,
                        media_config=defaultdict(dict),
                        max_new_tokens=max(1, int(max_tokens)),
                        return_output_ids_only=True,  # Get only generated tokens
                    )

                    # Decode only the generated portion
                    chunk_responses = self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, resp in enumerate(chunk_responses):
                        all_responses[chunk_start + i] = resp.strip()

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[LongVILA-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== LongVILA ({len(all_responses)} questions, mode={mode}) =====")
        for i, resp in enumerate(all_responses[:3]):
            print(f"Q{i+1}: {resp[:100] if resp else '(empty)'}...")
        if len(all_responses) > 3:
            print(f"... and {len(all_responses)-3} more")
        print("[model-output-batch-isolated] ===== END =====", flush=True)

        return all_responses

    def _process_single_isolated_context(
        self,
        ctx: Dict[str, Any],
        max_tokens: int,
    ) -> str:
        """Process a single isolated context (sequence mode only)."""
        mode = ctx.get('mode', 'sequence')

        if mode != 'sequence':
            # Video mode not supported
            raise NotImplementedError("Video mode isolated batching not supported for LongVILA")

        # Build prompt
        prompt_text = (
            f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
            f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
            f"{ctx['question_text']}"
        )

        if self.system_prompt:
            prompt_text = f"{self.system_prompt}\n\n{prompt_text}"

        conversation = [{"from": "human", "value": prompt_text}]
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True)
        input_ids = input_ids.unsqueeze(0).to(self.device)

        # Generate (get only new tokens, not input)
        output_ids = self.model.generate(
            input_ids=input_ids,
            media_config=defaultdict(dict),
            max_new_tokens=max(1, int(max_tokens)),
            return_output_ids_only=True,
        )

        response = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return response

    # ------------------------------------------------------------------
    # Helper routines
    # ------------------------------------------------------------------
    def _build_timeline(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                events.append(
                    {
                        "type": "video",
                        "time": segment["time_start"],
                        "video_id": video_id,
                        "segment": segment,
                    }
                )
        for text, timestamp in self.text_entries:
            events.append({"type": "text", "time": timestamp, "text": text})
        events.sort(key=lambda item: (item["time"], item["type"] != "video"))
        return events

    def _trim_frames(self, frames: torch.Tensor, budget: int) -> torch.Tensor:
        if not budget or budget <= 0 or frames.shape[0] <= budget:
            return frames
        indices = np.linspace(0, frames.shape[0] - 1, budget, dtype=np.int64)
        return frames[indices]

    def _build_prompt_and_media(
        self,
        timeline: List[Dict[str, Any]],
        question: str,
        max_frames_in_video: int,
    ) -> Tuple[str, List[torch.Tensor]]:
        combined_frames: Dict[int, torch.Tensor] = {}
        for video_id, segments in self.video_segments.items():
            frame_chunks = [segment["frames"] for segment in segments if segment["frames"].shape[0] > 0]
            if frame_chunks:
                combined_frames[video_id] = torch.cat(frame_chunks, dim=0)

        prompt_lines: List[str] = []
        media_videos: List[torch.Tensor] = []
        video_token_map: Dict[int, int] = {}
        self._video_was_truncated = False  # Reset flag

        for event in timeline:
            if event["type"] == "video":
                video_id = event["video_id"]
                if video_id in video_token_map or video_id not in combined_frames:
                    continue
                original_frames = combined_frames[video_id]
                frames = self._trim_frames(original_frames, max_frames_in_video)
                if frames.shape[0] < original_frames.shape[0]:
                    self._video_was_truncated = True  # Track truncation
                processed = self._preprocess_video_tensor(frames)
                processed = self._select_tsp_safe_frames(processed)
                media_videos.append(processed)
                token_index = len(media_videos)
                video_token_map[video_id] = token_index
                prompt_lines.append(f"{VIDEO_TOKEN}")
            else:
                if event["text"]:
                    prompt_lines.append(event["text"])

        if question:
            prompt_lines.append(question)

        prompt_body = "\n".join(line for line in prompt_lines if line).strip()
        if self.system_prompt:
            if prompt_body:
                prompt_body = f"{self.system_prompt}\n\n{prompt_body}"
            else:
                prompt_body = self.system_prompt

        return prompt_body, media_videos

    def _preprocess_video_tensor(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] not in (3, 4):
            raise ValueError(f"Expected video tensor of shape (T, 3, H, W); got {frames.shape}")

        frames_uint8 = frames[:, :3].to(dtype=torch.uint8, device="cpu")
        pil_frames = [
            Image.fromarray(frame.permute(1, 2, 0).numpy(), mode="RGB")
            for frame in frames_uint8
        ]

        enable_dynamic = (
            getattr(self.model.config, "image_aspect_ratio", "") == "dynamic"
            and getattr(self.model.config, "video_max_tiles", 1) > 1
        )

        processed = process_images(
            pil_frames,
            self.image_processor,
            self.model.config,
            enable_dynamic_res=enable_dynamic,
            max_tiles=getattr(self.model.config, "video_max_tiles", None),
        )
        return processed.to(device=self.device, dtype=self.model.dtype)

    @staticmethod
    def _ensure_tensor(video_frames: Any) -> torch.Tensor:
        if isinstance(video_frames, torch.Tensor):
            if video_frames.ndim != 4:
                raise ValueError(f"LongVILA expects frame tensor rank 4, got {video_frames.shape}")
            return video_frames.detach().cpu()
        raise TypeError("LongVILA requires torch.Tensor inputs from the frame sampler")

    def _clone_segments(self, segments: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        return {video_id: [self._clone_segment(segment) for segment in entries] for video_id, entries in segments.items()}

    @staticmethod
    def _clone_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
        clone = dict(segment)
        frames = segment.get("frames")
        if isinstance(frames, torch.Tensor):
            clone["frames"] = frames.clone()
        return clone

    def _select_tsp_safe_frames(self, frames: torch.Tensor) -> torch.Tensor:
        total = frames.shape[0]
        if total <= 0:
            return frames

        min_frames = 4
        if total < min_frames:
            last = frames[-1:].repeat(min_frames - total, 1, 1, 1)
            frames = torch.cat([frames, last], dim=0)
            total = frames.shape[0]

        pool = self._tsp_pool_size(total)
        remainder = total % pool
        if remainder != 0:
            padding = pool - remainder
            last = frames[-1:].repeat(padding, 1, 1, 1)
            frames = torch.cat([frames, last], dim=0)

        return frames

    def _tsp_pool_size(self, num_frames: int) -> int:
        # Use the configured TSP pool size from the model config instead of dynamic calculation
        # The video_encoder.pool_sizes config specifies the fixed temporal pool size
        video_encoder_config = getattr(self.model.config, "video_encoder", {})
        pool_sizes = video_encoder_config.get("pool_sizes", [[8, 1, 1]])
        # Extract the temporal pool size (first element of first pool_sizes entry)
        tsp_temporal_pool = pool_sizes[0][0] if pool_sizes and len(pool_sizes[0]) > 0 else 8
        return int(tsp_temporal_pool)

    def _get_state_memory_floats(self) -> float:
        total = 0.0
        for segments in self.video_segments.values():
            for segment in segments:
                frames = segment.get("frames")
                if torch.is_tensor(frames):
                    total += float(frames.numel())
        for text, _ in self.text_entries:
            total += float(self._count_text_tokens(text))
        return total

    def _count_text_tokens(self, text: str) -> int:
        text = text or ""
        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached

        tokenizer = self._tokenizer_ref
        encoding = tokenizer(
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

    def _estimate_ask_question_flops(
        self,
        *,
        vision_frames: int,
        lang_prompt_len: int,
        num_generated: int,
        do_backward: bool,
    ) -> float:
        # Read patch dimensions from the image processor when available; fall back
        # to LongVILA-R1-7B's documented 448x448 if the processor doesn't expose them.
        size = getattr(self.image_processor, "size", {}) or {}
        vision_height = int(size.get("height", 448))
        vision_width = int(size.get("width", 448))
        flops_breakdown = longvila_r1_7b_flops(
            vision_frames=vision_frames,
            vision_height=vision_height,
            vision_width=vision_width,
            lang_prompt_len=lang_prompt_len,
            num_generated=num_generated,
            do_backward=do_backward,
        )
        if isinstance(flops_breakdown, dict):
            return float(flops_breakdown.get("total_flops", 0.0))
        return float(flops_breakdown)
