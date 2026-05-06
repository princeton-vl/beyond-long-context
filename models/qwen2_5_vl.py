"""
Generic interface for video-language models with benchmarking capabilities.
All models must implement this interface for consistent comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import numpy as np
from dataclasses import dataclass
from models.base_interface import VideoLanguageModelInterface, PerformanceMetrics
from enum import Enum
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import copy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics.flops_calc import qwen_2_5_vl_7b_flops



import time

from utils.paths import get_model_cache_dir


CACHE_ROOT = get_model_cache_dir()
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)

    
class QwenFullVideo(VideoLanguageModelInterface):
    """
    Generic interface for video-language models.
    
    This interface defines the core methods that all video-language models
    must implement for consistent benchmarking and evaluation.
    """
    
    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        """Initialize QwenFullVideo with proper base class initialization."""
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )
    
    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Model-specific initialization logic.
        
        Args:
            **kwargs: Model-specific parameters. Supported keys:
                - min_pixels (Optional[int])
                - max_pixels (Optional[int])
        """
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }

        # Legacy callers sometimes forwarded max_gpu_mem; the argument is now
        # handled explicitly in the base class, so nothing further to do here.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            **model_kwargs,
        )

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs["cache_dir"] = CACHE_ROOT
        processor_kwargs["padding_side"] = "left"  # Required for decoder-only batched generation
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            **processor_kwargs,
        )
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()

    # ------------------------------------------------------------------
    # Hooks for subclasses (e.g. Qwen3 VL) to express their processor deltas
    # without duplicating the entire ask_question / ask_question_batch body.
    # ------------------------------------------------------------------

    def _build_video_payload(self, video_id, data):
        """Return (frames_for_processor, metadata_or_None, fps_or_None) for a single video.

        Subclasses override to attach a VideoMetadata object and/or move the
        frames tensor onto the device the processor expects. Default impl
        targets Qwen2.5 VL: ship frames on CPU, no metadata, fps=1.0.
        """
        frames = data.get('frames')
        if frames is None:
            return None, None, None
        if torch.is_tensor(frames) and frames.device.type != "cpu":
            frames = frames.to(device='cpu')
        # All videos in this pipeline use fps=1.0 for Qwen2.5
        return frames, None, 1.0

    def _apply_processor_video_kwargs(
        self,
        processor_kwargs,
        video_inputs,
        video_metadata,
        video_fps,
        *,
        batched: bool,
        batch_size: int = 1,
    ):
        """Mutate processor_kwargs in place to add videos / video_metadata / fps.

        Default behaviour matches Qwen2.5 VL:
            - flat list when ``batched=False`` (single ask_question path)
            - list-of-lists (one inner list per text) when ``batched=True``,
              even for chunk_size=1 (the batched processor expects this shape)
            - video_metadata is omitted (Qwen2.5 doesn't require it)

        Subclasses (Qwen3 VL) override to also pass video_metadata in matching
        nested form.
        """
        if not video_inputs:
            return

        if not batched:
            processor_kwargs["videos"] = video_inputs
        else:
            processor_kwargs["videos"] = [video_inputs for _ in range(batch_size)]

        coalesced_fps = self._coalesce_video_fps(video_fps)
        if coalesced_fps is not None:
            processor_kwargs["fps"] = coalesced_fps

    def _debug_print_prompts(self, *texts):
        """Dump the rendered chat prompt(s) when QWEN_DEBUG_PROMPTS is set."""
        if not os.environ.get("QWEN_DEBUG_PROMPTS"):
            return
        for text in texts:
            print("\n[model-input] ===== RAW CHAT PROMPT START =====")
            print(text)
            print("[model-input] ===== RAW CHAT PROMPT END =====", flush=True)

    def _debug_print_response(self, output_texts, batch_size=None, label="Qwen"):
        """Dump generated response(s) when QWEN_DEBUG_PROMPTS is set."""
        if not os.environ.get("QWEN_DEBUG_PROMPTS"):
            return
        if batch_size is None:
            print(f"\n[model-output] ===== RAW MODEL RESPONSE START ({label}) =====")
            print(output_texts[0] if output_texts else "")
            print(f"[model-output] ===== RAW MODEL RESPONSE END ({label}) =====", flush=True)
            return
        print(
            f"\n[model-output-batch] ===== BATCHED RESPONSE {label} "
            f"({len(output_texts)} questions, batch_size={batch_size}) ====="
        )
        for i, text in enumerate(output_texts[:3]):
            preview = text[:100] if text else "(empty)"
            print(f"Q{i+1}: {preview}")
        if len(output_texts) > 3:
            print(f"... and {len(output_texts)-3} more")
        print("[model-output-batch] ===== END =====", flush=True)

    def _collect_and_trim_videos(self, timeline_events, max_frames_in_video, sample_method):
        """Aggregate per-video frames across segments and apply max_frames_in_video.

        Returns a dict keyed by video_id with 'frames', 'total_duration',
        'total_frames', and 'frame_indices' (the post-trim selection over the
        original concatenated frames).
        """
        video_data = {}
        for event in timeline_events:
            if event['type'] != 'video':
                continue
            video_id = event['video_id']
            segment = event['segment']

            store = video_data.setdefault(
                video_id,
                {
                    'frames': None,
                    'total_duration': 0.0,
                    'total_frames': 0,
                },
            )

            segment_frames = segment['frames']
            if store['frames'] is None:
                store['frames'] = segment_frames
            else:
                # Ensure both tensors are on the same device before concatenation
                if (
                    isinstance(segment_frames, torch.Tensor)
                    and isinstance(store['frames'], torch.Tensor)
                    and segment_frames.device != store['frames'].device
                ):
                    segment_frames = segment_frames.to(store['frames'].device)
                store['frames'] = torch.cat([store['frames'], segment_frames], dim=0)

            store['total_duration'] = float(store['total_duration']) + float(segment['duration'])
            store['total_frames'] = int(store['total_frames']) + int(
                segment.get('num_frames', segment['frames'].shape[0])
            )

        for video_id, data in video_data.items():
            frames = data['frames']
            if frames is None:
                data['frame_indices'] = []
                continue

            frame_count = int(frames.shape[0])
            selection = np.arange(frame_count, dtype=int)
            if frame_count > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.sort(
                        np.random.choice(frame_count, max_frames_in_video, replace=False)
                    )
                else:  # TIME or SEGMENT
                    indices = np.linspace(0, frame_count - 1, max_frames_in_video, dtype=int)
                data['frames'] = frames[indices]
                selection = selection[indices]

            trimmed_frame_count = int(data['frames'].shape[0])
            original_frame_count = int(data['total_frames'])
            original_duration = float(data['total_duration'])
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data['total_duration'] = trimmed_frame_count / original_fps

            data['frame_indices'] = selection.astype(int).tolist()

        return video_data

    def add_video(self, video_frames: np.ndarray, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """
        Add video frames to the model's context.
        
        This method should mutate the internal context state and return nothing.
        The video frames should be integrated into the model's understanding.
        
        Args:
            video_frames: Video frames as numpy array with shape (num_frames, 3, height, width)
            time_start: The time the video frames starts. Must be after last was added
            time_end: The time the video frames end. Must be after last were added
            video_id: Optional identifier for the video (defaults to 0 if not provided)
        """
        # Validate time ordering
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")
        
        self.latest_time = time_end
        
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.from_numpy(video_frames)

        # Clone tensor to avoid memory sharing issues and keep a CPU copy
        video_frames = video_frames.clone()
        if video_frames.device.type != "cpu":
            video_frames = video_frames.cpu()
        
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Default video_id to 0 if not provided
        if video_id is None:
            video_id = 0
        
        if video_id not in self.video_segments:
            self.video_segments[video_id] = []
        
        segment = {
            'frames': video_frames,
            'time_start': time_start,
            'time_end': time_end,
            'duration': time_end - time_start,
            'num_frames': video_frames.shape[0]
        }
        
        self.video_segments[video_id].append(segment)
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if torch.is_tensor(video_frames) and video_frames.ndim == 4:
                vision_frames = int(video_frames.shape[0])
                vision_height = int(video_frames.shape[2])
                vision_width = int(video_frames.shape[3])
            else:
                vision_frames = 0
                vision_height = 0
                vision_width = 0

            add_video_flops = 0.0

            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                add_video_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )
    
    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        """
        Add text to the model's context.

        This method should mutate the internal context state and return nothing.
        The text should be integrated into the model's understanding.

        Args:
            text: Text string to add to context
            current_video_time: Current timestamp in video (seconds from start)
        """
        # Add text with current timestamp (text goes first if same time as video)
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Anchor text to its passed-in video timestamp (matches docstring + sibling
        # wrappers in glm45v / longvila). Previously stored self.latest_time, which
        # silently collapsed all text inserts to the most recent add_video time_end.
        self.text_entries.append((text, float(current_video_time)))
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            token_count = self._count_text_tokens(text)
            add_text_flops = 0.0

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                add_text_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )
    
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """
        Ask a question based on the current context.

        This method should generate a response based on all previously added
        video and text content without modifying the context.

        Args:
            question: Question to ask
            current_video_time: Current timestamp in video when question is asked (seconds from start)
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")

        Returns:
            Generated response as string
        """
        self._video_was_truncated = False  # Reset before any potential trimming

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Create timeline of events (videos and text)
        timeline_events = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append({
                    'type': 'video',
                    'time': segment['time_start'],
                    'video_id': video_id,
                    'segment': segment
                })

        for text, timestamp in self.text_entries:
            timeline_events.append({
                'type': 'text',
                'time': timestamp,
                'text': text
            })

        timeline_events.sort(key=lambda x: (x['time'], x['type'] == 'video'))

        # Aggregate per-video frames + apply max_frames_in_video
        video_data = self._collect_and_trim_videos(
            timeline_events, max_frames_in_video, sample_method
        )

        # Build message content + processor video payload via subclass-friendly hook
        content = []
        video_inputs = []
        video_metadata = []
        video_fps = []
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id in processed_video_ids:
                    continue

                frames_for_processor, metadata, fps = self._build_video_payload(
                    video_id, video_data[video_id]
                )
                if frames_for_processor is None or (
                    torch.is_tensor(frames_for_processor) and frames_for_processor.shape[0] == 0
                ):
                    processed_video_ids.add(video_id)
                    continue

                video_inputs.append(frames_for_processor)
                if metadata is not None:
                    video_metadata.append(metadata)
                video_fps.append(fps)

                content.append({"type": "video", "video": len(video_inputs) - 1})
                processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                content.append({"type": "text", "text": event['text']})

        # Add the question at the end
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self._debug_print_prompts(text)

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        self._apply_processor_video_kwargs(
            processor_kwargs,
            video_inputs,
            video_metadata,
            video_fps,
            batched=False,
        )

        inputs = self.processor(**processor_kwargs)
        # Move to model's input embedding device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self._debug_print_response(output_text, batch_size=None, label=type(self).__name__)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # Calculate FLOPs for the question answering
            if video_inputs:
                total_frames = sum([v.shape[0] for v in video_inputs])
                vision_height = video_inputs[0].shape[2]
                vision_width = video_inputs[0].shape[3]
            else:
                total_frames = 0
                vision_height = 0
                vision_width = 0

            flops_value = self._estimate_ask_question_flops(
                vision_frames=total_frames,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=int(inputs.input_ids.shape[1]),
                num_generated=len(generated_ids_trimmed[0]),
                do_backward=False
            )
            latest_context_time = 0.0
            for event in timeline_events:
                if event['type'] == 'video':
                    latest_context_time = max(latest_context_time, float(event['segment']['time_end']))
                else:
                    latest_context_time = max(latest_context_time, float(event['time']))
            question_timestamp = float(current_video_time) if current_video_time is not None else 0.0
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time
            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                flops_value,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
            )
        
        return output_text[0]

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME"
    ) -> List[str]:
        """
        Ask multiple questions in a batched inference call for improved performance.

        All questions share the same video/text context (timeline).
        """
        if not questions:
            return []

        if len(questions) == 1:
            # Single question - use regular path
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False  # Reset before any potential trimming

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build timeline events (shared across all questions)
        timeline_events = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append({
                    'type': 'video',
                    'time': segment['time_start'],
                    'video_id': video_id,
                    'segment': segment
                })

        for text, timestamp in self.text_entries:
            timeline_events.append({
                'type': 'text',
                'time': timestamp,
                'text': text
            })

        timeline_events.sort(key=lambda x: (x['time'], x['type'] == 'video'))

        video_data = self._collect_and_trim_videos(
            timeline_events, max_frames_in_video, sample_method
        )

        # Build shared content + processor payload via subclass-friendly hook
        shared_content = []
        video_inputs = []
        video_metadata = []
        video_fps = []
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id in processed_video_ids:
                    continue

                frames_for_processor, metadata, fps = self._build_video_payload(
                    video_id, video_data[video_id]
                )
                if frames_for_processor is None or (
                    torch.is_tensor(frames_for_processor) and frames_for_processor.shape[0] == 0
                ):
                    processed_video_ids.add(video_id)
                    continue

                video_inputs.append(frames_for_processor)
                if metadata is not None:
                    video_metadata.append(metadata)
                video_fps.append(fps)

                shared_content.append({"type": "video", "video": len(video_inputs) - 1})
                processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                shared_content.append({"type": "text", "text": event['text']})

        # Build batch of messages (one per question)
        batch_messages = []
        for question_text in questions:
            content = list(shared_content)  # Copy shared content
            content.append({"type": "text", "text": question_text})
            batch_messages.append({"role": "user", "content": content})

        # Apply chat template to all questions
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Get model device
        input_device = self.model.get_input_embeddings().weight.device

        # Batched inference with OOM retry (halve batch size on OOM)
        batch_size = len(questions)
        all_output_texts = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }
                    self._apply_processor_video_kwargs(
                        processor_kwargs,
                        video_inputs,
                        video_metadata,
                        video_fps,
                        batched=True,
                        batch_size=len(chunk_texts),
                    )

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(input_device)

                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                # Success - break out of retry loop
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    # Clear CUDA cache and halve batch size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[QwenFullVideo] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Reset results for retry
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        output_texts = all_output_texts

        self._debug_print_response(output_texts, batch_size=batch_size, label=type(self).__name__)

        if self.enable_metrics:
            # Aggregate metrics for the batch. We record per-call latency / peak
            # GPU memory once (not per question) because the batched generate()
            # returns a single allocator peak; per-question FLOPs attribution is
            # ambiguous when the prompt sequence lengths differ. We record the
            # aggregate using the LATEST context timestamp so curves still get
            # a sample point.
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latest_context_time = 0.0
            for event in timeline_events:
                if event['type'] == 'video':
                    latest_context_time = max(latest_context_time, float(event['segment']['time_end']))
                else:
                    latest_context_time = max(latest_context_time, float(event['time']))
            question_timestamp = float(current_video_time) if current_video_time else 0.0
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time

            state_mem = self._get_state_memory_floats()
            # FLOPs left at 0 for the aggregate batch entry (per-question
            # accounting would double-count vision). Per-question FLOPs are
            # captured in the single-question ask_question path.
            self._record_ask_question_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
            )

        return output_texts

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
                     For video mode: {'main_video_frames': tensor, 'candidate_video_frames': tensor,
                                     'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate
            max_frames_in_video: Max frames per video

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # NOTE: Do NOT shortcut to _process_single_context_isolated for batch_size=1.
        # That path uses add_video()+ask_question() which has different FPS/content logic.
        # Always use the batch code path for consistency.

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build separate messages for each question
        batch_messages = []
        all_videos_per_question = []

        for ctx in contexts:
            if is_sequence_mode:
                # Sequence mode: pure text
                prompt = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                content = [{"type": "text", "text": prompt}]
                message = {"role": "user", "content": content}
                batch_messages.append(message)
                all_videos_per_question.append([])

            else:
                # Video mode: build separate video list for this question
                main_frames = ctx.get('main_video_frames')
                candidate_frames = ctx.get('candidate_video_frames')

                videos_for_question = []
                content = []

                # Add main video
                if main_frames is not None and main_frames.shape[0] > 0:
                    # Sample to half of budget
                    if main_frames.shape[0] > max_frames_in_video // 2:
                        indices = np.linspace(0, main_frames.shape[0] - 1, max_frames_in_video // 2, dtype=int)
                        sampled_main = main_frames[indices]
                    else:
                        sampled_main = main_frames

                    sampled_main = sampled_main.to(device='cpu')
                    videos_for_question.append(sampled_main)
                    # Add text anchor BEFORE video (matches non-batched)
                    content.append({"type": "text", "text": "Here is a main video to remember:"})
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                # Add candidate clip - MODIFIED TO MATCH NON-BATCHED MODE
                if candidate_frames is not None and candidate_frames.shape[0] > 0:
                    # Add text anchor BEFORE video (matches non-batched)
                    content.append({"type": "text", "text": "\nHere is a candidate clip:\n"})
                    # Sample to remaining budget
                    total_main = sampled_main.shape[0] if main_frames is not None else 0
                    remaining = max_frames_in_video - total_main
                    if candidate_frames.shape[0] > remaining:
                        indices = np.linspace(0, candidate_frames.shape[0] - 1, max(1, remaining), dtype=int)
                        sampled_cand = candidate_frames[indices]
                    else:
                        sampled_cand = candidate_frames

                    sampled_cand = sampled_cand.to(device='cpu')
                    videos_for_question.append(sampled_cand)
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                content.append({"type": "text", "text": ctx['question_text']})
                message = {"role": "user", "content": content}
                batch_messages.append(message)
                all_videos_per_question.append(videos_for_question)

        # Apply chat template
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Get device
        input_device = self.model.get_input_embeddings().weight.device

        # Batch process with OOM retry
        batch_size = len(contexts)
        all_output_texts = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Prepare processor kwargs
                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }

                    if not is_sequence_mode:
                        # Video mode: get video lists for this chunk
                        chunk_videos = all_videos_per_question[chunk_start:chunk_end]
                        # Filter out empty lists
                        valid_videos = [v for v in chunk_videos if v]

                        if valid_videos:
                            processor_kwargs["videos"] = valid_videos
                            # Add fps to avoid warning about missing video metadata
                            processor_kwargs["fps"] = 1.0

                            # Build video_metadata for Qwen3 compatibility (optional for Qwen2.5)
                            try:
                                from transformers.video_utils import VideoMetadata
                                video_metadata_per_question = []
                                for videos_list in valid_videos:
                                    metadata_for_question = []
                                    for video_tensor in videos_list:
                                        if torch.is_tensor(video_tensor):
                                            metadata = VideoMetadata(
                                                total_num_frames=int(video_tensor.shape[0]),
                                                fps=1.0,
                                                width=int(video_tensor.shape[3]),
                                                height=int(video_tensor.shape[2]),
                                                duration=float(video_tensor.shape[0] / 1.0),
                                                video_backend="tensor",
                                                frames_indices=list(range(video_tensor.shape[0])),
                                            )
                                            metadata_for_question.append(metadata)
                                    video_metadata_per_question.append(metadata_for_question)

                                if video_metadata_per_question:
                                    processor_kwargs["video_metadata"] = video_metadata_per_question
                            except ImportError:
                                # VideoMetadata not available in older transformers, skip
                                pass

                    # Tokenize
                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(device=input_device, non_blocking=True)

                    # Generate
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max(1, int(max_tokens))
                        )

                    # Trim prompts
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]

                    # Decode
                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[QwenFull-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output (gated)
        if os.environ.get("QWEN_DEBUG_PROMPTS"):
            print(
                f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE QwenFull "
                f"({len(all_output_texts)} questions, batch_size={batch_size}, mode={mode}) ====="
            )
            for i, text in enumerate(all_output_texts[:3]):
                preview = text[:100] if text else "(empty)"
                print(f"Q{i+1}: {preview}...")
            if len(all_output_texts) > 3:
                print(f"... and {len(all_output_texts)-3} more")
            print("[model-output-batch-isolated] ===== END =====", flush=True)

        return all_output_texts

    # Note: _process_single_context_isolated was removed. ask_question_batch_isolated
    # always uses its batch code path even for batch_size=1 (see comment above) so
    # the helper had no callers.

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
        """Return the FLOPs estimate for a question turn."""

        flops_breakdown = qwen_2_5_vl_7b_flops(
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
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the model.
        
        Returns:
            Dictionary containing current context information
        """
        # Create timeline of all events
        timeline = []
        
        # Add video segments to timeline
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline.append({
                    'type': 'video',
                    'timestamp': segment['time_start'],
                    'video_id': video_id,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                })
        
        # Add text entries to timeline
        for text, timestamp in self.text_entries:
            timeline.append({
                'type': 'text',
                'timestamp': timestamp,
                'text': text
            })
        
        # Sort timeline by timestamp (text first if same time)
        timeline.sort(key=lambda x: (x['timestamp'], x['type'] == 'video'))
        
        return {
            'video_segments': dict(self.video_segments),
            'text_entries': list(self.text_entries),
            'latest_time': self.latest_time,
            'timeline': timeline
        }

    
    def clear_context(self) -> None:
        """
        Clear all context (video and text) from the model.

        This should reset the model to its initial state.
        """
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None

        # Additional memory cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Note: Don't reset metrics here - they should accumulate across questions

    def was_video_truncated(self) -> Optional[bool]:
        """
        Check if video frames were truncated in the last ask_question call.

        Returns:
            True if frames were truncated due to max_frames_in_video limit,
            False if all frames were used,
            None if no question has been asked yet
        """
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current model state to memory.

        Returns a deep copy of all state so GPU memory can be freed while preserving
        the ability to restore the exact state later. Preserves metrics across save/load.
        """
        # Deep copy video segments to CPU memory
        saved_video_segments = {}
        for video_id, segments in self.video_segments.items():
            saved_segments = []
            for segment in segments:
                # Move tensor to CPU and detach
                frames_cpu = segment['frames'].detach().cpu().clone() if torch.is_tensor(segment['frames']) else segment['frames']
                saved_segment = {
                    'frames': frames_cpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                }
                saved_segments.append(saved_segment)
            saved_video_segments[video_id] = saved_segments

        state = {
            'video_segments': saved_video_segments,
            'text_entries': copy.deepcopy(self.text_entries),
            'latest_time': self.latest_time,
            'enable_metrics': self.enable_metrics,
            '_metrics': None
        }

        # Save metrics if enabled (preserve across save/load cycles)
        if self.enable_metrics and self._metrics is not None:
            metrics_dict = {
                'latency_add_video': copy.deepcopy(self._metrics.latency_add_video),
                'latency_add_text': copy.deepcopy(self._metrics.latency_add_text),
                'flops_add_video': copy.deepcopy(self._metrics.flops_add_video),
                'flops_add_text': copy.deepcopy(self._metrics.flops_add_text),
                'state_memory_floats': copy.deepcopy(self._metrics.state_memory_floats),
                'state_memory_after_add_video': copy.deepcopy(self._metrics.state_memory_after_add_video),
                'state_memory_after_add_text': copy.deepcopy(self._metrics.state_memory_after_add_text),
                'state_memory_delta_add_video': copy.deepcopy(self._metrics.state_memory_delta_add_video),
                'state_memory_delta_add_text': copy.deepcopy(self._metrics.state_memory_delta_add_text),
                'peak_gpu_mem_increase_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_video),
                'peak_gpu_mem_increase_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_text),
                'peak_gpu_mem_absolute_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_video),
                'peak_gpu_mem_absolute_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_text),
                'video_timestamps_add_video': copy.deepcopy(self._metrics.video_timestamps_add_video),
                'video_timestamps_add_text': copy.deepcopy(self._metrics.video_timestamps_add_text),
                'first_oom_timestamp': self._metrics.first_oom_timestamp,
            }
            state['_metrics'] = metrics_dict

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Restores video segments (moving them back to model device), text entries, timing,
        and preserves metrics across save/load cycles.
        """
        # Get the model's input device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device

        # Restore video segments and move tensors back to model device
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                # Move tensor back to model device
                if torch.is_tensor(segment['frames']):
                    frames_gpu = segment['frames'].to(input_device)
                else:
                    frames_gpu = segment['frames']

                restored_segment = {
                    'frames': frames_gpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                }
                restored_segments.append(restored_segment)
            self.video_segments[video_id] = restored_segments

        # Restore other state
        self.text_entries = state['text_entries'].copy()
        self.latest_time = state['latest_time']

        # Restore metrics if present and metrics are enabled
        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
            from models.base_interface import PerformanceMetrics

            # If we don't have metrics yet, create new object
            if self._metrics is None:
                self._metrics = PerformanceMetrics()

            preserved_fields = {
                'latency_ask_question',
                'flops_ask_question',
                'state_memory_after_ask_question',
                'state_memory_delta_ask_question',
                'peak_gpu_mem_increase_ask_question',
                'peak_gpu_mem_absolute_ask_question',
                'video_timestamps_ask_question',
                'question_correctness_rate',
                'question_dont_know_rate',
                'question_answered_mask',
                'video_timestamps_question_outcome',
            }

            preserved = {
                name: copy.deepcopy(getattr(self._metrics, name, []))
                for name in preserved_fields
            }

            metrics_data = state['_metrics']
            self._metrics.latency_add_video = metrics_data.get('latency_add_video', []).copy()
            self._metrics.latency_add_text = metrics_data.get('latency_add_text', []).copy()
            self._metrics.flops_add_video = metrics_data.get('flops_add_video', []).copy()
            self._metrics.flops_add_text = metrics_data.get('flops_add_text', []).copy()
            self._metrics.state_memory_floats = metrics_data.get('state_memory_floats', []).copy()
            self._metrics.state_memory_after_add_video = metrics_data.get('state_memory_after_add_video', []).copy()
            self._metrics.state_memory_after_add_text = metrics_data.get('state_memory_after_add_text', []).copy()
            self._metrics.state_memory_delta_add_video = metrics_data.get('state_memory_delta_add_video', []).copy()
            self._metrics.state_memory_delta_add_text = metrics_data.get('state_memory_delta_add_text', []).copy()
            self._metrics.peak_gpu_mem_increase_add_video = metrics_data.get('peak_gpu_mem_increase_add_video', []).copy()
            self._metrics.peak_gpu_mem_increase_add_text = metrics_data.get('peak_gpu_mem_increase_add_text', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_video = metrics_data.get('peak_gpu_mem_absolute_add_video', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_text = metrics_data.get('peak_gpu_mem_absolute_add_text', []).copy()
            self._metrics.video_timestamps_add_video = metrics_data.get('video_timestamps_add_video', []).copy()
            self._metrics.video_timestamps_add_text = metrics_data.get('video_timestamps_add_text', []).copy()
            self._metrics.first_oom_timestamp = metrics_data.get('first_oom_timestamp')

            for name, value in preserved.items():
                setattr(self._metrics, name, value)

            self._sync_state_memory_tracking_from_metrics()

    def _get_tokenizer(self):
        if self._tokenizer_ref is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor does not expose a tokenizer; cannot measure text state size.")
            self._tokenizer_ref = tokenizer
        return self._tokenizer_ref

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached

        tokenizer = self._get_tokenizer()
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

    def _get_state_memory_floats(self) -> float:
        """Calculates the total memory usage of the stored state."""
        total_floats = 0.0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                frames_tensor = segment.get('frames')
                if torch.is_tensor(frames_tensor):
                    total_floats += float(frames_tensor.numel())

        for text, _ in self.text_entries:
            total_floats += float(self._count_text_tokens(text))
        return total_floats
    
