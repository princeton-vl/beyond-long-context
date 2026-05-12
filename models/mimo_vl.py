"""
MiMo-VL-7B streaming wrapper compatible with the shared benchmarking interface.

This module mirrors the Qwen full-video integration but targets the
XiaomiMiMo/MiMo-VL-7B-RL checkpoint directly so it can participate in the
evaluation harness without bespoke sampling logic.
"""

from typing import Any, Dict, Optional, Union, List
import numpy as np
from models.base_interface import VideoLanguageModelInterface, PerformanceMetrics
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from models.device_map_utils import build_max_memory_map
import copy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.flops_calc import mimo_vl_7b_flops

import time

from utils.paths import get_model_cache_dir


CACHE_ROOT = get_model_cache_dir()
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)


# ROUND2-v5 forced-think prefix. Manually appended to the rendered chat prompt
# so MiMo-VL-RL keeps reasoning instead of emitting </think> as its first token.
# `_THINK_PREFIX_OUT` is re-prepended to decoded outputs so downstream
# <think>...</think> parsing stays unchanged.
_THINK_PREFIX_IN = "<|im_start|>assistant\n<think>\nLet me think step by step. "
_THINK_PREFIX_OUT = "<think>\nLet me think step by step. "


class MimoVLVideo(VideoLanguageModelInterface):
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
        """Initialize MiMo-VL with proper base class initialization."""
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

        max_memory = None
        if max_gpu_mem is not None and torch.cuda.is_available():
            max_memory = build_max_memory_map(max_gpu_mem)

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "XiaomiMiMo/MiMo-VL-7B-RL",
            **model_kwargs,
        )

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs["cache_dir"] = CACHE_ROOT
        self.processor = AutoProcessor.from_pretrained(
            "XiaomiMiMo/MiMo-VL-7B-RL",
            **processor_kwargs,
        )
        # Fix padding side for decoder-only architecture
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

            
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

        # Clone tensor to avoid memory sharing issues
        video_frames = video_frames.clone()

        # Move tensor to model device to avoid device mismatch
        # Use input embeddings device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        video_frames = video_frames.to(input_device)
        
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
        
        self.text_entries.append((text, self.latest_time))
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                0.0,
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
        
        # Build message content based on timeline
        content = []
        video_inputs = []
        # Group segments by video_id to combine into single videos
        video_data = {}  # video_id -> {'frames': tensor, 'total_duration': float, 'total_frames': int}
        video_fps = []
        
        # First pass: collect all frames for each video
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']

                if video_id not in video_data:
                    video_data[video_id] = {
                        'frames': None,
                        'total_duration': 0.0,
                        'total_frames': 0,
                    }

                # Add all frames (we'll sample later)
                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'] = torch.cat(
                        [video_data[video_id]['frames'], segment['frames']], dim=0
                    )

                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(
                    segment.get('num_frames', segment['frames'].shape[0])
                )

        # Second pass: apply max_frames_in_video limit to each complete video
        self._video_was_truncated = False  # Reset flag
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames.shape[0] > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:  # TIME or SEGMENT (same logic)
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)

                data['frames'] = frames[indices]

            trimmed_frame_count = data['frames'].shape[0]
            original_frame_count = data['total_frames']
            original_duration = data['total_duration']
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data['total_duration'] = trimmed_frame_count / original_fps

        # Process timeline events for content
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                # Only add video to content once per video_id
                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    if frames_tensor is None or frames_tensor.shape[0] == 0:
                        continue

                    video_inputs.append(frames_tensor)

                    # All videos use fps=1.0
                    fps = 1.0
                    video_fps.append(fps)

                    content.append({"type": "video", "video": len(video_inputs) - 1})
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                content.append({"type": "text", "text": event['text']})

        # Add the question at the end
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        # ROUND2-v5 (task-neutral forced-think): v4 confirmed that forcing a
        # partial reasoning sentence INTO the prompt makes the model continue
        # reasoning. v5 uses a short task-neutral prefix ("Let me think step
        # by step.") to avoid leaking task-specific vocabulary into the
        # forced context.
        text_without_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = text_without_prompt + _THINK_PREFIX_IN
        # Process inputs
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            coalesced_fps = self._coalesce_video_fps(video_fps)
            if coalesced_fps is not None:
                processor_kwargs["fps"] = coalesced_fps
        inputs = self.processor(**processor_kwargs)
        # Move to model's input embedding device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max(1, int(max_tokens)))
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # ROUND2-v5: prepend the think-prefix we forced into the prompt back
        # onto the decoded output so downstream <think>...</think> parsing
        # works as expected.
        output_text = [_THINK_PREFIX_OUT + t for t in output_text]
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if video_inputs:
                total_frames = sum(int(v.shape[0]) for v in video_inputs)
                vision_height = int(video_inputs[0].shape[2])
                vision_width = int(video_inputs[0].shape[3])
            else:
                total_frames = 0
                vision_height = 0
                vision_width = 0

            input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
            prompt_length = int(input_ids.shape[1]) if input_ids is not None else 0
            generated_tokens = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

            flops_breakdown = mimo_vl_7b_flops(
                vision_frames=total_frames,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=prompt_length,
                num_generated=generated_tokens,
                do_backward=False,
            )
            flops_value = (
                float(flops_breakdown.get("total_flops", 0.0))
                if isinstance(flops_breakdown, dict)
                else float(flops_breakdown)
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
        sample_method: str = "TIME",
    ) -> List[str]:
        """
        Ask multiple questions in a batched inference call.
        All questions share the same video/text context.
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        # Build timeline (shared across all questions)
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

        # Collect video data
        video_data = {}
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']
                if video_id not in video_data:
                    video_data[video_id] = {'frames': None, 'total_duration': 0.0, 'total_frames': 0}
                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'] = torch.cat(
                        [video_data[video_id]['frames'], segment['frames']], dim=0
                    )
                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(segment.get('num_frames', segment['frames'].shape[0]))

        # Apply max_frames_in_video limit
        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames.shape[0] > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)
                data['frames'] = frames[indices]

        # Build shared content
        shared_content = []
        video_inputs = []
        video_fps = []
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames is not None and frames.shape[0] > 0:
                        video_inputs.append(frames)
                        duration = video_data[video_id]['total_duration']
                        fps_value = frames.shape[0] / duration if duration > 0 else None
                        video_fps.append(fps_value)
                        shared_content.append({'type': 'video', 'video': len(video_inputs) - 1})
                        processed_video_ids.add(video_id)
            elif event['type'] == 'text':
                shared_content.append({'type': 'text', 'text': event['text']})

        # Build batch messages
        batch_messages = []
        for question_text in questions:
            content = list(shared_content)
            content.append({'type': 'text', 'text': question_text})
            batch_messages.append({'role': 'user', 'content': content})

        # Apply chat template.
        # ROUND2-v5 (batched): mirror the single-question fix. Render with
        # add_generation_prompt=False, then manually suffix the assistant/think
        # prefix so MiMo-VL-RL does not emit </think> as its first token.
        # We re-prepend the same string onto decoded outputs below so the
        # downstream <think>...</think> parser keeps working.
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=False) + _THINK_PREFIX_IN
            for msg in batch_messages
        ]

        # Get model device
        input_device = self.model.get_input_embeddings().weight.device

        # Batched inference with OOM retry
        batch_size = len(questions)
        all_output_texts = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    processor_kwargs = {
                        'text': chunk_texts,
                        'padding': True,
                        'return_tensors': 'pt',
                    }
                    if video_inputs:
                        # Qwen2.5-VL (MimoVL) processor expects nested lists for batched videos
                        chunk_size = len(chunk_texts)
                        processor_kwargs['videos'] = [video_inputs for _ in range(chunk_size)]
                        coalesced_fps = self._coalesce_video_fps(video_fps)
                        if coalesced_fps is not None:
                            processor_kwargs['fps'] = coalesced_fps

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(input_device)

                    generated_ids = self.model.generate(**inputs, max_new_tokens=max(1, int(max_tokens)))
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    # ROUND2-v5 (batched): prepend the forced think-prefix back
                    # onto each decoded output so parsing sees a well-formed
                    # <think>...</think> block.
                    chunk_outputs = [_THINK_PREFIX_OUT + t for t in chunk_outputs]

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
                    print(f"[MimoVL] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch] ===== BATCHED RESPONSE MimoVL ({len(all_output_texts)} questions, batch_size={batch_size}) =====")
        for i, text in enumerate(all_output_texts[:3]):
            print(f"Q{i+1}: {text}")
        if len(all_output_texts) > 3:
            print(f"... and {len(all_output_texts)-3} more")
        print("[model-output-batch] ===== END =====", flush=True)

        return all_output_texts

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
        # That path has different FPS/content logic from the batch path.

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build separate messages for each question
        batch_messages = []
        all_videos_per_question = []

        for i, ctx in enumerate(contexts):
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

                # Add question text
                content.append({"type": "text", "text": ctx['question_text']})

                message = {"role": "user", "content": content}
                batch_messages.append(message)
                all_videos_per_question.append(videos_for_question)

        # Apply chat template to each message separately.
        # ROUND2-v5 (batched-isolated): same forced-think prefix trick as the
        # single-question path. Render with add_generation_prompt=False and
        # manually append the assistant/think opener; re-prepend the out
        # prefix onto decoded outputs below so parsing is unchanged.
        batch_texts = []
        for idx, msg in enumerate(batch_messages):
            text = self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
            batch_texts.append(text + _THINK_PREFIX_IN)

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

                    # Handle video inputs
                    if not is_sequence_mode:
                        chunk_videos = all_videos_per_question[chunk_start:chunk_end]
                        # Flatten videos: [[main1, cand1], [main2, cand2]] -> [main1, cand1, main2, cand2]
                        all_videos = []
                        for video_list in chunk_videos:
                            all_videos.extend(video_list)

                        if all_videos:
                            processor_kwargs["videos"] = all_videos
                            # Batched-isolated path: flat all_videos shape, fixed at fps=1.0.
                            processor_kwargs["fps"] = 1.0

                    # Tokenize
                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(input_device) if hasattr(v, 'to') else v
                             for k, v in inputs.items()}

                    # Generate
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max(1, int(max_tokens))
                        )

                    # Decode
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]
                    output_texts = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    # ROUND2-v5 (batched-isolated): prepend forced think-prefix
                    # back onto decoded outputs.
                    output_texts = [_THINK_PREFIX_OUT + t for t in output_texts]

                    # Store results
                    for i, text in enumerate(output_texts):
                        all_output_texts[chunk_start + i] = text

                # Success - break out of retry loop
                break

            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = batch_size // 2
                print(f"[OOM] Retrying with batch_size={batch_size}")
                torch.cuda.empty_cache()

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE MimoVL ({len(all_output_texts)} questions, batch_size={batch_size}, mode={mode}) =====")
        for i, text in enumerate(all_output_texts[:4]):
            preview = text[:150] if text else ""
            print(f"Q{i+1}: {preview}...")
        if len(all_output_texts) > 4:
            print(f"... and {len(all_output_texts)-4} more")
        print("[model-output-batch-isolated] ===== END =====", flush=True)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, peak_mem - baseline_mem) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            # Wire through the standard metrics helper for parity with single-question path.
            # FLOPs / video timestamp are not tracked in batched-isolated (per-context FLOPs
            # would require recomputing for each ctx; surface as 0.0 placeholders).
            self._record_ask_question_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                0.0,
                None,
            )

        return all_output_texts

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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Note: Don't reset metrics here - they should accumulate across questions

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
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
    
