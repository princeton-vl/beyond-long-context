# models/timechat.py
"""
TimeChatOnlineStreaming wrapper that conforms to VideoLanguageModelInterface.
Uses standard Qwen2VL model for video-language understanding.

- add_video: store raw frames + timestamps (CPU).
- ask_question: pack with AutoProcessor, then normalize tensor fields to CUDA.
- Frames are (T, 3, H, W) torch or numpy arrays.
"""

from typing import Any, Dict, Optional, Union, List, Tuple
import os
import sys
import time
import copy
import json
import numpy as np
import torch
from datetime import datetime
from dataclasses import fields

from transformers import AutoProcessor
from models.base_interface import VideoLanguageModelInterface, PerformanceMetrics
from metrics.flops_calc import timechat_online_flops
from models.device_map_utils import build_max_memory_map
from utils.paths import get_model_cache_dir

# Import the TimeChat demo model (has DTD methods)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_here, "..", "external")))
from TimeChat.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import transformers.image_utils as _hf_image_utils
import transformers.processing_utils as _hf_processing_utils

if not hasattr(_hf_image_utils, 'VideoInput'):
    _hf_image_utils.VideoInput = _hf_image_utils.ImageInput

if not hasattr(_hf_image_utils, 'make_batched_videos'):
    def _make_batched_videos(videos):
        return videos
    _hf_image_utils.make_batched_videos = _make_batched_videos

if not hasattr(_hf_processing_utils, 'MultiModalData'):
    class MultiModalData(dict):
        """Fallback for new Transformers multi-modal typing"""
        pass
    _hf_processing_utils.MultiModalData = MultiModalData



CACHE_ROOT = get_model_cache_dir()
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)


curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
DR_SAVE_PATH = f"drop_{curr_time}.jsonl"
DROP_MAX_VIDEOS = 1
reduce_videos = False


class TimeChatOnlineStreaming(VideoLanguageModelInterface):
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
            "wyccccc/TimeChatOnline-7B",
            **model_kwargs,
        )

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        # Required for batched generation with Qwen2.5-VL
        processor_kwargs["padding_side"] = "left"

        self.processor = AutoProcessor.from_pretrained(
            "wyccccc/TimeChatOnline-7B",
            cache_dir=CACHE_ROOT,
            trust_remote_code=True,
            **processor_kwargs,
        )
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        # Track token-drop logs for FLOP accounting
        self._drop_log_path = os.path.abspath(DR_SAVE_PATH)
        self._drop_log_line_count = 0
        self._last_drop_stats: Optional[Dict[str, Any]] = None

            
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

        # Clone tensor to avoid memory sharing issues and keep it on CPU in compact form
        video_frames = video_frames.clone().contiguous()
        if video_frames.dtype.is_floating_point:
            video_frames = torch.clamp(video_frames, 0.0, 1.0)
            video_frames = torch.round(video_frames * 255.0).to(torch.uint8)
        elif video_frames.dtype != torch.uint8:
            video_frames = video_frames.to(torch.uint8)
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

            add_video_flops = 0

            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                add_video_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_memory_total=state_mem,
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

            token_count = self._count_text_tokens(text)
            add_text_flops = 0.0

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                add_text_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_memory_total=state_mem,
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
        content: List[Dict[str, Any]] = []
        video_inputs: List[torch.Tensor] = []
        video_fps: List[float] = []

        video_data: Dict[Any, Dict[str, Any]] = {}

        for event in timeline_events:
            if event['type'] != 'video':
                continue

            video_id = event['video_id']
            segment = event['segment']

            if video_id not in video_data:
                video_data[video_id] = {'frames': None, 'total_duration': 0.0}

            if video_data[video_id]['frames'] is None:
                video_data[video_id]['frames'] = segment['frames']
            else:
                video_data[video_id]['frames'] = torch.cat(
                    [video_data[video_id]['frames'], segment['frames']],
                    dim=0,
                )

            video_data[video_id]['total_duration'] += segment['duration']

        # Apply max_frames_in_video limit to each video stream
        self._video_was_truncated = False  # Reset flag
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames is None:
                continue

            original_frame_count = frames.shape[0]
            if original_frame_count > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)

                data['frames'] = frames[indices]

                # Adjust duration proportionally to maintain consistent FPS after subsampling
                subsample_ratio = max_frames_in_video / original_frame_count
                data['total_duration'] *= subsample_ratio

        def _resize_to_reference(
            frames: torch.Tensor,
            reference: Optional[Tuple[int, int]],
        ) -> Tuple[torch.Tensor, Tuple[int, int]]:
            if reference is None:
                return frames, (frames.shape[2], frames.shape[3])

            height, width = reference
            if frames.shape[2] != height or frames.shape[3] != width:
                float_frames = frames.to(device="cpu", dtype=torch.float32)
                float_frames = torch.nn.functional.interpolate(
                    float_frames,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )
                frames = torch.round(torch.clamp(float_frames, 0.0, 255.0)).to(torch.uint8)
            return frames, reference

        if reduce_videos:
            reference_size: Optional[Tuple[int, int]] = None
            resized_frames = []
            combined_duration = 0.0

            for data in video_data.values():
                frames = data.get('frames')
                if frames is None:
                    continue

                frames, reference_size = _resize_to_reference(frames, reference_size)
                resized_frames.append(frames)
                combined_duration += data['total_duration']

            combined_video: Optional[Tuple[torch.Tensor, float]] = None
            if resized_frames:
                concatenated_frames = torch.cat(resized_frames, dim=0).to(torch.uint8)
                combined_video = (concatenated_frames, combined_duration)

            video_inserted = False
            for event in timeline_events:
                if event['type'] == 'text':
                    content.append({"type": "text", "text": event['text']})
                    continue

                if video_inserted or combined_video is None:
                    continue

                frames_tensor, duration = combined_video
                content.append({"type": "video"})
                video_inputs.append(frames_tensor.to(device="cpu", dtype=torch.uint8, copy=False).contiguous())
                fps_value = frames_tensor.shape[0] / duration if duration > 0 else 0.0
                video_fps.append(fps_value)
                video_inserted = True

            if combined_video is not None and not video_inserted:
                frames_tensor, duration = combined_video
                content.append({"type": "video"})
                video_inputs.append(frames_tensor.to(device="cpu", dtype=torch.uint8, copy=False).contiguous())
                fps_value = frames_tensor.shape[0] / duration if duration > 0 else 0.0
                video_fps.append(fps_value)
        else:
            reference_size: Optional[Tuple[int, int]] = None
            processed_video_ids = set()

            for event in timeline_events:
                if event['type'] == 'text':
                    content.append({"type": "text", "text": event['text']})
                    continue

                video_id = event['video_id']
                if video_id in processed_video_ids:
                    continue

                data = video_data.get(video_id)
                if not data:
                    continue

                frames = data.get('frames')
                if frames is None:
                    continue

                frames, reference_size = _resize_to_reference(frames, reference_size)
                frames_uint8 = frames.to(device="cpu", dtype=torch.uint8, copy=False).contiguous()

                content.append({"type": "video"})
                video_inputs.append(frames_uint8)

                duration = data['total_duration']
                fps_value = frames_uint8.shape[0] / duration if duration > 0 else 0.0
                video_fps.append(fps_value)

                processed_video_ids.add(video_id)
        # Add the question at the end
        content.append({
            "type": "text",
            "text": question
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process inputs - set do_rescale=False since videos are already in [0, 255] uint8 range
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = self.processor(**processor_kwargs)
        # Move to model's input embedding device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)
        drop_method = DROP_METHOD
        drop_video_limit = None
        if drop_method is not None and drop_method.lower() != "none":
            if len(video_inputs) > DROP_MAX_VIDEOS:
                drop_video_limit = DROP_MAX_VIDEOS

        generated_ids = self.model.generate(**inputs, 
            max_new_tokens=max_tokens,
            drop_method=drop_method,
            drop_threshold=DROP_THRESHOLD,
            drop_absolute=DROP_ABSOLUTE,
            dr_save_path=DR_SAVE_PATH,
            drop_max_videos=drop_video_limit)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # Calculate FLOPs for the question answering
            if video_inputs:
                total_frames = sum(v.shape[0] for v in video_inputs)
                vision_height = video_inputs[0].shape[2]
                vision_width = video_inputs[0].shape[3]
            else:
                total_frames = 0
                vision_height = 0
                vision_width = 0

            drop_stats = self._read_latest_drop_stats()
            tokens_dropped = int(drop_stats.get('drop', 0)) if drop_stats else 0
            tokens_total = drop_stats.get('total') if drop_stats else None

            flops_breakdown = timechat_online_flops(
                vision_frames=total_frames,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=int(inputs.input_ids.shape[1]),
                num_generated=len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0,
                tokens_dropped=tokens_dropped,
                tokens_total_before_drop=tokens_total,
                do_backward=False,
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
                float(flops_breakdown["total_flops"] if isinstance(flops_breakdown, dict) else flops_breakdown),
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_memory_total=state_mem,
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

        # Build timeline (shared across all questions) - copying logic from ask_question
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

        # Collect and process video data
        video_data: Dict[Any, Dict[str, Any]] = {}
        for event in timeline_events:
            if event['type'] != 'video':
                continue
            video_id = event['video_id']
            segment = event['segment']
            if video_id not in video_data:
                video_data[video_id] = {'frames': None, 'total_duration': 0.0}
            if video_data[video_id]['frames'] is None:
                video_data[video_id]['frames'] = segment['frames']
            else:
                video_data[video_id]['frames'] = torch.cat(
                    [video_data[video_id]['frames'], segment['frames']], dim=0
                )
            video_data[video_id]['total_duration'] += segment['duration']

        # Apply max_frames_in_video limit
        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames is None:
                continue
            original_frame_count = frames.shape[0]
            if original_frame_count > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)
                data['frames'] = frames[indices]
                subsample_ratio = max_frames_in_video / original_frame_count
                data['total_duration'] *= subsample_ratio

        # Build shared content
        shared_content: List[Dict[str, Any]] = []
        video_inputs: List[torch.Tensor] = []
        video_fps: List[float] = []
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames is not None and frames.shape[0] > 0:
                        video_inputs.append(frames)
                        duration = video_data[video_id]['total_duration']
                        fps_value = frames.shape[0] / duration if duration > 0 else 1.0
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

        # Apply chat template
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
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
                        # TimeChat uses same processor as Qwen2.5-VL - nested lists
                        chunk_size = len(chunk_texts)
                        processor_kwargs['videos'] = [video_inputs for _ in range(chunk_size)]

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(input_device)

                    drop_method = DROP_METHOD
                    drop_video_limit = None
                    if drop_method is not None and drop_method.lower() != "none":
                        if len(video_inputs) > DROP_MAX_VIDEOS:
                            drop_video_limit = DROP_MAX_VIDEOS

                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        drop_method=drop_method,
                        drop_threshold=DROP_THRESHOLD,
                        drop_absolute=DROP_ABSOLUTE,
                        dr_save_path=DR_SAVE_PATH,
                        drop_max_videos=drop_video_limit
                    )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
                    print(f"[TimeChat] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch] ===== BATCHED RESPONSE TimeChat ({len(all_output_texts)} questions, batch_size={batch_size}) =====")
        for i, text in enumerate(all_output_texts[:3]):
            print(f"Q{i+1}: {text}")
        if len(all_output_texts) > 3:
            print(f"... and {len(all_output_texts)-3} more")
        print("[model-output-batch] ===== END =====", flush=True)

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
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()

        # Additional memory cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated
        # Note: Don't reset metrics here - they should accumulate across questions

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current model state to memory.

        Returns a deep copy of all state so GPU memory can be freed while preserving
        the ability to restore the exact state later. Preserves metrics across save/load.
        """
        import copy

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
            include_fields = [
                'latency_add_video',
                'latency_add_text',
                'flops_add_video',
                'flops_add_text',
                'state_memory_floats',
                'state_memory_after_add_video',
                'state_memory_after_add_text',
                'state_memory_delta_add_video',
                'state_memory_delta_add_text',
                'peak_gpu_mem_increase_add_video',
                'peak_gpu_mem_increase_add_text',
                'peak_gpu_mem_absolute_add_video',
                'peak_gpu_mem_absolute_add_text',
                'video_timestamps_add_video',
                'video_timestamps_add_text',
            ]
            metrics_dict: Dict[str, Any] = {
                name: copy.deepcopy(getattr(self._metrics, name)) for name in include_fields
            }
            metrics_dict['first_oom_timestamp'] = self._metrics.first_oom_timestamp
            state['_metrics'] = metrics_dict

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Restores video segments (moving them back to model device), text entries, timing,
        and preserves metrics across save/load cycles.
        """
        # Restore video segments while keeping them on CPU uint8 tensors
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                frames_value = segment['frames']
                if torch.is_tensor(frames_value):
                    frames_cpu = frames_value.clone().contiguous().cpu()
                    if frames_cpu.dtype.is_floating_point:
                        frames_cpu = torch.clamp(frames_cpu, 0.0, 1.0)
                        frames_cpu = torch.round(frames_cpu * 255.0).to(torch.uint8)
                    elif frames_cpu.dtype != torch.uint8:
                        frames_cpu = frames_cpu.to(torch.uint8)
                else:
                    frames_cpu = frames_value

                restored_segment = {
                    'frames': frames_cpu,
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
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()

        # Restore metrics if present and metrics are enabled
        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
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
            include_fields = [
                'latency_add_video',
                'latency_add_text',
                'flops_add_video',
                'flops_add_text',
                'state_memory_floats',
                'state_memory_after_add_video',
                'state_memory_after_add_text',
                'state_memory_delta_add_video',
                'state_memory_delta_add_text',
                'peak_gpu_mem_increase_add_video',
                'peak_gpu_mem_increase_add_text',
                'peak_gpu_mem_absolute_add_video',
                'peak_gpu_mem_absolute_add_text',
                'video_timestamps_add_video',
                'video_timestamps_add_text',
            ]

            for name in include_fields:
                value = metrics_data.get(name, [])
                setattr(self._metrics, name, value.copy() if isinstance(value, list) else [])

            self._metrics.first_oom_timestamp = metrics_data.get('first_oom_timestamp')

            for name, value in preserved.items():
                setattr(self._metrics, name, value)

            self._sync_state_memory_tracking_from_metrics()

    def _read_latest_drop_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch the most recent token-drop statistics emitted by TimeChat."""
        drop_path = getattr(self, "_drop_log_path", None)
        if not drop_path or not os.path.exists(drop_path):
            return None

        try:
            with open(drop_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except OSError:
            return None

        if not lines:
            return None

        if self._drop_log_line_count is not None and len(lines) <= self._drop_log_line_count:
            # No new entries since last read; reuse cached stats
            return self._last_drop_stats

        latest_line = lines[-1]
        self._drop_log_line_count = len(lines)

        try:
            parsed = json.loads(latest_line)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        self._last_drop_stats = parsed
        return parsed

    def _get_state_memory_floats(self) -> float:
        """
        Calculates the total memory usage of the stored video frames state.
        """
        total_floats = 0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                frames_tensor = segment.get('frames')
                if torch.is_tensor(frames_tensor):
                    total_floats += frames_tensor.numel()

        for text, _ in self.text_entries:
            total_floats += self._count_text_tokens(text)
        return total_floats

    def _get_tokenizer(self):
        if self._tokenizer_ref is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor does not expose a tokenizer; cannot measure text state size")
            self._tokenizer_ref = tokenizer
        return self._tokenizer_ref

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cache = getattr(self, "_text_token_cache", None)
        if cache is None:
            cache = {}
            self._text_token_cache = cache
        cached = cache.get(text)
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
        cache[text] = count
        return count
    
