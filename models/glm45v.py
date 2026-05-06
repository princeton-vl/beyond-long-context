"""
GLM-4.5V video-language model implementation.
"""

import sys
import os
import time
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import infer_auto_device_map

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.base_interface import VideoLanguageModelInterface, PerformanceMetrics
from metrics.flops_calc import glm45v_flops
from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
from utils.paths import get_model_cache_dir

GLM_ATTENTION_IMPLEMENTATION = "sdpa"

# Per-GPU memory cap (GiB) used when constructing Accelerate's max_memory map.
# Using all available memory (~134 GiB on H100) paradoxically causes worse placement;
# 130 GiB leaves headroom for activations without forcing offload.
_PER_GPU_MEM_GIB_DEFAULT = 130

# Override the GLM-4.5V video processor's longest-edge cap. The default
# (47,040,000 = 6 * 4096 * 1916) silently downsamples high-res inputs;
# this value (1024 * 448 * 448) keeps native resolution for our benchmarks.
_GLM_LONGEST_EDGE = 1024 * 448 * 448


class GLM45V(VideoLanguageModelInterface):
    """
    GLM-4.5V video-language model implementation.

    Implements the VideoLanguageModelInterface for GLM-4.5V model
    with support for video and text context processing.
    """
    
    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        """Initialize GLM45V with proper base class initialization."""
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )


    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")
        cache_dir = get_model_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        self._min_pixels = min_pixels
        self._max_pixels = max_pixels

        if not torch.cuda.is_available():
            raise RuntimeError("GLM-4.5V requires CUDA devices for multi-GPU placement.")

        # ---- pick latest snapshot deterministically
        repo_dir = os.path.join(cache_dir, "models--zai-org--GLM-4.5V")
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        hashes: List[str] = []
        if os.path.isdir(snapshots_dir):
            snapshot_hashes = [
                h for h in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, h))
            ]
            hashes = sorted(
                snapshot_hashes,
                key=lambda h: os.path.getmtime(os.path.join(snapshots_dir, h)),
            )

        if hashes:
            local_ckpt_dir = os.path.join(snapshots_dir, hashes[-1])

            # Cap memory per GPU to avoid poor distribution by Accelerate.
            # See _PER_GPU_MEM_GIB_DEFAULT comment above.
            max_memory = {}
            per_gpu_limit = int(max_gpu_mem) if max_gpu_mem is not None else _PER_GPU_MEM_GIB_DEFAULT
            for i in range(torch.cuda.device_count()):
                max_memory[i] = f"{per_gpu_limit}GiB"
            max_memory["cpu"] = "0GiB"  # Disable CPU offload

            no_split = ["Glm4vMoeTextDecoderLayer", "Glm4vMoeVisionBlock"]

            with init_empty_weights():
                model_skel = Glm4vMoeForConditionalGeneration.from_pretrained(
                    local_ckpt_dir,
                    dtype=torch.bfloat16,
                    attn_implementation=GLM_ATTENTION_IMPLEMENTATION,
                )

            device_map = infer_auto_device_map(
                model_skel,
                max_memory=max_memory,
                no_split_module_classes=no_split,
                dtype=torch.bfloat16,
            )

            # max_memory["cpu"] = "0GiB" above forbids any CPU/disk offload, so the
            # offload_folder/offload_state_dict/offload_buffers kwargs would never
            # actually be consulted. Dropping them keeps the contract explicit.
            self.model = load_checkpoint_and_dispatch(
                model_skel,
                checkpoint=local_ckpt_dir,
                device_map=device_map,
                dtype=torch.bfloat16,
                no_split_module_classes=no_split,
            )
            self.model.eval()
            self.model.requires_grad_(False)
        else:
            print("[GLM45V] No local snapshot found; downloading model from Hugging Face.")
            self.model = Glm4vMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation=GLM_ATTENTION_IMPLEMENTATION,
            )

        self.processor = AutoProcessor.from_pretrained("zai-org/GLM-4.5V", cache_dir=cache_dir, trust_remote_code=True)
        # Disable silent video downsampling for GLM-4.5V (default longest_edge=47,040,000).
        self.processor.video_processor.size["longest_edge"] = _GLM_LONGEST_EDGE
        self._model_devices = self._resolve_model_devices()

        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None
        self._pending_vision_flops: float = 0.0
        self._pending_text_flops: float = 0.0

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()
    
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

        # Enforce pixel constraints if provided
        if torch.is_tensor(video_frames) and video_frames.ndim == 4:
            height = video_frames.shape[2]
            width = video_frames.shape[3]
            area = height * width

            target_area = None
            if self._max_pixels and area > self._max_pixels:
                target_area = self._max_pixels
            elif self._min_pixels and area < self._min_pixels:
                target_area = self._min_pixels

            if target_area and height > 0 and width > 0:
                scale = math.sqrt(target_area / float(area))
                new_h = max(1, int(round(height * scale)))
                new_w = max(1, int(round(width * scale)))

                if new_h != height or new_w != width:
                    dtype = video_frames.dtype
                    needs_conversion = not torch.is_floating_point(video_frames)
                    frames_for_resize = (
                        video_frames.float() / 255.0 if needs_conversion else video_frames
                    )
                    video_frames = F.interpolate(
                        frames_for_resize,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if needs_conversion:
                        video_frames = (video_frames * 255.0).to(dtype)

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()
        
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
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            self._reset_cuda_peak_memory()
            
            if torch.is_tensor(video_frames) and video_frames.ndim == 4:
                vision_frames = int(video_frames.shape[0])
                vision_height = int(video_frames.shape[2])
                vision_width = int(video_frames.shape[3])
            else:
                vision_frames = 0
                vision_height = 0
                vision_width = 0

            add_video_flops = self._estimate_vision_flops(
                vision_frames,
                vision_height,
                vision_width,
            )
            self._pending_vision_flops += add_video_flops

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
        """
        # Add text with current timestamp (text goes first if same time as video)
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()
        
        timestamp = max(0.0, float(current_video_time))
        self.text_entries.append((text, timestamp))
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            self._reset_cuda_peak_memory()

            token_count = self._count_text_tokens(text)
            add_text_flops = self._estimate_text_prompt_flops(token_count)
            self._pending_text_flops += add_text_flops

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                add_text_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                timestamp,
                state_mem,
            )
    
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """
        Ask a question based on the current context.
        
        This method should generate a response based on all previously added
        video and text content without modifying the context.
        
        Args:
            question: Question to ask
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")
            
        Returns:
            Generated response as string
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()
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
        video_data = {}  # video_id -> {'frames': Tensor, 'total_duration': float}
        video_fps = []
        video_metadata: List[Dict[str, Any]] = []
        
        # First pass: collect all frames for each video
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']
                
                if video_id not in video_data:
                    video_data[video_id] = {'frames': None, 'total_duration': 0.0}
                
                # Add all frames (we'll sample later)
                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'] = torch.cat([video_data[video_id]['frames'], segment['frames']], dim=0)
                
                video_data[video_id]['total_duration'] += segment['duration']
        
        # Second pass: apply max_frames_in_video limit to each complete video
        self._video_was_truncated = False  # Reset flag
        for video_id in video_data:
            frames = video_data[video_id]['frames']
            original_frame_count = frames.shape[0]
            if original_frame_count > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:  # TIME or SEGMENT (same logic)
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)

                video_data[video_id]['frames'] = frames[indices]

                # Adjust duration proportionally to maintain consistent FPS after subsampling
                subsample_ratio = max_frames_in_video / original_frame_count
                video_data[video_id]['total_duration'] *= subsample_ratio
        
        # Process timeline events for content
        processed_video_ids = set()
        
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                
                # Only add video to content once per video_id
                if video_id not in processed_video_ids:
                    content.append({"type": "video"})
                    
                    # Add to video inputs and calculate FPS
                    frames_tensor = video_data[video_id]['frames']
                    video_inputs.append(frames_tensor)
                    duration = float(video_data[video_id]['total_duration'])
                    safe_duration = float(max(duration, 1e-6))
                    frame_count = int(frames_tensor.shape[0])
                    fps = 1.0  # All videos use fps=1.0
                    height = int(frames_tensor.shape[2]) if frames_tensor.ndim >= 3 else None
                    width = int(frames_tensor.shape[3]) if frames_tensor.ndim >= 4 else None

                    video_fps.append(fps)
                    # Provide sequential indices so the GLM processor can recover per-frame timestamps
                    frame_indices = list(range(frame_count))
                    video_metadata.append(
                        {
                            "total_num_frames": frame_count,
                            "fps": fps,
                            "duration": duration if duration > 0 else safe_duration,
                            "video_backend": "tensor",
                            "frames_indices": frame_indices,
                            "height": height,
                            "width": width,
                        }
                    )
                    
                    processed_video_ids.add(video_id)
            
            elif event['type'] == 'text':
                content.append({"type": "text", "text": event['text']})
        
        # Add the question at the end
        content.append({
            "type": "text",
            "text": question
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, do_sample_frames=False
        )
        
        # Process inputs
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "do_sample_frames": False,
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            if video_metadata:
                processor_kwargs["video_metadata"] = video_metadata
            coalesced_fps = self._coalesce_video_fps(video_fps)
            if coalesced_fps is not None:
                processor_kwargs["fps"] = coalesced_fps
        inputs = self.processor(**processor_kwargs)

        # Remove token_type_ids if present, as GLM-4.5V model doesn't accept it
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')

        # Move inputs to the device of the first embedding layer (proper multi-GPU support)
        # Using next(parameters()) would only get cuda:0 even if model spans multiple GPUs
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is not None:
            model_device = input_embeddings.weight.device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max(1, int(max_tokens)))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            self._reset_cuda_peak_memory()
            # Calculate FLOPs for the question answering
            if video_inputs:
                total_frames = sum([v.shape[0] for v in video_inputs])
                vision_height = video_inputs[0].shape[2]
                vision_width = video_inputs[0].shape[3]
            else:
                total_frames = 0
                vision_height = 0
                vision_width = 0

            flops_breakdown = glm45v_flops(
                vision_frames=total_frames,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=int(inputs["input_ids"].shape[1]),
                num_generated=len(generated_ids_trimmed[0]),
                do_backward=False
            )
            total_flops = float(flops_breakdown.get("total_flops", 0.0))
            question_flops = max(total_flops - self._pending_vision_flops - self._pending_text_flops, 0.0)
            self._pending_vision_flops = 0.0
            self._pending_text_flops = 0.0
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
                question_flops,
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
        Ask multiple questions in a batched inference call.
        GLM-4.5V implementation with OOM retry (halves batch size on OOM).
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()

        # Build timeline (same for all questions)
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

        # Build shared content
        shared_content = []
        video_inputs = []
        video_data = {}
        video_fps = []

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

        # Apply frame limits
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

        processed_video_ids = set()
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    if frames_tensor is not None and frames_tensor.shape[0] > 0:
                        frames_tensor = frames_tensor.to(device='cpu')
                        video_inputs.append(frames_tensor)
                        total_duration = video_data[video_id]['total_duration']
                        if total_duration and total_duration > 0:
                            video_fps.append(frames_tensor.shape[0] / total_duration)
                        else:
                            video_fps.append(None)
                        shared_content.append({"type": "video", "video": len(video_inputs) - 1})
                        processed_video_ids.add(video_id)
            elif event['type'] == 'text':
                shared_content.append({"type": "text", "text": event['text']})

        # Build batch messages
        batch_messages = []
        for question_text in questions:
            content = list(shared_content)
            content.append({"type": "text", "text": question_text})
            batch_messages.append({"role": "user", "content": content})

        # Apply chat template to all
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, do_sample_frames=False)
            for msg in batch_messages
        ]

        # Get model device
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else torch.device("cuda")

        # Batched inference with OOM retry (halve batch size on OOM)
        batch_size = len(questions)
        all_output_texts = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Process this chunk
                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }
                    if video_inputs:
                        # Bug fix: GLM's processor expects 'videos' (matches the single-question
                        # path at the top of ask_question and the isolated batch path below).
                        # Passing 'images' silently routed clips through the still-image branch.
                        processor_kwargs["videos"] = video_inputs

                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    with torch.inference_mode():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=max(1, int(max_tokens)))

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
                    print(f"[GLM45V] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Reset results for retry
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        output_texts = all_output_texts

        # Debug output
        if output_texts:
            print(f"\n[model-output-batch] ===== BATCHED RESPONSE GLM ({len(output_texts)} questions, batch_size={batch_size}) =====")
            for i, text in enumerate(output_texts[:3]):
                print(f"Q{i+1}: {text}")
            if len(output_texts) > 3:
                print(f"... and {len(output_texts)-3} more")
            print("[model-output-batch] ===== END =====", flush=True)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time

        return output_texts

    def ask_question_batch_isolated(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
    ) -> List[str]:
        """
        TRUE parallel batching with isolated contexts per question.

        Each question gets its own complete input context.

        Args:
            contexts: List of context dicts, one per question.
                     For sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                        'question_text': str, 'mode': 'sequence'}
                     For video mode: {'main_video_frames': tensor, 'candidate_video_frames': tensor,
                                     'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate per response
            max_frames_in_video: Max frames per video

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # Match the truncation-flag reset that ask_question and ask_question_batch
        # already perform; was_video_truncated() must reflect this call only.
        self._video_was_truncated = False

        if len(contexts) == 1:
            # GLM: Use single-context path for batch_size=1.
            # The batch path's multi-video flattening in GLM's custom processor
            # doesn't properly segment videos per batch item, causing divergence.
            return [self._process_single_context(contexts[0], max_tokens, max_frames_in_video)]

        # Determine mode from first context
        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()

        # Build separate messages for each question
        batch_messages = []
        video_inputs_per_question = []

        for ctx in contexts:
            if is_sequence_mode:
                # Sequence mode: pure text input
                full_text = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )

                content = [{"type": "text", "text": full_text}]
                message = {"role": "user", "content": content}
                batch_messages.append(message)
                video_inputs_per_question.append(None)

            else:
                # Video mode: main video + candidate clip as SEPARATE videos
                main_frames = ctx.get('main_video_frames')
                candidate_frames = ctx.get('candidate_video_frames')

                # Build video list for this question
                videos_for_question = []
                content = []

                # Add main video first
                if main_frames is not None and main_frames.shape[0] > 0:
                    content.append({"type": "text", "text": "Here is a main video to remember:"})
                    # Sample to max_frames_in_video
                    if main_frames.shape[0] > max_frames_in_video:
                        indices = np.linspace(0, main_frames.shape[0] - 1,
                                            max_frames_in_video, dtype=int)
                        sampled_main = main_frames[indices]
                    else:
                        sampled_main = main_frames
                    videos_for_question.append(sampled_main)
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                # Add candidate clip second
                if candidate_frames is not None and candidate_frames.shape[0] > 0:
                    content.append({"type": "text", "text": "\nHere is a candidate clip:\n"})
                    # Sample to remaining budget
                    remaining_frames = max_frames_in_video - (sampled_main.shape[0] if main_frames is not None else 0)
                    if candidate_frames.shape[0] > remaining_frames:
                        indices = np.linspace(0, candidate_frames.shape[0] - 1,
                                            remaining_frames, dtype=int)
                        sampled_candidate = candidate_frames[indices]
                    else:
                        sampled_candidate = candidate_frames
                    videos_for_question.append(sampled_candidate)
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                # Add question text
                content.append({"type": "text", "text": ctx['question_text']})

                message = {"role": "user", "content": content}
                batch_messages.append(message)
                video_inputs_per_question.append(videos_for_question)

        # Apply chat template to each message
        batch_texts = []
        for message in batch_messages:
            text = self.processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
                do_sample_frames=False
            )
            batch_texts.append(text)

        # Get device
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings else torch.device("cuda")

        # Batch process with OOM retry
        current_batch_size = len(contexts)
        all_output_texts = [None] * len(contexts)

        while current_batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), current_batch_size):
                    chunk_end = min(chunk_start + current_batch_size, len(contexts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Prepare processor kwargs
                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                        "do_sample_frames": False,
                    }

                    # Handle video inputs
                    if not is_sequence_mode:
                        # For video mode, each question has its own video list
                        chunk_videos = video_inputs_per_question[chunk_start:chunk_end]

                        # GLM processor expects: videos as list of tensors
                        # Need to flatten: [[main1, cand1], [main2, cand2]] -> [main1, cand1, main2, cand2]
                        all_videos = []
                        for video_list in chunk_videos:
                            if video_list:
                                all_videos.extend(video_list)

                        if all_videos:
                            processor_kwargs["videos"] = all_videos
                            processor_kwargs["fps"] = 1.0
                            # Build video_metadata to match single-path behavior
                            video_metadata = []
                            for vid in all_videos:
                                fc = int(vid.shape[0])
                                video_metadata.append({
                                    "total_num_frames": fc,
                                    "fps": 1.0,
                                    "duration": float(fc),
                                    "video_backend": "tensor",
                                    "frames_indices": list(range(fc)),
                                    "height": int(vid.shape[2]) if vid.ndim >= 3 else None,
                                    "width": int(vid.shape[3]) if vid.ndim >= 4 else None,
                                })
                            processor_kwargs["video_metadata"] = video_metadata

                    # Tokenize and prepare inputs
                    inputs = self.processor(**processor_kwargs)

                    # Remove token_type_ids if present, as GLM-4.5V model doesn't accept it
                    if 'token_type_ids' in inputs:
                        inputs.pop('token_type_ids')

                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v
                             for k, v in inputs.items()}

                    # Forward pass
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max(1, int(max_tokens))
                        )

                    # Trim prompt tokens
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

                    # Store results
                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = current_batch_size
                    current_batch_size = current_batch_size // 2
                    print(f"[GLM45V-Isolated] OOM: Reducing batch size {old_batch_size} → {current_batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output
        if all_output_texts:
            print(f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE GLM "
                  f"({len(all_output_texts)} questions, batch_size={current_batch_size}, mode={mode}) =====")
            for i, text in enumerate(all_output_texts[:3]):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"Q{i+1}: {preview}")
            if len(all_output_texts) > 3:
                print(f"... and {len(all_output_texts)-3} more")
            print("[model-output-batch-isolated] ===== END =====", flush=True)

        # Metrics
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_memory_allocated() - baseline_mem
            self._metrics.latency_ask_question.append(latency)
            self._metrics.peak_gpu_mem_increase_ask_question.append(peak_mem)

        return all_output_texts

    def _process_single_context(self, ctx: Dict[str, Any], max_tokens: int, max_frames_in_video: int) -> str:
        """Helper to process a single context (fallback for batch size 1)."""
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            full_text = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            return self.ask_question(full_text, current_video_time=0.0, max_tokens=max_tokens)
        else:
            # For video mode single question, add videos and ask
            main_frames = ctx.get('main_video_frames')
            candidate_frames = ctx.get('candidate_video_frames')

            # Clear and rebuild context
            self.clear_context()

            # Add main video — duration = num_frames at 1fps
            if main_frames is not None:
                main_dur = float(main_frames.shape[0])  # 1fps → duration = frame count
                self.add_video(main_frames, time_start=0.0, time_end=main_dur, video_id="main_video")
            else:
                main_dur = 0.0

            # Add candidate clip — placed after main video
            cand_start = main_dur + 1.0
            if candidate_frames is not None:
                cand_dur = float(candidate_frames.shape[0])
                self.add_video(candidate_frames, time_start=cand_start, time_end=cand_start + cand_dur, video_id="candidate_clip")
            else:
                cand_dur = 0.0

            return self.ask_question(ctx['question_text'], current_video_time=cand_start + cand_dur, max_tokens=max_tokens)

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
        self._pending_vision_flops = 0.0
        self._pending_text_flops = 0.0

        # Additional memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            devices = getattr(self, "_model_devices", [])
            if devices:
                try:
                    prev_device = torch.cuda.current_device()
                except RuntimeError:
                    prev_device = None
                for device in devices:
                    try:
                        torch.cuda.empty_cache(device)
                    except TypeError:
                        torch.cuda.set_device(device)
                        torch.cuda.empty_cache()
                if prev_device is not None:
                    torch.cuda.set_device(prev_device)
            else:
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
            'pending_vision_flops': self._pending_vision_flops,
            'pending_text_flops': self._pending_text_flops,
            '_metrics': None
        }

        # Save metrics if enabled (preserve across save/load cycles)
        if self.enable_metrics and self._metrics is not None:
            metrics_dict = {
                'latency_add_video': copy.deepcopy(self._metrics.latency_add_video),
                'latency_add_text': copy.deepcopy(self._metrics.latency_add_text),
                'latency_ask_question': copy.deepcopy(self._metrics.latency_ask_question),
                'flops_add_video': copy.deepcopy(self._metrics.flops_add_video),
                'flops_add_text': copy.deepcopy(self._metrics.flops_add_text),
                'flops_ask_question': copy.deepcopy(self._metrics.flops_ask_question),
                'state_memory_floats': copy.deepcopy(self._metrics.state_memory_floats),
                'state_memory_after_add_video': copy.deepcopy(self._metrics.state_memory_after_add_video),
                'state_memory_after_add_text': copy.deepcopy(self._metrics.state_memory_after_add_text),
                'state_memory_after_ask_question': copy.deepcopy(self._metrics.state_memory_after_ask_question),
                'state_memory_delta_add_video': copy.deepcopy(self._metrics.state_memory_delta_add_video),
                'state_memory_delta_add_text': copy.deepcopy(self._metrics.state_memory_delta_add_text),
                'state_memory_delta_ask_question': copy.deepcopy(self._metrics.state_memory_delta_ask_question),
                'peak_gpu_mem_increase_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_video),
                'peak_gpu_mem_increase_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_text),
                'peak_gpu_mem_increase_ask_question': copy.deepcopy(self._metrics.peak_gpu_mem_increase_ask_question),
                'peak_gpu_mem_absolute_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_video),
                'peak_gpu_mem_absolute_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_text),
                'peak_gpu_mem_absolute_ask_question': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_ask_question),
                'video_timestamps_add_video': copy.deepcopy(self._metrics.video_timestamps_add_video),
                'video_timestamps_add_text': copy.deepcopy(self._metrics.video_timestamps_add_text),
                'video_timestamps_ask_question': copy.deepcopy(self._metrics.video_timestamps_ask_question),
                'question_correctness_rate': copy.deepcopy(self._metrics.question_correctness_rate),
                'question_dont_know_rate': copy.deepcopy(self._metrics.question_dont_know_rate),
                'question_answered_mask': copy.deepcopy(self._metrics.question_answered_mask),
                'video_timestamps_question_outcome': copy.deepcopy(self._metrics.video_timestamps_question_outcome),
            }
            state['_metrics'] = metrics_dict

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Restores video segments (moving them back to model device), text entries, timing,
        and preserves metrics across save/load cycles.
        """
        # Restore video segments and move tensors back to model device
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                frames = segment['frames']
                if torch.is_tensor(frames):
                    frames_restored = frames.clone()
                else:
                    frames_restored = frames

                restored_segment = {
                    'frames': frames_restored,
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
        self._pending_vision_flops = float(state.get('pending_vision_flops', 0.0))
        self._pending_text_flops = float(state.get('pending_text_flops', 0.0))
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()

        # Restore metrics if present and metrics are enabled
        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
            from models.base_interface import PerformanceMetrics

            # If we don't have metrics yet, create new object
            if self._metrics is None:
                self._metrics = PerformanceMetrics()

            # Restore only the accumulated add_video/add_text metrics (these should persist)
            # DO NOT restore ask_question metrics as they should not be duplicated
            metrics_data = state['_metrics']
            self._metrics.latency_add_video = metrics_data.get('latency_add_video', []).copy()
            self._metrics.latency_add_text = metrics_data.get('latency_add_text', []).copy()
            self._metrics.flops_add_video = metrics_data.get('flops_add_video', []).copy()
            self._metrics.flops_add_text = metrics_data.get('flops_add_text', []).copy()
            self._metrics.state_memory_floats = metrics_data.get('state_memory_floats', []).copy()
            self._metrics.state_memory_after_add_video = metrics_data.get('state_memory_after_add_video', []).copy()
            self._metrics.state_memory_after_add_text = metrics_data.get('state_memory_after_add_text', []).copy()
            self._metrics.state_memory_after_ask_question = metrics_data.get('state_memory_after_ask_question', []).copy()
            self._metrics.state_memory_delta_add_video = metrics_data.get('state_memory_delta_add_video', []).copy()
            self._metrics.state_memory_delta_add_text = metrics_data.get('state_memory_delta_add_text', []).copy()
            self._metrics.state_memory_delta_ask_question = metrics_data.get('state_memory_delta_ask_question', []).copy()
            self._metrics.peak_gpu_mem_increase_add_video = metrics_data.get('peak_gpu_mem_increase_add_video', []).copy()
            self._metrics.peak_gpu_mem_increase_add_text = metrics_data.get('peak_gpu_mem_increase_add_text', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_video = metrics_data.get('peak_gpu_mem_absolute_add_video', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_text = metrics_data.get('peak_gpu_mem_absolute_add_text', []).copy()
            self._metrics.video_timestamps_add_video = metrics_data.get('video_timestamps_add_video', []).copy()
            self._metrics.video_timestamps_add_text = metrics_data.get('video_timestamps_add_text', []).copy()
            self._metrics.video_timestamps_ask_question = metrics_data.get('video_timestamps_ask_question', []).copy()
            self._metrics.question_correctness_rate = metrics_data.get('question_correctness_rate', []).copy()
            self._metrics.question_dont_know_rate = metrics_data.get('question_dont_know_rate', []).copy()
            self._metrics.question_answered_mask = metrics_data.get('question_answered_mask', []).copy()
            self._metrics.video_timestamps_question_outcome = metrics_data.get('video_timestamps_question_outcome', []).copy()

            # Keep existing ask_question metrics (don't overwrite with saved state)
            self._sync_state_memory_tracking_from_metrics()

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
                raise RuntimeError("Processor does not expose a tokenizer; cannot measure text state size.")
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

    def _estimate_vision_flops(self, frame_count: int, height: int, width: int) -> float:
        if frame_count <= 0 or height <= 0 or width <= 0:
            return 0.0
        breakdown = glm45v_flops(
            vision_frames=frame_count,
            vision_height=height,
            vision_width=width,
            lang_prompt_len=frame_count,
            num_generated=0,
            do_backward=False,
        )
        return float(breakdown.get("vision_flops", 0.0))

    def _estimate_text_prompt_flops(self, text_tokens: int) -> float:
        if text_tokens <= 0:
            return 0.0
        breakdown = glm45v_flops(
            vision_frames=0,
            vision_height=0,
            vision_width=0,
            lang_prompt_len=text_tokens,
            num_generated=0,
            do_backward=False,
        )
        return float(breakdown.get("lang_prompt_flops", 0.0))
    
    def _resolve_model_devices(self) -> List[torch.device]:
        device_map = getattr(self.model, "hf_device_map", None)
        devices: List[torch.device] = []

        if device_map:
            for placement in device_map.values():
                if isinstance(placement, torch.device):
                    target = placement
                elif isinstance(placement, str):
                    if placement == "disk" or placement.startswith("cpu"):
                        raise RuntimeError(f"Model placement includes non-CUDA device: {placement}")
                    target = torch.device(placement)
                elif isinstance(placement, int):
                    target = torch.device("cuda", placement)
                else:
                    continue

                if target.type != "cuda":
                    raise RuntimeError(f"Model placement includes non-CUDA device: {target}")
                if target not in devices:
                    devices.append(target)

        if not devices:
            param = next(self.model.parameters())
            if param.device.type != "cuda":
                raise RuntimeError(f"Unexpected parameter device {param.device}; CUDA required.")
            devices.append(param.device)

        return devices

    def _cuda_memory_allocated(self) -> int:
        if not torch.cuda.is_available():
            return 0

        devices = getattr(self, "_model_devices", [])
        if not devices:
            return torch.cuda.memory_allocated()

        return sum(torch.cuda.memory_allocated(device) for device in devices)

    def _cuda_peak_memory(self) -> int:
        if not torch.cuda.is_available():
            return 0

        devices = getattr(self, "_model_devices", [])
        if not devices:
            return torch.cuda.max_memory_allocated()

        return sum(torch.cuda.max_memory_allocated(device) for device in devices)

    def _reset_cuda_peak_memory(self) -> None:
        if not torch.cuda.is_available():
            return

        devices = getattr(self, "_model_devices", [])
        if not devices:
            torch.cuda.reset_peak_memory_stats()
            return

        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
