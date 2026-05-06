"""
Phi-4-Multimodal streaming wrapper compatible with the shared benchmarking interface.

Targets the microsoft/Phi-4-multimodal-instruct checkpoint and adapts its
image-token prompt format ("<|image_i|>") to the streaming evaluation harness.
"""

from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
from models.base_interface import VideoLanguageModelInterface, PerformanceMetrics
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import copy
from PIL import Image
from accelerate import dispatch_model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import time

from utils.paths import get_model_cache_dir


CACHE_ROOT = get_model_cache_dir()

# Phi-4 multimodal maximum frames per video/context.
# Heuristic cap to prevent index overflow and OOM; not sourced from model config.
PHI4_MAX_FRAMES_PER_VIDEO = 768
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)

from metrics.flops_calc import phi4mm_flops

    
class PhiMultimodalVideo(VideoLanguageModelInterface):
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
        """Initialize Phi-Multimodal with proper base class initialization."""
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
        # Define model path (honor caller-supplied model_id; fall back to default)
        model_path = self.model_id or "microsoft/Phi-4-multimodal-instruct"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        multi_gpu = visible_gpus > 1

        model_kwargs = dict(
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            low_cpu_mem_usage=True,
        )

        if multi_gpu:
            # Load weights on CPU first, then dispatch evenly across GPUs.
            model_kwargs['device_map'] = 'cpu'
        elif torch.cuda.is_available():
            model_kwargs['device_map'] = 'auto'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        if multi_gpu:
            gpu_indices = list(range(visible_gpus))
            device_map: Dict[str, int] = {}

            def assign(module_name: str, device_idx: int) -> None:
                device_map[module_name] = gpu_indices[device_idx % visible_gpus]

            assign('model.embed_tokens', 0)
            assign('model.embed_dropout', 0)
            assign('lm_head', 0)
            assign('model.embed_tokens_extend', 0)

            layers = getattr(self.model.model, 'layers', [])
            for layer_idx, _ in enumerate(layers):
                assign(f'model.layers.{layer_idx}', layer_idx)

            assign('model.norm', len(layers) - 1 if layers else 0)

            self.model = dispatch_model(self.model, device_map=device_map)
        elif torch.cuda.is_available():
            # Use device_map="auto" instead of forcing to single GPU
            # This allows proper multi-GPU distribution
            from accelerate import infer_auto_device_map

            # Cap memory per GPU to avoid poor distribution by Accelerate
            max_memory = {}
            # Use max_gpu_mem if provided, otherwise default to 60 GiB
            per_gpu_limit = int(max_gpu_mem) if max_gpu_mem is not None else 60
            for i in range(torch.cuda.device_count()):
                max_memory[i] = f"{per_gpu_limit}GiB"
            max_memory["cpu"] = "0GiB"  # Disable CPU offload

            device_map = infer_auto_device_map(self.model, max_memory=max_memory)
            self.model = dispatch_model(self.model, device_map=device_map)

        self.temperature = 0.1
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()


    def add_video(self, video_frames: List, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
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

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        frames = self._normalize_video_frames(video_frames)
        if not frames:
            raise ValueError("PhiMultimodalVideo requires at least one frame per segment")

        frame_count = len(frames)
        height, width = self._infer_frame_dimensions(frames[0])
        state_floats = self._compute_frame_float_total(frames)

        # Default video_id to 0 if not provided
        if video_id is None:
            video_id = 0
        
        if video_id not in self.video_segments:
            self.video_segments[video_id] = []
        
        segment = {
            'frames': frames,
            'time_start': time_start,
            'time_end': time_end,
            'duration': time_end - time_start,
            'num_frames': frame_count,
            'state_floats': state_floats,
            'vision_stats': {
                'frames': frame_count,
                'height': height,
                'width': width,
            },
        }

        self.video_segments[video_id].append(segment)
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            vision_flops = 0.0
            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                vision_flops,
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

            token_count = self._count_text_tokens(text)
            text_flops = 0
            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                text_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )
    
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO, sample_method: str = "TIME") -> str:
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
        # Enforce hard limit on frames per video
        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

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
                    video_data[video_id]['frames'].extend(segment['frames'])

                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(
                    segment.get('num_frames', len(segment['frames']))
                )

        # Second pass: apply max_frames_in_video limit to each complete video
        self._video_was_truncated = False  # Reset flag
        for video_id, data in video_data.items():
            frames = data['frames']
            if len(frames) > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.random.choice(len(frames), max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:  # TIME or SEGMENT (same logic)
                    indices = np.linspace(0, len(frames) - 1, max_frames_in_video, dtype=int)

                data['frames'] = [frames[i] for i in indices]

            trimmed_frame_count = len(data['frames'])
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
        video_images = []
        input_text = ""
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                # Only add video to content once per video_id
                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames is None or len(frames) == 0:
                        continue

                    video_images.extend(frames)
                    input_text += "".join(f'<|image_{i}|>' for i in range(len(video_images) - len(frames), len(video_images)))
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                input_text += " " + event['text']
        
        # Always append the natural-language question text so Phi sees the instruction
        question_text = question.strip() if isinstance(question, str) else ""
        if question_text:
            if input_text.strip():
                input_text = input_text.strip() + "\n\n" + question_text
            else:
                input_text = question_text
        
        
        messages = [
            {'role': 'user', 'content': input_text },
        ]
        
        # Apply chat template
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = {
            'text': text,
            'return_tensors': 'pt',
        }
        if video_images:
            processor_kwargs['images'] = video_images
        inputs = self.processor(**processor_kwargs)

        # Move inputs to the model's input embedding device for proper multi-GPU support
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is not None:
            model_device = input_embeddings.weight.device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        target_max = max(1, int(max_tokens))
        generation_args = {
            'max_new_tokens': target_max,
            'temperature': self.temperature,
            'do_sample': True,
        }

        generation = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=self.generation_config,
        )
        if isinstance(generation, tuple):
            generate_ids = generation[0]
        else:
            generate_ids = generation
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self.enable_metrics:
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
            question_timestamp = float(current_video_time) if current_video_time is not None else 0.0
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time
            vision_frames_total, vision_height, vision_width = self._summarize_video_dimensions(video_images)
            text_token_total = sum(
                self._count_text_tokens(event['text'])
                for event in timeline_events
                if event['type'] == 'text'
            )
            text_token_total += self._count_text_tokens(question)

            num_generated = int(generate_ids.shape[1]) if generate_ids is not None else 0
            flops_breakdown = phi4mm_flops(
                vision_frames=vision_frames_total,
                vision_height=vision_height,
                vision_width=vision_width,
                text_tokens=text_token_total,
                num_generated=num_generated,
                do_backward=False,
            )
            total_flops = float(flops_breakdown.get('total_flops', 0.0))
            question_flops = total_flops
            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                question_flops,
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
        max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO,
        sample_method: str = "TIME",
    ) -> List[str]:
        """
        Ask multiple questions in a batched inference call.
        All questions share the same video/text context.
        """
        if not questions:
            return []

        # Enforce hard limit on frames per video
        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        # Build timeline and video data (shared across all questions)
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

        # Collect video frames
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
                    video_data[video_id]['frames'].extend(segment['frames'])
                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(segment.get('num_frames', len(segment['frames'])))

        # Apply max_frames_in_video limit
        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames and len(frames) > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(len(frames), max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, len(frames) - 1, max_frames_in_video, dtype=int)
                data['frames'] = [frames[i] for i in indices]

        # Build shared content (images and base text)
        processed_video_ids = set()
        video_images = []
        base_text = ""
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames:
                        video_images.extend(frames)
                        base_text += "".join(f'<|image_{i}|>' for i in range(len(video_images) - len(frames), len(video_images)))
                        processed_video_ids.add(video_id)
            elif event['type'] == 'text':
                base_text += " " + event['text']

        # Build batch of texts (one per question)
        batch_texts = []
        for question in questions:
            question_text = question.strip() if isinstance(question, str) else ""
            if question_text:
                if base_text.strip():
                    input_text = base_text.strip() + "\n\n" + question_text
                else:
                    input_text = question_text
            else:
                input_text = base_text

            messages = [{'role': 'user', 'content': input_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        # Get model device
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

        all_responses: List[Optional[str]] = [None] * len(questions)

        # Phi batching: replicate images and use PER-SAMPLE index resetting
        if video_images:
            # FIXED: Each sample independently uses indices 0 to N-1
            # This prevents index overflow when batch_size * num_images exceeds limits
            batch_size_val = len(questions)
            num_images_per_sample = len(video_images)
            all_images = video_images * batch_size_val

            # Build base text from timeline (without images)
            base_text_parts = []
            for event in timeline_events:
                if event['type'] == 'text':
                    base_text_parts.append(event['text'])
            base_text = " ".join(base_text_parts) if base_text_parts else ""

            # Rebuild batch_texts with PER-SAMPLE index resetting (0 to N-1 for each)
            rebatched_texts = []
            for idx, question_text in enumerate(questions):
                # Each sample uses indices 0 to num_images_per_sample-1
                # Phi-4 processor handles per-sample indexing correctly
                sample_input_text = "".join(
                    f'<|image_{i}|>'
                    for i in range(num_images_per_sample)
                )
                if base_text:
                    sample_input_text += " " + base_text

                if question_text:
                    if sample_input_text.strip():
                        sample_input_text = sample_input_text.strip() + "\n\n" + question_text
                    else:
                        sample_input_text = question_text

                messages = [{'role': 'user', 'content': sample_input_text}]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                rebatched_texts.append(text)

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
            }
            if self.temperature > 0:
                generation_args['temperature'] = self.temperature
                generation_args['do_sample'] = True
            else:
                generation_args['do_sample'] = False

            try:
                processor_kwargs = {
                    'text': rebatched_texts,
                    'images': all_images,
                    'return_tensors': 'pt',
                    'padding': True,
                }

                inputs = self.processor(**processor_kwargs)
                inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                generation = self.model.generate(**inputs, **generation_args, generation_config=self.generation_config)
                if isinstance(generation, tuple):
                    generate_ids = generation[0]
                else:
                    generate_ids = generation

                input_len = inputs['input_ids'].shape[1]
                generate_ids = generate_ids[:, input_len:]

                all_responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[Phi] OOM, falling back to sequential")
                    # Sequential fallback
                    all_responses = [None] * len(questions)
                    for i, text in enumerate(batch_texts):
                        try:
                            processor_kwargs = {'text': text, 'return_tensors': 'pt', 'images': video_images}
                            inputs = self.processor(**processor_kwargs)
                            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                            generation = self.model.generate(**inputs, **generation_args, generation_config=self.generation_config)
                            if isinstance(generation, tuple):
                                generate_ids = generation[0]
                            else:
                                generate_ids = generation
                            input_len = inputs['input_ids'].shape[1]
                            generate_ids = generate_ids[:, input_len:]
                            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            all_responses[i] = response
                        except RuntimeError:
                            all_responses[i] = ""
                else:
                    raise
        else:
            # True batching for text-only
            batch_size = len(questions)
            all_responses = [None] * len(questions)
            while batch_size >= 1:
                try:
                    for chunk_start in range(0, len(questions), batch_size):
                        chunk_end = min(chunk_start + batch_size, len(questions))
                        chunk_texts = batch_texts[chunk_start:chunk_end]

                        processor_kwargs = {
                            'text': chunk_texts,
                            'return_tensors': 'pt',
                            'padding': True,
                        }

                        inputs = self.processor(**processor_kwargs)
                        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                        generation_args = {
                            'max_new_tokens': max(1, int(max_tokens)),
                        }
                        if self.temperature > 0:
                            generation_args['temperature'] = self.temperature
                            generation_args['do_sample'] = True
                        else:
                            generation_args['do_sample'] = False

                        generation = self.model.generate(
                            **inputs,
                            **generation_args,
                            generation_config=self.generation_config,
                        )

                        if isinstance(generation, tuple):
                            generate_ids = generation[0]
                        else:
                            generate_ids = generation

                        input_len = inputs['input_ids'].shape[1]
                        generate_ids = generate_ids[:, input_len:]

                        chunk_responses = self.processor.batch_decode(
                            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )

                        for i, resp in enumerate(chunk_responses):
                            all_responses[chunk_start + i] = resp

                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and batch_size > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        old_batch_size = batch_size
                        batch_size = batch_size // 2
                        print(f"[Phi] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                        # Keep partial responses instead of resetting to None
                    else:
                        raise

        # Debug output
        print(f"\n[model-output-batch] ===== BATCHED RESPONSE Phi ({len(all_responses)} questions, batch_size={len(questions)}) =====")
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
        max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO,
    ) -> List[str]:
        """
        TRUE parallel tensor batching with isolated contexts per question.

        Each context gets its own independent frames and text - no contamination.

        Key design (FIXED):
        - Phi-4 MM supports per-sample index resetting (each sample uses 0 to N-1)
        - We replicate all frames from all contexts into one big list
        - Each context's prompt references indices 0 to M-1 independently

        Args:
            contexts: List of context dicts, one per question.
                     Sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                    'question_text': str, 'mode': 'sequence'}
                     Video mode: {'main_video_frames': List[Image], 'candidate_video_frames': List[Image],
                                'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate per response
            max_frames_in_video: Max frames per video (used for truncation, capped at 768)

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # Enforce hard limit on frames per video
        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

        if len(contexts) == 1:
            # Single context - process directly
            return [self._process_single_isolated_context(contexts[0], max_tokens, max_frames_in_video)]

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        # Get model device
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

        if is_sequence_mode:
            # Sequence mode: pure text, no images
            batch_texts = []
            for ctx in contexts:
                # Build prompt with main sequence, candidate sequence, and question
                prompt_text = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                messages = [{'role': 'user', 'content': prompt_text}]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            # Process batch with OOM retry
            all_responses = self._execute_text_batch(
                batch_texts=batch_texts,
                max_tokens=max_tokens,
                model_device=model_device,
            )

        else:
            # Video mode: each context has main + candidate frames
            all_responses = self._execute_video_isolated_batch(
                contexts=contexts,
                max_tokens=max_tokens,
                max_frames_in_video=max_frames_in_video,
                model_device=model_device,
            )

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== Phi-4 MM ({len(all_responses)} questions, mode={mode}) =====")
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
        max_frames_in_video: int,
    ) -> str:
        """Process a single isolated context (helper for batch_size=1)."""
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            # Text only
            prompt_text = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_embeddings = self.model.get_input_embeddings()
            model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

            processor_kwargs = {'text': text, 'return_tensors': 'pt'}
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
                'temperature': self.temperature,
                'do_sample': True,
            }

            generation = self.model.generate(
                **inputs,
                **generation_args,
                generation_config=self.generation_config,
            )

            if isinstance(generation, tuple):
                generate_ids = generation[0]
            else:
                generate_ids = generation

            input_len = inputs['input_ids'].shape[1]
            generate_ids = generate_ids[:, input_len:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response

        else:
            # Video mode with frames
            main_frames = ctx.get('main_video_frames', [])
            candidate_frames = ctx.get('candidate_video_frames', [])

            # Combine frames
            all_frames = list(main_frames[:max_frames_in_video // 2]) + list(candidate_frames[:max_frames_in_video // 2])

            # Build prompt with image tokens - MODIFIED TO MATCH NON-BATCHED MODE
            main_frame_count = len(main_frames[:max_frames_in_video // 2])
            cand_frame_count = len(candidate_frames[:max_frames_in_video // 2])

            prompt_text = "Here is a main video to remember:"
            prompt_text += "".join(f'<|image_{i}|>' for i in range(main_frame_count))
            prompt_text += "\n\nHere is a candidate clip:\n"
            prompt_text += "".join(f'<|image_{i}|>' for i in range(main_frame_count, len(all_frames)))
            prompt_text += "\n\n" + ctx['question_text']

            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_embeddings = self.model.get_input_embeddings()
            model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

            processor_kwargs = {
                'text': text,
                'images': all_frames,
                'return_tensors': 'pt',
            }
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
                'temperature': self.temperature,
                'do_sample': True,
            }

            generation = self.model.generate(
                **inputs,
                **generation_args,
                generation_config=self.generation_config,
            )

            if isinstance(generation, tuple):
                generate_ids = generation[0]
            else:
                generate_ids = generation

            input_len = inputs['input_ids'].shape[1]
            generate_ids = generate_ids[:, input_len:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response

    def _execute_text_batch(
        self,
        batch_texts: List[str],
        max_tokens: int,
        model_device: Any,
    ) -> List[str]:
        """Execute batched text-only inference with OOM retry."""
        batch_size = len(batch_texts)
        all_responses: List[Optional[str]] = [None] * len(batch_texts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(batch_texts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(batch_texts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    processor_kwargs = {
                        'text': chunk_texts,
                        'return_tensors': 'pt',
                        'padding': True,
                    }

                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    generation_args = {
                        'max_new_tokens': max(1, int(max_tokens)),
                        'temperature': self.temperature,
                        'do_sample': True,
                    }

                    generation = self.model.generate(
                        **inputs,
                        **generation_args,
                        generation_config=self.generation_config,
                    )

                    if isinstance(generation, tuple):
                        generate_ids = generation[0]
                    else:
                        generate_ids = generation

                    input_len = inputs['input_ids'].shape[1]
                    generate_ids = generate_ids[:, input_len:]

                    chunk_responses = self.processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, resp in enumerate(chunk_responses):
                        all_responses[chunk_start + i] = resp

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[Phi-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        return all_responses

    def _execute_video_isolated_batch(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int,
        max_frames_in_video: int,
        model_device: Any,
    ) -> List[str]:
        """
        Execute batched video inference with isolated contexts.

        FIXED: Each context uses PER-SAMPLE index resetting (0 to M-1).
        This prevents index overflow when batching many contexts.

        Image indexing scheme (NEW):
        - Context 0: images 0 to M-1
        - Context 1: images 0 to M-1 (RESET, not M to 2M-1!)
        - Context 2: images 0 to M-1 (RESET, not 2M to 3M-1!)
        - etc.

        Where M = len(main_frames) + len(candidate_frames) for that context.
        Phi-4 processor handles per-sample indexing correctly.
        """
        # Collect all frames and build texts with PER-SAMPLE index resetting
        all_frames = []
        batch_texts = []

        for ctx_idx, ctx in enumerate(contexts):
            # Get frames for this context
            main_frames_raw = ctx.get('main_video_frames', [])
            candidate_frames_raw = ctx.get('candidate_video_frames', [])

            # Truncate if needed
            main_frames = list(main_frames_raw[:max_frames_in_video // 2])
            candidate_frames = list(candidate_frames_raw[:max_frames_in_video // 2])

            # Add frames to global list
            context_frames = main_frames + candidate_frames
            all_frames.extend(context_frames)

            # Build prompt with PER-SAMPLE index resetting (0 to M-1 for each context)
            prompt_text = ""

            # MODIFIED TO MATCH NON-BATCHED MODE
            # Add text anchor BEFORE main video
            prompt_text += "Here is a main video to remember:"
            # Add image tokens for main video (starting from 0)
            for i in range(len(main_frames)):
                prompt_text += f'<|image_{i}|>'

            # Add text anchor BEFORE candidate clip
            prompt_text += "\n\nHere is a candidate clip:\n"
            # Add image tokens for candidate clip (continuing from len(main_frames))
            for i in range(len(candidate_frames)):
                prompt_text += f'<|image_{len(main_frames) + i}|>'
            prompt_text += "\n\n"

            # Add question
            prompt_text += ctx['question_text']

            # Apply chat template
            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        # Now execute batched inference with all frames and texts
        batch_size = len(contexts)
        all_responses: List[Optional[str]] = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Determine which frames this chunk needs
                    # Calculate frames needed for THIS chunk (from chunk_start to chunk_end)
                    frames_before_chunk = 0
                    frames_in_chunk = 0

                    # Count frames before this chunk
                    for i in range(chunk_start):
                        ctx = contexts[i]
                        main_len = min(len(ctx.get('main_video_frames', [])), max_frames_in_video // 2)
                        cand_len = min(len(ctx.get('candidate_video_frames', [])), max_frames_in_video // 2)
                        frames_before_chunk += main_len + cand_len

                    # Count frames in this chunk
                    for i in range(chunk_start, chunk_end):
                        ctx = contexts[i]
                        main_len = min(len(ctx.get('main_video_frames', [])), max_frames_in_video // 2)
                        cand_len = min(len(ctx.get('candidate_video_frames', [])), max_frames_in_video // 2)
                        frames_in_chunk += main_len + cand_len

                    # Extract frames for this chunk
                    chunk_frames = all_frames[frames_before_chunk:frames_before_chunk + frames_in_chunk]

                    processor_kwargs = {
                        'text': chunk_texts,
                        'images': chunk_frames,
                        'return_tensors': 'pt',
                        'padding': True,
                    }

                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    generation_args = {
                        'max_new_tokens': max(1, int(max_tokens)),
                        'temperature': self.temperature,
                        'do_sample': True,
                    }

                    generation = self.model.generate(
                        **inputs,
                        **generation_args,
                        generation_config=self.generation_config,
                    )

                    if isinstance(generation, tuple):
                        generate_ids = generation[0]
                    else:
                        generate_ids = generation

                    input_len = inputs['input_ids'].shape[1]
                    generate_ids = generate_ids[:, input_len:]

                    chunk_responses = self.processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, resp in enumerate(chunk_responses):
                        all_responses[chunk_start + i] = resp

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[Phi-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        return all_responses

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
                frames_obj = segment['frames']
                if torch.is_tensor(frames_obj):
                    frames_cpu = frames_obj.detach().cpu().clone()
                elif isinstance(frames_obj, list):
                    frames_cpu = [self._clone_frame(frame) for frame in frames_obj]
                else:
                    frames_cpu = frames_obj
                saved_segment = {
                    'frames': frames_cpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames'],
                    'state_floats': segment.get('state_floats'),
                    'vision_stats': copy.deepcopy(segment.get('vision_stats', {})),
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
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else next(self.model.parameters()).device

        # Restore video segments and move tensors back to model device
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                # Move tensor back to model device
                if torch.is_tensor(segment['frames']):
                    frames_gpu = segment['frames'].to(model_device)
                else:
                    frames_gpu = segment['frames']

                restored_segment = {
                    'frames': frames_gpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames'],
                    'state_floats': segment.get('state_floats'),
                    'vision_stats': copy.deepcopy(segment.get('vision_stats', {})),
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

    def _normalize_video_frames(self, video_frames: Any) -> List[Image.Image]:
        raw_frames: List[Any]
        if isinstance(video_frames, list):
            raw_frames = [self._clone_frame(frame) for frame in video_frames]
        elif torch.is_tensor(video_frames):
            tensor = video_frames.detach().cpu()
            if tensor.ndim != 4:
                raise ValueError("Expected 4D tensor for video frames (frames, C, H, W)")
            raw_frames = [tensor[i].clone() for i in range(tensor.shape[0])]
        elif isinstance(video_frames, np.ndarray):
            array = np.array(video_frames, copy=True)
            if array.ndim != 4:
                raise ValueError("Expected 4D ndarray for video frames")
            raw_frames = [np.array(array[i], copy=True) for i in range(array.shape[0])]
        else:
            raise TypeError("Unsupported video_frames type for PhiMultimodalVideo")

        pil_frames: List[Image.Image] = []
        for frame in raw_frames:
            pil_frames.append(self._frame_to_pil(frame))
        return pil_frames

    @staticmethod
    def _clone_frame(frame: Any) -> Any:
        if isinstance(frame, Image.Image):
            return frame.copy()
        if torch.is_tensor(frame):
            return frame.detach().cpu().clone()
        if isinstance(frame, np.ndarray):
            return np.array(frame, copy=True)
        return frame

    def _frame_to_pil(self, frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.copy()
        if torch.is_tensor(frame):
            array = frame.detach().cpu().numpy()
        elif isinstance(frame, np.ndarray):
            array = np.array(frame)
        else:
            raise TypeError("Unsupported frame type for PhiMultimodalVideo")

        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] not in (1, 3):
            raise ValueError("Expected frame data with channel dimension")
        return Image.fromarray(array)

    @staticmethod
    def _infer_frame_dimensions(frame: Any) -> Tuple[int, int]:
        if isinstance(frame, Image.Image):
            width, height = frame.size
            return int(height), int(width)
        if torch.is_tensor(frame):
            tensor = frame.detach().cpu()
            if tensor.ndim == 3:
                channels, h, w = tensor.shape
                if channels in (1, 3):
                    return int(h), int(w)
                return int(channels), int(h)
            if tensor.ndim >= 4:
                return int(tensor.shape[-2]), int(tensor.shape[-1])
            return 0, 0
        if isinstance(frame, np.ndarray):
            array = np.array(frame)
            if array.ndim == 3:
                if array.shape[0] in (1, 3):
                    return int(array.shape[1]), int(array.shape[2])
                return int(array.shape[0]), int(array.shape[1])
            if array.ndim >= 4:
                return int(array.shape[-2]), int(array.shape[-1])
            return 0, 0
        return 0, 0

    def _compute_frame_float_total(self, frames: List[Any]) -> float:
        total = 0.0
        for frame in frames or []:
            if torch.is_tensor(frame):
                total += float(frame.numel())
            elif isinstance(frame, np.ndarray):
                total += float(frame.size)
            elif isinstance(frame, Image.Image):
                array_view = np.asarray(frame)
                total += float(array_view.size)
        return total

    def _summarize_video_dimensions(self, frames: List[Any]) -> Tuple[int, int, int]:
        if not frames:
            return 0, 0, 0
        height, width = self._infer_frame_dimensions(frames[0])
        return len(frames), height, width

    def _get_state_memory_floats(self) -> float:
        """Calculates the total memory usage of the stored state."""
        total_floats = 0.0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                segment_total = segment.get('state_floats')
                if segment_total is not None:
                    total_floats += float(segment_total)
                else:
                    frames = segment.get('frames', [])
                    total_floats += self._compute_frame_float_total(frames)

        for text, _ in self.text_entries:
            total_floats += float(self._count_text_tokens(text))
        return total_floats
    
