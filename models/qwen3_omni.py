"""
Qwen3-Omni-30B-A3B-Instruct model implementation for video-language tasks.
This model supports text, video, and audio inputs with text and audio outputs.
"""

import json
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import time
import numpy as np
from models.base_interface import VideoLanguageModelInterface
import torch
from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from huggingface_hub import snapshot_download
import gc
import sys
import os
import copy
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics.flops_calc import qwen3_omni_30b_flops
from utils.paths import get_model_cache_dir


_MODEL_CACHE_SUBDIR = "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"


def _latest_snapshot(path: Path) -> Optional[Path]:
    """Return the newest snapshot directory if it exists."""

    if not path.exists() or not path.is_dir():
        return None

    try:
        candidates = [candidate for candidate in path.iterdir() if candidate.is_dir()]
    except FileNotFoundError:
        return None

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _required_weight_files(snapshot_dir: Path) -> Optional[set[str]]:
    """Return the set of expected weight shard filenames from index metadata."""

    index_files = list(snapshot_dir.glob("*.safetensors.index.json"))
    if not index_files:
        index_files = list(snapshot_dir.glob("**/*.safetensors.index.json"))
    if not index_files:
        index_files = list(snapshot_dir.glob("*.bin.index.json"))
    if not index_files:
        index_files = list(snapshot_dir.glob("**/*.bin.index.json"))

    if not index_files:
        return None

    expected_files: set[str] = set()
    for index_path in index_files:
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        weight_map = data.get("weight_map", {})
        if isinstance(weight_map, dict):
            expected_files.update(weight_map.values())

        shard_filenames = data.get("weight_files")
        if isinstance(shard_filenames, list):
            expected_files.update(shard_filenames)

    return expected_files if expected_files else None


def _snapshot_has_complete_weights(snapshot_dir: Path) -> bool:
    """Return True when required weight shards are present and complete."""

    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return False

    expected_files = _required_weight_files(snapshot_dir)
    if expected_files:
        for filename in expected_files:
            if not (snapshot_dir / filename).exists():
                return False
        return True

    # Non-indexed checkpoints fall back to single-file heuristics.
    if (snapshot_dir / "model.safetensors").exists():
        return True
    if (snapshot_dir / "pytorch_model.bin").exists():
        return True

    # If we only see shard-style files without an index, treat as incomplete so
    # we trigger a resync that fetches the index as well.
    shard_patterns = list(snapshot_dir.glob("model-*-of-*.safetensors"))
    shard_patterns += list(snapshot_dir.glob("pytorch_model-*-of-*.bin"))
    if shard_patterns:
        return False

    return False


def _disable_audio_output(config):
    """Disable talker/audio branches when the config exposes them."""

    if getattr(config, "enable_audio_output", False):
        config.enable_audio_output = False
        # Keep the nested configs so Hugging Face composite models do not
        # break when they attempt to propagate dtype or other attributes
        # across `config.sub_configs`. A few Transformers versions expect
        # every sub-config listed there to be non-null.
    return config

def _resolve_local_snapshot(cache_dir: str, model_id: str) -> Path:
    """Ensure a local snapshot exists for the given model and return its path."""

    repo_dir = Path(cache_dir) / _MODEL_CACHE_SUBDIR
    snapshots_dir = repo_dir / "snapshots"

    local_snapshot_dir = _latest_snapshot(snapshots_dir)

    if local_snapshot_dir is None or not _snapshot_has_complete_weights(local_snapshot_dir):
        allow_patterns = [
            "config.json",
            "**/config.json",
            "*.config.json",
            "vision_config.json",
            "**/vision_config.json",
            "text_config.json",
            "**/text_config.json",
            "processing_config.json",
            "**/processing_config.json",
            "tokenizer_config.json",
            "**/tokenizer_config.json",
            "*.index.json",
            "**/*.index.json",
            "*.py",
            "**/*.py",
            "*.safetensors",
            "**/*.safetensors",
            "*.bin",
            "**/*.bin",
        ]

        downloaded_dir = Path(
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True,
                allow_patterns=allow_patterns,
            )
        )

        if not _snapshot_has_complete_weights(downloaded_dir):
            raise RuntimeError(
                "Downloaded snapshot for "
                f"{model_id} is missing weight shards. Confirm Hugging Face access permissions."
            )

        local_snapshot_dir = downloaded_dir

    return local_snapshot_dir


class Qwen3Omni(VideoLanguageModelInterface):
    """
    Qwen3-Omni-30B-A3B-Instruct model interface for multimodal video-language tasks.

    This model supports text, video, and audio inputs with text and audio outputs.
    Uses the official Qwen3-Omni transformers implementation.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        """Initialize Qwen3Omni with proper base class initialization."""
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Faster model initialization for Qwen3-Omni using Accelerate.
        Streams checkpoint shards directly to GPUs with balanced placement.
        Works on 2–3x A40 or larger setups.
        """

        cache_dir = kwargs.pop("cache_dir", get_model_cache_dir())
        dtype_spec = kwargs.pop("dtype", "auto")
        processor_use_fast = kwargs.pop("use_fast_processor", False)

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-Omni requires CUDA for FLOPs harnesses.")

        dtype = self._resolve_torch_dtype(dtype_spec)

        # ---- pick latest snapshot dir (local cache)
        local_snapshot_dir = _resolve_local_snapshot(cache_dir, self.model_id)
        local_ckpt_dir = str(local_snapshot_dir)

        config = AutoConfig.from_pretrained(
            local_ckpt_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        if not hasattr(config, "initializer_range") or config.initializer_range is None:
            fallback = getattr(config, "initializer_factor", 0.02)
            try:
                config.initializer_range = float(fallback)
            except (TypeError, ValueError):
                config.initializer_range = 0.02
        config = _disable_audio_output(config)

        # Pure GPU placement; no CPU/disk offloading (offload dirs were never cleaned up
        # and contradicted the device_map="auto" GPU sharding strategy used here).
        load_kwargs = {
            "config": config,
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            local_ckpt_dir,
            **load_kwargs,
        )

        self.model.eval()
        self.model.requires_grad_(False)

        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                cache_dir=cache_dir,
                use_fast=processor_use_fast,
            )
        except (ImportError, AttributeError):
            if processor_use_fast:
                raise
            # Installed transformers build lacks the fast processor; fall back to the slow variant.
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                cache_dir=cache_dir,
                use_fast=False,
            )

        # Set left-padding for decoder-only batched generation.
        # `self.tokenizer` is a property that returns `self.processor.tokenizer`,
        # so the second hasattr/assignment block was redundant.
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()

    @property
    def tokenizer(self):
        """Access the tokenizer from the processor."""
        return self.processor.tokenizer

    def add_video(self, video_frames: np.ndarray, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """
        Add video frames to the model's context.

        Args:
            video_frames: Video frames as numpy array with shape (num_frames, 3, height, width)
            time_start: The time the video frames starts. Must be after last was added
            time_end: The time the video frames end. Must be after last were added
            video_id: Optional identifier for the video (defaults to 0 if not provided)
        """
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")

        self.latest_time = time_end

        if isinstance(video_frames, np.ndarray):
            video_frames = torch.from_numpy(video_frames)

        # Keep frames on CPU to avoid early massive GPU transfers
        video_frames = video_frames.clone()

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

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
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
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

        Args:
            text: Text string to add to context
            current_video_time: Current video time (unused in this implementation)
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        self.text_entries.append((text, self.latest_time))

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            token_count = self._count_text_tokens(text)
            add_text_flops = 0.0

            state_mem = self._get_state_memory_floats()
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            self._record_add_text_metrics(
                latency,
                add_text_flops,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )

    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256,
                    max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """
        Ask a question based on the current context.

        Args:
            question: Question to ask
            current_video_time: Current video time when question is asked
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")

        Returns:
            Generated response as string
        """
        self._video_was_truncated = False  # Reset flag
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

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

        content = []
        video_inputs = []
        processor_videos = []
        processor_video_fps = []
        video_data = {}

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']

                if video_id not in video_data:
                    video_data[video_id] = {'frames': None, 'total_duration': 0.0, 'segments': []}

                video_data[video_id]['segments'].append(segment)
                video_data[video_id]['total_duration'] += segment['duration']

        for video_id in video_data:
            segments = video_data[video_id]['segments']
            total_frames = sum(seg['frames'].shape[0] for seg in segments)

            if total_frames <= max_frames_in_video:
                frame_slices = [seg['frames'] for seg in segments]
            else:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    frame_slices = []
                    for segment in segments:
                        seg_frames = segment['frames']
                        proportion = seg_frames.shape[0] / total_frames
                        target_frames = max(1, int(max_frames_in_video * proportion))

                        if seg_frames.shape[0] <= target_frames:
                            frame_slices.append(seg_frames)
                        else:
                            indices = np.random.choice(seg_frames.shape[0], target_frames, replace=False)
                            indices = np.sort(indices)
                            frame_slices.append(seg_frames[indices])
                else:
                    # Sample evenly across the full timeline without materialising every frame
                    indices = np.linspace(0, total_frames - 1, max_frames_in_video, dtype=int)
                    frame_slices = []
                    offset = 0
                    for segment in segments:
                        seg_frames = segment['frames']
                        seg_len = seg_frames.shape[0]
                        mask = (indices >= offset) & (indices < offset + seg_len)
                        if np.any(mask):
                            local_indices = indices[mask] - offset
                            frame_slices.append(seg_frames[local_indices])
                        offset += seg_len

            if frame_slices:
                video_data[video_id]['frames'] = torch.cat(frame_slices, dim=0)
            else:
                first_segment = segments[0]['frames']
                empty = first_segment.new_empty((0,) + first_segment.shape[1:])
                video_data[video_id]['frames'] = empty

            # Adjust duration proportionally to maintain consistent FPS after subsampling
            final_frame_count = video_data[video_id]['frames'].shape[0]
            if final_frame_count != total_frames and total_frames > 0:
                subsample_ratio = final_frame_count / total_frames
                video_data[video_id]['total_duration'] *= subsample_ratio

            del video_data[video_id]['segments']

        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    fps_value = self._compute_video_fps(
                        frames_tensor.shape[0],
                        video_data[video_id]['total_duration'],
                    )

                    content.append({"type": "video"})
                    video_inputs.append(frames_tensor)
                    processor_videos.append(self._prepare_video_for_processor(frames_tensor))
                    processor_video_fps.append(fps_value)
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                content.append({"type": "text", "text": event['text']})

        content.append({
            "type": "text",
            "text": question
        })

        conversation = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Print raw prompt for debugging
        print("\n[model-input] ===== RAW CHAT PROMPT START (Qwen Omni) =====")
        print(text)
        print("[model-input] ===== RAW CHAT PROMPT END (Qwen Omni) =====", flush=True)

        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
            "return_attention_mask": True,
        }
        if processor_videos:
            processor_kwargs["videos"] = processor_videos
            fps_setting = processor_video_fps[0] if len(processor_video_fps) == 1 else float(
                sum(processor_video_fps) / len(processor_video_fps)
            )
            processor_kwargs["videos_kwargs"] = {
                "fps": fps_setting,
                "use_audio_in_video": False,
            }

        inputs = self.processor(**processor_kwargs)

        # Ensure multimodal tensors land on the same CUDA device and align float dtypes with the model.
        target_param = next(self.model.parameters())
        target_device = target_param.device
        target_dtype = target_param.dtype
        inputs = inputs.to(device=target_device, non_blocking=True)

        def _cast_to_model_dtype(obj):
            if torch.is_tensor(obj):
                return obj.to(dtype=target_dtype) if obj.is_floating_point() and obj.dtype != target_dtype else obj
            if isinstance(obj, dict):
                return {k: _cast_to_model_dtype(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cast_to_model_dtype(v) for v in obj]
            return obj

        for key in list(inputs.keys()):
            inputs[key] = _cast_to_model_dtype(inputs[key])

        generation_kwargs = {}
        max_new = max(1, int(max_tokens))
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            cloned_config = copy.deepcopy(generation_config)
            cloned_config.max_new_tokens = max_new
            generation_kwargs["generation_config"] = cloned_config
        generation_kwargs["max_new_tokens"] = max_new

        # Qwen Omni splits reasoning (thinker) and response (talker) generation. Only
        # talker_max_new_tokens bounds the final text output, so wire both stage-specific
        # kwargs when the model exposes them.
        stage_specific_limits = {}
        try:
            generate_signature = inspect.signature(self.model.generate)
            parameters = generate_signature.parameters
        except (ValueError, TypeError):
            parameters = {}

        if "talker_max_new_tokens" in parameters:
            stage_specific_limits["talker_max_new_tokens"] = max_new
        if "thinker_max_new_tokens" in parameters:
            stage_specific_limits.setdefault("thinker_max_new_tokens", max_new)

        generation_kwargs.update(stage_specific_limits)

        # Clear GPU cache before generation to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            generation = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=False,
                **generation_kwargs,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "bad_alloc" in str(e).lower():
                # Clean up and provide helpful error message
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"CUDA OOM during Qwen3-Omni generation. "
                    f"Frames: {sum(v.shape[0] for v in video_inputs) if video_inputs else 0}, "
                    f"Try reducing max_frames_in_video (currently {max_frames_in_video}). "
                    f"Original error: {e}"
                ) from e
            raise

        if isinstance(generation, tuple):
            text_result = generation[0]
        else:
            text_result = generation

        sequences_tensor = getattr(text_result, "sequences", None)

        # Save prompt_length before deleting inputs (needed for metrics later)
        prompt_length = int(inputs["input_ids"].shape[1])

        if sequences_tensor is not None:
            generated_text = self.processor.batch_decode(
                sequences_tensor[:, prompt_length :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        else:
            if isinstance(text_result, str):
                generated_text = [text_result]
            elif isinstance(text_result, (list, tuple)):
                generated_text = [str(item) for item in text_result]
            else:
                generated_text = [str(text_result)]

        # Print raw response for debugging
        result = generated_text[0] if generated_text else ""
        print("\n[model-output] ===== RAW MODEL RESPONSE START (Qwen Omni) =====")
        print(result)
        print("[model-output] ===== RAW MODEL RESPONSE END (Qwen Omni) =====", flush=True)

        # Clear large intermediate tensors to free memory
        del inputs
        del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if video_inputs:
                total_frames = sum([v.shape[0] for v in video_inputs])
                vision_height = video_inputs[0].shape[2]
                vision_width = video_inputs[0].shape[3]
            else:
                total_frames = 0
                vision_height = 0
                vision_width = 0

            # prompt_length already calculated before deleting inputs (line ~618)
            if sequences_tensor is not None:
                generated_tokens = max(0, sequences_tensor.shape[1] - prompt_length)
            else:
                generated_tokens = 0
                tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is not None and generated_text:
                    tokenized = tokenizer(
                        generated_text,
                        add_special_tokens=False,
                        padding=False,
                        return_attention_mask=False,
                    )
                    input_ids = tokenized.get("input_ids")
                    if input_ids:
                        generated_tokens = len(input_ids[0])
            flops_breakdown = qwen3_omni_30b_flops(
                vision_frames=total_frames,
                vision_height=vision_height,
                vision_width=vision_width,
                lang_prompt_len=prompt_length,
                num_generated=generated_tokens,
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
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            self._record_ask_question_metrics(
                latency,
                float(flops_breakdown["total_flops"] if isinstance(flops_breakdown, dict) else flops_breakdown),
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
            )

        result = generated_text[0] if generated_text else ""
        return result

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
        Qwen3Omni implementation with video support and OOM retry (halves batch size on OOM).
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build timeline (shared for all questions)
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

        # Build shared content with video support
        shared_content = []
        processor_videos = []
        processor_video_fps = []
        video_data = {}

        # First pass: collect video frames
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']

                if video_id not in video_data:
                    video_data[video_id] = {'frames': None, 'total_duration': 0.0, 'segments': []}

                video_data[video_id]['segments'].append(segment)
                video_data[video_id]['total_duration'] += segment['duration']

        # Apply frame limits
        for video_id in video_data:
            segments = video_data[video_id]['segments']
            total_frames = sum(seg['frames'].shape[0] for seg in segments)

            if total_frames <= max_frames_in_video:
                frame_slices = [seg['frames'] for seg in segments]
            else:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    frame_slices = []
                    for segment in segments:
                        seg_frames = segment['frames']
                        proportion = seg_frames.shape[0] / total_frames
                        target_frames = max(1, int(max_frames_in_video * proportion))

                        if seg_frames.shape[0] <= target_frames:
                            frame_slices.append(seg_frames)
                        else:
                            indices = np.random.choice(seg_frames.shape[0], target_frames, replace=False)
                            indices = np.sort(indices)
                            frame_slices.append(seg_frames[indices])
                else:
                    indices = np.linspace(0, total_frames - 1, max_frames_in_video, dtype=int)
                    frame_slices = []
                    offset = 0
                    for segment in segments:
                        seg_frames = segment['frames']
                        seg_len = seg_frames.shape[0]
                        mask = (indices >= offset) & (indices < offset + seg_len)
                        if np.any(mask):
                            local_indices = indices[mask] - offset
                            frame_slices.append(seg_frames[local_indices])
                        offset += seg_len

            if frame_slices:
                video_data[video_id]['frames'] = torch.cat(frame_slices, dim=0)
            else:
                first_segment = segments[0]['frames']
                empty = first_segment.new_empty((0,) + first_segment.shape[1:])
                video_data[video_id]['frames'] = empty

            # Adjust duration proportionally
            final_frame_count = video_data[video_id]['frames'].shape[0]
            if final_frame_count != total_frames and total_frames > 0:
                subsample_ratio = final_frame_count / total_frames
                video_data[video_id]['total_duration'] *= subsample_ratio

            del video_data[video_id]['segments']

        # Build shared content
        processed_video_ids = set()
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    fps_value = self._compute_video_fps(
                        frames_tensor.shape[0],
                        video_data[video_id]['total_duration'],
                    )

                    shared_content.append({"type": "video"})
                    processor_videos.append(self._prepare_video_for_processor(frames_tensor))
                    processor_video_fps.append(fps_value)
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                shared_content.append({"type": "text", "text": event['text']})

        # Build batch of conversations
        batch_conversations = []
        for question_text in questions:
            content = list(shared_content)
            content.append({"type": "text", "text": question_text})
            batch_conversations.append([{"role": "user", "content": content}])

        # Apply chat template to batch
        batch_texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in batch_conversations
        ]

        # Get model device and dtype
        target_param = next(self.model.parameters())
        model_device = target_param.device
        target_dtype = target_param.dtype

        def _cast_to_model_dtype(obj):
            if torch.is_tensor(obj):
                return obj.to(dtype=target_dtype) if obj.is_floating_point() and obj.dtype != target_dtype else obj
            if isinstance(obj, dict):
                return {k: _cast_to_model_dtype(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cast_to_model_dtype(v) for v in obj]
            return obj

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
                        "return_attention_mask": True,
                    }
                    if processor_videos:
                        # Qwen3-Omni processor expects nested lists for batched videos:
                        # videos = [[video1, video2], [video1, video2], ...]  # one list per text
                        chunk_size = len(chunk_texts)
                        processor_kwargs["videos"] = [processor_videos for _ in range(chunk_size)]
                        fps_setting = processor_video_fps[0] if len(processor_video_fps) == 1 else float(
                            sum(processor_video_fps) / len(processor_video_fps)
                        )
                        processor_kwargs["videos_kwargs"] = {
                            "fps": fps_setting,
                            "use_audio_in_video": False,
                        }

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(device=model_device, non_blocking=True)

                    for key in list(inputs.keys()):
                        inputs[key] = _cast_to_model_dtype(inputs[key])

                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max(1, int(max_tokens)),
                        thinker_max_new_tokens=max(1, int(max_tokens)),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    # Handle different output formats
                    if hasattr(generation, 'sequences'):
                        output_ids = generation.sequences
                    else:
                        output_ids = generation

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
                    ]

                    chunk_outputs = self.tokenizer.batch_decode(
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
                    print(f"[Qwen3Omni] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Reset results for retry
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        generated_texts = all_output_texts

        # Debug output
        if generated_texts:
            print(f"\n[model-output-batch] ===== BATCHED RESPONSE Qwen3Omni ({len(generated_texts)} questions, batch_size={batch_size}) =====")
            for i, text in enumerate(generated_texts[:3]):
                print(f"Q{i+1}: {text}")
            if len(generated_texts) > 3:
                print(f"... and {len(generated_texts)-3} more")
            print("[model-output-batch] ===== END =====", flush=True)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time

        return generated_texts

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
        # That path has different FPS/generate kwargs from the batch path.
        # Always use the batch code path for consistency.

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build separate conversations for each question
        conversations = []
        all_videos_per_question = []
        all_fps_values = []

        for ctx in contexts:
            if is_sequence_mode:
                # Sequence mode: pure text
                prompt = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                content = [{"type": "text", "text": prompt}]
                conversations.append([{"role": "user", "content": content}])
                all_videos_per_question.append(None)
                all_fps_values.append(None)

            else:
                # Video mode: build separate video list for this question
                main_frames = ctx.get('main_video_frames')
                candidate_frames = ctx.get('candidate_video_frames')

                videos_for_question = []
                total_frames = 0

                # Add main video
                if main_frames is not None and main_frames.shape[0] > 0:
                    # Sample to half of max_frames budget
                    if main_frames.shape[0] > max_frames_in_video // 2:
                        indices = np.linspace(0, main_frames.shape[0] - 1, max_frames_in_video // 2, dtype=int)
                        sampled_main = main_frames[indices]
                    else:
                        sampled_main = main_frames

                    main_prepared = self._prepare_video_for_processor(sampled_main)
                    videos_for_question.append(main_prepared)
                    total_frames += sampled_main.shape[0]

                # Add candidate clip
                if candidate_frames is not None and candidate_frames.shape[0] > 0:
                    # Sample to remaining budget
                    remaining = max_frames_in_video - total_frames
                    if candidate_frames.shape[0] > remaining:
                        indices = np.linspace(0, candidate_frames.shape[0] - 1, max(1, remaining), dtype=int)
                        sampled_cand = candidate_frames[indices]
                    else:
                        sampled_cand = candidate_frames

                    cand_prepared = self._prepare_video_for_processor(sampled_cand)
                    videos_for_question.append(cand_prepared)
                    total_frames += sampled_cand.shape[0]

                all_videos_per_question.append(videos_for_question)

                # Compute FPS
                fps_value = 1.0  # All videos use fps=1.0
                all_fps_values.append(fps_value)

                # Build content with video placeholders. The number of {"type":"video"}
                # entries must equal the number of videos in videos_for_question; the
                # natural-video benchmark sends only a main video, so the candidate-clip
                # placeholder must be omitted when there is no candidate.
                content = [
                    {"type": "text", "text": "Here is a main video to remember:"},
                    {"type": "video"},  # Main video
                ]
                if candidate_frames is not None and candidate_frames.shape[0] > 0:
                    content += [
                        {"type": "text", "text": "\nHere is a candidate clip:\n"},
                        {"type": "video"},  # Candidate clip
                    ]
                content.append({"type": "text", "text": ctx['question_text']})
                conversations.append([{"role": "user", "content": content}])

        # Apply chat template
        batch_texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        # Get device and dtype
        target_param = next(self.model.parameters())
        model_device = target_param.device
        target_dtype = target_param.dtype

        def _cast_to_model_dtype(obj):
            if torch.is_tensor(obj):
                return obj.to(dtype=target_dtype) if obj.is_floating_point() and obj.dtype != target_dtype else obj
            if isinstance(obj, dict):
                return {k: _cast_to_model_dtype(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cast_to_model_dtype(v) for v in obj]
            return obj

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
                        "return_attention_mask": True,
                    }

                    if not is_sequence_mode:
                        # Video mode: slice video lists for this chunk
                        chunk_videos = all_videos_per_question[chunk_start:chunk_end]
                        # Filter out None entries
                        valid_videos = [v for v in chunk_videos if v is not None]

                        if valid_videos:
                            processor_kwargs["videos"] = valid_videos
                            # Use average FPS from chunk
                            chunk_fps = [fps for fps in all_fps_values[chunk_start:chunk_end] if fps is not None]
                            avg_fps = sum(chunk_fps) / len(chunk_fps) if chunk_fps else 1.0
                            processor_kwargs["videos_kwargs"] = {
                                "fps": avg_fps,
                                "use_audio_in_video": False
                            }

                    # Tokenize and prepare inputs
                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(device=model_device, non_blocking=True)

                    # Cast to model dtype
                    for key in list(inputs.keys()):
                        inputs[key] = _cast_to_model_dtype(inputs[key])

                    # Generate — must pass thinker_max_new_tokens explicitly;
                    # the top-level max_new_tokens is blocked by thinker_kwargs
                    # default of 1024 (see Qwen3OmniMoe.generate() line ~3917)
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max(1, int(max_tokens)),
                        thinker_max_new_tokens=max(1, int(max_tokens)),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    # Handle output format
                    if hasattr(generation, 'sequences'):
                        output_ids = generation.sequences
                    else:
                        output_ids = generation

                    # Trim prompt tokens
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
                    ]

                    # Decode
                    chunk_outputs = self.tokenizer.batch_decode(
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
                    print(f"[Qwen3Omni-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE Qwen3Omni "
              f"({len(all_output_texts)} questions, batch_size={batch_size}, mode={mode}) =====")
        for i, text in enumerate(all_output_texts[:3]):
            preview = text[:100] if text else "(empty)"
            print(f"Q{i+1}: {preview}...")
        if len(all_output_texts) > 3:
            print(f"... and {len(all_output_texts)-3} more")
        print("[model-output-batch-isolated] ===== END =====", flush=True)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            # Could add metrics tracking here if needed

        return all_output_texts

    def _resolve_torch_dtype(self, dtype_spec: Any) -> torch.dtype:
        """Normalize dtype specifications to a concrete torch dtype."""
        if isinstance(dtype_spec, torch.dtype):
            return dtype_spec

        if dtype_spec is None or (isinstance(dtype_spec, str) and dtype_spec.lower() == "auto"):
            return self._default_torch_dtype()

        if isinstance(dtype_spec, str):
            normalized = dtype_spec.lower()
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if normalized in mapping:
                return mapping[normalized]
            raise ValueError(f"Unsupported dtype specification: {dtype_spec}")

        raise TypeError(f"dtype must be a torch.dtype or string identifier, received {type(dtype_spec)}")

    def _default_torch_dtype(self) -> torch.dtype:
        """Pick a performant default dtype for the available accelerator."""
        if torch.cuda.is_available():
            capability_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            if capability_major >= 8:
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _prepare_video_for_processor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Move video tensor to CPU for processor consumption without upcasting."""
        if not torch.is_tensor(video_tensor):
            raise TypeError("Expected video tensor input for processor preparation")

        prepared = video_tensor.detach()
        if prepared.device.type != "cpu":
            prepared = prepared.to("cpu")

        # Keep original dtype (uint8/float); avoid 4× size blow-up
        return prepared.contiguous()


    @staticmethod
    def _compute_video_fps(num_frames: int, duration_seconds: float) -> float:
        """Compute an FPS value for processor kwargs, defaulting to frame count when duration is tiny."""
        if num_frames <= 0:
            return 1.0

        if duration_seconds is None or duration_seconds <= 1e-6:
            return float(num_frames)

        return max(1.0, float(num_frames) / duration_seconds)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the model."""
        timeline = []

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

        for text, timestamp in self.text_entries:
            timeline.append({
                'type': 'text',
                'timestamp': timestamp,
                'text': text
            })

        timeline.sort(key=lambda x: (x['timestamp'], x['type'] == 'video'))

        return {
            'video_segments': dict(self.video_segments),
            'text_entries': list(self.text_entries),
            'latest_time': self.latest_time,
            'timeline': timeline
        }

    def clear_context(self) -> None:
        """Clear all context (video, text, audio) from the model."""
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                if 'frames' in segment and torch.is_tensor(segment['frames']):
                    del segment['frames']

        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        """Save the current model state to memory."""
        saved_video_segments = {}
        for video_id, segments in self.video_segments.items():
            saved_segments = []
            for segment in segments:
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
        """Load a previously saved model state."""
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                if 'frames' in segment and torch.is_tensor(segment['frames']):
                    del segment['frames']

        # Keep restored frames on CPU; avoid big device transfers at load time
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                frames_cpu = segment['frames'] if torch.is_tensor(segment['frames']) else segment['frames']
                if torch.is_tensor(frames_cpu):
                    frames_cpu = frames_cpu.clone()  # stay on CPU
                restored_segment = {
                    'frames': frames_cpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                }
                restored_segments.append(restored_segment)
            self.video_segments[video_id] = restored_segments

        self.text_entries = state['text_entries'].copy()
        self.latest_time = state['latest_time']

        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
            from models.base_interface import PerformanceMetrics

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


    def _get_state_memory_floats(self) -> float:
        """Calculate total memory usage of stored state."""
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
        if getattr(self, '_tokenizer_ref', None) is None:
            tokenizer = getattr(self.processor, 'tokenizer', None)
            if tokenizer is None:
                raise RuntimeError('Processor does not expose a tokenizer; cannot measure text state size.')
            self._tokenizer_ref = tokenizer
        return self._tokenizer_ref

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cache = getattr(self, '_text_token_cache', None)
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
        input_ids = encoding.get('input_ids') if isinstance(encoding, dict) else encoding.input_ids
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                count = len(input_ids[0])
            else:
                count = len(input_ids)
        elif hasattr(input_ids, '__len__'):
            count = len(input_ids)
        else:
            shape = getattr(input_ids, 'shape', None)
            count = int(shape[0]) if shape else 0
        cache[text] = count
        return count
