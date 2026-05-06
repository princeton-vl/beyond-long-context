#!/usr/bin/env python3
"""
Video multiple choice evaluation script for multi-question per video format.
Loads videos from JSON, presents main video + options, evaluates model responses.
"""

import sys
import os
import argparse
import atexit
import gc
import math
import copy
import logging
import hashlib
import random
import signal
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence, Set

from datasets.patternvideos_manifest import (
    QuestionEntry,
    VideoEntry,
    load_patternvideos_manifest,
)

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'frame_samplers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'metrics'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'processors'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Thermal throttling hook (disabled - the previous stub always returned False).
from gpu_monitor import GPUMonitor


import numpy as np
import torch

from processors import (
    VideoProcessor,
    QuestionProcessor,
    TextToVideoProcessor,
    SequenceProcessor,
    SequenceFormatter,
    CommaSeparatedSequenceFormatter,
    SpatialSequenceFormatter,
)
from wandb_logger import WandbLogger
from models.base_interface import PerformanceMetrics
from utils.question_csv_logger import write_question_log_csv


logging.getLogger("httpx").setLevel(logging.WARNING)

# Batching configuration
ENABLE_TEXT_BATCHING = os.environ.get('ENABLE_TEXT_BATCHING', '1') == '1'
ENABLE_VIDEO_BATCHING = os.environ.get('ENABLE_VIDEO_BATCHING', '1') == '1'
DEFAULT_BATCH_SIZE = int(os.environ.get('TEXT_BATCH_SIZE', '16'))  # Increased from 8 to 16 for sequence mode
VIDEO_BATCH_SIZE = int(os.environ.get('VIDEO_BATCH_SIZE', '16'))  # Increased from 8 to 16 for video mode

# Models to exclude from text/sequence batching (use sequential processing instead)
# longvila: Batched text generation produces nonsense outputs unrelated to prompts
# timechat: Has ask_question_batch() but not ask_question_batch_isolated()
# claude-opus*: API model — batched isolated contexts would just be parallel HTTP
# calls; leave as sequential so the harness's retry / backoff is simple.
TEXT_BATCHING_EXCLUDED_MODELS = {
    'longvila', 'timechat', 'm3_agent',
    'claude-opus', 'claude-opus-4-7', 'claude-opus-4-6',
    # OpenRouter API models — sequential calls, no batching benefit.
    'openrouter', 'gemini-3.1', 'gpt-5.5', 'grok-5',
}

# Models to exclude from video batching (use sequential processing instead)
# qwen3_full: all batched video responses identical "{1}" (bug investigation needed)
# minicpm: MiniCPM-2.6 uses streaming API that doesn't support parallel batching
# longvila: Media embedding architecture breaks in batch mode (queue-based fusion incompatible)
# timechat: Has ask_question_batch() but not ask_question_batch_isolated()
# claude-opus*: API model — see TEXT_BATCHING_EXCLUDED_MODELS note.
VIDEO_BATCHING_EXCLUDED_MODELS = {
    'qwen3_full', 'minicpm', 'longvila', 'timechat', 'm3_agent',
    'claude-opus', 'claude-opus-4-7', 'claude-opus-4-6',
    # OpenRouter API models — see TEXT_BATCHING_EXCLUDED_MODELS note.
    'openrouter', 'gemini-3.1', 'gpt-5.5', 'grok-5',
}

# Model-specific video batch sizes (for models that need smaller batches)
MODEL_SPECIFIC_VIDEO_BATCH_SIZE = {
    'mimo-vl': 4,  # MiMo-VL: batch_size=8 hangs at model.generate(), trying 4
    'glm45v': 8,   # GLM-4.5V: custom batch size for video mode
}

# Per-model maximum bucket lengths (VIDEO MODE ONLY - sequence/text mode unaffected)
# Key: model name, Value: max L{N} value (in frames/seconds)
# Note: These limits only apply when input_mode == "video"
# Sequence mode processes all buckets regardless of length (uses text tokens, not video frames)
MODEL_VIDEO_BUCKET_LIMITS = {
    'phi_multimodal': 512,         # Phi-4: up to L512
    'longvila': 256,               # LongVILA: up to L256
    'internvl-3-5': 512,           # InternVL 8B: up to L512
    'internvl-3-5-thinking': 512,  # InternVL 8B thinking: up to L512
}

def group_questions_into_batches(questions: List[QuestionEntry], batch_size: int) -> List[List[QuestionEntry]]:
    """Group questions into batches of specified size."""
    if batch_size <= 0:
        batch_size = 1
    return [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]


class DynamicBatchSizer:
    """Manages batch size with OOM-aware reduction."""

    def __init__(self, initial_size: int = 8, min_size: int = 1, max_size: int = 32):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.oom_count = 0

    def reduce_on_oom(self) -> int:
        """Halve batch size on OOM."""
        self.current_size = max(self.min_size, self.current_size // 2)
        self.oom_count += 1
        return self.current_size

    def increase_on_success(self) -> int:
        """Gradually increase batch size on successful runs."""
        if self.current_size < self.max_size and self.oom_count == 0:
            self.current_size = min(self.max_size, self.current_size + 1)
        return self.current_size

    def get_size(self) -> int:
        return self.current_size


_TERMINATION_REQUESTED = False

def _handle_termination(signum, frame):
    """Handle process termination signals and clean up GPU memory."""
    del frame
    global _TERMINATION_REQUESTED
    _TERMINATION_REQUESTED = True

    # Clean up GPU memory to avoid zombie processes holding GPU resources
    print(f"\nReceived termination signal {signum}, cleaning up GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            # Clear CUDA cache on all devices
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                try:
                    torch.cuda.empty_cache()
                    # Also synchronize to ensure all operations are complete
                    torch.cuda.synchronize(device_id)
                except Exception as e:
                    print(f"  Warning: Could not clean GPU {device_id}: {e}")

            print(f"  GPU memory cleared on {device_count} device(s)")

        # Force garbage collection
        import gc
        gc.collect()
        print("  Garbage collection completed")

    except Exception as e:
        print(f"  Warning: GPU cleanup failed: {e}")

    print("Cleanup complete, exiting...")

def termination_requested() -> bool:
    return _TERMINATION_REQUESTED

# Register handlers for multiple termination signals
signal.signal(signal.SIGTERM, _handle_termination)  # Normal termination (e.g., scancel)
signal.signal(signal.SIGINT, _handle_termination)   # Ctrl+C

# Register cleanup handler for normal exit
def _cleanup_on_exit():
    """Clean up GPU memory when process exits normally."""
    if _TERMINATION_REQUESTED:
        # Already cleaned up via signal handler
        return

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass  # Silent cleanup on normal exit

atexit.register(_cleanup_on_exit)


class RunStateManager:
    def __init__(self, state_path: Optional[str], resume: bool) -> None:
        self.enabled = bool(state_path)
        self.path = Path(state_path) if state_path else None
        self.state: Dict[str, Dict[str, Any]] = {"videos": {}, "question_results": []}
        if not self.enabled:
            return
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
                # Ensure question_results key exists for backwards compatibility
                self.state.setdefault("question_results", [])
            except Exception:
                logging.warning("Failed to load existing state from %s; starting fresh.", self.path)
                self.state = {"videos": {}, "question_results": []}
        elif resume and not self.path.exists():
            self.state = {"videos": {}, "question_results": []}
        else:
            self.state = {"videos": {}, "question_results": []}

    def _video_key(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> str:
        if bucket:
            # Include bucket in key to handle non-unique video indices across buckets
            key = f"{bucket}:{video_index}"
        else:
            key = str(video_index)
        if variant is not None:
            key = f"{key}_v{variant}"
        return key

    def _ensure_entry(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> Dict[str, Any]:
        videos = self.state.setdefault("videos", {})
        key = self._video_key(video_index, variant, bucket)
        entry = videos.get(key)
        if entry is None:
            entry = {"questions": {}, "completed": False}
            videos[key] = entry
        entry.setdefault("questions", {})
        entry.setdefault("completed", False)
        # Store bucket metadata for clarity (backwards compatible)
        entry.setdefault("bucket", bucket)
        entry.setdefault("video_index", video_index)
        entry.setdefault("variant", variant)
        return entry

    def get_completed_questions(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> Set[str]:
        if not self.enabled:
            return set()
        entry = self.state.get("videos", {}).get(self._video_key(video_index, variant, bucket))
        if not entry:
            return set()
        questions = entry.get("questions", {})
        return {qid for qid, status in questions.items() if status}

    def mark_question_complete(self, video_index: int, question_id: str, result: Optional[Dict[str, Any]] = None, variant: Optional[int] = None, bucket: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry = self._ensure_entry(video_index, variant, bucket)
        entry["questions"][question_id] = True
        # Save the question result for resume capability
        if result is not None:
            # Store full response (no truncation)
            response = result.get('response', '')

            # Store complete result data for metrics calculations
            result_data = {
                # Basic identifiers
                'video_id': result.get('video_index'),
                'video_index': video_index,  # Store parameter (more reliable)
                'variant': variant if variant is not None else 0,
                'bucket': bucket,  # Critical for uniquely identifying questions
                'question_id': result.get('question_id'),
                'question_order': result.get('question_order'),
                # Answer data
                'predicted': result.get('predicted'),
                'correct': result.get('correct'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': response,  # Full response, no truncation
                # Metrics for analysis
                'video_entropy': result.get('entropy_prefix_mean'),  # For entropy bucketing
                'entropy_prefix_mean': result.get('entropy_prefix_mean'),  # Keep original key too
                # Video-level entropy (empirical and analytic)
                'entropy_empirical_bits': result.get('entropy_empirical_bits'),
                'entropy_empirical_mean': result.get('entropy_empirical_mean'),
                'entropy_analytic_bits': result.get('entropy_analytic_bits'),
                'entropy_analytic_mean': result.get('entropy_analytic_mean'),
                'prefix_match_fraction': result.get('prefix_match_fraction'),  # For prefix analysis
                'correct_option_likelihood': result.get('correct_option_likelihood'),  # For likelihood analysis
                'num_options': result.get('num_options'),
                # Video/frame info (the "length of video" info you asked about)
                'frames_seen_before_question': result.get('frames_seen_before_question'),
                # Performance metrics
                'latency_ask_question': result.get('latency_ask_question'),
                'peak_gpu_mem_ask_question': result.get('peak_gpu_mem_ask_question'),
                'question_time': result.get('question_time'),
                # Additional context
                'question_mode': result.get('question_mode'),
                'sequence_format': result.get('sequence_format'),
                # Mode tracking
                'video_only_mode': result.get('video_only_mode'),
                'eval_mode': result.get('eval_mode'),
                'input_mode': result.get('input_mode'),
                # Per-question type
                'question_variant': result.get('question_variant'),
                'question_type': result.get('question_type'),
                # Token/frame limits and truncation
                'max_tokens': result.get('max_tokens'),
                'max_frames': result.get('max_frames'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'saw_all_frames': result.get('saw_all_frames'),
            }
            # Check if this result already exists to avoid duplicates
            question_results = self.state.setdefault("question_results", [])

            # Build signature for deduplication
            sig = (
                result_data.get('question_id'),
                result_data.get('video_index'),
                result_data.get('bucket'),
                result_data.get('question_time'),
                result_data.get('entropy_prefix_mean'),
            )

            # Check if already exists
            already_exists = False
            for existing in question_results:
                existing_sig = (
                    existing.get('question_id'),
                    existing.get('video_index'),
                    existing.get('bucket'),
                    existing.get('question_time'),
                    existing.get('entropy_prefix_mean'),
                )
                if sig == existing_sig:
                    already_exists = True
                    break

            if not already_exists:
                question_results.append(result_data)
            # else: silently skip duplicate (question already in results)

        self._save()

    def get_saved_results(self) -> List[Dict[str, Any]]:
        """Get all previously saved question results for resume."""
        if not self.enabled:
            return []
        return self.state.get("question_results", [])

    def mark_video_complete(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry = self._ensure_entry(video_index, variant, bucket)
        entry["completed"] = True
        self._save()

    def is_video_completed(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        entry = self.state.get("videos", {}).get(self._video_key(video_index, variant, bucket))
        return bool(entry and entry.get("completed"))

    def flush(self) -> None:
        if not self.enabled:
            return
        self._save()

    def _save(self) -> None:
        if not self.enabled or self.path is None:
            return
        tmp_path = self.path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp_path, self.path)


def _extract_bucket_from_video_data(video_data: Any) -> Optional[str]:
    """Extract bucket name from video_data.video_path.

    Returns the bucket name (e.g., 'UNIFORM_EVAL_L008_ELOW') or None if not found.
    This is crucial for handling non-unique video indices across buckets.
    """
    path = getattr(video_data, 'video_path', None) or getattr(video_data, 'main_video_path', None)
    if not path:
        return None
    path_str = str(path)
    parts = path_str.split('/')
    for part in parts:
        if 'UNIFORM_EVAL' in part or any(x in part for x in ['L008', 'L016', 'L032', 'L064', 'L128', 'L256', 'L512', 'L1024', 'L2048', 'L4096']):
            return part
    return None


def extract_bucket_length(bucket: str) -> Optional[int]:
    """Extract numeric length from bucket name (e.g., 'L064' -> 64).

    Used for sorting buckets by frame count and checking per-model limits.
    Returns None if bucket name doesn't contain a valid L{N} pattern.
    """
    if not bucket:
        return None
    import re
    match = re.search(r'L(\d+)', bucket)
    if match:
        return int(match.group(1))
    return None


def count_valid_completed_questions(
    completed_set: Set[str],
    video_data: Any,
    eval_mode: str,
) -> int:
    """
    Count how many completed questions are VALID for eval_mode.
    Cross-references completed_set with video_data to filter by question_variant.

    In 'sequential' mode: only counts sequential questions (skips spatial)
    Otherwise: counts all questions
    """
    valid_count = 0
    for i, question_data in enumerate(video_data.questions):
        question_id = question_data.question_id or f"video{video_data.video_index}_q{i}"
        if question_id not in completed_set:
            continue

        # Check if valid for eval_mode
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue  # Skip spatial in sequential mode

        valid_count += 1
    return valid_count


def count_total_valid_questions(
    video_data: Any,
    eval_mode: str,
) -> int:
    """
    Count total questions valid for eval_mode.

    In 'sequential' mode: only counts sequential questions (skips spatial)
    Otherwise: counts all questions
    """
    valid_count = 0
    for question_data in video_data.questions:
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue
        valid_count += 1
    return valid_count


def load_model_class(model_type: str) -> tuple:
    """Dynamically import and return the model class based on model type."""
    if model_type == "qwen_full":
        from qwen2_5_vl import QwenFullVideo
        return QwenFullVideo, "QwenFullVideo"
    elif model_type == "mimo-vl":
        from mimo_vl import MimoVLVideo
        return MimoVLVideo, "MiMo-VL-7B-RL"
    elif model_type == "phi_multimodal":
        from phi_4_mm import PhiMultimodalVideo
        return PhiMultimodalVideo, "Phi-4-multimodal"
    elif model_type == "qwen3_full":
        from qwen3_vl import Qwen3Dense
        return Qwen3Dense, "Qwen3Dense"
    elif model_type == "m3_agent":
        from models.m3_agent import M3Agent
        return M3Agent, "M3Agent"
    elif model_type == "minicpm":
        from minicpm_v_2_6 import MiniCPMVideo
        return MiniCPMVideo, "MiniCPM-2.6"
    elif model_type == "glm45v":
        from models.glm45v import GLM45V
        return GLM45V, "GLM-4.5V"
    elif model_type == "timechat":
        from models.timechat import TimeChatOnlineStreaming
        return TimeChatOnlineStreaming, "TimeChat-Online"
    elif model_type == "qwen3_omni":
        from models.qwen3_omni import Qwen3Omni
        return Qwen3Omni, "Qwen3-Omni"
    elif model_type.startswith("internvl-3-5"):
        from models.internvl_3_5 import InternVL35Model, InternVL35ThinkingModel

        if model_type.endswith("-thinking"):
            return InternVL35ThinkingModel, model_type.replace("-", " ").title()
        return InternVL35Model, model_type.replace("-", " ").title()
    elif model_type == "minicpm-4-5":
        from models.minicpm_v_4_5 import MiniCPM45Model

        return MiniCPM45Model, "MiniCPM-4.5"
    elif model_type == "longvila":
        from models.longvila import LongVILAModel

        return LongVILAModel, "LongVILA-R1-7B"
    elif model_type == "dummy_eval":
        from models.dummy_eval import DummyEvalModel

        return DummyEvalModel, "DummyEvalModel"
    elif model_type in ("claude-opus", "claude-opus-4-7", "claude-opus-4-6"):
        from models.claude_api import ClaudeAPIModel, DEFAULT_MODEL_ID

        # Map the alias to a specific API model_id. Pass via a small wrapper
        # class so ``model_id`` lands on the adapter without changing the
        # harness's load-by-name signature.
        if model_type == "claude-opus-4-6":
            api_model = "claude-opus-4-6"
            friendly = "Claude-Opus-4-6"
        else:
            api_model = DEFAULT_MODEL_ID  # claude-opus-4-7
            friendly = "Claude-Opus-4-7"

        class _ClaudeAPIModelBound(ClaudeAPIModel):
            def __init__(self, model_id=api_model, **kwargs):
                super().__init__(model_id=api_model, **kwargs)

        return _ClaudeAPIModelBound, friendly
    elif model_type in ("openrouter", "gemini-3.1", "gpt-5.5", "grok-5"):
        # Generic OpenRouter adapter. The actual upstream slug is resolved at
        # instantiate_model time from the CLI flag (--openrouter-model) or
        # a convenience alias. We return the raw class here and let the
        # instantiator bind model_id.
        from models.openrouter_api import OpenRouterAPIModel, KNOWN_SLUGS

        if model_type == "openrouter":
            friendly = "OpenRouter"
        else:
            # Friendly display name maps to the CLAUDE.md official names.
            friendly_map = {
                "gemini-3.1": "Gemini-3.1",
                "gpt-5.5": "GPT-5.5",
                "grok-5": "Grok-5",
            }
            friendly = friendly_map[model_type]
        return OpenRouterAPIModel, friendly
    else:
        print(f"❌ Unknown model type: {model_type}")
        sys.exit(1)



def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True if exception indicates a CUDA out-of-memory condition."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    message = str(exc).lower()
    if "out of memory" in message and "cuda" in message:
        return True
    if "cuda error" in message and "memory" in message:
        return True
    return False


def _clear_exception_traceback(exc: BaseException) -> None:
    """Detach traceback frames so large locals (e.g. CUDA tensors) can be reclaimed."""
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        try:
            frame.f_locals.clear()
        except Exception:
            pass
        tb = tb.tb_next

    try:
        import traceback  # Lazy import keeps startup fast

        traceback.clear_frames(exc.__traceback__)
    except Exception:
        pass

    try:
        exc.__traceback__ = None  # type: ignore[assignment]
    except Exception:
        pass
    exc.__cause__ = None
    exc.__context__ = None


def _lightweight_exception(exc: BaseException) -> BaseException:
    """Return an exception of the same type without traceback references."""
    try:
        clone = exc.__class__(*getattr(exc, "args", ()))
    except Exception:
        clone = RuntimeError(str(exc))
    _clear_exception_traceback(clone)
    return clone


def _reduced_frame_budget(current_max_frames: int) -> int:
    """Compute reduced frame budget after an OOM event (90% of previous, at least 1)."""
    if current_max_frames <= 1:
        return 1

    reduced = max(int(math.floor(current_max_frames * 0.90)), 1)
    if reduced == current_max_frames and current_max_frames > 1:
        reduced = current_max_frames - 1
    return max(reduced, 1)

class QuestionOOMRetry(RuntimeError):
    """Signal that question processing needs to be retried after reducing resources."""

    def __init__(
        self,
        question_index: int,
        partial_results: List[Dict[str, Any]],
        original_exception: BaseException,
        oom_timestamp: float,
        partial_metrics: Optional[PerformanceMetrics],
    ):
        super().__init__("CUDA OOM during question processing")
        self.question_index = question_index
        self.partial_results = partial_results
        self.original_exception_message = repr(original_exception)
        self.original_exception = _lightweight_exception(original_exception)
        _clear_exception_traceback(original_exception)
        self.oom_timestamp = float(oom_timestamp)
        self.partial_metrics = partial_metrics


def process_video_text_batched(
    video_data: VideoEntry,
    model: Any,
    sequence_processor: SequenceProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    input_mode: str,
    eval_mode: str,
    binary_questions: bool,
    predictive_questions: bool,
    verbose: bool,
    state_manager: Any,
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process questions in text/sequence mode with batching for improved performance.

    This function groups questions into batches and reuses the base sequence context
    to reduce overhead from repeated state loading.
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)

    # Extract video-level entropy features once
    def _get_video_entropy_features() -> Dict[str, Any]:
        metadata = getattr(video_data, 'metadata', None)
        if not metadata or not isinstance(metadata, dict):
            return {}
        entropy_overall = metadata.get('entropy_overall')
        if not isinstance(entropy_overall, dict):
            return {}
        features = {}
        empirical = entropy_overall.get('empirical_bits')
        if isinstance(empirical, dict):
            features['entropy_empirical_bits'] = empirical
            try:
                vals = [float(v) for v in empirical.values()]
                if vals:
                    features['entropy_empirical_mean'] = sum(vals) / len(vals)
            except (TypeError, ValueError):
                pass
        analytic = entropy_overall.get('analytic_bits')
        if isinstance(analytic, dict):
            features['entropy_analytic_bits'] = analytic
            try:
                vals = [float(v) for v in analytic.values()]
                if vals:
                    features['entropy_analytic_mean'] = sum(vals) / len(vals)
            except (TypeError, ValueError):
                pass
        return features

    video_entropy_features = _get_video_entropy_features()

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        if verbose:
            print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return []

    # Stream full sequence ONCE
    base_sequence_time, base_sequence_statements = sequence_processor.stream_full_sequences(
        model,
        base_time=0.0,
    )
    sequence_base_state = model.save_state()

    # Filter and prepare questions
    questions_to_process = []
    new_valid_count = 0

    for i, question_data in enumerate(video_data.questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            if verbose:
                print(f"  [LIMIT] Reached limit_questions={limit_questions} (completed={valid_completed}, new={new_valid_count}), stopping")
            break

        questions_to_process.append((i, question_data, question_identifier))
        new_valid_count += 1

    # Log progress with eval_mode-aware counts
    if verbose and valid_completed > 0:
        print(f"  [RESUME] Skipped {valid_completed}/{total_valid} valid questions (eval_mode={eval_mode}), processing {len(questions_to_process)} remaining")

    if not questions_to_process:
        print(f"No questions to process for video {video_index}")
        return []

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []
    question_offset = 0

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Check if model supports true batching
        supports_batching = hasattr(model, 'ask_question_batch') and callable(getattr(model, 'ask_question_batch', None))

        if supports_batching and len(batch) > 1:
            # TRUE BATCHING: Process entire batch in one model call
            model.load_state(sequence_base_state)
            actual_time = base_sequence_time

            # Build all question prompts with their options
            batch_prompts = []
            batch_metadata = []

            for q_local_idx, question_data in enumerate(batch):
                global_idx, original_question, question_identifier = questions_to_process[question_offset + q_local_idx]

                question_mode = (question_data.question_mode or "").strip().lower()

                # Build instruction
                if question_mode == "continuation":
                    prefix_text = question_processor._format_primary_prefix(question_data)  # pylint: disable=protected-access
                    if prefix_text:
                        instruction_text = (
                            "You just saw the token sequence. Determine which option continues the prefix "
                            f"[{prefix_text}] with the exact next tokens in the original sequence, preserving the original order with no skipped or inserted tokens. More than one continuation may exist; select the continuation that matches one of the provided options."
                        )
                    else:
                        instruction_text = (
                            "You just saw the token sequence with the highlighted prefix. Determine which option continues the prefix with the exact next tokens in the original sequence, in the same order and without adding extra tokens."
                        )
                else:
                    instruction_text = (
                        "You just saw the entire token sequence. Determine which option's subsequence appears somewhere in that sequence."
                    )

                # Build options text
                options_text = ""
                for option_idx, option in enumerate(question_data.options):
                    option_text = sequence_processor.build_option_statement(
                        option_idx,
                        option,
                        "",
                    )
                    options_text += f"\n{option_text}\n"

                # Build question text
                question_text = question_processor.build_question_text(question_data, len(question_data.options))

                # Combine into full prompt
                full_prompt = f"{instruction_text}\n{options_text}\n{question_text}"
                batch_prompts.append(full_prompt)
                batch_metadata.append((global_idx, question_data, question_identifier))

            # Single batched model call
            batch_responses = model.ask_question_batch(
                batch_prompts,
                current_video_time=actual_time,
                max_tokens=max_tokens
            )

            # Process batch responses
            for idx, response in enumerate(batch_responses):
                global_idx, question_data, question_identifier = batch_metadata[idx]

                # Extract answer and create result dict
                predicted_answer = question_processor.extract_answer(response)
                correct_answer = str(question_data.correct_answer_index)
                is_correct = predicted_answer.lower() == correct_answer.lower()
                is_dont_know = predicted_answer.lower() == str(question_data.dont_know_index).lower()

                result = {
                    'video_index': video_index,
                    'question_index': global_idx,
                    'question_id': question_identifier,
                    'predicted_answer': predicted_answer,
                    'predicted': predicted_answer,  # Alias
                    'correct_answer': correct_answer,
                    'correct': correct_answer,  # Alias
                    'is_correct': is_correct,
                    'is_dont_know': is_dont_know,
                    'response': response,
                    'num_options': len(question_data.options),
                    'frames_seen_before_question': 0,
                }

                # Extract token statistics
                token_stats = _extract_token_stats(model, response)
                result.update(token_stats)

                # Add max_tokens and calculate truncation
                result['max_tokens'] = max_tokens
                response_token_count = result.get('response_token_count')
                if response_token_count is not None and max_tokens:
                    result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
                else:
                    result['output_was_truncated'] = None

                # max_frames and saw_all_frames not applicable in text mode
                result['max_frames'] = None
                result['saw_all_frames'] = None

                # latency_ask_question and peak_gpu_mem_ask_question
                result['latency_ask_question'] = None
                result['peak_gpu_mem_ask_question'] = None

                # Add video-level entropy features
                result.update(video_entropy_features)

                # Add bucket field
                result['bucket'] = bucket

                all_results.append(result)

                # Save state
                if state_manager and state_manager.enabled:
                    state_manager.mark_question_complete(video_index, question_identifier, result, bucket=bucket)

                # Log to CSV
                if question_log_path and question_log_rows is not None:
                    csv_row = {
                        'video_id': video_index,
                        'variant': 0,
                        'question_id': question_identifier,
                        'question_order': global_idx,
                        'video_entropy': None,
                        'correct_answer': result.get('correct'),
                        'model_answer': result.get('predicted'),
                        'is_correct': result.get('is_correct'),
                        'is_dont_know': result.get('is_dont_know'),
                        'response': result.get('response', ''),
                        'num_options': result.get('num_options'),
                    }
                    question_log_rows.append(csv_row)

                if verbose:
                    print(f"    Q{global_idx+1}: {predicted_answer} ({'✅' if is_correct else '❌'})")

        else:
            # PARTIAL BATCHING: Process questions individually with context reuse
            for q_local_idx, question_data in enumerate(batch):
                global_idx, original_question, question_identifier = questions_to_process[question_offset + q_local_idx]

                # Restore base sequence state
                model.load_state(sequence_base_state)
                actual_time = base_sequence_time

                question_mode = (question_data.question_mode or "").strip().lower()

                # Add question instruction
                if question_mode == "continuation":
                    prefix_text = question_processor._format_primary_prefix(question_data)  # pylint: disable=protected-access
                    if prefix_text:
                        instruction_text = (
                            "You just saw the token sequence. Determine which option continues the prefix "
                            f"[{prefix_text}] with the exact next tokens in the original sequence, preserving the original order with no skipped or inserted tokens. More than one continuation may exist; select the continuation that matches one of the provided options."
                        )
                    else:
                        instruction_text = (
                            "You just saw the token sequence with the highlighted prefix. Determine which option continues the prefix with the exact next tokens in the original sequence, in the same order and without adding extra tokens."
                        )
                else:
                    instruction_text = (
                        "You just saw the entire token sequence. Determine which option's subsequence appears somewhere in that sequence."
                    )

                instruction_payload = f"\n\n{instruction_text}\n\n"
                model.add_text(instruction_payload, current_video_time=actual_time)

                # Add options
                running_time = actual_time
                for option_idx, option in enumerate(question_data.options):
                    option_text = sequence_processor.build_option_statement(
                        option_idx,
                        option,
                        "",
                    )
                    option_payload = f"\n{option_text}\n"
                    model.add_text(option_payload, current_video_time=running_time)
                    running_time += 1.0

                # Process question
                result = question_processor.process_single_question(
                    model,
                    question_data,
                    video_index,
                    max_tokens=max_tokens,
                    current_video_time=running_time,
                )
                # Add video-level entropy features
                result.update(video_entropy_features)

                # Add bucket field
                result['bucket'] = bucket

                all_results.append(result)

                # Save state
                if state_manager and state_manager.enabled:
                    state_manager.mark_question_complete(
                        video_index,
                        question_identifier,
                        result,
                        bucket=bucket
                    )

                # Log to CSV if requested
                if question_log_path and question_log_rows is not None:
                    csv_row = {
                        'video_id': video_index,
                        'variant': 0,
                        'question_id': question_identifier,
                        'question_order': global_idx,
                        'video_entropy': None,
                        'correct_answer': result.get('correct'),
                        'model_answer': result.get('predicted'),
                        'is_correct': result.get('is_correct'),
                        'is_dont_know': result.get('is_dont_know'),
                        'response': result.get('response', ''),
                        'num_options': result.get('num_options'),
                    }
                    question_log_rows.append(csv_row)

        question_offset += len(batch)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_sequence_native_binary_batched(
    video_data: VideoEntry,
    model: Any,
    sequence_processor: SequenceProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    eval_mode: str,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    Process native binary questions in sequence mode with batching.
    Each batch shares the main sequence but has multiple candidate sequences.
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    if verbose:
        print(f"\nProcessing Video {video_index} (bucket: {bucket or 'None'}) with sequence mode batching")

    # Filter questions
    questions_to_process = []
    sequential_count = 0

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode for sequential questions
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue  # Skip spatial questions

            if limit_questions is not None and sequential_count >= limit_questions:
                break
            sequential_count += 1
        else:
            if limit_questions is not None and len(questions_to_process) >= limit_questions:
                break

        if not question_data.is_native_binary:
            continue  # Only batch native binary questions

        questions_to_process.append((i, question_data, question_identifier))

    # Log skipped vs processing counts
    total_native_binary = len([q for q in questions if q.is_native_binary])
    skipped_count = total_native_binary - len(questions_to_process)
    if verbose and skipped_count > 0:
        print(f"  [RESUME] Skipped {skipped_count}/{total_native_binary} already-completed native binary questions, processing {len(questions_to_process)} remaining")

    if not questions_to_process:
        return []

    # Stream main sequence ONCE
    base_sequence_time, base_sequence_statements = sequence_processor.stream_full_sequences(
        model,
        base_time=0.0,
    )
    sequence_base_state = model.save_state()

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []
    supports_batching = hasattr(model, 'ask_question_batch') and callable(getattr(model, 'ask_question_batch', None))

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Restore base sequence state
        model.load_state(sequence_base_state)
        actual_time = base_sequence_time

        # Add ALL candidate sequences for this batch
        for question_data in batch:
            candidate = question_data.candidate
            if candidate and candidate.sequence:
                candidate_text = f"\nCandidate sequence: {', '.join(candidate.sequence)}\n"
                model.add_text(candidate_text, current_video_time=actual_time)
                actual_time += 1.0

        # Build batch of question texts
        batch_question_texts = []
        for question_data in batch:
            question_text = question_processor.build_native_binary_question_text(question_data, include_uncertain=True)
            batch_question_texts.append(question_text)

        # Process batch
        if supports_batching and len(batch) > 1:
            responses = model.ask_question_batch(
                batch_question_texts,
                current_video_time=actual_time,
                max_tokens=max_tokens,
            )
        else:
            # Sequential fallback
            responses = [model.ask_question(qt, current_video_time=actual_time, max_tokens=max_tokens) for qt in batch_question_texts]

        # Process each response
        for idx, (question_data, response) in enumerate(zip(batch, responses)):
            global_idx, original_question, question_identifier = questions_to_process[batch_idx * batch_size + idx]

            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            candidate_seq = ', '.join(question_data.candidate.sequence) if question_data.candidate else "N/A"

            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"    Q{global_idx+1}: {predicted_answer} ({status}) - Candidate: [{candidate_seq}]")

            # Build result
            result = {
                'video_index': video_index,
                'question_index': global_idx,
                'question_id': question_identifier,
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': 0,
                'question_order': question_data.question_order,
                'question_mode': question_data.question_mode,
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
                'sequence_context': list(base_sequence_statements),
                'sequence_format': 'comma',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # max_frames and saw_all_frames not applicable in sequence mode
            result['max_frames'] = None
            result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                # entropy_prefix
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            mean_entropy = sum(values) / len(values)
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = mean_entropy
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy from video_data
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            # Add candidate fields
            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            # CSV logging
            csv_row = {
                'video_id': video_index,
                'variant': 0,
                'bucket': result.get('bucket'),
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'is_native_binary': True,
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=0, bucket=bucket)
                completed_set.add(question_identifier)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_sequence_native_binary_batched_isolated(
    video_data: VideoEntry,
    model: Any,
    sequence_processor: SequenceProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    eval_mode: str,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    TRUE parallel batching for sequence mode native binary questions.
    Each question gets its own isolated context (main sequence + ONE candidate).
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    if verbose:
        print(f"\nProcessing Video {video_index} (bucket: {bucket or 'None'}) with TRUE sequence mode batching (isolated contexts)")

    # Check if model supports isolated batching
    if not hasattr(model, 'ask_question_batch_isolated'):
        raise RuntimeError(
            f"Model {type(model).__name__} does not support isolated batching. "
            f"Disable ENABLE_TEXT_BATCHING or implement ask_question_batch_isolated()."
        )

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        if verbose:
            print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return []

    # Filter questions
    questions_to_process = []
    new_valid_count = 0

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue

        # Check if this is native binary
        if not question_data.is_native_binary:
            continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            if verbose:
                print(f"  [LIMIT] Reached limit_questions={limit_questions} (completed={valid_completed}, new={new_valid_count}), stopping")
            break

        questions_to_process.append((i, question_data, question_identifier))
        new_valid_count += 1

    if not questions_to_process:
        return []

    # Log progress with eval_mode-aware counts
    if verbose and valid_completed > 0:
        print(f"  [RESUME] Skipped {valid_completed}/{total_valid} valid questions (eval_mode={eval_mode}), processing {len(questions_to_process)} remaining")

    # Get main sequence (shared reference, not added to context yet)
    sequences_used = video_data.metadata.get('sequences_used', {})
    main_sequence_tokens = sequences_used.get('S_tokens', [])
    main_sequence_text = ', '.join(str(t) for t in main_sequence_tokens)

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Build separate isolated context for EACH question
        question_contexts = []
        batch_metadata = []

        for q_idx, question_data in enumerate(batch):
            global_idx, original_question, question_identifier = questions_to_process[batch_idx * batch_size + q_idx]

            # Build isolated context for THIS question only
            candidate = question_data.candidate
            candidate_text = ', '.join(candidate.sequence) if candidate and candidate.sequence else ""

            question_prompt = question_processor.build_native_binary_question_text(
                question_data,
                include_uncertain=True
            )

            # Create context dict for this specific question
            ctx = {
                'main_sequence': main_sequence_text,
                'candidate_sequence': candidate_text,
                'question_text': question_prompt,
                'mode': 'sequence'
            }

            question_contexts.append(ctx)
            batch_metadata.append({
                'global_idx': global_idx,
                'question_data': question_data,
                'question_identifier': question_identifier,
                'candidate_text': candidate_text
            })

        # Call model with isolated contexts
        responses = model.ask_question_batch_isolated(
            contexts=question_contexts,
            max_tokens=max_tokens,
        )

        # Process responses
        for q_idx, (response, meta) in enumerate(zip(responses, batch_metadata)):
            question_data = meta['question_data']
            question_identifier = meta['question_identifier']
            global_idx = meta['global_idx']

            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"    Q{global_idx+1}: pred={predicted_answer} correct={correct_answer} ({status}) - Candidate: [{meta['candidate_text']}]")

            result = {
                'video_index': video_index,
                'question_index': global_idx,
                'question_id': question_identifier,
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': 0,
                'question_order': question_data.question_order,
                'question_mode': question_data.question_mode,
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
                'sequence_format': 'comma',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # max_frames and saw_all_frames not applicable in sequence mode
            result['max_frames'] = None
            result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                # entropy_prefix
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            mean_entropy = sum(values) / len(values)
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = mean_entropy
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy from video_data
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            csv_row = {
                'video_id': video_index,
                'variant': 0,
                'bucket': result.get('bucket'),
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'is_native_binary': True,
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=0, bucket=bucket)
                completed_set.add(question_identifier)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_video_native_binary_batched_isolated(
    video_data: VideoEntry,
    model: Any,
    video_processor: VideoProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    eval_mode: str,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    TRUE parallel batching for video mode native binary questions.
    Each question gets its own isolated context (main video + ONE candidate clip as separate videos).
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    if verbose:
        bucket_display = _extract_bucket_from_video_data(video_data)
        print(f"\nProcessing Video {video_index} (bucket: {bucket_display or 'None'}) with TRUE video mode batching (isolated contexts)")

    # Check if model supports isolated batching
    if not hasattr(model, 'ask_question_batch_isolated'):
        raise RuntimeError(
            f"Model {type(model).__name__} does not support isolated batching. "
            f"Disable ENABLE_VIDEO_BATCHING or implement ask_question_batch_isolated()."
        )

    # Load main video once
    print("Loading main video...")
    main_video_path = str(video_data.video_path)
    video_processor.load_main_video(main_video_path)
    basename = os.path.basename(main_video_path)
    print(f"Main video loaded: {basename}")

    # Get all main video frames from processor
    main_video_frames = video_processor.main_video_frames

    # Get frame count for frames_seen_before_question field
    main_video_frame_count = video_processor.frame_sampler.get_frame_count(main_video_frames)

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        if verbose:
            print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return []

    # Filter questions
    questions_to_process = []
    new_valid_count = 0

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue

        # Check if this is native binary
        if not question_data.is_native_binary:
            continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            if verbose:
                print(f"  [LIMIT] Reached limit_questions={limit_questions} (completed={valid_completed}, new={new_valid_count}), stopping")
            break

        questions_to_process.append((i, question_data, question_identifier))
        new_valid_count += 1

    # Log progress with eval_mode-aware counts
    if verbose and valid_completed > 0:
        print(f"  [RESUME] Skipped {valid_completed}/{total_valid} valid questions (eval_mode={eval_mode}), processing {len(questions_to_process)} remaining")

    if not questions_to_process:
        return []

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Build separate isolated context for EACH question
        question_contexts = []
        batch_metadata = []

        for q_idx, question_data in enumerate(batch):
            global_idx, original_question, question_identifier = questions_to_process[batch_idx * batch_size + q_idx]

            # Load candidate clip for this question
            candidate = question_data.candidate
            candidate_frames = None
            if candidate and candidate.clip_path:
                candidate_clips = video_processor.load_option_videos([candidate.clip_path])
                if candidate_clips:
                    candidate_frames = candidate_clips[0]

            question_prompt = question_processor.build_native_binary_question_text(
                question_data,
                include_uncertain=True
            )

            # Create context dict - keep main and candidate as SEPARATE videos
            # Clone frames to avoid sharing references across contexts
            # Handle different frame types properly
            from frame_samplers.internvl_sampler import InternVLSampledVideo
            import numpy as np

            # Clone main video frames
            if isinstance(main_video_frames, InternVLSampledVideo):
                cloned_main = InternVLSampledVideo(
                    pixel_values=main_video_frames.pixel_values.clone(),
                    num_patches_list=list(main_video_frames.num_patches_list),
                    fps=main_video_frames.fps
                )
            elif torch.is_tensor(main_video_frames):
                cloned_main = main_video_frames.clone()
            elif isinstance(main_video_frames, np.ndarray):
                cloned_main = main_video_frames.copy()
            else:
                cloned_main = main_video_frames

            # Clone candidate frames
            if candidate_frames is not None:
                if isinstance(candidate_frames, InternVLSampledVideo):
                    cloned_candidate = InternVLSampledVideo(
                        pixel_values=candidate_frames.pixel_values.clone(),
                        num_patches_list=list(candidate_frames.num_patches_list),
                        fps=candidate_frames.fps
                    )
                elif torch.is_tensor(candidate_frames):
                    cloned_candidate = candidate_frames.clone()
                elif isinstance(candidate_frames, np.ndarray):
                    cloned_candidate = candidate_frames.copy()
                else:
                    cloned_candidate = candidate_frames
            else:
                cloned_candidate = None

            ctx = {
                'main_video_frames': cloned_main,
                'candidate_video_frames': cloned_candidate,
                'question_text': question_prompt,
                'mode': 'video'
            }

            question_contexts.append(ctx)
            # Extract clip name from actual loaded frames, not just metadata
            candidate_clip_name = os.path.basename(candidate.clip_path) if (candidate and candidate_frames is not None) else "None"
            batch_metadata.append({
                'global_idx': global_idx,
                'question_data': question_data,
                'question_identifier': question_identifier,
                'candidate_clip': candidate_clip_name
            })

        # Call model with isolated contexts
        responses = model.ask_question_batch_isolated(
            contexts=question_contexts,
            max_tokens=max_tokens,
            max_frames_in_video=video_processor.model_max_frames,
        )

        # Process responses
        for q_idx, (response, meta) in enumerate(zip(responses, batch_metadata)):
            question_data = meta['question_data']
            question_identifier = meta['question_identifier']
            global_idx = meta['global_idx']

            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"    Q{global_idx+1}: pred={predicted_answer} correct={correct_answer} ({status}) - Clip: {meta['candidate_clip']}")

            result = {
                'video_index': video_index,
                'question_index': global_idx,
                'question_id': question_identifier,
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': main_video_frame_count,
                'question_order': question_data.question_order,
                'question_mode': question_data.question_mode,
                'eval_mode': 'spatial',
                'input_mode': 'video',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # Get max_frames and saw_all_frames from video_processor if available
            if video_processor:
                result['max_frames'] = getattr(video_processor, 'max_frames', None)
                result['saw_all_frames'] = getattr(video_processor, 'saw_all_frames', None)
            else:
                result['max_frames'] = None
                result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                # entropy_prefix
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            mean_entropy = sum(values) / len(values)
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = mean_entropy
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy from video_data
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            csv_row = {
                'video_id': video_index,
                'variant': 0,
                'bucket': result.get('bucket'),
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'is_native_binary': True,
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'eval_mode': 'spatial',
                'input_mode': 'video',
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=0, bucket=bucket)
                completed_set.add(question_identifier)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_video_native_binary_batched(
    video_data: VideoEntry,
    model: Any,
    video_processor: VideoProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    text_to_video_processor: Optional[TextToVideoProcessor],
    no_option_text: bool,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    Process native binary questions in video mode with batching.
    Each batch shares the main video but has multiple candidate clips.
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    # Load main video
    print(f"\nProcessing Video {video_index} (bucket: {bucket or 'None'}) with batching")
    print("Loading main video...")
    main_video_path = str(video_data.video_path)
    video_processor.load_main_video(main_video_path)
    print(f"Main video loaded: {os.path.basename(main_video_path)}")

    # Filter questions
    questions_to_process = []
    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue
        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue
        if limit_questions is not None and len(questions_to_process) >= limit_questions:
            break
        if not question_data.is_native_binary:
            continue  # Only batch native binary questions
        questions_to_process.append((i, question_data, question_identifier))

    # Log skipped vs processing counts
    total_native_binary = len([q for q in questions if q.is_native_binary])
    skipped_count = total_native_binary - len(questions_to_process)
    if verbose and skipped_count > 0:
        print(f"  [RESUME] Skipped {skipped_count}/{total_native_binary} already-completed native binary questions, processing {len(questions_to_process)} remaining")

    if not questions_to_process:
        return []

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []
    supports_batching = hasattr(model, 'ask_question_batch') and callable(getattr(model, 'ask_question_batch', None))

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Load main video ONCE for the batch
        model.clear_context()
        prefix_text = "Here is a main video to remember:"
        model.add_text(prefix_text, current_video_time=0.0)

        # Determine max question time in this batch
        max_question_time = max(float(q.question_time) for q in batch)
        actual_time = video_processor.add_main_video_up_to_time(model, max_question_time)

        # Add ALL candidate clips for this batch
        candidate_times = []
        for question_data in batch:
            candidate = question_data.candidate
            if candidate and candidate.clip_path:
                if no_option_text and text_to_video_processor:
                    text_frames = text_to_video_processor.create_option_label_frames_for_model(
                        "Candidate",
                        video_processor.frame_sampler,
                        duration_seconds=2.0,
                    )
                    text_frame_count = text_to_video_processor.get_frame_count_for_model(
                        text_frames,
                        video_processor.frame_sampler,
                    )
                    if text_frame_count > 0:
                        text_start_time = actual_time + 1
                        text_end_time = text_start_time + text_frame_count / video_processor.fps
                        actual_time = text_end_time
                        model.add_video(
                            video_frames=text_frames,
                            time_start=text_start_time,
                            time_end=text_end_time,
                            video_id=f"candidate_label_{batch_idx}"
                        )
                else:
                    candidate_intro = "\nHere is a candidate clip:\n"
                    model.add_text(candidate_intro, current_video_time=actual_time)

                # Load and add candidate clip
                candidate_clips = video_processor.load_option_videos([candidate.clip_path])
                if candidate_clips:
                    candidate_video = candidate_clips[0]
                    frame_count = video_processor.frame_sampler.get_frame_count(candidate_video)
                    if frame_count > 0:
                        candidate_start = actual_time + 1
                        candidate_end = candidate_start + frame_count / video_processor.fps
                        model.add_video(
                            video_frames=candidate_video,
                            time_start=candidate_start,
                            time_end=candidate_end,
                            video_id=f"candidate_clip_{batch_idx}"
                        )
                        actual_time = candidate_end
                        candidate_times.append(actual_time)

        # Build batch of question texts
        batch_question_texts = []
        for question_data in batch:
            question_text = question_processor.build_native_binary_question_text(question_data, include_uncertain=True)
            batch_question_texts.append(question_text)

        # Process batch
        if supports_batching and len(batch) > 1:
            responses = model.ask_question_batch(
                batch_question_texts,
                current_video_time=actual_time,
                max_tokens=max_tokens,
            )
        else:
            # Sequential fallback
            responses = [model.ask_question(qt, current_video_time=actual_time, max_tokens=max_tokens) for qt in batch_question_texts]

        # Process each response
        for question_data, response in zip(batch, responses):
            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            question_identifier = question_data.question_id or f"video{video_index}_q{batch_idx}"
            candidate_name = os.path.basename(question_data.candidate.clip_path) if question_data.candidate else "N/A"

            status_icon = '✅' if is_correct else ('❓' if is_dont_know else '❌')
            print(f"Q{question_data.question_order or '?'}: {candidate_name} -> Predicted: {predicted_answer}, Correct: {correct_answer} - {status_icon}")

            result = {
                'video_index': video_index,
                'question_id': question_data.question_id,
                'question_order': getattr(question_data, 'question_order', None),
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': 0,
                'question_mode': question_data.question_mode,
                'eval_mode': 'spatial',
                'input_mode': 'video',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # Get max_frames and saw_all_frames from video_processor if available
            if video_processor:
                result['max_frames'] = getattr(video_processor, 'max_frames', None)
                result['saw_all_frames'] = getattr(video_processor, 'saw_all_frames', None)
            else:
                result['max_frames'] = None
                result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = sum(values) / len(values)
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            # Log to CSV
            if question_log_path and question_log_rows is not None:
                csv_row = {
                    'video_id': video_index,
                    'variant': 0,
                    'bucket': result.get('bucket'),
                    'question_id': question_data.question_id,
                    'question_order': result.get('question_order'),
                    'video_entropy': result.get('entropy_prefix_mean'),
                    'correct_answer': correct_answer,
                    'model_answer': predicted_answer,
                    'is_correct': is_correct,
                    'is_dont_know': is_dont_know,
                    'response': response,
                    'num_options': 3,
                    'is_native_binary': True,
                    'question_type': result.get('question_type'),
                    'question_variant': result.get('question_variant'),
                    'question_time': result.get('question_time'),
                    'clip_start_time': question_data.clip_start_time,
                    'clip_end_time': question_data.clip_end_time,
                    'candidate_present': result.get('candidate_present'),
                    'candidate_clip_start': result.get('candidate_clip_start'),
                    'candidate_clip_end': result.get('candidate_clip_end'),
                    'has_unique_answer': result.get('has_unique_answer'),
                    'scenario': result.get('scenario'),
                    'response_token_count': result.get('response_token_count'),
                    'output_was_truncated': result.get('output_was_truncated'),
                    'eval_mode': 'spatial',
                    'input_mode': 'video',
                }
                question_log_rows.append(csv_row)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_video_with_questions(
    video_data: VideoEntry,
    model: Any,
    video_processor: Optional[VideoProcessor],
    question_processor: QuestionProcessor,
    max_tokens: int,
    max_frames: int,
    text_to_video_processor: TextToVideoProcessor = None,
    no_option_text: bool = False,
    start_question_index: int = 0,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    binary_questions: bool = False,
    wandb_logger: Optional[WandbLogger] = None,
    input_mode: str = "video",
    sequence_processor: Optional[SequenceProcessor] = None,
    sequence_format_label: Optional[str] = None,
    eval_mode: str = "spatial",
    completed_questions: Optional[Set[str]] = None,
    state_manager: Optional[RunStateManager] = None,
    question_log_path: Optional[str] = None,
    question_log_rows: Optional[List[Dict[str, Any]]] = None,
    limit_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process multiple questions for a single video with progressive main video addition."""

    def _extract_bucket_from_path(video_path: str) -> Optional[str]:
        """Extract bucket name from video path like 'videos/UNIFORM_EVAL_L008_ELOW/video_1_v0.mp4'."""
        if not video_path:
            return None
        # Split by / and look for pattern like UNIFORM_EVAL_*
        parts = video_path.split('/')
        for part in parts:
            if 'UNIFORM_EVAL' in part or 'L008' in part or 'L016' in part or 'L032' in part or 'L064' in part or 'L128' in part or 'L1024' in part:
                return part
        return None

    def _compute_entropy_features(question: QuestionEntry) -> Dict[str, Any]:
        metadata = getattr(question, 'metadata', None)
        if not metadata:
            return {}
        entropy_map = metadata.get('entropy_prefix')
        if not isinstance(entropy_map, dict) or not entropy_map:
            return {}
        try:
            values = [float(value) for value in entropy_map.values()]
        except (TypeError, ValueError):
            return {}
        if not values:
            return {}
        mean_entropy = sum(values) / len(values)
        return {
            'entropy_prefix_values': entropy_map,
            'entropy_prefix_mean': mean_entropy,
        }

    def _compute_video_entropy_features() -> Dict[str, Any]:
        """Extract video-level entropy from video_data.metadata.entropy_overall."""
        metadata = getattr(video_data, 'metadata', None)
        if not metadata or not isinstance(metadata, dict):
            return {}
        entropy_overall = metadata.get('entropy_overall')
        if not isinstance(entropy_overall, dict):
            return {}
        result = {}
        # Extract empirical_bits
        empirical = entropy_overall.get('empirical_bits')
        if isinstance(empirical, dict):
            result['entropy_empirical_bits'] = empirical
            try:
                empirical_values = [float(v) for v in empirical.values()]
                if empirical_values:
                    result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
            except (TypeError, ValueError):
                pass
        # Extract analytic_bits
        analytic = entropy_overall.get('analytic_bits')
        if isinstance(analytic, dict):
            result['entropy_analytic_bits'] = analytic
            try:
                analytic_values = [float(v) for v in analytic.values()]
                if analytic_values:
                    result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
            except (TypeError, ValueError):
                pass
        return result

    def _normalize_token_sequence(sequence: Sequence[Any]) -> List[str]:
        return [str(token) for token in sequence]

    def _count_sequence_occurrences(source: List[str], pattern: List[str]) -> int:
        if not source or not pattern or len(pattern) > len(source):
            return 0
        count = 0
        limit = len(source) - len(pattern) + 1
        for idx in range(limit):
            if source[idx: idx + len(pattern)] == pattern:
                count += 1
        return count

    def _compute_prefix_stats(question: QuestionEntry) -> Dict[str, Any]:
        sequences_used = video_data.metadata.get('sequences_used') if isinstance(video_data.metadata, dict) else None
        if not isinstance(sequences_used, dict) or not sequences_used:
            return {}

        option_count = len(question.options)
        if option_count == 0:
            return {}
        correct_index = question.correct_answer_index
        if not (0 <= correct_index < option_count):
            return {}
        option = question.options[correct_index]
        option_sequences = option.metadata.get('sequences') if isinstance(option.metadata, dict) else None
        fallback_sequence = option.metadata.get('sequence') if isinstance(option.metadata, dict) else None

        candidates: List[Tuple[str, List[Any]]] = []
        if isinstance(option_sequences, dict) and option_sequences:
            for key, value in option_sequences.items():
                if isinstance(value, list):
                    candidates.append((key, value))
        if not candidates and isinstance(fallback_sequence, list):
            candidates.append(('default', fallback_sequence))

        for seq_key, prefix_values in candidates:
            source_sequence = sequences_used.get(seq_key)
            if not isinstance(source_sequence, list):
                continue
            normalized_source = _normalize_token_sequence(source_sequence)
            normalized_prefix = _normalize_token_sequence(prefix_values)
            prefix_length = len(normalized_prefix)
            if prefix_length == 0 or prefix_length > len(normalized_source):
                continue
            max_positions = max(len(normalized_source) - prefix_length + 1, 0)
            if max_positions <= 0:
                continue
            occurrences = _count_sequence_occurrences(normalized_source, normalized_prefix)
            fraction = occurrences / max_positions if max_positions > 0 else 0.0
            return {
                'prefix_sequence_key': seq_key,
                'prefix_sequence_length': prefix_length,
                'prefix_total_positions': max_positions,
                'prefix_occurrences': occurrences,
                'prefix_match_fraction': fraction,
            }
        return {}

    def _compute_correct_option_likelihood(question: QuestionEntry) -> Optional[float]:
        options = question.options
        if not options:
            return None
        try:
            correct_index = int(question.correct_answer_index)
        except (TypeError, ValueError):
            return None
        if not (0 <= correct_index < len(options)):
            return None
        option_metadata = options[correct_index].metadata if isinstance(options[correct_index].metadata, dict) else None
        if not option_metadata:
            return None
        likelihood = option_metadata.get('likelihood_combined')
        if likelihood is not None:
            try:
                return float(likelihood)
            except (TypeError, ValueError):
                return None
        likelihoods = option_metadata.get('likelihoods')
        if isinstance(likelihoods, dict) and likelihoods:
            try:
                return float(max(likelihoods.values()))
            except (TypeError, ValueError):
                return None
        return None

    def _snapshot_metric_lengths(metrics_obj: Optional[PerformanceMetrics]) -> Optional[Dict[str, int]]:
        if metrics_obj is None:
            return None
        snapshot: Dict[str, int] = {}
        fields = getattr(metrics_obj, '__dataclass_fields__', {})
        for name in fields:
            value = getattr(metrics_obj, name, None)
            if isinstance(value, list):
                snapshot[name] = len(value)
        return snapshot

    def _restore_metrics_to_snapshot(
        metrics_obj: Optional[PerformanceMetrics],
        snapshot: Optional[Dict[str, int]],
    ) -> None:
        if metrics_obj is None or not snapshot:
            return
        for name, length in snapshot.items():
            value = getattr(metrics_obj, name, None)
            if isinstance(value, list) and len(value) > length:
                del value[length:]

    def _latest_metric_value(metric_name: str) -> Optional[float]:
        metrics_obj = getattr(model, '_metrics', None)
        if metrics_obj is None:
            return None
        values = getattr(metrics_obj, metric_name, None)
        if isinstance(values, list) and values:
            try:
                return float(values[-1])
            except (TypeError, ValueError):
                return None
        return None

    use_sequence_mode = input_mode == "sequence"
    if use_sequence_mode and sequence_processor is None:
        raise ValueError("Sequence mode requires a SequenceProcessor instance.")
    if not use_sequence_mode and video_processor is None:
        raise ValueError("Video mode requires a VideoProcessor instance.")

    video_index = video_data.video_index
    variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching save calls
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))
    for order, question in enumerate(questions, start=1):
        question.question_order = order
    completed_set = set(completed_questions or set())

    print(f"\nProcessing Video {video_index} with {len(questions)} questions")
    print("Pre clear context", flush=True)
    # Clear context completely for fresh start
    model.clear_context()
    print("Post clear context", flush=True)
    if not use_sequence_mode:
        # Load main video frames (but don't add to model yet)
        assert video_processor is not None
        main_video = video_processor.load_main_video(video_data.main_video_path)
        print("Post  load video", flush=True)
    else:
        main_video = None

    results: List[Dict[str, Any]] = [] if existing_results is None else list(existing_results)
    sequence_base_state = None
    base_sequence_time = 0.0
    base_sequence_statements: List[str] = []
    if use_sequence_mode and sequence_processor is not None:
        base_sequence_time, base_sequence_statements = sequence_processor.stream_full_sequences(
            model,
            base_time=0.0,
        )
        sequence_base_state = model.save_state()

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return results, True  # True = hit limit

    # Process each question progressively
    new_valid_count = 0  # Track new valid questions processed in THIS session

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            print(f"Skipping previously completed question {question_identifier}")
            continue

        # Filter questions based on eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                print(f"Skipping spatial question {question_identifier} in sequential mode")
                continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            print(f"Reached question limit ({limit_questions}) for video {video_index} (completed={valid_completed}, new={new_valid_count}), moving to next video")
            return results, True  # True = hit limit

        new_valid_count += 1

        metrics_snapshot: Optional[Dict[str, int]] = None
        metrics_obj = getattr(model, '_metrics', None)
        if metrics_obj is not None:
            metrics_snapshot = _snapshot_metric_lengths(metrics_obj)

        question_mode = (question_data.question_mode or "").strip().lower()

        # Thermal-pause hook removed (always disabled). If you need a thermal
        # throttling check, reintroduce it here.
        print("Starting question", flush=True)
        question_time = question_data.question_time
        print(f"\nProcessing Question {i+1} at time {question_time}s")

        base_state = None
        oom_record_time: Optional[float] = None

        # =====================================================================
        # Native Binary Question Handling
        # =====================================================================
        # Check if this is a native binary format question (single candidate with yes/no answer)
        is_native_binary = getattr(question_data, 'is_native_binary', False)
        if is_native_binary:
            try:
                print(f"[native-binary] Processing native binary question {question_identifier}")

                # Restore base state for sequence mode or clear context for video mode
                sequence_context_statements: List[str] = []
                if use_sequence_mode:
                    assert sequence_processor is not None
                    if sequence_base_state is not None:
                        model.load_state(sequence_base_state)
                    actual_time = base_sequence_time
                    sequence_context_statements = list(base_sequence_statements)

                    # For continuation mode, stream prefix
                    if question_mode == "continuation":
                        actual_time, prefix_statements = sequence_processor.stream_question_prefix(
                            model,
                            question_data,
                            base_time=actual_time,
                        )
                        if prefix_statements:
                            sequence_context_statements.extend(prefix_statements)

                    # Add candidate sequence
                    candidate = question_data.candidate
                    if candidate and candidate.sequence:
                        candidate_text = f"\nCandidate sequence: {', '.join(candidate.sequence)}\n"
                        model.add_text(candidate_text, current_video_time=actual_time)
                        actual_time += 1.0
                    frames_seen_before_question = 0
                else:
                    # Video mode: load main video and candidate clip
                    assert video_processor is not None
                    model.clear_context()
                    video_processor.reset_video_position()  # Reset cursor after clearing context
                    prefix_text = "Here is a main video to remember:"
                    model.add_text(prefix_text, current_video_time=0.0)
                    actual_time = video_processor.add_main_video_up_to_time(model, question_time)
                    frames_seen_before_question = video_processor.get_main_frames_streamed()

                    # Add candidate clip
                    candidate = question_data.candidate
                    if candidate and candidate.clip_path:
                        # Handle no_option_text mode: render label as video frames
                        if no_option_text and text_to_video_processor:
                            # Create "Candidate:" label as video frames
                            text_frames = text_to_video_processor.create_option_label_frames_for_model(
                                "Candidate",  # Label for native binary
                                video_processor.frame_sampler,
                                duration_seconds=2.0,
                            )
                            text_frame_count = text_to_video_processor.get_frame_count_for_model(
                                text_frames,
                                video_processor.frame_sampler,
                            )
                            if text_frame_count > 0:
                                text_start_time = actual_time + 1
                                text_end_time = text_start_time + text_frame_count / video_processor.fps
                                actual_time = text_end_time
                                model.add_video(
                                    video_frames=text_frames,
                                    time_start=text_start_time,
                                    time_end=text_end_time,
                                    video_id="candidate_label"
                                )
                        else:
                            # Normal mode: add text label
                            candidate_intro = "\nHere is a candidate clip:\n"
                            model.add_text(candidate_intro, current_video_time=actual_time)

                        # Load and add the candidate video clip
                        candidate_clips = video_processor.load_option_videos([candidate.clip_path])
                        if candidate_clips:
                            candidate_video = candidate_clips[0]
                            frame_count = video_processor.frame_sampler.get_frame_count(candidate_video)
                            if frame_count > 0:
                                candidate_start = actual_time + 1
                                candidate_end = candidate_start + frame_count / video_processor.fps
                                model.add_video(
                                    video_frames=candidate_video,
                                    time_start=candidate_start,
                                    time_end=candidate_end,
                                    video_id="candidate_clip"
                                )
                                actual_time = candidate_end

                oom_record_time = float(actual_time)

                # Process the native binary question.
                # LongVILA over-commits to the "{uncertain}" option on NO clips
                # (iterdebug baseline: 4/4 NO clips emitted {uncertain} instead
                # of {1}; full bench_short: 99/99 NO clips). Drop the uncertain
                # option for LongVILA only so it is forced into {0}/{1}.
                # LongVILA is in both TEXT_ and VIDEO_BATCHING_EXCLUDED_MODELS,
                # so this single-question path is the only live one for it.
                _use_uncertain_option = type(model).__name__ != 'LongVILAModel'
                result = question_processor.process_native_binary_question(
                    model,
                    question=question_data,
                    video_index=video_index,
                    max_tokens=max_tokens,
                    max_frames=max_frames,
                    current_video_time=actual_time,
                    include_uncertain=_use_uncertain_option,
                )

                # Add entropy features
                entropy_features = _compute_entropy_features(question_data)
                if entropy_features:
                    result.update(entropy_features)
                video_entropy_features = _compute_video_entropy_features()
                if video_entropy_features:
                    result.update(video_entropy_features)
                prefix_stats = _compute_prefix_stats(question_data)
                if prefix_stats:
                    result.update(prefix_stats)
                likelihood_value = _compute_correct_option_likelihood(question_data)
                if likelihood_value is not None:
                    result['correct_option_likelihood'] = likelihood_value

                # Add token metrics
                token_metrics = _extract_token_stats(model, result.get('response', ''))
                if token_metrics:
                    result.update(token_metrics)

                # Check if output was truncated
                response_tokens = result.get('response_token_count')
                if response_tokens is not None and max_tokens:
                    result['output_was_truncated'] = response_tokens >= (max_tokens - 5)
                    result['max_tokens'] = max_tokens
                else:
                    result['output_was_truncated'] = None

                # Add bucket information
                bucket_name = _extract_bucket_from_path(str(video_data.video_path))
                if bucket_name:
                    result['bucket'] = bucket_name

                # Add metadata to result
                result['frames_seen_before_question'] = frames_seen_before_question
                result['question_order'] = question_data.question_order
                result['question_mode'] = question_data.question_mode
                result['eval_mode'] = eval_mode
                result['input_mode'] = input_mode
                result['video_only_mode'] = no_option_text

                if use_sequence_mode:
                    result['sequence_context'] = list(sequence_context_statements)
                    result['sequence_format'] = sequence_format_label or 'comma'

                if hasattr(question_data, 'metadata') and question_data.metadata:
                    if 'question_variant' in question_data.metadata:
                        result['question_variant'] = question_data.metadata['question_variant']
                    if 'question_type' in question_data.metadata:
                        result['question_type'] = question_data.metadata['question_type']
                    # New fields from eval_membership dataset
                    if 'has_unique_answer' in question_data.metadata:
                        result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                    if 'scenario' in question_data.metadata:
                        result['scenario'] = question_data.metadata['scenario']

                # Add candidate-level fields for native binary questions
                if question_data.candidate:
                    result['candidate_present'] = question_data.candidate.present
                    result['candidate_clip_start'] = question_data.candidate.clip_start
                    result['candidate_clip_end'] = question_data.candidate.clip_end

                try:
                    result['question_time'] = float(question_time)
                except (TypeError, ValueError):
                    result['question_time'] = None

                results.append(result)

                # CSV logging
                csv_row = {
                    'video_id': video_data.video_index,
                    'variant': variant if variant is not None else 0,
                    'bucket': result.get('bucket'),
                    'question_id': question_identifier,
                    'question_order': result.get('question_order'),
                    'video_entropy': result.get('entropy_prefix_mean'),
                    'correct_answer': result.get('correct'),
                    'model_answer': result.get('predicted'),
                    'is_correct': result.get('is_correct'),
                    'is_dont_know': result.get('is_dont_know'),
                    'response': result.get('response', ''),
                    'num_options': result.get('num_options'),
                    'is_native_binary': True,
                    'question_type': result.get('question_type'),
                    'question_variant': result.get('question_variant'),
                    'question_time': result.get('question_time'),
                    'clip_start_time': question_data.clip_start_time,
                    'clip_end_time': question_data.clip_end_time,
                    'candidate_present': question_data.candidate.present if question_data.candidate else None,
                    'candidate_clip_start': question_data.candidate.clip_start if question_data.candidate else None,
                    'candidate_clip_end': question_data.candidate.clip_end if question_data.candidate else None,
                    'has_unique_answer': result.get('has_unique_answer'),
                    'scenario': result.get('scenario'),
                    'response_token_count': result.get('response_token_count'),
                    'output_was_truncated': result.get('output_was_truncated'),
                    'eval_mode': eval_mode,
                    'input_mode': input_mode,
                }

                if question_log_path and question_log_rows is not None:
                    question_log_rows.append(csv_row)

                if state_manager is not None:
                    state_manager.mark_question_complete(video_index, question_identifier, result, variant=variant, bucket=bucket)
                    completed_set.add(question_identifier)

                if wandb_logger:
                    wandb_logger.log_question_record(result)

                if termination_requested():
                    print("Termination requested after completing question; exiting question loop.")
                    break

                # Continue to next question
                continue

            except Exception as native_err:
                print(f"[native-binary] Error processing native binary question: {native_err}")
                if metrics_snapshot is not None:
                    _restore_metrics_to_snapshot(metrics_obj, metrics_snapshot)
                raise
        # =====================================================================
        # End Native Binary Question Handling
        # =====================================================================

        try:
            sequence_context_statements: List[str] = []
            if use_sequence_mode:
                assert sequence_processor is not None
                if sequence_base_state is not None:
                    model.load_state(sequence_base_state)
                actual_time = base_sequence_time
                sequence_context_statements = list(base_sequence_statements)
                if question_mode == "continuation":
                    actual_time, prefix_statements = sequence_processor.stream_question_prefix(
                        model,
                        question_data,
                        base_time=actual_time,
                    )
                    if prefix_statements:
                        sequence_context_statements.extend(prefix_statements)
                frames_seen_before_question = 0
                print("Finished streaming sequence context", flush=True)
            else:
                assert video_processor is not None
                # Add prefix text for new prompt format
                prefix_text = (
                    "Here is a main video to remember:"
                    if not use_sequence_mode
                    else "Here is a main sequence to remember:"
                )
                model.add_text(prefix_text, current_video_time=0.0)
                # Add main video up to this question's time point
                actual_time = video_processor.add_main_video_up_to_time(model, question_time)
                print("Finished one adding video", flush=True)
                frames_seen_before_question = video_processor.get_main_frames_streamed()
            oom_record_time = float(actual_time)

            # Save state after adding main video portion (before instruction/options)
            base_state = model.save_state()

            # Add question intro text after main video, before options
            if use_sequence_mode:
                # Keep sequence mode instruction as-is (complex logic for continuation vs exists)
                if question_mode == "continuation":
                    prefix_text = question_processor._format_primary_prefix(question_data)  # pylint: disable=protected-access
                    if prefix_text:
                        instruction_text = (
                            "You just saw the token sequence. Determine which option continues the prefix "
                            f"[{prefix_text}] with the exact next tokens in the original sequence, preserving the original order with no skipped or inserted tokens. More than one continuation may exist; select the continuation that matches one of the provided options."
                        )
                    else:
                        instruction_text = (
                            "You just saw the token sequence with the highlighted prefix. Determine which option continues the prefix with the exact next tokens in the original sequence, in the same order and without adding extra tokens."
                        )
                else:
                    instruction_text = (
                        "You just saw the entire token sequence. Determine which option's subsequence appears somewhere in that sequence."
                    )
                adjust_context = getattr(question_processor, 'adjust_text_for_context', None)
                if callable(adjust_context):
                    instruction_text = adjust_context(instruction_text)
                instruction_payload = f"\n\n{instruction_text}\n\n"
                model.add_text(instruction_payload, current_video_time=actual_time)
            else:
                # New concise format for spatial/video mode
                question_intro = "\nQuestion: Of the following options, which one appeared in the main video?\n"
                model.add_text(question_intro, current_video_time=actual_time)
            if use_sequence_mode:
                print(f"[sequence-mode] Instruction: {instruction_text}", flush=True)

            # Load and add this question's option videos with labels
            running_time = actual_time

            selected_binary_answer: Optional[str] = None
            selected_option_index: Optional[int] = None

            if binary_questions:
                seed_components = [
                    str(video_index),
                    str(i),
                    str(question_data.question_id),
                ]
                seed_material = "|".join(seed_components)
                seed_value = int(hashlib.sha256(seed_material.encode('utf-8')).hexdigest(), 16)
                rng = random.Random(seed_value)

                correct_option_index = int(question_data.correct_answer_index)
                if correct_option_index == question_data.dont_know_index:
                    raise ValueError(
                        "Binary question mode cannot be used when the ground truth is the uncertain option."
                    )

                should_answer_yes = rng.getrandbits(1) == 0
                if should_answer_yes or len(question_data.options) <= 1:
                    selected_option_index = correct_option_index
                    selected_binary_answer = "yes"
                else:
                    incorrect_indices = [
                        idx for idx in range(len(question_data.options)) if idx != correct_option_index
                    ]
                    if incorrect_indices:
                        selected_option_index = incorrect_indices[rng.randrange(len(incorrect_indices))]
                        selected_binary_answer = "no"
                    else:
                        selected_option_index = correct_option_index
                        selected_binary_answer = "yes"

                if selected_option_index is None:
                    raise ValueError("Binary question mode requires at least one playable option.")

                selected_option = question_data.options[selected_option_index]
                if use_sequence_mode:
                    option_sequence = [(selected_option_index, selected_option, None)]
                else:
                    assert video_processor is not None
                    option_videos = video_processor.load_option_videos([selected_option.clip_path])
                    if not option_videos:
                        raise ValueError("Failed to load the selected binary option clip.")
                    option_sequence = [(selected_option_index, selected_option, option_videos[0])]
            else:
                if use_sequence_mode:
                    option_sequence = [
                        (idx, option, None)
                        for idx, option in enumerate(question_data.options)
                    ]
                else:
                    assert video_processor is not None
                    option_paths = [option.clip_path for option in question_data.options]
                    option_videos = video_processor.load_option_videos(option_paths)
                    if len(option_videos) != len(option_paths):
                        raise ValueError(
                            "Mismatch between loaded option clips and manifest entries; expected "
                            f"{len(option_paths)} clips but received {len(option_videos)}."
                        )
                    option_sequence = [
                        (idx, option, option_videos[idx])
                        for idx, option in enumerate(question_data.options)
                    ]

            for option_label_index, option_entry, option_payload in option_sequence:
                label_suffix = ""

                if use_sequence_mode:
                    assert sequence_processor is not None
                    option_text = sequence_processor.build_option_statement(
                        option_label_index,
                        option_entry,
                        label_suffix,
                    )
                    option_payload = f"\n{option_text}\n"
                    model.add_text(option_payload, current_video_time=running_time)
                    running_time += 1.0
                    oom_record_time = float(running_time)
                    continue

                assert video_processor is not None
                option_text = f"\nOption {option_label_index}:\n"

                if no_option_text and text_to_video_processor:
                    text_frames = text_to_video_processor.create_option_label_frames_for_model(
                        option_label_index,
                        video_processor.frame_sampler,
                        duration_seconds=2.0,
                    )
                    text_frame_count = text_to_video_processor.get_frame_count_for_model(
                        text_frames,
                        video_processor.frame_sampler,
                    )

                    if text_frame_count > 0:
                        text_start_time = running_time + 1  # 1 second gap
                        text_end_time = text_start_time + text_frame_count / video_processor.fps
                        running_time = text_end_time

                        model.add_video(
                            video_frames=text_frames,
                            time_start=text_start_time,
                            time_end=text_end_time,
                            video_id=f"option_label_{option_label_index}"
                        )
                else:
                    model.add_text(option_text, current_video_time=running_time)

                option_video = option_payload
                frame_count = video_processor.frame_sampler.get_frame_count(option_video)
                if frame_count > 0:
                    option_start_time = running_time + 1  # 1 second gap
                    option_end_time = option_start_time + frame_count / video_processor.fps
                    running_time = option_end_time
                else:
                    raise ValueError("Option video has zero frames after sampling.")

                model.add_video(
                    video_frames=option_video,
                    time_start=option_start_time,
                    time_end=option_end_time,
                    video_id=option_label_index + 1
                )

                if option_end_time > 0:
                    oom_record_time = float(option_end_time)

            # Process the question with complex format
            oom_record_time = float(actual_time)
            result = question_processor.process_single_question(
                model,
                question=question_data,
                video_index=video_index,
                max_tokens=max_tokens,
                max_frames=max_frames,
                current_video_time=actual_time,
                binary_selected_option_index=selected_option_index,
                binary_correct_answer=selected_binary_answer,
            )
            entropy_features = _compute_entropy_features(question_data)
            if entropy_features:
                result.update(entropy_features)
            video_entropy_features = _compute_video_entropy_features()
            if video_entropy_features:
                result.update(video_entropy_features)
            prefix_stats = _compute_prefix_stats(question_data)
            if prefix_stats:
                result.update(prefix_stats)
            likelihood_value = _compute_correct_option_likelihood(question_data)
            if likelihood_value is not None:
                result['correct_option_likelihood'] = likelihood_value
            result['frames_seen_before_question'] = frames_seen_before_question
            result['question_order'] = question_data.question_order
            if use_sequence_mode:
                result['sequence_context'] = list(sequence_context_statements)
                result['sequence_format'] = sequence_format_label or 'comma'
            if question_data.question_mode:
                result['question_mode'] = question_data.question_mode

            # Add mode tracking fields
            result['video_only_mode'] = no_option_text
            result['eval_mode'] = eval_mode
            result['input_mode'] = input_mode

            # Add per-question type fields from JSON
            if hasattr(question_data, 'metadata'):
                metadata = question_data.metadata
                if metadata:
                    if 'question_variant' in metadata:
                        result['question_variant'] = metadata['question_variant']
                    if 'question_type' in metadata:
                        result['question_type'] = metadata['question_type']
                    # New fields from eval_membership dataset
                    if 'has_unique_answer' in metadata:
                        result['has_unique_answer'] = metadata['has_unique_answer']
                    if 'scenario' in metadata:
                        result['scenario'] = metadata['scenario']

            # Add candidate-level fields if available (for binary questions)
            if hasattr(question_data, 'candidate') and question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            token_metrics = _extract_token_stats(model, result.get('response', ''))
            if token_metrics:
                result.update(token_metrics)

            # Check if output was truncated (hit max_tokens limit)
            response_tokens = result.get('response_token_count')
            if response_tokens is not None and 'max_tokens' in result:
                # Consider truncated if within 5 tokens of limit (some models may overshoot slightly)
                result['output_was_truncated'] = response_tokens >= (result['max_tokens'] - 5)
            else:
                result['output_was_truncated'] = None

            latency_value = _latest_metric_value('latency_ask_question')
            if latency_value is not None:
                result['latency_ask_question'] = latency_value
            peak_gpu_value = _latest_metric_value('peak_gpu_mem_increase_ask_question')
            if peak_gpu_value is not None:
                result['peak_gpu_mem_ask_question'] = peak_gpu_value
            try:
                result['question_time'] = float(question_time)
            except (TypeError, ValueError):
                result['question_time'] = None
            results.append(result)

            # Prepare CSV log row with enhanced data
            csv_row = {
                'video_id': video_data.video_index,
                'variant': variant if variant is not None else 0,
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'eval_mode': eval_mode,
                'input_mode': input_mode,
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=variant, bucket=bucket)
                completed_set.add(question_identifier)

            if wandb_logger:
                wandb_logger.log_question_record(result)

            if termination_requested():
                print("Termination requested after completing question; exiting question loop.")
                break

            # Load state back to just main video (before instruction/options) for next question
            if i < len(questions) - 1:
                if not use_sequence_mode:
                    model.load_state(base_state)
                    if video_processor is not None:
                        video_processor.reset_to_main_video_state()  # Sync tracking

        except Exception as exc:
            if _is_cuda_oom_error(exc):
                oom_timestamp = (
                    float(question_data.question_time)
                    if oom_record_time is None
                    else oom_record_time
                )
                if hasattr(model, "record_first_oom_snapshot") and hasattr(model, "_metrics"):
                    model.record_first_oom_snapshot(oom_timestamp)
                # Attempt to revert to clean state before propagating retry request
                if base_state is not None:
                    try:
                        if hasattr(model, "recover_from_oom"):
                            model.recover_from_oom(base_state)
                        else:
                            model.clear_context()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
                                if callable(reset_peak):
                                    reset_peak()
                            model.load_state(base_state)
                    except Exception:
                        pass
                else:
                    try:
                        if hasattr(model, "teardown_after_oom"):
                            model.teardown_after_oom()
                        else:
                            model.clear_context()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
                                if callable(reset_peak):
                                    reset_peak()
                    except Exception:
                        pass
                if metrics_snapshot is not None:
                    _restore_metrics_to_snapshot(getattr(model, '_metrics', None), metrics_snapshot)
                    if hasattr(model, '_sync_state_memory_tracking_from_metrics'):
                        model._sync_state_memory_tracking_from_metrics()

                partial_metrics = None
                if getattr(model, '_metrics', None) is not None:
                    partial_metrics = copy.deepcopy(model._metrics)
                raise QuestionOOMRetry(
                    i,
                    copy.deepcopy(results),
                    exc,
                    oom_timestamp,
                    partial_metrics,
                ) from exc
            raise

    # Clear context only after ALL questions for this video
    model.clear_context()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return results and whether we stopped due to question limit
    hit_question_limit = (limit_questions is not None and
                          len(questions) > limit_questions)
    return results, hit_question_limit




def get_canonical_question_list(
    video_data: Any,
    eval_mode: str,
    limit_questions: Optional[int],
) -> List[str]:
    """
    Get the canonical list of question IDs that should be processed for this video/bucket.

    This is DETERMINISTIC - always returns the same list for the same inputs.
    The list represents the "first N" questions that qualify based on eval_mode.

    Args:
        video_data: Video entry with questions
        eval_mode: 'sequential' or 'spatial'
        limit_questions: Max questions to include (None = all)

    Returns:
        Ordered list of question_ids that should be processed
    """
    canonical_list = []

    for i, question_data in enumerate(video_data.questions):
        question_id = question_data.question_id or f"video{video_data.video_index}_q{i}"

        # Filter by eval_mode
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue  # Skip spatial in sequential mode

        canonical_list.append(question_id)

        # Stop once we reach the limit
        if limit_questions is not None and len(canonical_list) >= limit_questions:
            break

    return canonical_list


def complete_remaining_questions(
    videos: List[Any],
    model: Any,
    state_manager: Any,
    args: Any,
    question_processor: Any,
    sequence_formatter: Optional[Any],
    use_sequence_mode: bool,
) -> int:
    """
    Fill in gaps for incomplete buckets after main pass.

    Goes through each video/bucket and processes any missing questions
    from the CANONICAL list (first N questions as determined by get_canonical_question_list).

    This is idempotent - safe to call multiple times.
    Questions are added to the bucket's state as they're completed.

    Returns: Number of additional questions processed
    """
    print(f"\n{'='*60}")
    print("COMPLETION PASS - Filling in gaps for incomplete buckets")
    print(f"{'='*60}\n")

    completion_count = 0

    for video_data in videos:
        variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching save calls
        bucket = _extract_bucket_from_video_data(video_data)

        # Skip if already marked complete
        if state_manager.is_video_completed(video_data.video_index, variant, bucket):
            continue

        # Get the CANONICAL list of questions for this bucket (deterministic)
        canonical_questions = get_canonical_question_list(
            video_data,
            args.eval_mode,
            args.limit_questions
        )

        # Get completed questions
        completed_set = state_manager.get_completed_questions(video_data.video_index, variant, bucket)

        # Find missing questions from the canonical list
        missing_question_ids = [qid for qid in canonical_questions if qid not in completed_set]

        if not missing_question_ids:
            # No missing questions, mark complete
            print(f"Video {video_data.video_index} bucket {bucket}: All {len(canonical_questions)} canonical questions complete, marking bucket complete")
            state_manager.mark_video_complete(video_data.video_index, variant, bucket)
            continue

        print(f"Video {video_data.video_index} bucket {bucket}: {len(completed_set)}/{len(canonical_questions)} complete, processing {len(missing_question_ids)} remaining")

        # Build list of questions to process (in canonical order)
        questions_to_process = []
        for question_id in missing_question_ids:
            # Find the question data
            for i, question_data in enumerate(video_data.questions):
                qid = question_data.question_id or f"video{video_data.video_index}_q{i}"
                if qid == question_id:
                    questions_to_process.append((i, question_data, question_id))
                    break

        if not questions_to_process:
            # This shouldn't happen if missing_question_ids was non-empty, but safety check
            print(f"  ⚠ Could not find question data for missing questions, skipping")
            continue

        # Process missing questions (fill gaps to complete the canonical list)
        try:
            video_results = 0

            # Load video/sequence context
            if use_sequence_mode:
                if sequence_formatter is None:
                    print(f"  ✗ Sequence formatter not available, skipping")
                    continue

                # Create processor for this video (same pattern as main loop)
                raw_sequences = video_data.metadata.get('sequences_used')
                if not isinstance(raw_sequences, dict) or not raw_sequences:
                    print(f"  ✗ Video {video_data.video_index} missing 'sequences_used' metadata, skipping")
                    continue

                video_sequence_processor = SequenceProcessor(
                    sequences_used=raw_sequences,
                    formatter=sequence_formatter,
                    print_chunks=True,
                )

                # Stream sequences
                video_sequence_processor.stream_full_sequences(model, base_time=0.0)
            else:
                # Load video (simplified - just use question processor)
                pass

            for i, question_data, question_id in questions_to_process:
                try:
                    # Process single question
                    result = question_processor.process_single_question(
                        model=model,
                        question=question_data,
                        video_index=video_data.video_index,
                        max_tokens=args.max_tokens,
                    )

                    video_results += 1
                    completion_count += 1

                    # Save progress immediately to bucket state
                    state_manager.mark_question_complete(
                        video_data.video_index,
                        question_id,
                        result,
                        variant=variant,
                        bucket=bucket
                    )

                except Exception as e:
                    print(f"  Failed to process question {question_id}: {e}")
                    # Skip this question, continue with others
                    continue

            # Check if bucket is now complete
            # Re-fetch completed set to include newly completed questions
            completed_set_after = state_manager.get_completed_questions(video_data.video_index, variant, bucket)
            remaining = [qid for qid in canonical_questions if qid not in completed_set_after]

            if not remaining:
                print(f"  ✓ Bucket complete: all {len(canonical_questions)} canonical questions done")
                state_manager.mark_video_complete(video_data.video_index, variant, bucket)
            else:
                print(f"  ⚠ Processed {video_results} questions, {len(remaining)} still missing from canonical list")

        except Exception as e:
            print(f"  ✗ Could not complete bucket {bucket}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nCompletion pass finished. Processed {completion_count} additional questions.\n")
    return completion_count


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Video Multiple Choice Evaluation - Multi-question per video format",
        epilog="""
Examples:
  %(prog)s questions.json                                    # QwenFullVideo (default)
  %(prog)s questions.json --model m3_agent                  # M3Agent
  %(prog)s questions.json --model glm45v                    # GLM-4.5V
  %(prog)s questions.json --model timechat                 # TimeChat-Online
  %(prog)s questions.json --model qwen3_full               # Qwen3Dense
  %(prog)s questions.json --model qwen3_omni                # Qwen3-Omni
  %(prog)s questions.json --verbose                          # Show full questions/responses
  %(prog)s questions.json --fps 1 --max_frames 128          # Custom settings
  %(prog)s questions.json --no_describe --max_tokens 800    # Skip descriptions, custom token limit
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("json_path", help="Path to questions JSON file")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second to sample")
    parser.add_argument("--max_frames", type=int, default=5000, help="Maximum frames per video")
    parser.add_argument("--chunk_size", type=float, default=15,
                    help="Process videos in chunks of this many seconds. If not set, processes whole video at once.")
    parser.add_argument("--model", type=str, default="qwen_full",
                       choices=[
                           "qwen_full",
                           "mimo-vl",
                           "phi_multimodal",
                           "qwen3_full",
                            "minicpm",
                           "m3_agent",
                           "glm45v",
                           "timechat",
                           "qwen3_omni",
                           "internvl-3-5",
                           "internvl-3-5-thinking",
                           "internvl-3-5-38b",
                           "internvl-3-5-38b-thinking",
                           "internvl-3-5-30b-a3b",
                           "internvl-3-5-30b-a3b-thinking",
                           "minicpm-4-5",
                           "longvila",
                           "claude-opus",
                           "claude-opus-4-7",
                           "claude-opus-4-6",
                           "openrouter",
                           "gemini-3.1",
                           "gpt-5.5",
                           "grok-5",
                           "dummy_eval",
                       ],
                       help=(
                           "Model to use: qwen_full (default), mimo-vl, phi_multimodal, qwen3_full, "
                           "m3_agent, minicpm, glm45v, timechat, qwen3_omni, "
                           "internvl-3-5 (8B), internvl-3-5-38b, or internvl-3-5-30b-a3b (each with optional thinking variants), "
                           "minicpm-4-5, longvila, claude-opus / claude-opus-4-7 / claude-opus-4-6 (Anthropic API), "
                           "openrouter (generic; requires --openrouter-model <slug>) or gemini-3.1 / gpt-5.5 / grok-5 aliases, "
                           "or dummy_eval for integration tests"
                       ))
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default=None,
        help=(
            "OpenRouter model slug (e.g. 'google/gemini-3.1-pro', 'openai/gpt-5.5'). "
            "Required when --model openrouter; optional overrides for the gemini-3.1 / gpt-5.5 / grok-5 aliases."
        ),
    )
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--bucket-filter", type=str, action="append", default=None, help="Filter videos to only process those from the specified bucket (e.g., UNIFORM_EVAL_L008_ELOW). Can be passed multiple times; a video matches if ANY filter is a substring of its path.")
    parser.add_argument("--verbose", action="store_true", help="Print full questions and responses for debugging")
    parser.add_argument("--enable_metrics", action="store_true", help="Enable performance metrics collection and reporting")
    parser.add_argument(
        "--question-log-csv",
        type=str,
        help=(
            "Path to a CSV that records per-question results as video_id, question_id, "
            "video_entropy, correct_answer, and model_answer."
        ),
    )
    parser.add_argument("--no_describe", action="store_true", help="Omit option description requirement from questions")
    parser.add_argument("--describe", action="store_true", help="Add instruction to describe videos and thought process before answering (video mode only)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens to generate per response")
    parser.add_argument("--no_option_text", action="store_true", help="Convert option labels to video frames instead of text")
    parser.add_argument(
        "--limit_frames_on_oom",
        action="store_true",
        help=(
            "[DEPRECATED 2026-04-26] No longer reduces max_frames on OOM. "
            "Kept for backward-compat with old slurm scripts; if set, a warning "
            "is printed and the run will still exit loudly on the next OOM. "
            "Bump --gres=gpu:N or lower --max_frames in your script instead."
        ),
                       )
    parser.add_argument("--max_gpu_mem", type=float, default=None,
                        help="Per-GPU memory budget in GiB for VLM checkpoints (GPU0 uses 2/3 of this value). If not specified, automatically calculated as MODEL_SIZE_IN_GB/NUM_GPUS + 4.")
    parser.add_argument(
        "--qwen3-omni-fast-processor",
        action="store_true",
        help=(
            "Use the fast image processor for Qwen3-Omni. The default sticks to the slow path so the model still "
            "loads on environments missing the new Transformers fast processor build."
        ),
    )
    parser.add_argument(
        "--qwen3-thinking",
        action="store_true",
        help="Use the Qwen3 VL 8B Thinking checkpoint instead of the instruct variant.",
    )
    parser.add_argument(
        "--restart_on_oom",
        action="store_true",
        help="Reinitialize the model and retry the current video when a CUDA OOM is encountered.",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=35,
        help="Maximum number of retry attempts per video after a CUDA OOM before the run aborts.",
    )
    parser.add_argument(
        "--binary_questions",
        action="store_true",
        help="Switch to deterministic yes/no questions that show a single option clip per prompt.",
    )
    parser.add_argument(
        "--predictive-questions",
        action="store_true",
        help=("Ask each question about which option will occur next in the video, highlighting the small delay between the main clip and every option."),
    )
    parser.add_argument(
        "--input-mode",
        choices=["video", "sequence"],
        default="video",
        help="Use raw video playback (default) or text token sequences instead.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["spatial", "sequential"],
        default="spatial",
        help=(
            "Evaluation mode for sequence questions: 'spatial' shows sequences as (token, lane) tuples "
            "and asks all questions; 'sequential' shows only token sequences and filters to sequential-type questions only."
        ),
    )
    parser.add_argument(
        "--sequence-format",
        choices=["comma", "random-vocab"],
        default="comma",
        help="Formatter used in sequence mode. The random-vocab option reserves space for tokenizer-aware mapping.",
    )
    parser.add_argument(
        "--sequence-vocab-file",
        type=str,
        default=None,
        help="Path to a newline-delimited vocabulary consumed by the random-vocab formatter once implemented.",
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        default=None,
        help="Optional root directory that prefixes relative video or option clip paths from the manifest.",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to a JSON state file for per-question resume support.",
    )
    parser.add_argument(
        "--resume-state",
        action="store_true",
        help="Resume from the provided state file (requires --state-file).",
    )
    parser.add_argument(
        "--limit_questions",
        type=int,
        default=None,
        help="Limit number of questions to process per video (in sorted order, 0-indexed). "
             "When resuming, loads all previous results but only processes up to this limit per video.",
    )

    args = parser.parse_args()

    if args.resume_state and not args.state_file:
        parser.error("--resume-state requires --state-file")
    if args.state_file and (not args.resume_state) and Path(args.state_file).exists():
        parser.error(
            f"State file {args.state_file} already exists. Use --resume-state to continue or remove the file."
        )

    state_manager = RunStateManager(args.state_file, resume=args.resume_state)

    use_sequence_mode = args.input_mode == "sequence"
    if use_sequence_mode and args.sequence_format == "random-vocab":
        raise NotImplementedError(
            "The random-vocab sequence formatter is not implemented yet. Use --sequence-format comma "
            "for now and see README for the planned tokenizer-aligned mapping strategy."
        )

    # Load manifest entries and optionally limit the workload
    videos = load_patternvideos_manifest(
        args.json_path,
        require_video_assets=not use_sequence_mode,
        asset_root=args.asset_root,
    )

    # Filter by bucket if specified (supports multiple --bucket-filter flags; ANY-match)
    if args.bucket_filter:
        filters = args.bucket_filter if isinstance(args.bucket_filter, list) else [args.bucket_filter]
        original_count = len(videos)
        videos = [
            v for v in videos
            if any(bf in str(getattr(v, 'video_path', getattr(v, 'main_video_path', ''))) for bf in filters)
        ]
        print(f"Bucket filter {filters}: {len(videos)}/{original_count} videos match")
        if len(videos) == 0:
            print(f"ERROR: No videos found matching bucket filters {filters}")
            print(f"Available buckets in dataset:")
            buckets = set()
            for v in load_patternvideos_manifest(args.json_path, require_video_assets=False, asset_root=args.asset_root):
                path = str(getattr(v, 'video_path', getattr(v, 'main_video_path', '')))
                if '/' in path:
                    bucket = path.split('/')[1] if path.startswith('videos/') else path.split('/')[0]
                    buckets.add(bucket)
            for bucket in sorted(buckets):
                print(f"  - {bucket}")
            sys.exit(1)

    if args.limit:
        videos = videos[:args.limit]

    # Sort videos by bucket frame count (increasing order: L008, L016, L032, ...)
    # This ensures small buckets process before large ones for efficient resource usage
    def get_video_sort_key(video_data):
        bucket = _extract_bucket_from_video_data(video_data)
        bucket_length = extract_bucket_length(bucket)
        if bucket_length is not None:
            return bucket_length
        # Videos without buckets process last
        return float('inf')

    videos = sorted(videos, key=get_video_sort_key)
    print(f"Sorted {len(videos)} videos by frame count (L008, L016, L032, ...)")

    # Dynamically load the model class
    model_class, model_name = load_model_class(args.model)

    print(f"Starting evaluation with {model_name}")
    print(f"Processing {len(videos)} videos")
    print(f"Settings: fps={args.fps}, max_frames={args.max_frames}, verbose={args.verbose}")

    # Show GPU memory configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")

    print("Initializing model...")

    from utils.memory_utils import calculate_max_gpu_mem

    gpu_mem_aware = {
        "qwen_full",
        "mimo-vl",
        "qwen3_full",
        "glm45v",
        "timechat",
        "qwen3_omni",
        "internvl-3-5",
        "internvl-3-5-thinking",
        "internvl-3-5-38b",
        "internvl-3-5-38b-thinking",
        "internvl-3-5-30b-a3b",
        "internvl-3-5-30b-a3b-thinking",
        "longvila",
        "minicpm-4-5",
    }

    def _build_common_model_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"enable_metrics": args.enable_metrics}
        if args.model in gpu_mem_aware:
            max_gpu_mem_val = calculate_max_gpu_mem(args.model, override=args.max_gpu_mem)
            kwargs["max_gpu_mem"] = max_gpu_mem_val
            if args.verbose:
                print(f"Using max_gpu_mem={max_gpu_mem_val:.1f} GB per GPU")
        return kwargs

    def instantiate_model(max_frames: int) -> Any:
        """Create a fresh model instance with the provided frame budget."""
        common_kwargs = _build_common_model_kwargs()
        if args.model == "m3_agent":
            return model_class("ByteDance-Seed/M3-Agent-Control", **common_kwargs)
        if args.model == "minicpm":
            return model_class("openbmb/MiniCPM-o-2_6", **common_kwargs)
        if args.model == "glm45v":
            return model_class("zai-org/GLM-4.5V", **common_kwargs)
        if args.model == "timechat":
            return model_class("wyccccc/TimeChatOnline-7B", **common_kwargs)
        if args.model == "qwen3_full":
            return model_class(thinking=args.qwen3_thinking, **common_kwargs)
        if args.model == "qwen3_omni":
            omni_kwargs = dict(common_kwargs)
            if args.qwen3_omni_fast_processor:
                omni_kwargs["use_fast_processor"] = True
            return model_class("Qwen/Qwen3-Omni-30B-A3B-Instruct", **omni_kwargs)
        internvl_model_map = {
            "internvl-3-5": "OpenGVLab/InternVL3_5-8B",
            "internvl-3-5-thinking": "OpenGVLab/InternVL3_5-8B",
            "internvl-3-5-38b": "OpenGVLab/InternVL3_5-38B",
            "internvl-3-5-38b-thinking": "OpenGVLab/InternVL3_5-38B",
            "internvl-3-5-30b-a3b": "OpenGVLab/InternVL3_5-30B-A3B",
            "internvl-3-5-30b-a3b-thinking": "OpenGVLab/InternVL3_5-30B-A3B",
        }
        if args.model in internvl_model_map:
            ivl_kwargs = dict(common_kwargs)
            ivl_kwargs["generation_max_tokens"] = args.max_tokens
            return model_class(
                internvl_model_map[args.model],
                **ivl_kwargs,
            )
        if args.model == "minicpm-4-5":
            return model_class("openbmb/MiniCPM-V-4_5", **common_kwargs)
        if args.model == "mimo-vl":
            return model_class("XiaomiMiMo/MiMo-VL-7B-RL", **common_kwargs)
        if args.model == "Phi-4-multimodal":
            return model_class("microsoft/Phi-4-multimodal-instruct", **common_kwargs)
        if args.model == "longvila":
            return model_class("Efficient-Large-Model/LongVILA-R1-7B", **common_kwargs)
        if args.model in ("claude-opus", "claude-opus-4-7", "claude-opus-4-6"):
            # ClaudeAPIModel ignores max_gpu_mem; common_kwargs carries
            # enable_metrics. The bound class pins the API model id.
            return model_class(**common_kwargs)
        if args.model in ("openrouter", "gemini-3.1", "gpt-5.5", "grok-5"):
            # Resolve the slug. Precedence: explicit --openrouter-model wins,
            # else the known-alias table, else error out (for the generic
            # 'openrouter' key).
            from models.openrouter_api import KNOWN_SLUGS
            slug = args.openrouter_model or KNOWN_SLUGS.get(args.model)
            if not slug:
                raise ValueError(
                    "--model openrouter requires --openrouter-model <slug>. "
                    "Example: --model openrouter --openrouter-model google/gemini-3.1-pro"
                )
            or_kwargs = dict(common_kwargs)
            or_kwargs.pop("max_gpu_mem", None)  # OpenRouter ignores it
            return model_class(model_id=slug, **or_kwargs)
        return model_class("Qwen/Qwen2.5-VL-7B-Instruct", **common_kwargs)

    model = instantiate_model(args.max_frames)

    aggregate_metrics = None
    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
        metrics_cls = model._metrics.__class__  # type: ignore[attr-defined]
        aggregate_metrics = metrics_cls()

    def _metric_attr_names(metrics_obj: Any) -> List[str]:
        """Return list attributes that should be cleared or extended."""
        if metrics_obj is None:
            return []

        dataclass_fields = getattr(metrics_obj, '__dataclass_fields__', None)
        if dataclass_fields:
            return list(dataclass_fields.keys())

        return [name for name, value in vars(metrics_obj).items() if isinstance(value, list)]

    def _clear_metrics(metrics_obj: Any) -> None:
        """Clear all list-based metrics in place."""
        if metrics_obj is None:
            return

        for attr in _metric_attr_names(metrics_obj):
            attr_value = getattr(metrics_obj, attr, None)
            if isinstance(attr_value, list):
                attr_value.clear()

    def _extend_metrics(destination: Any, source: Any) -> None:
        """Append metrics from source into destination."""
        if destination is None or source is None:
            return

        dest_oom_timestamp = getattr(destination, 'first_oom_timestamp', None)
        src_oom_timestamp = getattr(source, 'first_oom_timestamp', None)
        if src_oom_timestamp is not None:
            if dest_oom_timestamp is None or src_oom_timestamp < dest_oom_timestamp:
                destination.first_oom_timestamp = float(src_oom_timestamp)

        for attr in _metric_attr_names(destination):
            dest_list = getattr(destination, attr, None)
            src_list = getattr(source, attr, None)
            if isinstance(dest_list, list) and isinstance(src_list, list):
                dest_list.extend(src_list)

    def _capture_first_oom(aggregate: Optional[PerformanceMetrics], timestamp: float) -> None:
        """Record the earliest OOM timestamp across the entire evaluation."""
        if aggregate is None:
            return

        current = getattr(aggregate, 'first_oom_timestamp', None)
        if current is None or float(timestamp) < current:
            aggregate.first_oom_timestamp = float(timestamp)

    def _has_metric_data(metrics_obj: Any) -> bool:
        """Return True if any tracked metric list contains data."""
        if metrics_obj is None:
            return False

        for attr in _metric_attr_names(metrics_obj):
            attr_value = getattr(metrics_obj, attr, None)
            if isinstance(attr_value, list) and attr_value:
                return True
        return False

    # Create processors
    question_processor = QuestionProcessor(
        args.verbose,
        args.no_describe,
        binary_questions=args.binary_questions,
        predictive_questions=args.predictive_questions,
        sequence_mode=use_sequence_mode,
        describe=args.describe,
    )
    sequence_formatter: Optional[SequenceFormatter] = None
    if use_sequence_mode:
        # Use SpatialSequenceFormatter for spatial mode, CommaSeparatedSequenceFormatter for sequential mode
        if args.eval_mode == "spatial":
            sequence_formatter = SpatialSequenceFormatter()
        else:
            sequence_formatter = CommaSeparatedSequenceFormatter()
    text_to_video_processor = (
        TextToVideoProcessor(fps=args.fps) if args.no_option_text and not use_sequence_mode else None
    )

    # Initialize wandb logger if metrics are enabled
    wandb_logger = WandbLogger(enabled=args.enable_metrics) if args.enable_metrics else None

    # Process all videos
    all_results = []
    correct_count = 0
    dont_know_count = 0
    question_outcomes: List[Dict[str, Any]] = []
    entropy_accuracy_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    prefix_fraction_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    correct_likelihood_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    # Initialize with saved results from previous runs (resume support)
    question_log_rows: List[Dict[str, Any]] = []
    if state_manager.enabled:
        saved_results = state_manager.get_saved_results()
        if saved_results:
            print(f"Loaded {len(saved_results)} question results from previous run(s)")
            question_log_rows.extend(saved_results)

    repeat_metrics = (
        {
            'pre': {'total': 0, 'correct': 0, 'dont_know': 0},
            'post': {'total': 0, 'correct': 0, 'dont_know': 0},
        }
        if args.predictive_questions
        else None
    )
    missing_repeat_warning_videos = set()
    invalid_repeat_warning_videos = set()
    missing_question_time_warnings = set()

    frame_budget = args.max_frames

    for i, video_data in enumerate(videos):
        variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching functions
        bucket = _extract_bucket_from_video_data(video_data)
        # Always print bucket info for clarity
        print(f"\n{'='*60}")
        print(f"Video {video_data.video_index} | Bucket: {bucket or 'None'} | Variant: {variant}")
        print(f"{'='*60}")

        # Check if model cannot process this bucket (VIDEO MODE ONLY - skip check entirely for sequence mode)
        # This allows per-model bucket limits (e.g., Phi-4 up to L512, LongVILA up to L256)
        # CRITICAL: Only applies when input_mode == "video" (sequence mode processes all buckets)
        if bucket and args.input_mode == "video":
            bucket_length = extract_bucket_length(bucket)
            model_limit = MODEL_VIDEO_BUCKET_LIMITS.get(args.model)

            if bucket_length is not None and model_limit is not None and bucket_length > model_limit:
                print(f"⚠️  SKIPPING: {args.model} cannot process bucket '{bucket}' "
                      f"({bucket_length} frames > {model_limit} frame limit for VIDEO mode)")
                print(f"    (Sequence mode would process this bucket without limits)")

                # Mark all questions as complete to skip on resume
                if state_manager.enabled:
                    question_ids = [
                        q.question_id or f"video{video_data.video_index}_q{qi}"
                        for qi, q in enumerate(video_data.questions)
                    ]
                    for qid in question_ids:
                        state_manager.mark_question_complete(
                            video_data.video_index, qid, result=None, variant=variant, bucket=bucket
                        )
                    state_manager.flush()
                continue
        # If input_mode == "sequence", the entire block above is skipped and all buckets process

        if args.verbose and i < 5:  # Debug first 5 videos
            state_key = state_manager._video_key(video_data.video_index, variant, bucket) if state_manager.enabled else None
            print(f"[DEBUG] State key: {state_key}")
        if state_manager.enabled and state_manager.is_video_completed(video_data.video_index, variant, bucket):
            print(f"Skipping completed video {video_data.video_index} variant {variant} bucket {bucket}")
            continue
        completed_questions = state_manager.get_completed_questions(video_data.video_index, variant, bucket) if state_manager.enabled else None
        if args.verbose and i < 5:  # Debug first 5 videos
            print(f"[DEBUG] Video {video_data.video_index}: completed_questions={completed_questions if completed_questions else 'EMPTY/None'}")
            if completed_questions:
                print(f"[DEBUG]   Sample: {list(completed_questions)[:5]}")
        current_max_frames = frame_budget
        oom_retry_count = 0
        video_results = None
        start_question_idx = 0
        partial_results: Optional[List[Dict[str, Any]]] = None
        carried_metrics: Optional[PerformanceMetrics] = None

        video_hit_question_limit = False  # Track if we stopped due to question limit

        while True:
            video_processor = None
            if not use_sequence_mode:
                video_processor = VideoProcessor(
                    args.model,
                    args.fps,
                    current_max_frames,
                    args.chunk_size,
                    model_max_frames=current_max_frames,
                )

            sequence_processor_instance: Optional[SequenceProcessor] = None
            if use_sequence_mode:
                raw_sequences = video_data.metadata.get('sequences_used')
                if not isinstance(raw_sequences, dict) or not raw_sequences:
                    raise ValueError(
                        f"Video {video_data.video_index} is missing 'sequences_used' details required for sequence mode."
                    )
                if sequence_formatter is None:
                    raise ValueError("Sequence formatter is not configured.")
                sequence_processor_instance = SequenceProcessor(
                    sequences_used=raw_sequences,
                    formatter=sequence_formatter,
                    print_chunks=True,
                )

            if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                if carried_metrics is not None:
                    model._metrics = copy.deepcopy(carried_metrics)
                    if hasattr(model, '_sync_state_memory_tracking_from_metrics'):
                        model._sync_state_memory_tracking_from_metrics()
                else:
                    _clear_metrics(model._metrics)
                    if hasattr(model, '_reset_state_memory_tracking'):
                        model._reset_state_memory_tracking()

            # Initialize dynamic batch sizer for text mode
            batch_sizer = None
            if args.input_mode == "sequence" and ENABLE_TEXT_BATCHING and sequence_processor_instance:
                batch_sizer = DynamicBatchSizer(initial_size=DEFAULT_BATCH_SIZE)
            # Initialize dynamic batch sizer for video mode (persists across videos
            # so batch reductions from earlier OOMs carry over).
            video_batch_sizer = None
            if args.input_mode == "video" and ENABLE_VIDEO_BATCHING and video_processor and args.model not in VIDEO_BATCHING_EXCLUDED_MODELS:
                _initial = MODEL_SPECIFIC_VIDEO_BATCH_SIZE.get(args.model, VIDEO_BATCH_SIZE)
                video_batch_sizer = DynamicBatchSizer(initial_size=_initial, min_size=1, max_size=_initial)

            try:
                # Use batched processing for text/sequence mode when enabled
                if args.input_mode == "sequence" and ENABLE_TEXT_BATCHING and sequence_processor_instance and batch_sizer and args.model not in TEXT_BATCHING_EXCLUDED_MODELS:
                    # Try with current batch size, reduce on OOM
                    max_batch_retries = 3
                    for batch_retry in range(max_batch_retries):
                        try:
                            current_batch_size = batch_sizer.get_size()
                            if args.verbose:
                                print(f"Attempting batch processing with batch_size={current_batch_size}")

                            # Use TRUE isolated batching for sequence mode
                            candidate_results = process_sequence_native_binary_batched_isolated(
                                video_data=video_data,
                                model=model,
                                sequence_processor=sequence_processor_instance,
                                question_processor=question_processor,
                                max_tokens=args.max_tokens,
                                batch_size=current_batch_size,
                                eval_mode=args.eval_mode,
                                verbose=args.verbose,
                                state_manager=state_manager if state_manager.enabled else None,
                                question_log_path=args.question_log_csv,
                                question_log_rows=question_log_rows,
                                limit_questions=args.limit_questions,
                                start_question_index=start_question_idx,
                                completed_set=completed_questions,
                            )
                            hit_limit = len(candidate_results) >= (args.limit_questions or float('inf'))

                            # Success! Maybe increase batch size for next video
                            batch_sizer.increase_on_success()
                            break

                        except RuntimeError as batch_exc:
                            # Check if it's a CUDA OOM error
                            if "out of memory" in str(batch_exc).lower() or "oom" in str(batch_exc).lower():
                                if current_batch_size == 1:
                                    # Can't reduce further, re-raise
                                    raise

                                new_size = batch_sizer.reduce_on_oom()
                                print(f"⚠️ OOM during batching with batch_size={current_batch_size}. Reducing to {new_size} and retrying...")

                                # Clear GPU memory
                                model.clear_context()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                if batch_retry == max_batch_retries - 1:
                                    raise  # Max retries reached
                            else:
                                raise  # Non-OOM error, don't retry
                elif args.input_mode == "video" and ENABLE_VIDEO_BATCHING and video_processor and args.model not in VIDEO_BATCHING_EXCLUDED_MODELS:
                    # Use batched processing for video mode native binary questions.
                    # Auto-halve on OOM (mirrors text-mode pattern at ~line 3974).
                    # video_batch_sizer was initialized above the try-block.
                    initial_video_batch = video_batch_sizer.max_size
                    max_batch_retries = max(3, int(math.log2(initial_video_batch)) + 2)
                    for batch_retry in range(max_batch_retries):
                        try:
                            current_batch_size = video_batch_sizer.get_size()
                            if args.verbose:
                                print(f"Attempting video batch processing with batch_size={current_batch_size}")

                            # Use TRUE isolated batching for video mode
                            if args.verbose and completed_questions:
                                print(f"[DEBUG] Passing {len(completed_questions)} completed questions to batching function")
                            candidate_results = process_video_native_binary_batched_isolated(
                                video_data=video_data,
                                model=model,
                                video_processor=video_processor,
                                question_processor=question_processor,
                                max_tokens=args.max_tokens,
                                batch_size=current_batch_size,
                                eval_mode=args.eval_mode,
                                verbose=args.verbose,
                                state_manager=state_manager if state_manager.enabled else None,
                                question_log_path=args.question_log_csv,
                                question_log_rows=question_log_rows,
                                limit_questions=args.limit_questions,
                                start_question_index=start_question_idx,
                                completed_set=completed_questions,
                            )
                            hit_limit = len(candidate_results) >= (args.limit_questions or float('inf'))
                            break

                        except RuntimeError as batch_exc:
                            if "out of memory" in str(batch_exc).lower() or "oom" in str(batch_exc).lower():
                                if current_batch_size == 1:
                                    # Can't reduce further, re-raise
                                    raise
                                new_size = video_batch_sizer.reduce_on_oom()
                                print(f"⚠️ OOM during video batching with batch_size={current_batch_size}. Reducing to {new_size} and retrying...")
                                # Clear GPU memory before retry
                                if hasattr(model, 'clear_context'):
                                    model.clear_context()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                if batch_retry == max_batch_retries - 1:
                                    raise  # Max retries reached
                            else:
                                raise  # Non-OOM error, don't retry

                else:
                    # Use original processing (video mode or batching disabled)
                    candidate_results, hit_limit = process_video_with_questions(
                        video_data,
                        model,
                        video_processor,
                        question_processor,
                        args.max_tokens,
                        current_max_frames,
                        text_to_video_processor,
                        args.no_option_text,
                        start_question_index=start_question_idx,
                        existing_results=partial_results,
                        binary_questions=args.binary_questions,
                        wandb_logger=wandb_logger,
                        input_mode=args.input_mode,
                        sequence_processor=sequence_processor_instance,
                        sequence_format_label=args.sequence_format,
                        eval_mode=args.eval_mode,
                        completed_questions=completed_questions,
                        state_manager=state_manager if state_manager.enabled else None,
                        question_log_path=args.question_log_csv,
                        question_log_rows=question_log_rows,
                        limit_questions=args.limit_questions,
                    )
                video_results = candidate_results
                video_hit_question_limit = hit_limit
                frame_budget = current_max_frames
                start_question_idx = 0
                partial_results = None
                oom_retry_count = 0
                carried_metrics = None
                break
            except QuestionOOMRetry as retry_exc:
                exc = retry_exc.original_exception
                handled = False

                if args.enable_metrics:
                    _capture_first_oom(aggregate_metrics, retry_exc.oom_timestamp)

                # 2026-04-26: silent frame reduction is permanently disabled.
                # Previously --limit_frames_on_oom would shrink current_max_frames
                # by 10% and retry, but rows written under the reduced budget had
                # a different (smaller) frame budget than configured, contaminating
                # accuracy stats. Always exit loudly on OOM now; the flag is a no-op.
                if args.limit_frames_on_oom:
                    print(
                        "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                        "Bump --gres=gpu:N or lower --max_frames in the script.",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise exc

                if args.restart_on_oom:
                    print(
                        "Restarting model after CUDA OOM to ensure clean state.",
                        flush=True,
                    )
                    try:
                        if hasattr(model, 'teardown_after_oom'):
                            model.teardown_after_oom()
                        elif hasattr(model, 'clear_context'):
                            model.clear_context()
                    except Exception:
                        pass
                    try:
                        if hasattr(model, 'close'):
                            model.close()
                    except Exception:
                        pass
                    try:
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    model = instantiate_model(current_max_frames)
                    handled = True
                    # Metrics instance belongs to the new model; ensure it is clean before retry.
                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        _clear_metrics(model._metrics)
                        if hasattr(model, '_reset_state_memory_tracking'):
                            model._reset_state_memory_tracking()

                if handled:
                    carried_metrics = retry_exc.partial_metrics
                    start_question_idx = retry_exc.question_index
                    partial_results = retry_exc.partial_results
                    oom_retry_count += 1
                    if oom_retry_count > args.max_oom_retries:
                        raise RuntimeError(
                            f"Exceeded maximum CUDA OOM retries ({args.max_oom_retries}) for video index {i} while using max_frames={current_max_frames}."
                        ) from exc
                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        _clear_metrics(model._metrics)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    if state_manager.enabled:
                        completed_questions = state_manager.get_completed_questions(video_data.video_index, variant, bucket)
                    continue

                # Not handled; re-raise original OOM exception chain.
                # 2026-04-26: print the clear stderr line first so slurm logs make
                # the failure mode obvious without scrolling through the trace.
                print(
                    "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                    "Bump --gres=gpu:N or lower --max_frames in the script.",
                    file=sys.stderr,
                    flush=True,
                )
                raise exc

            except Exception as exc:
                if _is_cuda_oom_error(exc):
                    handled = False

                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        timestamp = getattr(model._metrics, 'first_oom_timestamp', None)
                        if timestamp is not None:
                            _capture_first_oom(aggregate_metrics, timestamp)

                    # 2026-04-26: silent frame reduction is permanently disabled.
                    # See QuestionOOMRetry handler above for rationale.
                    if args.limit_frames_on_oom:
                        print(
                            "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                            "Bump --gres=gpu:N or lower --max_frames in the script.",
                            file=sys.stderr,
                            flush=True,
                        )
                        raise

                    if args.restart_on_oom:
                        print(
                            "Restarting model after CUDA OOM to ensure clean state.",
                            flush=True,
                        )
                        try:
                            if hasattr(model, 'teardown_after_oom'):
                                model.teardown_after_oom()
                            elif hasattr(model, 'clear_context'):
                                model.clear_context()
                        except Exception:
                            pass
                        try:
                            if hasattr(model, 'close'):
                                model.close()
                        except Exception:
                            pass
                        try:
                            if hasattr(model, 'shutdown'):
                                model.shutdown()
                        except Exception:
                            pass

                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                        model = instantiate_model(current_max_frames)
                        handled = True
                        # Metrics instance belongs to the new model; ensure it is clean before retry.
                        if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                            _clear_metrics(model._metrics)
                            if hasattr(model, '_reset_state_memory_tracking'):
                                model._reset_state_memory_tracking()

                    if handled:
                        start_question_idx = 0
                        partial_results = None
                        oom_retry_count += 1
                        if oom_retry_count > args.max_oom_retries:
                            raise RuntimeError(
                                f"Exceeded maximum CUDA OOM retries ({args.max_oom_retries}) for video index {i} while using max_frames={current_max_frames}."
                            ) from exc
                        if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                            _clear_metrics(model._metrics)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        continue

                if _is_cuda_oom_error(exc):
                    # 2026-04-26: print the clear stderr line before re-raising.
                    print(
                        "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                        "Bump --gres=gpu:N or lower --max_frames in the script.",
                        file=sys.stderr,
                        flush=True,
                    )
                raise

        # Only mark video as complete if we didn't stop due to question limit
        # This allows resuming with a higher limit to process additional questions
        if state_manager.enabled and not termination_requested() and not video_hit_question_limit:
            state_manager.mark_video_complete(video_data.video_index, variant=variant, bucket=bucket)
        all_results.extend(video_results)

        if termination_requested():
            print("Termination requested; stopping after current video.")
            break

        # Calculate video-level stats
        video_correct = sum(1 for result in video_results if result['is_correct'])
        video_accuracy = video_correct / len(video_results) if video_results else 0

        # Log to wandb if enabled
        if wandb_logger:
            additional_metrics = _summarize_video_question_stats(video_results)
            wandb_logger.log_video_completion(
                num_questions=len(video_results),
                video_accuracy=video_accuracy,
                model_metrics=model._metrics if hasattr(model, '_metrics') else None,
                model_name=model_name,
                extra_metrics=additional_metrics,
            )

        for result in video_results:
            if result['is_correct']:
                correct_count += 1
            if result['is_dont_know']:
                dont_know_count += 1

            question_time = result.get('question_time')
            if question_time is not None:
                question_outcomes.append(
                    {
                        'timestamp_sec': float(question_time),
                        'is_correct': 1 if result.get('is_correct') else 0,
                        'video_index': video_data.video_index,
                        'question_id': result.get('question_id'),
                    }
                )

            entropy_value = result.get('entropy_prefix_mean')
            if entropy_value is not None:
                entropy_accuracy_records.append((float(entropy_value), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

            prefix_fraction = result.get('prefix_match_fraction')
            if prefix_fraction is not None:
                prefix_fraction_records.append((float(prefix_fraction), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

            likelihood_value = result.get('correct_option_likelihood')
            if likelihood_value is not None:
                correct_likelihood_records.append((float(likelihood_value), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

        if repeat_metrics is not None:
            raw_repeat_time = video_data.metadata.get('repeated_time')
            video_index = video_data.video_index
            if raw_repeat_time is None:
                if video_index not in missing_repeat_warning_videos:
                    print(
                        f"⚠️  Video {video_index} is missing 'repeated_time'; "
                        "repeat-aware metrics will skip it.",
                        flush=True,
                    )
                    missing_repeat_warning_videos.add(video_index)
            else:
                try:
                    repeat_cutoff = float(raw_repeat_time)
                except (TypeError, ValueError):
                    if video_index not in invalid_repeat_warning_videos:
                        print(
                            f"⚠️  Video {video_index} has an invalid 'repeated_time' value "
                            "and will be ignored for repeat-aware metrics.",
                            flush=True,
                        )
                        invalid_repeat_warning_videos.add(video_index)
                else:
                    for result in video_results:
                        question_time = result.get('question_time')
                        if question_time is None:
                            question_identifier = result.get('question_id', '?')
                            question_key = (video_index, question_identifier)
                            if question_key not in missing_question_time_warnings:
                                print(
                                    f"⚠️  Question {question_identifier} in video {video_index} "
                                    "is missing 'question_time'; skipping repeat-aware metrics.",
                                    flush=True,
                                )
                                missing_question_time_warnings.add(question_key)
                            continue

                        bucket_key = 'pre' if float(question_time) < repeat_cutoff else 'post'
                        bucket = repeat_metrics[bucket_key]
                        bucket['total'] += 1
                        if result.get('is_correct'):
                            bucket['correct'] += 1
                        if result.get('is_dont_know'):
                            bucket['dont_know'] += 1

        if args.enable_metrics and aggregate_metrics is not None and getattr(model, '_metrics', None) is not None:
            _extend_metrics(aggregate_metrics, model._metrics)

    extra_final_metrics = _aggregate_global_question_stats(
        entropy_accuracy_records,
        prefix_fraction_records,
        correct_likelihood_records,
    )

    summary_metrics = None
    if args.enable_metrics:
        if aggregate_metrics is not None and _has_metric_data(aggregate_metrics):
            summary_metrics = aggregate_metrics
        elif getattr(model, '_metrics', None) is not None and _has_metric_data(model._metrics):
            summary_metrics = model._metrics

    if summary_metrics is not None:
        summary_cls = type(model)
        if hasattr(summary_cls, 'render_metrics_summary'):
            summary_cls.render_metrics_summary(summary_metrics)
        elif hasattr(model, 'print_metrics_summary'):
            model.print_metrics_summary()

        if hasattr(summary_metrics, 'video_timestamps_add_video') and hasattr(summary_cls, 'analyze_metrics'):
            try:
                summary_cls.analyze_metrics(summary_metrics, print_results=True)
            except Exception as exc:
                print(f"⚠️  Curve fitting analysis failed: {exc}")

    if state_manager.enabled:
        state_manager.flush()

    # Completion pass: Fill in gaps for incomplete buckets
    # This happens AFTER the limited main pass, so these questions don't count toward limit
    if state_manager.enabled and not termination_requested():
        try:
            completion_count = complete_remaining_questions(
                videos=videos,
                model=model,
                state_manager=state_manager,
                args=args,
                question_processor=question_processor,
                sequence_formatter=sequence_formatter,
                use_sequence_mode=use_sequence_mode,
            )
            print(f"Completion pass processed {completion_count} additional questions (not included in stats)")
        except Exception as e:
            print(f"⚠️ Completion pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Apply limit_questions PER BUCKET for stats
    if args.limit_questions is not None:
        from collections import defaultdict

        # If no new results (all buckets complete), use saved results from state for stats
        results_for_stats = all_results
        if len(all_results) == 0 and state_manager.enabled:
            saved_for_stats = state_manager.get_saved_results()
            if saved_for_stats:
                print(f"\nNo new results this run. Using {len(saved_for_stats)} saved results for stats calculation.")
                results_for_stats = saved_for_stats

        # Group by bucket
        bucket_results = defaultdict(list)
        for result in results_for_stats:
            bucket = result.get('bucket')
            if bucket:
                bucket_results[bucket].append(result)

        # Limit per bucket (respecting eval_mode)
        stats_results = []
        over_limit_buckets = []

        for bucket in sorted(bucket_results.keys()):
            bucket_data = bucket_results[bucket]

            # Filter by eval_mode
            valid_results = []
            for res in bucket_data:
                if args.eval_mode == 'sequential':
                    variant = res.get('question_variant', '').lower()
                    if variant == 'spatial':
                        continue
                valid_results.append(res)

            # Check if over limit
            if len(valid_results) > args.limit_questions:
                over_limit_buckets.append((bucket, len(valid_results)))

            # Take first limit_questions
            limited = valid_results[:args.limit_questions]
            stats_results.extend(limited)

        print(f"\nApplied limit of {args.limit_questions} per bucket (eval_mode={args.eval_mode}):")
        print(f"  Total results available: {len(results_for_stats)}")
        total_valid_all = sum(len([r for r in bucket_results[b]
                                    if args.eval_mode != 'sequential'
                                    or r.get('question_variant','').lower() != 'spatial'])
                              for b in bucket_results)
        print(f"  Valid for eval_mode: {total_valid_all}")
        print(f"  Used for stats: {len(stats_results)} ({len(bucket_results)} buckets)")

        # Warn about over-limit buckets (old state with extra questions)
        if over_limit_buckets:
            print(f"\n⚠️  WARNING: {len(over_limit_buckets)} buckets have more than {args.limit_questions} questions:")
            for bucket, count in over_limit_buckets:
                print(f"    {bucket}: {count} questions (using first {args.limit_questions} for stats, keeping all in state)")
    else:
        stats_results = all_results

    # Final cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print final results
    # ALWAYS recalculate correct_count and dont_know_count from stats_results
    # (stats_results may be limited per-bucket, but old counts were from all_results)
    correct_count = sum(1 for r in stats_results if r.get('is_correct', False))
    dont_know_count = sum(1 for r in stats_results if r.get('is_dont_know', False))

    total_questions = len(stats_results)
    accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    idk_rate = (dont_know_count / total_questions * 100) if total_questions > 0 else 0
    answered_questions = total_questions - dont_know_count
    answered_accuracy = (correct_count / answered_questions * 100) if answered_questions > 0 else 0

    # Calculate spatial vs sequential breakdown
    spatial_stats = {'total': 0, 'correct': 0, 'dont_know': 0}
    sequential_stats = {'total': 0, 'correct': 0, 'dont_know': 0}

    for result in stats_results:
        variant = result.get('question_variant', '').lower()
        is_correct = result.get('is_correct', False)
        is_dont_know = result.get('is_dont_know', False)

        if variant == 'spatial':
            spatial_stats['total'] += 1
            if is_correct:
                spatial_stats['correct'] += 1
            if is_dont_know:
                spatial_stats['dont_know'] += 1
        elif variant == 'sequential':
            sequential_stats['total'] += 1
            if is_correct:
                sequential_stats['correct'] += 1
            if is_dont_know:
                sequential_stats['dont_know'] += 1

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Don't Know Responses: {dont_know_count}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"IDK Rate: {idk_rate:.1f}%")
    print(f"Accuracy (excluding IDK): {answered_accuracy:.1f}%")

    # Print spatial vs sequential breakdown
    if spatial_stats['total'] > 0 or sequential_stats['total'] > 0:
        print(f"\nBreakdown by Question Type:")
        if spatial_stats['total'] > 0:
            spatial_acc = (spatial_stats['correct'] / spatial_stats['total'] * 100)
            spatial_answered = spatial_stats['total'] - spatial_stats['dont_know']
            spatial_acc_excl = (spatial_stats['correct'] / spatial_answered * 100) if spatial_answered > 0 else 0
            print(f"  Spatial: {spatial_stats['total']} questions, {spatial_acc:.1f}% accurate ({spatial_acc_excl:.1f}% excl. IDK)")
        if sequential_stats['total'] > 0:
            seq_acc = (sequential_stats['correct'] / sequential_stats['total'] * 100)
            seq_answered = sequential_stats['total'] - sequential_stats['dont_know']
            seq_acc_excl = (sequential_stats['correct'] / seq_answered * 100) if seq_answered > 0 else 0
            print(f"  Sequential: {sequential_stats['total']} questions, {seq_acc:.1f}% accurate ({seq_acc_excl:.1f}% excl. IDK)")

    if extra_final_metrics:
        _print_question_stat_summaries(extra_final_metrics)
    print(f"{'='*60}")

    # Compute and display per-bucket stats
    def compute_per_bucket_stats(results):
        from collections import defaultdict
        bucket_stats = defaultdict(lambda: {
            'spatial': {'correct': 0, 'total': 0, 'dont_know': 0},
            'sequential': {'correct': 0, 'total': 0, 'dont_know': 0},
        })

        for result in results:
            bucket = result.get('bucket')
            if not bucket:
                continue

            variant = result.get('question_variant', '').lower()
            category = 'spatial' if variant == 'spatial' else 'sequential'

            stats = bucket_stats[bucket][category]
            stats['total'] += 1
            if result.get('is_correct', False):
                stats['correct'] += 1
            if result.get('is_dont_know', False):
                stats['dont_know'] += 1

        return dict(bucket_stats)

    bucket_stats = compute_per_bucket_stats(stats_results)
    if bucket_stats:
        print(f"\n{'='*60}")
        print("PER-BUCKET METRICS")
        print(f"{'='*60}")

        for bucket in sorted(bucket_stats.keys()):
            print(f"\n{bucket}:")
            for category in ['spatial', 'sequential']:
                stats = bucket_stats[bucket][category]
                if stats['total'] == 0:
                    continue

                correct, total, dont_know = stats['correct'], stats['total'], stats['dont_know']
                answered = total - dont_know
                raw_acc = (correct / total * 100) if total > 0 else 0
                adj_acc = (correct / answered * 100) if answered > 0 else 0

                print(f"  {category.capitalize():12s}: Acc={raw_acc:5.1f}% (adj={adj_acc:5.1f}%) | answered={answered}/{total}")
        print(f"{'='*60}")

    # Compute and display temporal distribution for L032+ buckets
    def extract_length_from_bucket(bucket: str):
        """Extract length value from bucket name (e.g., 'L008' -> 8)."""
        import re
        match = re.search(r'L(\d+)', bucket)
        if match:
            return int(match.group(1))
        return None

    def compute_temporal_distribution(results):
        """Compute accuracy and recall by temporal position for L032+ buckets."""
        from collections import defaultdict
        # Structure: length -> fourth -> {correct, total, dont_know, tp, fn}
        length_fourth_stats = defaultdict(lambda: [
            {'correct': 0, 'total': 0, 'dont_know': 0, 'tp': 0, 'fn': 0}
            for _ in range(4)
        ])

        for result in results:
            bucket = result.get('bucket')
            if not bucket:
                continue

            length = extract_length_from_bucket(bucket)
            if not length or length < 32:
                continue  # Only L032+

            # Get clip timing
            clip_start = result.get('clip_start_time')
            clip_end = result.get('clip_end_time')
            if clip_start is None or clip_end is None:
                continue

            # Check if answer was yes (clip is present)
            # For native binary, check if correct answer is "0" (yes)
            correct_answer = result.get('correct')
            if correct_answer != '0':  # Not a "yes" question
                continue

            # Video duration equals the length (L032 = 32 seconds, etc.)
            video_duration = float(length)
            fourth_duration = video_duration / 4.0

            # Calculate which fourths the clip overlaps
            start_fourth = int(clip_start / fourth_duration)
            end_fourth = int(clip_end / fourth_duration)

            # Clamp to valid range [0, 3]
            start_fourth = max(0, min(3, start_fourth))
            end_fourth = max(0, min(3, end_fourth))

            # Determine fourths covered
            fourths_covered = list(range(start_fourth, end_fourth + 1))
            if len(fourths_covered) == 0:
                continue

            # Check correctness and predictions
            is_correct = result.get('is_correct', False)
            is_dont_know = result.get('is_dont_know', False)

            # For recall calculation
            predicted_yes = (result.get('predicted') == '0')
            actual_yes = True  # Filtered for answer=yes above

            # Increment each fourth this clip touches
            weight = 1.0 / len(fourths_covered) if len(fourths_covered) > 1 else 1.0

            for fourth_idx in fourths_covered:
                stats = length_fourth_stats[length][fourth_idx]
                stats['total'] += weight

                if is_correct:
                    stats['correct'] += weight
                if is_dont_know:
                    stats['dont_know'] += weight

                # For recall (precision omitted - always 100% for yes-only questions)
                if actual_yes and predicted_yes:
                    stats['tp'] += weight
                elif actual_yes and not predicted_yes and not is_dont_know:
                    stats['fn'] += weight

        return dict(length_fourth_stats)

    temporal_dist = compute_temporal_distribution(stats_results)
    if temporal_dist:
        print(f"\n{'='*60}")
        print("TEMPORAL METRICS BY FOURTH/QUARTER (L032+ buckets)")
        print(f"{'='*60}")
        print("For each fourth: metrics for questions where answer=yes (clip is present)")
        print("NOTE: 'No' questions excluded (undefined temporal position)")
        print("      Precision omitted (always 100% for yes-only questions)")

        for length in sorted(temporal_dist.keys()):
            fourth_stats = temporal_dist[length]

            # Calculate totals
            total_questions = sum(stats['total'] for stats in fourth_stats)

            print(f"\nL{length:04d} (total true questions: {total_questions:.1f}):")

            # Row 1: Fourth numbers
            print("  Fourth:         ", end="")
            for i in range(4):
                print(f"{i+1:12d}", end="  ")
            print()

            # Row 2: Question counts
            print("  Questions:      ", end="")
            for stats in fourth_stats:
                if stats['total'] > 0:
                    print(f"{stats['total']:12.1f}", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 3: Accuracy (raw)
            print("  Accuracy:       ", end="")
            for stats in fourth_stats:
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"{accuracy:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 4: Accuracy (adjusted for uncertainty)
            print("  Acc (adj IDK):  ", end="")
            for stats in fourth_stats:
                answered = stats['total'] - stats['dont_know']
                if answered > 0:
                    adj_acc = (stats['correct'] / answered) * 100
                    print(f"{adj_acc:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 5: Recall (precision omitted - always 100%)
            print("  Recall:         ", end="")
            for stats in fourth_stats:
                tp_fn = stats['tp'] + stats['fn']
                if tp_fn > 0:
                    recall = (stats['tp'] / tp_fn) * 100
                    print(f"{recall:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()
        print(f"{'='*60}")

    if repeat_metrics is not None:
        def _format_rate(numerator: int, denominator: int) -> str:
            if denominator == 0:
                return "N/A"
            rate = (numerator / denominator) * 100
            return f"{rate:.1f}%"

        pre_bucket = repeat_metrics['pre']
        post_bucket = repeat_metrics['post']

        print("Metrics Before Repeat:")
        print(f"  Questions: {pre_bucket['total']}")
        print(f"  Correctness rate: {_format_rate(pre_bucket['correct'], pre_bucket['total'])}")
        print(f"  IDK rate: {_format_rate(pre_bucket['dont_know'], pre_bucket['total'])}")

        print("Metrics After Repeat:")
        print(f"  Questions: {post_bucket['total']}")
        print(
            f"  Correctness rate: {_format_rate(post_bucket['correct'], post_bucket['total'])}"
        )
        print(f"  IDK rate: {_format_rate(post_bucket['dont_know'], post_bucket['total'])}")

    # Log final results to wandb
    if wandb_logger:
        wandb_logger.log_final_results(
            total_videos=len(videos),
            total_questions=total_questions,
            overall_accuracy=accuracy,
            idk_rate=idk_rate,
            model_name=model_name,
            extra_metrics=extra_final_metrics,
        )
        if question_outcomes:
            wandb_logger.log_question_accuracy_scatter(question_outcomes)
        wandb_logger.finish()

    if args.question_log_csv:
        try:
            destination = write_question_log_csv(args.question_log_csv, question_log_rows)
            print(f"Question log CSV written to {destination}", flush=True)
        except Exception as exc:
            print(
                f"Failed to write question log CSV to {args.question_log_csv}: {exc}",
                flush=True,
            )


def _summarize_video_question_stats(video_results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def _collect_means(result_key: str, metric_prefix: str) -> None:
        values = [float(result[result_key]) for result in video_results if result.get(result_key) is not None]
        if not values:
            return
        metrics[f'{metric_prefix}_mean'] = sum(values) / len(values)
        correct_values = [
            float(result[result_key])
            for result in video_results
            if result.get(result_key) is not None and result.get('is_correct')
        ]
        if correct_values:
            metrics[f'{metric_prefix}_correct_mean'] = sum(correct_values) / len(correct_values)

    _collect_means('entropy_prefix_mean', 'video_entropy')
    _collect_means('prefix_match_fraction', 'video_prefix_fraction')
    _collect_means('correct_option_likelihood', 'video_correct_likelihood')

    return metrics


def _aggregate_global_question_stats(
    entropy_records: List[Tuple[float, bool, bool]],
    prefix_records: List[Tuple[float, bool, bool]],
    likelihood_records: List[Tuple[float, bool, bool]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for records, key, label in (
        (entropy_records, 'entropy', 'Entropy'),
        (prefix_records, 'prefix_fraction', 'Prefix Fraction'),
        (likelihood_records, 'correct_likelihood', 'Correct Option Likelihood'),
    ):
        metric_summary = _summarize_metric_records(records, label)
        if metric_summary:
            summary[key] = metric_summary

    return summary


def _summarize_metric_records(
    records: List[Tuple[float, bool, bool]],
    label: str,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    values = [value for value, _, _ in records]
    summary = {
        'label': label,
        'mean': sum(values) / len(values),
    }
    correct_values = [value for value, is_correct, _ in records if is_correct]
    incorrect_values = [value for value, is_correct, _ in records if not is_correct]
    if correct_values:
        summary['mean_correct'] = sum(correct_values) / len(correct_values)
    if incorrect_values:
        summary['mean_incorrect'] = sum(incorrect_values) / len(incorrect_values)
    summary['bins'] = _build_accuracy_bins(records, label)
    return summary


def _build_accuracy_bins(records: List[Tuple[float, bool, bool]], label: str, bins: int = 3) -> List[Dict[str, Any]]:
    if not records:
        return []
    sorted_records = sorted(records, key=lambda item: item[0])
    bin_count = min(bins, len(sorted_records))
    if bin_count <= 0:
        return []
    base_size = max(1, len(sorted_records) // bin_count)
    result: List[Dict[str, Any]] = []
    start_index = 0
    for idx in range(bin_count):
        if idx == bin_count - 1:
            end_index = len(sorted_records)
        else:
            end_index = min(len(sorted_records), start_index + base_size)
        subset = sorted_records[start_index:end_index]
        start_index = end_index
        if not subset:
            continue

        # Calculate raw accuracy
        correct_count = sum(1 for _, is_correct, _ in subset if is_correct)
        accuracy = (correct_count / len(subset)) * 100.0

        # Calculate uncertainty-adjusted accuracy
        answered_count = sum(1 for _, _, is_dont_know in subset if not is_dont_know)
        adj_accuracy = (correct_count / answered_count * 100.0) if answered_count > 0 else 0.0

        value_range = (subset[0][0], subset[-1][0])
        result.append(
            {
                'label': f"{label} Bin {idx + 1}",
                'range': value_range,
                'accuracy': accuracy,
                'accuracy_adj': adj_accuracy,
                'count': len(subset),
            }
        )
    return result


def _print_question_stat_summaries(summary: Dict[str, Any]) -> None:
    for metric_key, data in summary.items():
        label = data.get('label', metric_key.replace('_', ' ').title())
        mean_value = data.get('mean')
        if mean_value is None:
            continue
        print(f"{label} Mean: {mean_value:.3f}")
        correct_mean = data.get('mean_correct')
        incorrect_mean = data.get('mean_incorrect')
        if correct_mean is not None or incorrect_mean is not None:
            print(
                f"  {label} (correct/incorrect): "
                f"{(correct_mean or 0.0):.3f} / {(incorrect_mean or 0.0):.3f}"
            )
        for bucket in data.get('bins', []):
            rng = bucket.get('range', (0.0, 0.0))
            acc = bucket.get('accuracy', 0.0)
            adj_acc = bucket.get('accuracy_adj', 0.0)
            print(
                f"  {bucket['label']}: accuracy {acc:.1f}% (adj {adj_acc:.1f}%)"
                f" (range {rng[0]:.3f}–{rng[1]:.3f}, n={bucket['count']})"
            )


def _extract_token_stats(model: Any, response: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    fallback_tokens = _estimate_token_count(response)
    stats['response_token_count'] = fallback_tokens

    provider = getattr(model, 'get_last_response_token_stats', None)
    if callable(provider):
        try:
            token_data = provider() or {}
        except Exception:
            token_data = {}
        if token_data:
            stats.update(token_data)
            total_tokens = token_data.get('total_output_tokens')
            if isinstance(total_tokens, (int, float)):
                stats['response_token_count'] = int(total_tokens)
    return stats


def _estimate_token_count(response: str) -> int:
    if not response:
        return 0
    tokens = response.strip().split()
    return len(tokens)


if __name__ == "__main__":
    main()
