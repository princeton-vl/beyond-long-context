"""FLOP validation helpers for the video-language benchmark."""

from __future__ import annotations

import dataclasses
from datetime import datetime
import functools
import gc
import inspect
import logging
import math
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

from utils.paths import get_model_cache_dir

try:
    from .flops_calc import vision_language_flops
except ImportError:  # pragma: no cover - allows running as a loose script
    from metrics.flops_calc import vision_language_flops  # type: ignore[import]


def compute_sample_qwen_flops(
    vision_frames: float = 8.0,
    vision_height: int = 448,
    vision_width: int = 448,
    prompt_tokens: int = 1024,
    generated_tokens: int = 256,
) -> float:
    """Return the reference FLOP estimate for Qwen2.5 VL using config defaults."""

    return vision_language_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
        vision_patch_size=14,
        vision_hidden_size=1280,
        num_full_attn_layers_vision=4,
        num_sliding_attn_layers_vision=28,
        num_mamba_layers_vision=0,
        num_mlp_layers_vision=32,
        vision_window_size=(2, 2),
        vision_is_2d=True,
        vision_num_attn_heads=16,
        vision_gqa=False,
        vision_gqa_groups=0,
        vision_mlp_expansion=3420 / 1280,
        merger=True,
        merger_mlp_expansion=1.0,
        lang_seq_len=prompt_tokens,
        lang_hidden_size=3584,
        num_full_attn_layers_lang=28,
        num_sliding_attn_layers_lang=0,
        num_mamba_layers_lang=0,
        num_mlp_layers_lang=28,
        lang_window_size=None,
        lang_is_2d=False,
        lang_num_attn_heads=28,
        lang_gqa=True,
        lang_gqa_groups=4,
        lang_mlp_expansion=18944 / 3584,
        swiglu=False,
        vocab_size=152_064,
        num_generated=generated_tokens,
        do_backward=False,
    )


# =============================================================================
# VALIDATION TESTER CODE
# =============================================================================

def extract_flops_from_profiler(prof):
    """Extract total FLOPs from PyTorch profiler events."""
    total_flops = 0
    for event in prof.events():
        if hasattr(event, 'flops') and event.flops is not None:
            total_flops += event.flops
    return total_flops

def profile_model_forward(model, inputs, max_new_tokens):
    """Profile a single forward pass and return actual FLOPs."""
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        profile_memory=False,
    ) as prof:
        with record_function("model_generation"):
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    use_cache=True  # Ensure KV caching is enabled
                )

    actual_flops = extract_flops_from_profiler(prof)
    return generated_ids, actual_flops, prof

def test_flop_accuracy():
    """
    Test FLOP calculations against actual Qwen2.5-VL model runs.
    Validates with different input lengths and measures actual vs predicted FLOPs using hardware profiling.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from torch.utils.flop_counter import FlopCounterMode

    print("\n" + "="*80)
    print("FLOP ACCURACY VALIDATION TEST")
    print("="*80)

    # Load model (using smaller cache dir as specified in CLAUDE.md)
    cache_dir = get_model_cache_dir()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir
    )
    # Use min_pixels/max_pixels divisible by 28*28=784 for clean vision tokenization
    min_pixels = 256 * 28 * 28  # 200,704 pixels
    max_pixels = 1280 * 28 * 28  # 1,003,520 pixels
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    print("Model loaded successfully")

    # Generate test images with dimensions divisible by 28
    def create_test_image(height, width):
        """Create a random test image with specified dimensions."""
        from PIL import Image
        import numpy as np
        # Create random RGB image
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    def create_test_video(num_frames, height, width):
        """Create a test video as an MP4 file."""
        import tempfile
        import cv2
        from PIL import Image
        import numpy as np

        # Create temporary MP4 file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8  # Standard frame rate
        video_writer = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

        # Generate and write frames
        for i in range(num_frames):
            # Create random frame
            frame_rgb = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        return temp_file.name

    # Test cases with varying complexity
    test_cases = [
        # {
        #     "name": "Short text only",
        #     "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}],
        #     "max_new_tokens": 5,
        #     "expected_vision_tokens": 0,
        #     "image_dims": None
        # },
        # {
        #     "name": "Small image (224x224)",
        #     "messages": [{
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": None},  # Will be filled with generated image
        #             {"type": "text", "text": "Describe this image briefly."}
        #         ]
        #     }],
        #     "max_new_tokens": 10,
        #     "expected_vision_tokens": None,
        #     "image_dims": (848, 848)  # 8x8 patches = 64 tokens
        # },
        # {
        #     "name": "Medium image (448x448)",
        #     "messages": [{
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": None},
        #             {"type": "text", "text": "Describe this image in detail."}
        #         ]
        #     }],
        #     "max_new_tokens": 20,
        #     "expected_vision_tokens": None,
        #     "image_dims": (448, 448)  # 32x32 patches = 1024 tokens
        # },
        # {
        #     "name": "Large image (672x672)",
        #     "messages": [{
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": None},
        #             {"type": "text", "text": "Describe this image in detail.Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Describe this image in detail.Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Describe this image in detail.Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Describe this image in detail.Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Describe this image in detail.Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze Analyze this image comprehensively.Analyze this image comprehensively.Analyze "}
        #         ]
        #     }],
        #     "max_new_tokens": 40,
        #     "expected_vision_tokens": None,
        #     "image_dims": (672, 672)  # 48x48 patches = 2304 tokens
        # },
        {
            "name": "Short video (16 frames, 224x224)",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "video": None},  # Will be filled with generated video
                    {"type": "text", "text": "Describe what happens in this short video."}
                ]
            }],
            "max_new_tokens": 200,
            "expected_vision_tokens": None,
            "video_frames": 16,  # Qwen will sample/process based on its internal logic
            "video_dims": (448, 448)  # Actual tokens depend on Qwen's processing
        },
        {
            "name": "Medium video (128 frames, 224x224)",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "video": None},  # Will be filled with generated video
                    {"type": "text", "text": "Analyze the sequence of events in this video."}
                ]
            }],
            "max_new_tokens": 200,
            "expected_vision_tokens": None,
            "video_frames": 128,  # Qwen will sample/process based on its internal logic
            "video_dims": (448, 448)  # Actual tokens depend on Qwen's processing
        },
        {
            "name": "Long video (768 frames, 224x224)",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "video": None},  # Will be filled with generated video
                    {"type": "text", "text": "Provide a detailed analysis of this long video sequence."}
                ]
            }],
            "max_new_tokens": 200,
            "expected_vision_tokens": None,
            "video_frames": 768,  # Qwen will sample/process based on its internal logic
            "video_dims": (448, 448)  # Actual tokens depend on Qwen's processing
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")

        # Generate test image if needed
        if test_case.get('image_dims') is not None:
            height, width = test_case['image_dims']
            print(f"Generating {height}x{width} test image (patches: {height//14}x{width//14} = {(height//14)*(width//14)})")
            test_image = create_test_image(height, width)
            # Replace None with actual image in messages
            for content_item in test_case['messages'][0]['content']:
                if content_item['type'] == 'image' and content_item['image'] is None:
                    content_item['image'] = test_image

        # Generate test video if needed
        if test_case.get('video_frames') is not None:
            num_frames = test_case['video_frames']
            height, width = test_case['video_dims']
            print(f"Generating video with {num_frames} frames ({height}x{width})")
            test_video = create_test_video(num_frames, height, width)
            # Replace None with actual video in messages
            for content_item in test_case['messages'][0]['content']:
                if content_item['type'] == 'video' and content_item['video'] is None:
                    content_item['video'] = test_video
        # Prepare inputs
        text = processor.apply_chat_template(
            test_case['messages'], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(test_case['messages'])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Measure input tokens
        input_token_count = inputs.input_ids.shape[1]
        print(f"Input tokens: {input_token_count}")
        #breakpoint()
        # Count vision tokens (if any) and get actual dimensions processed
        if video_inputs:
            actual_vision_height = max(video_inputs[0].shape[2], 448)  # Minimum height after Qwen's resizing
            actual_vision_width = max(video_inputs[0].shape[3], 448)   # Minimum width after Qwen's resizing
            effective_frames = video_inputs[0].shape[0] / 2 # For images, or actual processed frames for videos`
            vision_token_count = inputs['video_grid_thw'][0].prod().item()  # time * height_patches * width_patches
        elif image_inputs:
            actual_vision_height = max(image_inputs[0].shape[1], 448)  # Minimum height after Qwen's resizing
            actual_vision_width = max(image_inputs[0].shape[2], 448)   # Minimum width after Qwen's resizing
            effective_frames = 1  # Single image
            vision_token_count = inputs['image_grid_thw'][0].prod().item()  # height_patches * width_patches

        print(f"Vision tokens: {vision_token_count} (frames: {effective_frames}, dims: {actual_vision_height}x{actual_vision_width})")

        # Generate with hardware profiling to measure actual FLOPs
        print("Profiling generation...")
        generated_ids, actual_flops, prof = profile_model_forward(
            model, inputs, test_case['max_new_tokens']
        )

        # Measure actual generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        actual_generated_tokens = len(generated_ids_trimmed[0])
        total_context_len = input_token_count + actual_generated_tokens

        print(f"Generated tokens: {actual_generated_tokens}")
        print(f"Total context length: {total_context_len}")

        # Calculate predicted FLOPs using ACTUAL processed dimensions and token counts from Qwen
        predicted_flops = vision_language_flops(
            # Vision params (use actual processed dimensions and frames from Qwen)
            vision_frames=effective_frames if (image_inputs or video_inputs) else 0,
            vision_height=actual_vision_height if (image_inputs or video_inputs) else 0,
            vision_width=actual_vision_width if (image_inputs or video_inputs) else 0,
            vision_patch_size=14,
            vision_hidden_size=1280,
            num_full_attn_layers_vision=4 if (image_inputs or video_inputs) else 0,
            num_sliding_attn_layers_vision=28 if (image_inputs or video_inputs) else 0,
            num_mamba_layers_vision=0,
            num_mlp_layers_vision=32 if (image_inputs or video_inputs) else 0,

            # Vision settings
            vision_window_size=(8,8) if (image_inputs or video_inputs) else None,
            vision_is_2d=True if (image_inputs or video_inputs) else False,
            vision_num_attn_heads=16,
            vision_gqa=False,
            vision_mlp_expansion=(3420 / 1280),

            # Merger
            merger=bool(image_inputs or video_inputs),
            merger_mlp_expansion=1.0,

            # Language params (use actual input token count including vision tokens)
            lang_seq_len=input_token_count,
            lang_hidden_size=3584,
            num_full_attn_layers_lang=28,
            num_sliding_attn_layers_lang=0,
            num_mamba_layers_lang=0,
            num_mlp_layers_lang=28,

            # Language settings
            lang_num_attn_heads=28,
            lang_gqa=True,
            lang_gqa_groups=4,
            lang_mlp_expansion=(18944 / 3584),

            # Generation
            swiglu=False,
            vocab_size=152064,
            num_generated=actual_generated_tokens,
            do_backward=False,
        )

        # Calculate accuracy ratio
        flop_ratio = predicted_flops['total_flops'] / actual_flops if actual_flops > 0 else float('inf')

        result = {
            'test_name': test_case['name'],
            'input_tokens': input_token_count,
            'vision_tokens': vision_token_count,
            'generated_tokens': actual_generated_tokens,
            'total_context': total_context_len,
            'predicted_flops': predicted_flops['total_flops'],
            'actual_flops': actual_flops,
            'flop_ratio': flop_ratio,
            'vision_flops': predicted_flops['vision_flops'],
            'merger_flops': predicted_flops['merger_flops'],
            'lang_prompt_flops': predicted_flops['lang_prompt_flops'],
            'gen_flops': predicted_flops['gen_flops']
        }

        results.append(result)

        print(f"Predicted total FLOPs: {predicted_flops['total_flops']:,.0f}")
        print(f"Actual hardware FLOPs:   {actual_flops:,.0f}")
        print(f"Ratio (predicted/actual): {flop_ratio:.2f}x")
        print(f"  - Vision: {predicted_flops['vision_flops']:,.0f}")
        print(f"  - Merger: {predicted_flops['merger_flops']:,.0f}")
        print(f"  - Lang prompt: {predicted_flops['lang_prompt_flops']:,.0f}")
        print(f"  - Generation: {predicted_flops['gen_flops']:,.0f}")

        # Save profiler trace for detailed analysis
        prof_filename = f"profile_trace_{i+1}_{test_case['name'].replace(' ', '_')}.json"
        prof.export_chrome_trace(prof_filename)
        print(f"Profiler trace saved to: {prof_filename}")

        # Clean up temporary video files if any
        if test_case.get('video_frames') is not None:
            for content_item in test_case['messages'][0]['content']:
                if content_item['type'] == 'video' and isinstance(content_item['video'], str):
                    # Remove temporary MP4 file
                    if os.path.exists(content_item['video']):
                        os.unlink(content_item['video'])

    # Summary
    print(f"\n{'='*100}")
    print("FLOP ACCURACY SUMMARY")
    print("="*100)
    print(f"{'Test Name':<25} {'Input':<6} {'Vision':<6} {'Gen':<6} {'Predicted':<12} {'Actual':<12} {'Ratio':<8}")
    print("-" * 100)

    for result in results:
        print(f"{result['test_name']:<25} {result['input_tokens']:<6} {result['vision_tokens']:<6} "
              f"{result['generated_tokens']:<6} {result['predicted_flops']:<12,.0f} "
              f"{result['actual_flops']:<12,.0f} {result['flop_ratio']:<8.2f}x")

    # Calculate average ratio (excluding text-only which might have very different patterns)
    vision_results = [r for r in results if r['vision_tokens'] > 0]
    if vision_results:
        avg_ratio = sum(r['flop_ratio'] for r in vision_results) / len(vision_results)
        print(f"\nAverage ratio (vision tasks): {avg_ratio:.2f}x")

    print(f"\nRatio interpretation:")
    print(f"  - Ratio > 1.0: Our model overestimates FLOPs")
    print(f"  - Ratio < 1.0: Our model underestimates FLOPs")
    print(f"  - Ratio ≈ 1.0: Good accuracy!")
    print(f"\nProfiler traces saved for detailed kernel-level analysis.")
    print(f"Use chrome://tracing/ to view the .json trace files.")


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for manual execution via ``python -m metrics.flops_validation``."""

    args = list(sys.argv[1:] if argv is None else argv)
    run_full_validation = "--full-validation" in args

    reference_flops = compute_sample_qwen_flops()
    print(
        "Reference Qwen2.5-VL FLOP estimate (8 vision frames, 1024-prompt, 256-gen): "
        f"{reference_flops:,.0f}"
    )

    if run_full_validation:
        test_flop_accuracy()


if __name__ == "__main__":
    main()
