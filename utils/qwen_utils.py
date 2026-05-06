"""Utilities for pre-processing video inputs with Qwen vision models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


ProcessorOutput = Dict[str, object]


def get_video_frames(
    video_path: str,
    fps: int = 2,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[str, ProcessorOutput]:
    """Load video frames through the Qwen processor with optional resizing hints."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                }
            ],
        }
    ]

    processor_kwargs = {"use_fast": False}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        **processor_kwargs,
    )
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    _, video_inputs = process_vision_info(messages)
    return chat_prompt, video_inputs
