import math
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Union, Optional, Sequence, Dict, Any

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, GenerationConfig

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from utils.paths import get_scratch_subdir

# --- LongVILA helpers from remote code (correct ones!) ---
from media import extract_media           # from LongVILA-R1-7B repo
from mm_utils import process_image, process_images
from tokenizer_utils import tokenize_conversation
from constants import DEFAULT_IMAGE_TOKEN


@torch.inference_mode()
def generate_content(
    self,
    config,
    image_processor,
    prompt: Union[str, List],
    generation_config: Optional[GenerationConfig] = None,
    response_format=None,
):
    conversation = [{"from": "human", "value": prompt}]

    # Convert response format to logits processor (placeholder)
    xgr_logits_processor = None  # NOTE: currently unused in this function
    # Extract media from the conversation
    media = extract_media(conversation, config)
    conversation[0]['value'] = "Explain the differences between these two videos in 30 words or less. Video 1: <vila/video> and Video 2: <vila/video>"
    media_config = defaultdict(dict)
    for name in media:
        if name == "image":
            if len(media["image"]) == 1 and config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                # Use the passed-in image_processor
                config.image_processor = image_processor

                if config.image_aspect_ratio == "dynamic":
                    images = process_image(
                        media["image"][0],
                        config,
                        None,
                        enable_dynamic_res=True,
                    ).half()

                    # Expand image tokens to match number of image tiles
                    conversation[0]["value"] = conversation[0]["value"].replace(
                        DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                    )
                else:
                    # dynamic_s2
                    if isinstance(config.s2_scales, str):
                        config.s2_scales = list(map(int, config.s2_scales.split(",")))

                    images, block_sizes = process_image(
                        media["image"][0],
                        config,
                        None,
                        enable_dynamic_s2=True,
                    )
                    images = images.half()
                    media_config[name]["block_sizes"] = [block_sizes]
            else:
                images = process_images(
                    media["image"],
                    image_processor,
                    config,
                ).half()

            media[name] = [image for image in images]

        elif name == "video":
            if config.image_aspect_ratio == "dynamic" and config.video_max_tiles > 1:
                media[name] = [
                    process_images(
                        images,
                        image_processor,
                        config,
                        enable_dynamic_res=True,
                        max_tiles=config.video_max_tiles,
                    ).half()
                    for images in media[name]
                ]
            elif config.image_aspect_ratio == "dynamic_s2" and config.video_max_tiles > 1:
                config.image_processor = image_processor

                if isinstance(config.s2_scales, str):
                    config.s2_scales = list(map(int, config.s2_scales.split(",")))

                media[name] = [
                    torch.cat(
                        [
                            process_image(
                                image,
                                config,
                                None,
                                enable_dynamic_s2=True,
                                max_tiles=config.video_max_tiles,
                            )[0].half()
                            for image in images
                        ]
                    )
                    for images in media[name]
                ]
            else:
                media[name] = [
                    process_images(
                        images,
                        image_processor,
                        config,
                    ).half()
                    for images in media[name]
                ]
        else:
            raise ValueError(f"Unsupported media type: {name}")

    # Tokenize conversation (with generation prompt) and move to CUDA
    input_ids = tokenize_conversation(
        conversation,
        self.tokenizer,
        add_generation_prompt=True,
    ).unsqueeze(0).cuda()

    # ✅ Return media and media_config as well, so we don't have to recompute them
    return conversation, input_ids, media, media_config


# -----------------------------
# Model + simple driver
# -----------------------------
model_path = "Efficient-Large-Model/LongVILA-R1-7B"
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    _attn_implementation="flash_attention_2",
)

gen_config = GenerationConfig(max_new_tokens=400)
model.config.num_video_frames, model.config.fps = 128, 0

use_thinking = False
system_prompt_thinking = (
    "You are a helpful assistant. The user asks a question, and then you solve it.\n\n"
    "Please first think deeply about the question based on the given video, and then provide the final answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n"
    "Question: {question}"
)

base_prompt = "Explain the differences between these two videos in 60 words or less."
_SYNTH_DATASET_SEGMENTS = (
    "synthetic-video-memory-genv2",
    "output",
    "shapes-driving-ds",
)
video_path1 = get_scratch_subdir(*_SYNTH_DATASET_SEGMENTS, "video_01", "base_8fps_repeat_segment.mp4")
video_path2 = get_scratch_subdir(*_SYNTH_DATASET_SEGMENTS, "video_08", "base_8fps_repeat_segment.mp4")

if use_thinking:
    user_prompt = system_prompt_thinking.format(question=base_prompt)
else:
    user_prompt = base_prompt

# This matches the HF example: [prompt, {"path": video_path}]
prompt_list = [user_prompt, {"path": video_path1}, {"path": video_path2}]

# 1) Use your generate_content to get conversation, input_ids, media, media_config
conversation, input_ids, media, media_config = generate_content(
    self=model,
    config=model.config,
    image_processor=model.vision_tower.image_processor,
    prompt=prompt_list,
    generation_config=gen_config,
)

print("Conversation:", conversation)

# 2) Now call generate exactly like in modeling_vila.py,
#    but with the precomputed media + media_config.
xgr_logits_processor = None  # still placeholder, like in modeling_vila.generate_content 
for percentage in [10.0, 20.0, 30.0, 40.0, 10.0]:
    old_media = {}
    for i in range(len(media["video"])):
        num_frames = media["video"][i].shape[-4]
        target_frames = int(round((num_frames * 0.01 * percentage) / 4) * 4)
        old_media[i] = media["video"][i].clone()
        media["video"][i] = media["video"][i][:target_frames, :, : , :]
        print(
        f"\n=== Requested {percentage}% of sampled frames; "
        f"using {target_frames} frames ==="
        )
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            generation_config=gen_config,
            logits_processor=xgr_logits_processor,
        )
    except ValueError:
        if not gen_config.do_sample:
            raise
        import logging
        logging.warning("Generation failed with sampling, retrying with greedy decoding.")
        gen_config.do_sample = False
        output_ids = model.generate(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            generation_config=gen_config,
            logits_processor=xgr_logits_processor,
        )

    response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print("Response:", response)
    for i in range(len(media["video"])):
        media["video"][i] = old_media[i]
