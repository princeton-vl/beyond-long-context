import copy
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen2_5_vl import QwenFullVideo
from tests.state_restore_utils import make_video_tensor, require_cuda, set_seed

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

MODEL_ID = os.environ.get("QWEN_FULL_TEST_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="QwenFullVideo test requires PyTorch with CUDA support")
def test_qwen_full_state_roundtrip_preserves_generation():
    require_cuda("QwenFullVideo")

    def instantiate() -> QwenFullVideo:
        return QwenFullVideo(MODEL_ID, enable_metrics=False)

    initial_segments = [
        {
            "frames": make_video_tensor(12, 11),
            "start": 0.0,
            "end": 6.0,
            "text": "Introductory clip.",
        },
        {
            "frames": make_video_tensor(10, 12),
            "start": 6.0,
            "end": 10.0,
            "text": "Follow-up observations.",
        },
        {
            "frames": make_video_tensor(8, 13),
            "start": 10.0,
            "end": 14.0,
            "text": "Closing remarks before snapshot.",
        },
    ]

    later_segments = [
        {
            "frames": make_video_tensor(9, 21),
            "start": 14.5,
            "end": 18.0,
            "text": "Post-save sequence alpha.",
        },
        {
            "frames": make_video_tensor(7, 22),
            "start": 18.0,
            "end": 20.5,
            "text": "Post-save sequence beta.",
        },
    ]

    questions = [
        {
            "seed": 4100,
            "prompt": "Summarize the main events so far.",
            "time": 20.5,
            "max_tokens": 40,
            "ack": "Acknowledged summary A.",
            "ack_offset": 0.1,
        },
        {
            "seed": 4200,
            "prompt": "List key visual elements encountered.",
            "time": 20.6,
            "max_tokens": 40,
            "ack": "Acknowledged summary B.",
            "ack_offset": 0.1,
        },
        {
            "seed": 4300,
            "prompt": "Provide a concluding single sentence.",
            "time": 20.7,
            "max_tokens": 40,
            "ack": "Final acknowledgement recorded.",
            "ack_offset": 0.1,
        },
    ]

    pre_save_text = "Final note pre-save."
    pre_save_time = initial_segments[-1]["end"] + 0.2

    def add_segments(model: QwenFullVideo, segments: list[dict[str, object]]) -> None:
        for seg in segments:
            model.add_video(seg["frames"], time_start=seg["start"], time_end=seg["end"])
            model.add_text(seg["text"], current_video_time=seg["end"])

    def run_questions(model: QwenFullVideo) -> tuple[str, ...]:
        outputs: list[str] = []
        for q in questions:
            set_seed(q["seed"])
            answer = model.ask_question(
                q["prompt"],
                current_video_time=q["time"],
                max_tokens=q["max_tokens"],
            )
            outputs.append(answer)
            model.add_text(q["ack"], current_video_time=q["time"] + q["ack_offset"])
        return tuple(outputs)

    # Straight-through scenario: no saving, all segments added consecutively.
    direct_model = instantiate()
    add_segments(direct_model, initial_segments)
    direct_model.add_text(pre_save_text, current_video_time=pre_save_time)
    add_segments(direct_model, later_segments)
    direct_outputs = run_questions(direct_model)
    direct_model.clear_context()
    del direct_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save mid-way, continue on same instance.
    same_instance_model = instantiate()
    add_segments(same_instance_model, initial_segments)
    same_instance_model.add_text(pre_save_text, current_video_time=pre_save_time)
    initial_state = same_instance_model.save_state()
    add_segments(same_instance_model, later_segments)
    post_state = same_instance_model.save_state()
    same_instance_outputs = run_questions(same_instance_model)
    same_instance_model.clear_context()
    del same_instance_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save mid-way, resume on a fresh instance.
    resumed_model = instantiate()
    resumed_model.load_state(copy.deepcopy(initial_state))
    add_segments(resumed_model, later_segments)
    resumed_post_state = resumed_model.save_state()
    resumed_outputs = run_questions(resumed_model)
    resumed_model.clear_context()
    del resumed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the post-save snapshot multiple times to ensure determinism.
    repeated_outputs = []
    for snapshot in (post_state, resumed_post_state):
        repeat_model = instantiate()
        repeat_model.load_state(copy.deepcopy(snapshot))
        repeated_outputs.append(run_questions(repeat_model))
        repeat_model.clear_context()
        del repeat_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert (
        direct_outputs
        == same_instance_outputs
        == resumed_outputs
        == repeated_outputs[0]
        == repeated_outputs[1]
    )
