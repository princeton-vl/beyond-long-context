import copy
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen3_omni import Qwen3Omni
from tests.state_restore_utils import make_video_tensor, require_cuda, set_seed

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

MODEL_ID = os.environ.get("QWEN3_OMNI_TEST_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="Qwen3-Omni test requires PyTorch with CUDA support")
def test_qwen3_omni_state_roundtrip_preserves_generation():
    require_cuda("Qwen3-Omni")

    def instantiate() -> Qwen3Omni:
        return Qwen3Omni(MODEL_ID, enable_metrics=False)

    initial_segments = [
        {
            "frames": make_video_tensor(8, 701),
            "start": 0.0,
            "end": 4.0,
            "text": "Initial omni sequence.",
        },
        {
            "frames": make_video_tensor(8, 702),
            "start": 4.0,
            "end": 8.0,
            "text": "Secondary omni sequence.",
        },
    ]

    later_segments = [
        {
            "frames": make_video_tensor(7, 801),
            "start": 8.2,
            "end": 11.0,
            "text": "Omni post sequence A.",
        },
        {
            "frames": make_video_tensor(7, 802),
            "start": 11.0,
            "end": 13.5,
            "text": "Omni post sequence B.",
        },
    ]

    questions = [
        {
            "seed": 8100,
            "prompt": "Summarize the multimodal context so far.",
            "time": 13.5,
            "max_tokens": 48,
            "ack": "Acknowledged omni summary.",
            "ack_offset": 0.1,
        },
        {
            "seed": 8200,
            "prompt": "Describe notable audio or visual cues detected.",
            "time": 13.6,
            "max_tokens": 48,
            "ack": "Notable cues recorded.",
            "ack_offset": 0.1,
        },
    ]

    pre_save_text = "Omni pre-save note."
    pre_save_time = initial_segments[-1]["end"] + 0.2

    def add_segments(model: Qwen3Omni, segments: list[dict[str, object]]) -> None:
        for seg in segments:
            model.add_video(seg["frames"], time_start=seg["start"], time_end=seg["end"])
            model.add_text(seg["text"], current_video_time=seg["end"])

    def run_questions(model: Qwen3Omni) -> tuple[str, ...]:
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

    direct_model = instantiate()
    add_segments(direct_model, initial_segments)
    direct_model.add_text(pre_save_text, current_video_time=pre_save_time)
    add_segments(direct_model, later_segments)
    direct_outputs = run_questions(direct_model)
    direct_model.clear_context()
    del direct_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    resumed_model = instantiate()
    resumed_model.load_state(copy.deepcopy(initial_state))
    add_segments(resumed_model, later_segments)
    resumed_post_state = resumed_model.save_state()
    resumed_outputs = run_questions(resumed_model)
    resumed_model.clear_context()
    del resumed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
