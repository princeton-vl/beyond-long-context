import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - handled by pytest skip
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.minicpm_v_2_6 import MiniCPMVideo
from tests.state_restore_utils import require_cuda, set_seed

MODEL_ID = os.environ.get("MINICPM_TEST_MODEL", "openbmb/MiniCPM-o-2_6")


def _make_unit(num_frames: int, seed: int, width: int = 64, height: int = 64):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(num_frames):
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frames.extend(["<unit>", Image.fromarray(frame)])
    return frames


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="MiniCPM test requires PyTorch with CUDA support")
def test_minicpm_state_roundtrip_preserves_generation():
    require_cuda("MiniCPMVideo")

    def instantiate() -> MiniCPMVideo:
        try:
            return MiniCPMVideo(MODEL_ID, enable_metrics=False)
        except RuntimeError as exc:
            if "streaming_prefill" in str(exc):
                pytest.skip(str(exc))
            raise

    initial_segments = [
        {
            "unit": _make_unit(24, 101),
            "start": 0.0,
            "end": 6.0,
            "text": "Segment 0 concludes.",
        },
        {
            "unit": _make_unit(20, 102),
            "start": 6.0,
            "end": 12.0,
            "text": "Segment 1 adds more detail.",
        },
        {
            "unit": _make_unit(16, 103),
            "start": 12.0,
            "end": 18.0,
            "text": "Segment 2 wraps up the preface.",
        },
    ]

    later_segments = [
        {
            "unit": _make_unit(18, 201),
            "start": 18.5,
            "end": 24.5,
            "text": "Post-save segment A.",
        },
        {
            "unit": _make_unit(12, 202),
            "start": 24.5,
            "end": 27.5,
            "text": "Post-save segment B.",
        },
    ]

    questions = [
        {
            "seed": 3100,
            "prompt": "Provide a concise summary of everything so far.",
            "time": 27.5,
            "max_tokens": 60,
            "ack": "Acknowledged summary.",
            "ack_offset": 0.1,
        },
        {
            "seed": 3200,
            "prompt": "List the dominant colors you observed.",
            "time": 27.6,
            "max_tokens": 60,
            "ack": "Preparing for final report.",
            "ack_offset": 0.1,
        },
        {
            "seed": 3300,
            "prompt": "Give the final report in one sentence.",
            "time": 27.7,
            "max_tokens": 60,
            "ack": "Final report stored.",
            "ack_offset": 0.1,
        },
    ]

    pre_save_text = "Final remark before snapshot."
    pre_save_time = initial_segments[-1]["end"] + 0.5

    def add_segments(model: MiniCPMVideo, segments: list[dict[str, object]]) -> None:
        for seg in segments:
            model.add_video(seg["unit"], time_start=seg["start"], time_end=seg["end"])
            model.add_text(seg["text"], current_video_time=seg["end"])

    def run_questions(model: MiniCPMVideo) -> tuple[str, ...]:
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
