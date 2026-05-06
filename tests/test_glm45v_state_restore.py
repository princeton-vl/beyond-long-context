import copy
import os

import pytest

from tests.state_restore_utils import make_video_tensor, require_cuda, set_seed

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

MODEL_ID = os.environ.get("GLM45V_TEST_MODEL", "zai-org/GLM-4.5V")

from models.glm45v import GLM45V  # noqa: E402


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="GLM-4.5V test requires PyTorch with CUDA support")
def test_glm45v_state_roundtrip_preserves_generation():
    require_cuda("GLM45V")

    def instantiate() -> "GLM45V":
        return GLM45V(MODEL_ID, enable_metrics=False)

    initial_segments = [
        {
            "frames": make_video_tensor(8, 301),
            "start": 0.0,
            "end": 4.0,
            "text": "Intro clip.",
        },
        {
            "frames": make_video_tensor(8, 302),
            "start": 4.0,
            "end": 8.0,
            "text": "Detail clip.",
        },
    ]

    later_segments = [
        {
            "frames": make_video_tensor(6, 401),
            "start": 8.2,
            "end": 11.0,
            "text": "Post-save clip alpha.",
        },
        {
            "frames": make_video_tensor(6, 402),
            "start": 11.0,
            "end": 13.0,
            "text": "Post-save clip beta.",
        },
    ]

    questions = [
        {
            "seed": 6100,
            "prompt": "Provide a concise status update.",
            "time": 13.0,
            "max_tokens": 48,
            "ack": "Status update acknowledged.",
            "ack_offset": 0.1,
        },
        {
            "seed": 6200,
            "prompt": "Summarize notable visual details.",
            "time": 13.1,
            "max_tokens": 48,
            "ack": "Visual details logged.",
            "ack_offset": 0.1,
        },
    ]

    pre_save_text = "Closing note pre-save."
    pre_save_time = initial_segments[-1]["end"] + 0.2

    def add_segments(model, segments):
        for seg in segments:
            model.add_video(seg["frames"], time_start=seg["start"], time_end=seg["end"])
            model.add_text(seg["text"], current_video_time=seg["end"])

    def run_questions(model) -> tuple[str, ...]:
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
