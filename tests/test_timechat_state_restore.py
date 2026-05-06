import copy
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.timechat import TimeChatOnlineStreaming
from tests.state_restore_utils import make_video_tensor, require_cuda, set_seed

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

MODEL_ID = os.environ.get("TIMECHAT_TEST_MODEL", "wyccccc/TimeChatOnline-7B")


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="TimeChat test requires PyTorch with CUDA support")
def test_timechat_state_roundtrip_preserves_generation():
    require_cuda("TimeChatOnlineStreaming")

    def instantiate() -> TimeChatOnlineStreaming:
        instance = TimeChatOnlineStreaming(MODEL_ID, enable_metrics=False)
        tmp_dir = tempfile.mkdtemp(prefix="timechat_state_")
        instance._drop_log_path = os.path.join(tmp_dir, "drop_log.jsonl")
        return instance

    initial_segments = [
        {
            "frames": make_video_tensor(10, 101),
            "start": 0.0,
            "end": 5.0,
            "text": "Initial briefing.",
        },
        {
            "frames": make_video_tensor(8, 102),
            "start": 5.0,
            "end": 9.0,
            "text": "Secondary observations.",
        },
    ]

    later_segments = [
        {
            "frames": make_video_tensor(6, 201),
            "start": 9.5,
            "end": 12.0,
            "text": "Post-save capture A.",
        },
        {
            "frames": make_video_tensor(6, 202),
            "start": 12.0,
            "end": 14.0,
            "text": "Post-save capture B.",
        },
    ]

    questions = [
        {
            "seed": 5100,
            "prompt": "Summarize the sequences so far.",
            "time": 14.0,
            "max_tokens": 48,
            "ack": "Acknowledged summary.",
            "ack_offset": 0.1,
        },
        {
            "seed": 5200,
            "prompt": "List three notable motion patterns observed.",
            "time": 14.1,
            "max_tokens": 48,
            "ack": "Motion patterns logged.",
            "ack_offset": 0.1,
        },
    ]

    pre_save_text = "Closing comment before save."
    pre_save_time = initial_segments[-1]["end"] + 0.5

    def add_segments(model: TimeChatOnlineStreaming, segments: list[dict[str, object]]) -> None:
        for seg in segments:
            model.add_video(seg["frames"], time_start=seg["start"], time_end=seg["end"])
            model.add_text(seg["text"], current_video_time=seg["end"])

    def run_questions(model: TimeChatOnlineStreaming) -> tuple[str, ...]:
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
