"""Heavy FLOP-profiling tests for the V+L models.

These tests attach a PyTorch profiler to real generation calls and compare the
measured FLOPs with the theoretical counts from ``metrics.flops_calc``.  They
are deliberately expensive; set ``RUN_FLOP_PROFILE_TESTS=1`` to enable them.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.flops_calc import (  # noqa: E402
    glm45v_flops,
    minicpm_v_2_6_flops,
    qwen_2_5_vl_7b_flops,
    qwen3_omni_30b_flops,
    timechat_online_flops,
)


def _load_model_module(rel_path: str, module_name: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


GLM45V = _load_model_module("models/glm45v.py", "glm45v_module").GLM45V
MiniCPMVideo = _load_model_module("models/minicpm_v_2_6.py", "minicpm_module").MiniCPMVideo
Qwen3Omni = _load_model_module("models/qwen3_omni.py", "qwen3omni_module").Qwen3Omni
QwenFullVideo = _load_model_module("models/qwen2_5_vl.py", "qwenfull_module").QwenFullVideo
TimeChatOnlineStreaming = _load_model_module("models/timechat.py", "timechat_module").TimeChatOnlineStreaming

RUN_PROFILING = os.environ.get("RUN_FLOP_PROFILE_TESTS") == "1"


def _skip_unless_enabled() -> None:
    if not RUN_PROFILING:
        pytest.skip("Set RUN_FLOP_PROFILE_TESTS=1 to run FLOP profiling tests.")


def _require_cuda(name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{name} FLOP profiling requires CUDA support.")


def _build_text_for_tokens(counter: Callable[[str], int], target_tokens: int) -> str:
    token = "token"
    text = token
    while counter(text) < target_tokens:
        text += " " + token
    return text


@contextmanager
def _capture_processor_io(processor):
    records: dict[str, object] = {}
    original = processor.__call__

    def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        outputs = original(*args, **kwargs)
        try:
            records["input_ids"] = outputs["input_ids"].detach().cpu()
        except Exception:
            pass
        records["kwargs"] = kwargs
        return outputs

    processor.__call__ = wrapped  # type: ignore[assignment]
    try:
        yield records
    finally:
        processor.__call__ = original


@dataclass
class Scenario:
    frames: int
    text_tokens: int


def _make_video_array(num_frames: int, height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(num_frames, 3, height, width), dtype=np.uint8)
    return data


def _record_ratio(
    ratios: list[float],
    model_name: str,
    scenario: Scenario,
    max_tokens: int,
    measured: float,
    theoretical: float,
) -> None:
    if measured <= 0 or theoretical <= 0:
        pytest.skip(
            f"Non-positive FLOP measurement (measured={measured}, theoretical={theoretical}) for {model_name}."
        )
    ratio = theoretical / measured
    ratios.append(ratio)
    print(
        f"[{model_name}] frames={scenario.frames} tokens={max_tokens} -> pred/measured ratio={ratio:.4f}"
    )


def _assert_ratio_consistency(model_name: str, ratios: list[float], tolerance: float = 0.25) -> None:
    if not ratios:
        pytest.skip(f"No FLOP ratios captured for {model_name}.")
    ratio_max = max(ratios)
    ratio_min = min(ratios)
    if ratio_min <= 0:
        pytest.skip(f"Invalid ratio encountered for {model_name} (ratio_min={ratio_min}).")
    spread = ratio_max / ratio_min - 1.0
    assert spread <= tolerance, (
        f"Predicted/measured ratios vary too much for {model_name}: min={ratio_min:.4f}, "
        f"max={ratio_max:.4f}, spread={spread:.2%}"
    )


# -----------------------------------------------------------------------------
# Qwen Full Video
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_PROFILING, reason="Large FLOP-profiling suite disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flop_profile_qwen_full_video():
    _skip_unless_enabled()
    _require_cuda("QwenFullVideo")

    model = QwenFullVideo("Qwen/Qwen2.5-VL-7B-Instruct", enable_metrics=False)
    height = 224
    width = 224

    scenarios: Iterable[Scenario] = (
        Scenario(frames=100, text_tokens=100),
        Scenario(frames=250, text_tokens=100),
        Scenario(frames=768, text_tokens=100),
    )

    ratios: list[float] = []
    try:
        for scenario in scenarios:
            for max_tokens in (10, 1000):
                model.clear_context()
                frames = _make_video_array(scenario.frames, height, width, seed=scenario.frames)
                duration = scenario.frames / 15.0
                model.add_video(frames, time_start=0.0, time_end=duration)
                context_text = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                model.add_text(context_text, current_video_time=duration)

                question = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                question_time = duration + 0.1

                with _capture_processor_io(model.processor) as records:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                        answer = model.ask_question(
                            question,
                            current_video_time=question_time,
                            max_tokens=max_tokens,
                        )

                total_flops = sum(evt.flops for evt in prof.events() if evt.flops)
                prompt_ids = records.get("input_ids")
                if prompt_ids is None:
                    pytest.skip("Unable to capture processor input IDs for QwenFullVideo")
                prompt_len = int(prompt_ids.shape[-1])

                videos_kw = records.get("kwargs", {}).get("videos") if records.get("kwargs") else None
                if videos_kw is not None:
                    frame_height = videos_kw[0].shape[-2]
                    frame_width = videos_kw[0].shape[-1]
                    frame_count = sum(v.shape[0] for v in videos_kw)
                else:
                    frame_height = height
                    frame_width = width
                    frame_count = scenario.frames

                gen_tokens = model._count_text_tokens(answer)
                theoretical = qwen_2_5_vl_7b_flops(
                    vision_frames=frame_count,
                    vision_height=frame_height,
                    vision_width=frame_width,
                    lang_prompt_len=prompt_len,
                    num_generated=gen_tokens,
                    do_backward=False,
                )["total_flops"]

                _record_ratio(ratios, "QwenFullVideo", scenario, max_tokens, total_flops, theoretical)
    finally:
        model.clear_context()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assert_ratio_consistency("QwenFullVideo", ratios)


# -----------------------------------------------------------------------------
# TimeChat Online Streaming
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_PROFILING, reason="Large FLOP-profiling suite disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flop_profile_timechat():
    _skip_unless_enabled()
    _require_cuda("TimeChatOnlineStreaming")

    model = TimeChatOnlineStreaming("wyccccc/TimeChatOnline-7B", enable_metrics=False)
    height = 224
    width = 224

    scenarios: Iterable[Scenario] = (
        Scenario(frames=100, text_tokens=100),
        Scenario(frames=250, text_tokens=100),
        Scenario(frames=768, text_tokens=100),
    )

    ratios: list[float] = []
    try:
        for scenario in scenarios:
            for max_tokens in (10, 1000):
                model.clear_context()
                frames = _make_video_array(scenario.frames, height, width, seed=scenario.frames)
                duration = scenario.frames / 15.0
                model.add_video(frames, time_start=0.0, time_end=duration)
                context_text = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                model.add_text(context_text, current_video_time=duration)

                question = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                question_time = duration + 0.1

                with _capture_processor_io(model.processor) as records:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                        answer = model.ask_question(
                            question,
                            current_video_time=question_time,
                            max_tokens=max_tokens,
                        )

                total_flops = sum(evt.flops for evt in prof.events() if evt.flops)
                prompt_ids = records.get("input_ids")
                if prompt_ids is None:
                    pytest.skip("Unable to capture processor input IDs for TimeChat")
                prompt_len = int(prompt_ids.shape[-1])
                videos_kw = records.get("kwargs", {}).get("videos") if records.get("kwargs") else None
                if videos_kw is not None:
                    frame_height = videos_kw[0].shape[-2]
                    frame_width = videos_kw[0].shape[-1]
                    frame_count = sum(v.shape[0] for v in videos_kw)
                else:
                    frame_height = height
                    frame_width = width
                    frame_count = scenario.frames

                gen_tokens = model._count_text_tokens(answer)
                theoretical = timechat_online_flops(
                    vision_frames=frame_count,
                    vision_height=frame_height,
                    vision_width=frame_width,
                    lang_prompt_len=prompt_len,
                    num_generated=gen_tokens,
                    tokens_dropped=0,
                    tokens_total_before_drop=None,
                    do_backward=False,
                )["total_flops"]

                _record_ratio(ratios, "TimeChat", scenario, max_tokens, total_flops, theoretical)
    finally:
        model.clear_context()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assert_ratio_consistency("TimeChat", ratios)


# -----------------------------------------------------------------------------
# MiniCPM-V2.6 (streaming)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_PROFILING, reason="Large FLOP-profiling suite disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flop_profile_minicpm():
    _skip_unless_enabled()
    _require_cuda("MiniCPMVideo")

    model = MiniCPMVideo("openbmb/MiniCPM-o-2_6", enable_metrics=False)
    height = 448
    width = 448

    scenarios: Iterable[Scenario] = (
        Scenario(frames=100, text_tokens=100),
        Scenario(frames=250, text_tokens=100),
        Scenario(frames=768, text_tokens=100),
    )

    ratios: list[float] = []
    try:
        for scenario in scenarios:
            for max_tokens in (10, 1000):
                model.clear_context()
                # MiniCPM expects PIL "<unit>" content
                segment = _make_unit(scenario.frames, height, width, seed=scenario.frames)
                duration = scenario.frames / 6.0
                model.add_video(segment, time_start=0.0, time_end=duration)
                context_text = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                model.add_text(context_text, current_video_time=duration)

                question = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                question_time = duration + 0.1

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                    answer = model.ask_question(
                        question,
                        current_video_time=question_time,
                        max_tokens=max_tokens,
                    )

                total_flops = sum(evt.flops for evt in prof.events() if evt.flops)
                prompt_ids = model.tokenizer(question, return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ]
                prompt_len = int(prompt_ids.shape[-1])
                gen_tokens = (
                    model.tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
                    if answer
                    else 0
                )
                theoretical = minicpm_v_2_6_flops(
                    vision_frames=scenario.frames,
                    vision_height=height,
                    vision_width=width,
                    lang_prompt_len=prompt_len,
                    num_generated=gen_tokens,
                    do_backward=False,
                )["total_flops"]

                _record_ratio(ratios, "MiniCPM", scenario, max_tokens, total_flops, theoretical)
    finally:
        model.clear_context()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assert_ratio_consistency("MiniCPM", ratios)


def _make_unit(num_frames: int, height: int, width: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    content = []
    for _ in range(num_frames):
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        content.extend(["<unit>", Image.fromarray(frame)])
    return content


from PIL import Image  # noqa: E402  (used by _make_unit)


# -----------------------------------------------------------------------------
# GLM-4.5V (BF16)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_PROFILING, reason="Large FLOP-profiling suite disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flop_profile_glm45v():
    _skip_unless_enabled()
    _require_cuda("GLM45V")

    model = GLM45V("zai-org/GLM-4.5V", enable_metrics=False)
    height = 336
    width = 336

    scenarios: Iterable[Scenario] = (
        Scenario(frames=100, text_tokens=100),
        Scenario(frames=250, text_tokens=100),
        Scenario(frames=768, text_tokens=100),
    )

    ratios: list[float] = []
    try:
        for scenario in scenarios:
            for max_tokens in (10, 1000):
                model.clear_context()
                frames = _make_video_array(scenario.frames, height, width, seed=scenario.frames)
                duration = scenario.frames / 12.0
                model.add_video(frames, time_start=0.0, time_end=duration)
                context_text = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                model.add_text(context_text, current_video_time=duration)

                question = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                question_time = duration + 0.1

                with _capture_processor_io(model.processor) as records:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                        answer = model.ask_question(
                            question,
                            current_video_time=question_time,
                            max_tokens=max_tokens,
                        )

                total_flops = sum(evt.flops for evt in prof.events() if evt.flops)
                prompt_ids = records.get("input_ids")
                if prompt_ids is None:
                    pytest.skip("Unable to capture processor input IDs for GLM45V")
                prompt_len = int(prompt_ids.shape[-1])

                gen_tokens = model._count_text_tokens(answer)
                theoretical = glm45v_flops(
                    vision_frames=scenario.frames,
                    vision_height=height,
                    vision_width=width,
                    lang_prompt_len=prompt_len,
                    num_generated=gen_tokens,
                    do_backward=False,
                )["total_flops"]

                _record_ratio(ratios, "GLM45V", scenario, max_tokens, total_flops, theoretical)
    finally:
        model.clear_context()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assert_ratio_consistency("GLM45V", ratios)


# -----------------------------------------------------------------------------
# Qwen3 Omni
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_PROFILING, reason="Large FLOP-profiling suite disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flop_profile_qwen3_omni():
    _skip_unless_enabled()
    _require_cuda("Qwen3Omni")

    model = Qwen3Omni(enable_metrics=False)
    height = 224
    width = 224

    scenarios: Iterable[Scenario] = (
        Scenario(frames=100, text_tokens=100),
        Scenario(frames=250, text_tokens=100),
        Scenario(frames=768, text_tokens=100),
    )

    ratios: list[float] = []
    try:
        for scenario in scenarios:
            for max_tokens in (10, 1000):
                model.clear_context()
                frames = _make_video_array(scenario.frames, height, width, seed=scenario.frames)
                duration = scenario.frames / 15.0
                model.add_video(frames, time_start=0.0, time_end=duration)
                context_text = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                model.add_text(context_text, current_video_time=duration)

                question = _build_text_for_tokens(model._count_text_tokens, scenario.text_tokens)
                question_time = duration + 0.1

                with _capture_processor_io(model.processor) as records:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                        answer = model.ask_question(
                            question,
                            current_video_time=question_time,
                            max_tokens=max_tokens,
                        )

                total_flops = sum(evt.flops for evt in prof.events() if evt.flops)
                prompt_ids = records.get("input_ids")
                if prompt_ids is None:
                    pytest.skip("Unable to capture processor input IDs for Qwen3Omni")
                prompt_len = int(prompt_ids.shape[-1])

                gen_tokens = model._count_text_tokens(answer)
                theoretical = qwen3_omni_30b_flops(
                    vision_frames=scenario.frames,
                    vision_height=height,
                    vision_width=width,
                    lang_prompt_len=prompt_len,
                    num_generated=gen_tokens,
                    do_backward=False,
                )["total_flops"]

                _record_ratio(ratios, "Qwen3Omni", scenario, max_tokens, total_flops, theoretical)
    finally:
        model.clear_context()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(ratios)
    _assert_ratio_consistency("Qwen3Omni", ratios)


# -----------------------------------------------------------------------------
# Notes on M3-Agent: we profile its primary VLM components separately above.
# -----------------------------------------------------------------------------
