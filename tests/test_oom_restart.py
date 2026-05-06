"""Tests for the CUDA OOM recovery flow in main.py."""

import copy
import json
import sys
from types import SimpleNamespace

if "check_overheat" not in sys.modules:
    sys.modules["check_overheat"] = SimpleNamespace(pause_needed=lambda: False, pause=lambda: None)

import pytest

import main as main_module
from datasets.patternvideos_manifest import OptionEntry, QuestionEntry, VideoEntry


class _FakeMetrics:
    """Simple container that mimics the metric lists exposed by real models."""

    def __init__(self) -> None:
        self.latency_add_video = []
        self.latency_add_text = []
        self.latency_ask_question = []
        self.flops_add_video = []
        self.flops_add_text = []
        self.flops_ask_question = []
        self.state_memory_floats = []
        self.state_memory_after_add_video = []
        self.state_memory_after_add_text = []
        self.state_memory_after_ask_question = []
        self.state_memory_delta_add_video = []
        self.state_memory_delta_add_text = []
        self.state_memory_delta_ask_question = []
        self.peak_gpu_mem_increase_add_video = []
        self.peak_gpu_mem_increase_add_text = []
        self.peak_gpu_mem_increase_ask_question = []
        self.peak_gpu_mem_absolute_add_video = []
        self.peak_gpu_mem_absolute_add_text = []
        self.peak_gpu_mem_absolute_ask_question = []
        self.video_timestamps_add_video = []
        self.video_timestamps_add_text = []
        self.video_timestamps_ask_question = []
        self.question_correctness_rate = []
        self.question_dont_know_rate = []
        self.video_timestamps_question_outcome = []
        self.question_answered_mask = []


class _FakeModel:
    """Minimal stand-in for model classes used by main.py."""

    created_instances = 0
    last_summary_metrics = None
    last_analyzed_metrics = None
    analyze_print_results = None

    def __init__(self, *args, **kwargs):
        type(self).created_instances += 1
        self._metrics = _FakeMetrics()
        self._last_state_memory_total = 0.0

    def clear_context(self):
        return None

    def save_state(self):
        return {}

    def load_state(self, _state):
        return None

    def add_text(self, *_args, **_kwargs):
        return None

    def add_video(self, *_args, **_kwargs):
        return None

    def ask_question(self, *_args, **_kwargs):
        return "{0}"

    def close(self):
        return None

    def shutdown(self):
        return None

    def print_metrics_summary(self):
        return None

    def get_curve_fitting_analysis(self):
        return None

    @classmethod
    def render_metrics_summary(cls, metrics):
        cls.last_summary_metrics = metrics

    @classmethod
    def analyze_metrics(cls, metrics, print_results=True):
        cls.last_analyzed_metrics = metrics
        cls.analyze_print_results = print_results
        return {}

    def _reset_state_memory_tracking(self):
        self._last_state_memory_total = 0.0

    def _sync_state_memory_tracking_from_metrics(self):
        if self._metrics.state_memory_floats:
            self._last_state_memory_total = self._metrics.state_memory_floats[-1]
        else:
            self._last_state_memory_total = 0.0


class _FakeVideoProcessor:
    """Stub VideoProcessor that satisfies the constructor signature."""

    def __init__(self, *_args, **_kwargs):
        return None

    def get_main_frames_streamed(self) -> int:
        return 0


class _OOMTrackingModel:
    """Minimal model that tracks recover_from_oom usage inside process_video_with_questions."""

    def __init__(self) -> None:
        self.clear_calls = 0
        self.recover_calls = []
        self.teardown_calls = 0
        self.load_calls = []
        self.saved_states = []

    def clear_context(self) -> None:
        self.clear_calls += 1

    def save_state(self):
        state = {"state_id": len(self.saved_states)}
        self.saved_states.append(state)
        return state

    def load_state(self, state):
        self.load_calls.append(state)

    def add_text(self, *_args, **_kwargs):
        return None

    def add_video(self, *_args, **_kwargs):
        return None

    def recover_from_oom(self, state):
        self.recover_calls.append(state)
        self.teardown_after_oom()
        self.load_state(state)

    def teardown_after_oom(self):
        self.teardown_calls += 1
        self.clear_context()


class _StubVideoProcessor:
    """Lightweight video processor used to exercise process_video_with_questions."""

    def __init__(self) -> None:
        self.fps = 1.0
        self.frame_sampler = SimpleNamespace(get_frame_count=lambda _video: 1)
        self.loaded_main = False
        self.add_calls = []
        self.main_frames_added = 0

    def load_main_video(self, _video_path):
        self.loaded_main = True
        return ["frame"]

    def add_main_video_up_to_time(self, _model, target_time: float) -> float:
        self.add_calls.append(target_time)
        self.main_frames_added = int(target_time)
        return target_time

    def load_option_videos(self, option_paths):
        return [[path] for path in option_paths]

    def reset_to_main_video_state(self):
        return None

    def get_main_frames_streamed(self) -> int:
        return self.main_frames_added


def _setup_common_fakes(monkeypatch):
    """Install fake modules and helpers used across tests."""

    def fake_load_model_class(_model_type):
        return _FakeModel, "FakeModel"

    monkeypatch.setattr(main_module, "load_model_class", fake_load_model_class)
    monkeypatch.setattr(main_module, "VideoProcessor", _FakeVideoProcessor)
    monkeypatch.setattr(main_module, "QuestionProcessor", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(main_module, "TextToVideoProcessor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_module,
        "WandbLogger",
        lambda enabled=True: SimpleNamespace(
            log_video_completion=lambda **_kwargs: None,
            log_final_results=lambda **_kwargs: None,
            finish=lambda: None,
        ),
    )

    _install_cuda_stubs(monkeypatch)

    _FakeModel.last_summary_metrics = None
    _FakeModel.last_analyzed_metrics = None
    _FakeModel.analyze_print_results = None


def _install_cuda_stubs(monkeypatch):
    """Ensure main_module.torch exposes a CUDA namespace for tests."""

    cuda_iface = getattr(main_module.torch, "cuda", None)
    if cuda_iface is None:
        cuda_iface = SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda *args, **kwargs: None,
            OutOfMemoryError=RuntimeError,
        )
        setattr(main_module.torch, "cuda", cuda_iface)

    monkeypatch.setattr(main_module.torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(main_module.torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(
        main_module.torch.cuda,
        "reset_peak_memory_stats",
        lambda *args, **kwargs: None,
        raising=False,
    )


def _write_test_manifest(tmp_path, num_videos: int = 1):
    """Create a minimal evaluation manifest for tests."""

    videos = []
    for idx in range(num_videos):
        options = []
        for opt_idx in range(2):
            options.append(
                {
                    "label": f"Option {opt_idx}",
                    "sequence": [str(opt_idx)],
                    "present": opt_idx == (idx % 2),
                    "clip_path": f"option{idx}_{opt_idx}.mp4",
                    "clip_start": 0.0,
                    "clip_end": 1.0,
                    "present_by_seq": {"S1": opt_idx == (idx % 2)},
                    "likelihoods": {"S1": 1.0 if opt_idx == (idx % 2) else 0.0},
                    "likelihood_combined": 1.0 if opt_idx == (idx % 2) else 0.0,
                }
            )

        options.append(
            {
                "label": "Uncertain / IDK",
                "sequence": None,
                "present": False,
                "clip_path": None,
                "clip_start": None,
                "clip_end": None,
            }
        )

        videos.append(
            {
                "video_index": idx,
                "video_path": f"video_{idx}.mp4",
                "questions": [
                    {
                        "question_id": idx + 1,
                        "question": "Did you see this clip?",
                        "question_time": 1.0 + idx,
                        "options": options,
                        "correct_index": idx % 2,
                        "clip_start_time": 0.0,
                        "clip_end_time": 1.0,
                    }
                ],
            }
        )

    manifest = {"videos": videos}

    manifest_path = tmp_path / "questions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_process_video_recover_from_oom_invokes_model_hooks(monkeypatch):
    """process_video_with_questions must clear GPU state via recover_from_oom before retrying."""

    _install_cuda_stubs(monkeypatch)
    monkeypatch.setattr(main_module.check_overheat, "pause_needed", lambda: False)
    monkeypatch.setattr(main_module.check_overheat, "pause", lambda: None)

    model = _OOMTrackingModel()
    video_processor = _StubVideoProcessor()

    def _raise_oom(*_args, **_kwargs):
        raise main_module.torch.cuda.OutOfMemoryError("unit-test OOM")

    question_processor = SimpleNamespace(process_single_question=_raise_oom)

    question = QuestionEntry(
        question_id="1",
        prompt="",
        question_time=3.0,
        options=[
            OptionEntry(
                source_index=0,
                label="Option A",
                clip_path="option0.mp4",
                metadata={},
            )
        ],
        correct_answer_index=0,
        dont_know_index=1,
        clip_start_time=None,
        clip_end_time=None,
        metadata={},
    )
    video_data = VideoEntry(
        video_index=0,
        video_path="main.mp4",
        questions=[question],
        metadata={},
    )

    with pytest.raises(main_module.QuestionOOMRetry) as exc_info:
        main_module.process_video_with_questions(
            video_data,
            model,
            video_processor,
            question_processor,
            max_tokens=16,
            max_frames=32,
            text_to_video_processor=None,
            no_option_text=False,
        )

    assert exc_info.value.question_index == 0
    assert exc_info.value.oom_timestamp == pytest.approx(3.0)
    assert model.saved_states, "Model should persist a state snapshot before asking the question."
    expected_state = model.saved_states[0]
    assert model.recover_calls == [expected_state], "recover_from_oom must be invoked with the saved state."
    assert model.teardown_calls == 1, "OOM recovery should perform exactly one teardown before reload."
    assert model.load_calls == [expected_state], "State reload should occur after teardown."
    assert model.clear_calls == 2, "clear_context should run once initially and once during teardown."
    assert video_processor.loaded_main is True
    assert video_processor.add_calls == [3.0]


def test_restart_on_oom_recovers_latest_video(monkeypatch, tmp_path):
    """The restart_on_oom flag should reinstantiate the model and retry the failing video."""

    _setup_common_fakes(monkeypatch)

    _FakeModel.created_instances = 0
    call_counter = {"count": 0}

    def fake_process(*_args, **_kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise main_module.torch.cuda.OutOfMemoryError("test oom")
        return [{"is_correct": True, "is_dont_know": False}]

    monkeypatch.setattr(main_module, "process_video_with_questions", fake_process)

    manifest_path = _write_test_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(manifest_path),
            "--restart_on_oom",
            "--max_oom_retries",
            "5",
        ],
    )

    main_module.main()

    assert call_counter["count"] == 2, "Expected the failing video to be retried once after OOM."
    assert _FakeModel.created_instances == 2, "Model should be reinstantiated after the OOM event."


def test_max_oom_retries_limits_infinite_loops(monkeypatch, tmp_path):
    """Retries must stop once the configured max_oom_retries threshold is exceeded."""

    _setup_common_fakes(monkeypatch)

    _FakeModel.created_instances = 0
    call_counter = {"count": 0}

    def always_fail(*_args, **_kwargs):
        call_counter["count"] += 1
        raise main_module.torch.cuda.OutOfMemoryError("persistent oom")

    monkeypatch.setattr(main_module, "process_video_with_questions", always_fail)

    manifest_path = _write_test_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(manifest_path),
            "--restart_on_oom",
            "--max_oom_retries",
            "2",
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        main_module.main()

    assert "Exceeded maximum CUDA OOM retries" in str(exc_info.value)
    assert call_counter["count"] == 3, "Two retries plus the initial attempt should be recorded."


def test_metrics_accumulate_across_videos(monkeypatch, tmp_path):
    """Metrics should accumulate across videos when curve fitting runs."""

    _setup_common_fakes(monkeypatch)

    _FakeModel.created_instances = 0
    call_counter = {"count": 0}
    totals = {"last_total": 0.0}

    def populate_metrics(*args, **_kwargs):
        call_counter["count"] += 1
        model = args[1]
        metrics = model._metrics
        previous_total = totals["last_total"]
        new_total = float(call_counter["count"] * 50.0)
        delta = new_total - previous_total
        totals["last_total"] = new_total
        metrics.latency_add_video.append(0.1)
        metrics.video_timestamps_add_video.append(5.0 * call_counter["count"])
        metrics.latency_ask_question.append(float(call_counter["count"]))
        metrics.video_timestamps_ask_question.append(10.0 * call_counter["count"])
        metrics.state_memory_floats.append(new_total)
        metrics.state_memory_after_add_video.append(new_total)
        metrics.state_memory_delta_add_video.append(delta)
        metrics.state_memory_after_ask_question.append(new_total)
        metrics.state_memory_delta_ask_question.append(0.0)
        return [{"is_correct": True, "is_dont_know": False}]

    monkeypatch.setattr(main_module, "process_video_with_questions", populate_metrics)

    manifest_path = _write_test_manifest(tmp_path, num_videos=2)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(manifest_path),
            "--enable_metrics",
        ],
    )

    main_module.main()

    metrics_summary = _FakeModel.last_summary_metrics
    assert metrics_summary is not None, "Expected the summary to receive aggregated metrics."
    assert len(metrics_summary.latency_ask_question) == 2
    assert metrics_summary.latency_ask_question == [1.0, 2.0]

    analyzed = _FakeModel.last_analyzed_metrics
    assert analyzed is metrics_summary
    assert _FakeModel.analyze_print_results is True


def test_retried_video_preserves_question_metrics(monkeypatch, tmp_path):
    """Metrics collected before an OOM retry should persist into the successful run."""

    _setup_common_fakes(monkeypatch)

    call_counter = {"count": 0}
    state = {"partial_results": None}

    def fake_process(video_data, model, *_args, **_kwargs):
        call_counter["count"] += 1
        metrics = model._metrics
        if call_counter["count"] == 1:
            metrics.question_correctness_rate.append(1.0)
            metrics.question_dont_know_rate.append(0.0)
            metrics.question_answered_mask.append(1.0)
            metrics.video_timestamps_question_outcome.append(5.0)
            partial_results = [{"is_correct": True, "is_dont_know": False}]
            state["partial_results"] = partial_results
            exc = main_module.torch.cuda.OutOfMemoryError("simulated oom")
            raise main_module.QuestionOOMRetry(
                question_index=1,
                partial_results=copy.deepcopy(partial_results),
                original_exception=exc,
                oom_timestamp=5.0,
                partial_metrics=copy.deepcopy(metrics),
            )

        assert len(metrics.question_correctness_rate) == 1
        assert len(metrics.question_answered_mask) == 1
        metrics.question_correctness_rate.append(0.0)
        metrics.question_dont_know_rate.append(0.0)
        metrics.question_answered_mask.append(1.0)
        metrics.video_timestamps_question_outcome.append(10.0)

        prior = state.get("partial_results", [])
        return prior + [{"is_correct": False, "is_dont_know": False}]

    monkeypatch.setattr(main_module, "process_video_with_questions", fake_process)

    manifest_path = _write_test_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(manifest_path),
            "--enable_metrics",
            "--limit_frames_on_oom",
        ],
    )

    main_module.main()

    assert call_counter["count"] == 2

    metrics_summary = _FakeModel.last_summary_metrics
    assert metrics_summary is not None
    assert metrics_summary.question_correctness_rate == [1.0, 0.0]
    assert metrics_summary.question_answered_mask == [1.0, 1.0]
    assert metrics_summary.question_dont_know_rate == [0.0, 0.0]


def test_question_summary_handles_unanswered_entries():
    """summarize_questions should report intuitive counts and rates."""

    metrics = main_module.PerformanceMetrics()
    metrics.question_correctness_rate = [1.0, 0.0, 0.0]
    metrics.question_dont_know_rate = [0.0, 1.0, 0.0]
    metrics.question_answered_mask = [1.0, 1.0, 0.0]

    summary = metrics.summarize_questions()
    assert summary is not None
    assert summary.total == 3
    assert summary.answered == 2
    assert summary.unanswered == 1
    assert summary.correct == 1
    assert summary.dont_know == 1
    assert summary.overall_accuracy == pytest.approx(1 / 3)
    assert summary.answered_accuracy == pytest.approx(0.5)
    assert summary.dont_know_rate == pytest.approx(1 / 3)
    assert summary.answered_dont_know_rate == pytest.approx(0.5)
