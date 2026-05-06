"""Tests for shared FPS coalescing helper."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.base_interface import VideoLanguageModelInterface


class _Dummy(VideoLanguageModelInterface):
    """Minimal concrete subclass for testing the helper."""

    def _setup_model(self, **kwargs):
        self.model = None

    def add_video(self, *args, **kwargs):
        raise NotImplementedError

    def add_text(self, *args, **kwargs):
        raise NotImplementedError

    def ask_question(self, *args, **kwargs):
        raise NotImplementedError

    def get_state(self):
        return {}

    def clear_context(self):
        pass

    def save_state(self):
        return {}

    def load_state(self, state):
        pass


@pytest.fixture
def helper():
    return _Dummy("dummy")


def test_single_value(helper):
    assert helper._coalesce_video_fps([1.0]) == pytest.approx(1.0)


def test_multiple_identical_values(helper):
    assert helper._coalesce_video_fps([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_ignores_none(helper):
    assert helper._coalesce_video_fps([None, 2.5, None]) == pytest.approx(2.5)


def test_returns_none_when_empty(helper):
    assert helper._coalesce_video_fps([]) is None


def test_raises_on_mixed_values(helper):
    with pytest.raises(ValueError):
        helper._coalesce_video_fps([1.0, 2.0])
