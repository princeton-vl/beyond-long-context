from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

from models.base_interface import VideoLanguageModelInterface


class DummyEvalModel(VideoLanguageModelInterface):
    """Lightweight stand-in model for integration tests."""

    def _setup_model(self, **kwargs: Any) -> None:  # type: ignore[override]
        del kwargs
        self._context: List[Any] = []
        self._question_counter = 0

    def add_video(
        self,
        video_frames: Union[List[Any], Any],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:  # type: ignore[override]
        self._context.append(("video", time_start, time_end, video_id))

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:  # type: ignore[override]
        self._context.append(("text", text, current_video_time))

    def ask_question(  # type: ignore[override]
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        del question, current_video_time, max_tokens, max_frames_in_video, sample_method
        answer_index = self._question_counter % 4
        self._question_counter += 1
        return f"The answer is {{{answer_index}}}."

    def get_state(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "context": copy.deepcopy(self._context),
            "counter": self._question_counter,
        }

    def clear_context(self) -> None:  # type: ignore[override]
        self._context.clear()

    def save_state(self) -> Dict[str, Any]:  # type: ignore[override]
        return self.get_state()

    def load_state(self, state: Dict[str, Any]) -> None:  # type: ignore[override]
        self._context = copy.deepcopy(state.get("context", []))
        self._question_counter = int(state.get("counter", 0))
