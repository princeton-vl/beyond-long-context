"""Tests specific to question processing utilities."""

from datasets.patternvideos_manifest import OptionEntry, QuestionEntry
from processors.question_processor import QuestionProcessor


class _CaptureModel:
    """Minimal model stub that records question timing."""

    def __init__(self) -> None:
        self.last_current_time = None

    def ask_question(self, _question, current_video_time=0.0, **_kwargs):
        self.last_current_time = current_video_time
        return "{0}"


def _make_question(question_id: str = "7", correct_index: int = 0, question_time: float = 12.5) -> QuestionEntry:
    options = [
        OptionEntry(source_index=0, label="Option A", clip_path="opt0.mp4"),
        OptionEntry(source_index=1, label="Option B", clip_path="opt1.mp4"),
    ]
    return QuestionEntry(
        question_id=str(question_id),
        prompt="",
        question_time=question_time,
        options=options,
        correct_answer_index=correct_index,
        dont_know_index=len(options),
        clip_start_time=None,
        clip_end_time=None,
        metadata={},
    )


def test_process_single_question_passes_current_video_time():
    model = _CaptureModel()
    processor = QuestionProcessor(verbose=False, no_describe=True)

    question = _make_question()

    processor.process_single_question(
        model,
        question,
        video_index=0,
        max_tokens=32,
        max_frames=128,
        current_video_time=42.0,
    )

    assert model.last_current_time == 42.0


def test_process_single_question_falls_back_to_question_time():
    model = _CaptureModel()
    processor = QuestionProcessor(verbose=False, no_describe=True)

    question = _make_question(question_id="8", correct_index=1, question_time=33.3)

    processor.process_single_question(
        model,
        question,
        video_index=0,
        max_tokens=32,
        max_frames=128,
        current_video_time=None,
    )

    assert model.last_current_time == 33.3


def test_extract_answer_strips_punctuation_for_numeric():
    processor = QuestionProcessor(verbose=False, no_describe=True)

    response = "Explanation then final choice { 2.? }"

    assert processor.extract_answer(response) == "2"


def test_extract_answer_strips_punctuation_for_binary():
    processor = QuestionProcessor(verbose=False, no_describe=True, binary_questions=True)

    response = "Evidence {Yes! }"

    assert processor.extract_answer(response) == "Yes"
