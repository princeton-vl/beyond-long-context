"""Tests for the sequence-mode processor utilities."""

from datasets.patternvideos_manifest import OptionEntry, QuestionEntry
from processors.sequence_processor import SequenceProcessor, CommaSeparatedSequenceFormatter


class _DummyModel:
    def __init__(self) -> None:
        self.text_events = []

    def add_text(self, text, current_video_time=0.0):
        self.text_events.append((text, current_video_time))


def _make_question(prefix=None, mode="continuation"):
    if prefix is None:
        prefix = {"S1": ["1", "2", "3"]}
    option = OptionEntry(
        source_index=0,
        label="Option A",
        clip_path="",
        token_sequence=["4", "5"],
    )
    return QuestionEntry(
        question_id="q1",
        prompt="Prompt",
        question_time=0.0,
        options=[option],
        correct_answer_index=0,
        dont_know_index=1,
        clip_start_time=None,
        clip_end_time=None,
        metadata={},
        question_order=1,
        question_mode=mode,
        sequence_prefixes=prefix,
    )


def test_sequence_processor_streams_sequence_and_prefix():
    model = _DummyModel()
    processor = SequenceProcessor(
        sequences_used={"S1": ["9", "0"], "S2": ["7"]},
        formatter=CommaSeparatedSequenceFormatter(),
        print_chunks=False,
    )
    question = _make_question(prefix={"S1": ["1", "2"]})

    base_time, base_statements = processor.stream_full_sequences(model)
    assert base_statements == ["Sequence S1: 9, 0", "Sequence S2: 7"]
    assert model.text_events == [
        ("Sequence S1: 9, 0\n\n", 0.0),
        ("Sequence S2: 7\n\n", 1.0),
    ]

    cursor, prefix_statements = processor.stream_question_prefix(
        model,
        question,
        base_time=base_time,
    )

    assert prefix_statements == ["Prefix S1: 1, 2"]
    assert model.text_events[-1] == ("Prefix S1: 1, 2\n\n", 2.0)
    assert cursor == 3.0


def test_sequence_processor_formats_option_sequence():
    processor = SequenceProcessor(
        sequences_used={"S1": ["9"]},
        formatter=CommaSeparatedSequenceFormatter(),
        print_chunks=False,
    )
    option = OptionEntry(
        source_index=0,
        label="Option A",
        clip_path="",
        token_sequence=["7", "8", "9"],
    )

    statement = processor.build_option_statement(0, option, " (Option A)")

    assert "Option 0 (Option A):" in statement
    assert "7, 8, 9" in statement


def test_sequence_processor_exist_mode_streams_full_sequences():
    model = _DummyModel()
    processor = SequenceProcessor(
        sequences_used={"S1": ["4", "5", "6"]},
        formatter=CommaSeparatedSequenceFormatter(),
        print_chunks=False,
    )
    question = _make_question(prefix={}, mode="exist")

    cursor, statements = processor.stream_full_sequences(model)

    assert statements == ["Sequence S1: 4, 5, 6"]
    assert model.text_events == [("Sequence S1: 4, 5, 6\n\n", 0.0)]
    assert cursor == 1.0
