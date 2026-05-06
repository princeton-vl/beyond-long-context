"""Tests for the per-question CSV logging helper."""

from utils.question_csv_logger import QUESTION_LOG_HEADERS, write_question_log_csv


def test_write_question_log_csv(tmp_path):
    rows = [
        {
            "video_id": 7,
            "question_id": "q-1",
            "video_entropy": 0.25,
            "correct_answer": "3",
            "model_answer": "3",
        },
        {
            "video_id": 7,
            "question_id": "q-2",
            "video_entropy": None,
            "correct_answer": "4",
            "model_answer": "2",
        },
    ]

    destination = tmp_path / "logs" / "questions.csv"
    written_path = write_question_log_csv(str(destination), rows)

    assert written_path == destination
    assert destination.exists()

    contents = destination.read_text().strip().splitlines()
    assert contents[0] == ",".join(QUESTION_LOG_HEADERS)
    assert contents[1] == "7,q-1,0.25,3,3"
    assert contents[2] == "7,q-2,,4,2"
