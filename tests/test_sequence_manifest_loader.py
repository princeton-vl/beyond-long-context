"""Tests for manifest parsing in sequence mode."""

import json
from pathlib import Path

import pytest

from datasets.patternvideos_manifest import load_patternvideos_manifest


def _make_manifest(tmp_path: Path) -> Path:
    payload = {
        "videos": [
            {
                "video_index": 0,
                "video_path": "",
                "sequences_used": {"S1": ["1", "2", "3"]},
                "questions": [
                    {
                        "question": "Which sequence comes next?",
                        "question_time": 0,
                        "options": [
                            {
                                "label": "Option A",
                                "sequence": ["4", "5"],
                                "present": True,
                            },
                            {
                                "label": "Uncertain / IDK",
                                "sequence": None,
                            },
                        ],
                        "correct_index": 0,
                    }
                ],
            }
        ]
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_requires_assets_by_default(tmp_path):
    manifest_path = _make_manifest(tmp_path)

    with pytest.raises(ValueError):
        load_patternvideos_manifest(str(manifest_path))


def test_manifest_accepts_sequences_without_assets(tmp_path):
    manifest_path = _make_manifest(tmp_path)

    videos = load_patternvideos_manifest(str(manifest_path), require_video_assets=False)

    assert videos[0].video_path == ""
    question = videos[0].questions[0]
    option = question.options[0]
    assert option.clip_path == ""
    assert option.token_sequence == ["4", "5"]
