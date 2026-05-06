import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seq2vid.validate import validate_bucket


class BucketValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.bucket = Path(self.tmpdir.name) / "bucket"
        (self.bucket / "clips").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_clip(self, name: str, frames: list) -> Path:
        clip_path = self.bucket / "clips" / f"{name}.mp4"
        clip_path.write_bytes(b"fake")
        frames_path = clip_path.with_suffix(".frames.json")
        payload = {"clip": str(clip_path), "frames": frames, "fps": 10}
        frames_path.write_text(json.dumps(payload))
        return clip_path

    def _write_questions(self, questions: list) -> None:
        payload = {"videos": [{"questions": questions}]}
        (self.bucket / "questions.json").write_text(json.dumps(payload))

    def _basic_question(self, clip_name: str, tokens: list[str]) -> dict:
        return {
            "question_mode": "exists",
            "candidate": {
                "clip_path": str(self.bucket / "clips" / f"{clip_name}.mp4"),
                "sequences": {"S_tokens": tokens},
            },
        }

    def test_validator_accepts_well_formed_clip(self) -> None:
        frames = [
            {"items": [{"token": "1"}]},
            {"items": [{"token": "2"}]},
        ]
        self._write_clip("valid", frames)
        self._write_questions([self._basic_question("valid", ["1", "2"])])
        with mock.patch("seq2vid.validate._ffprobe_frame_count", return_value=2):
            errors = validate_bucket(self.bucket, max_frames=6)
        self.assertFalse(errors)

    def test_validator_detects_missing_final_token(self) -> None:
        frames = [
            {"items": [{"token": "1"}]},
            {"items": [{"token": "1"}]},
        ]
        self._write_clip("bad", frames)
        self._write_questions([self._basic_question("bad", ["1", "2"])])
        with mock.patch("seq2vid.validate._ffprobe_frame_count", return_value=2):
            errors = validate_bucket(self.bucket, max_frames=4)
        self.assertTrue(any("missing" in err for err in errors))

    def test_validator_detects_extra_option_clips(self) -> None:
        frames = [
            {"items": [{"token": "7"}]},
        ]
        self._write_clip("main", frames)
        extra = self.bucket / "clips" / "video_opt0_fake.mp4"
        extra.write_bytes(b"fake")
        extra.with_suffix(".frames.json").write_text(
            json.dumps({"clip": str(extra), "frames": frames})
        )
        self._write_questions([self._basic_question("main", ["7"])])
        with mock.patch("seq2vid.validate._ffprobe_frame_count", return_value=1):
            errors = validate_bucket(self.bucket, max_frames=4)
        self.assertTrue(any("unexpected option" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
