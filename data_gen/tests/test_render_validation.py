import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from seq2vid.render import (
    QuestionContext,
    _prepare_letter_rendering,
    _render_true_slice_clip,
    _resolve_media_path,
    _write_clip_frame_debug,
    _write_frame_debug,
    _write_validation_manifest,
    _clip_frames_match_video,
    run_render,
    _durations,
)
from seq2vid.simple_letter_renderer import render_sequence_frames
from vidgeom import load_template


class RenderValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name) / "bucket"
        (self.out_dir / "videos").mkdir(parents=True)
        (self.out_dir / "clips").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _create_file(self, relative: str, payload: bytes) -> Path:
        path = self.out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _make_context(self, frame_debug: list) -> QuestionContext:
        video_path = self._create_file("videos/video_debug.mp4", b"")
        return QuestionContext(
            template_path=Path("examples/template_conveyor_letters.yaml"),
            video_id="debug",
            seq_map={"S_tokens": ["1", "2"], "S_lanes": ["0", "1"]},
            seq_names=["S_tokens", "S_lanes"],
            entropy_cache={},
            counts_per_seq={},
            times_main=[0.0, 0.5],
            durations_main=[0.5, 0.5],
            median_step=0.5,
            video_end_time=1.0,
            video_paths=[str(video_path)],
            clips_dir=self.out_dir / "clips",
            clip_options=True,
            questions_only=False,
            questions_at_end=True,
            uniform_uncertain=False,
            rng=np.random.default_rng(0),
            question_min_len=3,
            ngram_max=4,
            fps_override=None,
            ffmpeg_crf=23,
            ffmpeg_preset="veryfast",
            ffmpeg_codec="libx264",
            video_job_seed=None,
            token_letters={"map": {"1": "A", "2": "B"}},
            token_labels={"1": "letter A", "2": "letter B"},
            hard_questions=False,
            video_fps=10,
            capture_frame_debug=True,
            frame_debug=frame_debug,
        )

    def test_validation_manifest_records_hashes(self) -> None:
        video_path = self._create_file("videos/video_1_v0.mp4", b"video-bytes")
        clip_path = self._create_file("clips/video_1_q0_false.mp4", b"clip-bytes")
        payload = [
            {
                "video_index": 1,
                "variant": 0,
                "video_path": str(video_path),
                "questions": [
                    {
                        "question_index": 0,
                        "question_variant": "spatial",
                        "candidate": {"clip_path": str(clip_path)},
                    }
                ],
            }
        ]
        _write_validation_manifest(self.out_dir, payload, log_progress=False)
        manifest_path = self.out_dir / "validation_manifest.json"
        data = json.loads(manifest_path.read_text())
        self.assertEqual(len(data["videos"]), 1)
        entry = data["videos"][0]
        self.assertEqual(entry["video_index"], 1)
        self.assertIn("sha256", entry["media"])
        self.assertEqual(entry["media"]["size_bytes"], video_path.stat().st_size)
        clip_entry = entry["questions"][0]["clip"]
        self.assertEqual(clip_entry["size_bytes"], clip_path.stat().st_size)

    def test_questions_only_runs_skip_video_media(self) -> None:
        clip_path = self._create_file("clips/video_3_q0_true.mp4", b"clip")
        payload = [
            {
                "video_index": 3,
                "variant": 0,
                "video_path": None,
                "questions": [
                    {
                        "question_index": 0,
                        "question_variant": "sequential",
                        "candidate": {"clip_path": str(clip_path)},
                    }
                ],
            }
        ]
        _write_validation_manifest(self.out_dir, payload, log_progress=False)
        manifest_path = self.out_dir / "validation_manifest.json"
        data = json.loads(manifest_path.read_text())
        self.assertIsNone(data["videos"][0]["media"])

    def test_resolve_media_path_honors_relative_tree(self) -> None:
        bucket_dir = Path(self.tmpdir.name) / "bucket"
        target = bucket_dir / "clips" / "video.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"video")
        reference = str(Path("bucket") / "clips" / "video.mp4")
        resolved = _resolve_media_path(reference, bucket_dir)
        self.assertEqual(resolved, target.resolve())

    def test_missing_media_raises(self) -> None:
        video_path = self._create_file("videos/video_2_v0.mp4", b"video-two")
        payload = [
            {
                "video_index": 2,
                "variant": 0,
                "video_path": str(video_path),
                "questions": [
                    {
                        "question_index": 0,
                        "question_variant": "sequential",
                        "candidate": {"clip_path": str(self.out_dir / "clips" / "missing.mp4")},
                    }
                ],
            }
        ]
        with self.assertRaises(FileNotFoundError):
            _write_validation_manifest(self.out_dir, payload, log_progress=False)

    def test_frame_debug_serialization(self) -> None:
        ctx = self._make_context([
            {
                "frame_index": 0,
                "time": 0.0,
                "view_bounds": {"y_min": 0.0, "y_max": 1.0},
                "items": [{"token": "1", "lane": 0, "visible": False, "visibility": "above_frame"}],
            },
            {
                "frame_index": 1,
                "time": 0.1,
                "items": [{"token": "2", "lane": 1, "visible": True, "visibility": "onscreen"}],
            },
        ])
        video_path = Path(ctx.video_paths[0])
        _write_frame_debug(video_path, ctx)
        data = json.loads(video_path.with_suffix(".frames.json").read_text())
        self.assertEqual(len(data["frames"]), 2)
        self.assertIn("view_bounds", data)
        self.assertFalse(data["frames"][0]["items"])
        self.assertEqual(data["frames"][0]["offscreen_items"][0]["letter"], "A")
        self.assertEqual(data["frames"][1]["items"][0]["letter"], "B")

    def test_clip_frame_debug_offsets(self) -> None:
        frame_debug = [
            {
                "frame_index": 0,
                "time": 0.0,
                "items": [{"token": "1", "lane": 0, "visible": True, "visibility": "onscreen"}],
            },
            {
                "frame_index": 1,
                "time": 0.2,
                "items": [{"token": "2", "lane": 1, "visible": True, "visibility": "onscreen"}],
            },
        ]
        ctx = self._make_context(frame_debug)
        clip_path = self._create_file("clips/debug_clip.mp4", b"")
        _write_clip_frame_debug(ctx, clip_path, clip_start=0.0, clip_end=0.2)
        data = json.loads(clip_path.with_suffix(".frames.json").read_text())
        self.assertEqual(len(data["frames"]), 2)
        self.assertAlmostEqual(data["frames"][1]["time"], 0.2, places=3)
        self.assertEqual(data["frames"][1]["clip_frame_index"], 1)
        self.assertEqual(data["frames"][0]["video_frame_index"], 0)

    def test_clip_frame_debug_handles_relative_records(self) -> None:
        ctx = self._make_context([])
        clip_path = self._create_file("clips/relative.mp4", b"")
        frame_records = [
            {"frame_index": 0, "time": 0.0, "items": [{"token": "1", "visible": True}]},
            {"frame_index": 1, "time": 0.5, "items": [{"token": "2", "visible": True}]},
        ]
        _write_clip_frame_debug(
            ctx,
            clip_path,
            clip_start=1.0,
            clip_end=2.0,
            frame_records=frame_records,
        )
        data = json.loads(clip_path.with_suffix(".frames.json").read_text())
        self.assertEqual(len(data["frames"]), 2)
        self.assertEqual(data["frames"][0]["clip_frame_index"], 0)
        self.assertIsNone(data["frames"][0]["video_frame_index"])
        self.assertAlmostEqual(data["frames"][0]["video_time"], 1.0)

    def test_clip_frames_match_video(self) -> None:
        video_frames = [
            {"frame_index": 0, "items": []},
            {"frame_index": 1, "items": [{"token": "1"}]},
            {"frame_index": 2, "items": [{"token": "1"}, {"token": "2"}]},
        ]
        clip_frames = [
            {"video_frame_index": 1, "items": [{"token": "1"}]},
            {"video_frame_index": 2, "items": [{"token": "1"}, {"token": "2"}]},
        ]
        self.assertTrue(_clip_frames_match_video(clip_frames, video_frames))

    def test_clip_frames_detect_mismatch(self) -> None:
        video_frames = [
            {"frame_index": 0, "items": []},
            {"frame_index": 1, "items": [{"token": "1"}]},
        ]
        clip_frames = [
            {"video_frame_index": 1, "items": [{"token": "2"}]},
        ]
        self.assertFalse(_clip_frames_match_video(clip_frames, video_frames))

    def test_letter_plan_covers_declared_tokens(self) -> None:
        template_path = Path("examples/template_conveyor_letters.yaml")
        seq_map = {
            "S_tokens": ["0", "1", "2", "3"],
            "S_lanes": ["0", "1", "1", "2"],
        }
        prep = _prepare_letter_rendering(template_path, "job_letters", seq_map)
        letter_map = prep.asset_plan["token_letters"]["map"]
        template_raw = load_template(str(template_path)).raw
        declared = [
            str(tok)
            for tok in (template_raw.get("vocab", {}).get("token_ids") or [])
        ]
        for tok in declared:
            self.assertIn(tok, letter_map)

    def test_true_clips_rerender_without_video_source(self) -> None:
        template_path = Path("examples/template_conveyor_letters.yaml")
        seq_map = {
            "S_tokens": ["0", "1", "2", "3"],
            "S_lanes": ["0", "1", "1", "2"],
        }
        prep = _prepare_letter_rendering(template_path, "video_letters", seq_map)
        times_main = prep.times["S_tokens"]
        durations_main = _durations(times_main)
        median_step = float(np.median(durations_main))
        video_end_time = times_main[-1] + durations_main[-1]
        _, frame_debug = render_sequence_frames(
            prep.plan,
            seq_map["S_tokens"],
            seq_map["S_lanes"],
        )
        ctx = QuestionContext(
            template_path=template_path,
            video_id="debug_true",
            seq_map=seq_map,
            seq_names=["S_tokens", "S_lanes"],
            entropy_cache={},
            counts_per_seq={},
            times_main=times_main,
            durations_main=durations_main,
            median_step=median_step,
            video_end_time=video_end_time,
            video_paths=[str(self.out_dir / "videos" / "missing.mp4")],
            clips_dir=self.out_dir / "clips",
            clip_options=True,
            questions_only=False,
            questions_at_end=True,
            uniform_uncertain=False,
            rng=np.random.default_rng(0),
            question_min_len=3,
            ngram_max=4,
            fps_override=None,
            ffmpeg_crf=23,
            ffmpeg_preset="veryfast",
            ffmpeg_codec="libx264",
            video_job_seed=None,
            token_letters=prep.asset_plan["token_letters"],
            token_labels={},
            hard_questions=False,
            video_fps=prep.plan.fps,
            capture_frame_debug=True,
            frame_debug=frame_debug,
            render_plan=prep.plan,
        )
        clip_path, clip_start, clip_end = _render_true_slice_clip(
            ctx,
            idx_q=0,
            clip_suffix="true_sequential",
            start_idx=0,
            length=2,
        )
        self.assertTrue(clip_path.exists())
        self.assertGreater(clip_path.stat().st_size, 0)
        payload = json.loads(clip_path.with_suffix(".frames.json").read_text())
        self.assertEqual(payload["frames"][0]["video_frame_index"], 0)
        self.assertAlmostEqual(clip_start, times_main[0])
        self.assertAlmostEqual(clip_end, times_main[0] + 2 * prep.plan.frame_duration)


class RenderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_sequence_file(self, name: str, tokens: Sequence[int]) -> Path:
        payload = {
            "sequences": [
                {
                    "seq_id": name,
                    "tokens": [str(t) for t in tokens],
                    "entropy": 4.0,
                    "entropy_units": "bits",
                    "length": len(tokens),
                    "vocab_size": len(set(tokens)),
                    "top_ngrams": [
                        {
                            "n": min(4, len(tokens)),
                            "ngram": [str(t) for t in tokens[: min(4, len(tokens))]],
                            "count": 1,
                        }
                    ],
                }
            ]
        }
        path = self.base / f"{name}.json"
        path.write_text(json.dumps(payload))
        return path

    def _ffprobe_frame_count(self, clip_path: Path) -> int:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0
        data = result.stdout.strip()
        return int(data) if data.isdigit() else 0

    def _load_frames(self, clip_path: Path) -> List[dict]:
        frames_path = clip_path.with_suffix(".frames.json")
        payload = json.loads(frames_path.read_text())
        return payload.get("frames", [])

    def _assert_clip_frames(self, clip_path: Path, candidate: Dict[str, Any]) -> None:
        frames = self._load_frames(clip_path)
        self.assertTrue(frames, f"No frame metadata for {clip_path}")
        tokens = [str(tok) for tok in candidate.get("sequences", {}).get("S_tokens", [])]
        lanes_seq = candidate.get("sequences", {}).get("S_lanes")
        lanes = [str(l) for l in lanes_seq] if lanes_seq else None
        for frame in frames:
            items = frame.get("items", [])
            self.assertLessEqual(len(items), 8)
            self.assertTrue(
                any(str(item.get("token")) in tokens for item in items),
                "Every frame must include a target token",
            )
        if not tokens:
            return
        base_index: Optional[int] = None
        for frame in frames:
            for item in frame.get("items", []):
                seq_idx = item.get("seq_index")
                if seq_idx is None:
                    continue
                token_val = str(item.get("token"))
                lane_val = str(item.get("lane")) if "lane" in item else None
                if token_val == tokens[0] and (lanes is None or lane_val == lanes[0]):
                    base_index = int(seq_idx)
                    break
            if base_index is not None:
                break
        self.assertIsNotNone(base_index, "First token never appears in clip")
        next_seq = base_index or 0
        matched = 0
        for frame in frames:
            progressed = True
            while progressed and matched < len(tokens):
                progressed = False
                target_token = tokens[matched]
                target_lane = lanes[matched] if lanes else None
                for item in frame.get("items", []):
                    seq_idx = item.get("seq_index")
                    if seq_idx is None or int(seq_idx) != next_seq:
                        continue
                    token_val = str(item.get("token"))
                    if token_val != target_token:
                        continue
                    lane_val = str(item.get("lane")) if "lane" in item else None
                    if target_lane is not None and lane_val != target_lane:
                        continue
                    matched += 1
                    next_seq += 1
                    progressed = True
                    break
        self.assertEqual(matched, len(tokens), "Clip is missing one or more tokens")
        last_token = tokens[-1]
        self.assertTrue(
            any(str(item.get("token")) == last_token for item in frames[-1].get("items", [])),
            "Final token never appears in the last frame",
        )
        max_seq_index = max(
            (int(item.get("seq_index")) for frame in frames for item in frame.get("items", []) if item.get("seq_index") is not None),
            default=next_seq,
        )
        self.assertLessEqual(max_seq_index, (base_index or 0) + len(tokens) - 1)
        ffprobe_frames = self._ffprobe_frame_count(clip_path)
        if ffprobe_frames:
            self.assertEqual(ffprobe_frames, len(frames))

    def _assert_no_extra_clips(self, clips_dir: Path) -> None:
        for clip_file in clips_dir.glob("*.mp4"):
            self.assertNotIn("_opt", clip_file.name)

    def test_membership_render_outputs_clips(self) -> None:
        tokens_file = self._write_sequence_file("S_tokens", list(range(1, 13)))
        lanes_file = self._write_sequence_file("S_lanes", [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        out_dir = self.base / "membership"
        run_render(
            template_path=Path("examples/template_conveyor_letters.yaml"),
            sequences_file=None,
            out_dir=out_dir,
            num_questions=2,
            log_progress=False,
            clip_options=True,
            questions_only=False,
            questions_at_end=True,
            ffmpeg_crf=32,
            ffmpeg_preset="ultrafast",
            ffmpeg_codec="libx264",
            max_videos=1,
            question_min_len=3,
            render_workers=1,
            assignment_seed=123,
            fps_override=10,
            uniform_uncertain=False,
            question_mode="exists",
            hide_question_text=False,
            sequence_sources={
                "S_tokens": tokens_file,
                "S_lanes": lanes_file,
            },
            spatial_question_fraction=0.5,
            hard_questions=False,
            capture_frame_debug=True,
            validate_outputs=False,
        )
        payload = json.loads((out_dir / "questions.json").read_text())
        clips_dir = out_dir / "clips"
        self._assert_no_extra_clips(clips_dir)
        for video in payload.get("videos", []):
            for question in video.get("questions", []):
                self.assertEqual(question.get("question_format"), "binary_yes_no")
                candidate = question.get("candidate", {})
                clip_path = Path(candidate.get("clip_path", ""))
                self.assertTrue(clip_path.exists(), "Missing candidate clip")
                self._assert_clip_frames(clip_path, candidate)
                self.assertEqual(
                    question.get("answer"),
                    "yes" if candidate.get("present") else "no",
                )
                self.assertEqual(candidate.get("clip_start"), question.get("clip_start_time"))
                self.assertEqual(candidate.get("clip_end"), question.get("clip_end_time"))

    def test_continuation_render_outputs_clips(self) -> None:
        tokens_file = self._write_sequence_file("S_tokens", list(range(0, 16)))
        lanes_file = self._write_sequence_file("S_lanes", [0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1])
        out_dir = self.base / "continuation"
        run_render(
            template_path=Path("examples/template_conveyor_letters.yaml"),
            sequences_file=None,
            out_dir=out_dir,
            num_questions=2,
            log_progress=False,
            clip_options=True,
            questions_only=False,
            questions_at_end=True,
            ffmpeg_crf=32,
            ffmpeg_preset="ultrafast",
            ffmpeg_codec="libx264",
            max_videos=1,
            question_min_len=4,
            render_workers=1,
            assignment_seed=456,
            fps_override=10,
            uniform_uncertain=False,
            question_mode="continuation",
            hide_question_text=False,
            sequence_sources={
                "S_tokens": tokens_file,
                "S_lanes": lanes_file,
            },
            spatial_question_fraction=0.5,
            hard_questions=False,
            capture_frame_debug=True,
            validate_outputs=False,
        )
        payload = json.loads((out_dir / "questions.json").read_text())
        clips_dir = out_dir / "clips"
        self._assert_no_extra_clips(clips_dir)
        for video in payload.get("videos", []):
            for question in video.get("questions", []):
                self.assertIn("prefix_clip_path", question)
                prefix_path = Path(question.get("prefix_clip_path", ""))
                self.assertTrue(prefix_path.exists(), "Missing prefix clip")
                prefix_candidate = {
                    "sequences": question.get("prefix", {}),
                }
                self._assert_clip_frames(prefix_path, prefix_candidate)
                candidate = question.get("candidate", {})
                clip_path = Path(candidate.get("clip_path", ""))
                self.assertTrue(clip_path.exists(), "Missing continuation clip")
                self._assert_clip_frames(clip_path, candidate)
                self.assertEqual(
                    question.get("answer"),
                    "yes" if candidate.get("present") else "no",
                )
                self.assertEqual(candidate.get("clip_start"), question.get("clip_start_time"))
                self.assertEqual(candidate.get("clip_end"), question.get("clip_end_time"))


if __name__ == "__main__":
    unittest.main()
