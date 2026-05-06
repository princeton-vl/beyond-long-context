import tempfile
import unittest
from pathlib import Path

import numpy as np

from seq2vid.render import (
    QuestionContext,
    _build_entropy_cache_map,
    _contains_joint_subsequence,
    _contains_subsequence,
    _build_exists_question,
    _collect_continuations,
    _ngram_counts,
)


class SpatialQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_context(self) -> QuestionContext:
        seq_tokens = [str(x) for x in (1, 3, 2, 5, 1, 4, 2, 6, 3, 5)]
        seq_lanes = [str(x) for x in (5, 5, 1, 2, 5, 3, 1, 3, 4, 2)]
        seq_map = {"S_tokens": seq_tokens, "S_lanes": seq_lanes}
        entropy_cache = _build_entropy_cache_map(seq_map)
        counts_per_seq = {name: _ngram_counts(tokens) for name, tokens in seq_map.items()}
        times_main = [float(i) for i in range(len(seq_tokens))]
        durations_main = [1.0] * len(times_main)
        return QuestionContext(
            template_path=Path("template_mock.yaml"),
            video_id="video_test",
            seq_map=seq_map,
            seq_names=["S_tokens", "S_lanes"],
            entropy_cache=entropy_cache,
            counts_per_seq=counts_per_seq,
            times_main=times_main,
            durations_main=durations_main,
            median_step=1.0,
            video_end_time=float(len(times_main)),
            video_paths=[],
            clips_dir=Path(self.tmpdir.name),
            clip_options=False,
            questions_only=True,
            questions_at_end=False,
            uniform_uncertain=False,
            rng=np.random.default_rng(0),
            question_min_len=2,
            ngram_max=6,
            fps_override=None,
            ffmpeg_crf=23,
            ffmpeg_preset="veryfast",
            ffmpeg_codec="libx264",
            video_job_seed=None,
        )

    def _make_constant_lane_context(self) -> QuestionContext:
        seq_tokens = [str(x) for x in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
        seq_lanes = ["0"] * len(seq_tokens)
        seq_map = {"S_tokens": seq_tokens, "S_lanes": seq_lanes}
        entropy_cache = _build_entropy_cache_map(seq_map)
        counts_per_seq = {name: _ngram_counts(tokens) for name, tokens in seq_map.items()}
        times_main = [float(i) for i in range(len(seq_tokens))]
        durations_main = [1.0] * len(times_main)
        return QuestionContext(
            template_path=Path("template_mock.yaml"),
            video_id="video_constant",
            seq_map=seq_map,
            seq_names=["S_tokens", "S_lanes"],
            entropy_cache=entropy_cache,
            counts_per_seq=counts_per_seq,
            times_main=times_main,
            durations_main=durations_main,
            median_step=1.0,
            video_end_time=float(len(times_main)),
            video_paths=[],
            clips_dir=Path(self.tmpdir.name),
            clip_options=False,
            questions_only=True,
            questions_at_end=False,
            uniform_uncertain=False,
            rng=np.random.default_rng(0),
            question_min_len=2,
            ngram_max=6,
            fps_override=None,
            ffmpeg_crf=23,
            ffmpeg_preset="veryfast",
            ffmpeg_codec="libx264",
            video_job_seed=None,
        )

    def _make_sparse_lane_context(self) -> QuestionContext:
        seq_tokens = [
            "8",
            "2",
            "5",
            "9",
            "6",
            "15",
            "8",
            "10",
            "2",
            "10",
            "0",
            "13",
            "3",
            "7",
            "5",
            "2",
            "15",
            "8",
            "10",
            "2",
            "10",
            "0",
            "15",
            "8",
            "10",
            "2",
            "10",
            "0",
            "15",
            "8",
            "10",
            "2",
            "3",
            "14",
            "15",
            "8",
            "10",
            "2",
            "10",
            "0",
        ]
        seq_lanes = [
            "1",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
            "2",
            "4",
        ]
        seq_map = {"S_tokens": seq_tokens, "S_lanes": seq_lanes}
        entropy_cache = _build_entropy_cache_map(seq_map)
        counts_per_seq = {name: _ngram_counts(tokens) for name, tokens in seq_map.items()}
        times_main = [float(i) * 0.6 for i in range(len(seq_tokens))]
        durations_main = [0.6] * len(seq_tokens)
        return QuestionContext(
            template_path=Path("template_mock.yaml"),
            video_id="video_sparse",
            seq_map=seq_map,
            seq_names=["S_tokens", "S_lanes"],
            entropy_cache=entropy_cache,
            counts_per_seq=counts_per_seq,
            times_main=times_main,
            durations_main=durations_main,
            median_step=0.6,
            video_end_time=times_main[-1] + durations_main[-1],
            video_paths=[],
            clips_dir=Path(self.tmpdir.name),
            clip_options=False,
            questions_only=True,
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
        )

    def test_spatial_distractors_use_joint_sequences(self) -> None:
        ctx = self._make_context()
        pos = 6
        question = ctx.make_exists_question(idx_q=0, pos=pos, max_stat_n=4, spatial=True)
        self.assertEqual(question["question_variant"], "spatial")
        self.assertEqual(question["question_type"], "spatial")
        candidate = question["candidate"]
        seqs = candidate["sequences"]
        self.assertGreaterEqual(len(seqs["S_tokens"]), 4)
        if question["answer"] == "yes":
            self.assertTrue(candidate["present"])
            self.assertTrue(
                _contains_joint_subsequence(ctx.seq_map, ctx.seq_names, seqs)
            )
        else:
            self.assertFalse(candidate["present"])
            self.assertFalse(
                _contains_joint_subsequence(ctx.seq_map, ctx.seq_names, seqs)
            )

    def test_contains_joint_subsequence_requires_alignment(self) -> None:
        seq_map = {
            "S_tokens": list("abcdef"),
            "S_lanes": list("112233"),
        }
        seq_names = ["S_tokens", "S_lanes"]
        candidate = {
            "S_tokens": list("bcd"),
            "S_lanes": list("122"),
        }
        self.assertTrue(_contains_joint_subsequence(seq_map, seq_names, candidate, end_idx=4))
        mismatched = {
            "S_tokens": list("bcd"),
            "S_lanes": list("223"),
        }
        self.assertFalse(_contains_joint_subsequence(seq_map, seq_names, mismatched, end_idx=5))

    def test_spatial_question_raises_when_lanes_constant(self) -> None:
        ctx = self._make_constant_lane_context()
        pos = len(ctx.tokens_main) - 1
        # First sequential question advances RNG as in production runs.
        _build_exists_question(ctx, idx_q=0, pos=pos, max_stat_n=ctx.ngram_max, spatial=False)
        with self.assertRaises(RuntimeError):
            _build_exists_question(
                ctx,
                idx_q=1,
                pos=pos,
                max_stat_n=ctx.ngram_max,
                spatial=True,
            )

    def test_spatial_continuation_min_length(self) -> None:
        ctx = self._make_sparse_lane_context()
        question = ctx.make_continuation_question(idx_q=0, spatial=True)
        self.assertEqual(question["question_type"], "spatial")
        prefix_len = len(question["prefix"]["S_tokens"])
        self.assertGreaterEqual(prefix_len, 4)
        candidate = question["candidate"]
        self.assertGreaterEqual(len(candidate["sequences"]["S_tokens"]), 4)
        forbidden = _collect_continuations(
            ctx,
            question["prefix"],
            len(question["prefix"]["S_tokens"]),
            len(candidate["sequences"]["S_tokens"]),
        )
        slices = candidate["sequences"]
        if question["answer"] == "yes":
            self.assertTrue(
                any(
                    all(slices[name] == cont[1][name] for name in ctx.seq_names)
                    for cont in forbidden
                )
            )
        else:
            self.assertFalse(
                any(
                    all(slices[name] == cont[1][name] for name in ctx.seq_names)
                    for cont in forbidden
                )
            )

    def test_false_spatial_question_never_uses_present_sequence(self) -> None:
        ctx = self._make_context()
        pos = 6
        for _ in range(64):
            question = ctx.make_exists_question(idx_q=0, pos=pos, max_stat_n=ctx.ngram_max, spatial=True)
            if question["answer"] == "no":
                slices = question["candidate"]["sequences"]
                self.assertTrue(
                    _contains_subsequence(ctx.seq_map[ctx.primary_seq_name], slices[ctx.primary_seq_name])
                )
                self.assertFalse(
                    _contains_joint_subsequence(ctx.seq_map, ctx.seq_names, slices)
                )
                break
        else:
            self.fail("Failed to sample a false spatial question")

    def test_false_sequential_question_never_repeats(self) -> None:
        ctx = self._make_context()
        pos = 5
        for _ in range(64):
            question = ctx.make_exists_question(idx_q=0, pos=pos, max_stat_n=ctx.ngram_max, spatial=False)
            if question["answer"] == "no":
                slices = question["candidate"]["sequences"]
                self.assertFalse(
                    _contains_subsequence(ctx.seq_map[ctx.primary_seq_name], slices[ctx.primary_seq_name])
                )
                break
        else:
            self.fail("Failed to sample a false sequential question")

    def test_spatial_question_raises_when_no_unused_lanes(self) -> None:
        ctx = self._make_constant_lane_context()
        pos = len(ctx.tokens_main) - 1
        with self.assertRaises(RuntimeError):
            ctx.make_exists_question(idx_q=0, pos=pos, max_stat_n=ctx.ngram_max, spatial=True)


if __name__ == "__main__":
    unittest.main()
