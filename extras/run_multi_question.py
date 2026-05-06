#!/usr/bin/env python3
"""
run_multi_question.py — Multi-question batching evaluation.

Tests models by asking 2 or 3 questions simultaneously in a single prompt.
Only for models/trials that scored >58% in the single-question baseline.

Usage:
    envctl run hf448 -- python run_multi_question.py DATASET \\
        --asset-root PATH --model phi_multimodal --num-questions 2 \\
        --input-mode sequence --eval-mode sequential \\
        --sequence-format comma \\
        --question-log-csv PATH --state-file PATH \\
        --bucket-filter "UNIFORM_EVAL_L256_" --resume-state

State/CSV files use multi{N}- prefix and never overwrite buckets-* files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add repo root to import path
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))

from datasets.patternvideos_manifest import (
    BinaryCandidateEntry,
    QuestionEntry,
    VideoEntry,
    load_patternvideos_manifest,
)
from processors.sequence_processor import (
    CommaSeparatedSequenceFormatter,
    SequenceProcessor,
)


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs profiling helper
# ─────────────────────────────────────────────────────────────────────────────

_PRECOMP_DEFAULT: Optional[Path] = None  # Set --poly-json-dir to enable polynomial range warnings.

# Map args.model → key used in precomputed JSON files
_MODEL_TO_POLY_KEY: Dict[str, str] = {
    'phi_multimodal':             'phi4mm',
    'qwen3_full':                 'qwen3dense',
    'qwen3_omni':                 'qwen3omni',
    'glm45v':                     'glm45v',
    'qwen_full':                  'qwen25video',
    'internvl-3-5':               'internvl8b',
    'internvl-3-5-thinking':      'internvl8b-thinking',
    'internvl-3-5-38b':           'internvl38b',
    'internvl-3-5-38b-thinking':  'internvl38b-thinking',
    'internvl-3-5-30b-a3b':       'internvl30ba3b',
    'internvl-3-5-30b-a3b-thinking': 'internvl30ba3b-thinking',
}


def _load_poly_ranges(poly_dir: Path, model_key: str, input_mode: str) -> Dict[int, Tuple]:
    """
    Return {bucket_length: (A, C, tok_min, tok_max)} for polynomial range checks.
    mode_str in filenames: 'text' or 'video_spatial' / 'video_sequential'.
    """
    poly_key = _MODEL_TO_POLY_KEY.get(model_key, model_key)
    mode_prefix = 'text' if input_mode == 'sequence' else 'video_spatial'
    ranges: Dict[int, Tuple] = {}
    if not poly_dir.exists():
        return ranges
    for fpath in poly_dir.glob(f'final_{mode_prefix}_len*_raw_accuracy.json'):
        m = re.search(r'len(\d+)', fpath.name)
        if not m:
            continue
        L = int(m.group(1))
        try:
            with open(fpath) as f:
                entries = {e['model']: e for e in json.load(f)}
            if poly_key in entries:
                e = entries[poly_key]
                tr = e.get('token_range', [None, None])
                ranges[L] = (e['A'], e['C'], tr[0], tr[1])
        except Exception:
            pass
    return ranges


def _pick_gflops(
    token_count: int,
    poly_ranges: Dict[int, Tuple],
    bucket: str,
) -> Tuple[Optional[float], str]:
    """
    Compute GFLOPs from polynomial for this response token count.
    Returns (gflops, source) where source is 'poly' or ''.
    Warns to stdout when token_count falls far outside the fit range
    (flag for follow-up measurement via run_flops.py).
    """
    m = re.search(r'L(\d+)', bucket)
    if not m or not poly_ranges:
        return None, ""
    L = int(m.group(1))
    if L not in poly_ranges:
        return None, ""

    A, C, tok_min, tok_max = poly_ranges[L]
    too_low  = tok_min is not None and token_count < tok_min * 0.5
    too_high = tok_max is not None and token_count > tok_max * 1.5

    if too_low:
        print(f"  [FLOPS WARN] token_count={token_count} far below poly min={tok_min:.0f} (L{L})"
              f" — add to FLOPS_MEASUREMENT_PLAN and re-run run_flops.py", flush=True)
    elif too_high:
        print(f"  [FLOPS WARN] token_count={token_count} far above poly max={tok_max:.0f} (L{L})"
              f" — add to FLOPS_MEASUREMENT_PLAN and re-run run_flops.py", flush=True)

    return A * token_count**2 + C, "poly"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-question batching evaluation (2 or 3 Qs per prompt)"
    )
    p.add_argument("json_path", help="Path to questions JSON dataset")
    p.add_argument("--asset-root", required=True, help="Root directory for video/sequence assets")
    p.add_argument("--model", required=True,
                   help="Model key (e.g. phi_multimodal, qwen3_full, internvl-3-5-30b-a3b)")
    p.add_argument("--num-questions", type=int, default=2, choices=[2, 3],
                   help="Number of questions per prompt group")
    p.add_argument("--input-mode", default="sequence", choices=["sequence", "video"],
                   help="Input modality")
    p.add_argument("--eval-mode", default="sequential", choices=["sequential", "spatial"],
                   help="Question variant to evaluate")
    p.add_argument("--sequence-format", default="comma",
                   help="Sequence token format (comma supported)")
    p.add_argument("--question-log-csv", required=True,
                   help="Output CSV path (never overwrites buckets-* files)")
    p.add_argument("--state-file", required=True,
                   help="Resumable state JSON (never overwrites buckets-* files)")
    p.add_argument("--bucket-filter", action="append", dest="bucket_filters", default=[],
                   metavar="BUCKET_PREFIX",
                   help="Only process videos whose path contains this string. Repeatable.")
    p.add_argument("--max-tokens", type=int, default=700,
                   help="Max generation tokens per prompt")
    p.add_argument("--max-frames", type=int, default=5000,
                   help="Max video frames to load")
    p.add_argument("--fps", type=float, default=1.0,
                   help="Frames per second for video loading")
    p.add_argument("--verbose", action="store_true",
                   help="Print prompts and responses")
    p.add_argument("--resume-state", action="store_true",
                   help="Skip already-completed groups (read from --state-file)")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Limit to first N videos (for local testing)")
    # Model-specific flags (passed through to model factory)
    p.add_argument("--qwen3-thinking", action="store_true",
                   help="Use Qwen3-VL thinking variant (for qwen3_full)")
    p.add_argument("--max-gpu-mem", type=float, default=None,
                   help="Override per-GPU memory limit (GB)")
    p.add_argument("--poly-json-dir", default=None,
                   help="Directory with final_*_raw_accuracy.json for polynomial range checks "
                        "(optional; if omitted, range warnings are skipped).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# State manager
# ─────────────────────────────────────────────────────────────────────────────

class MultiQStateManager:
    """Tracks completed question groups, keyed by bucket:videoId_vVariant."""

    def __init__(self, state_file: str, resume: bool) -> None:
        self._path = state_file
        self._state: Dict[str, Dict[str, bool]] = {}
        if resume and os.path.exists(state_file):
            with open(state_file) as f:
                self._state = json.load(f)
            print(f"[state] Loaded {sum(len(v) for v in self._state.values())} completed groups")

    @staticmethod
    def _key(video_id: int, variant: int, bucket: str) -> str:
        return f"{bucket}:{video_id}_v{variant}"

    def is_done(self, video_id: int, variant: int, bucket: str, group_id: str) -> bool:
        return self._state.get(self._key(video_id, variant, bucket), {}).get(group_id, False)

    def mark_done(self, video_id: int, variant: int, bucket: str, group_id: str) -> None:
        k = self._key(video_id, variant, bucket)
        self._state.setdefault(k, {})[group_id] = True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

_BASE_COLS = [
    "video_id", "variant", "bucket", "group_id", "num_questions",
    "video_entropy", "eval_mode", "input_mode", "model",
    "q1_id", "q1_correct", "q1_predicted", "q1_is_correct", "q1_is_dont_know",
    "q2_id", "q2_correct", "q2_predicted", "q2_is_correct", "q2_is_dont_know",
]
_Q3_COLS = [
    "q3_id", "q3_correct", "q3_predicted", "q3_is_correct", "q3_is_dont_know",
]
_TAIL_COLS = ["response_full", "response_token_count", "output_was_truncated", "gflops", "gflops_source", "timestamp"]


def _csv_header(n: int) -> List[str]:
    return _BASE_COLS + (_Q3_COLS if n == 3 else []) + _TAIL_COLS


class CsvLogger:
    def __init__(self, path: str, n: int) -> None:
        self._path = path
        self._header = _csv_header(n)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self._header)

    def write(self, row: Dict[str, Any]) -> None:
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self._header])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _seq_text(candidate: Optional[BinaryCandidateEntry]) -> str:
    """Format a candidate sequence as comma-separated text."""
    if candidate is None:
        return "(no sequence)"
    if candidate.sequence:
        return ", ".join(str(t) for t in candidate.sequence)
    seqs = (candidate.metadata or {}).get("sequences", {})
    if isinstance(seqs, dict):
        for v in seqs.values():
            if isinstance(v, list) and v:
                return ", ".join(str(t) for t in v)
    return "(no sequence)"


def build_sequence_prompt(questions: List[QuestionEntry], n: int) -> str:
    """Multi-question prompt for text/sequence mode."""
    parts = [
        "In the main sequence, tokens appeared in a specific order.",
        f"You will be asked {n} separate questions. Each provides a different candidate sequence.",
        "Your task: determine if each candidate appears as a contiguous subsequence within the",
        "main sequence (same order, consecutive, no gaps).",
        "",
    ]
    for i, q in enumerate(questions, 1):
        parts.append(f"Question {i} candidate: {_seq_text(q.candidate)}")
    fmt = " ".join(f"{i}:{{answer}}" for i in range(1, n + 1))
    example = " ".join(f"{i}:{{{'0' if i % 2 == 1 else '1'}}}" for i in range(1, n + 1))
    parts += [
        "",
        "Answer each question: {0}=yes (it appears), {1}=no (does not appear), {2}=unsure/abstain.",
        f"You MUST answer ALL {n} questions in this EXACT format on one line:",
        fmt,
        f"Example: {example}",
        "You may write {2} to abstain from any question you are unsure about.",
    ]
    return "\n".join(parts)


def build_video_prompt(n: int, eval_mode: str) -> str:
    """Multi-question prompt for video mode."""
    if eval_mode == "sequential":
        direction = (
            "Did this option video show events that exactly appeared in the main video, "
            "down to the order they appeared and which conveyer belt each letter was on?"
        )
    else:
        direction = (
            "Did this option video show events that exactly appeared SOMEWHERE in the main video, "
            "down to the order they appeared and which conveyer belt each letter was on?"
        )
    parts = [
        "In the main video, letters moved down three conveyer belts. "
        "A letter that appeared on a specific conveyer belt ALWAYS stayed on the same conveyer belt "
        "and only moved down, not left or right. "
        "The letters appeared sequentially, but multiple may have been on screen at the same time. "
        "The letters continued moving down the conveyer belt until they left the screen.",
        "",
        f"You will be asked {n} separate questions about {n} different option videos.",
        direction,
        "",
    ]
    for i in range(1, n + 1):
        parts.append(f"Question {i}: Examine option video {i} shown above.")
    fmt = " ".join(f"{i}:{{answer}}" for i in range(1, n + 1))
    example = " ".join(f"{i}:{{{'0' if i % 2 == 1 else '1'}}}" for i in range(1, n + 1))
    parts += [
        "",
        "Answer each question: {0}=yes, {1}=no, {2}=unsure/abstain.",
        f"You MUST answer ALL {n} questions in this EXACT format on one line:",
        fmt,
        f"Example: {example}",
        "You may write {2} to abstain.",
    ]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def _norm_ans(raw: str) -> str:
    c = raw.strip().lower().strip(".,!? ")
    if c in ("0", "yes"):
        return "0"
    if c in ("1", "no"):
        return "1"
    return "2"


def extract_answers(response: str, n: int) -> Dict[int, str]:
    """Extract N numbered answers from model response.
    Primary: i:{X} pattern. Fallback: Nth {X} in the text.
    """
    answers: Dict[int, str] = {}
    for i in range(1, n + 1):
        m = re.search(rf'(?<!\d){i}\s*:\s*\{{([^}}]*)\}}', response)
        if m:
            answers[i] = _norm_ans(m.group(1))
            continue
        brackets = re.findall(r'\{([^}]+)\}', response)
        answers[i] = _norm_ans(brackets[i - 1]) if i - 1 < len(brackets) else "2"
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Question grouping
# ─────────────────────────────────────────────────────────────────────────────

def make_groups(questions: List[QuestionEntry], n: int, seed: int) -> List[List[QuestionEntry]]:
    """Randomly partition questions into non-overlapping groups of size n."""
    rng = random.Random(seed)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    return [shuffled[i:i + n] for i in range(0, len(shuffled) - n + 1, n)]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bucket(video_data: VideoEntry) -> str:
    """Extract bucket label from video path (e.g. UNIFORM_EVAL_L256_ELOW)."""
    path = str(video_data.video_path or "")
    for part in path.split("/"):
        if "UNIFORM_EVAL" in part or any(
            x in part for x in ["L008", "L016", "L032", "L064", "L128", "L256", "L512", "L1024", "L2048"]
        ):
            return part
    return "UNKNOWN"


def _correct(q: QuestionEntry) -> str:
    raw = (q.binary_answer or "").strip().lower()
    return "0" if raw == "yes" else ("1" if raw == "no" else "2")


def _filter_questions(video_data: VideoEntry, eval_mode: str) -> List[QuestionEntry]:
    """Return questions matching the eval_mode variant."""
    out = []
    for q in sorted(video_data.questions, key=lambda x: float(x.question_time)):
        if not q.is_native_binary:
            continue
        qv = (q.metadata or {}).get("question_variant", "").lower()
        if eval_mode == "sequential" and qv == "spatial":
            continue
        if eval_mode == "spatial" and qv == "sequential":
            continue
        out.append(q)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing — SEQUENCE mode
# ─────────────────────────────────────────────────────────────────────────────

def _process_sequence(
    video_data: VideoEntry,
    model: Any,
    args: argparse.Namespace,
    state_mgr: MultiQStateManager,
    csv_log: CsvLogger,
) -> None:
    video_id = video_data.video_index
    variant = int((video_data.metadata or {}).get("variant", 0))
    bucket = _extract_bucket(video_data)

    valid_qs = _filter_questions(video_data, args.eval_mode)
    if len(valid_qs) < args.num_questions:
        if args.verbose:
            print(f"  skip: only {len(valid_qs)} questions, need {args.num_questions}")
        return

    # Build SequenceProcessor from video metadata
    raw_seqs = (video_data.metadata or {}).get("sequences_used", {})
    if not raw_seqs:
        print(f"  skip video {video_id}: missing sequences_used metadata")
        return

    formatter = CommaSeparatedSequenceFormatter()
    seq_proc = SequenceProcessor(
        sequences_used=raw_seqs,
        formatter=formatter,
        print_chunks=args.verbose,
    )

    # Stream main sequence once, save state
    model.clear_context()
    base_time, _ = seq_proc.stream_full_sequences(model, base_time=0.0)
    base_state = model.save_state()

    # Group questions, process each group
    seed = video_id * 10_000 + variant
    groups = make_groups(valid_qs, args.num_questions, seed)
    entropy = (video_data.metadata or {}).get("entropy_overall", "")

    for g_idx, group in enumerate(groups):
        gid = f"g{g_idx}"
        if state_mgr.is_done(video_id, variant, bucket, gid):
            if args.verbose:
                print(f"    {gid}: already done")
            continue

        # Restore context to just after main sequence
        model.load_state(base_state)
        t = base_time

        # Add each candidate sequence as context
        for i, q in enumerate(group, 1):
            txt = f"\nCandidate sequence {i}: {_seq_text(q.candidate)}\n"
            model.add_text(txt, current_video_time=t)
            t += 1.0

        # Ask multi-question prompt
        prompt = build_sequence_prompt(group, args.num_questions)
        if args.verbose:
            print(f"  Group {gid}, prompt:\n{prompt}")

        response = model.ask_question(prompt, current_video_time=t, max_tokens=args.max_tokens)
        if args.verbose:
            print(f"  Response: {response}")
        token_count = len(response.split())
        gflops, gflops_src = _pick_gflops(token_count, args._poly_ranges, bucket)

        answers = extract_answers(response, args.num_questions)
        _log_and_save(
            row_base={
                "video_id": video_id, "variant": variant, "bucket": bucket,
                "group_id": gid, "num_questions": args.num_questions,
                "video_entropy": entropy, "eval_mode": args.eval_mode,
                "input_mode": "sequence", "model": args.model,
            },
            group=group,
            answers=answers,
            response=response,
            gflops=gflops,
            gflops_source=gflops_src,
            csv_log=csv_log,
            state_mgr=state_mgr,
            video_id=video_id, variant=variant, bucket=bucket, gid=gid,
            verbose=args.verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processing — VIDEO mode
# ─────────────────────────────────────────────────────────────────────────────

def _process_video(
    video_data: VideoEntry,
    model: Any,
    args: argparse.Namespace,
    state_mgr: MultiQStateManager,
    csv_log: CsvLogger,
) -> None:
    from processors.video_processor import VideoProcessor

    video_id = video_data.video_index
    variant = int((video_data.metadata or {}).get("variant", 0))
    bucket = _extract_bucket(video_data)

    # Filter questions that have valid clip paths
    all_qs = _filter_questions(video_data, args.eval_mode)
    valid_qs = [
        q for q in all_qs
        if q.candidate and q.candidate.clip_path and os.path.exists(str(q.candidate.clip_path))
    ]
    if len(valid_qs) < args.num_questions:
        if args.verbose:
            print(f"  skip: only {len(valid_qs)} questions with valid clips, need {args.num_questions}")
        return

    main_path = str(video_data.video_path or "")
    if not os.path.exists(main_path):
        print(f"  skip: main video not found: {main_path}")
        return

    # Load main video frames once
    vp = VideoProcessor(args.model, fps=args.fps, max_frames=args.max_frames)
    vp.load_main_video(main_path)
    main_frames = vp.main_video_frames
    video_duration = float(getattr(vp, "video_duration", 0) or 10.0)
    # Compute actual fps from loaded frames so option clips use the same value,
    # preventing FPS-inconsistency errors on models that don't resample to args.fps.
    n_main = len(main_frames) if hasattr(main_frames, '__len__') else getattr(main_frames, 'shape', [0])[0]
    effective_fps = n_main / video_duration if video_duration > 0 and n_main > 0 else args.fps

    seed = video_id * 10_000 + variant
    groups = make_groups(valid_qs, args.num_questions, seed)
    entropy = (video_data.metadata or {}).get("entropy_overall", "")

    for g_idx, group in enumerate(groups):
        gid = f"g{g_idx}"
        if state_mgr.is_done(video_id, variant, bucket, gid):
            if args.verbose:
                print(f"    {gid}: already done")
            continue

        # Fresh context: main video + N option clips
        model.clear_context()
        model.add_video(main_frames, 0.0, video_duration, video_id=video_id)

        # Place each option clip sequentially after the main video so global
        # timestamps are strictly increasing (required by models without video_id isolation).
        # Use effective_fps (derived from main video) so all segments report the same fps.
        next_t = video_duration + 1.0
        for i, q in enumerate(group, 1):
            clips = vp.load_option_videos([str(q.candidate.clip_path)])
            if clips:
                n_opt = (len(clips[0]) if hasattr(clips[0], '__len__')
                         else getattr(clips[0], 'shape', [0])[0])
                c_dur = n_opt / effective_fps if n_opt > 0 else 5.0
                model.add_video(clips[0], next_t, next_t + c_dur, video_id=video_id * 100 + i)
                next_t += c_dur + 1.0
            else:
                print(f"  Warning: could not load option clip {i}: {q.candidate.clip_path}")

        prompt = build_video_prompt(args.num_questions, args.eval_mode)
        qt = next_t  # ask after all option clips
        if args.verbose:
            print(f"  Group {gid}, prompt:\n{prompt}")

        response = model.ask_question(
            prompt, current_video_time=qt,
            max_tokens=args.max_tokens, max_frames_in_video=args.max_frames)
        if args.verbose:
            print(f"  Response: {response}")
        token_count = len(response.split())
        gflops, gflops_src = _pick_gflops(token_count, args._poly_ranges, bucket)

        answers = extract_answers(response, args.num_questions)
        _log_and_save(
            row_base={
                "video_id": video_id, "variant": variant, "bucket": bucket,
                "group_id": gid, "num_questions": args.num_questions,
                "video_entropy": entropy, "eval_mode": args.eval_mode,
                "input_mode": "video", "model": args.model,
            },
            group=group,
            answers=answers,
            response=response,
            gflops=gflops,
            gflops_source=gflops_src,
            csv_log=csv_log,
            state_mgr=state_mgr,
            video_id=video_id, variant=variant, bucket=bucket, gid=gid,
            verbose=args.verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared logging helper
# ─────────────────────────────────────────────────────────────────────────────

def _log_and_save(
    row_base: Dict[str, Any],
    group: List[QuestionEntry],
    answers: Dict[int, str],
    response: str,
    csv_log: CsvLogger,
    state_mgr: MultiQStateManager,
    video_id: int,
    variant: int,
    bucket: str,
    gid: str,
    verbose: bool,
    gflops: Optional[float] = None,
    gflops_source: str = "",
) -> None:
    row = dict(row_base)
    row["response_full"] = response
    row["response_token_count"] = len(response.split())
    row["output_was_truncated"] = False
    row["gflops"] = f"{gflops:.1f}" if gflops is not None else ""
    row["gflops_source"] = gflops_source
    row["timestamp"] = datetime.now().isoformat()

    for i, q in enumerate(group, 1):
        correct = _correct(q)
        pred = answers.get(i, "2")
        qid = q.question_id or f"video{video_id}_q{i}"
        row[f"q{i}_id"] = qid
        row[f"q{i}_correct"] = correct
        row[f"q{i}_predicted"] = pred
        row[f"q{i}_is_correct"] = pred == correct
        row[f"q{i}_is_dont_know"] = pred == "2"

        if verbose:
            marker = "✅" if pred == correct else ("🤷" if pred == "2" else "❌")
            print(f"    Q{i}: pred={pred} correct={correct} {marker}")

    csv_log.write(row)
    state_mgr.mark_done(video_id, variant, bucket, gid)
    state_mgr.save()


# ─────────────────────────────────────────────────────────────────────────────
# Model instantiation (mirrors main.py factory)
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(args: argparse.Namespace) -> Any:
    from main import load_model_class
    from utils.memory_utils import calculate_max_gpu_mem

    model_class, model_name = load_model_class(args.model)
    print(f"Loading model: {model_name}")

    gpu_mem_aware = {
        "qwen_full", "mimo-vl", "qwen3_full", "glm45v", "timechat", "qwen3_omni",
        "internvl-3-5", "internvl-3-5-thinking",
        "internvl-3-5-38b", "internvl-3-5-38b-thinking",
        "internvl-3-5-30b-a3b", "internvl-3-5-30b-a3b-thinking",
        "longvila", "minicpm-4-5",
    }
    kw: Dict[str, Any] = {}
    if args.model in gpu_mem_aware:
        kw["max_gpu_mem"] = calculate_max_gpu_mem(args.model, override=args.max_gpu_mem)

    internvl_map = {
        "internvl-3-5":                   "OpenGVLab/InternVL3_5-8B",
        "internvl-3-5-thinking":          "OpenGVLab/InternVL3_5-8B",
        "internvl-3-5-38b":               "OpenGVLab/InternVL3_5-38B",
        "internvl-3-5-38b-thinking":      "OpenGVLab/InternVL3_5-38B",
        "internvl-3-5-30b-a3b":           "OpenGVLab/InternVL3_5-30B-A3B",
        "internvl-3-5-30b-a3b-thinking":  "OpenGVLab/InternVL3_5-30B-A3B",
    }

    if args.model == "glm45v":
        from models.glm45v import GLM45V
        return GLM45V("zai-org/GLM-4.5V", **kw)
    if args.model == "qwen3_full":
        return model_class(thinking=args.qwen3_thinking, **kw)
    if args.model == "qwen3_omni":
        return model_class("Qwen/Qwen3-Omni-30B-A3B-Instruct", **kw)
    if args.model in internvl_map:
        kw["generation_max_tokens"] = args.max_tokens
        return model_class(internvl_map[args.model], **kw)
    if args.model == "minicpm-4-5":
        return model_class("openbmb/MiniCPM-V-4_5", **kw)
    if args.model == "mimo-vl":
        return model_class("XiaomiMiMo/MiMo-VL-7B-RL", **kw)
    if args.model == "minicpm":
        return model_class("openbmb/MiniCPM-o-2_6", **kw)
    if args.model == "longvila":
        return model_class("Efficient-Large-Model/LongVILA-R1-7B", **kw)
    # Default: qwen_full (Qwen2.5-VL)
    return model_class("Qwen/Qwen2.5-VL-7B-Instruct", **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Safety check: never overwrite buckets-* files
    for path in [args.question_log_csv, args.state_file]:
        name = os.path.basename(path)
        if name.startswith("buckets-"):
            print(f"ERROR: refusing to write to {path} (looks like a baseline buckets-* file)")
            sys.exit(1)

    # Load polynomial ranges for out-of-range warnings (always) and profiling (if --measure-flops)
    poly_dir = Path(args.poly_json_dir) if args.poly_json_dir else _PRECOMP_DEFAULT
    args._poly_ranges = _load_poly_ranges(poly_dir, args.model, args.input_mode) if poly_dir else {}

    print("=" * 60)
    print(f"Multi-question eval: {args.num_questions}Q per prompt")
    print(f"Model:      {args.model}")
    print(f"Mode:       {args.input_mode}/{args.eval_mode}")
    print(f"CSV output: {args.question_log_csv}")
    print(f"State:      {args.state_file}")
    print(f"FLOPs poly: {len(args._poly_ranges)} ranges loaded (warns if response outside range)")
    print("=" * 60)

    # Load dataset
    require_video = (args.input_mode == "video")
    videos = load_patternvideos_manifest(
        args.json_path,
        require_video_assets=require_video,
        asset_root=args.asset_root,
    )

    # Apply bucket filters
    if args.bucket_filters:
        def _matches(v: VideoEntry) -> bool:
            path = str(v.video_path or "")
            return any(f in path for f in args.bucket_filters)
        videos = [v for v in videos if _matches(v)]
        print(f"After bucket filter ({args.bucket_filters}): {len(videos)} videos")

    if args.max_videos:
        videos = videos[: args.max_videos]
        print(f"Limited to {args.max_videos} videos (testing mode)")

    print(f"Processing {len(videos)} videos\n")

    state_mgr = MultiQStateManager(args.state_file, args.resume_state)
    csv_log = CsvLogger(args.question_log_csv, args.num_questions)
    model = _load_model(args)

    n_done = 0
    for idx, video_data in enumerate(videos, 1):
        bucket = _extract_bucket(video_data)
        vid = video_data.video_index
        print(f"[{idx}/{len(videos)}] video={vid} bucket={bucket}")

        try:
            if args.input_mode == "sequence":
                _process_sequence(video_data, model, args, state_mgr, csv_log)
            else:
                _process_video(video_data, model, args, state_mgr, csv_log)
        except Exception as exc:
            import traceback
            print(f"  ERROR video {vid}: {exc}")
            traceback.print_exc()

        n_done += 1
        if n_done % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Done. {n_done} videos processed.")
    print(f"CSV:   {args.question_log_csv}")
    print(f"State: {args.state_file}")


if __name__ == "__main__":
    main()
