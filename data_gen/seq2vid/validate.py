from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _ffprobe_frame_count(path: Path) -> int:
    cmd = [
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
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return 0
    data = result.stdout.strip()
    return int(data) if data.isdigit() else 0


def _resolve_path(reference: str, bucket_root: Path) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path
    candidates = [bucket_root / path, bucket_root.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (bucket_root / path).resolve()


def _frame_items(frame: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    return frame.get("items", []) or []


def _validate_frame_sequence(
    frames: Sequence[Dict[str, Any]],
    target_tokens: Sequence[str],
    max_frames: Optional[int],
    max_items_per_frame: int,
) -> List[str]:
    errors: List[str] = []
    if not frames:
        errors.append("clip has no frame metadata")
        return errors
    if max_frames is not None and len(frames) > max_frames:
        errors.append(f"clip has {len(frames)} frames (max {max_frames})")
    tokens = [str(t) for t in target_tokens]
    for idx, frame in enumerate(frames):
        items = _frame_items(frame)
        if len(items) > max_items_per_frame:
            errors.append(
                f"frame {idx} has {len(items)} items (max {max_items_per_frame})"
            )
    if not tokens:
        return errors
    first_token = tokens[0]
    last_token = tokens[-1]
    if first_token not in {str(it.get("token")) for it in _frame_items(frames[0])}:
        errors.append("first frame does not show the first target token")
    if last_token not in {str(it.get("token")) for it in _frame_items(frames[-1])}:
        errors.append("last frame does not show the final target token")
    target_idx = 0
    for frame in frames:
        items = {str(it.get("token")) for it in _frame_items(frame)}
        while target_idx < len(tokens) and tokens[target_idx] in items:
            target_idx += 1
    if target_idx < len(tokens):
        errors.append("clip is missing one or more target tokens")
    return errors


def _load_frames(clip_path: Path) -> Dict[str, Any]:
    frames_path = clip_path.with_suffix(".frames.json")
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    return json.loads(frames_path.read_text())


def _validate_clip(
    clip_path: Path,
    target_tokens: Sequence[str],
    max_frames: Optional[int],
    max_items: int,
) -> List[str]:
    errors: List[str] = []
    if not clip_path.exists():
        return [f"missing clip file {clip_path}"]
    try:
        payload = _load_frames(clip_path)
    except FileNotFoundError:
        return [f"missing frame metadata for {clip_path}"]
    frames = payload.get("frames", [])
    frame_errors = _validate_frame_sequence(frames, target_tokens, max_frames, max_items)
    for err in frame_errors:
        errors.append(f"{clip_path.name}: {err}")
    ffprobe_count = _ffprobe_frame_count(clip_path)
    if ffprobe_count and ffprobe_count != len(frames):
        errors.append(
            f"ffprobe reports {ffprobe_count} frames but metadata lists {len(frames)}"
        )
    return errors


def _primary_tokens(sequences: Dict[str, Sequence[str]]) -> Sequence[str]:
    if "S_tokens" in sequences:
        return sequences["S_tokens"]
    if sequences:
        first_key = next(iter(sequences))
        return sequences[first_key]
    return []


def _validate_question(
    question: Dict[str, Any],
    bucket_root: Path,
    max_frames: Optional[int],
    max_items: int,
) -> List[str]:
    errors: List[str] = []
    candidate = question.get("candidate") or {}
    clip_ref = candidate.get("clip_path")
    if not clip_ref:
        errors.append("candidate missing clip_path")
        return errors
    clip_path = _resolve_path(clip_ref, bucket_root)
    seqs = candidate.get("sequences") or {}
    target_tokens = _primary_tokens(seqs)
    errors.extend(_validate_clip(clip_path, target_tokens, max_frames, max_items))
    if question.get("question_mode") == "continuation":
        prefix_path = question.get("prefix_clip_path")
        if not prefix_path:
            errors.append("continuation question missing prefix clip")
        else:
            prefix_full = _resolve_path(prefix_path, bucket_root)
            prefix_tokens = _primary_tokens(question.get("prefix") or {})
            errors.extend(
                _validate_clip(prefix_full, prefix_tokens, max_frames, max_items)
            )
    return errors


def _find_extra_option_clips(clips_dir: Path) -> List[str]:
    errors: List[str] = []
    if not clips_dir.exists():
        return [f"missing clips directory {clips_dir}"]
    for clip_path in clips_dir.glob("*.mp4"):
        if "_opt" in clip_path.name:
            errors.append(f"unexpected option clip present: {clip_path.name}")
    return errors


def validate_bucket(
    bucket_dir: Path,
    max_frames: Optional[int] = 60,
    max_items: int = 8,
) -> List[str]:
    bucket_dir = bucket_dir.resolve()
    questions_path = bucket_dir / "questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(questions_path)
    payload = json.loads(questions_path.read_text())
    errors: List[str] = []
    for video in payload.get("videos", []):
        for question in video.get("questions", []):
            errors.extend(
                _validate_question(question, bucket_dir, max_frames, max_items)
            )
    errors.extend(_find_extra_option_clips(bucket_dir / "clips"))
    return errors


def _main() -> None:
    ap = argparse.ArgumentParser(description="Validate rendered bucket outputs")
    ap.add_argument("bucket", type=Path, help="Path to bucket directory")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--max-items", type=int, default=8)
    args = ap.parse_args()
    errors = validate_bucket(args.bucket, args.max_frames, args.max_items)
    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)
    print("Validation passed with no errors.")


if __name__ == "__main__":
    _main()
