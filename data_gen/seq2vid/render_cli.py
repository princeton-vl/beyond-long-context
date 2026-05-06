from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .buckets import BucketSpec, filter_buckets, load_bucket_config
from .render import run_render
from .template_utils import TemplateOverrideManager


def _parse_sequence_sources(values: Optional[List[str]]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not values:
        return mapping
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --sequence-source entry {raw!r}; expected seq_id=path")
        key, path = raw.split("=", 1)
        key = key.strip()
        value = path.strip()
        if not key or not value:
            raise ValueError(f"Invalid --sequence-source entry {raw!r}; missing key or path")
        if key in mapping:
            raise ValueError(f"Duplicate --sequence-source entry for {key!r}")
        mapping[key] = Path(value)
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Render videos from pre-generated sequences.")
    ap.add_argument("--template", default="examples/template_conveyor.yaml", help="Path to vidgeom template YAML")
    ap.add_argument("--sequences-file", default=None, help="Path to sequences.json from generation step (classic mode)")
    ap.add_argument("--sequences-root", default="runs_seq", help="Root directory containing per-bucket sequences (bucket mode)")
    ap.add_argument("--out-dir", default="runs_render", help="Directory for outputs (videos + questions.json)")
    ap.add_argument("--num-questions", type=int, default=None, help="Questions per video (overrides bucket defaults)")
    ap.add_argument("--clip-options", action="store_true", help="Render option clips (true from main video, false rendered) instead of sequences-only")
    ap.add_argument("--questions-only", action="store_true", help="Skip video rendering; only generate questions.json from sequences")
    question_group = ap.add_mutually_exclusive_group()
    question_group.add_argument(
        "--questions-at-end",
        dest="questions_at_end",
        action="store_true",
        help="Ask all questions after the video instead of interspersed during playback",
    )
    question_group.add_argument(
        "--questions-during",
        dest="questions_at_end",
        action="store_false",
        help="Interleave questions while the video is still playing",
    )
    ap.set_defaults(questions_at_end=True)
    ap.add_argument("--ffmpeg-crf", type=int, default=23, help="CRF value for libx264 (higher = smaller files, default 23)")
    ap.add_argument("--ffmpeg-preset", type=str, default="veryfast", help="ffmpeg preset for libx264 (e.g., veryfast, medium, slow)")
    ap.add_argument("--ffmpeg-codec", type=str, default="libx264", help="ffmpeg video codec (libx264, libx265, etc.)")
    ap.add_argument("--target-seq-lens", type=str, default=None, help="Comma-separated lengths per sequence (single value applies to all). Longer sequences are trimmed; shorter sequences raise an error.")
    ap.add_argument("--num-videos", type=int, default=None, help="Optional cap on number of videos to render (consumes sequences in order)")
    ap.add_argument("--question-min-len", type=int, default=3, help="Minimum n-gram length for questions (must fit stats and prefix)")
    ap.add_argument("--render-workers", type=int, default=1, help="Parallel render workers (per-video jobs)")
    ap.add_argument("--assignment-seed", type=int, default=None, help="Seed to shuffle sequences before assigning to videos (controls deterministic random order)")
    ap.add_argument("--fps", type=int, default=None, help="Override output FPS (template render.fps)")
    ap.add_argument("--log-progress", action="store_true", help="Enable verbose progress logging")
    ap.add_argument("--uniform-uncertain", action="store_true", help="Treat the Uncertain option as correct with uniform probability across all answers")
    ap.add_argument("--hard-questions", action="store_true", help="Use structured distractors instead of fully random unseen ones")
    ap.add_argument(
        "--question-mode",
        choices=["exists", "continuation"],
        default="exists",
        help="Question style: 'exists' reuses the legacy presence check, 'continuation' asks which chunk follows a prefix",
    )
    ap.add_argument(
        "--hide-question-text",
        action="store_true",
        help="Remove question text overlays from rendered videos",
    )
    ap.add_argument(
        "--sequence-source",
        action="append",
        default=None,
        help="Optional seq_id=path overrides to pull each template sequence from its own sequences.json",
    )
    ap.add_argument(
        "--spatial-question-fraction",
        type=float,
        default=0.0,
        help="Fraction (0-1) of questions to render in spatial mode",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip media validation/manifest generation",
    )
    ap.add_argument(
        "--debug-frame-metadata",
        action="store_true",
        help="Write per-frame token/letter debug files next to each video",
    )

    # Bucket mode controls
    ap.add_argument("--bucket-config", type=str, default=None, help="Path to bucket YAML to enable bucket-based rendering")
    ap.add_argument("--include-buckets", nargs="*", default=None, help="Glob(s) of bucket IDs to include in bucket mode")
    ap.add_argument("--exclude-buckets", nargs="*", default=None, help="Glob(s) of bucket IDs to exclude in bucket mode")
    ap.add_argument("--bucket-overwrite", action="store_true", help="Re-render buckets even if outputs already exist")

    args = ap.parse_args()

    sequence_sources = _parse_sequence_sources(args.sequence_source)

    if args.bucket_config:
        run_bucket_render_mode(args, sequence_sources)
        return

    if not args.sequences_file:
        raise ValueError("--sequences-file is required when not using --bucket-config")

    target_seq_lens = None
    if args.target_seq_lens:
        lens = [int(x) for x in args.target_seq_lens.split(",") if x.strip()]
        target_seq_lens = lens if lens else None

    run_render(
        template_path=Path(args.template),
        sequences_file=Path(args.sequences_file),
        out_dir=Path(args.out_dir),
        num_questions=args.num_questions or 3,
        clip_options=args.clip_options,
        questions_at_end=args.questions_at_end,
        ffmpeg_crf=args.ffmpeg_crf,
        ffmpeg_preset=args.ffmpeg_preset,
        ffmpeg_codec=args.ffmpeg_codec,
        target_seq_lens=target_seq_lens,
        max_videos=args.num_videos,
        question_min_len=args.question_min_len,
        render_workers=args.render_workers,
        assignment_seed=args.assignment_seed,
        fps_override=args.fps,
        questions_only=args.questions_only,
        log_progress=args.log_progress,
        uniform_uncertain=args.uniform_uncertain,
        question_mode=args.question_mode,
        hide_question_text=args.hide_question_text,
        sequence_sources=sequence_sources or None,
        spatial_question_fraction=args.spatial_question_fraction,
        hard_questions=args.hard_questions,
        validate_outputs=not args.skip_validation,
        capture_frame_debug=args.debug_frame_metadata,
    )


def run_bucket_render_mode(args: argparse.Namespace, sequence_sources_cli: Dict[str, Path]) -> None:
    cfg = load_bucket_config(Path(args.bucket_config))
    buckets = filter_buckets(cfg.list_buckets(), args.include_buckets, args.exclude_buckets)
    if not buckets:
        raise ValueError("No buckets matched the provided include/exclude filters")
    sequences_root = Path(args.sequences_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    override_mgr = TemplateOverrideManager()
    manifest_entries: List[dict] = []

    for spec in buckets:
        seq_dir = sequences_root / spec.bucket_id
        seq_file = seq_dir / "sequences.json"
        meta_status = "unknown"
        meta_reason: Optional[str] = None
        meta_path = seq_dir / "meta.json"
        if meta_path.exists():
            meta_data = json.loads(meta_path.read_text())
            meta_status = meta_data.get("status", "unknown")
            meta_reason = meta_data.get("failure_reason")
        if meta_status != "completed":
            print(
                f"[skip] bucket {spec.bucket_id}: status={meta_status} reason={meta_reason}",
                flush=True,
            )
            manifest_entries.append(
                {
                    "bucket_id": spec.bucket_id,
                    "sequences_file": str(seq_file),
                    "render_dir": "",
                    "status": meta_status,
                    "skip_reason": meta_reason or "incomplete",
                }
            )
            continue
        bucket_out = out_root / spec.bucket_id
        questions_path = bucket_out / "questions.json"
        if questions_path.exists() and not args.bucket_overwrite:
            print(f"[skip] bucket {spec.bucket_id}: outputs already exist (use --bucket-overwrite to rerender)")
            manifest_entries.append({
                "bucket_id": spec.bucket_id,
                "sequences_file": str(seq_file),
                "render_dir": str(bucket_out),
                "status": "skipped_existing",
            })
            continue
        bucket_out.mkdir(parents=True, exist_ok=True)
        profile = cfg.get_step_profile(spec.step_profile)
        template_path = override_mgr.build(
            Path(args.template),
            step_range=profile.step_range,
            fps=args.fps if args.fps is not None else profile.fps,
            hide_question_text=args.hide_question_text,
        )
        seq_names = _template_sequence_names(template_path)
        detected_sources = _detect_sequence_sources(seq_dir, seq_names)
        cli_sources = sequence_sources_cli or {}
        seq_sources_to_use: Optional[Dict[str, Path]] = None
        if detected_sources:
            seq_sources_to_use = detected_sources
        elif cli_sources:
            seq_sources_to_use = cli_sources
        if not seq_file.exists() and not seq_sources_to_use:
            reason = f"missing sequences.json at {seq_file}"
            print(f"[skip] bucket {spec.bucket_id}: {reason}")
            manifest_entries.append(
                {
                    "bucket_id": spec.bucket_id,
                    "sequences_file": str(seq_file),
                    "render_dir": "",
                    "status": "missing_sequences",
                    "skip_reason": reason,
                }
            )
            continue
        num_questions = args.num_questions if args.num_questions is not None else profile.num_questions
        max_videos = spec.target_videos
        if args.num_videos is not None:
            max_videos = min(max_videos, args.num_videos)
        run_render(
            template_path=template_path,
            sequences_file=seq_file if seq_file.exists() else None,
            out_dir=bucket_out,
            num_questions=num_questions,
            clip_options=args.clip_options,
            questions_at_end=args.questions_at_end,
            ffmpeg_crf=args.ffmpeg_crf,
            ffmpeg_preset=args.ffmpeg_preset,
            ffmpeg_codec=args.ffmpeg_codec,
            target_seq_lens=None,
            max_videos=max_videos,
            question_min_len=args.question_min_len,
            render_workers=args.render_workers,
            assignment_seed=args.assignment_seed,
            fps_override=None if args.fps is None else args.fps,
            questions_only=args.questions_only,
            log_progress=args.log_progress,
            uniform_uncertain=args.uniform_uncertain,
            question_mode=args.question_mode,
            hide_question_text=args.hide_question_text,
            sequence_sources=seq_sources_to_use,
            spatial_question_fraction=args.spatial_question_fraction,
            hard_questions=args.hard_questions,
            validate_outputs=not args.skip_validation,
            capture_frame_debug=args.debug_frame_metadata,
        )
        manifest_entries.append(
            {
                "bucket_id": spec.bucket_id,
                "sequences_file": str(seq_file),
                "render_dir": str(bucket_out),
                "step_profile": spec.step_profile,
                "step_range": list(profile.step_range),
                "fps": profile.fps if args.fps is None else args.fps,
                "num_questions": num_questions,
                "target_videos": spec.target_videos,
                "status": "completed",
            }
        )

    override_mgr.cleanup()
    manifest_path = out_root / "bucket_render_manifest.json"
    manifest_path.write_text(json.dumps({"buckets": manifest_entries}, indent=2))


def _template_sequence_names(path: Path) -> List[str]:
    data = yaml.safe_load(path.read_text()) or {}
    seqs = data.get("sequences", []) or []
    return [str(x) for x in seqs]


def _detect_sequence_sources(seq_dir: Path, seq_names: List[str]) -> Optional[Dict[str, Path]]:
    if not seq_names:
        return None
    mapping: Dict[str, Path] = {}
    found_any = False
    missing: List[str] = []
    for name in seq_names:
        candidate = seq_dir / name / "sequences.json"
        if candidate.exists():
            mapping[name] = candidate
            found_any = True
        else:
            missing.append(name)
    if not found_any:
        return None
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Per-sequence JSONs detected in {seq_dir}, but missing entries for: {missing_str}"
        )
    return mapping


if __name__ == "__main__":
    main()
