from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("[info] running:", " ".join(cmd))
    subprocess.check_call(cmd)


def _extend_patterns(cmd: List[str], flag: str, patterns: list[str] | None) -> None:
    if patterns:
        cmd.append(flag)
        cmd.extend(patterns)


def _matches_bucket(bucket_id: str, includes: list[str] | None, excludes: list[str] | None) -> bool:
    if includes:
        if not any(fnmatch(bucket_id, pat) for pat in includes):
            return False
    if excludes and any(fnmatch(bucket_id, pat) for pat in excludes):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Bucket orchestration utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Run bucketed sequence generation via seq2vid.gen_cli")
    p_gen.add_argument("--config", default="configs/buckets.yaml", help="Bucket config YAML path")
    p_gen.add_argument("--out-dir", default="runs_seq", help="Output root for per-bucket sequences")
    p_gen.add_argument("--include-buckets", nargs="*", default=None, help="Optional bucket ID globs to include")
    p_gen.add_argument("--exclude-buckets", nargs="*", default=None, help="Optional bucket ID globs to exclude")
    p_gen.add_argument("--bucket-batch-size", type=int, default=32, help="Sequences to request per generation batch")
    p_gen.add_argument("--bucket-seed", type=int, default=None, help="Seed for bucket sampling")
    p_gen.add_argument("--bucket-overwrite", action="store_true", help="Overwrite existing bucket outputs")
    p_gen.add_argument("--bucket-write-combined", action="store_true", help="Write combined sequences JSON")
    p_gen.add_argument("--bucket-max-no-progress", type=int, default=None, help="Override bucket no-progress retry limit")
    p_gen.add_argument("--gen-workers", type=int, default=None, help="Override --gen-workers for seq2vid.gen_cli")
    p_gen.add_argument("--discover-len-mult", type=float, default=None, help="Override discovery length multiplier")
    p_gen.add_argument("--max-attempts", type=int, default=None, help="Override max attempts")
    p_gen.add_argument("--top-k", type=int, default=None, help="Override top-k")
    p_gen.add_argument("--ngram-max", type=int, default=None, help="Override ngram-max")
    p_gen.add_argument("--max-rules", type=int, default=None, help="Override max rules")
    p_gen.add_argument("--log-progress", action="store_true", help="Enable verbose logging")

    p_render = sub.add_parser("render", help="Run bucketed rendering via seq2vid.render_cli")
    p_render.add_argument("--config", default="configs/buckets.yaml", help="Bucket config YAML path")
    p_render.add_argument("--template", default="examples/template_conveyor.yaml", help="Base template path")
    p_render.add_argument("--sequences-root", default="runs_seq", help="Root directory containing bucket sequences")
    p_render.add_argument("--out-dir", default="runs_render", help="Output root for renders")
    p_render.add_argument("--include-buckets", nargs="*", default=None, help="Optional bucket ID globs to include")
    p_render.add_argument("--exclude-buckets", nargs="*", default=None, help="Optional bucket ID globs to exclude")
    p_render.add_argument("--bucket-overwrite", action="store_true", help="Rerender even if outputs exist")
    p_render.add_argument("--num-questions", type=int, default=None, help="Override question count")
    p_render.add_argument("--num-videos", type=int, default=None, help="Override max videos per bucket")
    p_render.add_argument("--render-workers", type=int, default=None, help="Override render worker count")
    p_render.add_argument("--assignment-seed", type=int, default=None, help="Override assignment seed")
    p_render.add_argument("--fps", type=int, default=None, help="Override fps")
    p_render.add_argument("--ffmpeg-crf", type=int, default=None, help="Override CRF")
    p_render.add_argument("--ffmpeg-preset", type=str, default=None, help="Override ffmpeg preset")
    p_render.add_argument("--ffmpeg-codec", type=str, default=None, help="Override ffmpeg codec")
    p_render.add_argument("--question-min-len", type=int, default=None, help="Override question min length")
    p_render.add_argument("--questions-at-end", action="store_true", help="Ask questions after video")
    p_render.add_argument("--questions-only", action="store_true", help="Skip mp4 rendering")
    p_render.add_argument("--clip-options", action="store_true", help="Render clips for each option")
    p_render.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation manifest generation",
    )
    p_render.add_argument(
        "--debug-frame-metadata",
        action="store_true",
        help="Write per-frame token/letter debug files",
    )
    p_render.add_argument("--log-progress", action="store_true", help="Enable verbose logging")
    p_render.add_argument("--uniform-uncertain", action="store_true", help="Allow Uncertain to be correct with uniform probability")
    p_render.add_argument(
        "--question-mode",
        choices=["exists", "continuation"],
        default="exists",
        help="Question style for render_cli",
    )
    p_render.add_argument(
        "--hide-question-text",
        action="store_true",
        help="Disable HUD/question text overlays in rendered videos",
    )
    p_render.add_argument(
        "--sequence-source",
        action="append",
        default=None,
        help="Pass-through seq_id=path overrides for seq2vid.render_cli",
    )
    p_render.add_argument(
        "--spatial-question-fraction",
        type=float,
        default=0.0,
        help="Fraction (0-1) of questions to render in spatial mode",
    )

    p_manifest = sub.add_parser("manifest", help="Combine generation + render manifests")
    p_manifest.add_argument("--sequences-manifest", default="runs_seq/bucket_generation_manifest.json", help="Path to generation manifest")
    p_manifest.add_argument("--render-manifest", default="runs_render/bucket_render_manifest.json", help="Path to render manifest")
    p_manifest.add_argument("--output", default="bucket_dataset.json", help="Output JSON path")

    p_collect = sub.add_parser(
        "collect_questions",
        help="Merge per-bucket questions.json files into a single dataset",
    )
    p_collect.add_argument("--render-root", default="runs_render", help="Root directory containing per-bucket render outputs")
    p_collect.add_argument("--output", default="questions_dataset.json", help="Output JSON path for consolidated questions")
    p_collect.add_argument("--questions-file", default="questions.json", help="Filename to load within each bucket directory")
    p_collect.add_argument("--include-buckets", nargs="*", default=None, help="Optional bucket ID globs to include")
    p_collect.add_argument("--exclude-buckets", nargs="*", default=None, help="Optional bucket ID globs to exclude")

    args = ap.parse_args()
    if args.cmd == "generate":
        cmd = [
            sys.executable,
            "-m",
            "seq2vid.gen_cli",
            f"--bucket-config={args.config}",
            f"--out-dir={args.out_dir}",
            f"--bucket-batch-size={args.bucket_batch_size}",
        ]
        if args.bucket_seed is not None:
            cmd.append(f"--bucket-seed={args.bucket_seed}")
        if args.bucket_overwrite:
            cmd.append("--bucket-overwrite")
        if args.bucket_write_combined:
            cmd.append("--bucket-write-combined")
        if args.bucket_max_no_progress is not None:
            cmd.append(f"--bucket-max-no-progress={args.bucket_max_no_progress}")
        _extend_patterns(cmd, "--include-buckets", args.include_buckets)
        _extend_patterns(cmd, "--exclude-buckets", args.exclude_buckets)
        if args.log_progress:
            cmd.append("--log-progress")
        if args.gen_workers is not None:
            cmd.append(f"--gen-workers={args.gen_workers}")
        if args.discover_len_mult is not None:
            cmd.append(f"--discover-len-mult={args.discover_len_mult}")
        if args.max_attempts is not None:
            cmd.append(f"--max-attempts={args.max_attempts}")
        if args.top_k is not None:
            cmd.append(f"--top-k={args.top_k}")
        if args.ngram_max is not None:
            cmd.append(f"--ngram-max={args.ngram_max}")
        if args.max_rules is not None:
            cmd.append(f"--max-rules={args.max_rules}")
        _run(cmd)
    elif args.cmd == "render":
        cmd = [
            sys.executable,
            "-m",
            "seq2vid.render_cli",
            f"--bucket-config={args.config}",
            f"--template={args.template}",
            f"--sequences-root={args.sequences_root}",
            f"--out-dir={args.out_dir}",
        ]
        _extend_patterns(cmd, "--include-buckets", args.include_buckets)
        _extend_patterns(cmd, "--exclude-buckets", args.exclude_buckets)
        if args.bucket_overwrite:
            cmd.append("--bucket-overwrite")
        if args.num_questions is not None:
            cmd.append(f"--num-questions={args.num_questions}")
        if args.num_videos is not None:
            cmd.append(f"--num-videos={args.num_videos}")
        if args.render_workers is not None:
            cmd.append(f"--render-workers={args.render_workers}")
        if args.assignment_seed is not None:
            cmd.append(f"--assignment-seed={args.assignment_seed}")
        if args.fps is not None:
            cmd.append(f"--fps={args.fps}")
        if args.ffmpeg_crf is not None:
            cmd.append(f"--ffmpeg-crf={args.ffmpeg_crf}")
        if args.ffmpeg_preset is not None:
            cmd.append(f"--ffmpeg-preset={args.ffmpeg_preset}")
        if args.ffmpeg_codec is not None:
            cmd.append(f"--ffmpeg-codec={args.ffmpeg_codec}")
        if args.question_min_len is not None:
            cmd.append(f"--question-min-len={args.question_min_len}")
        if args.questions_at_end:
            cmd.append("--questions-at-end")
        if args.questions_only:
            cmd.append("--questions-only")
        if args.clip_options:
            cmd.append("--clip-options")
        if args.uniform_uncertain:
            cmd.append("--uniform-uncertain")
        cmd.append(f"--question-mode={args.question_mode}")
        if args.hide_question_text:
            cmd.append("--hide-question-text")
        if args.log_progress:
            cmd.append("--log-progress")
        if args.skip_validation:
            cmd.append("--skip-validation")
        if args.debug_frame_metadata:
            cmd.append("--debug-frame-metadata")
        if args.sequence_source:
            for entry in args.sequence_source:
                cmd.append(f"--sequence-source={entry}")
        if args.spatial_question_fraction is not None:
            cmd.append(f"--spatial-question-fraction={args.spatial_question_fraction}")
        _run(cmd)
    elif args.cmd == "manifest":
        seq_manifest = Path(args.sequences_manifest)
        render_manifest = Path(args.render_manifest)
        if not seq_manifest.exists():
            raise FileNotFoundError(seq_manifest)
        if not render_manifest.exists():
            raise FileNotFoundError(render_manifest)
        seq_data = json.loads(seq_manifest.read_text())
        render_data = json.loads(render_manifest.read_text())
        render_map = {entry.get("bucket_id"): entry for entry in render_data.get("buckets", [])}
        combined = []
        for entry in seq_data.get("buckets", []):
            bucket_id = entry.get("bucket_id")
            combined.append(
                {
                    "bucket_id": bucket_id,
                    "sequences_file": entry.get("sequences_file"),
                    "meta": entry,
                    "render": render_map.get(bucket_id),
                }
            )
        Path(args.output).write_text(json.dumps({"buckets": combined}, indent=2))
        print(f"[write] {args.output} ({len(combined)} buckets)")
    else:
        render_root = Path(args.render_root)
        if not render_root.exists():
            raise FileNotFoundError(render_root)
        entries = []
        bucket_count = 0
        for bucket_dir in sorted(render_root.iterdir()):
            if not bucket_dir.is_dir():
                continue
            bucket_id = bucket_dir.name
            if not _matches_bucket(bucket_id, args.include_buckets, args.exclude_buckets):
                continue
            q_path = bucket_dir / args.questions_file
            if not q_path.exists():
                print(f"[skip] bucket {bucket_id}: missing {args.questions_file}")
                continue
            try:
                payload = json.loads(q_path.read_text())
            except json.JSONDecodeError as exc:
                print(f"[skip] bucket {bucket_id}: failed to parse {q_path} ({exc})")
                continue
            bucket_count += 1
            questions_at_end = bool(payload.get("questions_at_end", False))
            for video in payload.get("videos", []):
                item = dict(video)
                item["bucket_from"] = str(q_path.resolve())
                item["bucket_id"] = bucket_id
                item.setdefault("questions_at_end", questions_at_end)
                entries.append(item)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"videos": entries}, indent=2))
        print(f"[write] {output_path} ({len(entries)} videos from {bucket_count} buckets)")


if __name__ == "__main__":
    main()
