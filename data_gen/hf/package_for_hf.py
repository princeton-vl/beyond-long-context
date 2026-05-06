#!/usr/bin/env python3
"""Package bucket render outputs into a Hugging Face-friendly directory."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional, Tuple


def _list_buckets(root: Path, includes: Optional[List[str]], excludes: Optional[List[str]]) -> List[str]:
    buckets: List[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if includes and not any(fnmatch(name, pat) for pat in includes):
            continue
        if excludes and any(fnmatch(name, pat) for pat in excludes):
            continue
        buckets.append(name)
    return buckets


def _ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output directory {path} already exists (use --overwrite to replace it)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_media(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _rewrite_video_entry(
    entry: dict,
    bucket: str,
    dest_root: Path,
    include_clips: bool,
) -> Tuple[dict, List[Tuple[Path, Path]]]:
    copies: List[Tuple[Path, Path]] = []
    updated = dict(entry)
    video_path = Path(entry.get("video_path", ""))
    if video_path and video_path.exists():
        dest_video = dest_root / "videos" / bucket / video_path.name
        copies.append((video_path, dest_video))
        updated["video_path"] = str(Path("videos") / bucket / video_path.name)
    else:
        updated["video_path"] = ""

    return updated, copies


def _rewrite_questions(
    payload: dict,
    bucket: str,
    dest_root: Path,
    include_clips: bool,
) -> Tuple[dict, List[Tuple[Path, Path]]]:
    updated = dict(payload)
    media_copies: List[Tuple[Path, Path]] = []
    videos = []
    for video_entry in payload.get("videos", []):
        new_entry, copies = _rewrite_video_entry(
            video_entry,
            bucket,
            dest_root,
            include_clips,
        )
        media_copies.extend(copies)
        # rewrite nested question option clip paths
        new_questions = []
        for question in new_entry.get("questions", []):
            q = dict(question)
            if "options" in q:
                opts = []
                for option in q.get("options", []):
                    opt = dict(option)
                    clip_path = opt.get("clip_path")
                    if clip_path and include_clips:
                        src = Path(option["clip_path"])
                        if src.exists():
                            dest = dest_root / "clips" / bucket / src.name
                            media_copies.append((src, dest))
                            opt["clip_path"] = str(Path("clips") / bucket / src.name)
                        else:
                            opt["clip_path"] = ""
                    else:
                        opt["clip_path"] = ""
                    opts.append(opt)
                q["options"] = opts
            candidate = q.get("candidate")
            if candidate is not None:
                cand = dict(candidate)
                clip_path = cand.get("clip_path")
                if clip_path and include_clips:
                    src = Path(clip_path)
                    if src.exists():
                        dest = dest_root / "clips" / bucket / src.name
                        media_copies.append((src, dest))
                        cand["clip_path"] = str(Path("clips") / bucket / src.name)
                    else:
                        cand["clip_path"] = ""
                else:
                    cand["clip_path"] = ""
                q["candidate"] = cand
            new_questions.append(q)
        new_entry["questions"] = new_questions
        videos.append(new_entry)
    updated["videos"] = videos
    return updated, media_copies


def _package_bucket(
    bucket: str,
    render_root: Path,
    sequences_root: Path,
    out_root: Path,
    questions_file: str,
    include_clips: bool,
) -> Tuple[int, dict]:
    bucket_render = render_root / bucket
    bucket_seq_dir = sequences_root / bucket
    questions_path = bucket_render / questions_file
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing {questions_file} for bucket {bucket} at {questions_path}")
    seq_path = bucket_seq_dir / "sequences.json"
    if not seq_path.exists():
        seq_path = bucket_seq_dir / "sequences.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences.json for bucket {bucket} under {bucket_seq_dir}")

    payload = json.loads(questions_path.read_text())
    rewritten, media_copies = _rewrite_questions(
        payload,
        bucket,
        out_root,
        include_clips,
    )

    # Copy questions payload
    out_questions = out_root / "questions" / f"{bucket}.json"
    out_questions.parent.mkdir(parents=True, exist_ok=True)
    out_questions.write_text(json.dumps(rewritten, indent=2))

    # Copy sequences
    out_sequences = out_root / "sequences" / f"{bucket}.json"
    out_sequences.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seq_path, out_sequences)

    # Copy per-bucket meta if present
    meta_path = bucket_seq_dir / "meta.json"
    if meta_path.exists():
        out_meta = out_root / "sequences" / f"{bucket}_meta.json"
        shutil.copy2(meta_path, out_meta)

    copied_files = set()
    for src, dest in media_copies:
        if src in copied_files:
            continue
        _copy_media(src, dest)
        copied_files.add(src)

    video_count = len(rewritten.get("videos", []))
    return video_count, rewritten


def _copy_manifest(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _create_archive(src_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Package bucket outputs for Hugging Face uploads")
    ap.add_argument("--render-root", required=True, type=Path)
    ap.add_argument("--sequences-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--questions-file", default="questions.json")
    ap.add_argument("--include-buckets", nargs="*", default=None)
    ap.add_argument("--exclude-buckets", nargs="*", default=None)
    ap.add_argument("--archive", type=Path, default=None, help="Optional .tar.gz path to create after packaging")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataset-name", default="spatial_uniform_eval")
    ap.add_argument(
        "--skip-clips",
        action="store_true",
        help="Do not copy per-question clip files (only keep full videos)",
    )
    args = ap.parse_args()

    render_root = args.render_root
    sequences_root = args.sequences_root
    out_dir = args.out_dir

    out_dir = out_dir.resolve()
    _ensure_clean_dir(out_dir, args.overwrite)

    buckets = _list_buckets(render_root, args.include_buckets, args.exclude_buckets)
    if not buckets:
        raise RuntimeError("No buckets matched the provided include/exclude filters")

    total_videos = 0
    merged_questions: List[dict] = []
    include_clips = not args.skip_clips
    for bucket in buckets:
        count, rewritten = _package_bucket(
            bucket=bucket,
            render_root=render_root,
            sequences_root=sequences_root,
            out_root=out_dir,
            questions_file=args.questions_file,
            include_clips=include_clips,
        )
        total_videos += count
        merged_questions.extend(rewritten.get("videos", []))

    # copy manifests if present
    _copy_manifest(render_root / "bucket_render_manifest.json", out_dir / "meta" / "bucket_render_manifest.json")
    _copy_manifest(sequences_root / "bucket_generation_manifest.json", out_dir / "meta" / "bucket_generation_manifest.json")

    merged_path = out_dir / "questions_dataset.json"
    merged_payload = {
        "dataset_name": args.dataset_name,
        "bucket_count": len(buckets),
        "video_count": total_videos,
        "videos": merged_questions,
        "clips_included": include_clips,
    }
    merged_path.write_text(json.dumps(merged_payload, indent=2))

    meta = {
        "dataset_name": args.dataset_name,
        "render_root": str(render_root),
        "sequences_root": str(sequences_root),
        "questions_file": args.questions_file,
        "bucket_count": len(buckets),
        "video_count": total_videos,
        "buckets": buckets,
        "clips_included": include_clips,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    if args.archive:
        _create_archive(out_dir, args.archive)


if __name__ == "__main__":
    main()
