from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.data.video_features import pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an embedding plan JSONL and shard summaries.")
    parser.add_argument("--questions", type=Path, required=True, help="Path to questions.json manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Embedding output directory")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Override root for relative video/clip paths (defaults to questions parent)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=-1,
        help="Filter videos whose tokenized sequences exceed this limit",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1024,
        help="Number of videos per shard (rounded up for the last shard)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_items, shard_specs, skipped, root_path = pipeline.build_plan(
        manifest_path=args.questions.resolve(),
        output_dir=output_dir,
        manifest_root=args.manifest_root,
        shard_size=max(1, args.shard_size),
        max_seq_len=args.max_seq_len,
    )
    summary = pipeline.write_plan(
        plan_items=plan_items,
        shard_specs=shard_specs,
        skipped_videos=skipped,
        manifest_path=args.questions.resolve(),
        output_dir=output_dir,
        root_path=root_path,
        max_seq_len=args.max_seq_len,
    )
    stats = {
        "plan_path": str(summary.plan_path),
        "metadata_path": str(summary.metadata_path),
        "total_videos": summary.total_videos,
        "skipped_videos": summary.skipped_videos,
        "shards": [
            {
                "index": spec.index,
                "count": spec.count,
                "tags": list(spec.tags),
            }
            for spec in shard_specs
        ],
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
