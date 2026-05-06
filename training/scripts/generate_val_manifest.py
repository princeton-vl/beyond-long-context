#!/usr/bin/env python
"""CLI for generating balanced synthetic validation manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from calibrated_memory.valset.generation import (
    DEFAULT_BUCKETS,
    ValGenerationConfig,
    generate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["membership", "continuation"], required=True)
    parser.add_argument("--num-sequences", type=int, default=5000)
    parser.add_argument("--queries-per-sequence", type=int, default=15)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--min-query-len", type=int, default=3)
    parser.add_argument("--max-query-len", type=int, default=7)
    parser.add_argument("--cont-len", type=int, default=4, help="Continuation length (continuation only).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--token-offset", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ValGenerationConfig(
        task=args.task,
        num_sequences=args.num_sequences,
        queries_per_sequence=args.queries_per_sequence,
        vocab_size=args.vocab_size,
        min_query_len=args.min_query_len,
        max_query_len=args.max_query_len,
        cont_len=args.cont_len,
        seed=args.seed,
        token_offset=args.token_offset,
        buckets=DEFAULT_BUCKETS,
    )
    result = generate_manifest(cfg)
    output_dir = args.output_dir or _default_output_folder(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "questions.json"
    manifest_path.write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    config_path = output_dir / "generator_config.json"
    config_path.write_text(json.dumps(vars(cfg), indent=2, default=str), encoding="utf-8")
    _write_bucket_summary(output_dir / "bucket_summary.csv", result.bucket_counts)
    _write_question_summary(output_dir / "question_spans.csv", result.question_rows)
    print(f"Wrote manifest to {manifest_path}")


def _default_output_folder(task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("val") / task.replace("_", "-") / f"run-{timestamp}"


def _write_bucket_summary(path: Path, bucket_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["bucket,count"]
    lines.extend(f"{bucket},{count}" for bucket, count in bucket_counts.items())
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_question_summary(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["video_index,question_index,bucket,stream_length,truth_kind,concerned_ranges"]
    for row in rows:
        ranges = ";".join(f"[{start}:{end}]" for start, end in row.concerned_ranges)
        lines.append(
            f"{row.video_index},{row.question_index},{row.bucket_id},{row.stream_length},{row.truth_kind},\"{ranges}\""
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
