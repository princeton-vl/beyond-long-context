"""Utility CLI for generating and evaluating low-entropy validation sets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from calibrated_memory.valset import (
    PowerBucket,
    SyntheticValGenerator,
    ValGenerationConfig,
)
from calibrated_memory.valset.evaluator import ValidationConfig, run_validation


BUCKETS: Sequence[PowerBucket] = (
    PowerBucket("seq16-32", 16, 32),
    PowerBucket("seq32-64", 32, 64),
    PowerBucket("seq64-128", 64, 128),
    PowerBucket("seq128-256", 128, 256),
    PowerBucket("seq256-512", 256, 512),
    PowerBucket("seq512-1024", 512, 1024),
    PowerBucket("seq1024-2048", 1024, 2048, inclusive_upper=True),
)


def _write_bucket_summary(path: Path, bucket_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "count"])
        for bucket in sorted(bucket_counts):
            writer.writerow([bucket, bucket_counts[bucket]])


def _write_question_spans(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_index",
                "question_index",
                "bucket",
                "stream_length",
                "truth_kind",
                "concerned_ranges",
            ]
        )
        for row in rows:
            ranges = ";".join(f"[{start}:{end}]" for start, end in row.concerned_ranges) or "[]"
            writer.writerow(
                [
                    row.video_index,
                    row.question_index,
                    row.bucket_id,
                    row.stream_length,
                    row.truth_kind,
                    ranges,
                ]
            )


def cmd_generate(args: argparse.Namespace) -> None:
    bucket_count = len(BUCKETS)
    if args.streams % bucket_count != 0:
        raise SystemExit(
            f"--streams must be divisible by {bucket_count} to maintain equal buckets; got {args.streams}."
        )
    config = ValGenerationConfig(
        task="membership",
        num_sequences=args.streams,
        queries_per_sequence=args.queries,
        vocab_size=args.vocab_size,
        token_offset=args.token_offset,
        min_query_len=args.min_query_len,
        max_query_len=args.max_query_len,
        cont_len=args.cont_len,
        seed=args.seed,
        buckets=BUCKETS,
    )
    result = SyntheticValGenerator(config).build()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generator_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=lambda x: str(x)),
        encoding="utf-8",
    )
    (output_dir / "questions.json").write_text(
        json.dumps(result.manifest, indent=2),
        encoding="utf-8",
    )
    _write_bucket_summary(output_dir / "bucket_summary.csv", result.bucket_counts)
    _write_question_spans(output_dir / "question_spans.csv", result.question_rows)
    print(
        f"Generated {args.streams} streams with {args.queries} queries each under {output_dir}/questions.json"
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = ValidationConfig(
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        task="membership",
        checkpoint_name=args.checkpoint_name,
        token_offset=args.token_offset,
        cont_len=args.cont_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )
    metrics = run_validation(config)
    print("Per-bucket metrics:")
    for row in metrics.bucket_metrics:
        print(
            f"  {row.bucket_id}: acc={row.accuracy:.3f} cov={row.coverage:.3f} abstain={row.abstention_rate:.3f} "
            f"uncertain_error={row.uncertain_truth_error_pct:.3f} option_abstain={row.option_abstain_pct:.3f}"
        )
    print("Per-bucket eighth metrics:")
    for row in metrics.bucket_eighth_metrics:
        print(
            f"  {row.bucket_id} eighth={row.eighth_index}: acc={row.accuracy:.3f} cov={row.coverage:.3f} abstain={row.abstention_rate:.3f} "
            f"uncertain_error={row.uncertain_truth_error_pct:.3f} option_abstain={row.option_abstain_pct:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-entropy validation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate a synthetic validation manifest")
    gen.add_argument("--output-dir", type=Path, required=True, help="Directory for manifest + metadata")
    gen.add_argument("--streams", type=int, default=5040, help="Total stream count (must be divisible by 7)")
    gen.add_argument("--queries", type=int, default=15, help="Queries per stream")
    gen.add_argument("--vocab-size", type=int, default=16)
    gen.add_argument("--token-offset", type=int, default=32)
    gen.add_argument("--min-query-len", type=int, default=3)
    gen.add_argument("--max-query-len", type=int, default=7)
    gen.add_argument("--cont-len", type=int, default=4)
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=cmd_generate)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a manifest")
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--checkpoint-name", default="best.ckpt")
    eval_parser.add_argument("--token-offset", type=int, default=None)
    eval_parser.add_argument("--cont-len", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--output-dir", type=Path, default=None)
    eval_parser.add_argument("--no-plots", action="store_true")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
