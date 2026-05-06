from __future__ import annotations

import argparse
from pathlib import Path

from .generate_spatial import run_spatial_generation


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paired token + lane sequences per bucket")
    ap.add_argument("--config", default="configs/buckets_spatial.yaml", help="Spatial bucket config path")
    ap.add_argument("--out-dir", default="buckets_spatial/runs_seq", help="Output directory for spatial sequences")
    ap.add_argument("--bucket-batch-size", type=int, default=32, help="Sequences requested per batch")
    ap.add_argument("--gen-workers", type=int, default=8, help="Parallel workers for generation")
    ap.add_argument("--bucket-seed", type=int, default=123, help="Seed for bucket sampling")
    ap.add_argument("--include-buckets", nargs="*", default=None, help="Optional list of bucket IDs to include")
    ap.add_argument("--exclude-buckets", nargs="*", default=None, help="Optional list of bucket IDs to exclude")
    ap.add_argument("--log-progress", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    run_spatial_generation(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        bucket_batch_size=args.bucket_batch_size,
        gen_workers=args.gen_workers,
        bucket_seed=args.bucket_seed,
        include_buckets=args.include_buckets,
        exclude_buckets=args.exclude_buckets,
        log_progress=args.log_progress,
    )


if __name__ == "__main__":
    main()
