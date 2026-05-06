#!/usr/bin/env python
"""Run checkpoint evaluation on a generated validation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from calibrated_memory.valset.evaluator import ValidationConfig, run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing config.json.")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the generated questions.json manifest.")
    parser.add_argument("--task", choices=["membership", "continuation"], required=True)
    parser.add_argument("--cont-len", type=int, default=None, help="Override continuation length when --task=continuation.")
    parser.add_argument("--token-offset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip writing SVG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ValidationConfig(
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        task=args.task,
        checkpoint_name=args.checkpoint_name,
        token_offset=args.token_offset,
        cont_len=args.cont_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )
    metrics = run_validation(cfg)
    print(f"Evaluated {sum(row.question_count for row in metrics.bucket_metrics)} questions.")


if __name__ == "__main__":
    main()
