from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.data.video_features import pipeline
from calibrated_memory.data.video_features.backbones import build_backbone


def _plan_paths(output_dir: Path, plan_path: Path | None, meta_path: Path | None) -> tuple[Path, Path]:
    base = output_dir / "plan"
    resolved_plan = plan_path or (base / "plan.jsonl")
    resolved_meta = meta_path or (base / "plan_meta.json")
    return resolved_plan.resolve(), resolved_meta.resolve()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute dataset-level embedding mean for later normalization.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Embedding output directory")
    parser.add_argument("--plan", type=Path, default=None, help="Optional override for plan JSONL path")
    parser.add_argument(
        "--plan-metadata",
        type=Path,
        default=None,
        help="Optional override for plan metadata JSON",
    )
    parser.add_argument("--backbone", default="videomae-base", help="Backbone name")
    parser.add_argument(
        "--backbone-option",
        action="append",
        default=[],
        help="Override backbone config via key=value entries",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="Sampling FPS for zero-mean estimation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for backbone forward pass")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="auto", help="Device for mean computation (default: auto)")
    parser.add_argument("--sample-videos", type=int, default=32, help="Videos to sample for mean estimation")
    parser.add_argument("--sample-frames", type=int, default=256, help="Frames per sampled video")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument(
        "--mean-path",
        type=Path,
        default=None,
        help="Output location for cached mean tensor (default: <output-dir>/zero_mean.pt)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing mean cache if present")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    plan_path, meta_path = _plan_paths(output_dir, args.plan, args.plan_metadata)
    metadata = pipeline.load_plan_metadata(meta_path)
    plan_items = list(pipeline.iter_plan_items(plan_path, shard_index=None, output_dir=output_dir))
    if not plan_items:
        raise ValueError("Plan contained no entries. Did you run embed_plan.py?")
    overrides = {}
    for pair in args.backbone_option:
        if "=" not in pair:
            raise ValueError(f"Expected key=value format for --backbone-option, got: {pair}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backbone = build_backbone(
        args.backbone,
        device=device,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        overrides=overrides,
    )
    rng = random.Random(args.seed)
    mean_tensor, sampled = pipeline.estimate_embedding_mean(
        plan_items,
        backbone=backbone,
        rng=rng,
        fps=args.fps,
        sample_videos=args.sample_videos,
        sample_frames=args.sample_frames,
        batch_size=args.batch_size,
    )
    payload = {
        "mean": mean_tensor.to(torch.float32).cpu(),
        "sampled_videos": sampled,
        "sampled_frames_per_video": args.sample_frames,
        "fps": args.fps,
        "backbone": args.backbone,
        "overrides": overrides,
    }
    mean_path = (args.mean_path or (output_dir / "zero_mean.pt")).resolve()
    if mean_path.exists() and not args.force:
        raise FileExistsError(f"Mean cache already exists at {mean_path}; pass --force to overwrite")
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, mean_path)
    stats = {
        "mean_path": str(mean_path),
        "embed_dim": int(mean_tensor.size(0)),
        "sampled_videos": sampled,
        "fps": args.fps,
        "device": str(device),
    }
    metadata.setdefault("preprocessing", {})
    metadata["preprocessing"]["zero_mean_cache"] = str(mean_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
