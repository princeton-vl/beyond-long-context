from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.data.video_features import pipeline


def _default_plan_metadata(output_dir: Path, meta: Path | None) -> Path:
    if meta is not None:
        return meta.resolve()
    return (output_dir / "plan" / "plan_meta.json").resolve()


def _gather_shard_manifests(output_dir: Path, overrides: Sequence[Path] | None) -> list[Path]:
    if overrides:
        return [path.resolve() for path in overrides]
    shard_dir = output_dir / "shards"
    return sorted(shard_dir.glob("shard_*.json"))


def _load_mean_payload(mean_path: Path | None) -> tuple[torch.Tensor | None, dict[str, object]]:
    if mean_path is None or not mean_path.exists():
        return None, {}
    payload = torch.load(mean_path, map_location="cpu")
    if isinstance(payload, dict) and "mean" in payload:
        metadata = {k: v for k, v in payload.items() if k != "mean"}
        return payload["mean"].to(torch.float32), metadata
    if isinstance(payload, torch.Tensor):
        return payload.to(torch.float32), {}
    raise ValueError(f"Unsupported mean cache format at {mean_path}")


def _parse_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in entries:
        if "=" not in pair:
            raise ValueError(f"Expected key=value format, got: {pair}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _filters_from_metadata(metadata: dict[str, object]) -> dict[str, object] | None:
    max_seq_len = int(metadata.get("max_seq_len", -1))
    skipped = int(metadata.get("skipped_videos", 0))
    total_after = int(metadata.get("total_videos", 0))
    if max_seq_len <= 0 and skipped == 0:
        return None
    checked = total_after + skipped
    return {
        "max_seq_len": max_seq_len if max_seq_len > 0 else None,
        "checked_videos": checked,
        "skipped_videos": skipped,
    }


def _infer_embed_dim(manifest_paths: Sequence[Path]) -> int:
    for path in manifest_paths:
        payload = json.loads(path.read_text())
        for entry in payload.get("videos", []):
            stream = entry.get("stream_embeddings") or {}
            dim = stream.get("embed_dim")
            if dim:
                return int(dim)
    raise ValueError("Unable to infer embed_dim from shard manifests")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge shard manifests into a final embedding manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-metadata", type=Path, default=None)
    parser.add_argument("--mean-path", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument(
        "--shard-manifest",
        action="append",
        default=None,
        help="Optional explicit shard manifest path (repeatable)",
    )
    parser.add_argument("--backbone", required=True, help="Backbone name used for embeddings")
    parser.add_argument(
        "--backbone-option",
        action="append",
        default=[],
        help="Backbone overrides recorded in final manifest",
    )
    parser.add_argument("--fps", type=float, required=True, help="FPS used when sampling frames")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--device", default="cuda", help="Device descriptor to record in the manifest")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    metadata_path = _default_plan_metadata(output_dir, args.plan_metadata)
    metadata = pipeline.load_plan_metadata(metadata_path)
    shard_paths = _gather_shard_manifests(output_dir, args.shard_manifest)
    expected = len(metadata.get("shards", []))
    if expected and len(shard_paths) < expected:
        raise ValueError(f"Expected {expected} shard manifests but found {len(shard_paths)}")
    mean_path = args.mean_path or (output_dir / "zero_mean.pt")
    mean_tensor, mean_meta = _load_mean_payload(mean_path)
    overrides = _parse_overrides(args.backbone_option)
    filters = _filters_from_metadata(metadata)
    embed_dim = _infer_embed_dim(shard_paths)
    backbone_meta = {
        "name": args.backbone,
        "options": overrides,
        "embed_dim": embed_dim,
        "fps": args.fps,
        "dtype": args.dtype,
        "device": args.device,
    }
    metadata["backbone"] = backbone_meta
    manifest_path = (args.output_manifest or (output_dir / "embedding_manifest.json")).resolve()
    pipeline.merge_manifests(
        shard_manifests=shard_paths,
        metadata=metadata,
        embedding_mean=mean_tensor,
        zero_mean_meta=mean_meta,
        filters=filters,
        output_manifest=manifest_path,
    )
    print(json.dumps({"manifest": str(manifest_path), "shards": len(shard_paths)}, indent=2))


if __name__ == "__main__":
    main()
