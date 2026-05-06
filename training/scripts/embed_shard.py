from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.data.video_features import pipeline


def _default_plan_paths(output_dir: Path, plan: Path | None, meta: Path | None) -> tuple[Path, Path]:
    plan_dir = output_dir / "plan"
    return (
        (plan or (plan_dir / "plan.jsonl")).resolve(),
        (meta or (plan_dir / "plan_meta.json")).resolve(),
    )


def _load_mean(mean_path: Path | None) -> torch.Tensor | None:
    if mean_path is None or not mean_path.exists():
        return None
    payload = torch.load(mean_path, map_location="cpu")
    if isinstance(payload, dict) and "mean" in payload:
        return payload["mean"].to(torch.float32)
    if isinstance(payload, torch.Tensor):
        return payload.to(torch.float32)
    raise ValueError(f"Unsupported mean cache format at {mean_path}")


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in values:
        if "=" not in pair:
            raise ValueError(f"Expected key=value format, got '{pair}'")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _load_existing_manifest(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    entries = {}
    for entry in payload.get("videos", []):
        order_index = int(entry.get("order_index", entry.get("video_index", 0)))
        entries[order_index] = entry
    return entries


def _write_manifest(entries: dict[int, dict[str, object]], path: Path) -> None:
    ordered = [entries[idx] for idx in sorted(entries.keys())]
    pipeline.save_shard_manifest(ordered, path)


def _format_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "inf"
    seconds = max(0.0, seconds)
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class _ProgressLogger:
    def __init__(self, total: int, initial: int = 0):
        self.total = total
        self.count = initial
        self.start = time.time()

    def __call__(self, order_index: int, entry: dict[str, object], stats: dict[str, object] | None) -> None:
        self.count += 1
        elapsed = max(1e-6, time.time() - self.start)
        rate = self.count / elapsed
        remaining = max(0, self.total - self.count)
        eta = remaining / rate if rate > 0 and remaining > 0 else 0.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = [
            f"[{ts}] embedded {self.count}/{self.total}",
            f"rate={rate:.2f} vid/s",
            f"eta={_format_duration(eta)}",
            f"order={order_index}",
        ]
        if stats:
            if "decode_sec" in stats:
                summary.append(f"decode={stats['decode_sec']:.2f}s")
            if "embed_sec" in stats:
                summary.append(f"embed={stats['embed_sec']:.2f}s")
        print(" ".join(summary), flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed a shard of the plan on the current host")
    parser.add_argument("--output-dir", type=Path, required=True, help="Embedding output directory")
    parser.add_argument("--plan", type=Path, default=None, help="Plan JSONL path (defaults to <output>/plan/plan.jsonl)")
    parser.add_argument(
        "--plan-metadata",
        type=Path,
        default=None,
        help="Plan metadata path (defaults to <output>/plan/plan_meta.json)",
    )
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index to embed")
    parser.add_argument("--backbone", default="videomae-base")
    parser.add_argument(
        "--backbone-option",
        action="append",
        default=[],
        help="Override backbone config via key=value entries",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma separated list of devices (e.g. cuda:0,cuda:1). Defaults to single --device",
    )
    parser.add_argument("--num-workers", type=int, default=-1, help="Worker processes (defaults to len(devices))")
    parser.add_argument("--cpu-workers", type=int, default=0, help="Decode workers feeding GPUs")
    parser.add_argument("--prefetch-limit", type=int, default=4, help="Buffered decoded samples per GPU worker")
    parser.add_argument(
        "--dispatch-mode",
        choices=("queue", "stride"),
        default="queue",
        help="Assignment strategy when >1 GPU worker",
    )
    parser.add_argument("--mean-path", type=Path, default=None, help="Path to zero-mean cache")
    parser.add_argument("--log-interval", type=float, default=60.0, help="Seconds between throughput logs")
    parser.add_argument("--resume", action="store_true", help="Skip entries already present in shard manifest")
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=None,
        help="Optional override for shard manifest path",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=-1,
        help="Debug helper: only embed the first N videos from the shard",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    plan_path, meta_path = _default_plan_paths(output_dir, args.plan, args.plan_metadata)
    metadata = pipeline.load_plan_metadata(meta_path)
    shard_index = args.shard_index
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = (args.shard_manifest or shard_dir / f"shard_{shard_index:04d}.json").resolve()
    existing_entries = _load_existing_manifest(shard_manifest_path) if args.resume else {}
    plan_items = [
        item
        for item in pipeline.iter_plan_items(plan_path, shard_index=shard_index, output_dir=output_dir)
    ]
    if args.max_videos > 0:
        plan_items = plan_items[: args.max_videos]
    if not plan_items:
        raise ValueError(f"No plan entries assigned to shard {shard_index}")
    pending = [item for item in plan_items if item.order_index not in existing_entries]
    if not pending:
        print(json.dumps({"status": "complete", "shard_index": shard_index}))
        return
    overrides = _parse_overrides(args.backbone_option)
    device_list = pipeline.parse_device_list(args.device, args.devices)
    mean_tensor = _load_mean(args.mean_path or (output_dir / "zero_mean.pt"))
    root_path = Path(metadata.get("root_path", output_dir))
    start_time = time.time()
    print(
        json.dumps(
            {
                "event": "shard_start",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "shard_index": shard_index,
                "pending_videos": len(pending),
                "devices": device_list,
                "cpu_workers": args.cpu_workers,
            }
        ),
        flush=True,
    )
    initial_completed = len(plan_items) - len(pending)
    progress_logger = _ProgressLogger(total=len(plan_items), initial=initial_completed)
    results = pipeline.embed_work_items(
        pending,
        backbone=args.backbone,
        overrides=overrides,
        fps=args.fps,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device_list=device_list,
        num_workers=args.num_workers,
        cpu_workers=args.cpu_workers,
        prefetch_limit=args.prefetch_limit,
        dispatch_mode=args.dispatch_mode,
        embedding_mean=mean_tensor,
        root_path=root_path,
        log_interval=args.log_interval,
        progress_callback=progress_logger,
    )
    duration = max(1e-6, time.time() - start_time)
    manifest_entries = dict(existing_entries)
    for order_index, entry, stats in results:
        entry["order_index"] = order_index
        manifest_entries[order_index] = entry
    _write_manifest(manifest_entries, shard_manifest_path)
    gpu_stats = [pipeline.describe_gpu(device) for device in device_list]
    summary = {
        "shard_index": shard_index,
        "videos_embedded": len(results),
        "videos_remaining": len(plan_items) - len(manifest_entries),
        "duration_sec": duration,
        "throughput_vps": len(results) / duration,
        "gpu_stats": gpu_stats,
        "manifest": str(shard_manifest_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
