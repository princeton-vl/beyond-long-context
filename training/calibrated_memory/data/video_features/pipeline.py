"""Modular embedding pipeline primitives for multi-stage processing."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

import torch
import torch.multiprocessing as mp
from safetensors.torch import safe_open, save_file

from .backbones import FeatureBackbone, build_backbone
from .frame_sampler import sample_video_frames
from .utils import is_uncertain_option

try:  # pragma: no cover - optional dependency installed on GPU workers
    import pynvml
except Exception:  # pragma: no cover - NVML optional
    pynvml = None


_PLAN_VERSION = 1


@dataclass(frozen=True)
class VideoWorkItem:
    """Work item that fully describes how to embed a single video."""

    order_index: int
    entry: dict[str, Any]
    video_path: Path
    tensor_path: Path
    shard_index: int
    tags: tuple[str, ...]


@dataclass
class DecodedVideo:
    order_index: int
    entry: dict[str, Any]
    tensor_path: Path
    stream_frames: torch.Tensor
    timestamps: torch.Tensor
    option_frames: dict[tuple[int, int], torch.Tensor]
    option_clip_paths: dict[tuple[int, int], str | None]
    decode_time: float = 0.0


@dataclass(frozen=True)
class ShardSpec:
    index: int
    start: int
    end: int
    count: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PlanSummary:
    plan_path: Path
    metadata_path: Path
    shards: list[ShardSpec]
    total_videos: int
    skipped_videos: int
    output_dir: Path
    root_path: Path
    questions_path: Path


def _json_copy(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _event_record(event: str, **payload: Any) -> str:
    record = {"event": event, "timestamp": datetime.now().isoformat(timespec="seconds")}
    record.update(payload)
    return json.dumps(record)


def _slugify(value: str | None, *, default: str) -> str:
    slug_chars = []
    value = value or ""
    for ch in value.lower():
        if ch.isalnum():
            slug_chars.append(ch)
        else:
            slug_chars.append("-")
    slug = "".join(slug_chars).strip("-")
    return slug or default


def tensor_filename(entry: dict[str, Any]) -> str:
    try:
        video_index = int(entry["video_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each video entry must include an integer video_index") from exc
    bucket_source = entry.get("bucket_id") or entry.get("bucket_from") or ""
    bucket_slug = _slugify(str(bucket_source), default=f"bucket{video_index:05d}")
    try:
        variant = int(entry.get("variant", 0))
    except (TypeError, ValueError):
        variant = 0
    return f"video{video_index:05d}_{bucket_slug}_v{variant:02d}.safetensors"


def resolve_manifest_root(manifest_path: Path, override: Path | None) -> Path:
    if override is None:
        return manifest_path.parent.resolve()
    candidate = Path(override).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    return candidate


def resolve_asset_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return candidate


def _sequence_token_count(entry: dict[str, Any]) -> int | None:
    sequences = entry.get("sequences_used") or {}
    total = 0
    if isinstance(sequences, dict) and sequences:
        for payload in sequences.values():
            if isinstance(payload, (list, tuple)):
                total += len(payload)
            elif isinstance(payload, dict):
                tokens = payload.get("tokens") if isinstance(payload, dict) else None
                if isinstance(tokens, (list, tuple)):
                    total += len(tokens)
        if total > 0:
            return total
    sequence = entry.get("sequence")
    if isinstance(sequence, (list, tuple)) and sequence:
        return len(sequence)
    return None


def _question_mode(question: dict[str, Any]) -> str:
    return str(question.get("question_mode") or "exists").lower()


def _tags_for_entry(entry: dict[str, Any]) -> tuple[str, ...]:
    modes = {_question_mode(q) for q in entry.get("questions", []) or []}
    tags: list[str] = []
    if any(mode.startswith("continuation") or mode.startswith("sequence") for mode in modes):
        tags.append("sequential")
    if any(mode in {"exists", "membership"} for mode in modes) or not tags:
        tags.append("spatial")
    return tuple(sorted(set(tags)))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_plan(
    *,
    manifest_path: Path,
    output_dir: Path,
    manifest_root: Path | None,
    shard_size: int,
    max_seq_len: int,
) -> tuple[list[VideoWorkItem], list[ShardSpec], int, Path]:
    manifest = json.loads(manifest_path.read_text())
    videos = list(manifest.get("videos", []))
    if not videos:
        raise ValueError("Manifest contained no videos to embed")
    root_path = resolve_manifest_root(manifest_path, manifest_root)
    tensor_dir = output_dir / "videos"
    _ensure_dir(tensor_dir)
    entries: list[VideoWorkItem] = []
    skipped = 0
    shard_size = max(1, shard_size) if shard_size > 0 else len(videos)
    for order_index, raw_entry in enumerate(videos):
        entry = _json_copy(raw_entry)
        resolved = resolve_asset_path(entry.get("video_path"), root_path)
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(
                f"Video path missing for entry index {order_index}: {entry.get('video_path')}"
            )
        token_count = _sequence_token_count(entry)
        if max_seq_len > 0 and token_count is not None and token_count > max_seq_len:
            skipped += 1
            continue
        entry["_resolved_video_path"] = str(resolved)
        tensor_name = tensor_filename(entry)
        tensor_path = tensor_dir / tensor_name
        tags = _tags_for_entry(entry)
        shard_index = len(entries) // shard_size
        work_item = VideoWorkItem(
            order_index=order_index,
            entry=entry,
            video_path=resolved,
            tensor_path=tensor_path,
            shard_index=shard_index,
            tags=tags,
        )
        entries.append(work_item)
    if not entries:
        raise ValueError("No videos satisfied the configured constraints")
    shard_specs: list[ShardSpec] = []
    shard_groups: dict[int, list[VideoWorkItem]] = {}
    for item in entries:
        shard_groups.setdefault(item.shard_index, []).append(item)
    sorted_shards = sorted(shard_groups.items())
    for shard_index, items in sorted_shards:
        shard_specs.append(
            ShardSpec(
                index=shard_index,
                start=min(it.order_index for it in items),
                end=max(it.order_index for it in items),
                count=len(items),
                tags=_collect_shard_tags(items),
            )
        )
    return entries, shard_specs, skipped, root_path


def _collect_shard_tags(items: Sequence[VideoWorkItem]) -> tuple[str, ...]:
    tags: set[str] = set()
    for item in items:
        tags.update(item.tags)
    return tuple(sorted(tags))


def write_plan(
    *,
    plan_items: list[VideoWorkItem],
    shard_specs: list[ShardSpec],
    skipped_videos: int,
    manifest_path: Path,
    output_dir: Path,
    root_path: Path,
    max_seq_len: int,
) -> PlanSummary:
    plan_dir = output_dir / "plan"
    _ensure_dir(plan_dir)
    plan_path = plan_dir / "plan.jsonl"
    metadata_path = plan_dir / "plan_meta.json"
    data: list[str] = []
    for item in plan_items:
        record = {
            "order_index": item.order_index,
            "video_path": str(item.video_path),
            "tensor_relpath": str(item.tensor_path.relative_to(output_dir)),
            "entry": item.entry,
            "shard_index": item.shard_index,
            "tags": list(item.tags),
        }
        data.append(json.dumps(record))
    plan_path.write_text("\n".join(data) + "\n")
    metadata = {
        "version": _PLAN_VERSION,
        "created_at": time.time(),
        "questions_path": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_videos": len(plan_items),
        "skipped_videos": skipped_videos,
        "root_path": str(root_path),
        "max_seq_len": max_seq_len,
        "shards": [
            {
                "index": spec.index,
                "start": spec.start,
                "end": spec.end,
                "count": spec.count,
                "tags": list(spec.tags),
            }
            for spec in shard_specs
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return PlanSummary(
        plan_path=plan_path,
        metadata_path=metadata_path,
        shards=shard_specs,
        total_videos=len(plan_items),
        skipped_videos=skipped_videos,
        output_dir=output_dir.resolve(),
        root_path=root_path,
        questions_path=manifest_path.resolve(),
    )


def load_plan_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text())
    version = int(payload.get("version", 0))
    if version != _PLAN_VERSION:
        raise ValueError(
            f"Unsupported plan metadata version {version}; expected {_PLAN_VERSION}."
        )
    return payload


def iter_plan_items(
    plan_path: Path,
    *,
    shard_index: int | None,
    output_dir: Path,
) -> Iterator[VideoWorkItem]:
    with plan_path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if shard_index is not None and int(record.get("shard_index", -1)) != shard_index:
                continue
            entry = record["entry"]
            video_path = Path(record["video_path"]).expanduser()
            tensor_relpath = Path(record["tensor_relpath"])
            tensor_path = output_dir / tensor_relpath
            tags = tuple(record.get("tags", []))
            yield VideoWorkItem(
                order_index=int(record["order_index"]),
                entry=entry,
                video_path=video_path,
                tensor_path=tensor_path,
                shard_index=int(record.get("shard_index", 0)),
                tags=tags,
            )

def _sample_indices(count: int, limit: int, rng: random.Random) -> Iterable[int]:
    if limit <= 0 or count <= limit:
        return range(count)
    return sorted(rng.sample(range(count), limit))


def _limit_frames(frames: torch.Tensor, limit: int, rng: random.Random) -> torch.Tensor:
    if limit <= 0 or frames.size(0) <= limit:
        return frames
    indices = torch.tensor(list(_sample_indices(frames.size(0), limit, rng)), dtype=torch.long)
    return frames.index_select(0, indices)


def _frames_to_float(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype != torch.float32:
        return frames.to(torch.float32) / 255.0
    return frames


def _normalize_question_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _assert_single_question_index(questions: list[dict[str, Any]], video_index: Any) -> None:
    normalized = {
        idx
        for idx in (_normalize_question_index(q.get("question_index")) for q in questions)
        if idx is not None
    }
    if len(normalized) > 1:
        ordered = ", ".join(str(idx) for idx in sorted(normalized))
        raise NotImplementedError(
            "Multiple question_index values per video are not supported yet "
            f"(video {video_index}, found: {ordered})."
        )


def decode_work_item(
    item: VideoWorkItem,
    *,
    fps: float,
    root_path: Path,
) -> DecodedVideo:
    sampled_frames, timestamps = sample_video_frames(item.video_path, fps=fps)
    questions = item.entry.get("questions", []) or []
    _assert_single_question_index(questions, item.entry.get("video_index"))
    option_frames: dict[tuple[int, int], torch.Tensor] = {}
    option_clip_paths: dict[tuple[int, int], str | None] = {}
    for question_idx, question in enumerate(questions):
        for opt_idx, option in enumerate(question.get("options", []) or []):
            clip_file = resolve_asset_path(option.get("clip_path"), root_path)
            frames = _extract_option_frames(
                question,
                option,
                item.video_path,
                fps,
                clip_file,
                sampled_frames,
            )
            option_frames[(question_idx, opt_idx)] = frames
            option_clip_paths[(question_idx, opt_idx)] = str(clip_file) if clip_file else None
    return DecodedVideo(
        order_index=item.order_index,
        entry=item.entry,
        tensor_path=item.tensor_path,
        stream_frames=sampled_frames,
        timestamps=timestamps,
        option_frames=option_frames,
        option_clip_paths=option_clip_paths,
    )


def _extract_option_frames(
    question: dict[str, Any],
    option: dict[str, Any],
    video_path: Path,
    target_fps: float,
    clip_path: Path | None,
    stream_frames: torch.Tensor | None,
) -> torch.Tensor:
    bounds = _resolve_clip_bounds(question, option)
    if bounds and stream_frames is not None:
        start, end = bounds
        sliced = _slice_stream_frames(stream_frames, target_fps, start, end)
        if sliced.numel() > 0:
            return sliced
    if clip_path:
        if not clip_path.exists():
            raise FileNotFoundError(f"Option clip not found: {clip_path}")
        frames, _ = sample_video_frames(clip_path, fps=target_fps)
        return frames
    if not bounds:
        return torch.zeros(0, 1, 1, 3, dtype=torch.uint8)
    start, end = bounds
    frames, _ = sample_video_frames(video_path, fps=target_fps, start=start, end=end)
    return frames


def _resolve_clip_bounds(question: dict[str, Any], option: dict[str, Any]) -> tuple[float, float] | None:
    def _first(keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in option:
                return option[key]
            if key in question:
                return question[key]
        return None

    start = _first(("clip_start_time", "clip_start"))
    end = _first(("clip_end_time", "clip_end"))
    if start is None or end is None:
        return None
    return float(start), float(end)


def _slice_stream_frames(stream_frames: torch.Tensor, fps: float, start: float, end: float) -> torch.Tensor:
    if stream_frames.numel() == 0 or end <= start:
        return torch.zeros(0, 1, 1, 3, dtype=stream_frames.dtype)
    start_idx = max(0, int(math.floor(start * fps)))
    end_idx = min(stream_frames.size(0), max(start_idx + 1, int(math.ceil(end * fps))))
    return stream_frames[start_idx:end_idx].clone()


def embed_decoded_video(
    decoded: DecodedVideo,
    *,
    backbone: FeatureBackbone,
    batch_size: int,
    embedding_mean: torch.Tensor | None,
) -> dict[str, Any]:
    stream_frames = _frames_to_float(decoded.stream_frames)
    embeddings = backbone.embed_frames(stream_frames, batch_size)
    embeddings = apply_embedding_mean(embeddings, embedding_mean)
    tensor_store: dict[str, torch.Tensor] = {
        "stream_embeddings": embeddings.contiguous(),
        "stream_timestamps": decoded.timestamps.contiguous(),
    }
    option_meta: dict[tuple[int, int], dict[str, Any]] = {}
    for (question_idx, opt_idx), frames in decoded.option_frames.items():
        opt_frame_batch = _frames_to_float(frames)
        opt_embeddings = backbone.embed_frames(opt_frame_batch, batch_size)
        opt_embeddings = apply_embedding_mean(opt_embeddings, embedding_mean)
        tensor_key = f"question_{question_idx}_opt{opt_idx}"
        if tensor_key in tensor_store:
            raise ValueError(
                f"Duplicate tensor key '{tensor_key}' for video {decoded.entry.get('video_index')}"
            )
        tensor_store[tensor_key] = opt_embeddings.contiguous()
        option_meta[(question_idx, opt_idx)] = {
            "count": int(opt_embeddings.size(0)),
            "clip_path": decoded.option_clip_paths.get((question_idx, opt_idx)),
        }
    manifest_entry = _build_manifest_entry(
        decoded.entry,
        timestamps=decoded.timestamps,
        embed_dim=backbone.embed_dim,
        video_file=decoded.tensor_path,
        option_meta=option_meta,
    )
    save_file(tensor_store, str(decoded.tensor_path))
    return manifest_entry


def apply_embedding_mean(tensor: torch.Tensor, mean: torch.Tensor | None) -> torch.Tensor:
    if mean is None or tensor.numel() == 0:
        return tensor
    return tensor - mean.to(tensor.device)


def _build_manifest_entry(
    entry: dict[str, Any],
    *,
    timestamps: torch.Tensor,
    embed_dim: int,
    video_file: Path,
    option_meta: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    new_entry = _json_copy(entry)
    questions: list[dict[str, Any]] = new_entry.get("questions", []) or []
    _assert_single_question_index(questions, entry.get("video_index"))
    new_entry["stream_embeddings"] = {
        "file": str(video_file),
        "embeddings_key": "stream_embeddings",
        "timestamps_key": "stream_timestamps",
        "frame_count": int(timestamps.size(0)),
        "embed_dim": embed_dim,
    }
    for question_idx, question in enumerate(questions):
        last_time = timestamps[-1].item() if timestamps.numel() > 0 else 0.0
        question_time = float(question.get("question_time", last_time))
        cutoff = int((timestamps <= question_time).sum().item()) if timestamps.numel() > 0 else 0
        question["stream_cutoff"] = cutoff
        normalized_index = _normalize_question_index(question.get("question_index"))
        if normalized_index is None:
            question["question_index"] = question_idx
        for opt_idx, option in enumerate(question.get("options", []) or []):
            opt_meta = option_meta.get((question_idx, opt_idx))
            if opt_meta is None:
                continue
            option_payload = _json_copy(option)
            clip_value = opt_meta.get("clip_path")
            if clip_value:
                option_payload["clip_path"] = clip_value
            option_payload["is_uncertain"] = is_uncertain_option(option)
            option_payload["embedding"] = {
                "file": str(video_file),
                "key": f"question_{question_idx}_opt{opt_idx}",
                "count": int(opt_meta["count"]),
                "embed_dim": embed_dim,
            }
            question["options"][opt_idx] = option_payload
    return new_entry


def estimate_embedding_mean(
    items: List[VideoWorkItem],
    *,
    backbone: FeatureBackbone,
    rng: random.Random,
    fps: float,
    sample_videos: int,
    sample_frames: int,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    order = list(range(len(items)))
    rng.shuffle(order)
    if sample_videos > 0:
        order = order[: min(sample_videos, len(order))]
    accum: torch.Tensor | None = None
    total = 0
    for idx in order:
        entry = items[idx].entry
        video_path_value = entry.get("_resolved_video_path") or entry.get("video_path")
        if not video_path_value:
            continue
        video_path = Path(video_path_value)
        frames, _ = sample_video_frames(video_path, fps=fps)
        if frames.numel() == 0:
            continue
        frames = _frames_to_float(frames)
        frames = _limit_frames(frames, sample_frames, rng)
        embeds = backbone.embed_frames(frames, batch_size)
        if embeds.numel() == 0:
            continue
        if accum is None:
            accum = torch.zeros(embeds.size(1), dtype=torch.float32)
        accum += embeds.sum(dim=0)
        total += embeds.size(0)
    if accum is None or total == 0:
        raise ValueError("Zero-mean preprocessing requested but no frames were sampled.")
    return accum / total, len(order)

def _float_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _maybe_clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _materialize_mean(serialized: list[float] | None) -> torch.Tensor | None:
    if serialized is None:
        return None
    return torch.tensor(serialized, dtype=torch.float32)


def _serialize_mean(mean_tensor: torch.Tensor | None) -> list[float] | None:
    if mean_tensor is None:
        return None
    return mean_tensor.to(torch.float32).cpu().tolist()


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def parse_device_list(device_arg: str, devices_arg: str | None) -> list[str]:
    if devices_arg:
        values = [token.strip() for token in devices_arg.split(",") if token.strip()]
        if not values:
            raise ValueError("--devices requires at least one entry (e.g. cuda:0,cuda:1)")
        return values
    device = _device_from_arg(device_arg)
    return [str(device)]


def build_device_assignments(device_list: list[str], worker_count: int) -> list[str]:
    if not device_list:
        raise ValueError("No devices available")
    worker_count = max(1, worker_count)
    assignments: list[str] = []
    for idx in range(worker_count):
        assignments.append(device_list[idx % len(device_list)])
    return assignments


@dataclass(frozen=True)
class _WorkerConfig:
    backbone: str
    overrides: dict[str, str]
    dtype_name: str
    fps: float
    batch_size: int
    root_path: str
    embedding_mean: list[float] | None


@dataclass(frozen=True)
class _DecodeWorkerConfig:
    fps: float
    root_path: str


def _materialize_worker_config(
    config: _WorkerConfig,
    *,
    device_str: str,
) -> tuple[FeatureBackbone, torch.Tensor | None]:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = _float_dtype(config.dtype_name)
    backbone = build_backbone(
        config.backbone,
        device=device,
        dtype=dtype,
        overrides=dict(config.overrides),
    )
    return backbone, _materialize_mean(config.embedding_mean)


def _inline_worker_entry(
    worker_id: int,
    device_str: str,
    assignments: list[VideoWorkItem] | None,
    task_queue: mp.Queue | None,
    result_queue: mp.Queue,
    config: _WorkerConfig,
) -> None:
    try:
        backbone, embedding_mean = _materialize_worker_config(config, device_str=device_str)
        root_path = Path(config.root_path)
        iterator: Iterable[VideoWorkItem]
        if assignments is not None:
            iterator = assignments
        else:
            iterator = _task_iterator(task_queue)
        for item in iterator:
            entry = embed_video_item(
                item,
                backbone=backbone,
                fps=config.fps,
                batch_size=config.batch_size,
                embedding_mean=embedding_mean,
                root_path=root_path,
            )
            result_queue.put((item.order_index, entry))
    except Exception as exc:  # pragma: no cover - worker exceptions bubble up
        result_queue.put(("error", worker_id, str(exc)))


def _gpu_worker_entry(
    worker_id: int,
    device_str: str,
    decoded_queue: mp.Queue,
    result_queue: mp.Queue,
    config: _WorkerConfig,
) -> None:
    try:
        backbone, embedding_mean = _materialize_worker_config(config, device_str=device_str)
        while True:
            payload = decoded_queue.get()
            if payload is None:
                break
            start = time.perf_counter()
            print(
                _event_record(
                    "embed_begin",
                    worker_id=worker_id,
                    order_index=payload.order_index,
                    device=device_str,
                ),
                flush=True,
            )
            entry = embed_decoded_video(
                payload,
                backbone=backbone,
                batch_size=config.batch_size,
                embedding_mean=embedding_mean,
            )
            embed_time = time.perf_counter() - start
            stats = {"embed_sec": embed_time, "device": device_str}
            decode_time = getattr(payload, "decode_time", 0.0)
            if decode_time:
                stats["decode_sec"] = decode_time
            result_queue.put((payload.order_index, entry, stats))
    except Exception as exc:  # pragma: no cover
        result_queue.put(("error", worker_id, str(exc)))


def _cpu_worker_entry(
    worker_id: int,
    task_queue: mp.Queue,
    decoded_queue: mp.Queue,
    result_queue: mp.Queue,
    config: _DecodeWorkerConfig,
) -> None:
    try:
        root_path = Path(config.root_path)
        while True:
            item = task_queue.get()
            if item is None:
                break
            start = time.perf_counter()
            print(
                _event_record(
                    "decode_begin",
                    worker_id=worker_id,
                    order_index=item.order_index,
                ),
                flush=True,
            )
            decoded = decode_work_item(item, fps=config.fps, root_path=root_path)
            decoded.decode_time = time.perf_counter() - start
            print(
                _event_record(
                    "decode_complete",
                    worker_id=worker_id,
                    order_index=item.order_index,
                    decode_sec=decoded.decode_time,
                ),
                flush=True,
            )
            decoded_queue.put(decoded)
    except Exception as exc:  # pragma: no cover
        result_queue.put(("error", worker_id, str(exc)))


def _task_iterator(task_queue: mp.Queue | None) -> Iterable[VideoWorkItem]:
    if task_queue is None:
        return []
    while True:
        item = task_queue.get()
        if item is None:
            break
        yield item


def embed_video_item(
    item: VideoWorkItem,
    *,
    backbone: FeatureBackbone,
    fps: float,
    batch_size: int,
    embedding_mean: torch.Tensor | None,
    root_path: Path,
) -> dict[str, Any]:
    decoded = decode_work_item(item, fps=fps, root_path=root_path)
    return embed_decoded_video(
        decoded,
        backbone=backbone,
        batch_size=batch_size,
        embedding_mean=embedding_mean,
    )


def _run_single_worker(
    pending: list[VideoWorkItem],
    *,
    device: str,
    worker_config: _WorkerConfig,
    progress_callback: Callable[[int, dict[str, Any], dict[str, Any]], None] | None,
) -> list[tuple[int, dict[str, Any]]]:
    backbone, embedding_mean = _materialize_worker_config(worker_config, device_str=device)
    root_path = Path(worker_config.root_path)
    results: list[tuple[int, dict[str, Any]]] = []
    for item in pending:
        entry = embed_video_item(
            item,
            backbone=backbone,
            fps=worker_config.fps,
            batch_size=worker_config.batch_size,
            embedding_mean=embedding_mean,
            root_path=root_path,
        )
        results.append((item.order_index, entry))
        if progress_callback is not None:
            progress_callback(item.order_index, entry, {})
    return results


def _assign_stride_work(pending: list[VideoWorkItem], worker_count: int) -> list[list[VideoWorkItem]]:
    assignments = [[] for _ in range(worker_count)]
    for idx, item in enumerate(pending):
        assignments[idx % worker_count].append(item)
    return assignments


def _run_multi_worker(
    pending: list[VideoWorkItem],
    *,
    worker_config: _WorkerConfig,
    device_assignments: list[str],
    dispatch_mode: str,
    progress_callback: Callable[[int, dict[str, Any], dict[str, Any]], None] | None,
) -> list[tuple[int, dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue() if dispatch_mode == "queue" else None
    result_queue = ctx.Queue()
    assignments: list[list[VideoWorkItem]] | None = None
    if dispatch_mode == "stride":
        assignments = _assign_stride_work(pending, len(device_assignments))
    processes: list[mp.Process] = []
    for worker_id, device_str in enumerate(device_assignments):
        worker_assignments = assignments[worker_id] if assignments is not None else None
        proc = ctx.Process(
            target=_inline_worker_entry,
            args=(
                worker_id,
                device_str,
                worker_assignments,
                task_queue,
                result_queue,
                worker_config,
            ),
        )
        proc.start()
        processes.append(proc)
    if dispatch_mode == "queue" and task_queue is not None:
        for item in pending:
            task_queue.put(item)
        for _ in device_assignments:
            task_queue.put(None)
    results: list[tuple[int, dict[str, Any]]] = []
    collected = 0
    total = len(pending)
    try:
        while collected < total:
            message = result_queue.get()
            if isinstance(message, tuple) and len(message) >= 2 and message[0] == "error":
                raise RuntimeError(f"Worker {message[1]} failed: {message[2]}")
            order_index, entry = message
            if progress_callback is not None:
                progress_callback(order_index, entry, {})
            results.append((order_index, entry))
            collected += 1
        return results
    finally:
        for proc in processes:
            proc.join()
        result_queue.close()
        if task_queue is not None:
            task_queue.close()


def _run_pipeline_workers(
    pending: list[VideoWorkItem],
    *,
    worker_config: _WorkerConfig,
    decode_config: _DecodeWorkerConfig,
    device_assignments: list[str],
    cpu_workers: int,
    prefetch_limit: int,
    log_interval: float,
    progress_callback: Callable[[int, dict[str, Any], dict[str, Any]], None] | None,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    decoded_queue: mp.Queue = ctx.Queue(maxsize=prefetch_limit)
    result_queue: mp.Queue = ctx.Queue()
    gpu_processes: list[mp.Process] = []
    for worker_id, device_str in enumerate(device_assignments):
        proc = ctx.Process(
            target=_gpu_worker_entry,
            args=(worker_id, device_str, decoded_queue, result_queue, worker_config),
        )
        proc.start()
        gpu_processes.append(proc)
    cpu_processes: list[mp.Process] = []
    for cpu_id in range(cpu_workers):
        proc = ctx.Process(
            target=_cpu_worker_entry,
            args=(cpu_id, task_queue, decoded_queue, result_queue, decode_config),
        )
        proc.start()
        cpu_processes.append(proc)
    for item in pending:
        task_queue.put(item)
    total = len(pending)
    results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    collected = 0
    last_log = time.time()
    try:
        while collected < total:
            message = result_queue.get()
            if isinstance(message, tuple) and len(message) >= 2 and message[0] == "error":
                raise RuntimeError(f"Worker {message[1]} failed: {message[2]}")
            if len(message) == 3:
                order_index, entry, stats = message
            else:
                order_index, entry = message
                stats = {}
            results.append((order_index, entry, stats))
            if progress_callback is not None:
                progress_callback(order_index, entry, stats)
            collected += 1
            if log_interval > 0 and time.time() - last_log >= log_interval:
                backlog = None
                if hasattr(decoded_queue, "qsize"):
                    try:
                        backlog = decoded_queue.qsize()
                    except NotImplementedError:
                        backlog = None
                backlog_str = backlog if backlog is not None else "n/a"
                print(
                    f"[embedder] processed {collected}/{total} entries decoded_backlog={backlog_str}"
                )
                last_log = time.time()
    finally:
        for _ in cpu_processes:
            task_queue.put(None)
        for proc in cpu_processes:
            proc.join()
        for _ in gpu_processes:
            decoded_queue.put(None)
        for proc in gpu_processes:
            proc.join()
        task_queue.close()
        decoded_queue.close()
        result_queue.close()
    return results

def embed_work_items(
    pending: list[VideoWorkItem],
    *,
    backbone: str,
    overrides: dict[str, str],
    fps: float,
    batch_size: int,
    dtype: str,
    device_list: list[str],
    num_workers: int,
    cpu_workers: int,
    prefetch_limit: int,
    dispatch_mode: str,
    embedding_mean: torch.Tensor | None,
    root_path: Path,
    log_interval: float,
    progress_callback: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    if not pending:
        return []
    serialized_mean = _serialize_mean(embedding_mean)
    worker_config = _WorkerConfig(
        backbone=backbone,
        overrides=dict(overrides),
        dtype_name=dtype,
        fps=fps,
        batch_size=batch_size,
        root_path=str(root_path),
        embedding_mean=serialized_mean,
    )
    worker_count = num_workers if num_workers > 0 else len(device_list)
    device_assignments = build_device_assignments(device_list, worker_count)
    if cpu_workers > 0:
        if prefetch_limit <= 0:
            raise ValueError("--prefetch-limit must be positive when CPU workers are enabled")
        decode_config = _DecodeWorkerConfig(fps=fps, root_path=str(root_path))
        return _run_pipeline_workers(
            pending,
            worker_config=worker_config,
            decode_config=decode_config,
            device_assignments=device_assignments,
            cpu_workers=cpu_workers,
            prefetch_limit=prefetch_limit,
            log_interval=log_interval,
            progress_callback=progress_callback,
        )
    if len(device_assignments) == 1:
        single_results = _run_single_worker(
            pending,
            device=device_assignments[0],
            worker_config=worker_config,
            progress_callback=progress_callback,
        )
        return [(order, entry, {}) for order, entry in single_results]
    multi = _run_multi_worker(
        pending,
        worker_config=worker_config,
        device_assignments=device_assignments,
        dispatch_mode=dispatch_mode,
        progress_callback=progress_callback,
    )
    return [(order, entry, {}) for order, entry in multi]


def save_shard_manifest(entries: list[dict[str, Any]], path: Path) -> None:
    payload = {"videos": entries}
    path.write_text(json.dumps(payload, indent=2))


def merge_manifests(
    *,
    shard_manifests: Sequence[Path],
    metadata: dict[str, Any],
    embedding_mean: torch.Tensor | None,
    zero_mean_meta: dict[str, Any] | None,
    filters: dict[str, Any] | None,
    output_manifest: Path,
) -> None:
    videos: list[dict[str, Any]] = []
    for manifest_path in shard_manifests:
        payload = json.loads(manifest_path.read_text())
        videos.extend(payload.get("videos", []))
    videos.sort(key=lambda entry: int(entry.get("order_index", entry.get("video_index", 0))))
    manifest = {
        "source_manifest": metadata.get("questions_path"),
        "backbone": metadata.get("backbone", {}),
        "videos": videos,
    }
    if embedding_mean is not None:
        manifest["preprocessing"] = {
            "zero_mean": True,
            "embedding_mean": embedding_mean.to(torch.float32).cpu().tolist(),
        }
        if zero_mean_meta:
            manifest["preprocessing"].update(zero_mean_meta)
    if filters:
        manifest["filters"] = filters
    output_manifest.write_text(json.dumps(manifest, indent=2))


def describe_gpu(device: str) -> dict[str, Any]:
    if pynvml is None:  # pragma: no cover - NVML optional
        return {"device": device, "utilization": None, "memory_utilization": None}
    try:
        pynvml.nvmlInit()
        if device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
        else:
            index = 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = float(memory.total) if memory.total else 1.0
        return {
            "device": device,
            "utilization": float(util.gpu),
            "memory_utilization": (float(memory.used) / total) * 100.0,
        }
    except Exception:
        return {"device": device, "utilization": None, "memory_utilization": None}


__all__ = [
    "VideoWorkItem",
    "DecodedVideo",
    "ShardSpec",
    "PlanSummary",
    "build_plan",
    "write_plan",
    "load_plan_metadata",
    "iter_plan_items",
    "tensor_filename",
    "resolve_manifest_root",
    "resolve_asset_path",
    "decode_work_item",
    "embed_decoded_video",
    "embed_video_item",
    "estimate_embedding_mean",
    "embed_work_items",
    "save_shard_manifest",
    "merge_manifests",
    "parse_device_list",
    "build_device_assignments",
    "describe_gpu",
]
