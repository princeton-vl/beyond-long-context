from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataclasses import dataclass, field

from vidgeom import load_template
from vidgeom.rng import derive_seed
from rule_discovery.simulate import LZCausalCache, build_lz_causal_cache
from .template_utils import TemplateOverrideManager
from .simple_letter_renderer import (
    LetterRenderPlan,
    build_letter_render_plan,
    create_letter_plan,
    encode_frames_to_mp4,
    render_sequence_frames,
    render_slice_frames,
)

QUESTION_MODE_EXISTS = "exists"
QUESTION_MODE_CONTINUATION = "continuation"
QUESTION_MODES = {QUESTION_MODE_EXISTS, QUESTION_MODE_CONTINUATION}
LOG2E = math.log2(math.e)

SPRITE_NAME_MAP = {
    "icon_000": "skull",
    "icon_001": "green fireball",
    "icon_003": "speech bubble",
    "icon_005": "two pink hearts",
    "icon_006": "sleep icon",
    "icon_007": "lightning trio",
    "icon_008": "flaming skull",
    "icon_009": "water droplets",
    "icon_010": "anatomical heart",
    "icon_011": "lungs",
    "icon_012": "stomach",
    "icon_013": "brain",
    "icon_014": "flexed bicep",
    "icon_015": "green arrow up",
    "icon_016": "red arrow down",
    "icon_025": "arrow-pierced skull",
    "icon_029": "armor",
    "icon_031": "ring of fire",
    "icon_040": "campfire",
    "icon_041": "tent",
    "icon_042": "hammer and anvil",
    "icon_058": "axe",
}


@dataclass
class LetterRenderData:
    plan: LetterRenderPlan
    seed: int
    times: Dict[str, List[float]]
    asset_plan: Dict[str, Any]
    primary_seq: str
    tokens_main: List[str]
    lane_seq: List[str]


def _prepare_letter_rendering(
    template_path: Path,
    job_id: str,
    sequences: Dict[str, Sequence[str]],
    fps_override: Optional[int] = None,
) -> LetterRenderData:
    template = load_template(str(template_path))
    raw = template.raw
    render_cfg = raw.get("render", {}) or {}
    width, height = render_cfg.get("resolution", [448, 448])
    width = int(width)
    height = int(height)
    template_fps = int(render_cfg.get("fps", 10))
    fps = int(fps_override) if fps_override is not None else template_fps
    if fps <= 0:
        raise ValueError("fps must be > 0")
    scene_cfg = raw.get("scene", {}) or {}
    belts = int(scene_cfg.get("belts", 3))
    lane_pad = float(scene_cfg.get("lanes_pad", 0.05))
    vocab_cfg = raw.get("vocab", {}) or {}
    letter_cfg = (vocab_cfg.get("token_letters") or {})
    base_seed = int(raw.get("seed", 12345))
    seed_to_use = derive_seed(base_seed, job_id)
    declared_tokens = {
        str(tok)
        for tok in (vocab_cfg.get("token_ids") or [])
        if str(tok).strip()
    }
    observed_tokens = {str(tok) for seq in sequences.values() for tok in seq}
    unique_tokens = sorted(observed_tokens | declared_tokens)
    letter_rng = np.random.default_rng(derive_seed(seed_to_use, "token_letters"))
    letter_plan = create_letter_plan(unique_tokens, letter_cfg, letter_rng)
    render_plan = build_letter_render_plan(width, height, fps, belts, lane_pad, letter_plan)
    seq_names = list(sequences.keys())
    if not seq_names:
        raise ValueError("No sequences provided for rendering")
    primary_seq = seq_names[0]
    tokens_main = [str(tok) for tok in sequences[primary_seq]]
    lane_seq: List[str] = []
    for name in seq_names[1:]:
        if "lane" in name.lower():
            lane_seq = [str(tok) for tok in sequences[name]]
            break
    if not lane_seq:
        lane_seq = ["0"] * len(tokens_main)
    if len(lane_seq) < len(tokens_main):
        lane_seq.extend([lane_seq[-1] if lane_seq else "0"] * (len(tokens_main) - len(lane_seq)))
    frame_duration = render_plan.frame_duration
    times = {
        name: [idx * frame_duration for idx in range(len(seq))]
        for name, seq in sequences.items()
    }
    asset_plan = {"token_letters": letter_plan}
    return LetterRenderData(
        plan=render_plan,
        seed=seed_to_use,
        times=times,
        asset_plan=asset_plan,
        primary_seq=primary_seq,
        tokens_main=tokens_main,
        lane_seq=lane_seq,
    )


def _contains_subsequence(seq: Sequence[str], subseq: Sequence[str]) -> bool:
    n = len(subseq)
    if n == 0 or n > len(seq):
        return False
    for i in range(len(seq) - n + 1):
        if list(seq[i : i + n]) == list(subseq):
            return True
    return False


def _contains_joint_subsequence(
    seq_map: Dict[str, Sequence[str]],
    seq_names: Sequence[str],
    slices: Dict[str, Sequence[str]],
    end_idx: Optional[int] = None,
) -> bool:
    """Return True if the joint slices appear with all sequences aligned."""
    if not seq_names:
        return False
    primary = seq_names[0]
    subseq = slices.get(primary, [])
    length = len(subseq)
    if length == 0:
        return False
    seq_len = len(seq_map.get(primary, []))
    max_end = seq_len if end_idx is None else min(seq_len, end_idx + 1)
    if max_end < length:
        return False
    limit = max_end - length + 1
    for start in range(limit):
        if all(seq_map[name][start : start + length] == list(slices[name]) for name in seq_names):
            return True
    return False


def _ngram_counts(
    seq_tokens: Sequence[str],
    min_len: int = 2,
    max_len: int = 6,
) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    n_tokens = len(seq_tokens)
    for n in range(min_len, max_len + 1):
        if n > n_tokens:
            break
        for i in range(0, n_tokens - n + 1):
            g = tuple(seq_tokens[i : i + n])
            counts[g] = counts.get(g, 0) + 1
    return counts


def _subseq_likelihood_from_counts(seq_len: int, counts: Dict[Tuple[str, ...], int], subseq: Sequence[str]) -> float:
    n = len(subseq)
    if n == 0 or n > seq_len:
        return 0.0
    total = seq_len - n + 1
    if total <= 0:
        return 0.0
    c = counts.get(tuple(subseq), 0)
    return c / float(total)


def _sprite_names_from_plan(plan: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not plan:
        return {}
    sprite_map = plan.get("map", {}) or {}
    names: Dict[str, str] = {}
    for token, sprite_id in sprite_map.items():
        label = SPRITE_NAME_MAP.get(sprite_id, sprite_id)
        names[str(token)] = label
    return names


def _token_labels_from_asset_plan(plan: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not plan:
        return {}
    labels = _sprite_names_from_plan(plan.get("token_sprites") or {})
    letter_plan = plan.get("token_letters") or {}
    for token, letter in (letter_plan.get("map", {}) or {}).items():
        clean = str(letter or "").strip()
        if not clean:
            clean = str(token)
        labels[str(token)] = f"letter {clean.upper()}"
    return labels


def _describe_sequence(tokens: Sequence[str], lanes: Sequence[str], name_lookup: Dict[str, str]) -> str:
    parts: List[str] = []
    total = len(tokens)
    for idx, tok in enumerate(tokens):
        label = name_lookup.get(str(tok), f"token {tok}")
        lane = lanes[idx] if idx < len(lanes) else "?"
        if idx == 0:
            prefix = "First"
        elif idx == total - 1:
            prefix = "Finally"
        else:
            prefix = "Then"
        parts.append(f"{prefix} {label} travels down conveyor {lane}")
    if not parts:
        return "(no tokens)"
    return ". ".join(parts) + "."


def _pick_positions(seq_len: int, num_questions: int, rng: np.random.Generator, min_len: int) -> List[int]:
    """Pick question positions uniformly from the valid prefix range."""
    if seq_len < min_len:
        raise ValueError(f"Sequence length {seq_len} shorter than required min question length {min_len}")
    start = max(0, min_len - 1)
    positions: List[int] = []
    for _ in range(num_questions):
        pos = int(rng.integers(start, seq_len))
        positions.append(pos)
    return positions


def _durations(time_list: List[float]) -> List[float]:
    if len(time_list) < 2:
        return [0.5]
    diffs = [time_list[i + 1] - time_list[i] for i in range(len(time_list) - 1)]
    if any(d <= 0 for d in diffs):
        raise RuntimeError("Non-increasing token times detected")
    tail = float(np.median(diffs))
    return diffs + [tail]


def _tokens_to_int_array(tokens: Sequence[str]) -> np.ndarray:
    return np.array([int(t) for t in tokens], dtype=np.int64)


def _build_entropy_cache_map(seq_map: Dict[str, List[str]]) -> Dict[str, LZCausalCache]:
    return {name: build_lz_causal_cache(_tokens_to_int_array(tokens)) for name, tokens in seq_map.items()}


def _render_sequence_to_video(
    template_path: Path,
    job_id: str,
    sequences: Dict[str, Sequence[str]],
    out_dir: Path,
    fps_override: Optional[int] = None,
    ffmpeg_crf: int = 23,
    ffmpeg_preset: str = "veryfast",
    ffmpeg_codec: str = "libx264",
    capture_frame_debug: bool = False,
) -> Tuple[List[str], Dict[str, List[float]], int, Dict[str, Any], List[Dict[str, Any]], LetterRenderPlan]:
    prep = _prepare_letter_rendering(template_path, job_id, sequences, fps_override)
    frames, frame_meta = render_sequence_frames(prep.plan, prep.tokens_main, prep.lane_seq)
    if not frames:
        raise RuntimeError("Cannot render an empty sequence")
    video_path = out_dir / f"{job_id}_v0.mp4"
    encode_frames_to_mp4(
        frames,
        prep.plan.fps,
        str(video_path),
        crf=ffmpeg_crf,
        preset=ffmpeg_preset,
        codec=ffmpeg_codec,
    )
    frame_records = frame_meta if capture_frame_debug else []
    return [str(video_path)], prep.times, prep.seed, prep.asset_plan, frame_records, prep.plan


def _load_sequence_entries(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    seq_entries = data.get("sequences", [])
    if not seq_entries:
        raise ValueError(f"No sequences found in sequences file {path}")
    return seq_entries


def _slice_time_bounds(ctx: QuestionContext, start_idx: int, length: int) -> Tuple[float, float]:
    if start_idx < 0 or start_idx + length > len(ctx.tokens_main):
        raise RuntimeError("Slice indices out of bounds for clip extraction")
    if not ctx.render_plan:
        raise RuntimeError("Render plan is required for clip extraction")
    start_time = ctx.times_main[start_idx]
    end_time = start_time + length * ctx.render_plan.frame_duration
    return start_time, end_time


def _render_true_slice_clip(
    ctx: QuestionContext,
    idx_q: int,
    clip_suffix: str,
    start_idx: int,
    length: int,
) -> Tuple[Path, float, float]:
    if not ctx.render_plan:
        raise RuntimeError("Render plan is required for clip generation")
    if start_idx < 0 or start_idx + length > len(ctx.tokens_main):
        raise RuntimeError("Slice indices out of bounds for clip extraction")
    true_slices = {
        name: ctx.seq_map[name][start_idx : start_idx + length]
        for name in ctx.seq_names
    }
    tokens = list(true_slices.get(ctx.primary_seq_name, []))
    if len(tokens) != length:
        raise RuntimeError("True slice length mismatch")
    lanes = ctx.lane_values_for_slices(true_slices, length)
    frames, frame_meta = render_slice_frames(ctx.render_plan, tokens, lanes)
    clip_path = ctx.clips_dir / f"{ctx.video_id}_q{idx_q}_{clip_suffix}.mp4"
    encode_frames_to_mp4(
        frames,
        ctx.render_plan.fps,
        str(clip_path),
        crf=ctx.ffmpeg_crf,
        preset=ctx.ffmpeg_preset,
        codec=ctx.ffmpeg_codec,
    )
    clip_start, clip_end = _slice_time_bounds(ctx, start_idx, length)
    if ctx.capture_frame_debug:
        records = []
        use_absolute = False
        if ctx.frame_debug:
            records = ctx.frame_debug[start_idx : start_idx + length]
            use_absolute = True
        if not records:
            records = frame_meta
            use_absolute = False
        _write_clip_frame_debug(
            ctx,
            clip_path,
            clip_start=clip_start,
            clip_end=clip_end,
            frame_records=records,
            use_absolute=use_absolute,
        )
    return clip_path, clip_start, clip_end


def _render_fake_slice_clip(
    ctx: QuestionContext,
    idx_q: int,
    clip_suffix: str,
    candidate_slices: Dict[str, List[str]],
    start_slice: int,
    length: int,
) -> Tuple[Path, float, float]:
    if not ctx.render_plan:
        raise RuntimeError("Render plan is required for clip generation")
    tokens = list(candidate_slices.get(ctx.primary_seq_name, []))
    if len(tokens) != length:
        raise RuntimeError("Candidate slice length mismatch")
    lanes = ctx.lane_values_for_slices(candidate_slices, length)
    frames, frame_meta = render_slice_frames(ctx.render_plan, tokens, lanes)
    clip_path = ctx.clips_dir / f"{ctx.video_id}_q{idx_q}_{clip_suffix}.mp4"
    encode_frames_to_mp4(
        frames,
        ctx.render_plan.fps,
        str(clip_path),
        crf=ctx.ffmpeg_crf,
        preset=ctx.ffmpeg_preset,
        codec=ctx.ffmpeg_codec,
    )
    clip_start, clip_end = _slice_time_bounds(ctx, start_slice, length)
    if ctx.capture_frame_debug:
        _write_clip_frame_debug(
            ctx,
            clip_path,
            clip_start=clip_start,
            clip_end=clip_end,
            frame_records=frame_meta,
        )
    return clip_path, clip_start, clip_end


def _token_letter_lookup(ctx: QuestionContext) -> Dict[str, str]:
    plan = ctx.token_letters or {}
    raw = plan.get("map", {}) if isinstance(plan, dict) else {}
    return {str(tok): str(letter) for tok, letter in (raw or {}).items() if str(letter).strip()}


def _split_debug_items(
    items: Sequence[Dict[str, Any]],
    letter_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    visible_items: List[Dict[str, Any]] = []
    offscreen_items: List[Dict[str, Any]] = []
    for item in items or []:
        data = dict(item)
        token = str(data.get("token"))
        data["token"] = token
        letter = letter_map.get(token)
        if letter:
            data["letter"] = letter
        if data.get("visible"):
            visible_items.append(data)
        else:
            offscreen_items.append(data)
    return visible_items, offscreen_items


def _write_frame_debug(video_path: Path, ctx: QuestionContext) -> None:
    if not ctx.capture_frame_debug or not ctx.frame_debug:
        return
    letter_map = _token_letter_lookup(ctx)
    frames: List[Dict[str, Any]] = []
    view_bounds = None
    for record in ctx.frame_debug:
        visible, offscreen = _split_debug_items(record.get("items", []), letter_map)
        entry = {
            "frame_index": int(record.get("frame_index", -1)),
            "time": float(record.get("time", 0.0)),
            "items": visible,
        }
        if offscreen or record.get("offscreen_items"):
            entry["offscreen_items"] = offscreen
        if "view_bounds" in record and view_bounds is None:
            view_bounds = dict(record.get("view_bounds", {}))
        frames.append(entry)
    payload = {
        "video": str(video_path),
        "fps": ctx.video_fps or 30,
        "frames": frames,
    }
    if view_bounds:
        payload["view_bounds"] = view_bounds
    video_path.with_suffix(".frames.json").write_text(json.dumps(payload, indent=2))


def _write_clip_frame_debug(
    ctx: QuestionContext,
    clip_path: Path,
    clip_start: float,
    clip_end: float,
    frame_records: Optional[Sequence[Dict[str, Any]]] = None,
    use_absolute: bool = False,
) -> None:
    if not ctx.capture_frame_debug:
        return
    records = frame_records if frame_records is not None else ctx.frame_debug
    if not records:
        return
    letter_map = _token_letter_lookup(ctx)
    frames: List[Dict[str, Any]] = []
    eps = 1e-6
    use_relative = frame_records is not None and not use_absolute
    for record in records:
        t = float(record.get("time", 0.0))
        if use_relative:
            rel_time = t
            video_frame_index = None
            video_time = clip_start + t
        else:
            if t < clip_start - eps or t > clip_end + eps:
                continue
            rel_time = t - clip_start
            video_frame_index = record.get("frame_index")
            video_time = t
        visible, offscreen = _split_debug_items(record.get("items", []), letter_map)
        entry = {
            "clip_frame_index": len(frames),
            "video_frame_index": video_frame_index,
            "video_time": video_time,
            "time": rel_time,
            "items": visible,
        }
        if offscreen:
            entry["offscreen_items"] = offscreen
        frames.append(entry)
    if not frames:
        return
    payload = {
        "clip": str(clip_path),
        "fps": ctx.video_fps or 30,
        "offset_start": clip_start,
        "frames": frames,
    }
    clip_path.with_suffix(".frames.json").write_text(json.dumps(payload, indent=2))


def _resolve_media_path(reference: str, bucket_root: Path) -> Path:
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = (bucket_root.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _media_record(path_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "size_bytes": size, "sha256": h.hexdigest()}


def _write_validation_manifest(out_dir: Path, videos_payload: Sequence[Dict[str, Any]], log_progress: bool = False) -> None:
    records: List[Dict[str, Any]] = []
    for entry in videos_payload:
        media_info = _media_record(entry.get("video_path")) if entry.get("video_path") else None
        questions_info: List[Dict[str, Any]] = []
        for question in entry.get("questions", []):
            candidate = question.get("candidate", {})
            clip_info = _media_record(candidate.get("clip_path")) if candidate.get("clip_path") else None
            questions_info.append(
                {
                    "question_index": question.get("question_index"),
                    "question_variant": question.get("question_variant"),
                    "clip": clip_info,
                }
            )
        records.append(
            {
                "video_index": entry.get("video_index"),
                "variant": entry.get("variant"),
                "media": media_info,
                "questions": questions_info,
            }
        )
    manifest = {"videos": records}
    path = out_dir / "validation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    if log_progress:
        print(f"[validate] wrote {path}", flush=True)


def _clip_frames_match_video(
    clip_frames: Sequence[Dict[str, Any]],
    video_frames: Sequence[Dict[str, Any]],
) -> bool:
    lookup = {frame.get("frame_index"): frame for frame in video_frames}
    for entry in clip_frames:
        idx = entry.get("video_frame_index")
        if idx is None:
            continue
        base = lookup.get(idx)
        if base is None:
            return False
        if entry.get("items", []) != base.get("items", []):
            return False
    return True


@dataclass
class QuestionContext:
    template_path: Path
    video_id: str
    seq_map: Dict[str, List[str]]
    seq_names: List[str]
    entropy_cache: Dict[str, LZCausalCache]
    counts_per_seq: Dict[str, Dict[Tuple[str, ...], int]]
    times_main: List[float]
    durations_main: List[float]
    median_step: float
    video_end_time: float
    video_paths: List[str]
    clips_dir: Path
    clip_options: bool
    questions_only: bool
    questions_at_end: bool
    uniform_uncertain: bool
    rng: np.random.Generator
    question_min_len: int
    ngram_max: int
    fps_override: Optional[int]
    ffmpeg_crf: int
    ffmpeg_preset: str
    ffmpeg_codec: str
    video_job_seed: Optional[int] = None
    token_letters: Dict[str, Any] = field(default_factory=dict)
    hard_questions: bool = False
    vocab_upper: Dict[str, int] = field(default_factory=dict)
    primary_seq_name: str = field(init=False)
    secondary_seq_names: List[str] = field(init=False)
    lane_seq_name: Optional[str] = field(init=False)
    text_options: bool = False
    token_labels: Dict[str, str] = field(default_factory=dict)
    video_fps: Optional[int] = None
    capture_frame_debug: bool = False
    frame_debug: List[Dict[str, Any]] = field(default_factory=list)
    render_plan: Optional[LetterRenderPlan] = None

    def __post_init__(self) -> None:
        if not self.seq_names:
            raise ValueError("QuestionContext requires at least one sequence")
        self.primary_seq_name = self.seq_names[0]
        self.secondary_seq_names = list(self.seq_names[1:])
        self.lane_seq_name = next((name for name in self.secondary_seq_names if "lane" in name.lower()), None)
        self.tokens_main = self.seq_map[self.seq_names[0]]
        self.total_tokens = len(self.tokens_main)
        self.vocab_max = {
            name: max(int(t) for t in tokens) if tokens else 1
            for name, tokens in self.seq_map.items()
        }
        self.vocab_limits: Dict[str, int] = {}
        for name in self.seq_map:
            observed = self.vocab_max.get(name, 0)
            default_upper = observed + 1 if observed >= 0 else 1
            override = int(self.vocab_upper.get(name, 0)) if name in self.vocab_upper else 0
            limit = max(default_upper, override)
            if limit <= 0:
                limit = 1
            self.vocab_limits[name] = limit

    def vocab_limit(self, name: str) -> int:
        return max(1, self.vocab_limits.get(name, 1))

    def prefix_entropy_bits(self, prefix_len: int) -> Dict[str, float]:
        prefix_len = max(0, prefix_len)
        return {name: cache.entropy_bits(prefix_len) for name, cache in self.entropy_cache.items()}

    def make_exists_question(
        self, idx_q: int, pos: int, max_stat_n: int, spatial: bool
    ) -> Optional[Dict[str, Any]]:
        return _build_exists_question(self, idx_q, pos, max_stat_n, spatial=spatial)

    def make_continuation_question(self, idx_q: int, spatial: bool) -> Optional[Dict[str, Any]]:
        return _build_continuation_question(self, idx_q, spatial=spatial)

    def presence_sequences(self, spatial: bool) -> List[str]:
        if spatial and self.secondary_seq_names and self.hard_questions:
            return self.secondary_seq_names
        # Non-hard questions (and all sequential ones) only enforce presence/absence
        # on the primary token stream.
        return [self.primary_seq_name]

    def lane_values_for_slices(
        self,
        slices: Dict[str, Sequence[str]],
        length: int,
    ) -> List[str]:
        if self.lane_seq_name and self.lane_seq_name in slices:
            lanes = [str(val) for val in slices[self.lane_seq_name]]
        else:
            lanes = []
        if len(lanes) < length:
            fill = lanes[-1] if lanes else "0"
            lanes.extend([fill] * (length - len(lanes)))
        return lanes[:length]


def _build_exists_question(
    ctx: QuestionContext, idx_q: int, pos: int, max_stat_n: int, spatial: bool = False
) -> Optional[Dict[str, Any]]:
    tokens_main = ctx.tokens_main
    rng = ctx.rng
    spatial_active = spatial and bool(ctx.secondary_seq_names)
    enforce_absent = True
    max_question_tokens = max(1, len(tokens_main) - 1)
    max_len_allowed = min(6, max_question_tokens, pos + 1, max_stat_n)
    min_required = ctx.question_min_len
    if spatial_active:
        min_required = max(min_required, 4)
    if max_len_allowed < min_required:
        raise RuntimeError(
            f"Not enough prefix to ask length>={min_required} at position {pos}"
        )
    if max_len_allowed == min_required:
        length = min_required
    else:
        length = int(rng.integers(min_required, max_len_allowed + 1))
    max_start_pos = max(0, min(pos - length + 1, len(tokens_main) - length))
    start_true = int(rng.integers(0, max_start_pos + 1)) if max_start_pos >= 0 else 0
    true_slices = {name: ctx.seq_map[name][start_true : start_true + length] for name in ctx.seq_names}
    presence_seq_names = ctx.presence_sequences(spatial_active)
    primary_vocab_size = max(1, ctx.vocab_limit(ctx.primary_seq_name))
    primary_token_pool = [str(i) for i in range(primary_vocab_size)]
    lane_value_pool = {
        name: sorted({str(val) for val in ctx.seq_map[name]})
        for name in ctx.secondary_seq_names
    }
    if ctx.render_plan and ctx.render_plan.belts > 0:
        default_lane_values = [str(i) for i in range(ctx.render_plan.belts)]
    else:
        observed = sorted({str(val) for values in lane_value_pool.values() for val in values})
        default_lane_values = observed or ["0"]
    if ctx.render_plan and ctx.render_plan.belts > 0:
        default_lane_values = [str(i) for i in range(ctx.render_plan.belts)]
    else:
        observed = sorted({str(val) for values in lane_value_pool.values() for val in values})
        default_lane_values = observed or ["0"]

    def _random_lane_sequence(name: str, seq_len: int) -> List[str]:
        pool = lane_value_pool.get(name)
        values = pool if pool else default_lane_values
        if not values:
            values = ["0"]
        return [str(values[int(rng.integers(0, len(values)))]) for _ in range(seq_len)]

    def _random_primary_sequence(seq_len: int) -> List[str]:
        if seq_len <= 0:
            return []
        pool = primary_token_pool or ["0"]
        return [str(pool[int(rng.integers(0, len(pool)))]) for _ in range(seq_len)]

    if spatial_active:
        max_fake_attempts = 60
    elif enforce_absent:
        max_fake_attempts = 80
    else:
        max_fake_attempts = 20

    def _is_duplicate_candidate(
        candidate: Dict[str, List[str]],
        existing_fake_slices: Optional[Sequence[Dict[str, List[str]]]],
    ) -> bool:
        if not existing_fake_slices:
            return False
        for other in existing_fake_slices:
            if all(candidate.get(name, []) == other.get(name, []) for name in ctx.seq_names):
                return True
        return False

    def _spatial_candidate_valid(
        candidate: Dict[str, List[str]],
        existing_fake_slices: Optional[Sequence[Dict[str, List[str]]]],
    ) -> bool:
        if any(candidate.get(nm, []) == true_slices[nm] for nm in ctx.secondary_seq_names):
            return False
        if _is_duplicate_candidate(candidate, existing_fake_slices):
            return False
        return True

    def _sample_fake_slices_hard(
        existing_fake_slices: Optional[Sequence[Dict[str, List[str]]]] = None,
        max_attempts: int = max_fake_attempts,
    ) -> Optional[Dict[str, Any]]:
        # First attempt biased sampling/mutations and, if those fail, fall back to
        # uniform sampling that still enforces the "not yet seen" constraint.
        existing_fake_slices = existing_fake_slices or []
        hybrid_threshold = max(1, max_attempts // 2)
        for attempt_idx in range(max_attempts):
            if spatial_active:
                candidate = {name: list(true_slices[name]) for name in ctx.seq_names}
                for nm in ctx.secondary_seq_names:
                    candidate[nm] = _random_lane_sequence(nm, length)
                if not _spatial_candidate_valid(candidate, existing_fake_slices):
                    continue
                if any(
                    candidate.get(nm, []) == true_slices[nm] for nm in ctx.secondary_seq_names
                ):
                    continue
                start_fake = start_true
            else:
                start_fake = int(rng.integers(0, len(tokens_main) - length + 1))
                base_slices = {
                    name: ctx.seq_map[name][start_fake : start_fake + length] for name in ctx.seq_names
                }
                candidate = {name: list(base_slices[name]) for name in ctx.seq_names}
                candidate[ctx.primary_seq_name] = _random_primary_sequence(length)
                if any(candidate[nm] == true_slices[nm] for nm in presence_seq_names):
                    continue
                if enforce_absent:
                    present_here = any(
                        _contains_subsequence(ctx.seq_map[nm][: pos + 1], candidate[nm])
                        for nm in presence_seq_names
                    )
                    if present_here:
                        continue
            return {"slices": candidate, "start": start_fake}

        return None

    def _candidate_present(candidate: Dict[str, List[str]]) -> bool:
        if spatial_active:
            return all(
                candidate.get(nm, []) == true_slices[nm]
                for nm in ctx.secondary_seq_names
            )
        primary = ctx.primary_seq_name
        seq_tokens = candidate.get(primary, [])
        if pos < 0:
            return False
        prefix = ctx.seq_map[primary][: pos + 1]
        return _contains_subsequence(prefix, seq_tokens)

    def _sample_fake_slices_easy(
        existing_fake_slices: Optional[Sequence[Dict[str, List[str]]]] = None,
        max_attempts: int = max_fake_attempts,
    ) -> Optional[Dict[str, Any]]:
        existing_fake_slices = existing_fake_slices or []
        span = max(1, len(tokens_main) - length + 1)
        for _ in range(max_attempts):
            candidate: Dict[str, List[str]] = {}
            start_fake = int(rng.integers(0, span))
            base_slices = {
                name: ctx.seq_map[name][start_fake : start_fake + length]
                for name in ctx.seq_names
            }
            for name in ctx.seq_names:
                if name in ctx.secondary_seq_names:
                    if spatial_active:
                        candidate[name] = _random_lane_sequence(name, length)
                    else:
                        candidate[name] = list(true_slices[name])
                elif spatial_active:
                    candidate[name] = list(true_slices[name])
                else:
                    candidate[name] = _random_primary_sequence(length)
            if _is_duplicate_candidate(candidate, existing_fake_slices):
                continue
            if any(candidate[name] == true_slices[name] for name in presence_seq_names):
                continue
            if spatial_active and any(
                candidate.get(nm, []) == true_slices[nm] for nm in ctx.secondary_seq_names
            ):
                continue
            if enforce_absent and _candidate_present(candidate):
                continue
            return {"slices": candidate, "start": start_fake}
        return None

    def _sample_fake_slices(
        existing_fake_slices: Optional[Sequence[Dict[str, List[str]]]] = None,
        max_attempts: int = max_fake_attempts,
    ) -> Optional[Dict[str, Any]]:
        if ctx.hard_questions:
            return _sample_fake_slices_hard(existing_fake_slices, max_attempts)
        cand = _sample_fake_slices_easy(existing_fake_slices, max_attempts)
        if cand is None:
            cand = _sample_fake_slices_hard(existing_fake_slices, max_attempts)
        return cand

    def _sample_single_fake() -> Optional[Dict[str, Any]]:
        cand = _sample_fake_slices()
        if cand is None and spatial_active:
            return None
        return cand

    fake_candidate = _sample_single_fake()
    if fake_candidate is None:
        return None

    answer_is_true = bool(ctx.rng.integers(0, 2))
    candidate_spec = {"slices": true_slices, "start": start_true}
    if not answer_is_true:
        candidate_spec = fake_candidate

    qt = ctx.video_end_time if ctx.questions_at_end else ctx.times_main[pos]
    clip_start_time: Optional[float] = None
    clip_end_time: Optional[float] = None
    clip_path_str = ""
    variant_label = "spatial" if spatial_active else "sequential"
    clip_suffix_template = "{}_{}".format("{}", variant_label)
    if ctx.clip_options and not ctx.questions_only:
        if not ctx.capture_frame_debug:
            raise RuntimeError("clip_options requires capture_frame_debug=True")
        clip_suffix = clip_suffix_template.format("true" if answer_is_true else "false")
        if answer_is_true:
            clip_path, clip_start_time, clip_end_time = _render_true_slice_clip(
                ctx,
                idx_q,
                clip_suffix,
                candidate_spec["start"],
                length,
            )
        else:
            clip_path, clip_start_time, clip_end_time = _render_fake_slice_clip(
                ctx,
                idx_q,
                clip_suffix,
                candidate_spec["slices"],
                candidate_spec["start"],
                length,
            )
        clip_path_str = str(clip_path)

    candidate_entry = {
        "sequence": list(candidate_spec["slices"].get(ctx.primary_seq_name, [])),
        "sequences": candidate_spec["slices"],
        "clip_path": clip_path_str,
        "clip_start": clip_start_time,
        "clip_end": clip_end_time,
        "present": answer_is_true,
    }

    question = {
        "question": (
            "Did this sequence appear anywhere in the video?"
            if ctx.questions_at_end
            else "Did this sequence appear in the video so far?"
        ),
        "question_mode": QUESTION_MODE_EXISTS,
        "question_format": "binary_yes_no",
        "answer": "yes" if answer_is_true else "no",
        "candidate": candidate_entry,
        "scenario": "single_true",
        "question_index": pos,
        "question_time": qt,
        "clip_start_time": clip_start_time,
        "clip_end_time": clip_end_time,
        "entropy_prefix": ctx.prefix_entropy_bits(pos + 1),
        "asked_after_video": ctx.questions_at_end,
        "has_unique_answer": True,
        "question_variant": variant_label,
        "question_type": variant_label,
    }
    return question


def _build_continuation_question(
    ctx: QuestionContext, idx_q: int, spatial: bool = False
) -> Optional[Dict[str, Any]]:
    rng = ctx.rng
    spatial_active = spatial and bool(ctx.secondary_seq_names)
    presence_seq_names = ctx.presence_sequences(spatial_active)
    prefix_len = answer_len = 4
    max_prefix_start = ctx.total_tokens - (prefix_len + answer_len)
    if max_prefix_start < 0:
        raise RuntimeError("Not enough tokens for continuation question")
    prefix_start = int(rng.integers(0, max_prefix_start + 1))
    prefix_slices = {
        name: ctx.seq_map[name][prefix_start : prefix_start + prefix_len]
        for name in ctx.seq_names
    }
    continuations = _collect_continuations(ctx, prefix_slices, prefix_len, answer_len)
    if not continuations:
        raise RuntimeError("Could not find valid continuation prefix")

    rng.shuffle(continuations)
    true_start, true_slices = continuations[0]
    target_start = prefix_start + prefix_len
    for start, slices in continuations:
        if start == target_start:
            true_start, true_slices = start, slices
            break

    key_names = ctx.seq_names if spatial_active else (presence_seq_names or ctx.seq_names)
    key_fn = lambda item: tuple(tuple(item[1][name]) for name in key_names)
    real_set = {key_fn(item) for item in continuations}
    unique_flag = len(real_set) == 1

    primary_vocab_size = max(1, ctx.vocab_limit(ctx.primary_seq_name))
    primary_token_pool = [str(i) for i in range(primary_vocab_size)]
    lane_value_pool = {
        name: sorted({str(val) for val in ctx.seq_map[name]})
        for name in ctx.secondary_seq_names
    }

    def _random_primary_sequence(seq_len: int) -> List[str]:
        if seq_len <= 0:
            return []
        pool = primary_token_pool or ["0"]
        return [str(pool[int(rng.integers(0, len(pool)))]) for _ in range(seq_len)]

    def _random_lane_sequence(name: str, seq_len: int) -> List[str]:
        pool = lane_value_pool.get(name)
        values = pool if pool else default_lane_values
        if not values:
            values = ["0"]
        return [str(values[int(rng.integers(0, len(values)))]) for _ in range(seq_len)]

    def _sample_fake_candidate() -> Optional[Dict[str, Any]]:
        attempts = 0
        while attempts < 30:
            attempts += 1
            if spatial_active:
                candidate = {name: list(true_slices[name]) for name in ctx.seq_names}
                for nm in ctx.secondary_seq_names:
                    candidate[nm] = _random_lane_sequence(nm, answer_len)
            else:
                candidate = {name: list(true_slices[name]) for name in ctx.seq_names}
                candidate[ctx.primary_seq_name] = _random_primary_sequence(answer_len)
            key = tuple(tuple(candidate[name]) for name in key_names)
            if key in real_set:
                continue
            return {"slices": candidate, "start": true_start}
        return None

    fake_spec = _sample_fake_candidate()
    if fake_spec is None:
        return None

    qa_end_idx = true_start + answer_len - 1
    qt = ctx.video_end_time if ctx.questions_at_end else ctx.times_main[qa_end_idx]
    answer_is_true = bool(ctx.rng.integers(0, 2))
    variant_label = "spatial" if spatial_active else "sequential"

    prefix_clip_path = ""
    prefix_clip_start: Optional[float] = None
    prefix_clip_end: Optional[float] = None
    if ctx.clip_options and not ctx.questions_only:
        if not ctx.capture_frame_debug:
            raise RuntimeError("clip_options requires capture_frame_debug=True")
        clip_path, clip_start, clip_end = _render_true_slice_clip(
            ctx,
            idx_q,
            "prefix",
            prefix_start,
            prefix_len,
        )
        prefix_clip_path = str(clip_path)
        prefix_clip_start = clip_start
        prefix_clip_end = clip_end

    candidate_spec = {"slices": true_slices, "start": true_start}
    if not answer_is_true:
        candidate_spec = fake_spec

    clip_start_time: Optional[float] = None
    clip_end_time: Optional[float] = None
    clip_path_str = ""
    clip_suffix_template = "{}_{}".format("{}", variant_label)
    if ctx.clip_options and not ctx.questions_only:
        if not ctx.capture_frame_debug:
            raise RuntimeError("clip_options requires capture_frame_debug=True")
        clip_suffix = clip_suffix_template.format("true" if answer_is_true else "false")
        if answer_is_true:
            clip_path, clip_start_time, clip_end_time = _render_true_slice_clip(
                ctx,
                idx_q,
                clip_suffix,
                candidate_spec["start"],
                answer_len,
            )
        else:
            clip_path, clip_start_time, clip_end_time = _render_fake_slice_clip(
                ctx,
                idx_q,
                clip_suffix,
                candidate_spec["slices"],
                candidate_spec["start"],
                answer_len,
            )
        clip_path_str = str(clip_path)

    candidate_entry = {
        "sequence": list(candidate_spec["slices"].get(ctx.primary_seq_name, [])),
        "sequences": candidate_spec["slices"],
        "clip_path": clip_path_str,
        "clip_start": clip_start_time,
        "clip_end": clip_end_time,
        "present": answer_is_true,
    }

    question = {
        "question": f"Does this sequence follow {prefix_slices[ctx.seq_names[0]]}?",
        "question_mode": QUESTION_MODE_CONTINUATION,
        "question_format": "binary_yes_no",
        "prefix": prefix_slices,
        "prefix_clip_path": prefix_clip_path,
        "prefix_clip_start": prefix_clip_start,
        "prefix_clip_end": prefix_clip_end,
        "answer": "yes" if answer_is_true else "no",
        "candidate": candidate_entry,
        "scenario": "single_true",
        "question_index": qa_end_idx,
        "question_time": qt,
        "clip_start_time": clip_start_time,
        "clip_end_time": clip_end_time,
        "asked_after_video": ctx.questions_at_end,
        "has_unique_answer": unique_flag,
        "question_variant": variant_label,
        "question_type": variant_label,
    }
    return question


def _collect_continuations(
    ctx: QuestionContext,
    prefix_slices: Dict[str, List[str]],
    prefix_len: int,
    answer_len: int,
) -> List[Tuple[int, Dict[str, List[str]]]]:
    continuations: List[Tuple[int, Dict[str, List[str]]]] = []
    total = ctx.total_tokens
    for start in range(0, total - (prefix_len + answer_len) + 1):
        if all(
            ctx.seq_map[name][start : start + prefix_len] == prefix_slices[name]
            for name in ctx.seq_names
        ):
            cont_start = start + prefix_len
            cont_slice = {
                name: ctx.seq_map[name][cont_start : cont_start + answer_len]
                for name in ctx.seq_names
            }
            continuations.append((cont_start, cont_slice))
    return continuations


def run_render(
    template_path: Path,
    sequences_file: Optional[Path],
    out_dir: Path,
    num_questions: int,
    log_progress: bool,
    clip_options: bool = False,
    questions_only: bool = False,
    questions_at_end: bool = False,
    ffmpeg_crf: int = 23,
    ffmpeg_preset: str = "veryfast",
    ffmpeg_codec: str = "libx264",
    target_seq_lens: Optional[List[int]] = None,
    max_videos: Optional[int] = None,
    question_min_len: int = 3,
    render_workers: int = 1,
    assignment_seed: Optional[int] = None,
    fps_override: Optional[int] = None,
    uniform_uncertain: bool = False,
    question_mode: str = QUESTION_MODE_EXISTS,
    hide_question_text: bool = False,
    sequence_sources: Optional[Dict[str, Path]] = None,
    spatial_question_fraction: float = 0.0,
    hard_questions: bool = False,
    capture_frame_debug: bool = False,
    validate_outputs: bool = True,
) -> None:
    if question_mode not in QUESTION_MODES:
        raise ValueError(f"Unknown question_mode={question_mode!r}")
    if question_min_len <= 0:
        raise ValueError("question_min_len must be > 0")
    if sequences_file is None and not sequence_sources:
        raise ValueError("Either sequences_file or sequence_sources must be provided")
    if spatial_question_fraction < 0.0 or spatial_question_fraction > 1.0:
        raise ValueError("spatial_question_fraction must be between 0 and 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    clips_dir = out_dir / "clips"
    videos_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    sequence_sources = sequence_sources or None

    override_mgr = TemplateOverrideManager()
    template_to_use = template_path
    if hide_question_text:
        template_to_use = override_mgr.build(
            template_path,
            hide_question_text=True,
        )

    template_raw = load_template(str(template_to_use)).raw
    seq_names = list(template_raw.get("sequences", []))
    if not seq_names:
        raise ValueError("Template must declare 'sequences:'; none found")
    render_cfg = template_raw.get("render", {}) or {}
    template_fps = int(render_cfg.get("fps", 30))
    effective_fps = int(fps_override) if fps_override is not None else template_fps

    vocab_upper: Dict[str, int] = {}
    vocab_cfg = template_raw.get("vocab") or {}
    token_ids = vocab_cfg.get("token_ids") or []
    if token_ids:
        try:
            max_token = max(int(t) for t in token_ids)
            vocab_upper[seq_names[0]] = max_token + 1
        except (TypeError, ValueError):
            pass

    if target_seq_lens:
        # allow single length to broadcast to all sequences
        if len(target_seq_lens) == 1 and len(seq_names) > 1:
            target_seq_lens = target_seq_lens * len(seq_names)
        if len(target_seq_lens) != len(seq_names):
            raise ValueError("target_seq_lens must match number of template sequences")

    job_entry_maps = _make_job_entry_maps(
        seq_names=seq_names,
        sequences_file=sequences_file,
        sequence_sources=sequence_sources,
        max_videos=max_videos,
        assignment_seed=assignment_seed,
    )

    if not job_entry_maps:
        raise ValueError("Not enough sequences to render a single video")

    total_videos = len(job_entry_maps)
    rng_master = np.random.default_rng(assignment_seed)

    jobs = [
        (vid_idx, entry_map, int(rng_master.integers(0, 2**32 - 1)))
        for vid_idx, entry_map in enumerate(job_entry_maps)
    ]

    videos_payload: List[Dict[str, Any]] = []
    if render_workers <= 1:
        for vid_idx, entry_map, seed in jobs:
            videos_payload.append(
                _render_job(
                    vid_idx,
                    entry_map,
                    seed,
                    template_to_use,
                    seq_names,
                    target_seq_lens,
                    num_questions,
                    clip_options,
                    questions_only,
                    questions_at_end,
                    ffmpeg_crf,
                    ffmpeg_preset,
                    ffmpeg_codec,
                    question_min_len,
                    videos_dir,
                    clips_dir,
                    log_progress,
                    total_videos,
                    fps_override,
                    uniform_uncertain,
                    question_mode,
                    spatial_question_fraction,
                    hard_questions,
                    effective_fps,
                    capture_frame_debug,
                    vocab_upper,
                )
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=render_workers) as ex:
            futs = {
                ex.submit(
                    _render_job,
                    vid_idx,
                    entry_map,
                    seed,
                    template_to_use,
                    seq_names,
                    target_seq_lens,
                    num_questions,
                    clip_options,
                    questions_only,
                    questions_at_end,
                    ffmpeg_crf,
                    ffmpeg_preset,
                    ffmpeg_codec,
                    question_min_len,
                    videos_dir,
                    clips_dir,
                    log_progress,
                    total_videos,
                    fps_override,
                    uniform_uncertain,
                    question_mode,
                    spatial_question_fraction,
                    hard_questions,
                    effective_fps,
                    capture_frame_debug,
                    vocab_upper,
                ): vid_idx
                for vid_idx, entry_map, seed in jobs
            }
            for fut in as_completed(futs):
                videos_payload.append(fut.result())

    videos_payload.sort(key=lambda x: x["video_index"])

    qa_payload = {
        "template": str(template_to_use),
        "videos": videos_payload,
        "sequences_file": str(sequences_file) if sequences_file else None,
        "questions_at_end": questions_at_end,
    }
    if sequence_sources:
        qa_payload["sequence_sources"] = {name: str(path) for name, path in sequence_sources.items()}
    qa_path = out_dir / "questions.json"
    qa_path.write_text(json.dumps(qa_payload, indent=2))
    if log_progress:
        print(f"[write] {qa_path}", flush=True)
    override_mgr.cleanup()


def _render_job(
    job_idx: int,
    entry_map: Dict[str, dict],
    seed: int,
    template_path: Path,
    seq_names: List[str],
    target_seq_lens: Optional[List[int]],
    num_questions: int,
    clip_options: bool,
    questions_only: bool,
    questions_at_end: bool,
    ffmpeg_crf: int,
    ffmpeg_preset: str,
    ffmpeg_codec: str,
    question_min_len: int,
    videos_dir: Path,
    clips_dir: Path,
    log_progress: bool,
    total_videos: int,
    fps_override: Optional[int],
    uniform_uncertain: bool,
    question_mode: str,
    spatial_question_fraction: float,
    hard_questions: bool,
    video_fps: int,
    capture_frame_debug: bool,
    vocab_upper: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    if vocab_upper is None:
        vocab_upper = {}
    if question_mode not in QUESTION_MODES:
        raise ValueError(f"Unknown question_mode={question_mode!r}")
    rng = np.random.default_rng(seed)
    seq_map: Dict[str, List[str]] = {}
    counts_per_seq: Dict[str, Dict[Tuple[str, ...], int]] = {}
    analytic_bits: Dict[str, float] = {}
    for idx_name, seq_name in enumerate(seq_names):
        if seq_name not in entry_map:
            raise ValueError(f"Missing sequence data for {seq_name!r} in video {job_idx+1}")
        entry = entry_map[seq_name]
        tokens = list(entry["tokens"])
        tgt_len = target_seq_lens[idx_name] if target_seq_lens else None
        if tgt_len is not None:
            if len(tokens) < tgt_len:
                raise ValueError(
                    f"Sequence {entry.get('seq_id', seq_name)} shorter than required length {tgt_len}"
                )
            tokens = tokens[:tgt_len]
        seq_map[seq_name] = tokens
        counts_per_seq[seq_name] = _ngram_counts(tokens, min_len=2, max_len=6)
        ent_val = float(entry.get("entropy", 0.0))
        ent_units = str(entry.get("entropy_units", "bits")).lower()
        if ent_units in ("nat", "nats"):
            ent_val *= LOG2E
        analytic_bits[seq_name] = ent_val

    entropy_cache: Dict[str, LZCausalCache] = _build_entropy_cache_map(seq_map)

    lens = {len(v) for v in seq_map.values()}
    if len(lens) != 1:
        raise ValueError(f"Sequences for video {job_idx+1} have mismatched lengths: {lens}")

    if log_progress:
        print(f"[render start] idx={job_idx+1}/{total_videos} seq_ids={seq_names}", flush=True)

    video_id = f"video_{job_idx+1}"
    video_paths: List[str] = []
    time_map: Dict[str, List[float]] = {}
    video_job_seed: Optional[int] = None
    asset_plan: Dict[str, Any] = {}
    frame_debug_records: List[Dict[str, Any]] = []
    render_plan: Optional[LetterRenderPlan] = None
    if not questions_only:
        (
            video_paths,
            time_map,
            video_job_seed,
            asset_plan,
            frame_debug_records,
            render_plan,
        ) = _render_sequence_to_video(
            template_path,
            video_id,
            seq_map,
            videos_dir,
            fps_override=fps_override,
            ffmpeg_crf=ffmpeg_crf,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_codec=ffmpeg_codec,
            capture_frame_debug=capture_frame_debug,
        )
    else:
        prep = _prepare_letter_rendering(template_path, video_id, seq_map, fps_override)
        video_paths = []
        time_map = prep.times
        video_job_seed = prep.seed
        asset_plan = prep.asset_plan
        frame_debug_records = []
        render_plan = prep.plan
    tokens_main = seq_map[seq_names[0]]
    times_main = time_map.get(seq_names[0], [])
    if len(times_main) != len(tokens_main):
        raise RuntimeError("Timing map length mismatch for main sequence")

    durations_main = _durations(times_main)
    median_step = float(np.median(durations_main)) if durations_main else 0.5
    if median_step <= 0:
        median_step = 0.5
    if times_main:
        video_end_time = times_main[-1] + durations_main[-1]
    else:
        video_end_time = float(len(tokens_main)) * median_step

    max_stat_n = None
    for seq_name in seq_names:
        ent = entry_map[seq_name]
        top_stats = ent.get("top_ngrams", [])
        if not top_stats:
            raise ValueError(f"No top_ngrams found for sequence {ent.get('seq_id')}")
        n_here = max(item.get("n", 0) for item in top_stats)
        max_stat_n = n_here if max_stat_n is None else min(max_stat_n, n_here)
    if max_stat_n is None or max_stat_n < 1:
        raise ValueError("Could not determine max n-gram length from stats")

    total_tokens = len(tokens_main)
    if questions_at_end and total_tokens < question_min_len and num_questions:
        raise ValueError(
            f"Sequence length {total_tokens} shorter than required min question length {question_min_len}"
        )
    if questions_at_end:
        positions = [max(0, total_tokens - 1)] * num_questions
    else:
        positions = _pick_positions(total_tokens, num_questions, rng, min_len=question_min_len)

    if render_plan is None:
        raise RuntimeError("Missing render plan for question context")

    ctx = QuestionContext(
        template_path=template_path,
        video_id=video_id,
        seq_map=seq_map,
        seq_names=seq_names,
        entropy_cache=entropy_cache,
        counts_per_seq=counts_per_seq,
        times_main=times_main,
        durations_main=durations_main,
        median_step=median_step,
        video_end_time=video_end_time,
        video_paths=video_paths,
        clips_dir=clips_dir,
        clip_options=clip_options,
        questions_only=questions_only,
        questions_at_end=questions_at_end,
        uniform_uncertain=uniform_uncertain,
        rng=rng,
        question_min_len=question_min_len,
        ngram_max=max_stat_n,
        fps_override=fps_override,
        ffmpeg_crf=ffmpeg_crf,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_codec=ffmpeg_codec,
        video_job_seed=video_job_seed,
        token_letters=asset_plan.get("token_letters", {}),
        token_labels=_token_labels_from_asset_plan(asset_plan),
        hard_questions=hard_questions,
        video_fps=video_fps,
        capture_frame_debug=capture_frame_debug,
        frame_debug=frame_debug_records,
        render_plan=render_plan,
        vocab_upper=vocab_upper,
    )

    if ctx.capture_frame_debug and not questions_only and video_paths:
        _write_frame_debug(Path(video_paths[0]), ctx)

    questions: List[Dict[str, object]] = []
    spatial_fraction = max(0.0, min(1.0, spatial_question_fraction))
    supports_spatial = bool(ctx.secondary_seq_names)

    for idx_q, pos in enumerate(positions):
        use_spatial = False
        if spatial_fraction > 0 and supports_spatial:
            use_spatial = bool(ctx.rng.random() < spatial_fraction)
        question: Optional[Dict[str, Any]]
        if question_mode == QUESTION_MODE_CONTINUATION:
            question = ctx.make_continuation_question(idx_q, spatial=use_spatial)
        else:
            question = ctx.make_exists_question(idx_q, pos, max_stat_n, spatial=use_spatial)
        if question is None:
            if log_progress:
                mode_desc = question_mode
                if question_mode == QUESTION_MODE_EXISTS and use_spatial:
                    mode_desc = f"{mode_desc}-spatial"
                print(
                    f"[render warn] skipping question {idx_q + 1} ({mode_desc}) due to distractor generation failure",
                    flush=True,
                )
            continue
        questions.append(question)

    empirical_bits = {name: cache.entropy_bits() for name, cache in entropy_cache.items()}

    payload = {
        "video_index": job_idx + 1,
        "variant": 0,
        "video_path": video_paths[0] if video_paths else "",
        "sequences_used": seq_map,
        "entropy_overall": {
            "empirical_bits": empirical_bits,
            "analytic_bits": analytic_bits,
        },
        "questions": questions,
        "questions_at_end": questions_at_end,
    }
    if log_progress:
        print(f"[render done] idx={job_idx+1}", flush=True)
    return payload
def _make_job_entry_maps(
    seq_names: List[str],
    sequences_file: Optional[Path],
    sequence_sources: Optional[Dict[str, Path]],
    max_videos: Optional[int],
    assignment_seed: Optional[int],
) -> List[Dict[str, dict]]:
    rng_assign = np.random.default_rng(assignment_seed)
    job_entry_maps: List[Dict[str, dict]] = []

    if sequence_sources:
        seq_entries_by_name: Dict[str, List[dict]] = {}
        for seq_name in seq_names:
            path = sequence_sources.get(seq_name)
            if path is None:
                raise ValueError(f"Missing sequence source for template sequence {seq_name!r}")
            seq_entries_by_name[seq_name] = list(_load_sequence_entries(path))
        lengths = {len(entries) for entries in seq_entries_by_name.values()}
        if not lengths:
            raise ValueError("No sequences loaded from sequence sources")
        if len(lengths) != 1:
            raise ValueError(f"Sequence source counts differ across files: {lengths}")
        pool_len = next(iter(lengths))
        if pool_len == 0:
            raise ValueError("No sequences available from provided sequence sources")
        order = list(range(pool_len))
        rng_assign.shuffle(order)
        total_jobs = pool_len if max_videos is None else min(pool_len, int(max_videos))
        for idx in range(total_jobs):
            sel = order[idx]
            entry_map = {name: seq_entries_by_name[name][sel] for name in seq_names}
            job_entry_maps.append(entry_map)
        return job_entry_maps

    if sequences_file is None:
        raise ValueError("sequences_file must be provided when sequence_sources is not set")
    seq_entries = list(_load_sequence_entries(sequences_file))
    rng_assign.shuffle(seq_entries)
    seqs_per_video = len(seq_names)
    total_possible = len(seq_entries) // seqs_per_video
    if total_possible == 0:
        raise ValueError("Not enough sequences to render a single video")
    total_jobs = total_possible if max_videos is None else min(total_possible, int(max_videos))
    cursor = 0
    for _ in range(total_jobs):
        entry_slice = seq_entries[cursor : cursor + seqs_per_video]
        if len(entry_slice) < seqs_per_video:
            break
        job_entry_maps.append({seq_names[i]: entry_slice[i] for i in range(seqs_per_video)})
        cursor += seqs_per_video
    return job_entry_maps
