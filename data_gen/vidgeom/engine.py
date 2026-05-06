from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Callable
import heapq
import math
import numpy as np

from .assets import (
    build_asset_store,
    AssetSpec,
    assign_unique_token_colors,
    assign_token_sprites,
    assign_token_shapes,
    assign_token_letters,
)
from .events import TokenEvent, SceneEvent
from .rng import derive_seed, make_rng
from .renderer import GeometryRenderer, RenderConfig
from .template import Template

def _get_fallback_spec(vocab: Dict[str, Any]) -> Optional[AssetSpec]:
    fb = (vocab or {}).get("fallback", None)
    if not fb:
        return None
    return AssetSpec(
        type=str(fb.get("type", "procedural.box")),
        params=fb.get("params", {}) or {},
        seed=fb.get("seed", None),
    )


@dataclass
class VideoJob:
    """Runtime-provided job; sequences determine the actual video."""
    id: str
    sequences: Dict[str, List[str]]
    meta: Dict[str, Any] = None
    seed: Optional[int] = None

class _Scheduler:
    def __init__(self):
        self._token_heap: List[TokenEvent] = []
        self._scene_heap: List[SceneEvent] = []

    def schedule_token(self, ev: TokenEvent) -> None:
        heapq.heappush(self._token_heap, ev)

    def schedule_scene(self, ev: SceneEvent) -> None:
        heapq.heappush(self._scene_heap, ev)

    def pop_due_tokens(self, t: float) -> List[TokenEvent]:
        out = []
        while self._token_heap and self._token_heap[0].t <= t + 1e-9:
            out.append(heapq.heappop(self._token_heap))
        return out

    def pop_due_scene(self, t: float) -> List[SceneEvent]:
        out = []
        while self._scene_heap and self._scene_heap[0].t <= t + 1e-9:
            out.append(heapq.heappop(self._scene_heap))
        return out

    def peek_max_time(self) -> float:
        mx = 0.0
        if self._token_heap:
            mx = max(mx, max(ev.t for ev in self._token_heap))
        if self._scene_heap:
            mx = max(mx, max(ev.t for ev in self._scene_heap))
        return mx

    def has_pending(self) -> bool:
        return bool(self._token_heap or self._scene_heap)

@dataclass
class VideoInstance:
    job: VideoJob
    template: Template
    variant_idx: int
    seed: int
    sequences: Dict[str, List[str]]
    scene: Any
    rules: List[Any]
    assets: Any
    scheduler: _Scheduler
    fps: int
    width: int
    height: int
    duration: float

def _build_token_schedule(
    sequences: Dict[str, List[str]],
    timing: Dict[str, Any],
    scheduler: _Scheduler,
    rng: Optional[np.random.Generator],
):
    if not sequences:
        return
    base_step = float(timing.get("step_duration", 0.6))
    step_range = timing.get("step_duration_range")
    offsets = timing.get("per_sequence_offset", {}) or {}
    seq_ids = list(sequences.keys())
    primary = seq_ids[0]
    primary_tokens = sequences[primary]
    offset_primary = float(offsets.get(primary, 0.0))

    def _next_step() -> float:
        if step_range and rng is not None and len(step_range) == 2:
            lo, hi = float(step_range[0]), float(step_range[1])
            return float(rng.uniform(min(lo, hi), max(lo, hi)))
        return base_step

    t_primary = offset_primary
    for idx, tok in enumerate(primary_tokens):
        for seq_id in seq_ids:
            toks = sequences.get(seq_id, [])
            if idx >= len(toks):
                continue
            delta = float(offsets.get(seq_id, 0.0)) - offset_primary
            t_event = t_primary + delta
            scheduler.schedule_token(
                TokenEvent(
                    t=t_event,
                    token=str(toks[idx]),
                    seq_id=str(seq_id),
                    index=idx,
                    meta={"seq_index": idx},
                )
            )
        t_primary += _next_step()

def instantiate(template: Template, job: VideoJob) -> List[VideoInstance]:
    """Create one or more VideoInstances from a job, based on template variants."""
    base_seed = int(template.raw.get("seed", 12345))
    job_seed = int(job.seed) if job.seed is not None else derive_seed(base_seed, job.id)
    vcfg = template.variants
    per_job = int(vcfg.get("per_job", 1))
    mapping_resample = bool(vcfg.get("mapping_resample", True))

    render_cfg = template.render
    width, height = render_cfg.get("resolution", [448, 448])
    fps = int(render_cfg.get("fps", 30))

    instances: List[VideoInstance] = []
    for vidx in range(per_job):
        seed = derive_seed(job_seed, "variant", vidx)
        rng = make_rng(seed)
        # mapping/prototypes
        vocab = template.vocab
        mapping = (vocab.get("mapping", {}) or {})
        asset_seed = derive_seed(seed, "assets")
        arng = make_rng(asset_seed)
        job_meta = job.meta or {}
        tokens = sorted({str(tok) for seq in job.sequences.values() for tok in seq})
        extra_ids: set[str] = set()
        vocab_ids = vocab.get("token_ids") or []
        for tid in vocab_ids:
            extra_ids.add(str(tid))
        override_colors = job_meta.get("token_colors")
        override_sprites = job_meta.get("token_sprites")
        if override_colors:
            extra_ids.update(str(tok) for tok in override_colors.keys())
        if override_sprites and isinstance(override_sprites, dict):
            extra_ids.update(str(tok) for tok in override_sprites.get("map", {}).keys())
        if extra_ids:
            tokens = sorted(set(tokens) | extra_ids)

        token_colors = override_colors or None
        unique_cfg_raw = vocab.get("unique_token_colors", True)
        unique_cfg: Dict[str, Any] = {}
        if isinstance(unique_cfg_raw, dict):
            enable_unique = True
            unique_cfg = unique_cfg_raw
        else:
            enable_unique = bool(unique_cfg_raw)
        if enable_unique and tokens and token_colors is None:
            color_seed = derive_seed(seed, "token_colors")
            color_rng = make_rng(color_seed)
            token_colors = assign_unique_token_colors(tokens, color_rng, unique_cfg)
        token_sprites = override_sprites or None
        sprite_cfg = vocab.get("token_sprites")
        if tokens and token_sprites is None and sprite_cfg is not None:
            sprite_seed = derive_seed(seed, "token_sprites")
            sprite_rng = make_rng(sprite_seed)
            token_sprites = assign_token_sprites(tokens, sprite_rng, sprite_cfg)
        override_shapes = job_meta.get("token_shapes")
        token_shapes = override_shapes or None
        shape_cfg_raw = vocab.get("unique_token_shapes")
        enable_shapes = False
        shape_cfg: Dict[str, Any] = {}
        allow_shape_reuse = False
        if isinstance(shape_cfg_raw, dict):
            enable_shapes = True
            shape_cfg = shape_cfg_raw
            allow_shape_reuse = bool(shape_cfg.get("allow_reuse", False))
        elif shape_cfg_raw:
            enable_shapes = True
            allow_shape_reuse = False
        if enable_shapes and tokens and token_shapes is None:
            shape_seed = derive_seed(seed, "token_shapes")
            shape_rng = make_rng(shape_seed)
            library = shape_cfg.get("library")
            if library:
                library = [str(entry) for entry in library]
            token_shapes = assign_token_shapes(tokens, shape_rng, library, allow_reuse=allow_shape_reuse)
        override_letters = job_meta.get("token_letters")
        token_letters = override_letters or None
        letter_cfg = vocab.get("token_letters")
        if tokens and token_letters is None and letter_cfg is not None:
            letter_seed = derive_seed(seed, "token_letters")
            letter_rng = make_rng(letter_seed)
            token_letters = assign_token_letters(tokens, letter_rng, letter_cfg)
        assets = build_asset_store(
            mapping,
            arng,
            base_seed=asset_seed,
            fallback=_get_fallback_spec(vocab),
            token_colors=token_colors,
            token_sprites=token_sprites,
            token_shapes=token_shapes,
            token_letters=token_letters,
            tokens=tokens,
        )

        scene = template.make_scene()
        scene.reset(template.scene_cfg, assets, rng)

        rules = template.make_rules()

        scheduler = _Scheduler()
        _build_token_schedule(job.sequences, template.timing, scheduler, rng)

        # Determine duration: from last token time + tail
        last_t = scheduler.peek_max_time()
        tail = float(getattr(scene, "duration_tail", 1.0))
        duration = last_t + tail

        instances.append(VideoInstance(
            job=job,
            template=template,
            variant_idx=vidx,
            seed=seed,
            sequences=job.sequences,
            scene=scene,
            rules=rules,
            assets=assets,
            scheduler=scheduler,
            fps=fps,
            width=int(width),
            height=int(height),
            duration=float(duration),
        ))
    return instances

def frame_generator(
    instance: VideoInstance,
    frame_observer: Optional[Callable[[int, float, Optional[Any]], None]] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yields (t, frame_rgb_uint8) for each frame."""
    renderer = GeometryRenderer(instance.assets, RenderConfig(width=instance.width, height=instance.height))
    dt = 1.0 / instance.fps
    t = 0.0
    # We keep an rng for rules separate from asset rng; deterministic
    rng = make_rng(instance.seed ^ 0xA5A5A5A5)

    n_frames = max(1, int(math.ceil(instance.duration * instance.fps)))
    tail = float(getattr(instance.scene, "duration_tail", 0.0))
    fi = 0
    while True:
        if fi >= n_frames:
            pending_max = instance.scheduler.peek_max_time()
            if not instance.scheduler.has_pending():
                break
            required_duration = max(instance.duration, pending_max + tail)
            n_frames = max(n_frames + 1, int(math.ceil(required_duration * instance.fps)))
            instance.duration = required_duration
            continue
        t = fi * dt
        # process token events
        for ev in instance.scheduler.pop_due_tokens(t):
            consumed = False
            for rule in instance.rules:
                if rule.on_token(ev, instance.scene, instance.scheduler, rng):
                    consumed = True
                    break
            if not consumed:
                instance.scene.on_token(ev.token, ev.seq_id, ev.t, ev.meta)

        # process scheduled scene events (from rules)
        for sev in instance.scheduler.pop_due_scene(t):
            for rule in instance.rules:
                rule.on_scene_event(sev, instance.scene, instance.scheduler, rng)

        # step scene
        instance.scene.step(t, dt)

        # scene can emit events too
        if hasattr(instance.scene, "pop_events"):
            for sev in instance.scene.pop_events():
                for rule in instance.rules:
                    rule.on_scene_event(sev, instance.scene, instance.scheduler, rng)

        draw = instance.scene.draw(t)
        debug_meta = None
        if hasattr(instance.scene, "frame_debug_snapshot"):
            try:
                debug_meta = instance.scene.frame_debug_snapshot()
            except Exception:
                debug_meta = None
        if frame_observer is not None:
            try:
                frame_observer(fi, t, debug_meta)
            except Exception:
                pass
        frame = renderer.render(draw)
        yield t, frame
        fi += 1
