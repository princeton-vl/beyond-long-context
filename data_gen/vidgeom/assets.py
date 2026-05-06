from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import itertools
import math
import numpy as np
import cv2

from .draw import CirclePrim, LinePrim, PolyPrim, Primitive, Prototype, Color, ImagePrim
from .distributions import sample_params
from .rng import make_rng
from .sprites import available_sprite_ids, get_sprite


def _resolve_sprite_manifest(path: Optional[str]) -> Path:
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            base = Path(__file__).resolve().parent
            candidate = (base / path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Sprite manifest '{candidate}' not found")
        return candidate
    from .sprites import _DEFAULT_MANIFEST  # type: ignore

    manifest = _DEFAULT_MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(f"Default sprite manifest '{manifest}' not found")
    return manifest

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _rgb(c) -> Color:
    if isinstance(c, (list, tuple)) and len(c) == 3:
        if all(isinstance(v, (int, np.integer)) for v in c):
            return (int(c[0]), int(c[1]), int(c[2]))
        return (int(_clamp01(float(c[0]))*255), int(_clamp01(float(c[1]))*255), int(_clamp01(float(c[2]))*255))
    raise ValueError(f"Invalid color: {c}")

def _rgb_to_bgr_tuple(color: Color) -> Tuple[int, int, int]:
    return (int(color[2]), int(color[1]), int(color[0]))

def _darken(color: Color, f: float) -> Color:
    return (int(color[0]*f), int(color[1]*f), int(color[2]*f))

def _lighten(color: Color, f: float) -> Color:
    return (int(color[0] + (255-color[0])*f), int(color[1] + (255-color[1])*f), int(color[2] + (255-color[2])*f))

def _normalize_color_entry(entry: Any) -> Optional[Color]:
    if isinstance(entry, str):
        key = entry.strip().lower().replace("_", " ")
        key = " ".join(key.split())
        if key in _COLOR_NAME_TABLE:
            return _COLOR_NAME_TABLE[key]
    try:
        return _rgb(entry)
    except Exception:
        return None

_COLOR_NAME_TABLE: Dict[str, Color] = {
    "red": (220, 32, 41),
    "orange": (248, 142, 52),
    "yellow": (255, 215, 64),
    "green": (62, 176, 82),
    "navy": (32, 52, 128),
    "purple": (138, 84, 201),
    "black": (24, 24, 28),
    "white": (248, 248, 248),
    "gray": (130, 136, 145),
    "pink": (245, 115, 185),
    "forest green": (36, 120, 68),
    "cyan": (0, 173, 202),
    "magenta": (212, 48, 150),
    "maroon": (124, 34, 58),
    "beige": (235, 210, 170),
    "olive": (118, 126, 46),
}

DEFAULT_COLOR_ORDER: Tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "navy",
    "purple",
    "black",
    "white",
    "gray",
    "pink",
    "forest green",
    "cyan",
    "magenta",
    "maroon",
    "beige",
    "olive",
)

DEFAULT_TOKEN_COLOR_PALETTE: Tuple[Color, ...] = tuple(
    _COLOR_NAME_TABLE[name] for name in DEFAULT_COLOR_ORDER
)

def apply_token_color(proto: Prototype, base_color: Optional[Tuple[Color, Color, Color]]) -> Prototype:
    if base_color is None:
        return proto
    colors = list(base_color)
    max_idx = len(colors) - 1
    for prim in proto.primitives:
        if isinstance(prim, (CirclePrim, PolyPrim, LinePrim)):
            level = int(max(0, min(max_idx, round(getattr(prim, 'z', 0)))))
            prim.color = colors[level]
    return proto

def _resolve_palette(palette_cfg: Optional[Sequence[Any]]) -> List[Color]:
    if palette_cfg is None:
        return list(DEFAULT_TOKEN_COLOR_PALETTE)
    resolved: List[Color] = []
    for entry in palette_cfg:
        color = _normalize_color_entry(entry)
        if color is None:
            continue
        if color not in resolved:
            resolved.append(color)
    if not resolved:
        return list(DEFAULT_TOKEN_COLOR_PALETTE)
    return resolved

def assign_unique_token_colors(
    tokens: Sequence[str],
    rng: np.random.Generator,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Color, Color, Color]]:
    cfg = cfg or {}
    unique = sorted({str(tok) for tok in tokens})
    if not unique:
        return {}

    palette = _resolve_palette(cfg.get('palette'))
    colors: Dict[str, Tuple[Color, Color, Color]] = {}
    used: set[Color] = set()
    palette_iter = iter(palette)
    for token in unique:
        try:
            primary = next(palette_iter)
        except StopIteration:
            while True:
                candidate = tuple(int(x) for x in rng.integers(0, 256, size=3))
                if candidate not in used:
                    primary = candidate
                    break
        used.add(primary)
        accent = _darken(primary, 0.4)
        tertiary = _lighten(primary, 0.25)
        colors[token] = (primary, accent, tertiary)
    return colors

def _make_polygon(points: List[Tuple[float, float]], color: Color, z: float = 0.0) -> PolyPrim:
    return PolyPrim(points=points, color=color, z=z)

def _triangle_points(width: float, height: float) -> List[Tuple[float, float]]:
    return [(-width / 2, -height / 2), (width / 2, -height / 2), (0.0, height / 2)]

def _rectangle_points(width: float, height: float) -> List[Tuple[float, float]]:
    hw, hh = width / 2, height / 2
    return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

def _rotated_rectangle_points(width: float, height: float, angle_rad: float) -> List[Tuple[float, float]]:
    pts = _rectangle_points(width, height)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        (x * c - y * s, x * s + y * c)
        for x, y in pts
    ]

def _star_points(r_outer: float, r_inner: float) -> List[Tuple[float, float]]:
    pts = []
    for k in range(10):
        a = math.pi * k / 5
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((math.cos(a) * r, math.sin(a) * r))
    return pts

def _build_shape(name: str, colors: Tuple[Color, Color, Color]) -> List[Primitive]:
    c0, c1, c2 = colors
    if name == 'circle':
        return [CirclePrim(center=(0.0, 0.0), radius=0.48, color=c0, z=0)]
    if name == 'square':
        return [_make_polygon(_rectangle_points(0.92, 0.92), c0, z=0)]
    if name == 'square_circle':
        return [
            _make_polygon(_rectangle_points(0.92, 0.92), c0, z=0),
            CirclePrim(center=(0.0, 0.0), radius=0.34, color=c1, z=1),
        ]
    if name == 'circle_square':
        return [
            CirclePrim(center=(0.0, 0.0), radius=0.48, color=c0, z=0),
            _make_polygon(_rectangle_points(0.36, 0.36), c1, z=1),
        ]
    if name == 'triangle':
        return [_make_polygon(_triangle_points(0.94, 0.84), c0, z=0)]
    if name == 'rect_triangle':
        return [
            _make_polygon(_rectangle_points(0.85, 0.48), c0, z=0),
            _make_polygon([(x, y + 0.45) for x, y in _triangle_points(0.70, 0.50)], c1, z=1),
        ]
    if name == 'star':
        return [_make_polygon(_star_points(0.48, 0.24), c0, z=0)]
    if name == 't_shape':
        return [
            _make_polygon(_rectangle_points(0.22, 0.96), c0, z=0),
            _make_polygon([(x, y + 0.36) for x, y in _rectangle_points(0.92, 0.24)], c1, z=1),
        ]
    if name == 'x_shape':
        arm_main = _rotated_rectangle_points(0.22, 1.08, math.pi / 4)
        arm_cross = _rotated_rectangle_points(0.22, 1.08, -math.pi / 4)
        return [
            _make_polygon(arm_main, c0, z=0),
            _make_polygon(arm_cross, c1, z=1),
        ]
    if name == 'l_shape':
        return [
            _make_polygon([(x - 0.32, y) for x, y in _rectangle_points(0.24, 0.96)], c0, z=0),
            _make_polygon([(x + 0.24, y - 0.34) for x, y in _rectangle_points(0.72, 0.24)], c1, z=1),
        ]
    if name == 'venn':
        return [
            CirclePrim(center=(-0.26, 0.0), radius=0.34, color=c0, z=0),
            CirclePrim(center=(0.26, 0.0), radius=0.34, color=c1, z=1),
        ]
    if name == 'square_venn':
        return [
            _make_polygon(_rectangle_points(0.92, 0.92), c0, z=0),
            CirclePrim(center=(-0.18, 0.0), radius=0.24, color=c1, z=1),
            CirclePrim(center=(0.18, 0.0), radius=0.24, color=c2, z=2),
        ]
    if name == 'hollow_rect':
        prims = []
        w, h, t = 0.94, 0.94, 0.12
        prims.append(_make_polygon(_rectangle_points(t, h), c0, z=0))
        prims.append(_make_polygon([(x + (w/2 - t/2), y) for x, y in _rectangle_points(t, h)], c0, z=0))
        prims.append(_make_polygon([(x, y + (h/2 - t/2)) for x, y in _rectangle_points(w, t)], c1, z=1))
        prims.append(_make_polygon([(x, y - (h/2 - t/2)) for x, y in _rectangle_points(w, t)], c1, z=1))
        return prims
    if name == 'diag_squares':
        offset = 0.32
        return [
            _make_polygon([(x - offset, y + offset) for x, y in _rectangle_points(0.38, 0.38)], c0, z=0),
            _make_polygon([(x + offset, y - offset) for x, y in _rectangle_points(0.38, 0.38)], c1, z=1),
        ]
    if name == 'circle_star':
        return [
            CirclePrim(center=(0.0, 0.0), radius=0.48, color=c0, z=0),
            _make_polygon(_star_points(0.25, 0.12), c1, z=1),
        ]
    if name == 'triangle_star':
        return [
            _make_polygon(_triangle_points(0.94, 0.80), c0, z=0),
            _make_polygon([(x, y + 0.46) for x, y in _star_points(0.20, 0.10)], c1, z=1),
        ]
    if name == 'hourglass':
        return [
            _make_polygon(_triangle_points(0.68, 0.50), c0, z=0),
            _make_polygon([(x, -y) for x, y in _triangle_points(0.68, 0.50)], c1, z=1),
        ]
    if name == 'rect_two_circles':
        return [
            _make_polygon(_rectangle_points(0.86, 0.40), c0, z=0),
            CirclePrim(center=(-0.26, 0.32), radius=0.20, color=c1, z=1),
            CirclePrim(center=(0.26, 0.32), radius=0.20, color=c2, z=2),
        ]
    if name == 'four_circles':
        offsets = [(-0.30, 0.30), (0.30, 0.30), (-0.30, -0.30), (0.30, -0.30)]
        palette_cycle = [c0, c1, c2, c0]
        return [
            CirclePrim(center=offset, radius=0.22, color=palette_cycle[i % len(palette_cycle)], z=i % 2)
            for i, offset in enumerate(offsets)
        ]
    return [CirclePrim(center=(0.0, 0.0), radius=0.48, color=c0, z=0)]

SHAPE_LIBRARY = [
    'circle',
    'square',
    'square_circle',
    'circle_square',
    'triangle',
    'rect_triangle',
    'star',
    't_shape',
    'x_shape',
    'l_shape',
    'venn',
    'hollow_rect',
    'diag_squares',
    'circle_star',
    'triangle_star',
    'hourglass',
    'square_venn',
    'rect_two_circles',
    'four_circles',
]

def assign_token_shapes(
    tokens: Sequence[str],
    rng: np.random.Generator,
    shape_library: Optional[Sequence[str]] = None,
    *,
    allow_reuse: bool = False,
) -> Dict[str, str]:
    unique = sorted({str(tok) for tok in tokens})
    if not unique:
        return {}
    library = list(shape_library or SHAPE_LIBRARY)
    if len(unique) > len(library) and not allow_reuse:
        raise RuntimeError("Not enough predefined shapes to assign all tokens uniquely")
    if rng is not None:
        library = list(library)
        rng.shuffle(library)
    if allow_reuse and library:
        cycle_iter = itertools.cycle(library)
        return {tok: next(cycle_iter) for tok in unique}
    return {tok: library[idx] for idx, tok in enumerate(unique)}

DEFAULT_LETTER_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _resolve_letter_alphabet(raw: Optional[Any]) -> List[str]:
    if raw is None:
        base = list(DEFAULT_LETTER_ALPHABET)
    elif isinstance(raw, str):
        base = [ch for ch in raw if ch.strip()]
    else:
        base = []
        for entry in raw:
            if not entry:
                continue
            s = str(entry)
            if not s:
                continue
            base.append(s[0])
    seen = set()
    ordered: List[str] = []
    for ch in base:
        key = ch.strip()
        if not key:
            continue
        key = key[0]
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    if not ordered:
        ordered = list(DEFAULT_LETTER_ALPHABET)
    return ordered


def assign_token_letters(
    tokens: Sequence[str],
    rng: np.random.Generator,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or {}
    unique = sorted({str(tok) for tok in tokens})
    if not unique:
        return {"map": {}, "style": {}}
    alphabet = _resolve_letter_alphabet(cfg.get('alphabet'))
    uppercase = bool(cfg.get('uppercase', True))
    if uppercase:
        alphabet = [ch.upper() for ch in alphabet]
    else:
        alphabet = [ch.lower() for ch in alphabet]
    if len(alphabet) < len(unique):
        raise RuntimeError("Not enough letters to cover the requested token vocabulary")
    pool = list(alphabet)
    if rng is not None and bool(cfg.get('shuffle', True)):
        rng.shuffle(pool)
    mapping = {tok: pool[idx] for idx, tok in enumerate(unique)}

    style_cfg: Dict[str, Any] = dict(cfg.get('style', {}))
    if 'color' in cfg:
        style_cfg['color'] = _rgb(cfg['color'])
    elif 'color' in style_cfg:
        style_cfg['color'] = _rgb(style_cfg['color'])
    else:
        style_cfg['color'] = (245, 245, 245)
    if 'outline_color' in cfg:
        val = cfg['outline_color']
        style_cfg['outline_color'] = None if val is None else _rgb(val)
    elif 'outline_color' in style_cfg:
        val = style_cfg['outline_color']
        style_cfg['outline_color'] = None if val is None else _rgb(val)
    else:
        style_cfg['outline_color'] = (30, 30, 35)
    style_cfg['uppercase'] = uppercase if 'uppercase' not in style_cfg else bool(style_cfg['uppercase'])
    for numeric_key in ('font_scale', 'thickness', 'outline_thickness', 'image_size'):
        if numeric_key in cfg:
            style_cfg[numeric_key] = cfg[numeric_key]
    return {
        'map': mapping,
        'style': style_cfg,
    }


def assign_token_sprites(
    tokens: Sequence[str],
    rng: np.random.Generator,
    sprite_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sprite_cfg = sprite_cfg or {}
    unique = sorted({str(tok) for tok in tokens})
    if not unique:
        return {}

    manifest_path = _resolve_sprite_manifest(sprite_cfg.get('manifest'))
    sprite_ids = available_sprite_ids(str(manifest_path))
    pool = list(sprite_ids)
    allowed = sprite_cfg.get('allowed')
    if allowed:
        allowed_set = {str(a).strip() for a in allowed}
        pool = [sid for sid in pool if sid in allowed_set]
    rng.shuffle(pool)
    limit = sprite_cfg.get('limit')
    if limit is not None:
        limit = int(limit)
        if limit > 0:
            pool = pool[:limit]
    if len(unique) > len(pool):
        raise RuntimeError(
            f"Sprite catalog '{manifest_path}' only provides {len(pool)} entries; need {len(unique)} unique tokens"
        )
    mapping = {tok: pool[idx] for idx, tok in enumerate(unique)}
    return {
        'manifest': str(manifest_path),
        'map': mapping,
    }


def _resolve_sprite_manifest(path: Optional[str]) -> Path:
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (Path(__file__).resolve().parent / candidate).resolve()
        return candidate
    from .sprites import _DEFAULT_MANIFEST  # type: ignore

    return _DEFAULT_MANIFEST

def make_sculpture2d(params: Dict[str, Any], rng: np.random.Generator) -> Prototype:
    palette = params.get('palette', [
        (210, 210, 220), (240, 200, 180), (180, 220, 200),
        (200, 170, 240), (150, 210, 230), (220, 180, 150),
        (190, 210, 120), (120, 190, 210),
        (240, 120, 120), (120, 240, 120), (120, 120, 240),
        (240, 200, 120), (120, 200, 240), (200, 120, 240),
    ])
    if isinstance(palette, (list, tuple)):
        if len(palette) == 3 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in palette):
            palette = [_rgb(palette)]
        elif len(palette) > 0 and isinstance(palette[0], (list, tuple)):
            palette = [_rgb(p) for p in palette]
        else:
            palette = [(210, 210, 220)]
    else:
        palette = [(210, 210, 220)]
    base = palette[int(rng.integers(0, len(palette)))]

    allow_two = bool(params.get('allow_two_colors', True))
    two_color_prob = float(params.get('two_color_prob', 0.65))
    accent = base
    if allow_two and len(palette) > 1:
        if bool(params.get('force_two_colors', False)) or (rng.random() < two_color_prob):
            candidates = [c for c in palette if c != base]
            if candidates:
                accent = candidates[int(rng.integers(0, len(candidates)))]

    tertiary = _lighten(accent, 0.3)
    shape = SHAPE_LIBRARY[int(rng.integers(0, len(SHAPE_LIBRARY)))]
    prims = _build_shape(shape, (base, accent, tertiary))
    return Prototype(primitives=prims, radius_hint=0.7)

def make_template_sculpture(shape: str, colors: Tuple[Color, Color, Color]) -> Prototype:
    return Prototype(primitives=_build_shape(shape, colors), radius_hint=0.7)

def make_composite_sculpture(params: Dict[str, Any], rng: np.random.Generator) -> Prototype:
    count = int(params.get('count', rng.integers(2, 4)))
    offset_scale = float(params.get('offset_scale', 0.28))
    lock_palette = bool(params.get('lock_palette_per_composite', True))

    base_palette = params.get('palette')
    palette_subset = None
    if lock_palette and isinstance(base_palette, (list, tuple)) and len(base_palette) >= 2:
        k = int(min(len(base_palette), rng.integers(2, 4)))
        palette_subset = [base_palette[int(i)] for i in rng.choice(len(base_palette), size=k, replace=False)]
    prims: List[Primitive] = []
    for _ in range(count):
        local_params = dict(params)
        local_params['rot'] = rng.uniform(0, 2*math.pi)
        if palette_subset is not None:
            local_params['palette'] = palette_subset
        proto = make_sculpture2d(local_params, rng)
        ox = rng.uniform(-offset_scale, offset_scale)
        oy = rng.uniform(-offset_scale, offset_scale)
        for prim in proto.primitives:
            if isinstance(prim, CirclePrim):
                prims.append(CirclePrim(center=(prim.center[0]+ox, prim.center[1]+oy), radius=prim.radius, color=prim.color, z=prim.z))
            elif isinstance(prim, PolyPrim):
                prims.append(PolyPrim(points=[(x+ox, y+oy) for x, y in prim.points], color=prim.color, z=prim.z))
            elif isinstance(prim, LinePrim):
                prims.append(LinePrim(p0=(prim.p0[0]+ox, prim.p0[1]+oy), p1=(prim.p1[0]+ox, prim.p1[1]+oy), width=prim.width, color=prim.color, z=prim.z))
    return Prototype(primitives=prims, radius_hint=0.85)

def make_gear(params: Dict[str, Any], rng: np.random.Generator) -> Prototype:
    teeth = int(params.get('teeth', 12))
    r_in = float(params.get('r_in', 0.18))
    r_out = float(params.get('r_out', 0.28))
    color = _rgb(params.get('color', (220, 190, 120)))
    outline = _darken(color, 0.55)
    pts = []
    for k in range(teeth*2):
        a = 2*math.pi*k/(teeth*2)
        r = r_out if (k % 2 == 0) else r_in
        pts.append((math.cos(a)*r, math.sin(a)*r))
    prims = [
        PolyPrim(points=[(x*1.06, y*1.06) for x, y in pts], color=outline, z=-0.01),
        PolyPrim(points=pts, color=color, z=0.0),
        CirclePrim(center=(0.0, 0.0), radius=r_in*0.45, color=_darken(color, 0.65), z=0.02),
        CirclePrim(center=(0.0, 0.0), radius=r_in*0.28, color=_lighten(color, 0.35), z=0.03),
    ]
    return Prototype(primitives=prims, radius_hint=r_out*1.1)

def make_box(params: Dict[str, Any], rng: np.random.Generator) -> Prototype:
    w = float(params.get('w', 0.46))
    h = float(params.get('h', 0.34))
    bevel = float(params.get('bevel', 0.08))
    color = _rgb(params.get('color', (120, 200, 160)))
    dark = _darken(color, 0.60)
    hi = _lighten(color, 0.30)

    bx, by = w/2, h/2
    b = bevel * min(w, h)
    pts = [(-bx+b, -by), (bx-b, -by), (bx, -by+b), (bx, by-b), (bx-b, by), (-bx+b, by), (-bx, by-b), (-bx, -by+b)]
    prims = [
        PolyPrim(points=[(x*1.03, y*1.03) for x, y in pts], color=dark, z=-0.01),
        PolyPrim(points=pts, color=color, z=0.0),
        LinePrim(p0=(-bx+b, -by+0.01), p1=(bx-b, -by+0.01), width=0.02, color=hi, z=0.02),
        LinePrim(p0=(-bx+b, -by+0.01), p1=(-bx+0.02, by-b), width=0.02, color=hi, z=0.02),
    ]
    return Prototype(primitives=prims, radius_hint=max(w, h)*0.7)


def make_letter_prototype(letter: str, style: Dict[str, Any]) -> Prototype:
    char = str(letter or "?")
    if not char:
        char = "?"
    uppercase = bool(style.get('uppercase', True))
    char = char.upper() if uppercase else char.lower()
    color = _rgb(style.get('color', (245, 245, 245)))
    outline_color = style.get('outline_color')
    outline_rgb = None if outline_color is None else _rgb(outline_color)
    font_scale = float(style.get('font_scale', 5.2))
    thickness = max(1, int(style.get('thickness', 14)))
    if outline_rgb is not None:
        outline_thickness = max(thickness + 4, int(style.get('outline_thickness', thickness + 4)))
    else:
        outline_thickness = thickness
    image_size = max(64, int(style.get('image_size', 256)))

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    ((text_w, text_h), baseline) = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, (image_size - text_w) // 2)
    y = min(image_size - 1, max(text_h + baseline, (image_size + text_h) // 2))
    if outline_rgb is not None:
        cv2.putText(
            canvas,
            char,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _rgb_to_bgr_tuple(outline_rgb),
            outline_thickness,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        char,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        _rgb_to_bgr_tuple(color),
        thickness,
        cv2.LINE_AA,
    )
    alpha = np.max(canvas, axis=2, keepdims=True)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pixels = np.concatenate([rgb, alpha], axis=2)
    prim = ImagePrim(pixels=pixels, width=1.0, height=1.0, z=0.0)
    return Prototype(primitives=[prim], radius_hint=0.65)


def make_sprite_prototype(sprite_id: str, manifest_path: Optional[str]) -> Prototype:
    sprite = get_sprite(sprite_id, manifest_path)
    prim = ImagePrim(pixels=sprite.image.copy(), width=1.0, height=1.0, z=0.0)
    return Prototype(primitives=[prim], radius_hint=0.6)

GENERATOR_REGISTRY = {
    'procedural.sculpture2d': make_sculpture2d,
    'procedural.sculpture2d.composite': make_composite_sculpture,
    'procedural.gear': make_gear,
    'procedural.box': make_box,
}

@dataclass
class AssetSpec:
    type: str
    params: Dict[str, Any]
    seed: Optional[int] = None

class AssetStore:
    def __init__(
        self,
        *,
        fallback: Optional[AssetSpec] = None,
        fallback_base_seed: int = 0,
        token_colors_plan: Optional[Dict[str, Tuple[Color, Color, Color]]] = None,
        token_sprites_plan: Optional[Dict[str, Any]] = None,
        token_shapes_plan: Optional[Dict[str, str]] = None,
        token_letters_plan: Optional[Dict[str, Any]] = None,
        planned_tokens: Optional[Sequence[str]] = None,
    ):
        self.prototypes: Dict[str, Prototype] = {}
        self.fallback: Optional[AssetSpec] = fallback
        self.fallback_base_seed: int = int(fallback_base_seed)
        self.token_colors_plan = dict(token_colors_plan or {})
        self.token_sprites_plan = dict(token_sprites_plan or {})
        self.token_shapes_plan = dict(token_shapes_plan or {})
        self.token_letters_plan = dict(token_letters_plan or {})
        self.planned_tokens = list(planned_tokens or [])

    def add(self, asset_id: str, proto: Prototype):
        self.prototypes[asset_id] = proto

    def get(self, asset_id: str) -> Prototype:
        if asset_id in self.prototypes:
            return self.prototypes[asset_id]
        if self.fallback is None:
            raise KeyError(f"Unknown asset_id '{asset_id}' and no vocab.fallback specified.")
        seed = (self.fallback_base_seed ^ (hash(asset_id) & 0xFFFFFFFF)) & 0xFFFFFFFF
        srng = make_rng(seed)
        gen = GENERATOR_REGISTRY.get(self.fallback.type)
        if gen is None:
            raise ValueError(f"Unknown fallback asset generator type: {self.fallback.type}")
        params = sample_params(self.fallback.params or {}, srng)
        if 'color' not in params:
            h = hash(asset_id) & 0xFFFFFFFF
            params['color'] = (80 + (h % 140), 80 + ((h >> 8) % 140), 80 + ((h >> 16) % 140))
        proto = gen(params, srng)
        self.prototypes[asset_id] = proto
        return proto

    def export_plan(self) -> Dict[str, Any]:
        return {
            'token_colors': dict(self.token_colors_plan),
            'token_sprites': dict(self.token_sprites_plan),
            'token_shapes': dict(self.token_shapes_plan),
            'token_letters': dict(self.token_letters_plan),
            'planned_tokens': list(self.planned_tokens),
        }

def build_asset_store(
    vocab_mapping: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
    base_seed: int,
    *,
    fallback: Optional[AssetSpec] = None,
    token_colors: Optional[Dict[str, Tuple[Color, Color, Color]]] = None,
    token_sprites: Optional[Dict[str, Any]] = None,
    token_shapes: Optional[Dict[str, str]] = None,
    token_letters: Optional[Dict[str, Any]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> AssetStore:
    planned_tokens = sorted({str(tok) for tok in tokens}) if tokens else []
    store = AssetStore(
        fallback=fallback,
        fallback_base_seed=base_seed,
        token_colors_plan=token_colors,
        token_sprites_plan=token_sprites,
        token_shapes_plan=token_shapes,
        token_letters_plan=token_letters,
        planned_tokens=planned_tokens,
    )
    sprite_manifest = None
    sprite_map: Dict[str, str] = {}
    if token_sprites:
        sprite_manifest = token_sprites.get('manifest')
        sprite_map = dict(token_sprites.get('map', {}))
    shape_map = dict(token_shapes or {})
    letter_plan = token_letters or {}
    letter_map: Dict[str, str] = dict(letter_plan.get('map', {}))
    letter_style: Dict[str, Any] = dict(letter_plan.get('style', {}))
    for sym, spec in (vocab_mapping or {}).items():
        atype = str(spec.get('type', 'procedural.sculpture2d'))
        params_spec = spec.get('params', {}) or {}
        params = sample_params(params_spec, rng)
        seed = spec.get('seed', None)
        if seed is None:
            seed = int(rng.integers(0, 2**31 - 1))
        srng = make_rng((base_seed ^ (hash(sym) & 0xFFFFFFFF) ^ int(seed)) & 0xFFFFFFFF)
        gen = GENERATOR_REGISTRY.get(atype)
        if gen is None:
            raise ValueError(f"Unknown asset generator type: {atype}")
        proto = gen(params, srng)
        plan_color = token_colors.get(sym) if token_colors else None
        if plan_color is not None:
            proto = apply_token_color(proto, plan_color)
        store.add(sym, proto)

    for tok in planned_tokens:
        if tok in store.prototypes:
            continue
        if sprite_map:
            sprite_id = sprite_map.get(tok)
            if sprite_id is None:
                raise KeyError(f"Missing sprite mapping for token '{tok}'")
            store.add(tok, make_sprite_prototype(sprite_id, sprite_manifest))
            continue
        if tok in letter_map:
            proto = make_letter_prototype(letter_map[tok], letter_style)
            store.add(tok, proto)
            continue
        if tok in shape_map:
            if not token_colors or tok not in token_colors:
                raise KeyError(f"Token '{tok}' missing color assignment required for shape prototypes")
            proto = make_template_sculpture(shape_map[tok], token_colors[tok])
            store.add(tok, proto)
            continue
        if fallback is not None:
            store.get(tok)
            continue
        raise KeyError(f"No asset mapping or fallback available for token '{tok}'")
    return store
