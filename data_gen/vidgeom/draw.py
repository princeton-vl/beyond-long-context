from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

Color = Tuple[int, int, int]  # RGB
Vec2 = Tuple[float, float]

@dataclass
class CirclePrim:
    center: Vec2          # local coords
    radius: float         # local units (relative to prototype space)
    color: Color
    z: float = 0.0

@dataclass
class PolyPrim:
    points: List[Vec2]    # local coords
    color: Color
    z: float = 0.0

@dataclass
class LinePrim:
    p0: Vec2
    p1: Vec2
    width: float
    color: Color
    z: float = 0.0

@dataclass
class ImagePrim:
    pixels: Any  # BGRA uint8 array
    width: float = 1.0
    height: float = 1.0
    z: float = 0.0

Primitive = Union[CirclePrim, PolyPrim, LinePrim, ImagePrim]

@dataclass
class Prototype:
    primitives: List[Primitive]
    # optional metadata for culling / convenience
    radius_hint: float = 0.5

@dataclass
class DrawCmd:
    kind: Literal[
        "fill",
        "gradient",
        "stripe_fill",
        "rect",
        "poly",
        "circle",
        "line",
        "text",
        "instance",
    ]
    z: float = 0.0
    # Generic payload. Renderer interprets by kind.
    data: Dict[str, Any] = field(default_factory=dict)

DrawList = List[DrawCmd]
