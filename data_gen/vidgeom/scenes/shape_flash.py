from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..draw import DrawCmd, DrawList
from ..events import SceneEvent


def _resolve_color(value: Optional[Any], default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return default
    return default


@dataclass
class FlashToken:
    token: str
    t_spawn: float


class ShapeFlashScene:
    """Displays centered sculptures sequentially on a dark background."""

    def __init__(self) -> None:
        self.cfg: Dict[str, Any] = {}
        self.assets = None
        self.rng = None
        self.items: List[FlashToken] = []
        self.events: List[SceneEvent] = []
        self.hold_seconds = 1.5
        self.item_scale = 0.32
        self.tail_seconds = 1.0
        self.background_color = (0, 0, 0)
        self.frame_color = (28, 28, 32)
        self.glow_color = (46, 46, 52)

    def reset(self, cfg: Dict[str, Any], asset_store, rng):
        self.cfg = cfg or {}
        self.assets = asset_store
        self.rng = rng
        self.items.clear()
        self.events.clear()
        self.hold_seconds = float(self.cfg.get("hold_seconds", 1.5))
        self.item_scale = float(self.cfg.get("item_scale", 0.32))
        self.tail_seconds = float(self.cfg.get("tail_seconds", self.hold_seconds + 0.4))
        self.background_color = _resolve_color(self.cfg.get("background_color"), (0, 0, 0))
        self.frame_color = _resolve_color(self.cfg.get("frame_color"), (32, 32, 38))
        self.glow_color = _resolve_color(self.cfg.get("glow_color"), (46, 46, 58))

    @property
    def duration_tail(self) -> float:
        return max(self.tail_seconds, self.hold_seconds)

    def on_token(self, token: str, seq_id: str, t: float, meta: Dict[str, Any]):
        del seq_id, meta
        self.items.append(FlashToken(token=str(token), t_spawn=float(t)))

    def step(self, t: float, dt: float):
        del dt
        cutoff = t - self.hold_seconds - 1e-6
        self.items = [it for it in self.items if it.t_spawn > cutoff]

    def pop_events(self) -> List[SceneEvent]:
        evs = self.events
        self.events = []
        return evs

    def draw(self, t: float) -> DrawList:
        draw: DrawList = []
        draw.append(
            DrawCmd(kind="fill", z=-10, data=dict(color=self.background_color))
        )
        draw.append(
            DrawCmd(
                kind="rect",
                z=-9,
                data=dict(rect=(0.08, 0.10, 0.84, 0.80), color=self.frame_color),
            )
        )
        draw.append(
            DrawCmd(
                kind="rect",
                z=-8,
                data=dict(rect=(0.18, 0.20, 0.64, 0.60), color=self.glow_color),
            )
        )

        for item in self.items:
            if not (item.t_spawn <= t <= item.t_spawn + self.hold_seconds):
                continue
            draw.append(
                DrawCmd(
                    kind="instance",
                    z=10,
                    data=dict(
                        asset_id=item.token,
                        pos=(0.5, 0.5),
                        scale=self.item_scale,
                        rot=0.0,
                    ),
                )
            )
        return draw
