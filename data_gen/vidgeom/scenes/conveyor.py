from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import numpy as np

from ..draw import DrawCmd, DrawList
from ..events import SceneEvent

@dataclass
class ConveyorItem:
    token: str
    lane: int
    t_spawn: float
    rot0: float
    wobble: float
    seq_index: int

class ConveyorSceneV2:
    """Geometry-only conveyor factory scene.

    Coordinates are normalized to [0,1].
    Belts move downwards; items spawn near top and exit bottom.
    """
    def __init__(self):
        self.cfg: Dict[str, Any] = {}
        self.rng: Optional[np.random.Generator] = None
        self.assets = None
        self.items: List[ConveyorItem] = []
        self.events: List[SceneEvent] = []
        self.flash_until: float = -1.0
        self._frame_debug: List[Dict[str, Any]] = []
        self._frame_debug_meta: Dict[str, Any] = {}
        self.view_y_min = 0.0
        self.view_y_max = 1.0
        self.visible_margin_top = 0.0
        self.visible_margin_bottom = 0.0

    def reset(self, cfg: Dict[str, Any], asset_store, rng: np.random.Generator):
        self.cfg = cfg or {}
        self.assets = asset_store
        self.rng = rng
        self.items.clear()
        self.events.clear()
        self.flash_until = -1.0

        self.belts = int(self.cfg.get("belts", 1))
        self.belt_speed = float(self.cfg.get("belt_speed", 0.65))  # normalized units per second
        self.item_scale = float(self.cfg.get("item_scale", 0.10))
        self.lanes_pad = float(self.cfg.get("lanes_pad", 0.08))
        self.scanner_y = float(self.cfg.get("scanner_y", 0.38))
        self.scanner_flash = 0.0  # disable flash by default
        self.tail_seconds = float(self.cfg.get("tail_seconds", 1.2))
        self.show_hud_label = bool(self.cfg.get("show_hud_label", False))
        self.show_lane_labels = bool(self.cfg.get("show_lane_labels", False))
        self.enable_wobble = bool(self.cfg.get("enable_wobble", True))
        self.enable_rotation = bool(self.cfg.get("enable_rotation", True))
        self.wobble_scale = float(self.cfg.get("lateral_wobble", 0.012))
        self.spawn_y = float(self.cfg.get("spawn_y", 0.05))
        self.view_y_min = float(self.cfg.get("view_y_min", 0.0))
        self.view_y_max = float(self.cfg.get("view_y_max", 1.0))
        self.visible_margin_top = float(self.cfg.get("visible_margin_top", 0.0))
        self.visible_margin_bottom = float(self.cfg.get("visible_margin_bottom", 0.0))

    def _token_hash(self, token: str) -> int:
        return hash(token) & 0xFFFFFFFF

    def _token_orientation(self, token: str) -> float:
        if not self.enable_rotation:
            return 0.0
        h = self._token_hash(token)
        return -0.6 + (h / 0xFFFFFFFF) * 1.2

    def _token_wobble(self, token: str) -> float:
        if not self.enable_wobble:
            return 0.0
        h = self._token_hash(token)
        return 0.4 + ((h >> 8) / 0xFFFFFF) * 0.6

    @property
    def duration_tail(self) -> float:
        return self.tail_seconds

    def on_token(self, token: str, seq_id: str, t: float, meta: Dict[str, Any]):
        lane = int(meta.get("lane", 0))
        lane = max(0, min(self.belts - 1, lane))
        seq_index = int(meta.get("seq_index", meta.get("index", -1)))
        # spawn
        rot0 = self._token_orientation(token)
        wobble = self._token_wobble(token)
        self.items.append(
            ConveyorItem(
                token=str(token),
                lane=lane,
                t_spawn=float(t),
                rot0=rot0,
                wobble=wobble,
                seq_index=seq_index,
            )
        )

    def step(self, t: float, dt: float):
        # Remove items past bottom; trigger scanner flashes
        alive = []
        for it in self.items:
            y = self._item_y(it, t)
            if y < 1.15:
                alive.append(it)
            # scanner
            if (y >= self.scanner_y) and (y < self.scanner_y + self.belt_speed * dt * 1.5):
                self.flash_until = max(self.flash_until, t + self.scanner_flash)
        self.items = alive

    def pop_events(self) -> List[SceneEvent]:
        evs = self.events
        self.events = []
        return evs

    def _lane_x(self, lane: int) -> float:
        # belts centered in middle; lanes side-by-side
        total_w = 1.0 - 2*self.lanes_pad
        if self.belts == 1:
            return 0.5
        lane_w = total_w / self.belts
        return self.lanes_pad + lane_w*(lane + 0.5)

    def _item_y(self, it: ConveyorItem, t: float) -> float:
        return self.spawn_y + (t - it.t_spawn) * self.belt_speed

    def _item_visibility_state(self, y: float) -> Dict[str, Any]:
        half = 0.5 * self.item_scale
        top = y - half
        bottom = y + half
        min_visible_top = self.view_y_min + self.visible_margin_top
        max_visible_bottom = self.view_y_max + self.visible_margin_bottom
        if top < min_visible_top:
            return {"visible": False, "visibility": "above_frame"}
        if bottom > max_visible_bottom:
            return {"visible": True, "visibility": "leaving_bottom"}
        return {"visible": True, "visibility": "onscreen"}

    def draw(self, t: float) -> DrawList:
        draw: DrawList = []
        self._frame_debug = []
        self._frame_debug_meta = {
            "view_bounds": {
                "y_min": self.view_y_min + self.visible_margin_top,
                "y_max": self.view_y_max + self.visible_margin_bottom,
            }
        }
        # Background: industrial wall gradient + floor
        draw.append(DrawCmd(kind="gradient", z=-10, data=dict(rect=(0,0,1,1), c0=(26,30,38), c1=(18,20,26), vertical=True)))
        draw.append(DrawCmd(kind="rect", z=-9, data=dict(rect=(0,0.78,1,0.22), color=(12,14,18))))

        # Overhead lights (animated intensity)
        flick = 0.5 + 0.5*math.sin(2*math.pi*(0.17*t) + 1.2)
        light_c = (int(80 + 40*flick), int(85 + 45*flick), int(90 + 55*flick))
        for k in range(3):
            x = 0.18 + k*0.32
            draw.append(DrawCmd(kind="rect", z=-8, data=dict(rect=(x-0.08, 0.03, 0.16, 0.05), color=light_c)))
            draw.append(DrawCmd(kind="rect", z=-8, data=dict(rect=(x-0.075, 0.035, 0.15, 0.04), color=(200,200,210))))

        # Belts
        total_w = 1.0 - 2*self.lanes_pad
        lane_w = total_w / self.belts
        phase = (t * self.belt_speed) % 1.0
        for lane in range(self.belts):
            x0 = self.lanes_pad + lane*lane_w + lane_w*0.08
            bw = lane_w*0.84
            # belt base
            draw.append(DrawCmd(kind="rect", z=-5, data=dict(rect=(x0, 0.10, bw, 0.85), color=(25, 27, 32))))
            # moving stripes
            draw.append(DrawCmd(kind="stripe_fill", z=-4, data=dict(
                rect=(x0, 0.10, bw, 0.85),
                c0=(28,30,36),
                c1=(40,44,52),
                period=0.10,
                width=0.045,
                angle=math.pi/3,
                phase=phase + lane*0.07,
            )))
            # rails
            draw.append(DrawCmd(kind="line", z=-3, data=dict(p0=(x0,0.10), p1=(x0,0.95), width=0.004, color=(85,90,96))))
            draw.append(DrawCmd(kind="line", z=-3, data=dict(p0=(x0+bw,0.10), p1=(x0+bw,0.95), width=0.004, color=(85,90,96))))

        if self.show_lane_labels:
            base_y = 0.045
            label_h = 0.05
            label_w = min(0.14, max(0.08, lane_w * 0.4)) if self.belts else 0.12
            for lane in range(self.belts):
                x = self._lane_x(lane)
                draw.append(
                    DrawCmd(
                        kind="rect",
                        z=-2,
                        data=dict(
                            rect=(x - label_w / 2, base_y - label_h / 2, label_w, label_h),
                            color=(30, 32, 42),
                        ),
                    )
                )
                draw.append(
                    DrawCmd(
                        kind="text",
                        z=5,
                        data=dict(
                            pos=(x - label_w / 4, base_y + 0.012),
                            text=str(lane),
                            color=(230, 230, 235),
                            scale=0.55,
                            thickness=1,
                        ),
                    )
                )

        # Scanner line
        draw.append(DrawCmd(kind="line", z=-1, data=dict(p0=(0.06,self.scanner_y), p1=(0.94,self.scanner_y), width=0.003, color=(160,190,255))))

        # Flash overlay if active (approximate by bright rectangle)
        if t < self.flash_until:
            a = 0.5 + 0.5*math.sin(40*(self.flash_until - t))
            c = (int(60*a), int(80*a), int(120*a))
            draw.append(DrawCmd(kind="rect", z=50, data=dict(rect=(0,0,1,1), color=c)))

        # Items
        cleaned: List[ConveyorItem] = []
        for it in self.items:
            y = self._item_y(it, t)
            x = self._lane_x(it.lane)
            # slight lateral wobble
            wob = 0.0
            if self.enable_wobble and self.wobble_scale > 0.0:
                wob = self.wobble_scale * math.sin(2*math.pi*(0.9*(t - it.t_spawn)) + it.rot0) * it.wobble
            rot = it.rot0
            vis_info = self._item_visibility_state(y)
            if vis_info.get("visible"):
                draw.append(
                    DrawCmd(
                        kind="instance",
                        z=10 + y,
                        data=dict(
                            asset_id=it.token,
                            pos=(x + wob, y),
                            scale=self.item_scale,
                            rot=rot,
                        ),
                    )
                )
            self._frame_debug.append(
                {
                    "token": it.token,
                    "lane": it.lane,
                    "x": x + wob,
                    "y": y,
                    "time": t,
                    "spawn_time": it.t_spawn,
                    "seq_index": it.seq_index,
                    "visible": vis_info["visible"],
                    "visibility": vis_info["visibility"],
                }
            )
            if y < 1.2:
                cleaned.append(it)
        self.items = cleaned

        # HUD label
        if self.show_hud_label:
            draw.append(DrawCmd(kind="text", z=100, data=dict(pos=(0.02, 0.98), text=f"Conveyor v2 | belts={self.belts}", color=(220,220,225), scale=0.45, thickness=1)))
        return draw

    def frame_debug_snapshot(self) -> Dict[str, Any]:
        return {
            "items": list(self._frame_debug),
            "view_bounds": dict(self._frame_debug_meta.get("view_bounds", {})),
        }
