from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import numpy as np

from ..draw import DrawCmd, DrawList
from ..events import SceneEvent

@dataclass
class HubItem:
    token: str
    t_spawn: float
    state: str  # "in", "done"
    x: float
    y: float
    dest_x: float
    dest_y: float
    rot0: float
    path_phase: float
    slot: int
    t_arrive: Optional[float] = None

class SortingHubSceneGeom:
    """Junction sorting hub with a controllable gate."""
    def __init__(self):
        self.cfg: Dict[str, Any] = {}
        self.rng: Optional[np.random.Generator] = None
        self.assets = None
        self.items: List[HubItem] = []
        self.events: List[SceneEvent] = []
        self.num_slots: int = 3
        self.slot_x: List[float] = []
        self.slot_y: float = 0.82

    def reset(self, cfg: Dict[str, Any], asset_store, rng: np.random.Generator):
        self.cfg = cfg or {}
        self.assets = asset_store
        self.rng = rng
        self.items.clear()
        self.events.clear()
        self.speed = float(self.cfg.get("belt_speed", 0.55))
        self.item_scale = float(self.cfg.get("item_scale", 0.10))
        self.tail_seconds = float(self.cfg.get("tail_seconds", 1.4))
        self.num_slots = int(self.cfg.get("num_slots", 4))
        self.num_slots = max(1, min(8, self.num_slots))
        self.slot_y = float(self.cfg.get("slot_y", 0.82))
        xs = np.linspace(0.12, 0.88, self.num_slots)
        self.slot_x = [float(x) for x in xs]

    @property
    def duration_tail(self) -> float:
        return self.tail_seconds

    def on_token(self, token: str, seq_id: str, t: float, meta: Dict[str, Any]):
        # Spawn inbound items; meta may contain a target slot index
        rot0 = float(self.rng.uniform(-0.7, 0.7)) if self.rng is not None else 0.0
        slot = meta.get("slot", None)
        if slot is None:
            slot = hash(token) % self.num_slots
        slot = max(0, min(self.num_slots - 1, int(slot)))
        dest_x = self.slot_x[slot]
        dest_y = self.slot_y
        self.items.append(
            HubItem(
                token=str(token),
                t_spawn=float(t),
                state="in",
                x=0.5,
                y=0.03,
                dest_x=dest_x,
                dest_y=dest_y,
                rot0=rot0,
                path_phase=float(self.rng.uniform(0, 1)) if self.rng is not None else 0.0,
                slot=slot,
            )
        )

    def step(self, t: float, dt: float):
        # Move items along paths: in -> destination slot -> done
        alive: List[HubItem] = []
        latest_by_slot: Dict[int, HubItem] = {}
        for it in self.items:
            if it.state == "in":
                dx = it.dest_x - it.x
                dy = it.dest_y - it.y
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    dist = 1e-6
                step = self.speed * dt
                if dist <= step:
                    it.x = it.dest_x
                    it.y = it.dest_y + self.speed * dt * 0.3
                    if it.y >= 1.05:
                        it.state = "done"
                    else:
                        it.state = "parked"
                        it.t_arrive = t
                else:
                    it.x += dx / dist * step
                    it.y += dy / dist * step
            if it.state in ("in", "parked"):
                alive.append(it)
                latest = latest_by_slot.get(it.slot)
                if latest is None or it.t_spawn >= latest.t_spawn:
                    latest_by_slot[it.slot] = it
        # keep only newest per slot if parked; keep inbound regardless; expire after dwell
        filtered: List[HubItem] = []
        for it in alive:
            if it.state == "parked":
                if latest_by_slot.get(it.slot) is it:
                    if it.t_arrive is None or (t - it.t_arrive) < 1.0:
                        filtered.append(it)
            else:
                filtered.append(it)
        self.items = filtered

    def pop_events(self) -> List[SceneEvent]:
        evs = self.events
        self.events = []
        return evs

    def draw(self, t: float) -> DrawList:
        draw: DrawList = []
        draw.append(DrawCmd(kind="gradient", z=-10, data=dict(rect=(0,0,1,1), c0=(16,18,24), c1=(8,10,14), vertical=True)))
        # inbound belt from top into hub
        phase = (t * self.speed) % 1.0
        draw.append(DrawCmd(kind="rect", z=-6, data=dict(rect=(0.42,0.05,0.16,0.70), color=(22,24,30))))
        draw.append(DrawCmd(kind="stripe_fill", z=-5, data=dict(rect=(0.42,0.05,0.16,0.70), c0=(24,26,32), c1=(40,44,55),
                                                          period=0.10, width=0.045, angle=math.pi/2, phase=phase*0.3)))

        # Output belts/slots
        slot_w = 0.12
        slot_h = 0.10
        for i, x in enumerate(self.slot_x):
            # belt from hub center to slot
            x_start = min(x, 0.5)
            bw = abs(x - 0.5)
            draw.append(DrawCmd(kind="rect", z=-6, data=dict(rect=(x_start, self.slot_y-0.12, bw, 0.10), color=(22,24,30))))
            draw.append(DrawCmd(kind="stripe_fill", z=-5, data=dict(rect=(x_start, self.slot_y-0.12, bw, 0.10),
                                                              c0=(24,26,32), c1=(40,44,55),
                                                              period=0.10, width=0.045, angle=0.0 if x >= 0.5 else math.pi, phase=phase*0.3 + 0.05*i)))
            # slot box
            draw.append(DrawCmd(kind="rect", z=-4, data=dict(rect=(x - slot_w/2, self.slot_y, slot_w, slot_h), color=(26,28,34))))
            draw.append(DrawCmd(kind="rect", z=-3, data=dict(rect=(x - slot_w/2 + 0.01, self.slot_y+0.01, slot_w-0.02, slot_h-0.02), color=(34,36,44))))
            draw.append(DrawCmd(kind="text", z=5, data=dict(pos=(x, self.slot_y + slot_h/2), text=str(i), color=(180,200,220), scale=0.35, thickness=1)))

        # Items (inbound + newest per slot)
        for it in self.items:
            rot = it.rot0 + 0.12*math.sin(2*math.pi*0.6*(t - it.t_spawn))
            draw.append(DrawCmd(kind="instance", z=10+it.y, data=dict(asset_id=it.token, pos=(it.x, it.y), scale=self.item_scale, rot=rot)))

        draw.append(DrawCmd(kind="text", z=100, data=dict(pos=(0.02, 0.98), text=f"SortingHub | slots={self.num_slots}", color=(220,220,225), scale=0.45, thickness=1)))
        return draw
