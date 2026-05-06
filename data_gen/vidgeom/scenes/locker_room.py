from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from ..draw import DrawCmd, DrawList
from ..events import SceneEvent

@dataclass
class Actor:
    actor_id: int
    token: str
    plan: List[Dict[str, Any]]
    plan_idx: int
    t_spawn: float
    state: str
    x: float
    y: float
    carry: List[str] = field(default_factory=list)
    door_open: float = 0.0  # 0 closed, 1 open
    hold_until: float = 0.0
    target: Optional[int] = None
    last_locker: Optional[int] = None
    target_pos: Tuple[float, float] = (0.0, 0.0)
    exit_pos: Tuple[float, float] = (1.2, 0.5)
    entry_side: str = "left"

class LockerRoomSceneGeom:
    """Geometry locker room with lockers, simple actors, and item interactions.

    Intended to be driven by rules (e.g., locker_couple_people_items).
    """
    def __init__(self):
        self.cfg: Dict[str, Any] = {}
        self.rng: Optional[np.random.Generator] = None
        self.assets = None
        self.actors: List[Actor] = []
        self.events: List[SceneEvent] = []
        self._next_actor_id = 1
        self.lockers_inventory: Dict[int, List[str]] = {}
        self.item_queue: List[str] = []  # shared staging rack

    def reset(self, cfg: Dict[str, Any], asset_store, rng: np.random.Generator):
        self.cfg = cfg or {}
        self.assets = asset_store
        self.rng = rng
        self.actors.clear()
        self.events.clear()
        self._next_actor_id = 1
        self.lockers_inventory.clear()
        self.item_queue.clear()

        self.rows = int(self.cfg.get("rows", 2))
        self.cols = int(self.cfg.get("cols", 8))
        self.walk_speed = float(self.cfg.get("walk_speed", 0.35))  # normalized units per second
        self.item_scale = float(self.cfg.get("item_scale", 0.075))
        self.tail_seconds = float(self.cfg.get("tail_seconds", 2.0))
        self.locker_capacity = int(self.cfg.get("locker_capacity", 4))
        self.allow_repeat_lockers = bool(self.cfg.get("allow_repeat_lockers", True))
        self.show_items = bool(self.cfg.get("show_items", True))
        # prefill some inventories for visual richness
        for lid in range(self.rows*self.cols):
            self.lockers_inventory[lid] = []

    @property
    def duration_tail(self) -> float:
        return self.tail_seconds

    def _locker_pos(self, locker_id: int) -> tuple[float,float,float,float]:
        # returns rect (x,y,w,h) normalized
        r = locker_id // self.cols
        c = locker_id % self.cols
        x0 = 0.08 + c*(0.84/self.cols)
        y0 = 0.12 + r*(0.54/self.rows)
        w = (0.84/self.cols)*0.92
        h = (0.54/self.rows)*0.88
        return x0, y0, w, h

    def _default_plan(self, token: str, rng: np.random.Generator) -> List[Dict[str, Any]]:
        lid = int(hash(token) % max(1, self.rows*self.cols))
        return [{"locker": lid, "action": "place", "item": token}]

    def _set_target(self, actor: Actor, locker: Optional[int]):
        if locker is None:
            actor.target = None
            actor.target_pos = actor.exit_pos
            return
        actor.target = locker
        x0, y0, w, h = self._locker_pos(locker)
        actor.target_pos = (x0 + w*0.5, y0 + h*0.90)
        actor.y = actor.target_pos[1]

    def spawn_actor(self, token: str, t: float, rng: np.random.Generator,
                    plan: Optional[List[Dict[str, Any]]] = None,
                    locker_capacity: Optional[int] = None) -> int:
        actor_id = self._next_actor_id
        self._next_actor_id += 1
        plan = plan or self._default_plan(token, rng)
        self.locker_capacity = int(locker_capacity or self.locker_capacity)
        locker = int(plan[0]["locker"])
        # choose entry side deterministically-ish per token
        sides = ["left", "right", "top", "bottom"]
        entry_side = sides[hash(token) % len(sides)]
        if entry_side == "left":
            spawn_pos = (-0.1, 0.15 + float(rng.random())*0.7)
            exit_pos = (-0.15, spawn_pos[1])
        elif entry_side == "right":
            spawn_pos = (1.1, 0.15 + float(rng.random())*0.7)
            exit_pos = (1.15, spawn_pos[1])
        elif entry_side == "top":
            spawn_pos = (0.15 + float(rng.random())*0.7, -0.1)
            exit_pos = (spawn_pos[0], -0.15)
        else:
            spawn_pos = (0.15 + float(rng.random())*0.7, 1.1)
            exit_pos = (spawn_pos[0], 1.15)

        x0, y0, w, h = self._locker_pos(locker)
        target_pos = (x0 + w*0.5, y0 + h*0.90)
        self.actors.append(Actor(
            actor_id=actor_id,
            token=str(token),
            plan=plan,
            plan_idx=0,
            t_spawn=float(t),
            state="enter",
            x=spawn_pos[0],
            y=spawn_pos[1],
            carry=[],
            door_open=0.0,
            hold_until=0.0,
            target=locker,
            last_locker=None,
            target_pos=target_pos,
            exit_pos=exit_pos,
            entry_side=entry_side,
        ))
        return actor_id

    def _perform_action(self, actor: Actor, t: float):
        if actor.plan_idx >= len(actor.plan):
            return
        step = actor.plan[actor.plan_idx]
        locker = int(step["locker"])
        inv = self.lockers_inventory.get(locker, [])
        action = step.get("action", "place")
        item = step.get("item", actor.token)
        if action == "noop":
            actor.door_open = 1.0
            actor.hold_until = t + 0.8
            actor.state = "open"
            actor.last_locker = locker
            actor.plan_idx += 1
            if actor.plan_idx < len(actor.plan):
                next_locker = int(actor.plan[actor.plan_idx]["locker"])
                self._set_target(actor, next_locker)
            else:
                self._set_target(actor, None)
            return
        if action == "take":
            if inv:
                if self.rng is not None:
                    idx = int(self.rng.integers(0, len(inv)))
                else:
                    idx = -1
                actor.carry.append(inv.pop(idx))
        else:
            if len(inv) < self.locker_capacity:
                # place carried first
                if actor.carry:
                    inv.append(actor.carry.pop(0))
                else:
                    inv.append(str(item))
        self.lockers_inventory[locker] = inv
        actor.door_open = 1.0
        actor.hold_until = t + 0.8
        actor.state = "open"
        actor.last_locker = locker
        actor.plan_idx += 1
        # set next target if any
        if actor.plan_idx < len(actor.plan):
            next_locker = int(actor.plan[actor.plan_idx]["locker"])
            self._set_target(actor, next_locker)
        else:
            self._set_target(actor, None)

    def on_token(self, token: str, seq_id: str, t: float, meta: Dict[str, Any]):
        # If no orchestrator is provided, fallback: spawn simple actor with default plan.
        if seq_id.lower().endswith("items") or seq_id == "S2":
            self.item_queue.append(str(token))
            return
        self.spawn_actor(str(token), t=t, rng=self.rng)

    def step(self, t: float, dt: float):
        # process self.events into external events
        # (controller will pop and dispatch, but we also keep ones scheduled internally)
        # update actor movement/state
        busy_lockers: Dict[int, int] = {}
        for a in self.actors:
            # locker is busy if actor is currently at/open on that locker
            if a.state in ("at_locker", "open") and a.target is not None:
                busy_lockers[a.target] = a.actor_id

        for a in self.actors:
            tx, ty = a.target_pos
            if a.state in ("enter", "move", "wait"):
                # if heading to a locker that is currently busy (by someone else) and close, wait
                busy_other = (a.target is not None and a.target in busy_lockers and busy_lockers[a.target] != a.actor_id)
                dx = tx - a.x
                dy = ty - a.y
                dist = math.hypot(dx, dy)
                step = self.walk_speed * dt
                if busy_other and dist < 0.06:
                    a.state = "wait"
                else:
                    a.state = "move" if a.state == "wait" else a.state
                    if dist <= step:
                        a.x, a.y = tx, ty
                        if a.target is None:
                            a.state = "leave"
                        else:
                            a.state = "at_locker"
                    else:
                        if dist > 0:
                            a.x += step * dx / dist
                            a.y += step * dy / dist
            if a.state == "at_locker":
                a.door_open = min(1.0, a.door_open + dt*4.0)
                self._perform_action(a, t)
            elif a.state == "open":
                if t >= a.hold_until:
                    if a.target is None:
                        a.state = "leave"
                        a.target_pos = a.exit_pos
                        a.door_open = 0.0
                    else:
                        a.state = "move"
            elif a.state == "move" and a.target is None:
                a.state = "leave"
                a.target_pos = a.exit_pos
                a.door_open = 0.0
            elif a.state == "leave":
                a.door_open = max(0.0, a.door_open - dt*3.0)
                dx = a.exit_pos[0] - a.x
                dy = a.exit_pos[1] - a.y
                dist = math.hypot(dx, dy)
                step = self.walk_speed * dt * 1.1
                if dist <= step:
                    a.x, a.y = a.exit_pos
                else:
                    a.x += step * dx / dist
                    a.y += step * dy / dist
        # cleanup actors offscreen
        self.actors = [a for a in self.actors if (-0.3 < a.x < 1.3 and -0.3 < a.y < 1.3)]

    def pop_events(self) -> List[SceneEvent]:
        evs = self.events
        self.events = []
        return evs

    def draw(self, t: float) -> DrawList:
        draw: DrawList = []
        # Background: warm locker room wall + floor
        draw.append(DrawCmd(kind="gradient", z=-10, data=dict(rect=(0,0,1,1), c0=(32,28,26), c1=(18,16,16), vertical=True)))
        draw.append(DrawCmd(kind="rect", z=-9, data=dict(rect=(0,0.78,1,0.22), color=(20,18,18))))
        # floor stripes
        for k in range(8):
            y = 0.80 + k*0.025
            c = (26,24,24) if k % 2 == 0 else (22,20,20)
            draw.append(DrawCmd(kind="rect", z=-8, data=dict(rect=(0,y,1,0.018), color=c)))

        # Lockers grid
        for lid in range(self.rows*self.cols):
            x0, y0, w, h = self._locker_pos(lid)
            # base locker
            base = (54, 62, 70) if (lid % 2 == 0) else (48, 56, 64)
            draw.append(DrawCmd(kind="rect", z=-5, data=dict(rect=(x0, y0, w, h), color=base)))
            # inner panel
            draw.append(DrawCmd(kind="rect", z=-4, data=dict(rect=(x0+w*0.06, y0+h*0.06, w*0.88, h*0.88), color=(32, 36, 42))))
            # handle
            draw.append(DrawCmd(kind="rect", z=-3, data=dict(rect=(x0+w*0.82, y0+h*0.42, w*0.06, h*0.16), color=(150, 150, 155))))
        # Draw open doors and inventory hint
        open_by_locker: Dict[int, float] = {}
        for a in self.actors:
            if a.last_locker is not None:
                open_by_locker[a.last_locker] = max(open_by_locker.get(a.last_locker, 0.0), a.door_open)

        for lid, op in open_by_locker.items():
            if op <= 0.05:
                continue
            x0, y0, w, h = self._locker_pos(lid)
            # door polygon that swings open by reducing width; simple perspective cheat
            swing = 0.9*op
            door_w = w*(1.0 - 0.85*swing)
            # Door anchored at left edge
            pts = [(x0, y0), (x0+door_w, y0+h*0.05), (x0+door_w, y0+h*0.95), (x0, y0+h)]
            draw.append(DrawCmd(kind="poly", z=2, data=dict(points=pts, color=(62, 70, 80))))
            # inside glow
            glow = int(40 + 80*op)
            draw.append(DrawCmd(kind="rect", z=1, data=dict(rect=(x0+w*0.15, y0+h*0.20, w*0.60, h*0.60), color=(glow,glow,glow))))

        # Locker contents (no occlusion: grid layout)
        cell_rows = 2
        cell_cols = max(2, int(math.ceil(self.locker_capacity / cell_rows)))
        if self.show_items:
            for lid, inv in self.lockers_inventory.items():
                if not inv or open_by_locker.get(lid, 0.0) <= 0.0:
                    continue
                x0, y0, w, h = self._locker_pos(lid)
                for idx, tok in enumerate(inv[: self.locker_capacity]):
                    r = idx // cell_cols
                    c = idx % cell_cols
                    px = x0 + w*0.2 + c*(w*0.25)
                    py = y0 + h*0.25 + r*(h*0.25)
                    draw.append(DrawCmd(kind="instance", z=4, data=dict(asset_id=str(tok), pos=(px, py), scale=self.item_scale*0.7, rot=0.0)))

        # Locker numbers
        lid_num = 0
        for lid in range(self.rows*self.cols):
            x0, y0, w, h = self._locker_pos(lid)
            draw.append(DrawCmd(kind="text", z=6, data=dict(pos=(x0+w*0.10, y0+h*0.90), text=str(lid_num), color=(200,200,205), scale=0.28, thickness=1)))
            lid_num += 1

        # Actors + carried items
        for a in self.actors:
            # body diversity via token-derived RNG
            arng = np.random.default_rng(abs(hash(a.token)) & 0xFFFFFFFF)
            palette = [
                (200, 180, 150), (150, 200, 180), (180, 150, 210),
                (180, 200, 120), (120, 180, 200), (210, 160, 120),
                (120, 200, 150), (190, 170, 220),
            ]
            base_c = palette[int(arng.integers(0, len(palette)))]
            accent_c = tuple(min(255, int(c + arng.integers(30, 90))) for c in base_c)
            mode = arng.choice(["capsule", "triangle", "round", "square"])
            body_h = 0.085
            body_w = 0.04 if mode != "triangle" else 0.05
            if mode == "capsule":
                draw.append(DrawCmd(kind="rect", z=9, data=dict(rect=(a.x-body_w/2, a.y-body_h, body_w, body_h*1.1), color=_darken(base_c, 0.45))))
                draw.append(DrawCmd(kind="circle", z=9, data=dict(center=(a.x, a.y-body_h), radius=body_w*0.55, color=_darken(base_c, 0.45))))
            elif mode == "triangle":
                pts = [(a.x, a.y-body_h*1.2), (a.x-body_w, a.y+body_h*0.1), (a.x+body_w, a.y+body_h*0.1)]
                draw.append(DrawCmd(kind="poly", z=9, data=dict(points=pts, color=_darken(base_c, 0.5))))
            elif mode == "round":
                draw.append(DrawCmd(kind="circle", z=9, data=dict(center=(a.x, a.y-body_h*0.4), radius=body_w*0.9, color=_darken(base_c, 0.45))))
            else:
                draw.append(DrawCmd(kind="rect", z=9, data=dict(rect=(a.x-body_w/2, a.y-body_h*1.05, body_w, body_h*1.4), color=_darken(base_c, 0.5))))

            # head
            draw.append(DrawCmd(kind="circle", z=10, data=dict(center=(a.x, a.y-0.08), radius=0.030, color=accent_c)))
            # belt/stripe
            draw.append(DrawCmd(kind="rect", z=10, data=dict(rect=(a.x-0.024, a.y-0.045, 0.048, 0.012), color=_lighten(base_c, 0.35))))
            # accessories
            if arng.random() < 0.5:
                draw.append(DrawCmd(kind="rect", z=11, data=dict(rect=(a.x-0.022, a.y-0.095, 0.044, 0.012), color=_darken(accent_c, 0.6))))  # hat brim
                draw.append(DrawCmd(kind="rect", z=11, data=dict(rect=(a.x-0.016, a.y-0.110, 0.032, 0.018), color=_darken(accent_c, 0.4))))
            if arng.random() < 0.35:
                draw.append(DrawCmd(kind="circle", z=8, data=dict(center=(a.x-0.03, a.y-0.01), radius=0.016, color=_darken(accent_c,0.3))))  # shoulder bag
                draw.append(DrawCmd(kind="line", z=8, data=dict(p0=(a.x-0.03, a.y-0.01), p1=(a.x+0.01, a.y-0.06), width=0.005, color=_darken(accent_c,0.5))))
            # label dot by token
            dot_c = (accent_c[2], accent_c[0], accent_c[1])
            draw.append(DrawCmd(kind="circle", z=12, data=dict(center=(a.x, a.y-0.015), radius=0.012, color=dot_c)))
            # carried item(s)
            if a.carry:
                offs = 0.0
                for ci in a.carry[:2]:
                    draw.append(DrawCmd(kind="instance", z=12, data=dict(asset_id=ci, pos=(a.x+0.045, a.y-0.03-offs), scale=self.item_scale, rot=0.2*math.sin(2*math.pi*0.7*(t-a.t_spawn)))))
                    offs += 0.03

        # Queued items on rack
        for i, tok in enumerate(self.item_queue[:6]):
            x = 0.12 + i*0.12
            y = 0.69
            draw.append(DrawCmd(kind="instance", z=6, data=dict(asset_id=tok, pos=(x, y), scale=self.item_scale*0.85, rot=0.0)))

        draw.append(DrawCmd(kind="text", z=100, data=dict(pos=(0.02, 0.98), text=f"LockerRoom | queued={len(self.item_queue)}", color=(230,230,235), scale=0.45, thickness=1)))
        return draw

def _darken(color, f: float):
    return (int(color[0]*f), int(color[1]*f), int(color[2]*f))

def _lighten(color, f: float):
    return (int(color[0] + (255-color[0])*f), int(color[1] + (255-color[1])*f), int(color[2] + (255-color[2])*f))

def _token_rng(token: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(token)) & 0xFFFFFFFF)
