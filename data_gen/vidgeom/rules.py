from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Type

import numpy as np
from .events import TokenEvent, SceneEvent


class Scheduler(Protocol):
    def schedule_token(self, ev: TokenEvent) -> None: ...
    def schedule_scene(self, ev: SceneEvent) -> None: ...


class Rule:
    """A small, easy-to-add rule.

    Return True from on_token() to consume the event and prevent scene.on_token().
    """
    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        return False

    def on_scene_event(self, ev: SceneEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> None:
        return None


RULE_REGISTRY: Dict[str, Type[Rule]] = {}


def register_rule(name: str):
    def deco(cls):
        RULE_REGISTRY[name] = cls
        return cls
    return deco


@register_rule("conveyor_interleaver")
class ConveyorInterleaver(Rule):
    """Assign lanes for multiple interleaved sequences on the conveyor."""
    def __init__(self, seq_ids: Sequence[str], mode: str = "shared",
                 seq_to_lane: Optional[Dict[str, int]] = None,
                 default_lane: int = 0, belts: int = 1):
        self.seq_ids = [str(s) for s in seq_ids]
        self.mode = str(mode)
        self.seq_to_lane = {str(k): int(v) for k, v in (seq_to_lane or {}).items()}
        self.default_lane = int(default_lane)
        self.belts = max(1, int(belts))

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        if ev.seq_id not in self.seq_ids:
            return False
        if self.mode == "shared":
            lane = self.default_lane
        elif self.mode == "random":
            lane = int(rng.integers(0, self.belts)) if rng is not None else self.default_lane
        else:
            lane = self.seq_to_lane.get(ev.seq_id)
            if lane is None:
                lane = hash(ev.seq_id) % self.belts
        ev.meta["lane"] = max(0, min(self.belts - 1, int(lane)))
        return False


@register_rule("conveyor_lane_from_sequence")
class ConveyorLaneFromSequence(Rule):
    """Routes a primary conveyor sequence using lanes provided by a control sequence."""

    def __init__(self, seq_main: str = "S_items", seq_lane: str = "S_lane", belts: int = 1, default_lane: int = 0):
        self.seq_main = str(seq_main)
        self.seq_lane = str(seq_lane)
        self.belts = max(1, int(belts))
        self.default_lane = max(0, int(default_lane))
        self._lane_queue: List[int] = []

    def _lane_from_token(self, token: str) -> int:
        try:
            raw = int(token)
        except Exception:
            raw = hash(token)
        lane = raw % self.belts
        if lane < 0:
            lane += self.belts
        return lane

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        belts = max(1, self.belts)

        if ev.seq_id == self.seq_lane:
            lane = self._lane_from_token(ev.token)
            lane = max(0, min(belts - 1, lane))
            self._lane_queue.append(lane)
            return True  # consume control sequence tokens so they do not spawn items

        if ev.seq_id != self.seq_main:
            return False

        lane_val = ev.meta.get("lane")
        lane: int
        if lane_val is not None:
            try:
                lane = int(lane_val)
            except Exception:
                lane = self.default_lane
        elif self._lane_queue:
            lane = self._lane_queue.pop(0)
        else:
            lane = self.default_lane

        lane %= belts
        if lane < 0:
            lane += belts
        ev.meta["lane"] = lane
        return False


@register_rule("lane_policy_markov")
class LanePolicyMarkov(Rule):
    """Sets ev.meta['lane'] using a Markov persistence rule."""
    def __init__(self, belts: int = 1, p_same: float = 0.75):
        self.belts = int(belts)
        self.p_same = float(p_same)
        self._last_lane: Optional[int] = None

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        if self.belts <= 1:
            ev.meta["lane"] = 0
            return False
        if self._last_lane is None:
            lane = int(rng.integers(0, self.belts))
        else:
            if float(rng.random()) < self.p_same:
                lane = self._last_lane
            else:
                choices = [i for i in range(self.belts) if i != self._last_lane]
                lane = int(choices[int(rng.integers(0, len(choices)))])
        self._last_lane = lane
        ev.meta["lane"] = lane
        return False


@register_rule("spawn_spacing")
class SpawnSpacing(Rule):
    """Enforces time gaps between spawns per lane by delaying token events."""

    def __init__(
        self,
        min_gap_seconds: float = 0.15,
        default_lane: int = 0,
        max_gap_seconds: Optional[float] = None,
        per_lane: bool = True,
    ):
        self.min_gap = max(0.0, float(min_gap_seconds))
        self.max_gap = None
        if max_gap_seconds is not None:
            max_gap = float(max_gap_seconds)
            if max_gap > self.min_gap:
                self.max_gap = max_gap
        self.default_lane = int(default_lane)
        self.per_lane = bool(per_lane)
        self._last_t_by_lane: Dict[int, float] = {}
        self._gap_target_by_lane: Dict[int, float] = {}

    def _next_gap(self, lane: int, rng: Optional[np.random.Generator]) -> float:
        if self.max_gap is None:
            return self.min_gap
        if rng is None:
            return self.max_gap
        return float(rng.uniform(self.min_gap, self.max_gap))

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        key_lane = int(ev.meta.get("lane", self.default_lane)) if self.per_lane else -1
        last = self._last_t_by_lane.get(key_lane, -1e9)
        target_gap = self._gap_target_by_lane.get(key_lane)
        if target_gap is None:
            target_gap = self._next_gap(key_lane, rng)
            self._gap_target_by_lane[key_lane] = target_gap

        desired_time = last + target_gap
        if ev.t < desired_time:
            scheduler.schedule_token(
                TokenEvent(t=desired_time, token=ev.token, seq_id=ev.seq_id, index=ev.index, meta=dict(ev.meta))
            )
            return True

        self._last_t_by_lane[key_lane] = ev.t
        if self.max_gap is not None:
            self._gap_target_by_lane[key_lane] = self._next_gap(key_lane, rng)
        else:
            self._gap_target_by_lane.pop(key_lane, None)
        return False


@register_rule("route_by_control")
class RouteByControl(Rule):
    """Routing for sorting hub."""
    def __init__(self, seq_main: str = "S1", seq_ctrl: Optional[str] = "S_ctrl", num_slots: int = 4):
        self.seq_main = str(seq_main)
        self.seq_ctrl = str(seq_ctrl) if seq_ctrl is not None else None
        self.num_slots = int(num_slots)
        self.queue: List[int] = []

    def _slot_from_token(self, token: str) -> int:
        return hash(token) % max(1, self.num_slots)

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        if ev.seq_id == self.seq_ctrl and self.seq_ctrl is not None:
            slot = self._slot_from_token(ev.token)
            self.queue.append(slot)
            return True  # consume control tokens

        if ev.seq_id != self.seq_main:
            return False

        if self.queue:
            slot = self.queue.pop(0)
        else:
            slot = self._slot_from_token(ev.token)
        ev.meta["slot"] = slot
        return False


@register_rule("locker_orchestrator")
class LockerOrchestrator(Rule):
    """Orchestrates locker-room behavior with optional locker and item control sequences."""
    def __init__(
        self,
        people_seq: str = "S_people",
        lockers_seq: Optional[str] = None,
        items_seq: Optional[str] = None,
        min_lockers_per_person: int = 1,
        max_lockers_per_person: int = 3,
        locker_capacity: int = 4,
        allow_repeat_lockers: bool = True,
        unique_lockers_per_person: bool = True,
        place_prob: float = 0.6,
        take_prob: float = 0.4,
        allow_items_without_item_seq: bool = True,
    ):
        self.people_seq = str(people_seq)
        self.lockers_seq = str(lockers_seq) if lockers_seq is not None else None
        self.items_seq = str(items_seq) if items_seq is not None else None
        self.min_lockers = int(min_lockers_per_person)
        self.max_lockers = int(max_lockers_per_person)
        self.locker_capacity = int(locker_capacity)
        self.allow_repeat_lockers = bool(allow_repeat_lockers)
        self.unique_lockers_per_person = bool(unique_lockers_per_person)
        self.place_prob = float(place_prob)
        self.take_prob = float(take_prob)
        self.allow_items_without_item_seq = bool(allow_items_without_item_seq)

        self._locker_tokens: List[str] = []
        self._item_tokens: List[str] = []

    def _locker_from_token(self, tok: str, total: int) -> int:
        return hash(tok) % max(1, total)

    def _sample_route(self, rng: np.random.Generator, total_lockers: int, person_token: str, k: int) -> List[int]:
        route: List[int] = []
        for i in range(k):
            lid = None
            if self._locker_tokens:
                lid = self._locker_from_token(self._locker_tokens.pop(0), total_lockers)
            if lid is None:
                lid = self._locker_from_token(f"{person_token}:{i}", total_lockers)
            if self.unique_lockers_per_person and not self.allow_repeat_lockers:
                tries = 0
                while lid in route and tries < total_lockers * 2:
                    lid = (lid + 1) % max(1, total_lockers)
                    tries += 1
            route.append(lid)
        return route

    def _action_from_item(self, tok: str) -> tuple[str, Optional[str]]:
        try:
            val = int(tok)
        except Exception:
            val = hash(tok)
        if val % 2 == 0:
            return "place", tok
        return "take", tok

    def _build_plan(self, rng: np.random.Generator, person_token: str, total_lockers: int) -> List[Dict[str, Any]]:
        k = int(rng.integers(self.min_lockers, self.max_lockers + 1))
        route = self._sample_route(rng, total_lockers, person_token, k)
        plan: List[Dict[str, Any]] = []
        for lid in route:
            if self.items_seq is None and not self.allow_items_without_item_seq:
                plan.append({"locker": lid, "action": "noop", "item": None})
                continue
            if self._item_tokens:
                tok = self._item_tokens.pop(0)
                action, item = self._action_from_item(tok)
            else:
                if rng.random() < self.place_prob:
                    action, item = "place", person_token
                else:
                    action, item = "take", None
                    if not self.allow_items_without_item_seq:
                        action = "noop"
            plan.append({"locker": lid, "action": action, "item": item})
        return plan

    def on_token(self, ev: TokenEvent, scene: Any, scheduler: Scheduler, rng: np.random.Generator) -> bool:
        if ev.seq_id == self.lockers_seq and self.lockers_seq is not None:
            self._locker_tokens.append(ev.token)
            return True
        if ev.seq_id == self.items_seq and self.items_seq is not None:
            self._item_tokens.append(ev.token)
            return True
        if ev.seq_id != self.people_seq:
            return False
        if not hasattr(scene, "spawn_actor"):
            return False

        total_lockers = max(1, int(getattr(scene, "rows", 2) * getattr(scene, "cols", 8)))
        plan = self._build_plan(rng, ev.token, total_lockers)
        actor_id = scene.spawn_actor(token=ev.token, t=ev.t, rng=rng, plan=plan, locker_capacity=self.locker_capacity)
        return True
