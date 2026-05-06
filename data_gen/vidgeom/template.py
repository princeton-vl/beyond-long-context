from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml

from .rules import RULE_REGISTRY, Rule
from .scenes.conveyor import ConveyorSceneV2
from .scenes.locker_room import LockerRoomSceneGeom
from .scenes.sorting_hub import SortingHubSceneGeom
from .scenes.shape_flash import ShapeFlashScene

SCENE_REGISTRY = {
    "conveyor_v2": ConveyorSceneV2,
    "locker_room_v1_geom": LockerRoomSceneGeom,
    "sorting_hub_v1_geom": SortingHubSceneGeom,
    "shape_flash_v1": ShapeFlashScene,
}

@dataclass
class Template:
    raw: Dict[str, Any]

    @property
    def video_type(self) -> str:
        return str(self.raw.get("video_type", "conveyor_v2"))

    @property
    def render(self) -> Dict[str, Any]:
        return self.raw.get("render", {}) or {}

    @property
    def timing(self) -> Dict[str, Any]:
        return self.raw.get("timing", {}) or {}

    @property
    def variants(self) -> Dict[str, Any]:
        return self.raw.get("variants", {}) or {}

    @property
    def vocab(self) -> Dict[str, Any]:
        return self.raw.get("vocab", {}) or {}

    @property
    def rules_cfg(self) -> List[Dict[str, Any]]:
        return self.raw.get("rules", []) or []

    @property
    def scene_cfg(self) -> Dict[str, Any]:
        return self.raw.get("scene", {}) or {}

    def make_scene(self):
        cls = SCENE_REGISTRY.get(self.video_type)
        if cls is None:
            raise ValueError(f"Unknown video_type/scene: {self.video_type}")
        return cls()

    def make_rules(self) -> List[Rule]:
        rules: List[Rule] = []
        for r in self.rules_cfg:
            name = str(r.get("name"))
            params = r.get("params", {}) or {}
            cls = RULE_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown rule: {name}")
            rules.append(cls(**params))
        return rules

def load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Template YAML must parse to a dict.")
    return Template(raw=raw)
