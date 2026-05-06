from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
import json
import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = _THIS_DIR / "static_icons" / "manifest.json"


def _resolve_manifest(path: Optional[str]) -> Path:
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (_THIS_DIR / candidate).resolve()
        return candidate
    return _DEFAULT_MANIFEST


@dataclass(frozen=True)
class SpriteTile:
    id: str
    image: np.ndarray  # BGRA uint8
    width: int
    height: int


@lru_cache(maxsize=None)
def _load_catalog(resolved_path: str) -> Dict[str, SpriteTile]:
    manifest_path = Path(resolved_path)
    data = json.loads(manifest_path.read_text())
    sprites: Dict[str, SpriteTile] = {}
    for entry in data.get("sprites", []):
        sprite_id = str(entry.get("id"))
        if not sprite_id:
            continue
        rel_path = entry.get("path")
        if not rel_path:
            continue
        img_path = manifest_path.parent / rel_path
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim != 3:
            continue
        if img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        sprites[sprite_id] = SpriteTile(
            id=sprite_id,
            image=img.copy(),
            width=int(entry.get("width", img.shape[1])),
            height=int(entry.get("height", img.shape[0])),
        )
    if len(sprites) < 16:
        raise RuntimeError(
            f"Sprite catalog '{manifest_path}' only has {len(sprites)} usable entries; need at least 16."
        )
    return sprites


def available_sprite_ids(manifest_path: Optional[str] = None) -> List[str]:
    manifest = _resolve_manifest(manifest_path)
    return list(_load_catalog(str(manifest)).keys())


def get_sprite(sprite_id: str, manifest_path: Optional[str] = None) -> SpriteTile:
    manifest = _resolve_manifest(manifest_path)
    catalog = _load_catalog(str(manifest))
    try:
        return catalog[sprite_id]
    except KeyError as exc:
        raise KeyError(f"Unknown sprite id '{sprite_id}' in catalog {manifest}") from exc
