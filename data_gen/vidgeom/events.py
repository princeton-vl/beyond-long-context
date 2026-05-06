from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(order=True)
class TokenEvent:
    t: float
    token: str = field(compare=False)
    seq_id: str = field(compare=False)
    index: int = field(compare=False)
    meta: Dict[str, Any] = field(default_factory=dict, compare=False)

@dataclass(order=True)
class SceneEvent:
    t: float
    name: str = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
