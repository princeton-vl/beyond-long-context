from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Seed:
    value: int

def _int_hash(*parts: object) -> int:
    """Stable-ish 64-bit hash for seeds (not cryptographic)."""
    h = 1469598103934665603  # FNV offset basis
    for p in parts:
        s = str(p).encode("utf-8")
        for b in s:
            h ^= b
            h *= 1099511628211
            h &= 0xFFFFFFFFFFFFFFFF
    return int(h)

def derive_seed(base_seed: int, *parts: object) -> int:
    return _int_hash(base_seed, *parts) & 0xFFFFFFFF

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)
