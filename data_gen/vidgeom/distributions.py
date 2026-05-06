from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import numpy as np

Spec = Any

def sample(spec: Spec, rng: np.random.Generator) -> Any:
    """Sample a value from a small distribution spec.

    Supported forms:
      - scalar (returned as-is)
      - list (returned as-is)
      - dict with one of:
          {uniform: [lo, hi]}
          {randint: [lo, hi]}   # inclusive lo, exclusive hi
          {choice: [v1, v2, ...]}
          {wchoice: {v1: w1, v2: w2, ...}}  # keys can be scalars or JSON-serializable lists
    """
    if isinstance(spec, dict):
        if "uniform" in spec:
            lo, hi = spec["uniform"]
            return float(rng.uniform(lo, hi))
        if "randint" in spec:
            lo, hi = spec["randint"]
            return int(rng.integers(lo, hi))
        if "choice" in spec:
            arr = spec["choice"]
            return arr[int(rng.integers(0, len(arr)))]
        if "wchoice" in spec:
            wmap = spec["wchoice"]
            keys = list(wmap.keys())
            weights = np.array([float(wmap[k]) for k in keys], dtype=np.float64)
            weights = weights / weights.sum()
            idx = int(rng.choice(len(keys), p=weights))
            k = keys[idx]
            # If key is a JSON-encoded list/obj (common in YAML), try to parse it.
            # But YAML keys are usually strings; we keep raw.
            return k
    return spec

def sample_params(params: Dict[str, Spec], rng: np.random.Generator) -> Dict[str, Any]:
    return {k: sample(v, rng) for k, v in (params or {}).items()}
