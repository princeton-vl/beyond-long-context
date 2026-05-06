from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def validate_probs(p: np.ndarray, name: str = "probs") -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(p < 0):
        raise ValueError(f"{name} must be non-negative")
    s = float(p.sum())
    if not np.isclose(s, 1.0):
        raise ValueError(f"{name} must sum to 1 (got {s})")
    return p


def make_base_probs(k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    return np.full(k, 1.0 / k, dtype=float)


class CdfSampler:
    def __init__(self, probs: np.ndarray):
        probs = validate_probs(probs, "base_probs")
        self.cdf = np.cumsum(probs)
        self.cdf[-1] = 1.0

    def pick(self, rng: np.random.Generator) -> int:
        r = float(rng.random())
        return int(np.searchsorted(self.cdf, r, side="right"))


@dataclass(frozen=True)
class Rule:
    """
    A rule matches by prefix (longest match wins).
    If it matches, it activates with probability `reliability`.
    If it activates, it emits an output according to out_probs (deterministic if one-hot).
    If it does NOT activate, caller should fall back to base distribution.
    """
    prefix: Tuple[int, ...]
    out_probs: np.ndarray
    reliability: float  # in [0, 1]
    forced: Optional[int]
    cdf: Optional[np.ndarray]

    @staticmethod
    def _from_probs(prefix: Tuple[int, ...], out_probs: np.ndarray, reliability: float) -> "Rule":
        pref = tuple(int(x) for x in prefix)
        p = validate_probs(np.asarray(out_probs, dtype=float), "out_probs")

        rel = float(reliability)
        if rel < 0.0 or rel > 1.0:
            raise ValueError("reliability must be in [0, 1]")

        ones = np.where(np.isclose(p, 1.0))[0]
        forced = int(ones[0]) if len(ones) == 1 else None

        cdf = None
        if forced is None:
            cdf = np.cumsum(p)
            cdf[-1] = 1.0

        return Rule(prefix=pref, out_probs=p, reliability=rel, forced=forced, cdf=cdf)

    @staticmethod
    def deterministic(prefix: Iterable[int], output: int, alphabet_size: int) -> "Rule":
        pref = tuple(int(x) for x in prefix)
        k = int(alphabet_size)
        if not (0 <= output < k):
            raise ValueError("output out of range")
        probs = np.zeros(k, dtype=float)
        probs[output] = 1.0
        return Rule._from_probs(pref, probs, reliability=1.0)

    @staticmethod
    def probabilistic(
        prefix: Iterable[int],
        support_outputs: Iterable[int],
        support_weights: Iterable[float],
        alphabet_size: int,
        reliability: float,
    ) -> "Rule":
        """
        support_outputs: indices with nonzero probability (size m)
        support_weights: arbitrary positive weights (size m), normalized internally
        reliability: activation probability
        """
        pref = tuple(int(x) for x in prefix)
        k = int(alphabet_size)

        outs = [int(o) for o in support_outputs]
        w = np.asarray(list(support_weights), dtype=float)
        if len(outs) != len(w) or len(outs) == 0:
            raise ValueError("support_outputs and support_weights must be same nonzero length")
        if any(o < 0 or o >= k for o in outs):
            raise ValueError("support output out of range")
        if np.any(w < 0):
            raise ValueError("support weights must be non-negative")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("support weights must sum to > 0")

        w = w / s
        probs = np.zeros(k, dtype=float)
        for o, ww in zip(outs, w):
            probs[o] = ww

        return Rule._from_probs(pref, probs, reliability=reliability)

    def _pick_output(self, rng: np.random.Generator) -> int:
        if self.forced is not None:
            return self.forced
        r = float(rng.random())
        return int(np.searchsorted(self.cdf, r, side="right"))

    def try_apply(self, rng: np.random.Generator) -> Optional[int]:
        """
        Returns:
          - int output if rule activates
          - None if rule does not activate (caller should use base distribution)
        """
        if self.reliability >= 1.0:
            return self._pick_output(rng)
        if float(rng.random()) < self.reliability:
            return self._pick_output(rng)
        return None

    def describe(self, max_items: int = 6) -> str:
        """
        Compact string for history printing.
        Deterministic example: (3,1)->4
        Probabilistic example: (3,1) r=0.05 -> {4:0.70, 2:0.20, 0:0.10}
        """
        if self.forced is not None and np.isclose(self.reliability, 1.0):
            return f"{self.prefix} -> {self.forced}"

        # show only support entries (nonzero)
        support = np.flatnonzero(self.out_probs > 0)
        pairs = [(int(i), float(self.out_probs[i])) for i in support]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:max_items]

        inside = ", ".join(f"{i}:{p:.2f}" for i, p in pairs)
        return f"{self.prefix} r={self.reliability:.2f} -> {{{inside}}}"

    def output_entropy(self) -> float:
        """
        Shannon entropy (nats) of this rule's output distribution.
        Reliability is not applied here.
        """
        p = self.out_probs
        mask = p > 0
        if not np.any(mask):
            return 0.0
        p = p[mask].astype(float)
        return float(-(p * np.log(p)).sum())


def ruleset_entropy(rules: Iterable["Rule"], weighted_by_reliability: bool = True) -> float:
    """
    Aggregate entropy over a ruleset.

    If weighted_by_reliability=True (default), each rule's output entropy is
    scaled by its activation probability; this approximates contribution to
    branching when rules actually fire.
    """
    total = 0.0
    for r in rules:
        weight = r.reliability if weighted_by_reliability else 1.0
        total += weight * r.output_entropy()
    return float(total)
