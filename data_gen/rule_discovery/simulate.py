from __future__ import annotations

import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .automaton import RuleAutomaton
from .config import NGramStat
from .rules import CdfSampler, Rule, validate_probs



@dataclass
class SimulationResult:
    num_sims: int
    sim_len: int
    seqs: np.ndarray            # shape (num_sims, sim_len)
    base_count: int
    rule_counts: np.ndarray     # shape (num_rules,)

    @property
    def total_steps(self) -> int:
        return int(self.num_sims) * int(self.sim_len)

    @property
    def base_usage(self) -> float:
        return self.base_count / float(self.total_steps)

    @property
    def rule_usage(self) -> np.ndarray:
        if self.rule_counts.size == 0:
            return self.rule_counts.astype(float)
        return self.rule_counts / float(self.total_steps)

    @property
    def entropy(self) -> float:
        """
        Empirical entropy (bits) of generated symbols across all trajectories.
        """
        flat = self.seqs.ravel()
        if flat.size == 0:
            return 0.0
        counts = np.bincount(flat)
        p = counts[counts > 0].astype(float)
        p = p / p.sum()
        logp = np.log2(p)
        return float(-(p * logp).sum())

    def entropy_rate(self, order: int = 2) -> float:
        """
        Empirical conditional entropy H(X_t | X_{t-order+1:t-1}) in bits.
        order=1 gives the same as entropy() on marginals.
        """
        return entropy_rate(self.seqs, order=order)


def simulate(
    base_probs: np.ndarray,
    rules: List[Rule],
    num_sims: int,
    sim_len: int,
    seed: Optional[int],
    precompute_transitions: bool,
    parallel: Optional[bool] = None,
    max_workers: Optional[int] = None,
) -> SimulationResult:
    base_probs = validate_probs(np.asarray(base_probs, dtype=float), "base_probs")
    k = len(base_probs)
    num_sims = int(num_sims)
    sim_len = int(sim_len)
    if num_sims <= 0 or sim_len <= 0:
        raise ValueError("num_sims and sim_len must be positive")

    autom = RuleAutomaton(rules, alphabet_size=k, precompute_transitions=precompute_transitions)
    base_sampler = CdfSampler(base_probs)

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(num_sims)

    def _run_one(child_seed: np.random.SeedSequence):
        rng = np.random.default_rng(child_seed)
        seq = np.empty(sim_len, dtype=int)
        rule_counts_local = np.zeros(len(rules), dtype=int)
        base_count_local = 0
        state = 0

        for t in range(sim_len):
            ridx = autom.best_rule_idx[state]
            if ridx == -1:
                base_count_local += 1
                sym = base_sampler.pick(rng)
            else:
                out = rules[ridx].try_apply(rng)
                if out is None:
                    # rule matched but did not activate -> base takes over
                    base_count_local += 1
                    sym = base_sampler.pick(rng)
                else:
                    rule_counts_local[ridx] += 1
                    sym = int(out)

            seq[t] = sym
            state = autom.step_state(state, sym)

        return seq, base_count_local, rule_counts_local

    total_steps = num_sims * sim_len
    if parallel is None:
        parallel = (num_sims >= 8 and total_steps >= 200_000)

    if parallel:
        workers = max_workers or min(num_sims, (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_one, child_seeds))
    else:
        results = [_run_one(s) for s in child_seeds]

    seqs = np.stack([r[0] for r in results], axis=0)
    base_count = int(sum(r[1] for r in results))

    if len(rules) == 0:
        rule_counts = np.zeros(0, dtype=int)
    else:
        rule_counts = np.sum(np.stack([r[2] for r in results], axis=0), axis=0)

    return SimulationResult(
        num_sims=num_sims,
        sim_len=sim_len,
        seqs=seqs,
        base_count=base_count,
        rule_counts=rule_counts,
    )


def top_ngrams(
    seqs: np.ndarray,
    min_n: int,
    max_n: int,
    top_k: int,
) -> Dict[int, List[NGramStat]]:
    """
    Works on:
      - 1D seq: treated as a single trajectory
      - 2D seqs: shape (num_sims, sim_len), counts n-grams within each row
                (does NOT allow n-grams to cross between trajectories)
    pct is computed relative to total number of windows across all trajectories.
    """
    if min_n <= 0 or max_n < min_n:
        raise ValueError("invalid min_n/max_n")

    arr = np.asarray(seqs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("seqs must be 1D or 2D")

    num_sims, sim_len = arr.shape
    out: Dict[int, List[NGramStat]] = {}

    for n in range(min_n, max_n + 1):
        windows_per = sim_len - n + 1
        if windows_per <= 0:
            out[n] = []
            continue

        total_windows = num_sims * windows_per
        counts: Dict[Tuple[int, ...], int] = defaultdict(int)

        for s in range(num_sims):
            row = arr[s]
            for i in range(windows_per):
                g = tuple(int(x) for x in row[i:i + n])
                counts[g] += 1

        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out[n] = [NGramStat(ngram=g, count=c, pct=c / total_windows) for g, c in items]

    return out


def entropy_rate(seqs: np.ndarray, order: int = 2) -> float:
    """
    Plug-in estimator of conditional entropy H(X_t | X_{t-order+1:t-1}) in bits.
    order=1 reduces to marginal entropy.
    """
    if order <= 0:
        raise ValueError("order must be positive")

    arr = np.asarray(seqs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("seqs must be 1D or 2D")

    num_sims, sim_len = arr.shape
    windows_per = sim_len - order + 1
    if windows_per <= 0:
        return 0.0

    prefix_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    joint_counts: Dict[Tuple[int, ...], int] = defaultdict(int)

    for s in range(num_sims):
        row = arr[s]
        for i in range(windows_per):
            joint = tuple(int(x) for x in row[i:i + order])
            pref = joint[:-1]
            joint_counts[joint] += 1
            prefix_counts[pref] += 1

    total_windows = float(num_sims * windows_per)
    h = 0.0
    for joint, cj in joint_counts.items():
        pref = joint[:-1]
        cp = prefix_counts[pref]
        p_joint = cj / total_windows
        p_cond = cj / cp
        h -= p_joint * math.log2(p_cond)
    return float(h)


@dataclass
class LZCausalCache:
    """Caches causal Lempel-Ziv match lengths to reuse entropy queries."""

    lambdas: np.ndarray

    @property
    def length(self) -> int:
        return int(self.lambdas.size)

    def entropy_bits(self, prefix_len: Optional[int] = None) -> float:
        n = self.length if prefix_len is None else max(0, min(int(prefix_len), self.length))
        if n <= 1:
            return 0.0
        total = self._prefix_lambda_sum(n)
        if total <= 0.0 or not math.isfinite(total):
            return 0.0
        return float((n * math.log2(n)) / total)

    def _prefix_lambda_sum(self, prefix_len: int) -> float:
        total = 0.0
        for i in range(prefix_len):
            limit = prefix_len - i + 1
            lam = float(self.lambdas[i])
            total += lam if lam <= limit else limit
        return total


def build_lz_causal_cache(seqs: Sequence[int] | np.ndarray) -> LZCausalCache:
    arr = np.asarray(seqs, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("build_lz_causal_cache expects a 1D sequence")
    lambdas = np.asarray(_lz_causal_match_lengths(arr), dtype=np.int64)
    return LZCausalCache(lambdas=lambdas)


def lz_entropy_rate_causal(seqs: np.ndarray) -> float:
    """Causal Lempel-Ziv entropy-rate estimator (bits per symbol)."""

    if seqs.ndim == 1:
        arr = seqs
    elif seqs.ndim == 2:
        arr = seqs.ravel()
    else:
        raise ValueError("seqs must be 1D or 2D")
    if arr.size == 0:
        return 0.0
    cache = build_lz_causal_cache(arr)
    return cache.entropy_bits()


def _suffix_array(s: np.ndarray) -> List[int]:
    n = len(s)
    k = 1
    sa = list(range(n))
    rank = list(int(x) for x in s)
    tmp = [0] * n

    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev = sa[i - 1]
            curr = sa[i]
            tmp[curr] = tmp[prev]
            if (rank[prev], rank[prev + k] if prev + k < n else -1) != (
                rank[curr],
                rank[curr + k] if curr + k < n else -1,
            ):
                tmp[curr] += 1
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa


def _lcp_array(s: np.ndarray, sa: List[int]) -> List[int]:
    n = len(s)
    rank = [0] * n
    for i, idx in enumerate(sa):
        rank[idx] = i

    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1
    return lcp


def _lz_causal_match_lengths(s: np.ndarray) -> List[int]:
    n = len(s)
    if n == 0:
        return []
    sa = _suffix_array(s)
    lcp = _lcp_array(s, sa)
    rank = [0] * n
    starts = [0] * n
    for r, idx in enumerate(sa):
        rank[idx] = r
        starts[r] = idx

    left = _nearest_smaller_indices(starts)
    right = _nearest_smaller_indices(starts, reverse=True)
    rmq = _RmqStructure(lcp)

    lambdas = [1] * n
    for idx in range(n):
        r = rank[idx]
        best = 0
        left_rank = left[r]
        if left_rank != -1:
            best = rmq.lcp_between(sa, r, left_rank)
        right_rank = right[r]
        if right_rank != -1:
            best = max(best, rmq.lcp_between(sa, r, right_rank))
        lambdas[idx] = int(best) + 1
    return lambdas


def _nearest_smaller_indices(values: Sequence[int], reverse: bool = False) -> List[int]:
    n = len(values)
    out = [-1] * n
    it = range(n - 1, -1, -1) if reverse else range(n)
    stack: List[int] = []
    for i in it:
        while stack and values[stack[-1]] >= values[i]:
            stack.pop()
        if stack:
            out[i] = stack[-1]
        stack.append(i)
    return out


class _RmqStructure:
    """Range-minimum query helper for LCP lookups."""

    def __init__(self, arr: Sequence[int]):
        self.arr = list(arr)
        n = len(self.arr)
        self.logs = [0] * (n + 1)
        for i in range(2, n + 1):
            self.logs[i] = self.logs[i // 2] + 1
        max_k = self.logs[n] + 1 if n else 1
        self.table: List[List[int]] = [self.arr[:]]
        for k in range(1, max_k):
            span = 1 << (k - 1)
            row_len = max(0, n - (1 << k) + 1)
            row: List[int] = [0] * row_len
            for i in range(row_len):
                row[i] = min(self.table[k - 1][i], self.table[k - 1][i + span])
            self.table.append(row)

    def query(self, left: int, right: int) -> int:
        if left > right:
            return 0
        length = right - left + 1
        if length <= 0:
            return 0
        k = self.logs[length]
        span = 1 << k
        row = self.table[k]
        return min(row[left], row[right - span + 1])

    def lcp_between(self, sa: Sequence[int], rank_a: int, rank_b: int) -> int:
        if rank_a == rank_b:
            idx = sa[rank_a]
            return len(sa) - idx
        left = min(rank_a, rank_b) + 1
        right = max(rank_a, rank_b)
        return self.query(left, right)
