from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DiscoveryConfig:
    # Simulation shape
    num_sims: int
    sim_len: int

    @property
    def total_steps(self) -> int:
        return int(self.num_sims) * int(self.sim_len)

    # Rule mode: "deterministic" or "probabilistic"
    rule_mode: str

    # Probabilistic rule activation probability r ~ Uniform[min,max]
    prob_reliability_min: float
    prob_reliability_max: float

    # Support size m chosen uniformly from [1..max_support], capped by divisor if set
    prob_support_max_divisor: Optional[float]

    # Rule counts (max_rules=None means no cap)
    min_rules: int
    max_rules: Optional[int]
    enforce_min_rules: bool

    # Proposal n-grams
    proposal_min_len: int
    proposal_max_len: int
    proposal_top_k: int

    # Skip proposing prefixes below this count (None => derived)
    proposal_min_prefix_count: Optional[int]

    # Reporting n-grams
    report_min_len: int
    report_max_len: int
    report_top_k: int

    # Acceptance bounds (fractions of total_steps = K*N); set to None to disable
    min_rule_usage: Optional[float]
    max_rule_usage: Optional[float]
    # Allow entropy increase up to this amount after half the tries in an iteration (None disables)
    max_entropy_increase: Optional[float]
    # Disallow entropy drops larger than this (absolute) in a single accepted step (None disables)
    max_entropy_decrease: Optional[float]
    # Reject proposals that would drive entropy below this floor (None disables)
    min_entropy: Optional[float]
    # Prune rules that haven't fired in this many iterations (None disables)
    prune_after_iters: Optional[int]

    # Search effort
    max_prefix_tries: int

    # Proposal bias
    proposal_temperature: float

    # Output selection bias for proposed outputs/support symbols: "base" or "uniform"
    output_choice: str

    # Prefix length selection: "uniform" or "weighted"
    length_selection: str

    # Performance
    precompute_transitions: bool

    # Parallelism toggles for simulation/generation
    simulate_parallel: Optional[bool]
    simulate_max_workers: Optional[int]

    # History printing controls (printing itself is controlled in main.py)
    print_rejections: bool
    max_rejections_printed_per_iter: int
    max_rows_per_n_final: int
    max_rows_per_n_history: Optional[int]  # None => half of final

    # Entropy-based stopping (optional, uses LZ estimator)
    entropy_target: Optional[float]
    # When entropy_target is not reached in an iteration, multiply proposal_temperature by this factor
    entropy_temp_multiplier: float

    # Randomness
    seed: Optional[int]


@dataclass(frozen=True)
class NGramStat:
    ngram: Tuple[int, ...]
    count: int
    pct: float


@dataclass(frozen=True)
class ProposalAttempt:
    iteration: int
    n: int
    prefix: Tuple[int, ...]
    prefix_count_before: int

    proposed_rule: str

    accepted: bool
    reason: str

    # fraction of total steps where the NEW rule fired
    cand_new_rule_usage: float


@dataclass
class IterationHistory:
    iteration: int
    rules_before: int
    base_usage_before: float

    attempts: List[ProposalAttempt]

    accepted: bool
    accepted_rule: Optional[str]

    # post-accept snapshots
    base_usage_after: Optional[float]
    rule_usage_after: Optional[np.ndarray]
    top_after: Optional[Dict[int, List[NGramStat]]]

    # entropy snapshot after acceptance (bits/symbol)
    entropy_bits: Optional[float]

    # snapshot of rules after acceptance (strings), aligns with rule_usage_after
    rules_after: Optional[List[str]]
