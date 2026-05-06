from __future__ import annotations

import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from .automaton import RuleAutomaton
from .rules import Rule, CdfSampler, validate_probs, ruleset_entropy


@dataclass(frozen=True)
class CompiledGenerator:
    """
    A compiled, reusable sampler for your final discovered distribution.
    Builds the automaton once; can generate sequences efficiently many times.
    """
    base_probs: np.ndarray
    rules: List[Rule]
    automaton: RuleAutomaton
    base_sampler: CdfSampler
    vocab_size: int
    precompute_transitions: bool

    def generate(
        self,
        num_sims: int = 1,
        sim_len: int = 1000,
        seed: Optional[int] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates sequences from the compiled distribution.
        Returns shape (num_sims, sim_len) if num_sims>1 else shape (sim_len,).
        """
        num_sims = int(num_sims)
        sim_len = int(sim_len)
        if num_sims <= 0 or sim_len <= 0:
            raise ValueError("num_sims and sim_len must be positive")

        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(num_sims)

        def _run_one(child_seed: np.random.SeedSequence) -> np.ndarray:
            rng = np.random.default_rng(child_seed)
            seq = np.empty(sim_len, dtype=int)
            state = 0
            for t in range(sim_len):
                ridx = self.automaton.best_rule_idx[state]
                if ridx == -1:
                    sym = self.base_sampler.pick(rng)
                else:
                    out = self.rules[ridx].try_apply(rng)
                    if out is None:
                        sym = self.base_sampler.pick(rng)
                    else:
                        sym = int(out)

                seq[t] = sym
                state = self.automaton.step_state(state, sym)
            return seq

        total_steps = num_sims * sim_len
        if parallel is None:
            # Conservative heuristic: threading often doesn't help for small loops under CPython
            parallel = (num_sims >= 8 and total_steps >= 200_000)

        if parallel:
            workers = max_workers or min(num_sims, (os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                seqs = list(ex.map(_run_one, child_seeds))
        else:
            seqs = [_run_one(s) for s in child_seeds]

        arr = np.stack(seqs, axis=0)
        if num_sims == 1:
            return arr[0]
        return arr

    @property
    def ruleset_entropy(self) -> float:
        """
        Reliability-weighted entropy (nats) of all rule output distributions.
        """
        return ruleset_entropy(self.rules, weighted_by_reliability=True)


def compile_generator(
    base_probs: np.ndarray,
    rules: List[Rule],
    precompute_transitions: bool = True,
) -> CompiledGenerator:
    base_probs = validate_probs(np.asarray(base_probs, dtype=float), "base_probs")
    V = len(base_probs)
    autom = RuleAutomaton(rules, alphabet_size=V, precompute_transitions=precompute_transitions)
    base_sampler = CdfSampler(base_probs)
    return CompiledGenerator(
        base_probs=base_probs,
        rules=rules,
        automaton=autom,
        base_sampler=base_sampler,
        vocab_size=V,
        precompute_transitions=precompute_transitions,
    )
