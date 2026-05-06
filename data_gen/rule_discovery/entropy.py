from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .automaton import RuleAutomaton
from .rules import Rule, validate_probs


LOG2E = math.log2(math.e)


@dataclass
class AnalyticEntropyResult:
    """Container for analytic entropy computations over the rule automaton."""

    entropy_bits: float
    state_stationary: np.ndarray
    state_entropies: np.ndarray
    context_labels: List[Tuple[int, ...]]
    emission_probs: np.ndarray


def _state_labels(automaton: RuleAutomaton) -> List[Tuple[int, ...]]:
    labels: List[Tuple[int, ...]] = [tuple() for _ in range(len(automaton.nodes))]
    q = deque([0])
    while q:
        state = q.popleft()
        prefix = labels[state]
        for sym, child in automaton.nodes[state].children.items():
            labels[child] = prefix + (int(sym),)
            q.append(child)
    return labels


def _state_emission_matrix(
    automaton: RuleAutomaton, rules: Sequence[Rule], base_probs: np.ndarray
) -> np.ndarray:
    num_states = len(automaton.nodes)
    vocab = base_probs.size
    emissions = np.tile(base_probs[None, :], (num_states, 1))

    for state in range(num_states):
        ridx = automaton.best_rule_idx[state]
        if ridx == -1:
            continue
        rule = rules[ridx]
        emissions[state] = rule.reliability * rule.out_probs + (1.0 - rule.reliability) * base_probs
    return emissions


def _state_transition_matrix(
    automaton: RuleAutomaton,
    emissions: np.ndarray,
) -> np.ndarray:
    num_states = emissions.shape[0]
    vocab = emissions.shape[1]
    trans = np.zeros((num_states, num_states), dtype=float)

    use_table = automaton.trans is not None
    for state in range(num_states):
        row = trans[state]
        for sym in range(vocab):
            p = emissions[state, sym]
            if p <= 0.0:
                continue
            if use_table:
                nxt = automaton.trans[state][sym]
            else:
                nxt = automaton.step_state(state, sym)
            row[nxt] += p

    row_sums = trans.sum(axis=1, keepdims=True)
    np.divide(trans, row_sums, out=trans, where=row_sums > 0)
    return trans


def _stationary_distribution(
    trans: np.ndarray,
    tol: float = 1e-12,
    max_iters: int = 10000,
) -> np.ndarray:
    n = trans.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)
    for _ in range(max_iters):
        new_pi = pi @ trans
        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            pi = new_pi
            break
        pi = new_pi
    total = pi.sum()
    if total <= 0:
        return np.full(n, 0.0, dtype=float)
    return pi / total


def _state_entropies(emissions: np.ndarray) -> np.ndarray:
    num_states = emissions.shape[0]
    ent = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        probs = emissions[state]
        mask = probs > 0
        if not np.any(mask):
            continue
        logp = np.log2(probs[mask])
        ent[state] = float(-np.sum(probs[mask] * logp))
    return ent


def analytic_entropy(
    base_probs: np.ndarray,
    rules: Sequence[Rule],
    automaton: Optional[RuleAutomaton] = None,
    *,
    tol: float = 1e-12,
    max_iters: int = 10000,
    precompute_transitions: bool = True,
) -> AnalyticEntropyResult:
    """Compute analytic entropy rate (bits/symbol) from rules + base distribution."""

    probs = validate_probs(np.asarray(base_probs, dtype=float), "base_probs")
    vocab = probs.size

    if automaton is None:
        automaton = RuleAutomaton(
            list(rules),
            alphabet_size=vocab,
            precompute_transitions=precompute_transitions,
        )

    emissions = _state_emission_matrix(automaton, rules, probs)
    trans = _state_transition_matrix(automaton, emissions)
    stationary = _stationary_distribution(trans, tol=tol, max_iters=max_iters)
    state_ent = _state_entropies(emissions)
    entropy_bits = float(stationary @ state_ent)
    labels = _state_labels(automaton)

    return AnalyticEntropyResult(
        entropy_bits=entropy_bits,
        state_stationary=stationary,
        state_entropies=state_ent,
        context_labels=labels,
        emission_probs=emissions,
    )
