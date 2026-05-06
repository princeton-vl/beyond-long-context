import math
import unittest
from itertools import product
from typing import List

import numpy as np

from rule_discovery.entropy import analytic_entropy
from rule_discovery.simulate import build_lz_causal_cache, lz_entropy_rate_causal
from rule_discovery.rules import Rule, make_base_probs
from rule_discovery.automaton import RuleAutomaton


class AnalyticEntropyTests(unittest.TestCase):
    def test_uniform_no_rules_matches_log2_vocab(self):
        base = make_base_probs(5)
        stats = analytic_entropy(base, [])
        self.assertAlmostEqual(stats.entropy_bits, math.log2(5), places=9)

    def test_matches_bruteforce_context_chain(self):
        base = np.array([0.5, 0.5], dtype=float)
        rules = [
            Rule.deterministic(prefix=(1, 1), output=0, alphabet_size=2),
            Rule.probabilistic(
                prefix=(0,),
                support_outputs=[0, 1],
                support_weights=[3.0, 1.0],
                alphabet_size=2,
                reliability=0.5,
            ),
        ]
        stats = analytic_entropy(base, rules)
        brute = self._bruteforce_entropy(base, rules, context_len=2)
        self.assertAlmostEqual(stats.entropy_bits, brute, places=9)

    def _bruteforce_entropy(self, base, rules, context_len):
        autom = RuleAutomaton(rules, alphabet_size=len(base), precompute_transitions=True)
        contexts = list(product(range(len(base)), repeat=context_len)) or [tuple()]
        idx = {ctx: i for i, ctx in enumerate(contexts)}
        num_states = len(contexts)
        trans = np.zeros((num_states, num_states), dtype=float)
        ent = np.zeros(num_states, dtype=float)

        for i, ctx in enumerate(contexts):
            state = 0
            for sym in ctx:
                state = autom.step_state(state, sym)
            ridx = autom.best_rule_idx[state]
            if ridx == -1:
                emis = base
            else:
                rule = rules[ridx]
                emis = rule.reliability * rule.out_probs + (1.0 - rule.reliability) * base
            mask = emis > 0
            if np.any(mask):
                ent[i] = float(-np.sum(emis[mask] * np.log2(emis[mask])))
            for sym, p in enumerate(emis):
                if p <= 0:
                    continue
                if context_len == 0:
                    nxt = ()
                else:
                    nxt = ctx[1:] + (sym,)
                j = idx[nxt]
                trans[i, j] += p

        row_sums = trans.sum(axis=1, keepdims=True)
        np.divide(trans, row_sums, out=trans, where=row_sums > 0)
        pi = np.full(num_states, 1.0 / num_states, dtype=float)
        for _ in range(10000):
            new_pi = pi @ trans
            if np.linalg.norm(new_pi - pi, ord=1) < 1e-12:
                pi = new_pi
                break
            pi = new_pi
        pi /= pi.sum()
        return float(pi @ ent)


class CausalLZEntropyTests(unittest.TestCase):
    def test_constant_sequence_has_zero_entropy(self):
        arr = np.zeros(50, dtype=int)
        cache = build_lz_causal_cache(arr)
        self.assertLess(cache.entropy_bits(), 0.3)
        self.assertLess(cache.entropy_bits(10), 1.0)

    def test_matches_naive_small_sequence(self):
        seq = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
        cache = build_lz_causal_cache(seq)
        expected = _naive_causal_entropy(seq)
        self.assertAlmostEqual(cache.entropy_bits(), expected, places=9)
        for k in range(1, len(seq) + 1):
            self.assertAlmostEqual(cache.entropy_bits(k), _naive_causal_entropy(seq[:k]), places=9)

    def test_function_alias(self):
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int)
        cache = build_lz_causal_cache(seq)
        self.assertAlmostEqual(cache.entropy_bits(), lz_entropy_rate_causal(seq))


def _naive_causal_entropy(seq: np.ndarray) -> float:
    n = len(seq)
    if n <= 1:
        return 0.0
    lambdas: List[int] = []
    for i in range(n):
        best = 0
        for j in range(i):
            L = 0
            while i + L < n and seq[j + L] == seq[i + L]:
                L += 1
            if L > best:
                best = L
        lambdas.append(best + 1)
    total = sum(lambdas)
    if total <= 0:
        return 0.0
    return float((n * math.log2(n)) / total)


if __name__ == "__main__":
    unittest.main()
