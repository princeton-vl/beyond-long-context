from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DiscoveryConfig, IterationHistory, NGramStat, ProposalAttempt
from .entropy import analytic_entropy
from .simulate import simulate, top_ngrams


def _pick_length(rng: np.random.Generator, candidates: Dict[int, List[NGramStat]], mode: str) -> int:
    ns = [n for n, stats in candidates.items() if stats]
    if not ns:
        raise ValueError("no candidate lengths")

    if mode == "uniform":
        return int(rng.choice(ns))

    if mode == "weighted":
        weights = np.array([sum(s.pct for s in candidates[n]) for n in ns], dtype=float)
        weights = weights / weights.sum()
        return int(rng.choice(ns, p=weights))

    raise ValueError(f"unknown length_selection: {mode!r}")


def _pick_prefix(
    rng: np.random.Generator,
    mined: Dict[int, List[NGramStat]],
    existing: set[Tuple[int, ...]],
    tried: set[Tuple[int, ...]],
    temperature: float,
    length_selection: str,
    min_prefix_count: int,
) -> Optional[Tuple[int, ...]]:
    filtered: Dict[int, List[NGramStat]] = {}
    for n, stats in mined.items():
        keep = [
            s for s in stats
            if s.ngram not in existing
            and s.ngram not in tried
            and s.count >= min_prefix_count
            and s.count > 1
        ]
        if keep:
            filtered[n] = keep

    if not filtered:
        return None

    n = _pick_length(rng, filtered, length_selection)
    stats = filtered[n]

    w = np.array([s.pct ** float(temperature) for s in stats], dtype=float)
    w = w / w.sum()
    idx = int(rng.choice(len(stats), p=w))
    return stats[idx].ngram


def discover_rules(base_probs: np.ndarray, cfg: DiscoveryConfig):
    import math
    import numpy as np

    from .rules import Rule, validate_probs
    from .simulate import simulate, top_ngrams
    from .generator import compile_generator

    base_probs = validate_probs(np.asarray(base_probs, dtype=float), "base_probs")
    V = len(base_probs)
    rng = np.random.default_rng(cfg.seed)

    min_prefix_count = cfg.proposal_min_prefix_count
    if min_prefix_count is None:
        min_usage = cfg.min_rule_usage if cfg.min_rule_usage is not None else 0.0
        if cfg.rule_mode == "probabilistic" and min_usage > 0:
            denom = max(cfg.prob_reliability_max, 1e-9)
            min_prefix_count = int(math.ceil((min_usage * cfg.total_steps) / denom))
        else:
            min_prefix_count = int(math.ceil(min_usage * cfg.total_steps))

    rules: List[Rule] = []
    history: List[IterationHistory] = []
    temperature = cfg.proposal_temperature
    last_seen: Dict[int, int] = {}

    sim = simulate(
        base_probs=base_probs,
        rules=rules,
        num_sims=cfg.num_sims,
        sim_len=cfg.sim_len,
        seed=int(rng.integers(0, 2**32 - 1)),
        precompute_transitions=cfg.precompute_transitions,
        parallel=cfg.simulate_parallel,
        max_workers=cfg.simulate_max_workers,
    )

    def _entropy_stats(ruleset: List[Rule]):
        return analytic_entropy(
            base_probs=base_probs,
            rules=ruleset,
            precompute_transitions=cfg.precompute_transitions,
        )

    entropy_stats = _entropy_stats(rules)
    current_entropy_bits = entropy_stats.entropy_bits

    def _pick_one_symbol(exclude: Optional[int] = None) -> int:
        if cfg.output_choice == "uniform":
            while True:
                x = int(rng.integers(0, V))
                if exclude is None or x != exclude:
                    return x
        elif cfg.output_choice == "base":
            if exclude is None:
                return int(rng.choice(V, p=base_probs))
            mask = np.ones(V, dtype=bool)
            mask[int(exclude)] = False
            cand = np.nonzero(mask)[0]
            p = base_probs[mask]
            p = p / p.sum()
            return int(rng.choice(cand, p=p))
        else:
            raise ValueError(f"unknown output_choice: {cfg.output_choice!r}")

    def _max_support_size() -> int:
        if cfg.prob_support_max_divisor is None:
            return V
        d = float(cfg.prob_support_max_divisor)
        if d <= 0:
            raise ValueError("prob_support_max_divisor must be > 0")
        return max(1, int(math.ceil(V / d)))

    def _propose_rule(prefix: Tuple[int, ...]) -> Rule:
        # Ban self-unigram output/support element
        exclude = int(prefix[0]) if len(prefix) == 1 else None

        if cfg.rule_mode == "deterministic":
            out = _pick_one_symbol(exclude=exclude)
            return Rule.deterministic(prefix, out, alphabet_size=V)

        if cfg.rule_mode == "probabilistic":
            rel = float(rng.uniform(cfg.prob_reliability_min, cfg.prob_reliability_max))
            max_support = _max_support_size()
            m = int(rng.integers(1, max_support + 1))

            # pick m distinct outputs
            candidates = list(range(V))
            if exclude is not None and exclude in candidates:
                candidates.remove(exclude)
            if len(candidates) < m:
                m = len(candidates)

            if cfg.output_choice == "uniform":
                outs = [int(x) for x in rng.choice(candidates, size=m, replace=False)]
            elif cfg.output_choice == "base":
                cand_arr = np.array(candidates, dtype=int)
                p = base_probs[cand_arr]
                p = p / p.sum()
                outs = [int(x) for x in rng.choice(cand_arr, size=m, replace=False, p=p)]
            else:
                raise ValueError(f"unknown output_choice: {cfg.output_choice!r}")

            weights = rng.random(m)  # random positive weights, normalized inside Rule.probabilistic
            return Rule.probabilistic(prefix, outs, weights, alphabet_size=V, reliability=rel)

        raise ValueError(f"unknown rule_mode: {cfg.rule_mode!r}")

    max_iters = cfg.max_rules if cfg.max_rules is not None else float("inf")
    it = 1
    while it <= max_iters:
        if cfg.entropy_target is not None and current_entropy_bits <= cfg.entropy_target:
            break

        iter_hist = IterationHistory(
            iteration=it,
            rules_before=len(rules),
            base_usage_before=sim.base_usage,
            attempts=[],
            accepted=False,
            accepted_rule=None,
            base_usage_after=None,
            rule_usage_after=None,
            top_after=None,
            rules_after=None,
            entropy_bits=None,
        )

        # Helper to prune rules by indices and refresh sim/entropy
        def _prune_indices(to_prune: List[int]) -> None:
            nonlocal sim, current_entropy_bits, last_seen, rules
            if not to_prune:
                return
            to_prune = sorted(set(to_prune), reverse=True)
            for idx in to_prune:
                if 0 <= idx < len(rules):
                    del rules[idx]
            # reindex last_seen after deletions
            new_last: Dict[int, int] = {}
            for old_idx, seen_iter in last_seen.items():
                shift = sum(1 for p in to_prune if p < old_idx)
                new_idx = old_idx - shift
                if 0 <= new_idx < len(rules):
                    new_last[new_idx] = seen_iter
            last_seen = new_last
            sim = simulate(
                base_probs=base_probs,
                rules=rules,
                num_sims=cfg.num_sims,
                sim_len=cfg.sim_len,
                seed=int(rng.integers(0, 2**32 - 1)),
                precompute_transitions=cfg.precompute_transitions,
                parallel=cfg.simulate_parallel,
                max_workers=cfg.simulate_max_workers,
            )
            entropy_stats = _entropy_stats(rules)
            current_entropy_bits = entropy_stats.entropy_bits

        # Update activity timestamps for existing rules based on the current simulation
        if cfg.prune_after_iters is not None:
            for idx, count in enumerate(sim.rule_counts):
                if count > 0:
                    last_seen[idx] = it

        # Enforce max_rule_usage on existing rules; prune offenders and re-run loop
        if cfg.max_rule_usage is not None and len(rules) > 0:
            over = [i for i, u in enumerate(sim.rule_usage) if u > cfg.max_rule_usage]
            if over:
                _prune_indices(over)
                continue

        mined = top_ngrams(
            sim.seqs,
            min_n=cfg.proposal_min_len,
            max_n=cfg.proposal_max_len,
            top_k=cfg.proposal_top_k,
        )

        existing = {r.prefix for r in rules}
        tried_prefixes: set[Tuple[int, ...]] = set()

        accepted = False
        accepted_sim = None
        accepted_rule_obj: Optional[Rule] = None
        accepted_entropy: Optional[float] = None
        accepted_replace_idx: Optional[int] = None

        for _ in range(cfg.max_prefix_tries):
            prefix = _pick_prefix(
                rng=rng,
                mined=mined,
                existing=existing,
                tried=tried_prefixes,
                temperature=temperature,
                length_selection=cfg.length_selection,
                min_prefix_count=min_prefix_count,
            )
            if prefix is None:
                break

            tried_prefixes.add(prefix)

            n = len(prefix)
            prefix_count_before = 0
            for s in mined.get(n, []):
                if s.ngram == prefix:
                    prefix_count_before = s.count
                    break

            cand_rule = _propose_rule(prefix)
            cand_rules = rules + [cand_rule]

            cand_sim = simulate(
                base_probs=base_probs,
                rules=cand_rules,
                num_sims=cfg.num_sims,
                sim_len=cfg.sim_len,
                seed=int(rng.integers(0, 2**32 - 1)),
                precompute_transitions=cfg.precompute_transitions,
                parallel=cfg.simulate_parallel,
                max_workers=cfg.simulate_max_workers,
            )

            usage = cand_sim.rule_usage
            new_u = float(usage[-1]) if usage.size else 0.0

            ok = True
            reason = "ok"
            if usage.size:
                min_usage_bound = cfg.min_rule_usage if cfg.min_rule_usage is not None else 0.0
                # Always require the new rule to fire at least once on average
                min_usage_bound = max(min_usage_bound, 1.0 / float(cfg.total_steps))
                if new_u < min_usage_bound:
                    ok = False
                    reason = f"LOW: new={new_u*100:.2f}% < {min_usage_bound*100:.2f}%"
                elif cfg.max_rule_usage is not None and new_u > cfg.max_rule_usage:
                    ok = False
                    reason = f"HIGH: new={new_u*100:.2f}% > {cfg.max_rule_usage*100:.2f}%"

            cand_stats = None
            if ok:
                cand_stats = _entropy_stats(cand_rules)
                cand_entropy = cand_stats.entropy_bits
                allow_increase = None
                if cfg.max_entropy_increase is not None:
                    allow_increase = (
                        cfg.max_entropy_increase
                        if len(iter_hist.attempts) >= cfg.max_prefix_tries // 2
                        else 0.0
                    )
                max_drop = cfg.max_entropy_decrease

                if allow_increase is not None and cand_entropy > current_entropy_bits + allow_increase:
                    ok = False
                    reason = (
                        f"ENTROPY_UP: cand={cand_entropy:.4f} > "
                        f"cur={current_entropy_bits:.4f} (allow +{allow_increase:.4f})"
                    )
                elif max_drop is not None and cand_entropy < current_entropy_bits - max_drop:
                    ok = False
                    reason = (
                        f"ENTROPY_DROP: cand={cand_entropy:.4f} < "
                        f"cur={current_entropy_bits:.4f} - {max_drop:.4f}"
                    )
                elif cfg.min_entropy is not None and cand_entropy < cfg.min_entropy:
                    ok = False
                    reason = (
                        f"ENTROPY_LOW: cand={cand_entropy:.4f} < "
                        f"min_entropy={cfg.min_entropy:.4f}"
                    )

            iter_hist.attempts.append(
                ProposalAttempt(
                    iteration=it,
                    n=n,
                    prefix=prefix,
                    prefix_count_before=prefix_count_before,
                    proposed_rule=cand_rule.describe(),
                    accepted=ok,
                    reason=reason,
                    cand_new_rule_usage=new_u,
                )
            )

            if ok:
                accepted = True
                accepted_sim = cand_sim
                accepted_rule_obj = cand_rule
                accepted_entropy = cand_entropy
                if cand_stats is not None:
                    entropy_stats = cand_stats
                break

        iter_hist.accepted = accepted

        if accepted:
            assert accepted_rule_obj is not None
            rules.append(accepted_rule_obj)
            sim = accepted_sim  # type: ignore[assignment]
            if accepted_entropy is not None:
                current_entropy_bits = float(accepted_entropy)

            iter_hist.accepted_rule = accepted_rule_obj.describe()
            iter_hist.base_usage_after = sim.base_usage
            iter_hist.rule_usage_after = sim.rule_usage
            iter_hist.top_after = top_ngrams(
                sim.seqs,
                min_n=cfg.report_min_len,
                max_n=cfg.report_max_len,
                top_k=cfg.report_top_k,
            )
            iter_hist.rules_after = [r.describe() for r in rules]
            iter_hist.entropy_bits = current_entropy_bits
            if cfg.prune_after_iters is not None:
                # Refresh activity for any rule that fired in this simulation
                for idx, count in enumerate(sim.rule_counts):
                    if count > 0:
                        last_seen[idx] = it

        history.append(iter_hist)

        if cfg.prune_after_iters is not None:
            to_prune: List[int] = []
            for idx in range(len(rules)):
                seen_iter = last_seen.get(idx, -1)
                if it - seen_iter >= cfg.prune_after_iters:
                    to_prune.append(idx)
            if to_prune:
                _prune_indices(to_prune)

        if not accepted:
            if cfg.entropy_target is not None and current_entropy_bits > cfg.entropy_target:
                temperature *= cfg.entropy_temp_multiplier
            it += 1
            continue

        it += 1

    if cfg.enforce_min_rules and len(rules) < cfg.min_rules:
        raise RuntimeError(f"Could not reach min_rules={cfg.min_rules}; got {len(rules)}")

    final_top = top_ngrams(sim.seqs, cfg.report_min_len, cfg.report_max_len, cfg.report_top_k)

    # NEW: compile an efficient generator for production sampling
    generator = compile_generator(base_probs, rules, precompute_transitions=cfg.precompute_transitions)

    return rules, sim, final_top, history, generator
