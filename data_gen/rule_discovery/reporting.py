from __future__ import annotations
from typing import List

from .config import DiscoveryConfig, IterationHistory
from .entropy import analytic_entropy


def print_iteration(h: IterationHistory, cfg: DiscoveryConfig) -> None:
    hist_rows = cfg.max_rows_per_n_history
    if hist_rows is None:
        hist_rows = max(1, cfg.max_rows_per_n_final // 2)

    print("\n" + "=" * 78)
    print(
        f"ITER {h.iteration} | rules_before={h.rules_before} | "
        f"base_before={h.base_usage_before*100:.2f}%"
    )

    if not h.attempts:
        print("No proposals.")
    else:
        print("Proposals (prefix(count_before), n : rule):")
        rej_printed = 0
        for a in h.attempts:
            left = f"{a.prefix} (c={a.prefix_count_before}, n={a.n}) : {a.proposed_rule}"
            if a.accepted:
                print(f"  {left} : ACCEPT (new_usage={a.cand_new_rule_usage*100:.2f}%)")
            else:
                if cfg.print_rejections and rej_printed < cfg.max_rejections_printed_per_iter:
                    print(f"  {left} : REJECT ({a.reason})")
                    rej_printed += 1

        if cfg.print_rejections:
            total_rej = sum(1 for a in h.attempts if not a.accepted)
            if total_rej > rej_printed:
                print(f"  ... {total_rej - rej_printed} more rejection(s) not shown")

    if not h.accepted:
        print("Result: FAILED to add a rule.")
        return

    print(f"Result: ACCEPTED rule: {h.accepted_rule}")
    print(f"After: base={h.base_usage_after*100:.2f}%  rules={h.rules_before+1}")
    if h.entropy_bits is not None:
        print(f"  Entropy rate (analytic): {h.entropy_bits:.4f} bits")

    if h.rules_after is not None and h.rule_usage_after is not None:
        usage = h.rule_usage_after
        print("Rules and usage (% steps):")
        for i, rule_str in enumerate(h.rules_after):
            u = usage[i] * 100.0
            marker = "  <== new" if (h.accepted_rule is not None and rule_str == h.accepted_rule) else ""
            print(f"  rule[{i:02d}] {rule_str}  applied={u:6.2f}%{marker}")

    if h.top_after:
        print("\nTop n-grams (history view; pct within n):")
        for n in sorted(h.top_after.keys()):
            rows = h.top_after[n][:hist_rows]
            if not rows:
                continue
            print(f"  n={n}:")
            for s in rows:
                print(f"    {s.ngram} : {s.pct*100:6.2f}% (count={s.count})")


def print_final(rules, sim, top, cfg: DiscoveryConfig, base_probs) -> None:
    print("\n" + "#" * 78)
    print("FINAL SUMMARY")
    print(f"Mode: {cfg.rule_mode}")
    print(f"Trajectories: K={sim.num_sims}, N={sim.sim_len}  (total steps={sim.total_steps})")
    stats = analytic_entropy(
        base_probs=base_probs,
        rules=rules,
        precompute_transitions=cfg.precompute_transitions,
    )
    print(f"Entropy rate (analytic): {stats.entropy_bits:.4f} bits")
    print(f"Base applied: {sim.base_usage*100:.2f}%")
    print(f"Rules: {len(rules)}")
    if len(rules) < cfg.min_rules:
        print(f"WARNING: min_rules={cfg.min_rules} but only accepted {len(rules)} rule(s).")

    if rules:
        usage = sim.rule_usage
        print("\nFinal rules (rule, usage % of steps):")
        for i, r in enumerate(rules):
            print(f"  rule[{i:02d}] {r.describe()}  applied={usage[i]*100:6.2f}%")

    print("\nTop n-grams (final; pct within n):")
    for n in sorted(top.keys()):
        rows = top[n][:cfg.max_rows_per_n_final]
        if not rows:
            continue
        print(f"  n={n}:")
        for s in rows:
            print(f"    {s.ngram} : {s.pct*100:6.2f}% (count={s.count})")
