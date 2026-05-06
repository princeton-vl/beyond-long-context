import math
import numpy as np

from rule_discovery.config import DiscoveryConfig
from rule_discovery.rules import make_base_probs
from rule_discovery.discover import discover_rules
from rule_discovery.reporting import print_iteration, print_final


LOG2E = math.log2(math.e)


def main() -> None:
    # Printing toggles
    PRINT_PROGRESS = True   # per-iteration history
    PRINT_FINAL = True      # final summary

    # Base distribution
    V = 16
    base = make_base_probs(V)  # uniform base

    cfg = DiscoveryConfig(
        # Simulation shape
        num_sims=1,
        sim_len=500,

        # Rule mode
        rule_mode="probabilistic",  # "deterministic" or "probabilistic"
        prob_reliability_min=0.90,
        prob_reliability_max=0.99,
        prob_support_max_divisor=6.0,

        # Proposal search
        proposal_min_len=1,
        proposal_max_len=5,
        proposal_top_k=10,
        proposal_min_prefix_count=None,
        proposal_temperature=1.0,
        length_selection="weighted",  # "uniform" or "weighted"
        output_choice="base",         # "base" or "uniform"
        max_prefix_tries=30,

        # Rule targets and acceptance
        min_rules=0,
        max_rules=200,
        enforce_min_rules=False,
        min_rule_usage=0.01,
        max_rule_usage=0.20,
        max_entropy_increase=0.0,
        max_entropy_decrease=0.2 * LOG2E,
        min_entropy=0.3 * LOG2E,
        entropy_target=0.5 * LOG2E,          # bits/symbol threshold
        entropy_temp_multiplier=1.0,
        prune_after_iters=3,

        # Reporting
        report_min_len=2,
        report_max_len=6,
        report_top_k=30,
        print_rejections=True,
        max_rejections_printed_per_iter=2,
        max_rows_per_n_final=10,
        max_rows_per_n_history=None,

        # Performance / parallelism
        precompute_transitions=True,
        simulate_parallel=None,
        simulate_max_workers=None,

        # Randomness
        seed=None,  # set to int for reproducible runs
    )

    rules, sim, top, history, generator = discover_rules(base, cfg)

    if PRINT_PROGRESS:
        for h in history:
            print_iteration(h, cfg)

    if PRINT_FINAL:
        print_final(rules, sim, top, cfg, base)

    if PRINT_FINAL:
        seq = generator.generate(num_sims=1, sim_len=cfg.sim_len * 2, seed=123)
        print("\nGenerated sequence length:", len(seq))
        print("First 25 symbols:", seq[:25])


if __name__ == "__main__":
    main()
