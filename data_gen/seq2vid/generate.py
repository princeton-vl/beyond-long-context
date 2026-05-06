from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from rule_discovery.config import DiscoveryConfig
from rule_discovery.discover import discover_rules
from rule_discovery.entropy import analytic_entropy
from rule_discovery.rules import make_base_probs

LOG2E = math.log2(math.e)


def ints_to_tokens(seq: Sequence[int]) -> List[str]:
    return [str(int(x)) for x in seq]



def _ngram_counts(seq_tokens: Sequence[str], min_len: int = 1, max_len: int = 6) -> dict:
    counts = {}
    n_tokens = len(seq_tokens)
    for n in range(min_len, max_len + 1):
        if n > n_tokens:
            break
        for i in range(0, n_tokens - n + 1):
            g = tuple(seq_tokens[i : i + n])
            counts[g] = counts.get(g, 0) + 1
    return counts


def _top_ngrams(tokens: Sequence[str], min_len: int = 1, max_len: int = 6, top_k: int = 500) -> List[dict]:
    counts = _ngram_counts(tokens, min_len=min_len, max_len=max_len)
    out = []
    for n in range(min_len, max_len + 1):
        items = [(g, c) for g, c in counts.items() if len(g) == n]
        total = sum(c for _, c in items) or 1
        items.sort(key=lambda x: x[1], reverse=True)
        if len(items) > top_k:
            items = items[:top_k]
        for g, c in items:
            out.append({"n": n, "ngram": list(g), "count": c, "pct": c / float(total)})
    return out


def _generate_task(
    job: "GenerationJob",
    discover_len_mult: float,
    max_rules: int,
    rule_mode: str,
    max_attempts: int,
    top_k: int,
    ngram_max: int,
    total: int,
    log_progress: bool,
) -> GeneratedSequence:
    rng_local = np.random.default_rng(job.seed)
    if max_rules <= 0:
        tokens = [str(int(rng_local.integers(0, job.vocab_size))) for _ in range(job.seq_len)]
        ent = math.log2(job.vocab_size) if job.vocab_size > 1 else 0.0
        if ent < job.entropy_min or ent > job.entropy_max:
            raise RuntimeError(
                f"Uniform sampling entropy {ent:.3f} out of bounds for seq_id={job.seq_id}"
            )
        top_stats = _top_ngrams(tokens, min_len=1, max_len=ngram_max, top_k=top_k)
        return GeneratedSequence(
            seq_id=job.seq_id,
            tokens=tokens,
            entropy=ent,
            vocab_size=job.vocab_size,
            length=len(tokens),
            config={"entropy_min": job.entropy_min, "entropy_max": job.entropy_max},
            top_ngrams=top_stats,
            rule_mode="uniform",
            max_prefix_len=0,
            prefix_histogram={},
            ruleset_entropy=0.0,
        )
    attempts = 0
    last_ent = None
    while attempts < max_attempts:
        attempts += 1
        cfg = _build_config(
            vocab_size=job.vocab_size,
            seq_len=int(job.seq_len * discover_len_mult),
            entropy_min=job.entropy_min,
            entropy_max=job.entropy_max,
            max_rules=max_rules,
            rule_mode=rule_mode,
            proposal_min_len=job.proposal_min_len,
            proposal_max_len=job.proposal_max_len,
            min_rule_usage=job.min_rule_usage,
            max_rule_usage=job.max_rule_usage,
            seed=int(rng_local.integers(0, 2**32 - 1)),
            disable_entropy_drop_guard=job.disable_entropy_drop_guard,
        )
        base = make_base_probs(job.vocab_size)
        rules, sim, top, history, generator = discover_rules(base, cfg)
        _ = (sim, top, history)
        seq_ints = generator.generate(
            num_sims=1, sim_len=job.seq_len, seed=int(rng_local.integers(0, 2**32 - 1))
        )
        tokens = ints_to_tokens(seq_ints)
        entropy_stats = analytic_entropy(
            base_probs=generator.base_probs,
            rules=generator.rules,
            automaton=generator.automaton,
            precompute_transitions=generator.precompute_transitions,
        )
        ent = entropy_stats.entropy_bits
        last_ent = ent
        if ent < job.entropy_min or ent > job.entropy_max:
            continue
        top_stats = _top_ngrams(tokens, min_len=1, max_len=ngram_max, top_k=top_k)
        prefix_lengths = [len(r.prefix) for r in rules]
        prefix_hist: Dict[int, int] = {}
        for L in prefix_lengths:
            prefix_hist[L] = prefix_hist.get(L, 0) + 1
        max_prefix = max(prefix_lengths) if prefix_lengths else 0
        ruleset_entropy = float(getattr(generator, "ruleset_entropy", 0.0))
        return GeneratedSequence(
            seq_id=job.seq_id,
            tokens=tokens,
            entropy=ent,
            vocab_size=job.vocab_size,
            length=len(tokens),
            config={"entropy_min": job.entropy_min, "entropy_max": job.entropy_max},
            top_ngrams=top_stats,
            rule_mode=rule_mode,
            max_prefix_len=max_prefix,
            prefix_histogram=prefix_hist,
            ruleset_entropy=ruleset_entropy,
        )
    raise RuntimeError(
        f"Failed to hit entropy bounds for seq_id={job.seq_id} after {max_attempts} attempts "
        f"(last entropy={last_ent})"
    )


def _build_config(
    vocab_size: int,
    seq_len: int,
    entropy_min: float,
    entropy_max: float,
    max_rules: int,
    rule_mode: str,
    proposal_min_len: int,
    proposal_max_len: int,
    min_rule_usage: Optional[float],
    max_rule_usage: Optional[float],
    seed: Optional[int],
    disable_entropy_drop_guard: bool,
) -> DiscoveryConfig:
    rng = np.random.default_rng(seed)
    target = float(rng.uniform(entropy_min, entropy_max))
    max_entropy_increase = 0.0
    max_entropy_decrease = None if disable_entropy_drop_guard else 0.3 * LOG2E

    return DiscoveryConfig(
        num_sims=1,
        sim_len=seq_len,
        rule_mode=rule_mode,
        prob_reliability_min=0.9,
        prob_reliability_max=0.99,
        prob_support_max_divisor=6.0,
        proposal_min_len=int(proposal_min_len),
        proposal_max_len=int(proposal_max_len),
        proposal_top_k=40,
        proposal_min_prefix_count=None,
        proposal_temperature=1.0,
        length_selection="weighted",
        output_choice="base",
        max_prefix_tries=30,
        min_rules=0,
        max_rules=max_rules,
        enforce_min_rules=False,
        min_rule_usage=min_rule_usage,
        max_rule_usage=max_rule_usage,
        max_entropy_increase=max_entropy_increase,
        max_entropy_decrease=max_entropy_decrease,
        min_entropy=entropy_min,
        entropy_target=target,
        entropy_temp_multiplier=1.0,
        prune_after_iters=3,
        report_min_len=2,
        report_max_len=6,
        report_top_k=30,
        print_rejections=False,
        max_rejections_printed_per_iter=0,
        max_rows_per_n_final=10,
        max_rows_per_n_history=None,
        precompute_transitions=True,
        simulate_parallel=None,
        simulate_max_workers=None,
        seed=seed,
    )


@dataclass
class GeneratedSequence:
    seq_id: str
    tokens: List[str]
    entropy: float
    vocab_size: int
    length: int
    config: dict
    top_ngrams: List[dict]
    rule_mode: str
    max_prefix_len: int
    prefix_histogram: Dict[int, int]
    ruleset_entropy: float


@dataclass(frozen=True)
class GenerationJob:
    index: int
    seq_id: str
    seq_len: int
    vocab_size: int
    entropy_min: float
    entropy_max: float
    proposal_min_len: int
    proposal_max_len: int
    min_rule_usage: Optional[float]
    max_rule_usage: Optional[float]
    disable_entropy_drop_guard: bool
    seed: int


def _broadcast_list(values, fallback, n: int):
    seq = list(values) if values is not None else []
    if not seq:
        seq = [fallback]
    if len(seq) < n:
        seq += [seq[-1]] * (n - len(seq))
    return seq[:n]


def _prepare_generation_jobs(
    seq_ids: List[str],
    seq_lens: List[int],
    vocab_sizes: List[int],
    entropy_mins: List[float],
    entropy_maxs: List[float],
    proposal_min_lens: List[int],
    proposal_max_lens: List[int],
    min_rule_usages: List[Optional[float]],
    max_rule_usages: List[Optional[float]],
    disable_entropy_drop_guards: List[bool],
    rng: np.random.Generator,
) -> List[GenerationJob]:
    n = len(seq_ids)
    jobs: List[GenerationJob] = []
    for idx in range(n):
        jobs.append(
            GenerationJob(
                index=idx,
                seq_id=seq_ids[idx],
                seq_len=seq_lens[idx],
                vocab_size=vocab_sizes[idx],
                entropy_min=entropy_mins[idx],
                entropy_max=entropy_maxs[idx],
                proposal_min_len=proposal_min_lens[idx],
                proposal_max_len=proposal_max_lens[idx],
                min_rule_usage=min_rule_usages[idx],
                max_rule_usage=max_rule_usages[idx],
                disable_entropy_drop_guard=disable_entropy_drop_guards[idx],
                seed=int(rng.integers(0, 2**32 - 1)),
            )
        )
    return jobs


def _execute_generation_jobs(
    jobs: List[GenerationJob],
    gen_workers: int,
    discover_len_mult: float,
    max_rules: int,
    rule_mode: str,
    max_attempts: int,
    top_k: int,
    ngram_max: int,
    log_progress: bool,
    skip_failures: bool,
) -> List[GeneratedSequence]:
    total = len(jobs)
    results: List[GeneratedSequence] = []

    def _run_job(job: GenerationJob) -> GeneratedSequence:
        if log_progress:
            print(f"[gen start] {job.index+1}/{total} id={job.seq_id}", flush=True)
        return _generate_task(
            job,
            discover_len_mult=discover_len_mult,
            max_rules=max_rules,
            rule_mode=rule_mode,
            max_attempts=max_attempts,
            top_k=top_k,
            ngram_max=ngram_max,
            total=total,
            log_progress=log_progress,
        )

    if gen_workers <= 1:
        for job in jobs:
            try:
                results.append(_run_job(job))
            except RuntimeError as exc:
                if skip_failures:
                    if log_progress:
                        print(f"[gen skip] id={job.seq_id} reason={exc}", flush=True)
                    continue
                raise
            finally:
                if log_progress:
                    print(f"[gen done] {job.index+1}/{total}", flush=True)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=gen_workers) as ex:
        future_map = {}
        for job in jobs:
            if log_progress:
                print(f"[gen submit] {job.index+1}/{total} id={job.seq_id}", flush=True)
            fut = ex.submit(
                _generate_task,
                job,
                discover_len_mult,
                max_rules,
                rule_mode,
                max_attempts,
                top_k,
                ngram_max,
                total,
                log_progress,
            )
            future_map[fut] = job
        completed = 0
        for fut in as_completed(future_map):
            job = future_map[fut]
            try:
                results.append(fut.result())
            except RuntimeError as exc:
                if skip_failures:
                    if log_progress:
                        print(f"[gen skip] id={job.seq_id} reason={exc}", flush=True)
                    continue
                raise
            finally:
                completed += 1
                if log_progress:
                    print(f"[gen done] {completed}/{total} id={job.seq_id}", flush=True)
    return results

def run_generation(
    out_dir: Path,
    num_seqs: int,
    seq_ids: List[str],
    seq_lens: List[int],
    discover_len_mult: float,
    vocab_sizes: List[int],
    seed: Optional[int],
    entropy_mins: List[float],
    entropy_maxs: List[float],
    max_rules: int,
    rule_mode: str,
    max_attempts: int,
    top_k: int,
    ngram_max: int,
    gen_workers: int,
    log_progress: bool,
    proposal_min_lens: Optional[List[int]] = None,
    proposal_max_lens: Optional[List[int]] = None,
    min_rule_usages: Optional[List[Optional[float]]] = None,
    max_rule_usages: Optional[List[Optional[float]]] = None,
    disable_entropy_drop_guard: Optional[List[bool]] = None,
    skip_failures: bool = False,
) -> List[GeneratedSequence]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seq = len(seq_ids)
    seq_lens = _broadcast_list(seq_lens, 120, n_seq)
    vocab_sizes = _broadcast_list(vocab_sizes, 16, n_seq)
    entropy_mins = _broadcast_list(entropy_mins, 0.15, n_seq)
    entropy_maxs = _broadcast_list(entropy_maxs, 0.45, n_seq)
    proposal_min_lens = _broadcast_list(proposal_min_lens, 1, n_seq)
    proposal_max_lens = _broadcast_list(proposal_max_lens, 5, n_seq)
    min_rule_usages = _broadcast_list(min_rule_usages, 0.01, n_seq)
    max_rule_usages = _broadcast_list(max_rule_usages, 0.20, n_seq)
    disable_entropy_drop_guard = _broadcast_list(disable_entropy_drop_guard, False, n_seq)

    for mn_len, mx_len in zip(proposal_min_lens, proposal_max_lens):
        if mn_len <= 0 or mx_len <= 0:
            raise ValueError("proposal min/max lengths must be positive")
        if mn_len > mx_len:
            raise ValueError("proposal_min_len cannot exceed proposal_max_len")

    if any(l <= 0 for l in seq_lens):
        raise ValueError("All sequence lengths must be positive")
    if any(v <= 0 for v in vocab_sizes):
        raise ValueError("All vocab sizes must be positive")
    if any(mn < 0 or mx < 0 for mn, mx in zip(entropy_mins, entropy_maxs)):
        raise ValueError("Entropy bounds must be non-negative")
    for mn, mx, sid in zip(entropy_mins, entropy_maxs, seq_ids):
        if mn > mx:
            raise ValueError(f"entropy_min > entropy_max for seq_id={sid}")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if ngram_max < 1:
        raise ValueError("ngram_max must be >= 1")
    jobs = _prepare_generation_jobs(
        seq_ids=seq_ids,
        seq_lens=seq_lens,
        vocab_sizes=vocab_sizes,
        entropy_mins=entropy_mins,
        entropy_maxs=entropy_maxs,
        proposal_min_lens=proposal_min_lens,
        proposal_max_lens=proposal_max_lens,
        min_rule_usages=min_rule_usages,
        max_rule_usages=max_rule_usages,
        disable_entropy_drop_guards=disable_entropy_drop_guard,
        rng=rng,
    )

    results = _execute_generation_jobs(
        jobs=jobs,
        gen_workers=gen_workers,
        discover_len_mult=discover_len_mult,
        max_rules=max_rules,
        rule_mode=rule_mode,
        max_attempts=max_attempts,
        top_k=top_k,
        ngram_max=ngram_max,
        log_progress=log_progress,
        skip_failures=skip_failures,
    )

    payload = {"sequences": [sequence_to_dict(r) for r in results]}
    out_path = out_dir / "sequences.json"
    out_path.write_text(json.dumps(payload, indent=2))
    if log_progress:
        print(f"[write] {out_path}")
    return results


def sequence_to_dict(r: GeneratedSequence) -> dict:
    return {
        "seq_id": r.seq_id,
        "tokens": r.tokens,
        "entropy": r.entropy,
        "vocab_size": r.vocab_size,
        "length": r.length,
        "top_ngrams": r.top_ngrams,
        "top_ngram_mass": {
            str(n): float(sum(item["pct"] for item in r.top_ngrams if item["n"] == n))
            for n in range(1, (max(item["n"] for item in r.top_ngrams) if r.top_ngrams else 0) + 1)
        },
        "rule_mode": r.rule_mode,
        "max_prefix_len": r.max_prefix_len,
        "prefix_histogram": r.prefix_histogram,
        "ruleset_entropy": r.ruleset_entropy,
        "ruleset_entropy_units": "nats",
        "entropy_units": "bits",
    }
