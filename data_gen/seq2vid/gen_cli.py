from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .buckets import BucketSpec, filter_buckets, load_bucket_config
from .generate import run_generation, sequence_to_dict


def _parse_csv_ints(s: Optional[str], fallback: int, count: int) -> List[int]:
    if s is None:
        return [fallback] * count
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return [fallback] * count
    if len(parts) < count:
        parts += [parts[-1]] * (count - len(parts))
    return [int(p) for p in parts[:count]]


def _parse_csv_floats(s: Optional[str], fallback: float, count: int) -> List[float]:
    if s is None:
        return [fallback] * count
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return [fallback] * count
    if len(parts) < count:
        parts += [parts[-1]] * (count - len(parts))
    return [float(p) for p in parts[:count]]


def _validate_bucket_specs(specs: Iterable[BucketSpec], default_vocab: int) -> None:
    for spec in specs:
        vocab = spec.vocab_size or default_vocab
        if vocab <= 0:
            raise ValueError(f"Bucket {spec.bucket_id} has non-positive vocab_size")
        if spec.seq_len_min <= 0 or spec.seq_len_max < spec.seq_len_min:
            raise ValueError(
                f"Bucket {spec.bucket_id} has invalid seq_len range [{spec.seq_len_min}, {spec.seq_len_max}]"
            )
        if spec.entropy_min < 0 or spec.entropy_max < spec.entropy_min:
            raise ValueError(
                f"Bucket {spec.bucket_id} has invalid entropy range [{spec.entropy_min}, {spec.entropy_max}]"
            )
        max_rules = spec.max_rules if spec.max_rules is not None else None
        if max_rules == 0:
            uniform_entropy = math.log2(vocab) if vocab > 1 else 0.0
            if uniform_entropy < spec.entropy_min or uniform_entropy > spec.entropy_max:
                raise ValueError(
                    f"Bucket {spec.bucket_id} cannot satisfy entropy bounds with max_rules=0 "
                    f"(uniform entropy={uniform_entropy:.3f} bits, range=[{spec.entropy_min}, {spec.entropy_max}])"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sequences and save to JSON.")
    ap.add_argument("--out-dir", default="runs_seq", help="Directory for outputs (sequences.json)")
    ap.add_argument("--num-seqs", type=int, default=1, help="Total sequences to generate (auto-named)")
    ap.add_argument("--seq-lens", type=str, default="120", help="Comma-separated lengths (single value broadcasts)")
    ap.add_argument("--sequences", type=str, default="", help="Deprecated; ignored. Names are auto-assigned.")
    ap.add_argument("--discover-len-mult", type=float, default=4.0, help="Multiplier for discovery sim length")
    ap.add_argument("--max-attempts", type=int, default=10, help="Max attempts per sequence to hit entropy bounds")
    ap.add_argument("--vocab-sizes", type=str, default="16", help="Comma-separated per-sequence vocab sizes (single value broadcasts)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    ap.add_argument(
        "--entropy-mins",
        type=str,
        default="0.15",
        help="Comma-separated per-sequence entropy mins in bits/symbol (single value broadcasts)",
    )
    ap.add_argument(
        "--entropy-maxs",
        type=str,
        default="0.45",
        help="Comma-separated per-sequence entropy maxs in bits/symbol (single value broadcasts)",
    )
    ap.add_argument(
        "--disable-entropy-drop-guard",
        action="store_true",
        help="Disable the entropy drop guard (allows larger downward jumps)",
    )
    # Backward compatibility
    ap.add_argument("--disable-entropy-guards", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--top-k", type=int, default=500, help="Top-k ngrams to store per length")
    ap.add_argument("--ngram-max", type=int, default=6, help="Maximum n-gram length to store (min is 1)")
    ap.add_argument("--max-rules", type=int, default=100, help="Maximum rules for discovery")
    ap.add_argument("--rule-mode", type=str, default="probabilistic", choices=["probabilistic", "deterministic"], help="Rule mode for discovery")
    ap.add_argument("--gen-workers", type=int, default=1, help="Parallel workers for sequence generation")
    ap.add_argument("--proposal-min-lens", type=str, default=None, help="Optional comma-separated proposal min prefix lengths (broadcasted)")
    ap.add_argument("--proposal-max-lens", type=str, default=None, help="Optional comma-separated proposal max prefix lengths (broadcasted)")
    ap.add_argument("--log-progress", action="store_true", help="Enable verbose progress logging")

    # Bucket mode controls
    ap.add_argument("--bucket-config", type=str, default=None, help="Path to bucket YAML to enable bucket-based generation")
    ap.add_argument("--include-buckets", nargs="*", default=None, help="Glob(s) of bucket IDs to include in bucket mode")
    ap.add_argument("--exclude-buckets", nargs="*", default=None, help="Glob(s) of bucket IDs to exclude in bucket mode")
    ap.add_argument("--bucket-batch-size", type=int, default=16, help="Sequences to request per generation batch in bucket mode")
    ap.add_argument("--bucket-seed", type=int, default=None, help="Seed for bucket sampling (seq lens and RNG)")
    ap.add_argument("--bucket-overwrite", action="store_true", help="Overwrite existing bucket outputs")
    ap.add_argument("--bucket-write-combined", action="store_true", help="Write combined sequences JSON across all processed buckets")
    ap.add_argument(
        "--bucket-max-no-progress",
        type=int,
        default=3,
        help="Max consecutive batches without new sequences before aborting a bucket",
    )

    args = ap.parse_args()

    if args.bucket_config:
        run_bucket_mode(args)
        return

    seq_ids = [f"SEQ_{i+1}" for i in range(args.num_seqs)]
    seq_lens = _parse_csv_ints(args.seq_lens, 120, len(seq_ids))
    vocab_sizes = _parse_csv_ints(args.vocab_sizes, 16, len(seq_ids))
    entropy_mins = _parse_csv_floats(args.entropy_mins, 0.15, len(seq_ids))
    entropy_maxs = _parse_csv_floats(args.entropy_maxs, 0.45, len(seq_ids))
    proposal_min_lens = _parse_csv_ints(args.proposal_min_lens, 1, len(seq_ids))
    proposal_max_lens = _parse_csv_ints(args.proposal_max_lens, 5, len(seq_ids))

    disable_entropy_drop_guard = bool(args.disable_entropy_drop_guard or args.disable_entropy_guards)

    run_generation(
        out_dir=Path(args.out_dir),
        num_seqs=len(seq_ids),
        seq_ids=seq_ids,
        seq_lens=seq_lens,
        discover_len_mult=args.discover_len_mult,
        vocab_sizes=vocab_sizes,
        seed=args.seed,
        entropy_mins=entropy_mins,
        entropy_maxs=entropy_maxs,
        max_rules=args.max_rules,
        rule_mode=args.rule_mode,
        max_attempts=args.max_attempts,
        top_k=args.top_k,
        ngram_max=args.ngram_max,
        gen_workers=args.gen_workers,
        log_progress=args.log_progress,
        proposal_min_lens=proposal_min_lens,
        proposal_max_lens=proposal_max_lens,
        disable_entropy_drop_guard=[disable_entropy_drop_guard],
    )


def run_bucket_mode(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = load_bucket_config(Path(args.bucket_config))
    all_specs = cfg.list_buckets()
    buckets = filter_buckets(all_specs, args.include_buckets, args.exclude_buckets)
    if not buckets:
        raise ValueError("No buckets matched the provided include/exclude filters")

    rng = np.random.default_rng(args.bucket_seed)
    default_vocab = _parse_csv_ints(args.vocab_sizes, 16, 1)[0]
    _validate_bucket_specs(buckets, default_vocab)
    default_min_rule_usage = 0.01
    default_max_rule_usage = 0.20
    max_no_progress = max(1, int(args.bucket_max_no_progress))

    manifest_path = out_root / "bucket_generation_manifest.json"
    manifest_map: Dict[str, dict] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing_manifest = {}
        for entry in existing_manifest.get("buckets", []):
            bid = entry.get("bucket_id")
            if bid:
                manifest_map[bid] = entry
    combined_sequences: List[dict] = []

    for spec in buckets:
        bucket_dir = out_root / spec.bucket_id
        bucket_dir.mkdir(parents=True, exist_ok=True)
        seq_path = bucket_dir / "sequences.json"
        meta_path = bucket_dir / "meta.json"
        existing_sequences: List[dict] = []
        if seq_path.exists() and not args.bucket_overwrite:
            data = json.loads(seq_path.read_text())
            existing_sequences = list(data.get("sequences", []))
        accepted = existing_sequences[:]
        rejected_entropy = 0
        rejected_prefix = 0
        batch_index = 0
        no_progress_batches = 0
        failed = False
        failure_reason = None
        low_entropy_bucket = spec.entropy_max <= 0.45
        bucket_max_rules = spec.max_rules if spec.max_rules is not None else args.max_rules
        while len(accepted) < spec.target_sequences and not failed:
            batch_index += 1
            batch_size = min(args.bucket_batch_size, spec.target_sequences - len(accepted))
            seq_ids = [f"{spec.bucket_id}_SEQ_{len(accepted)+i+1}" for i in range(batch_size)]
            seq_lens = [spec.pick_seq_len(rng) for _ in seq_ids]
            entropy_mins = [spec.entropy_min] * batch_size
            entropy_maxs = [spec.entropy_max] * batch_size
            vocab_sizes = [spec.vocab_size or default_vocab] * batch_size
            proposal_min_lens = [1] * batch_size
            proposal_max_lens = [spec.max_prefix_len] * batch_size
            min_rule_usage = None if low_entropy_bucket else default_min_rule_usage
            max_rule_usage = None if low_entropy_bucket else default_max_rule_usage
            min_rule_usage_list = [min_rule_usage] * batch_size
            max_rule_usage_list = [max_rule_usage] * batch_size
            disable_guards_flag = bool(
                spec.disable_entropy_drop_guard
                or args.disable_entropy_drop_guard
                or args.disable_entropy_guards
            )
            disable_entropy_drop_guard_list = [disable_guards_flag] * batch_size
            bucket_max_attempts = args.max_attempts
            if low_entropy_bucket and bucket_max_attempts < 50:
                bucket_max_attempts = 50
            tmp_dir = Path(tempfile.mkdtemp(prefix="bucket_gen_"))
            accepted_before = len(accepted)
            try:
                results = run_generation(
                    out_dir=tmp_dir,
                    num_seqs=batch_size,
                    seq_ids=seq_ids,
                    seq_lens=seq_lens,
                    discover_len_mult=args.discover_len_mult,
                    vocab_sizes=vocab_sizes,
                    seed=int(rng.integers(0, 2**32 - 1)),
                    entropy_mins=entropy_mins,
                    entropy_maxs=entropy_maxs,
                    max_rules=bucket_max_rules,
                    rule_mode=spec.rule_mode,
                    max_attempts=bucket_max_attempts,
                    top_k=args.top_k,
                    ngram_max=args.ngram_max,
                    gen_workers=args.gen_workers,
                    log_progress=args.log_progress,
                    proposal_min_lens=proposal_min_lens,
                    proposal_max_lens=proposal_max_lens,
                    min_rule_usages=min_rule_usage_list,
                    max_rule_usages=max_rule_usage_list,
                    disable_entropy_drop_guard=disable_entropy_drop_guard_list,
                    skip_failures=True,
                )
            except RuntimeError as exc:
                print(f"[warn] bucket {spec.bucket_id} batch {batch_index} failed: {exc}")
                results = []
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            for res in results:
                if res.entropy < spec.entropy_min or res.entropy > spec.entropy_max:
                    rejected_entropy += 1
                    continue
                if res.max_prefix_len > spec.max_prefix_len:
                    rejected_prefix += 1
                    continue
                accepted.append(sequence_to_dict(res))
                if len(accepted) >= spec.target_sequences:
                    break

            if len(accepted) == accepted_before:
                no_progress_batches += 1
                if no_progress_batches >= max_no_progress:
                    failed = True
                    failure_reason = (
                        f"no_progress_after_{max_no_progress}_batches"
                    )
                    print(
                        f"[warn] bucket {spec.bucket_id} giving up after {max_no_progress} batches without progress",
                        flush=True,
                    )
            else:
                no_progress_batches = 0

        payload = {"bucket_id": spec.bucket_id, "sequences": accepted}
        seq_path.parent.mkdir(parents=True, exist_ok=True)
        seq_path.write_text(json.dumps(payload, indent=2))
        status = "completed" if (len(accepted) >= spec.target_sequences and not failed) else "failed"
        meta = {
            "bucket_id": spec.bucket_id,
            "num_sequences": len(accepted),
            "target_sequences": spec.target_sequences,
            "rejected_entropy": rejected_entropy,
            "rejected_prefix": rejected_prefix,
            "seq_len_range": [spec.seq_len_min, spec.seq_len_max],
            "entropy_range": [spec.entropy_min, spec.entropy_max],
            "entropy_units": "bits",
            "max_prefix_len": spec.max_prefix_len,
            "rule_mode": spec.rule_mode,
            "max_rules": bucket_max_rules,
            "step_profile": spec.step_profile,
            "vocab_size": spec.vocab_size or default_vocab,
            "sequences_file": str(seq_path),
            "meta_file": str(meta_path),
            "status": status,
            "max_no_progress_batches": max_no_progress,
        }
        if status != "completed":
            meta["failure_reason"] = failure_reason or "incomplete"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        manifest_map[spec.bucket_id] = meta
        if args.bucket_write_combined:
            combined_sequences.extend(accepted)

    manifest_entries: List[dict] = []
    for spec in all_specs:
        entry = manifest_map.get(spec.bucket_id)
        if entry:
            manifest_entries.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"buckets": manifest_entries}, indent=2))
    if args.bucket_write_combined:
        combined_path = out_root / "sequences_combined.json"
        combined_path.write_text(json.dumps({"sequences": combined_sequences}, indent=2))


if __name__ == "__main__":
    main()
