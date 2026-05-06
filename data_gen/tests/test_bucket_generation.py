import math
from pathlib import Path

import pytest

from seq2vid.buckets import BucketSpec, StepProfile
from seq2vid.gen_cli import _validate_bucket_specs
from seq2vid.generate import run_generation


def test_uniform_generation_hits_upper_bound(tmp_path: Path) -> None:
    results = run_generation(
        out_dir=tmp_path,
        num_seqs=1,
        seq_ids=["SEQ_1"],
        seq_lens=[6],
        discover_len_mult=1.0,
        vocab_sizes=[16],
        seed=0,
        entropy_mins=[3.5],
        entropy_maxs=[4.0],
        max_rules=0,
        rule_mode="probabilistic",
        max_attempts=1,
        top_k=10,
        ngram_max=4,
        gen_workers=1,
        log_progress=False,
    )
    assert len(results) == 1
    assert math.isclose(results[0].entropy, math.log2(16), rel_tol=1e-5)


def test_validate_buckets_rejects_impossible_uniform() -> None:
    spec = BucketSpec(
        bucket_id="TEST",
        seq_len_min=4,
        seq_len_max=8,
        entropy_min=3.9,
        entropy_max=3.95,
        max_prefix_len=2,
        rule_mode="probabilistic",
        target_sequences=1,
        target_videos=1,
        step_profile="SMALL",
        vocab_size=16,
        disable_entropy_drop_guard=False,
        max_rules=0,
    )
    with pytest.raises(ValueError):
        _validate_bucket_specs([spec], default_vocab=16)


def test_validate_buckets_accepts_uniform_when_bounds_okay() -> None:
    spec = BucketSpec(
        bucket_id="TEST_OK",
        seq_len_min=4,
        seq_len_max=8,
        entropy_min=3.0,
        entropy_max=4.0,
        max_prefix_len=2,
        rule_mode="probabilistic",
        target_sequences=1,
        target_videos=1,
        step_profile="SMALL",
        vocab_size=16,
        disable_entropy_drop_guard=False,
        max_rules=0,
    )
    _validate_bucket_specs([spec], default_vocab=16)
