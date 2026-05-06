from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml


LOG2E = math.log2(math.e)


@dataclass(frozen=True)
class StepProfile:
    name: str
    step_range: Sequence[float]
    fps: int
    num_questions: int


@dataclass(frozen=True)
class BucketSpec:
    bucket_id: str
    seq_len_min: int
    seq_len_max: int
    entropy_min: float
    entropy_max: float
    max_prefix_len: int
    rule_mode: str
    target_sequences: int
    target_videos: int
    step_profile: str
    vocab_size: Optional[int] = None
    disable_entropy_drop_guard: bool = False
    max_rules: Optional[int] = None

    def pick_seq_len(self, rng: np.random.Generator) -> int:
        if self.seq_len_min >= self.seq_len_max:
            return int(self.seq_len_min)
        return int(rng.integers(self.seq_len_min, self.seq_len_max + 1))


@dataclass(frozen=True)
class BucketConfig:
    step_profiles: Dict[str, StepProfile]
    buckets: Dict[str, BucketSpec]

    def get_step_profile(self, name: str) -> StepProfile:
        if name not in self.step_profiles:
            raise KeyError(f"Unknown step_profile {name!r}")
        return self.step_profiles[name]

    def list_buckets(self) -> List[BucketSpec]:
        return [self.buckets[k] for k in sorted(self.buckets.keys())]


def load_bucket_config(path: Path) -> BucketConfig:
    data = yaml.safe_load(path.read_text())
    if data is None:
        raise ValueError(f"Bucket config {path} is empty")
    raw_profiles = data.get("step_profiles", {})
    if not raw_profiles:
        raise ValueError("Bucket config missing step_profiles")
    step_profiles: Dict[str, StepProfile] = {}
    for name, cfg in raw_profiles.items():
        step_range = cfg.get("step_range")
        if not step_range or len(step_range) != 2:
            raise ValueError(f"step_profile {name} missing step_range")
        step_profiles[name] = StepProfile(
            name=name,
            step_range=tuple(float(x) for x in step_range),
            fps=int(cfg.get("fps", 1)),
            num_questions=int(cfg.get("num_questions", 40)),
        )

    raw_buckets = data.get("buckets", [])
    if not raw_buckets:
        raise ValueError("Bucket config missing buckets list")

    buckets: Dict[str, BucketSpec] = {}
    for entry in raw_buckets:
        bucket_id = str(entry.get("id"))
        if not bucket_id:
            raise ValueError("Bucket entry missing id")
        seq_cfg = entry.get("seq_len") or {}
        ent_cfg = entry.get("entropy") or {}
        seq_min = int(seq_cfg.get("min", seq_cfg.get("value", 0)))
        seq_max = int(seq_cfg.get("max", seq_cfg.get("value", seq_min)))
        ent_units = str(ent_cfg.get("units", "nats")).lower()
        ent_min = float(ent_cfg.get("min", 0.0))
        ent_max = float(ent_cfg.get("max", ent_min))
        if ent_units in ("nat", "nats"):
            ent_min *= LOG2E
            ent_max *= LOG2E
        elif ent_units in ("bit", "bits"):
            pass
        else:
            raise ValueError(f"Unknown entropy units {ent_units!r} for bucket {bucket_id}")
        max_prefix_len = int(entry.get("max_prefix_len", 1))
        rule_mode = entry.get("rule_mode", "probabilistic")
        step_profile = entry.get("step_profile")
        if step_profile not in step_profiles:
            raise ValueError(f"Bucket {bucket_id} references unknown step_profile {step_profile!r}")
        target_sequences = int(entry.get("target_sequences", entry.get("target_videos", 0)))
        target_videos = int(entry.get("target_videos", target_sequences))
        if target_sequences <= 0:
            raise ValueError(f"Bucket {bucket_id} has non-positive target_sequences")
        disable_entropy_drop_guard = bool(
            entry.get("disable_entropy_drop_guard", entry.get("disable_entropy_guards", False))
        )
        max_rules = entry.get("max_rules")
        if max_rules is not None:
            max_rules = int(max_rules)
            if max_rules < 0:
                raise ValueError(f"Bucket {bucket_id} has negative max_rules={max_rules}")
        buckets[bucket_id] = BucketSpec(
            bucket_id=bucket_id,
            seq_len_min=seq_min,
            seq_len_max=seq_max,
            entropy_min=ent_min,
            entropy_max=ent_max,
            max_prefix_len=max_prefix_len,
            rule_mode=str(rule_mode),
            max_rules=max_rules,
            target_sequences=target_sequences,
            target_videos=target_videos,
            step_profile=step_profile,
            vocab_size=(entry.get("vocab_size") if entry.get("vocab_size") is not None else None),
            disable_entropy_drop_guard=disable_entropy_drop_guard,
        )

    return BucketConfig(step_profiles=step_profiles, buckets=buckets)


def filter_buckets(
    specs: Iterable[BucketSpec],
    includes: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
) -> List[BucketSpec]:
    selected: List[BucketSpec] = []
    for spec in specs:
        if includes and not any(fnmatch.fnmatch(spec.bucket_id, pat) for pat in includes):
            continue
        if excludes and any(fnmatch.fnmatch(spec.bucket_id, pat) for pat in excludes):
            continue
        selected.append(spec)
    return selected
