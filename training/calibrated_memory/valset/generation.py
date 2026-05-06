"""Synthetic manifest builder for binary membership/continuation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from calibrated_memory.data.sequences.common import TOKEN_OFFSET, YES_TOKEN, NO_TOKEN
from calibrated_memory.data.sequences.question_generator import (
    _draw_membership_queries,
    _draw_continuation_queries,
)


@dataclass(frozen=True)
class PowerBucket:
    name: str
    min_len: int
    max_len: int
    inclusive_upper: bool = False

    def sample_length(self, generator: torch.Generator) -> int:
        upper = self.max_len if self.inclusive_upper else self.max_len - 1
        if upper < self.min_len:
            raise ValueError(f"Bucket {self.name} has invalid range {self.min_len}-{self.max_len}")
        if self.min_len == upper:
            return self.min_len
        return int(torch.randint(self.min_len, upper + 1, (1,), generator=generator).item())


DEFAULT_BUCKETS: Sequence[PowerBucket] = (
    PowerBucket("seq16-32", 16, 32),
    PowerBucket("seq32-64", 32, 64),
    PowerBucket("seq64-128", 64, 128),
    PowerBucket("seq128-256", 128, 256),
    PowerBucket("seq256-512", 256, 512),
    PowerBucket("seq512-1024", 512, 1024),
    PowerBucket("seq1024-2048", 1024, 2048, inclusive_upper=True),
)


@dataclass(frozen=True)
class ValGenerationConfig:
    task: str
    num_sequences: int
    queries_per_sequence: int
    vocab_size: int = 16
    token_offset: int = TOKEN_OFFSET
    min_query_len: int = 3
    max_query_len: int = 7
    cont_len: int = 4
    seed: int = 0
    buckets: Sequence[PowerBucket] = field(default_factory=lambda: DEFAULT_BUCKETS)


@dataclass(frozen=True)
class QuestionRow:
    video_index: int
    question_index: int
    bucket_id: str
    stream_length: int
    concerned_ranges: Sequence[tuple[int, int]]
    truth_kind: str


@dataclass(frozen=True)
class ValGenerationResult:
    manifest: dict[str, Any]
    bucket_counts: dict[str, int]
    question_rows: list[QuestionRow]


class SyntheticValGenerator:
    def __init__(self, config: ValGenerationConfig) -> None:
        if config.task not in {"membership", "continuation"}:
            raise ValueError(f"Unsupported validation task '{config.task}'")
        self.config = config
        self.generator = torch.Generator().manual_seed(int(config.seed))

    def build(self) -> ValGenerationResult:
        buckets = list(self.config.buckets)
        if not buckets:
            raise ValueError("At least one bucket is required")
        counts = self._allocate_counts(len(buckets))
        videos: list[dict[str, Any]] = []
        question_rows: list[QuestionRow] = []
        video_index = 0
        global_q_idx = 0
        for bucket, count in zip(buckets, counts):
            for _ in range(count):
                stream = self._sample_stream(bucket)
                manifest_stream = [str(tok - self.config.token_offset) for tok in stream]
                if self.config.task == "membership":
                    questions, question_rows = self._build_membership_questions(
                        stream,
                        manifest_stream,
                        bucket,
                        video_index,
                        global_q_idx,
                        question_rows,
                    )
                else:
                    questions, question_rows = self._build_continuation_questions(
                        stream,
                        manifest_stream,
                        bucket,
                        video_index,
                        global_q_idx,
                        question_rows,
                    )
                global_q_idx += len(questions)
                videos.append(
                    {
                        "video_index": video_index,
                        "variant": f"{self.config.task}-val",
                        "bucket_id": bucket.name,
                        "bucket_from": bucket.min_len,
                        "sequences_used": {"S_tokens": manifest_stream},
                        "entropy_overall": {
                            "empirical_bits": {"S_tokens": float(self.config.vocab_size)}
                        },
                        "questions": questions,
                        "questions_at_end": True,
                    }
                )
                video_index += 1
        manifest = {"task": self.config.task, "videos": videos}
        bucket_counts = {bucket.name: count for bucket, count in zip(buckets, counts)}
        return ValGenerationResult(manifest=manifest, bucket_counts=bucket_counts, question_rows=question_rows)

    def _allocate_counts(self, bucket_count: int) -> list[int]:
        base = self.config.num_sequences // bucket_count
        remainder = self.config.num_sequences % bucket_count
        counts = [base] * bucket_count
        for idx in range(remainder):
            counts[idx] += 1
        return counts

    def _sample_stream(self, bucket: PowerBucket) -> list[int]:
        length = bucket.sample_length(self.generator)
        high = self.config.token_offset + self.config.vocab_size
        stream = torch.randint(
            self.config.token_offset,
            high,
            (length,),
            generator=self.generator,
        )
        return stream.tolist()

    def _build_membership_questions(
        self,
        stream: list[int],
        manifest_stream: list[int],
        bucket: PowerBucket,
        video_index: int,
        start_index: int,
        rows: list[QuestionRow],
    ) -> tuple[list[dict[str, Any]], list[QuestionRow]]:
        pairs = _draw_membership_queries(
            stream,
            count=self.config.queries_per_sequence,
            min_len=self.config.min_query_len,
            max_len=self.config.max_query_len,
            generator=self.generator,
            vocab_size=self.config.vocab_size,
            token_offset=self.config.token_offset,
        )
        order = torch.randperm(len(pairs), generator=self.generator).tolist()
        questions: list[dict[str, Any]] = []
        for local_idx, idx in enumerate(order):
            candidate, is_present = pairs[idx]
            label = YES_TOKEN if is_present else NO_TOKEN
            q_idx = start_index + local_idx
            candidate_manifest = [str(tok - self.config.token_offset) for tok in candidate]
            ranges = (
                _find_subsequence_ranges(manifest_stream, candidate_manifest)
                if label == YES_TOKEN
                else []
            )
            questions.append(
                {
                    "question_index": q_idx,
                    "question_mode": "exists",
                    "question_format": "binary_yes_no",
                    "answer": "yes" if label == YES_TOKEN else "no",
                    "candidate": {
                        "sequence": candidate_manifest,
                        "present": label == YES_TOKEN,
                    },
                    "concerned_ranges": [{"start": start, "end": end} for start, end in ranges],
                }
            )
            rows.append(
                QuestionRow(
                    video_index=video_index,
                    question_index=q_idx,
                    bucket_id=bucket.name,
                    stream_length=len(stream),
                    concerned_ranges=ranges,
                    truth_kind="yes" if label == YES_TOKEN else "no",
                )
            )
        return questions, rows


    def _build_continuation_questions(
        self,
        stream: list[int],
        manifest_stream: list[int],
        bucket: PowerBucket,
        video_index: int,
        start_index: int,
        rows: list[QuestionRow],
    ) -> tuple[list[dict[str, Any]], list[QuestionRow]]:
        queries = _draw_continuation_queries(
            stream,
            count=self.config.queries_per_sequence,
            min_len=self.config.min_query_len,
            max_len=self.config.max_query_len,
            cont_len=self.config.cont_len,
            generator=self.generator,
            vocab_size=self.config.vocab_size,
            token_offset=self.config.token_offset,
        )
        if not queries:
            return [], rows
        questions: list[dict[str, Any]] = []
        for local_idx, (prefix, candidate, label_token) in enumerate(queries):
            q_idx = start_index + local_idx
            prefix_manifest = [str(tok - self.config.token_offset) for tok in prefix]
            candidate_manifest = [str(tok - self.config.token_offset) for tok in candidate]
            ranges = (
                _find_continuation_ranges(manifest_stream, prefix_manifest, candidate_manifest)
                if label_token == YES_TOKEN
                else []
            )
            questions.append(
                {
                    "question_index": q_idx,
                    "question_mode": "continuation",
                    "question_format": "binary_continuation",
                    "answer": "yes" if label_token == YES_TOKEN else "no",
                    "prefix": {"sequence": prefix_manifest},
                    "candidate": {
                        "sequence": candidate_manifest,
                        "present": label_token == YES_TOKEN,
                    },
                    "concerned_ranges": [{"start": start, "end": end} for start, end in ranges],
                }
            )
            rows.append(
                QuestionRow(
                    video_index=video_index,
                    question_index=q_idx,
                    bucket_id=bucket.name,
                    stream_length=len(stream),
                    concerned_ranges=ranges,
                    truth_kind="yes" if label_token == YES_TOKEN else "no",
                )
            )
        return questions, rows


def _find_subsequence_ranges(stream: Sequence[int], subseq: Sequence[int]) -> list[tuple[int, int]]:
    if not subseq or len(subseq) > len(stream):
        return []
    ranges: list[tuple[int, int]] = []
    length = len(subseq)
    limit = len(stream) - length + 1
    for start in range(max(0, limit)):
        if list(stream[start : start + length]) == list(subseq):
            ranges.append((start, start + length))
    return ranges


def _find_continuation_ranges(
    stream: Sequence[int],
    prefix: Sequence[int],
    continuation: Sequence[int],
) -> list[tuple[int, int]]:
    if not prefix or not continuation:
        return []
    prefix_len = len(prefix)
    cont_len = len(continuation)
    limit = len(stream) - (prefix_len + cont_len) + 1
    ranges: list[tuple[int, int]] = []
    for start in range(max(0, limit)):
        if list(stream[start : start + prefix_len]) != list(prefix):
            continue
        cont_start = start + prefix_len
        if list(stream[cont_start : cont_start + cont_len]) == list(continuation):
            ranges.append((cont_start, cont_start + cont_len))
    return ranges


def generate_manifest(config: ValGenerationConfig) -> ValGenerationResult:
    """Convenience wrapper that mirrors the legacy API."""

    return SyntheticValGenerator(config).build()
