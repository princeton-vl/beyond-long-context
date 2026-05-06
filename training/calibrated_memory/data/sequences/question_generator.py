"""Utility builders for binary membership and continuation queries."""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Sequence

import torch

from .common import (
    CANDIDATE_SEPARATOR,
    IGNORE_INDEX,
    LABEL_SEPARATOR,
    NO_TOKEN,
    QUERY_END_SEPARATOR,
    STREAM_QUERY_SEPARATOR,
    TOKEN_OFFSET,
    YES_TOKEN,
)


def _ensure_generator(generator: torch.Generator | None) -> torch.Generator:
    gen = generator or torch.Generator()
    return gen


def _coin_flip(generator: torch.Generator) -> bool:
    return bool(int(torch.randint(0, 2, (1,), generator=generator).item()))


def _sample_span(
    tokens: Sequence[int],
    *,
    min_len: int,
    max_len: int,
    tail_room: int = 0,
    generator: torch.Generator,
) -> tuple[int, int]:
    """Return ``(start, length)`` ensuring ``tail_room`` tokens remain after the slice."""

    stream_len = len(tokens)
    if min_len <= 0 or max_len < min_len:
        raise ValueError("min_len must be > 0 and <= max_len")
    if stream_len < min_len + tail_room:
        raise RuntimeError("Stream is too short to satisfy the requested span length")

    max_start = stream_len - (min_len + tail_room)
    attempts = 0
    max_attempts = max(32, stream_len * 2)
    while attempts < max_attempts:
        attempts += 1
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        available = stream_len - start - tail_room
        allowed_max = min(max_len, available)
        if allowed_max < min_len:
            continue
        if allowed_max == min_len:
            length = min_len
        else:
            length = int(torch.randint(min_len, allowed_max + 1, (1,), generator=generator).item())
        if length > 0:
            return start, length
    raise RuntimeError("Failed to sample a valid span within the allotted attempts")


def _has_subsequence(tokens: Sequence[int], subseq: Sequence[int]) -> bool:
    length = len(subseq)
    if length == 0 or length > len(tokens):
        return False
    limit = len(tokens) - length + 1
    for start in range(limit):
        if tokens[start : start + length] == list(subseq):
            return True
    return False


def _resolve_vocab_bounds(
    tokens: Sequence[int],
    vocab_size: int | None,
    token_offset: int,
) -> tuple[int, int]:
    base = max(token_offset, TOKEN_OFFSET)
    if vocab_size is not None and vocab_size > 0:
        return base, base + vocab_size
    stream_min = min(tokens) if tokens else base
    stream_max = max(tokens) if tokens else stream_min
    high = max(stream_max + 1, stream_min + 1)
    return min(stream_min, base), max(high, base + 1)


def _sample_absent_slice(
    tokens: Sequence[int],
    *,
    length: int,
    vocab_low: int,
    vocab_high: int,
    generator: torch.Generator,
) -> list[int]:
    attempts = 0
    max_attempts = 256
    while attempts < max_attempts:
        attempts += 1
        candidate = torch.randint(
            vocab_low,
            vocab_high,
            (length,),
            generator=generator,
        ).tolist()
        if not _has_subsequence(tokens, candidate):
            return candidate
    raise RuntimeError("Unable to sample a negative membership slice absent from the stream")


def _collect_followers_for_prefix(
    tokens: Sequence[int],
    prefix: Sequence[int],
    *,
    cont_len: int,
) -> set[tuple[int, ...]]:
    followers: set[tuple[int, ...]] = set()
    prefix_len = len(prefix)
    limit = len(tokens) - (prefix_len + cont_len) + 1
    if prefix_len <= 0 or cont_len <= 0 or limit <= 0:
        return followers
    for start in range(limit):
        if tokens[start : start + prefix_len] == list(prefix):
            follower = tuple(tokens[start + prefix_len : start + prefix_len + cont_len])
            if len(follower) == cont_len:
                followers.add(follower)
    return followers


def _sample_absent_continuation(
    followers: set[tuple[int, ...]],
    *,
    length: int,
    vocab_low: int,
    vocab_high: int,
    generator: torch.Generator,
) -> list[int]:
    attempts = 0
    max_attempts = 256
    while attempts < max_attempts:
        attempts += 1
        candidate = torch.randint(
            vocab_low,
            vocab_high,
            (length,),
            generator=generator,
        ).tolist()
        if tuple(candidate) not in followers:
            return candidate
    raise RuntimeError("Unable to sample a continuation negative distinct from observed followers")


def _draw_membership_queries(
    tokens: Sequence[int],
    *,
    count: int,
    min_len: int,
    max_len: int,
    generator: torch.Generator,
    vocab_size: int | None,
    token_offset: int,
) -> list[tuple[list[int], bool]]:
    if count <= 0:
        return []
    vocab_low, vocab_high = _resolve_vocab_bounds(tokens, vocab_size, token_offset)
    queries: list[tuple[list[int], bool]] = []
    for _ in range(count):
        start, length = _sample_span(
            tokens,
            min_len=min_len,
            max_len=max_len,
            generator=generator,
        )
        subseq = list(tokens[start : start + length])
        is_present = _coin_flip(generator)
        if not is_present:
            subseq = _sample_absent_slice(
                tokens,
                length=length,
                vocab_low=vocab_low,
                vocab_high=vocab_high,
                generator=generator,
            )
        queries.append((subseq, is_present))
    return queries


def _draw_continuation_queries(
    tokens: Sequence[int],
    *,
    count: int,
    min_len: int,
    max_len: int,
    cont_len: int,
    generator: torch.Generator,
    vocab_size: int | None,
    token_offset: int,
) -> list[tuple[list[int], list[int], int]]:
    if count <= 0:
        return []
    if cont_len <= 0:
        raise ValueError("cont_len must be positive")
    vocab_low, vocab_high = _resolve_vocab_bounds(tokens, vocab_size, token_offset)
    queries: list[tuple[list[int], list[int], int]] = []
    for _ in range(count):
        start, prefix_len = _sample_span(
            tokens,
            min_len=min_len,
            max_len=max_len,
            tail_room=cont_len,
            generator=generator,
        )
        prefix = list(tokens[start : start + prefix_len])
        follower_start = start + prefix_len
        true_candidate = list(tokens[follower_start : follower_start + cont_len])
        followers = _collect_followers_for_prefix(
            tokens,
            prefix,
            cont_len=cont_len,
        )
        if not followers:
            raise RuntimeError("Failed to locate a continuation for the sampled prefix")
        is_present = _coin_flip(generator)
        candidate = (
            true_candidate
            if is_present
            else _sample_absent_continuation(
                followers,
                length=cont_len,
                vocab_low=vocab_low,
                vocab_high=vocab_high,
                generator=generator,
            )
        )
        label_token = YES_TOKEN if is_present else NO_TOKEN
        queries.append((prefix, candidate, label_token))
    return queries


def build_samples(
    streams: Iterable[Sequence[int]],
    query_fn: Callable[[Sequence[int]], list],
    task: Literal["membership", "continuation"] = "membership",
    cont_len: int = 3,
    vocab_size: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int, int, int]:
    """Construct (input, labels, stream_len) triples for the supported tasks."""

    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    vocab_max = -1
    max_input_len = 0

    for tokens in streams:
        if not tokens or len(tokens) < 2:
            continue

        if vocab_size is not None:
            vocab_max = max(vocab_max, TOKEN_OFFSET + vocab_size - 1)
        else:
            vocab_max = max(vocab_max, max(tokens))
        vocab_max = max(
            vocab_max,
            STREAM_QUERY_SEPARATOR,
            LABEL_SEPARATOR,
            QUERY_END_SEPARATOR,
            CANDIDATE_SEPARATOR,
            YES_TOKEN,
            NO_TOKEN,
        )
        queries = query_fn(tokens)
        if not queries:
            continue

        stream_len = len(tokens)
        input_tokens = list(tokens)
        labels = [IGNORE_INDEX] * len(tokens)

        for q in queries:
            if task == "membership":
                subseq, is_present = q  # (list[int], bool)
                if subseq:
                    vocab_max = max(vocab_max, max(subseq))

                input_tokens.append(STREAM_QUERY_SEPARATOR)
                labels.append(IGNORE_INDEX)

                input_tokens.extend(subseq)
                labels.extend([IGNORE_INDEX] * len(subseq))

                label_token = YES_TOKEN if is_present else NO_TOKEN

                input_tokens.append(LABEL_SEPARATOR)
                labels.append(label_token)

                input_tokens.append(QUERY_END_SEPARATOR)
                labels.append(IGNORE_INDEX)

            elif task == "continuation":
                prefix, candidate, label_token = q
                if prefix:
                    vocab_max = max(vocab_max, max(prefix))
                if candidate:
                    vocab_max = max(vocab_max, max(candidate))
                vocab_max = max(vocab_max, label_token)

                input_tokens.append(STREAM_QUERY_SEPARATOR)
                labels.append(IGNORE_INDEX)

                input_tokens.extend(prefix)
                labels.extend([IGNORE_INDEX] * len(prefix))

                input_tokens.append(CANDIDATE_SEPARATOR)
                labels.append(IGNORE_INDEX)

                input_tokens.extend(candidate)
                labels.extend([IGNORE_INDEX] * len(candidate))

                input_tokens.append(LABEL_SEPARATOR)
                labels.append(label_token)

                input_tokens.append(QUERY_END_SEPARATOR)
                labels.append(IGNORE_INDEX)

            else:
                raise ValueError(f"Unknown task={task}")

        seq = torch.tensor(input_tokens, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        stream_len_tensor = torch.tensor(stream_len, dtype=torch.long)
        samples.append((seq, label_tensor, stream_len_tensor))
        max_input_len = max(max_input_len, seq.numel())

    if not samples:
        raise RuntimeError("Failed to build any examples.")

    pad_id = vocab_max + 1
    vocab_total = pad_id + 1
    return samples, pad_id, vocab_total, max_input_len


def build_continuation_samples(
    streams: Iterable[Sequence[int]],
    unique_sequences: int,
    *,
    cont_len: int = 3,
    min_len: int = 3,
    max_len: int = 7,
    generator: torch.Generator | None = None,
    vocab_size: int | None = None,
    token_offset: int = TOKEN_OFFSET,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int, int, int]:
    """Build binary continuation queries using prefix/candidate pairs."""

    rng = _ensure_generator(generator)

    def query_fn(tokens: Sequence[int]) -> list[tuple[list[int], list[int], int]]:
        if len(tokens) < min_len + cont_len:
            raise RuntimeError(
                f"Stream of length {len(tokens)} cannot host continuation spans of length {min_len}+{cont_len}"
            )
        return _draw_continuation_queries(
            tokens,
            count=unique_sequences,
            min_len=min_len,
            max_len=max_len,
            cont_len=cont_len,
            generator=rng,
            vocab_size=vocab_size,
            token_offset=token_offset,
        )

    return build_samples(streams, query_fn, task="continuation", cont_len=cont_len, vocab_size=vocab_size)


def build_membership_samples(
    streams: Iterable[Sequence[int]],
    unique_sequences: int,
    *,
    min_len: int = 3,
    max_len: int = 7,
    cont_len: int = 3,
    generator: torch.Generator | None = None,
    vocab_size: int | None = None,
    token_offset: int = TOKEN_OFFSET,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int, int, int]:
    """Build yes/no membership datasets with independent 50/50 labels."""
    rng = _ensure_generator(generator)

    def query_fn(tokens: Sequence[int]) -> list[tuple[list[int], bool]]:
        if len(tokens) < min_len:
            raise RuntimeError(
                f"Stream of length {len(tokens)} cannot host membership spans with min_len={min_len}"
            )
        return _draw_membership_queries(
            tokens,
            count=unique_sequences,
            min_len=min_len,
            max_len=max_len,
            generator=rng,
            vocab_size=vocab_size,
            token_offset=token_offset,
        )

    return build_samples(streams, query_fn, task="membership", vocab_size=vocab_size)
