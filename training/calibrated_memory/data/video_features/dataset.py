from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset

from calibrated_memory.data.sequences.common import (
    CANDIDATE_SEPARATOR,
    IGNORE_INDEX,
    LABEL_SEPARATOR,
    NO_TOKEN,
    QUERY_END_SEPARATOR,
    STREAM_QUERY_SEPARATOR,
    TOKEN_OFFSET,
    UNCERTAIN_TOKEN,
    YES_TOKEN,
)
from calibrated_memory.data.sequences.metadata_utils import build_video_metadata, resolve_entropy_value


def _mean(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    total = sum(float(v) for v in values)
    return total / len(values)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    if lower == upper:
        return float(sorted_values[lower])
    lower_val = float(sorted_values[lower])
    upper_val = float(sorted_values[upper])
    weight = position - lower
    return lower_val + (upper_val - lower_val) * weight


def _compute_tertiles(values: list[float]) -> list[float] | None:
    data = sorted(float(v) for v in values if v is not None)
    if len(data) < 3:
        return None
    return [_quantile(data, 1.0 / 3.0), _quantile(data, 2.0 / 3.0)]


def _question_mode(question: dict[str, Any]) -> str:
    return str(question.get("question_mode") or "exists").lower()


def _resolve_video_entropy(video: dict[str, Any]) -> float | None:
    entropy_payload = video.get("entropy_overall")
    if entropy_payload is None:
        return None
    if isinstance(entropy_payload, (int, float)):
        return float(entropy_payload)
    sequence_keys = list(video.get("sequence_keys") or [])
    if not sequence_keys:
        sequences_used = video.get("sequences_used") or {}
        if isinstance(sequences_used, dict):
            sequence_keys = [key for key in sequences_used if key]
    if not sequence_keys:
        fallback_key = video.get("sequence_key")
        if fallback_key:
            sequence_keys = [fallback_key]
    for key in sequence_keys:
        resolved = resolve_entropy_value(entropy_payload, key)
        if resolved is not None:
            return resolved
    if isinstance(entropy_payload, dict):
        for value in entropy_payload.values():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _resolve_answer_token(question: dict[str, Any]) -> int:
    answer = str(question.get("answer") or "").lower()
    if answer == "yes":
        return YES_TOKEN
    if answer == "no":
        return NO_TOKEN
    if answer == "uncertain":
        return UNCERTAIN_TOKEN
    raise ValueError(f"Unsupported answer '{question.get('answer')}' in embedding manifest")


def _question_entropy_value(question: dict[str, Any]) -> float | None:
    payload = question.get("entropy_prefix")
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


@dataclass(frozen=True)
class _VideoEntry:
    stream_file: Path
    stream_key: str
    timestamp_key: str
    frame_count: int
    questions: list[dict[str, Any]]
    question_metadata: list[dict[str, Any]]


class VideoFeatureDataset(Dataset):
    """Dataset that consumes precomputed frame embeddings for binary tasks."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        max_videos: int = -1,
        task: str = "membership",
        cont_len: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        normalized_task = str(task).lower()
        if normalized_task not in {"membership", "continuation"}:
            raise ValueError("VideoFeatureDataset only supports membership or continuation tasks")
        self.task = normalized_task
        self.cont_len = int(cont_len or 0)
        if self.task == "continuation" and self.cont_len <= 0:
            raise ValueError("Continuation tasks require cont_len > 0")
        manifest = json.loads(Path(manifest_path).read_text())
        videos_raw = list(manifest.get("videos", []))
        if max_videos > 0:
            videos_raw = videos_raw[:max_videos]
        if not videos_raw:
            raise ValueError("Embedding manifest contained no videos")
        backbone_info = manifest.get("backbone", {})
        self.embed_dim = int(backbone_info.get("embed_dim", 0))
        if self.embed_dim <= 0:
            raise ValueError("Embed dim missing from manifest")
        entries: list[_VideoEntry] = []
        video_metadata: list[dict[str, Any]] = []
        sample_metadata: list[dict[str, Any]] = []
        entropy_values: list[float] = []
        candidate_lengths: list[float] = []
        video_length_values: list[float] = []
        video_entropy_values: list[float] = []
        total_questions: list[int] = []
        max_input_len = 0
        length_limit = int(max_seq_len) if max_seq_len and max_seq_len > 0 else 0
        skipped_over_limit = 0
        for video in videos_raw:
            stream_info = video.get("stream_embeddings") or {}
            stream_path = Path(stream_info.get("file", "")).expanduser()
            if not stream_path.exists():
                raise FileNotFoundError(f"Stream embedding file missing: {stream_path}")
            stream_key = str(stream_info.get("embeddings_key", "stream_embeddings"))
            timestamp_key = str(stream_info.get("timestamps_key", "stream_timestamps"))
            frame_count = int(stream_info.get("frame_count", 0))
            question_payload = self._select_questions(video)
            prepared_questions: list[dict[str, Any]] = []
            prepared_metadata: list[dict[str, Any]] = []
            entropy_samples: list[float] = []
            token_budget = frame_count
            for question in question_payload:
                candidate = question.get("candidate") or {}
                candidate_meta = candidate.get("embedding") or {}
                candidate_len = int(candidate_meta.get("count", 0))
                if candidate_len <= 0:
                    continue
                answer_token = _resolve_answer_token(question)
                if answer_token not in (YES_TOKEN, NO_TOKEN, UNCERTAIN_TOKEN):
                    continue
                question_meta = self._extract_question_metadata(question, candidate_len)
                prepared_questions.append(question)
                prepared_metadata.append(question_meta)
                entropy_value = question_meta.get("entropy_prefix")
                if entropy_value is not None:
                    entropy_samples.append(float(entropy_value))
                candidate_lengths.append(float(candidate_len))
                token_budget += self._query_token_cost(candidate_len)
            if not prepared_questions:
                continue
            if length_limit > 0 and token_budget > length_limit:
                skipped_over_limit += 1
                continue
            entry = _VideoEntry(
                stream_file=stream_path,
                stream_key=stream_key,
                timestamp_key=timestamp_key,
                frame_count=frame_count,
                questions=prepared_questions,
                question_metadata=prepared_metadata,
            )
            entries.append(entry)
            max_input_len = max(max_input_len, token_budget)
            entropy_value = _resolve_video_entropy(video)
            if entropy_value is not None:
                video_entropy_values.append(float(entropy_value))
            video_meta_payload = build_video_metadata(video, stream_length=frame_count)
            if not video_meta_payload:
                video_meta_payload = {
                    "stream_length": int(frame_count),
                    "length_value": float(frame_count),
                    "video_index": video.get("video_index"),
                }
                if entropy_value is not None:
                    video_meta_payload["entropy_value"] = entropy_value
            video_metadata.append(dict(video_meta_payload))
            sample_metadata.append(dict(video_meta_payload))
            entropy_values.extend(entropy_samples)
            video_length_values.append(float(frame_count))
            total_questions.append(len(prepared_questions))
        if not entries:
            raise ValueError("No videos satisfied the configured filters for VideoFeatureDataset")
        self.entries = entries
        self.video_metadata = video_metadata
        self.sample_metadata = sample_metadata
        self.max_input_len = max_input_len
        self.pad_id = TOKEN_OFFSET + 1
        self.vocab_size = self.pad_id + 1
        self.metadata_summary = {
            "source": "video_features",
            "video_count": len(entries),
            "task": self.task,
            "cont_len": self.cont_len if self.task == "continuation" else None,
            "avg_questions_per_video": _mean(total_questions),
            "target_length_tertiles": _compute_tertiles(candidate_lengths),
            "stream_length_tertiles": _compute_tertiles(video_length_values),
            "entropy_prefix_tertiles": _compute_tertiles(entropy_values),
            "video_entropy_tertiles": _compute_tertiles(video_entropy_values),
            "skipped_videos_over_limit": skipped_over_limit,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        with safe_open(entry.stream_file, framework="pt") as handle:
            stream_embeddings = handle.get_tensor(entry.stream_key).to(torch.float32)
            timestamps = handle.get_tensor(entry.timestamp_key).to(torch.float32)
            tensor_cache: Dict[str, torch.Tensor] = {}

            def _load_candidate(candidate: dict[str, Any]) -> torch.Tensor:
                info = candidate.get("embedding") or {}
                key = info.get("key")
                if not key:
                    return torch.zeros(0, self.embed_dim, dtype=torch.float32, device=stream_embeddings.device)
                cached = tensor_cache.get(key)
                if cached is None:
                    tensor_cache[key] = handle.get_tensor(key).to(torch.float32)
                    cached = tensor_cache[key]
                return cached.to(stream_embeddings.device)

            if self.task == "continuation":
                (
                    query_ids,
                    query_labels,
                    query_embeddings,
                    query_embedding_mask,
                ) = self._build_continuation_queries(
                    entry.questions,
                    _load_candidate,
                    stream_embeddings,
                )
            else:
                (
                    query_ids,
                    query_labels,
                    query_embeddings,
                    query_embedding_mask,
                ) = self._build_membership_queries(
                    entry.questions,
                    _load_candidate,
                    stream_embeddings.device,
                )
        stream_len = stream_embeddings.size(0)
        stream_ids = torch.zeros(stream_len, dtype=torch.long)
        stream_labels = torch.full((stream_len,), IGNORE_INDEX, dtype=torch.long)
        full_ids = torch.cat([stream_ids, query_ids])
        full_labels = torch.cat([stream_labels, query_labels])
        extras = {
            "stream_embeddings": stream_embeddings,
            "query_embeddings": query_embeddings,
            "query_embedding_mask": query_embedding_mask,
            "metadata": {
                "video": dict(self.video_metadata[idx]),
                "queries": [dict(meta) for meta in entry.question_metadata],
            },
        }
        return full_ids, full_labels, torch.tensor(stream_len, dtype=torch.long), extras

    def _select_questions(self, video: dict[str, Any]) -> list[dict[str, Any]]:
        questions = []
        for question in video.get("questions", []) or []:
            mode = _question_mode(question)
            if self.task == "continuation" and mode == "continuation":
                questions.append(question)
            elif self.task == "membership" and mode == "exists":
                questions.append(question)
        return questions

    def _build_membership_queries(
        self,
        questions: List[dict[str, Any]],
        loader: Callable[[dict[str, Any]], torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids: list[int] = []
        labels: list[int] = []
        embeds: list[torch.Tensor] = []
        mask: list[bool] = []
        zero_vec = torch.zeros(self.embed_dim, dtype=torch.float32, device=device)
        content_token = TOKEN_OFFSET
        inserted = 0
        for question in questions:
            candidate = loader(question.get("candidate") or {})
            if candidate.dim() == 1:
                candidate = candidate.unsqueeze(0)
            if candidate.size(0) == 0:
                continue
            label_token = _resolve_answer_token(question)
            ids.append(STREAM_QUERY_SEPARATOR)
            labels.append(IGNORE_INDEX)
            embeds.append(zero_vec)
            mask.append(False)
            for row in candidate:
                ids.append(content_token)
                labels.append(IGNORE_INDEX)
                embeds.append(row)
                mask.append(True)
            ids.append(LABEL_SEPARATOR)
            labels.append(label_token)
            embeds.append(zero_vec)
            mask.append(False)

            ids.append(QUERY_END_SEPARATOR)
            labels.append(IGNORE_INDEX)
            embeds.append(zero_vec)
            mask.append(False)
            inserted += 1
        if inserted == 0:
            raise ValueError("Failed to build membership queries from embeddings")
        query_ids = torch.tensor(ids, dtype=torch.long)
        query_labels = torch.tensor(labels, dtype=torch.long)
        query_embeddings = torch.stack(embeds, dim=0)
        query_embedding_mask = torch.tensor(mask, dtype=torch.bool, device=device)
        return query_ids, query_labels, query_embeddings, query_embedding_mask

    def _build_continuation_queries(
        self,
        questions: List[dict[str, Any]],
        loader: Callable[[dict[str, Any]], torch.Tensor],
        stream_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cont_len <= 0:
            raise ValueError("Continuation queries require cont_len > 0")
        ids: list[int] = []
        labels: list[int] = []
        embeds: list[torch.Tensor] = []
        mask: list[bool] = []
        zero_vec = torch.zeros(self.embed_dim, dtype=torch.float32, device=stream_embeddings.device)
        content_token = TOKEN_OFFSET
        inserted = 0
        for question in questions:
            prefix = self._extract_prefix(stream_embeddings, question)
            if prefix is None:
                continue
            candidate = loader(question.get("candidate") or {})
            if candidate.dim() == 1:
                candidate = candidate.unsqueeze(0)
            if candidate.size(0) == 0:
                continue
            label_token = _resolve_answer_token(question)
            ids.append(STREAM_QUERY_SEPARATOR)
            labels.append(IGNORE_INDEX)
            embeds.append(zero_vec)
            mask.append(False)
            for row in prefix:
                ids.append(content_token)
                labels.append(IGNORE_INDEX)
                embeds.append(row)
                mask.append(True)
            ids.append(CANDIDATE_SEPARATOR)
            labels.append(IGNORE_INDEX)
            embeds.append(zero_vec)
            mask.append(False)
            for row in candidate:
                ids.append(content_token)
                labels.append(IGNORE_INDEX)
                embeds.append(row)
                mask.append(True)
            ids.append(LABEL_SEPARATOR)
            labels.append(label_token)
            embeds.append(zero_vec)
            mask.append(False)

            ids.append(QUERY_END_SEPARATOR)
            labels.append(IGNORE_INDEX)
            embeds.append(zero_vec)
            mask.append(False)
            inserted += 1
        if inserted == 0:
            raise ValueError("Failed to build continuation queries from embeddings")
        query_ids = torch.tensor(ids, dtype=torch.long)
        query_labels = torch.tensor(labels, dtype=torch.long)
        query_embeddings = torch.stack(embeds, dim=0)
        query_embedding_mask = torch.tensor(mask, dtype=torch.bool, device=stream_embeddings.device)
        return query_ids, query_labels, query_embeddings, query_embedding_mask

    def _extract_prefix(
        self,
        stream_embeddings: torch.Tensor,
        question: dict[str, Any],
    ) -> list[torch.Tensor] | None:
        cutoff = int(question.get("stream_cutoff", stream_embeddings.size(0)))
        end = min(cutoff, stream_embeddings.size(0))
        start = max(0, end - self.cont_len)
        if end - start < self.cont_len:
            return None
        prefix = stream_embeddings[start:end]
        return [row.clone() for row in prefix]

    def _extract_question_metadata(
        self,
        question: dict[str, Any],
        candidate_len: int,
    ) -> dict[str, Any]:
        entropy_value = _question_entropy_value(question)
        metadata = {
            "question_index": question.get("question_index"),
            "entropy_prefix": entropy_value,
            "target_length": float(candidate_len),
            "question_mode": _question_mode(question),
            "answer": question.get("answer"),
        }
        if self.task == "continuation":
            metadata["prefix_length"] = float(self.cont_len)
        return metadata

    def _query_token_cost(self, candidate_len: int) -> int:
        if self.task == "continuation":
            return 3 + self.cont_len + candidate_len
        return 2 + candidate_len
