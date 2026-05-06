
"""Utilities for configurable text embedding backends used by M3-Agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency at runtime
    openai = None

from transformers import AutoModel, AutoTokenizer

import torch.nn.functional as F


from utils.paths import get_model_cache_dir


CACHE_ROOT = get_model_cache_dir()

QWEN_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _default_device() -> str:
    """Return sensible default device string."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class EmbeddingResult:
    """Return value for embedding operations."""

    vectors: List[List[float]]
    provider: str


class TextEmbeddingBackend:
    """Configurable text embedding backend for OpenAI or local HF models."""

    def __init__(
        self,
        model_name: str,
        *,
        api_config: Optional[Dict[str, Dict[str, str]]] = None,
        target_dim: int = 1536,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.api_config = api_config or {}
        self.target_dim = target_dim
        self.device = self._resolve_device(device)
        self.provider = self._infer_provider(model_name)
        self.qwen_query_task = QWEN_QUERY_TASK

        # Lazy members for local embedding
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._local_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        # Pre-cache API client configuration
        self._api_key, self._api_base = self._resolve_api_credentials(model_name)

        if self.provider == "openai" and openai is None:
            raise RuntimeError(
                "openai package is required for OpenAI embedding backend but is not installed"
            )

    @staticmethod
    def _infer_provider(model_name: str) -> str:
        lowered = (model_name or "").lower()
        if not model_name:
            return "local"

        if lowered.startswith("text-embedding-") or lowered.startswith("gpt") or lowered.startswith("text-ada"):
            return "openai"
        if lowered.startswith("gemini"):
            return "openai"  # Treated as HTTP API-style backend
        if "qwen" in lowered:
            return "qwen"
        return "local"

    def _resolve_api_credentials(self, model_name: str) -> tuple[Optional[str], Optional[str]]:
        entry = self.api_config.get(model_name) or self.api_config.get(model_name.lower())
        api_key = None
        api_base = None
        if entry:
            api_key = entry.get("api_key") or None
            api_base = entry.get("base_url") or None

        # Environment variables take precedence for OpenAI-compatible APIs
        env_keys = ["OPENAI_LAB_KEY", "OPENAI_API_KEY"]
        for env in env_keys:
            if os.getenv(env):
                api_key = os.environ[env]
                break
        if os.getenv("OPENAI_BASE_URL"):
            api_base = os.environ["OPENAI_BASE_URL"]
        return api_key, api_base

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is None or device == "auto":
            return _default_device()
        return device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: Iterable[str], *, batch_size: int = 32, role: str = "document") -> EmbeddingResult:
        """Create embeddings for a batch of texts."""
        texts = ["" if text is None else str(text) for text in texts]
        if not texts:
            return EmbeddingResult([], self.provider)

        prepared_texts = self._prepare_texts(texts, role)

        if self.provider == "openai":
            vectors = self._embed_openai(prepared_texts, batch_size=batch_size)
        else:
            vectors = self._embed_local(prepared_texts, batch_size=batch_size)
        return EmbeddingResult(vectors, self.provider)

    def embed_text(self, text: str, *, role: str = "document") -> EmbeddingResult:
        """Create embedding for a single text."""
        return self.embed_texts([text], role=role)

    def embed_queries(self, queries: Iterable[str], *, batch_size: int = 32) -> EmbeddingResult:
        """Create embeddings for query texts using query-specific prompting."""
        return self.embed_texts(list(queries), batch_size=batch_size, role="query")

    def _prepare_texts(self, texts: Iterable[str], role: str) -> List[str]:
        """Prepare texts for embedding based on provider and role."""
        if role not in {"document", "query"}:
            raise ValueError(f"Unsupported text role: {role}")

        if self.provider == "qwen" and role == "query":
            return [self._format_qwen_query(text) for text in texts]
        return list(texts)

    def _format_qwen_query(self, query: str) -> str:
        """Format query using Qwen detailed instruction template."""
        return f"Instruct: {self.qwen_query_task}\nQuery:{query}"

    def token_lengths(self, texts: Iterable[str], *, role: str = "document") -> List[int]:
        """Return tokenizer token counts for the provided texts."""
        texts = ["" if text is None else str(text) for text in texts]
        if not texts:
            return []

        prepared = self._prepare_texts(texts, role)
        if self.provider != "qwen":
            return [0] * len(prepared)

        self._ensure_local_model()
        assert self._tokenizer is not None

        max_length = 8192
        encoded = self._tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = encoded.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            return [int(mask.sum().item()) for mask in attention_mask]

        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return [int(seq.shape[-1]) for seq in input_ids]
        if isinstance(input_ids, list):
            return [len(seq) for seq in input_ids]
        return [0] * len(prepared)

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states using the last non-padding token (Qwen recommendation)."""
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]
    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------
    def _embed_openai(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if not self._api_key:
            print("WARNING: No OpenAI API key configured; returning zero embeddings")
            return [[0.0] * self.target_dim for _ in texts]

        client_kwargs: Dict[str, str] = {"api_key": self._api_key}
        if self._api_base:
            client_kwargs["base_url"] = self._api_base
        client = openai.OpenAI(**client_kwargs)

        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    dimensions=self.target_dim,
                )
                vectors.extend(item.embedding for item in response.data)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"OpenAI embedding request failed: {exc}")
                vectors.extend([[0.0] * self.target_dim for _ in batch])
        return vectors

    # ------------------------------------------------------------------
    # Local HF backend
    # ------------------------------------------------------------------
    def _ensure_local_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        trust_remote = "qwen" in (self.model_name or "").lower()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote,
            cache_dir=CACHE_ROOT,
        )
        if self.provider == "qwen":
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote,
            cache_dir=CACHE_ROOT,
            torch_dtype=self._local_dtype,
        )
        self._model.to(self.device)
        self._model.eval()

    def _embed_local(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        self._ensure_local_model()
        assert self._tokenizer is not None and self._model is not None

        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_length = 8192 if self.provider == "qwen" else 1024
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, (list, tuple)):
                    hidden_states = outputs[0]
                else:  # pragma: no cover - defensive
                    raise RuntimeError("Unexpected model output format for embeddings")

                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    raise RuntimeError("attention_mask missing from tokenizer outputs")

                if self.provider == "qwen":
                    pooled = self._last_token_pool(hidden_states, attention_mask)
                else:
                    mask = attention_mask.unsqueeze(-1)
                    masked_hidden = hidden_states * mask
                    summed = masked_hidden.sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    pooled = summed / counts

                pooled = pooled.to(torch.float32)

                if pooled.shape[1] > self.target_dim:
                    pooled = pooled[:, : self.target_dim]
                elif pooled.shape[1] < self.target_dim:
                    pad_width = self.target_dim - pooled.shape[1]
                    pooled = F.pad(pooled, (0, pad_width))

                pooled = F.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors


def load_api_config(config_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive load handling
        print(f"Warning: failed to load API config {config_path}: {exc}")
        return {}
