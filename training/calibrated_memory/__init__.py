"""Core QA-Ego package."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_hf_cache() -> None:
    default_cache = Path(__file__).resolve().parents[1] / ".hf_cache"
    legacy_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(legacy_cache or default_cache)


_configure_hf_cache()
