"""Central utilities for frequently used filesystem locations."""
from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT_ENV = "STREAMING_SCRATCH_ROOT"
MODEL_CACHE_ENV = "STREAMING_MODEL_CACHE"
TEMP_ROOT_ENV = "STREAMING_TEMP_ROOT"

_DEFAULT_SCRATCH_ROOT = Path("/workspace-vast/shmublu/scratch")
_DEFAULT_MODEL_CACHE = Path("/workspace-vast/shmublu/scratch/cache")
_DEFAULT_TEMP_ROOT = _DEFAULT_MODEL_CACHE

_TEMP_ENV_VARS = ("TMPDIR", "TEMP", "TMP")


def _prime_temp_environment() -> None:
    """Ensure Python's tempfile stack points at the shared cache root."""

    configured = os.environ.get(TEMP_ROOT_ENV)
    base = Path(configured) if configured else _DEFAULT_TEMP_ROOT
    base.mkdir(parents=True, exist_ok=True)
    for env_name in _TEMP_ENV_VARS:
        os.environ.setdefault(env_name, str(base))


_prime_temp_environment()


def get_scratch_root() -> Path:
    """Return the configured scratch root as a Path."""
    return Path(os.environ.get(SCRATCH_ROOT_ENV, _DEFAULT_SCRATCH_ROOT))


def get_scratch_subdir(*parts: str) -> str:
    """Join `parts` under the scratch root and return string path."""
    return str(get_scratch_root().joinpath(*parts))


def get_model_cache_dir() -> str:
    """Return the configured Hugging Face cache directory."""
    configured = os.environ.get(MODEL_CACHE_ENV)
    cache_path = Path(configured) if configured else _DEFAULT_MODEL_CACHE
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)
