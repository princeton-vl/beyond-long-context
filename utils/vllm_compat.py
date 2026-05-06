"""Compatibility helpers for third-party vLLM integration."""

from __future__ import annotations

import threading
from importlib import import_module
from typing import Any

_patch_lock = threading.Lock()
_disabled_tqdm_patched = False


def ensure_vllm_disabled_tqdm_patch() -> None:
    """Disable progress bars without triggering duplicate-kwarg errors.

    Hugging Face Hub >= 0.26 passes an explicit ``disable`` flag to the
    ``tqdm_class`` provided via :func:`snapshot_download`. Recent versions of
    vLLM override that class with ``DisabledTqdm`` but still forward their own
    ``disable=True`` argument, which now raises ``TypeError`` because ``disable``
    is specified twice. We wrap the vLLM-provided class so it drops the incoming
    argument before delegating to ``tqdm``.
    """

    global _disabled_tqdm_patched
    if _disabled_tqdm_patched:
        return

    with _patch_lock:
        if _disabled_tqdm_patched:
            return

        weight_utils = import_module(
            "vllm.model_executor.model_loader.weight_utils")

        base_tqdm = weight_utils.tqdm

        class _DisabledTqdm(base_tqdm):  # type: ignore[misc]
            __slots__ = ()

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.pop("disable", None)
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                self.disable = True

        _DisabledTqdm.__name__ = "DisabledTqdm"
        weight_utils.DisabledTqdm = _DisabledTqdm
        _disabled_tqdm_patched = True


__all__ = ["ensure_vllm_disabled_tqdm_patch"]
