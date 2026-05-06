"""Runtime patches that mitigate known Qwen2.5-VL memory issues."""

from __future__ import annotations

import logging
from typing import Callable

import torch


LOGGER = logging.getLogger(__name__)


def apply_tensor_memory_safe_operations() -> None:
    """Wrap ``torch.Tensor.__iadd__`` to avoid in-place overlap issues."""

    original_iadd: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.Tensor.__iadd__

    def safe_iadd(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        try:
            return original_iadd(self, other)
        except RuntimeError as exc:  # pragma: no cover - guard rails for rare backend bug
            message = str(exc)
            if "more than one element of the written-to tensor refers to a single memory location" not in message:
                raise

            result = self + other
            self.data = result.data
            LOGGER.warning("Converted in-place add to out-of-place to avoid overlapping tensor write.")
            return self

    if getattr(torch.Tensor.__iadd__, "__patched_by_streaming_memory__", False):
        LOGGER.debug("Tensor __iadd__ already patched; skipping reapplication.")
        return

    safe_iadd.__patched_by_streaming_memory__ = True  # type: ignore[attr-defined]
    torch.Tensor.__iadd__ = safe_iadd  # type: ignore[assignment]
    LOGGER.info("Tensor memory-safe in-place addition patch applied.")


def patch_qwen2_5_vl_memory_overlap() -> None:
    """Patch Qwen2.5-VL forward passes to defend against cache overlap and FA2 issues."""

    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # pylint: disable=import-error
            Qwen2_5_VLAttention,
            Qwen2_5_VLModel,
        )
    except ImportError as exc:  # pragma: no cover - import depends on optional package
        LOGGER.warning("Could not import Qwen2.5-VL modules: %s", exc)
        LOGGER.warning("Memory patches will apply once transformers loads the model class.")
        return

    original_forward = Qwen2_5_VLModel.forward
    original_attn_forward = Qwen2_5_VLAttention.forward

    def patched_forward(self, *args, **kwargs):  # type: ignore[override]
        try:
            return original_forward(self, *args, **kwargs)
        except RuntimeError as exc:
            message = str(exc)
            if "more than one element of the written-to tensor refers to a single memory location" in message:
                cache_position = kwargs.get("cache_position")
                if cache_position is not None:
                    kwargs["cache_position"] = cache_position.clone().detach()
                    LOGGER.warning("Cloned cache_position to avoid overlapping memory during forward pass.")
                return original_forward(self, *args, **kwargs)
            if "numel() == 0" in message and "max()" in message:
                LOGGER.warning("Falling back to eager attention after flash-attention empty tensor error.")
                previous_impl = getattr(self, "_attn_implementation", None)
                setattr(self, "_attn_implementation", "eager")
                try:
                    return original_forward(self, *args, **kwargs)
                finally:
                    if previous_impl is not None:
                        setattr(self, "_attn_implementation", previous_impl)
            raise

    def patched_attn_forward(self, *args, **kwargs):  # type: ignore[override]
        try:
            return original_attn_forward(self, *args, **kwargs)
        except RuntimeError as exc:
            message = str(exc)
            if "numel() == 0" in message and "max()" in message:
                LOGGER.warning("Switching Qwen2.5-VL attention to eager mode for this call.")
                previous_impl = getattr(self.config, "_attn_implementation", None)
                self.config._attn_implementation = "eager"
                try:
                    return original_attn_forward(self, *args, **kwargs)
                finally:
                    if previous_impl is not None:
                        self.config._attn_implementation = previous_impl
            raise

    Qwen2_5_VLModel.forward = patched_forward  # type: ignore[assignment]
    Qwen2_5_VLAttention.forward = patched_attn_forward  # type: ignore[assignment]
    LOGGER.info("Applied Qwen2.5-VL memory overlap and flash-attention guards.")


def apply_all_patches() -> None:
    """Apply every available runtime patch."""

    apply_tensor_memory_safe_operations()
    patch_qwen2_5_vl_memory_overlap()


if __name__ == "__main__":  # pragma: no cover
    apply_all_patches()
