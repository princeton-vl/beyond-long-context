"""Backend for the Log-Linear Mamba2 model (log-linear attention)."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import torch
from fla.models.log_linear_mamba2 import LogLinearMamba2Config, LogLinearMamba2Model
from fla.layers.log_linear_mamba2 import LogLinearMamba2

from .fla_backend import FlaSequenceBackend, _build_fla_config

import torch


logger = logging.getLogger(__name__)


_LOW_SHARED_MEMORY_STATE_SIZE = 64
_DEFAULT_STATE_SIZE = 128
_STATE_SIZE_SHARED_MEMORY_THRESHOLD = 120_000


def _force_safe_loglinear_forward(
    model: torch.nn.Module, training_dtype: str | None = None
) -> None:
    """Keep LogLinearMamba2 blocks on the unfused path to avoid Triton crashes.

    The fused training kernel occasionally raises ``cudaErrorIllegalAddress`` on
    long sequences with Triton 3.5.x. Flipping ``module.training`` to ``False``
    for the duration of each forward pass pushes the module down its stable
    eval-mode implementation while preserving gradients.
    """

    def _pre_hook(module: LogLinearMamba2, _inputs):
        if getattr(module, "_log_linear_forced_eval", False):
            return
        module._log_linear_forced_eval = True
        module._log_linear_prev_training = module.training
        module._log_linear_swapped_dtype = False
        module.training = False
        if training_dtype == "float32":
            first_param = next(module.parameters(), None)
            needs_swap = first_param is not None and first_param.dtype != torch.float32
            if needs_swap:
                module.to(dtype=torch.float32)
                module._log_linear_swapped_dtype = True

    def _post_hook(module: LogLinearMamba2, _inputs, output):
        if not getattr(module, "_log_linear_forced_eval", False):
            return output
        module.training = getattr(module, "_log_linear_prev_training", module.training)
        module._log_linear_forced_eval = False
        module._log_linear_prev_training = None
        if training_dtype == "float32" and getattr(module, "_log_linear_swapped_dtype", False):
            module.to(dtype=torch.bfloat16)
        module._log_linear_swapped_dtype = False
        return output

    for submodule in model.modules():
        if isinstance(submodule, LogLinearMamba2):
            submodule.register_forward_pre_hook(_pre_hook)
            submodule.register_forward_hook(_post_hook)


def _gather_shared_memory_limits() -> Sequence[int]:
    """Return the shared-memory limit for the active CUDA device."""

    if not torch.cuda.is_available():
        return ()

    def _device_limit(index: int) -> int:
        props = torch.cuda.get_device_properties(index)
        preferred = getattr(props, "shared_memory_per_block_optin", 0) or 0
        fallback = getattr(props, "shared_memory_per_block", 0) or 0
        return max(preferred, fallback)

    try:
        current = torch.cuda.current_device()
        return (_device_limit(current),)
    except Exception:  # pragma: no cover - conservative fallback
        limits: list[int] = []
        for index in range(torch.cuda.device_count()):
            limits.append(_device_limit(index))
        return tuple(limits)


def _resolve_state_size(
    desired_state_size: int,
    *,
    shared_memory_limits: Iterable[int] | None = None,
) -> tuple[int, str | None]:
    """Ensure the requested state size is supported by the active CUDA device."""

    limits = tuple(shared_memory_limits) if shared_memory_limits is not None else tuple(_gather_shared_memory_limits())
    if not limits:
        return desired_state_size, None
    smallest_limit = min(limits)
    if (
        desired_state_size > _LOW_SHARED_MEMORY_STATE_SIZE
        and smallest_limit < _STATE_SIZE_SHARED_MEMORY_THRESHOLD
    ):
        raise RuntimeError(
            "LogLinearMambaBackend requires GPUs with at least "
            f"{_STATE_SIZE_SHARED_MEMORY_THRESHOLD} bytes of shared memory per block "
            f"to run with state_size={desired_state_size}, but the active device only "
            f"exposes {smallest_limit}. Either lower --backend-option state_size or "
            "schedule the job on hardware with a larger shared-memory budget."
        )
    return desired_state_size, None


class LogLinearMambaBackend(FlaSequenceBackend):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        ctx_len: int = 256,
        allow_unstable_fused_kernel: bool = False,
        training_dtype: str | None = "float32",
        attn_mode: str | None = None,
        **config_overrides: Any,
    ) -> None:
        self.config_cls = LogLinearMamba2Config
        self.model_cls = LogLinearMamba2Model
        overrides = dict(config_overrides)
        resolved_attn_mode = overrides.pop("attn_mode", attn_mode)
        allow_unstable_fused_kernel = bool(allow_unstable_fused_kernel)
        desired_state_size = int(overrides.get("state_size", _DEFAULT_STATE_SIZE))
        chunk_size = int(overrides.get("chunk_size", 64))
        if chunk_size != 64:
            raise ValueError(
                "LogLinearMambaBackend only supports chunk_size=64 because the upstream"
                " chunked kernels gate on that constant."
            )
        overrides["chunk_size"] = chunk_size
        resolved_state_size, downgrade_notice = _resolve_state_size(desired_state_size)
        overrides["state_size"] = resolved_state_size
        if downgrade_notice:
            logger.warning(downgrade_notice)

        base_config = _build_fla_config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            ctx_len=ctx_len,
            overrides=overrides,
        )
        if resolved_attn_mode is not None:
            base_config["attn_mode"] = str(resolved_attn_mode)
        super().__init__(
            embed_dim=embed_dim,
            config_kwargs=base_config,
            ctx_len=ctx_len,
        )
        if training_dtype == "float32":
            self.encoder = self.encoder.to(dtype=torch.float32)

        n_groups = int(getattr(self.config, "n_groups", 1))
        if n_groups != 1:
            raise ValueError(
                "LogLinearMambaBackend currently requires n_groups=1 because the"
                " chunked log-linear kernels do not support grouped parameters on this build."
            )

        if not allow_unstable_fused_kernel:
            _force_safe_loglinear_forward(self.encoder, training_dtype=training_dtype)


__all__ = ["LogLinearMambaBackend"]
