"""flops_estimator — first-principles FLOPs estimators for 16 video-LLM models.

Each per-model function takes:
    frames:            list of {'height': int, 'width': int}
    n_in_text_tokens:  int (text prompt tokens)
    n_out_text_tokens: int (generated tokens)
and returns a dict with at least 'total' (or 'total_flops') in matmul-FLOPs.

Counting convention: matmul-only, 2*A*B*C per (A,B)·(B,C); softmax/norms/RoPE
are not counted. See per-file module docstrings for citations and the
per-function vision-encoder audits.
"""

from .flops_qwen import (
    flops_qwen2_5_vl_7b,
    flops_qwen3_vl_8b,
    flops_qwen3_vl_8b_thinking,
    flops_qwen3_omni_30b_a3b,
)
from .flops_internvl import (
    flops_internvl3_5_8b,
    flops_internvl3_5_8b_thinking,
    flops_internvl3_5_30b_a3b,
    flops_internvl3_5_30b_a3b_thinking,
    flops_internvl3_5_38b,
    flops_internvl3_5_38b_thinking,
)
from .flops_glm_minicpm import (
    flops_glm45v,
    flops_minicpmv45,
    flops_minicpmv26,
)
from .flops_longvila_mimo_phi import (
    flops_longvila,
    flops_mimo_vl,
    flops_phi4_mm,
)

__all__ = [
    # Qwen
    "flops_qwen2_5_vl_7b",
    "flops_qwen3_vl_8b",
    "flops_qwen3_vl_8b_thinking",
    "flops_qwen3_omni_30b_a3b",
    # InternVL3.5
    "flops_internvl3_5_8b",
    "flops_internvl3_5_8b_thinking",
    "flops_internvl3_5_30b_a3b",
    "flops_internvl3_5_30b_a3b_thinking",
    "flops_internvl3_5_38b",
    "flops_internvl3_5_38b_thinking",
    # GLM / MiniCPM
    "flops_glm45v",
    "flops_minicpmv45",
    "flops_minicpmv26",
    # LongVILA / MiMo / Phi
    "flops_longvila",
    "flops_mimo_vl",
    "flops_phi4_mm",
]
