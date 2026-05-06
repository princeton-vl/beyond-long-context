"""Single entry point mapping CSV state_keys -> per-model FLOPs functions.

Intended use
------------
Other tooling in this repo (CSV generators, paper figures) refers to models
by short `state_key` strings (e.g. 'glm45v', 'internvl38bthinking'). This
module exposes `MODEL_FUNCTIONS`, a dict mapping each state_key to its
matching `flops_*` function from `flops_estimator`, plus a small
`compute_all(...)` helper for batch evaluation.

All functions take the same signature:
    fn(frames, n_in_text_tokens, n_out_text_tokens) -> dict
where `frames` is a list of {'height': int, 'width': int}. Each returned
dict contains at least a 'total' (or 'total_flops') key in raw FLOPs.

Thinking variants share architecture with their base counterparts -- the
runtime difference is only in `n_out_text_tokens` (the <think>...</think>
chain). We therefore map *_thinking state_keys to the same architecture
function (or to a thinking-named alias when the file already defines one).
"""

from __future__ import annotations

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


# state_key -> FLOPs function. State_keys are the canonical short names used
# in our paper/figure CSVs; thinking-variant keys point to the same arch.
MODEL_FUNCTIONS = {
    # Qwen family
    "qwen25video":          flops_qwen2_5_vl_7b,
    "qwen3dense":           flops_qwen3_vl_8b,
    "qwen3densethinking":   flops_qwen3_vl_8b_thinking,
    "qwen3omni":            flops_qwen3_omni_30b_a3b,
    # InternVL3.5 family
    "internvl8b":           flops_internvl3_5_8b,
    "internvl8bthinking":   flops_internvl3_5_8b_thinking,
    "internvl30ba3b":       flops_internvl3_5_30b_a3b,
    "internvl30ba3bthinking": flops_internvl3_5_30b_a3b_thinking,
    "internvl38b":          flops_internvl3_5_38b,
    "internvl38bthinking":  flops_internvl3_5_38b_thinking,
    # GLM-4.5V & MiniCPM family
    "glm45v":               flops_glm45v,
    "minicpm45":            flops_minicpmv45,
    "minicpmstream":        flops_minicpmv26,   # MiniCPM-V 2.6 (streaming).
    # LongVILA / MiMo / Phi
    "longvila":             flops_longvila,
    "mimovl":               flops_mimo_vl,
    "phi4mm":               flops_phi4_mm,
}


# Pretty display names (matching the canonical names we use in graphs/paper).
DISPLAY_NAMES = {
    "qwen25video":            "Qwen2.5-VL (7B)",
    "qwen3dense":             "Qwen3-VL (8B)",
    "qwen3densethinking":     "Qwen3-VL-Thinking (8B)",
    "qwen3omni":              "Qwen3-Omni (30B, A3B)",
    "internvl8b":             "InternVL3.5 V (8B)",
    "internvl8bthinking":     "InternVL3.5 V Thinking (8B)",
    "internvl30ba3b":         "InternVL3.5 V (30B, A3B)",
    "internvl30ba3bthinking": "InternVL3.5-30B-Thinking (30B, A3B)",
    "internvl38b":            "InternVL3.5 V (38B)",
    "internvl38bthinking":    "InternVL3.5 V Thinking (38B)",
    "glm45v":                 "GLM-4.5V (104B, A12B)",
    "minicpm45":              "MiniCPM-V 4.5 (9B)",
    "minicpmstream":          "MiniCPM V 2.6 (8B)",
    "longvila":               "LongVILA (7B)",
    "mimovl":                 "MIMO-VL (7B)",
    "phi4mm":                 "Phi-4-MM (6B)",
}


def _get_total(d: dict) -> float:
    """Some files return 'total', others 'total_flops'. Normalise here."""
    if "total_flops" in d:
        return float(d["total_flops"])
    if "total" in d:
        return float(d["total"])
    raise KeyError(f"FLOPs dict has no 'total' or 'total_flops' field: {d.keys()}")


def compute_all(frames, n_in_text_tokens, n_out_text_tokens) -> dict[str, dict]:
    """Run every registered model with the same input. Returns
    {state_key: full result dict from the function}."""
    out = {}
    for key, fn in MODEL_FUNCTIONS.items():
        out[key] = fn(frames, n_in_text_tokens, n_out_text_tokens)
    return out


if __name__ == "__main__":
    # Default test point (per task spec):
    frames = [{"height": 448, "width": 448}] * 8
    n_in = 128
    n_out = 64

    print(f"FLOPs at frames=8x(448x448), n_in={n_in}, n_out={n_out}")
    print("=" * 78)
    print(f"{'state_key':<24}  {'display name':<38}  {'total (PF)':>10}")
    print("-" * 78)
    results = compute_all(frames, n_in, n_out)
    rows = [(k, DISPLAY_NAMES[k], _get_total(r)) for k, r in results.items()]
    rows.sort(key=lambda r: r[2])
    for key, name, tot in rows:
        print(f"{key:<24}  {name:<38}  {tot/1e15:>10.4f}")
