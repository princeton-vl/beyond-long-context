"""Validation runner: each registered model evaluated at two test points.

Test cases:
  A) 8 frames @ 448x448, n_in_text=128, n_out_text=64
  B) 32 frames @ 448x448, n_in_text=128, n_out_text=64

Prints a single comparison table with the matmul-FLOPs total for each.

Run:
    python -m flops_estimator.validate
"""

from __future__ import annotations

from .flops_all_models import MODEL_FUNCTIONS, DISPLAY_NAMES, _get_total


def _component(d: dict, *keys: str) -> float:
    for k in keys:
        if k in d:
            return float(d[k])
    return 0.0


def _row(key: str, fn, frames, n_in: int, n_out: int) -> tuple:
    r = fn(frames, n_in, n_out)
    return (
        key,
        _component(r, "vision", "vision_flops"),
        _component(r, "connector", "connector_flops"),
        _component(r, "llm_prefill", "llm_prefill_flops"),
        _component(r, "llm_decode", "llm_decode_flops"),
        _get_total(r),
    )


def _print_block(title: str, frames, n_in: int, n_out: int) -> None:
    print(title)
    print("-" * 100)
    print(f"{'state_key':<24}  {'display name':<38}  "
          f"{'vision':>9} {'conn':>8} {'prefill':>9} {'decode':>8} {'TOTAL (PF)':>11}")
    rows = [_row(k, fn, frames, n_in, n_out) for k, fn in MODEL_FUNCTIONS.items()]
    rows.sort(key=lambda r: r[5])
    for key, vis, con, pre, dec, tot in rows:
        name = DISPLAY_NAMES[key]
        print(f"{key:<24}  {name:<38}  "
              f"{vis/1e15:>9.4f} {con/1e15:>8.4f} {pre/1e15:>9.4f} "
              f"{dec/1e15:>8.4f} {tot/1e15:>11.4f}")
    print()


if __name__ == "__main__":
    print("=" * 100)
    _print_block(
        "Test A: 8 frames @ 448x448, n_in_text=128, n_out_text=64  [matmul PFLOPs]",
        [{"height": 448, "width": 448}] * 8, 128, 64,
    )
    _print_block(
        "Test B: 32 frames @ 448x448, n_in_text=128, n_out_text=64 [matmul PFLOPs]",
        [{"height": 448, "width": 448}] * 32, 128, 64,
    )
