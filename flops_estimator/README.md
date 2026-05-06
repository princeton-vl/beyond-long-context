# flops_estimator

First-principles, **matmul-only** FLOPs estimators for the 16 video-LLMs in this
project. Every constant in this directory is sourced from each model's
HuggingFace `config.json` and verified against its reference modeling code; see
the per-function `VISION-ENCODER AUDIT` blocks for line-level citations and
`AUDITS.md` for the consolidated audit history.

## What this module computes

Each `flops_*` function:

```python
fn(frames, n_in_text_tokens, n_out_text_tokens) -> dict
# frames:            list[{'height': int, 'width': int}]
# n_in_text_tokens:  int  (text prompt tokens entering the LLM)
# n_out_text_tokens: int  (generated tokens; thinking variants pass a larger
#                          n_out to capture the chain-of-thought)
```

The returned dict is the per-component matmul FLOPs:

```
vision_flops, connector_flops, llm_prefill_flops, llm_decode_flops, total
+ matching *_elementwise keys, elementwise_total, total_with_elementwise
```

A few files use `total_flops` instead of `total`; `flops_all_models._get_total`
normalises this. Phi-4-MM additionally returns `vision_lora`.

## Counting convention

- **Matmul only.** Every `(A, B) · (B, C)` matmul costs `2 * A * B * C`.
  Softmax / norms / RoPE / activations / embedding lookups are **not** counted
  in the headline `total`; they live in the `*_elementwise` fields and are
  included in `total_with_elementwise` for callers that want them.
- **Causal mask.** Full `N²` per attention matmul (Chinchilla / Kaplan
  scaling-law convention). Applied uniformly across all 16 models, so
  cross-model ratios are unaffected by this choice.
- **GQA.** Counted at the actual projection dims; the softmax-V product stays
  proportional to `n_q_heads` because the K/V are broadcast per query head.
- **MoE.** Router (dense Linear) + top-k SwiGLU experts; shared experts are
  counted whenever the model has them.
- **Vision LoRA** (Phi-4-MM only). `2 * (in*r + r*out)` per token per adapted
  Linear, summed across all prefill + decode tokens.

## Files

| File | Role |
|---|---|
| `flops_qwen.py` | Qwen2.5-VL, Qwen3-VL (+ Thinking), Qwen3-Omni |
| `flops_internvl.py` | InternVL3.5 8B / 30B-A3B / 38B (+ Thinking) |
| `flops_glm_minicpm.py` | GLM-4.5V, MiniCPM-V 4.5, MiniCPM-V 2.6 |
| `flops_longvila_mimo_phi.py` | LongVILA-R1, MiMo-VL, Phi-4-MM |
| `flops_all_models.py` | `MODEL_FUNCTIONS` (state_key → fn), `DISPLAY_NAMES`, `compute_all` |
| `elementwise.py` | Per-element FLOPs helpers (norms / softmax / RoPE / ...) |
| `_resize_helpers.py` | `smart_resize` (Qwen-family) + `minicpm_slice_geometry` |
| `validate.py` | Comparison runs at the canonical 8/32-frame test points |
| `__init__.py` | Re-exports the 16 functions |
| `AUDITS.md` | Consolidated forensic audit history (resolution audits, frame-scaling tables, hardcoded-value audits, truncation audits) |
| `MEASURED_VS_PREDICTED.png` and `measurement_vs_prediction_*.csv` | Latest measured-vs-predicted artefacts |

## How to call

```python
from flops_estimator.flops_all_models import MODEL_FUNCTIONS, DISPLAY_NAMES

frames = [{"height": 448, "width": 448}] * 8
fn = MODEL_FUNCTIONS["glm45v"]                  # state_key → function
result = fn(frames, n_in_text_tokens=128, n_out_text_tokens=64)
print(DISPLAY_NAMES["glm45v"], result["total"])
```

The 16 supported `state_key` values are listed in `MODEL_FUNCTIONS` and in
`DISPLAY_NAMES` (which maps them to the canonical paper names, e.g.
`"glm45v" → "GLM-4.5V (104B, A12B)"`).

## Resolution / slicing

All per-frame ViT seq-length math goes through `_resize_helpers.py`:

- **Qwen-family + GLM-4.5V + MiMo-VL.** Each frame's `(H, W)` is run through
  `smart_resize(H, W, factor=patch*spatial_merge)` (verbatim port of
  `transformers/.../image_processing_qwen2_vl.py:smart_resize`). Outputs are
  rounded to multiples of `factor`, with total pixels clamped to
  `[56², 14²·4·1280] = [3136, 1003520]`. At 448×448 this is a no-op; at
  high-res it drives the ViT seq length up.
- **MiniCPM-V 2.6 / 4.5.** Each frame's `(H, W)` drives
  `minicpm_slice_geometry(H, W)` which replicates the upstream
  `image_processing_minicpmv.py:get_sliced_grid + slice_image +
  get_sliced_images`, returning one `(h, w)` per ViT forward (1 thumbnail at
  448-base, +`grid_x·grid_y` sub-crops). The function then loops the SigLIP
  ViT once per slice. At 448×448 the grid is empty so 1 forward; at
  896×896 it is `1 + 2·2 = 5` forwards per frame.
- **LongVILA.** `_longvila_tokens_per_frame(H, W)` returns
  `(H/14)·(W/14)/4` (mlp_downsample_2x2_fix); this drives both the connector
  compute and the post-pool LLM-token count.
- **Phi-4-MM.** `_phi4_hd_geometry(H, W)` returns the dyhd grid + per-frame
  LLM tokens (`273 + 272 + 1 = 545` at 448²; scales linearly with sub-crops).
- **InternVL3.5.** `_tiles_per_frame(H, W, cfg)` calls `_closest_grid` to pick
  the `(i, j)` tile-grid minimising aspect error within
  `[min_dynamic_patch=1, max_dynamic_patch=12]`, and adds a thumbnail when
  `i·j > 1`. Each tile's ViT pass runs at 448×448.

## Caveats (must read)

1. **Silent input downsampling at high N for two models.** Qwen3-VL (and its
   Thinking variant) and GLM-4.5V enforce a *video-wide* pixel budget inside
   the `*VideoProcessor`. With a 448×448 input feed, Qwen3-VL starts
   downsampling at N=128 (96×96 at N=1024); GLM-4.5V starts at N=256 (196×196
   at N=1024). The other 14 models keep frames at 448×448 up to N=1024. This
   estimator does **not** apply that budget — it computes FLOPs at the H/W
   you pass in. If you want to compare against real measurements you must
   downsample the input frames yourself before calling the estimator. See the
   `MAX_PIXELS` discussion in `AUDITS.md`.
2. **Caller-set conventions.** Two numbers are *caller* conventions, not
   modeling-code constants:
   - **MiniCPM-V 4.5** packs 6 frames per resampler call. The resampler itself
     accepts any `temporal_ids` grouping; "6" comes from the model card's
     "compresses 6 frames -> 64 tokens" and from typical inference scripts.
   - **InternVL3.5** uses `max_num=1` for video (no dynamic tiling). The chat
     preprocessor lets the caller pass any `max_num`; we adopt the
     model-card-recommended video setting.
3. **Phi-4-MM `find_closest_aspect_ratio`** uses a naive `ceil` instead of the
   shipped greedy minimiser. Equal at our square test fixtures; can diverge
   on irregular aspect ratios. Documented in the per-function
   `CALLER CONTRACT` docstring.
4. **Elementwise opt-out.** The `total` field is the matmul-only number used
   by all paper figures. `total_with_elementwise` adds the per-element
   contribution from `elementwise.py`. Sensitivity is <1% on every model;
   `AUDITS.md` flags those constants as "convention".
5. **Flash-attention and `torch.profiler`.** Profiler runs (in
   `measurement_vs_prediction_rows.csv`) under-count attention because
   `flash_attn::_flash_attn_varlen_forward` reports `flops=0`. The
   prediction-vs-measurement gap at high N is mostly this, not estimator
   error.

## Validation artefacts

- `MEASURED_VS_PREDICTED.png` — scatter of per-row predictions vs profiler
  measurements.
- `measurement_vs_prediction_rows.csv` — per-row ratios.
- `measurement_vs_prediction_summary.csv`, `…_worst.csv` — per-model digest +
  worst outliers.

For deeper history (resolution audits, frame-scaling tables, hardcoded-value
audit, truncation audit) see `AUDITS.md`.
