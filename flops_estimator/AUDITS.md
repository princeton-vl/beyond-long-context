# AUDITS — consolidated

This file consolidates the historical audit / critique / investigation
documents for `flops_estimator/`. The originals live unchanged in
`_old_md/` for forensic detail. Per-section bullets reference the source file.

Test fixture used throughout (unless explicitly stated otherwise):
`frames = [{height: 448, width: 448}] * 8`, `n_in_text = 128`,
`n_out_text = 64` (or `2048` for thinking variants).

---

## 1. Hardcoded-value findings (HARDCODED_AUDIT.md, CRITIQUE.md)

All four CRITICAL findings are RESOLVED. Quick status:

| Finding | Severity | Status |
|---|---|---|
| MiniCPM-V 4.5 / 2.6 multi-slice was hard-coded to 448² regardless of caller H/W | CRITICAL | RESOLVED — `_resize_helpers.minicpm_slice_geometry` is now a verbatim port of the upstream slicer, wired into both `flops_minicpmv45` and `flops_minicpmv26`. |
| LongVILA `LV_VIS_TOKENS_PER_FRAME` constant | CRITICAL | RESOLVED — replaced by `_longvila_tokens_per_frame(h, w) = (h//14)*(w//14)//4`. |
| Qwen `_round_to_patch_grid` was naive | MEDIUM | RESOLVED — upgraded to `smart_resize` (verbatim port). |
| GLM-4.5V `// patch` truncation | MEDIUM | RESOLVED — `_glm_snap` → `smart_resize`. |
| MiMo-VL `// patch` truncation | MEDIUM | RESOLVED — `smart_resize`. |
| Phi-4-MM `n_patches_per_crop` used caller H/W | MEDIUM | RESOLVED — each crop is fixed at 448² inside `dynamic_preprocess`; now hard-coded to `(448 // 14)² = 1024`. |
| Phi-4-MM `find_closest_aspect_ratio` uses naive `ceil` | MEDIUM | NOT FIXED — equal at square test fixtures; documented in the `CALLER CONTRACT` docstring. |
| Vocab consts, fallback `vit_tile=336`, `__main__` test fixtures, head-dim guesses | LOW | NOT FIXED — no FLOPs impact at any test point. |

The CRITIQUE.md "fabricated LongVILA pool formula" finding was superseded by
the per-batch override discovered in `__embed_media_tokens` — see Section 2.

## 2. LongVILA temporal pool — final resolution (FINAL_AUDIT.md)

`media_encoder.py:TSPVideoEncoder.__init__` sets the **constructor default**
`pool_sizes = [[8, 1, 1]]`. But `modeling_vila.py:__embed_media_tokens` (lines
614–625) defines `round_up_to_bucket(x)` and runs a **per-batch override**
that fires every forward (training and inference). The runtime override wins,
producing the bucketed schedule:

| N (input frames) | bucket | pool_t = 4·bucket | n_temporal = N // pool_t | LLM tokens (= n_temporal · 256) |
|---|---|---|---|---|
| ≤ 256 | 1 | 4 | floor(N/4) | floor(N/4) · 256 |
| ≤ 512 | 2 | 8 | floor(N/8) | floor(N/8) · 256 |
| ≤ 1024 | 4 | 16 | floor(N/16) | floor(N/16) · 256 |
| > 1024 | 8 | 32 | floor(N/32) | floor(N/32) · 256 |

At N=8: `pool_t=4`, `n_temporal=2` → 512 LLM tokens, total ≈ **0.0173 PF**.
At N=1024: 64 slots × 256 = 16,384 LLM tokens, total ≈ **1.342 PF**. The
inline note at `flops_longvila_mimo_phi.py:51-71` cites both source blocks.

## 3. Packing rules (PACKING_RULES.md)

Per-model post-LLM token formulas. All counts measured at H=W=448.

| Model | Formula (post-LLM tokens) | N=1 | N=7 | N=8 | N=9 | N=100 | N=1024 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL (7B) | `ceil(N/2) * (H/14/2) * (W/14/2)` | 256 | 1024 | 1024 | 1280 | 12800 | 131072 |
| Qwen3-VL (8B) / Thinking | `ceil(N/2) * (H/16/2) * (W/16/2)` | 196 | 784 | 784 | 980 | 9800 | 100352 |
| Qwen3-Omni (30B, A3B) | identical to Qwen3-VL | 196 | 784 | 784 | 980 | 9800 | 100352 |
| InternVL3.5 (all sizes, ± Thinking) | `N * tiles(H,W) * 256`, tiles=1 at 448² | 256 | 1792 | 2048 | 2304 | 25600 | 262144 |
| GLM-4.5V (104B, A12B) | `ceil(N/2) * (H/14/2) * (W/14/2)` | 256 | 1024 | 1024 | 1280 | 12800 | 131072 |
| MiniCPM-V 4.5 (9B) | `ceil(N/6) * 64` (6-frame packs) | 64 | 128 | 128 | 128 | 1088 | 10944 |
| MiniCPM V 2.6 (8B) | `N * 64` (per-frame resampler) | 64 | 448 | 512 | 576 | 6400 | 65536 |
| LongVILA (7B) [video] | `max(1, N // pool_t(N)) * 256`, bucketed | 256 | 256 | 512 | 512 | 6400 | 16384 |
| LongVILA (7B) [image] | `N * 256` | 256 | 1792 | 2048 | 2304 | 25600 | 262144 |
| MIMO-VL (7B) | `ceil(N/2) * (H/14/2) * (W/14/2)` | 256 | 1024 | 1024 | 1280 | 12800 | 131072 |
| Phi-4-MM (6B) | `N * 545` (HD-transform, 448x448 input) | 545 | 3815 | 4360 | 4905 | 54500 | 558080 |

## 4. Frame-scaling totals (FRAME_SCALING.md)

Total FLOPs in PFLOPs (matmul + elementwise = `total_with_elementwise`),
H = W = 448, n_in = 128, n_out = 64. Phi-4-MM additionally includes the
always-active vision LoRA (r=256).

| Model | N=1 | N=8 | N=16 | N=64 | N=128 | N=512 | N=1024 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL (7B) | 0.007 | 0.022 | 0.043 | 0.203 | 0.502 | 4.377 | 15.091 |
| Qwen3-VL (8B) (± Thinking) | 0.006 | 0.018 | 0.037 | 0.217 | 0.637 | 7.490 | 28.174 |
| Qwen3-Omni (30B, A3B) | 0.003 | 0.010 | 0.023 | 0.170 | 0.561 | 7.565 | 29.324 |
| InternVL3.5 V (8B) (± Thinking) | 0.007 | 0.040 | 0.082 | 0.441 | 1.199 | 12.462 | 45.386 |
| InternVL3.5 V (30B, A3B) (± Thinking) | 0.003 | 0.022 | 0.050 | 0.355 | 1.135 | 14.769 | 56.823 |
| InternVL3.5 V (38B) (± Thinking) | 0.040 | 0.246 | 0.498 | 2.383 | 5.890 | 50.811 | 174.370 |
| GLM-4.5V (104B, A12B) | 0.014 | 0.044 | 0.086 | 0.388 | 0.925 | 7.398 | 24.692 |
| MiniCPM-V 4.5 (9B) | 0.005 | 0.013 | 0.022 | 0.080 | 0.158 | 0.630 | 1.274 |
| MiniCPM V 2.6 (8B) | 0.004 | 0.018 | 0.033 | 0.127 | 0.257 | 1.187 | 2.810 |
| LongVILA (7B) [video] | 0.007 | 0.017 | 0.032 | 0.127 | 0.265 | 0.835 | 1.342 |

## 5. Silent input downsampling at high N (MAX_PIXELS_INVESTIGATION.md, USER_CODE_DOWNSAMPLE_AUDIT.md, DOWNSAMPLE_REPORT.md)

Two of the 16 models silently downsample 448×448 inputs at large N because
their `*VideoProcessor` enforces a *video-wide* pixel budget. The estimator
does **not** apply this — callers must downsample inputs themselves if they
need apples-to-apples agreement with profiler measurements.

| state_key | downsamples? | first triggers at | actual H×W at N=1024 |
|---|---|---|---|
| `qwen3dense`, `qwen3densethinking` | YES | N ≥ 128 | 96 × 96 |
| `glm45v` | YES | N ≥ 256 | 196 × 196 |
| All other 14 models | NO | — | 448 × 448 (up to N=1024) |

Exact tables (verified against the live `AutoProcessor`):

**Qwen3-VL** (`size = {"shortest_edge": 4096, "longest_edge": 25_165_824}`,
factor=32):

| N | actual H×W | post-merge visual tokens |
|---|---|---|
| 8 | 448×448 | 784 |
| 64 | 448×448 | 6,272 |
| 128 | 416×416 | 10,816 |
| 256 | 288×288 | 10,368 |
| 512 | 192×192 | 9,216 |
| 1024 | 128×128 | 8,192 |

**GLM-4.5V** (`size = {"shortest_edge": 12544, "longest_edge": 47_040_000}`,
factor=28):

| N | actual H×W | post-merge visual tokens |
|---|---|---|
| 8 | 448×448 | 1,024 |
| 128 | 448×448 | 16,384 |
| 256 | 420×420 | 28,800 |
| 512 | 280×280 | 25,600 |
| 1024 | 196×196 | 25,088 |

The 30–130× prediction/measurement gap at high-N video is mostly **NOT**
caused by silent downsampling: it is dominated by `torch.profiler`
under-counting flash-attention FLOPs (`flash_attn::_flash_attn_varlen_forward`
reports `flops=0`). For Qwen2.5-VL / Qwen3-Omni / MiMo-VL — which do **not**
downsample — the entire prediction-vs-measurement gap is the missing
flash-attn term. For Qwen3-VL / GLM-4.5V the gap is the product of (a)
missing flash-attn FLOPs in measurement AND (b) genuinely degraded inputs.

## 6. Architectural truncation at N=1024 (TRUNCATION_AUDIT_13.md)

Beyond per-frame pixel budgets, several models silently drop frames at very
high N because they hit LLM context limits or per-image caps. Summary at
synthetic 448×448 video, N=1024, fps=1:

| state_key | frames truncated at N=1024? | mechanism |
|---|---|---|
| `qwen25video` | NO | per-frame budget; LLM ctx fits |
| `qwen3dense`, `qwen3densethinking` | NO (downsampled instead) | per-video budget |
| `qwen3omni` | NO | per-frame; LLM ctx fits |
| `glm45v` | NO (downsampled instead) | per-video budget |
| `mimovl` | NO | per-frame; LLM ctx fits |
| `internvl*` | NO (1 patch/frame in video mode) | `max_num=1` caller convention |
| `minicpm45`, `minicpmstream` | NO | per-frame slicing |
| `longvila` | YES — bucketed pool truncates LLM-side | `pool_t=16` for N≤1024 → 64 slots |
| `phi4mm` | NO | per-frame; HD-transform fits |

Detail and exact LLM-context numbers in `_old_md/TRUNCATION_AUDIT_13.md`.

## 7. Remaining caller-set conventions (REMAINING_ESTIMATES.md)

Two numbers are **caller conventions**, not modeling-code constants. Both are
flagged in the per-function `CALLER CONTRACT` docstrings.

1. **MiniCPM-V 4.5 `pack_size = 6`.** The resampler's `batch_attn_forward`
   flattens whatever group it is given via `temporal_ids`. The "6" appears
   in the model card and conventional inference scripts; we adopt it.
2. **InternVL3.5 `max_num = 1` for video.** The chat preprocessor accepts any
   `max_num`. We adopt the model-card-recommended video setting (no dynamic
   tiling).

## 8. Outstanding documentation contradictions

None blocking. The CRITIQUE.md "fabricated pool formula" claim was retracted
by FINAL_AUDIT.md (the override exists at runtime, in `__embed_media_tokens`,
not in the encoder's `__init__`). All other audit documents are consistent
with the post-fix codebase.

## 9. Out-of-scope

`_old_md/LZ_ENTROPY.md` is a Lempel-Ziv complexity table for the synthetic
benchmark sequences. It does not describe FLOPs and was filed here only by
proximity. Left in `_old_md/` for now.
