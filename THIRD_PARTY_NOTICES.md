# Third-Party Notices

This repository vendors several upstream projects under their original
licenses. The top-level `LICENSE` (MIT) covers only first-party code in
this repository; it does **not** relicense any vendored component below.
Each vendored file or directory remains under its upstream license, and
the corresponding upstream `LICENSE` is included in this repository at
the location indicated.

Where one upstream contributes several single-file modules, its `LICENSE`
lives once under `external_licenses/` at the repository root and is
cross-referenced from the relevant entries below rather than duplicated.

---

## Vendored directories

### `external/LongVILA/`

- **Upstream:** NVlabs/LongVILA — https://github.com/NVlabs/LongVILA
- **License:** Apache License 2.0
- **In-tree LICENSE:** `external/LongVILA/LICENSE`

### `external/TimeChat/`

- **Upstream:** RenShuhuai-Andy/TimeChat — https://github.com/RenShuhuai-Andy/TimeChat
- **License:** BSD 3-Clause
- **In-tree LICENSE:** `external/TimeChat/LICENSE`

### `external/TimeChat/qwen2_5_vl/`

- **Upstream:** Qwen2.5-VL reference modeling code from `huggingface/transformers`
  (https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)
- **License:** Apache License 2.0 (Copyright 2018- The Hugging Face team)
- **In-tree LICENSE:** `external/TimeChat/qwen2_5_vl/LICENSE`

### `training/calibrated_memory/backend/models/external/ttt_fast/ThunderKittens/`

- **Upstream:** HazyResearch/ThunderKittens — https://github.com/HazyResearch/ThunderKittens
- **License:** MIT License (Copyright 2024 HazyResearch)
- **In-tree LICENSE:** `training/calibrated_memory/backend/models/external/ttt_fast/ThunderKittens/LICENSE`

### `training/calibrated_memory/backend/models/external/ttt_fast/` (parent module)

- **Upstream provenance:** Derivative of `test-time-training/ttt-lm-pytorch`
  (MIT). The upstream `ttt.py` defines the TTT primitive that
  `modeling_ttt.py` extends. Additionally, `generation.py` carries a
  per-file header stating "Modified from
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py"
  (Apache-2.0).
- **In-tree LICENSE:** `external_licenses/ttt-lm-pytorch.LICENSE` (MIT)
  and `external_licenses/mamba.LICENSE` (Apache 2.0) for the respective
  derivative portions.

### `training/calibrated_memory/backend/models/external/ttt_fast/triton_kernel/`

- **Upstream:** First-party. Triton kernels for fast TTT inference
  (`activations.py`, `fused_gate_outln.py`, `ttt_linear_decode.py`,
  `ttt_mlp_decode.py`) authored as part of this work. Released under
  the repository-level `LICENSE` (MIT).
- **xformers attribution:** `activations.py:9` carries the inline
  comment "from xformers impl." for the `gelu_tl` GELU-approximation
  helper (with `diff_gelu_tl` derived from the same approximation). Per
  the upstream xformers BSD 3-Clause license, the attribution is
  retained inline and the LICENSE text is included at
  `external_licenses/xformers.LICENSE`.

### `data_gen/`

- **Origin:** First-party. The `data_gen/vidgeom/` package and the
  `data_gen/seq2vid/` CLI suite are authored as part of this work and
  covered by the repository-level `LICENSE` (MIT).

---

## Single-file vendored modules under `training/calibrated_memory/backend/models/external/`

Individual `.py` files copied or adapted from upstream projects. Each
upstream `LICENSE` is included once at the listed path in
`external_licenses/`.

| File | Upstream | License | In-tree LICENSE |
|---|---|---|---|
| `compressive_transformer.py` | lucidrains/compressive-transformer-pytorch — https://github.com/lucidrains/compressive-transformer-pytorch | MIT | `external_licenses/compressive-transformer-pytorch.LICENSE` |
| `deltaformer_parallel_debug.py` | fla-org/flash-linear-attention — https://github.com/fla-org/flash-linear-attention (Copyright 2023-2025 Songlin Yang, Yu Zhang) | MIT | `external_licenses/flash-linear-attention.LICENSE` |
| `log_linear_mamba_chunk_debug.py` | Derivative of `fla-org/flash-linear-attention` (imports `from fla.ops.utils ...` and `from fla.utils ...`); debug/instrumentation variant of an `fla` chunk-scan op. | MIT | `external_licenses/flash-linear-attention.LICENSE` |
| `mamba2.py` | state-spaces/mamba — https://github.com/state-spaces/mamba (Copyright 2024 Tri Dao, Albert Gu) | Apache License 2.0 | `external_licenses/mamba.LICENSE` |
| `memory_mosaics_blocks.py` | Meta Platforms, Inc.; per-file header states "derived from nanoGPT" (karpathy/nanoGPT — https://github.com/karpathy/nanoGPT). Released under MIT terms consistent with the nanoGPT base. | MIT | `external_licenses/nanoGPT.LICENSE` |
| `stm.py` | thaihungle/SAM — https://github.com/thaihungle/SAM (Self-Attentive Associative Memory; Copyright 2020 Tony) | MIT | `external_licenses/SAM.LICENSE` |
| `ttt.py` | test-time-training/ttt-lm-pytorch — https://github.com/test-time-training/ttt-lm-pytorch | MIT | `external_licenses/ttt-lm-pytorch.LICENSE` |

---

## Top-level summary of in-tree license files

```
LICENSE                                                              # MIT, first-party
external/LongVILA/LICENSE                                            # Apache 2.0
external/TimeChat/LICENSE                                            # BSD 3-Clause
external/TimeChat/qwen2_5_vl/LICENSE                                 # Apache 2.0
external_licenses/
  ├── compressive-transformer-pytorch.LICENSE                        # MIT
  ├── flash-linear-attention.LICENSE                                 # MIT
  ├── mamba.LICENSE                                                  # Apache 2.0
  ├── nanoGPT.LICENSE                                                # MIT
  ├── SAM.LICENSE                                                    # MIT
  ├── ttt-lm-pytorch.LICENSE                                         # MIT
  └── xformers.LICENSE                                               # BSD 3-Clause
training/calibrated_memory/backend/models/external/ttt_fast/
  └── ThunderKittens/LICENSE                                         # MIT
```
