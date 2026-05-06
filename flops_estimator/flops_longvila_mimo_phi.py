"""
First-principles FLOPs equations for LongVILA, MiMo-VL, and Phi-4-MM.

Methodology
-----------
We count matmul FLOPs only, using 2 * a * b * c per matmul (one mul + one add per
mac). We ignore softmax, normalization, activation functions, and elementwise
ops. Bias is folded into the dominant matmul cost (negligible).

Each transformer layer (decoder, vision encoder, LLM) is decomposed into:
  - Attention input projection: Q + K + V (with GQA/MQA accounted via num_kv_heads)
  - Attention output projection: O
  - Attention scores QK^T:  2 * H * S * D_h * S  (H = full heads, D_h = head dim)
  - Attention values  AV:   2 * H * S * S * D_h
  - MLP: depends on gating (SwiGLU = 3 linears, classic = 2)

CHANGES (Pass 3 — exhaustive verification against actual modeling code)
-----------------------------------------------------------------------
1. **MiMo-VL windowed attention IS active.** The released MiMo-VL config sets
   ``window_size=112`` and ``fullatt_block_indexes=[7, 15, 23, 31]``. The
   transformers Qwen2_5_VLVisionTransformer.forward() obeys these unconditionally
   (modeling_qwen2_5_vl.py:520-531) — there is no "disable windowing" toggle.
   So 28 of 32 vision blocks use windowed attention with
       vit_merger_window_size = window_size // spatial_merge_size // patch_size
                              = 112 // 2 // 14 = 4 (in merged-token units)
   = 8x8 = 64 raw patches per window. Each "full" layer attends over the full
   flattened sequence (all temporal pairs concatenated). Citation:
   transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:438-477,520-531

2. **MiMo-VL ViT runs on the FULL packed sequence, not per pair.** Qwen2.5-VL
   patch-embeds the entire (grid_t, grid_h, grid_w) volume into a single
   sequence of length grid_t*grid_h*grid_w. Each block computes QKV/O/MLP
   over that full sequence; cu_seqlens (full vs window) only changes the
   attention-scores mask. Citation: modeling_qwen2_5_vl.py:222-269,479-536.

3. **MiMo-VL PatchMerger: Linear(5120 -> 5120) -> GELU -> Linear(5120 -> 4096).**
   Confirmed in modeling_qwen2_5_vl.py:135-148. Previous code used
   (5120 -> 4096 -> 4096), which was wrong.

4. **Phi-4-MM HD-transform LLM tokens for 448x448 = 545 per frame, NOT 256.**
   processing_phi4mm.py: 448x448 -> 1x1 sub-grid (so 1 sub-crop) + 1 global
   thumbnail = 2 SigLIP forwards. The image_token_compression=avg_pool_2d takes
   each crop's 32x32 to 16x16 (256 tokens). Then modeling_phi4mm.py builds:
       glb_img : 16 rows of 16 tokens + 16 sub_GN row-separators -> 272 tokens
       sub_img : 16 rows of 16 tokens + 16 sub_GN row-separators -> 272 tokens
       total   : sub_img(272) + glb_GN(1) + glb_img(272) = 545 tokens
   All 545 go through img_projection and reach the LLM as visual tokens.
   Citation: processing_phi4mm.py:188-244, modeling_phi4mm.py:320-406, the
   num_img_tokens formula (=256+1+sum(mask)+mask[:,0].sum()+16=545).

5. **LongVILA TSP pool: pool_sizes[0][0] is MUTATED per-batch via
   ``round_up_to_bucket`` (re-corrected 2026-04-27).**
   The constructor in ``media_encoder.py:TSPVideoEncoder.__init__`` sets
   ``self.pool_sizes = [[8, 1, 1]]`` — but ``modeling_vila.py:738-739`` then
   OVERWRITES ``pool_sizes[0][0]`` per batch using
       pool_t = 4 * round_up_to_bucket(num_video_frames / 256)
   where ``round_up_to_bucket`` returns 1, 2, 4, or 8 (clamped at 8). Buckets:
       N <=  256: bucket=1, pool_t = 4
       N <=  512: bucket=2, pool_t = 8
       N <= 1024: bucket=4, pool_t = 16
       N >  1024: bucket=8, pool_t = 32
   Pool function is ``view(-1, pool_t).mean(dim+1)`` per ``media_encoder.py:103-104``;
   it requires ``N % pool_t == 0`` (frames not divisible would error in real
   inference; we clamp to ``max(1, N//pool_t)`` for FLOPs accounting at small N).
   For N=8 -> pool_t=4 -> n_temporal=2 -> 512 LLM tokens (NOT 256).
   For N=1024 -> pool_t=16 -> n_temporal=64 -> 16384 LLM tokens.
   Sources (verbatim): ``self.pool_sizes = [[8, 1, 1]]`` —
   ``TSPVideoEncoder.__init__`` (``media_encoder.py:115``); the per-batch
   override is ``modeling_vila.py:739``:
     ``self.encoders[name].pool_sizes[0][0] = 4 * round_up_to_bucket(num_video_frames / 256)``
   in ``Efficient-Large-Model/LongVILA-R1-7B``.

6. Phi-4-MM Vision LoRA confirmed: r=256, alpha=512, modules
   qkv_proj/o_proj/gate_up_proj/down_proj on every LLM layer (32). Module
   shapes (Phi4MMSdpaAttention/Phi4MMMLP, modeling_phi4mm.py:1086-1088,1026):
       qkv_proj    : Linear(3072, 5120)   # 24*128 + 2*8*128
       o_proj      : Linear(3072, 3072)
       gate_up_proj: Linear(3072, 16384)
       down_proj   : Linear(8192, 3072)
   LoRA cost per token, per layer = sum_modules 2*(in*r + r*out).

7. Phi-4-MM tied embeddings confirmed (config.json:tie_word_embeddings=true).
   FLOPs unchanged (the lm_head matmul still happens; tying only saves params).
"""

from __future__ import annotations

from .elementwise import (
    rmsnorm_flops, layernorm_flops, residual_flops, bias_flops,
    rope_flops, rope_flops_decode,
    softmax_flops_attention, softmax_flops_attention_windowed,
    softmax_flops_decode,
    silu_flops, gelu_exact_flops, gelu_tanh_flops,
    lm_head_softmax_decode,
)
from ._resize_helpers import smart_resize


# --- generic per-layer cost helpers --------------------------------------- #

def _mha_layer_flops(
    seq_len: int,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    ffn: int,
    mlp_kind: str,  # "swiglu" (3 linears) or "gelu" (2 linears)
) -> int:
    """FLOPs for a single transformer block forward pass (matmuls only).

    Attention is FULL self-attention over `seq_len` (no windowing).
    """
    head_dim = hidden // n_heads
    kv_proj_dim = n_kv_heads * head_dim  # GQA shrinks K, V

    # Attention input projections: Q (D x D), K (D x kv_proj_dim), V (D x kv_proj_dim)
    flops_q = 2 * seq_len * hidden * hidden
    flops_k = 2 * seq_len * hidden * kv_proj_dim
    flops_v = 2 * seq_len * hidden * kv_proj_dim
    # Output projection O: (D x D)
    flops_o = 2 * seq_len * hidden * hidden

    # Attention scores QK^T: per head, S x D_h * D_h x S => 2*S*S*D_h, times H heads.
    # GQA broadcasts KV across query groups, so the outer loop is full H heads.
    flops_qkt = 2 * n_heads * seq_len * seq_len * head_dim
    # AV: S x S * S x D_h => 2*S*S*D_h per head, times H
    flops_av = 2 * n_heads * seq_len * seq_len * head_dim

    if mlp_kind == "swiglu":
        # gate (D x ffn), up (D x ffn), down (ffn x D)
        flops_mlp = 2 * seq_len * hidden * ffn * 3
    elif mlp_kind == "gelu":
        # fc1 (D x ffn), fc2 (ffn x D)
        flops_mlp = 2 * seq_len * hidden * ffn * 2
    else:
        raise ValueError(f"unknown mlp_kind={mlp_kind}")

    return flops_q + flops_k + flops_v + flops_o + flops_qkt + flops_av + flops_mlp


def _windowed_attn_layer_flops(
    total_seq: int,
    window_seq: int,
    n_windows: int,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    ffn: int,
    mlp_kind: str,
) -> int:
    """Vision block where QKV/O/MLP scale with the FULL packed sequence but the
    attention scores & AV matmuls are computed inside non-overlapping windows.
    """
    head_dim = hidden // n_heads
    kv_proj_dim = n_kv_heads * head_dim

    flops_q = 2 * total_seq * hidden * hidden
    flops_k = 2 * total_seq * hidden * kv_proj_dim
    flops_v = 2 * total_seq * hidden * kv_proj_dim
    flops_o = 2 * total_seq * hidden * hidden

    # Each window's QK^T and AV (assuming uniform window_seq).
    flops_qkt = 2 * n_heads * window_seq * window_seq * head_dim * n_windows
    flops_av = 2 * n_heads * window_seq * window_seq * head_dim * n_windows

    if mlp_kind == "swiglu":
        flops_mlp = 2 * total_seq * hidden * ffn * 3
    elif mlp_kind == "gelu":
        flops_mlp = 2 * total_seq * hidden * ffn * 2
    else:
        raise ValueError(f"unknown mlp_kind={mlp_kind}")

    return flops_q + flops_k + flops_v + flops_o + flops_qkt + flops_av + flops_mlp


def _llm_decode_flops(
    n_prefill: int,
    n_decode: int,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    ffn: int,
    n_layers: int,
    vocab_size: int,
    tied_embeddings: bool,
    mlp_kind: str = "swiglu",
) -> int:
    """Autoregressive decode of n_decode tokens after a prefill of n_prefill."""
    head_dim = hidden // n_heads
    kv_proj_dim = n_kv_heads * head_dim

    proj_per_step = 2 * 1 * (hidden * hidden + 2 * hidden * kv_proj_dim + hidden * hidden)
    if mlp_kind == "swiglu":
        mlp_per_step = 2 * 1 * hidden * ffn * 3
    elif mlp_kind == "gelu":
        mlp_per_step = 2 * 1 * hidden * ffn * 2
    else:
        raise ValueError(mlp_kind)
    fixed_per_step = (proj_per_step + mlp_per_step) * n_layers

    # Attention summed across steps:
    #   sum_{i=0..n_decode-1} (n_prefill + i + 1) = n_decode*n_prefill + n_decode*(n_decode+1)/2
    s = n_decode * n_prefill + n_decode * (n_decode + 1) // 2
    attn_total = 4 * n_heads * head_dim * s * n_layers

    lm_head = 2 * n_decode * hidden * vocab_size
    _ = tied_embeddings  # tying does not change matmul shape.

    fixed_total = fixed_per_step * n_decode
    return fixed_total + attn_total + lm_head


def _llm_prefill_flops(
    seq_len: int,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    ffn: int,
    n_layers: int,
    vocab_size: int,
    mlp_kind: str = "swiglu",
) -> int:
    """Total LLM prefill FLOPs (matmul only): n_layers transformer blocks
    plus a single lm_head pass. The lm_head only fires once at the end of
    prefill (one logits row), so its term is `2 * 1 * hidden * vocab_size`."""
    layer = _mha_layer_flops(seq_len, hidden, n_heads, n_kv_heads, ffn, mlp_kind)
    lm_head = 2 * 1 * hidden * vocab_size
    return layer * n_layers + lm_head


def _lora_flops_per_token(in_dim: int, out_dim: int, r: int) -> int:
    """One LoRA branch over a single token: 2*(in*r + r*out) FLOPs."""
    return 2 * (in_dim * r + r * out_dim)


# ---------------------------------------------------------------------------
# Elementwise per-block helpers
# ---------------------------------------------------------------------------

def _siglip_block_elem(N: int, hidden: int, n_heads: int, ffn: int) -> int:
    """SigLIP-style ViT block: LayerNorm, qkv_bias=True, learned PE (no RoPE),
    GELU exact, biases on FFN linears."""
    norms = 2 * layernorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_bias = bias_flops(N, 3 * hidden)
    o_bias = bias_flops(N, hidden)
    attn_sm = softmax_flops_attention(N, n_heads)
    act = gelu_exact_flops(N, ffn)
    ffn_bias = bias_flops(N, ffn) + bias_flops(N, hidden)
    return norms + residuals + qkv_bias + o_bias + attn_sm + act + ffn_bias


def _qwen25_vit_block_elem(N: int, hidden: int, n_heads: int, ffn: int,
                           window_tokens: int | None) -> int:
    """Qwen2.5-VL ViT block (used by MiMo-VL): LayerNorm, qkv_bias=True,
    2D MRoPE, SwiGLU SiLU, biases on FFN linears."""
    head_dim = hidden // n_heads
    norms = 2 * layernorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_bias = bias_flops(N, 3 * hidden)
    o_bias = bias_flops(N, hidden)
    rope = rope_flops(N, head_dim, n_heads, n_heads)
    if window_tokens is not None and window_tokens < N:
        attn_sm = softmax_flops_attention_windowed(N, window_tokens, n_heads)
    else:
        attn_sm = softmax_flops_attention(N, n_heads)
    act = silu_flops(N, ffn)
    gateup_mul = N * ffn
    ffn_bias = bias_flops(N, ffn) * 2 + bias_flops(N, hidden)
    return norms + residuals + qkv_bias + o_bias + rope + attn_sm + act + gateup_mul + ffn_bias


def _llm_dense_block_elem_prefill(
    N: int, hidden: int, n_q: int, n_kv: int, head_dim: int, ffn: int,
    *, has_qk_norm: bool, has_qkv_bias: bool, mlp_kind: str = 'swiglu',
) -> int:
    """Elementwise FLOPs for ONE dense LLM block at prefill length N (RMSNorm,
    optional QKV bias / qk-norm, RoPE, softmax, FFN activation). Multiply by
    n_layers in the caller."""
    norms = 2 * rmsnorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    qk_norm = (rmsnorm_flops(N, head_dim) * (n_q + n_kv)) if has_qk_norm else 0
    rope = rope_flops(N, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_attention(N, n_q)
    if mlp_kind == 'swiglu':
        ffn_elem = silu_flops(N, ffn) + N * ffn
    else:
        ffn_elem = gelu_tanh_flops(N, ffn)
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem


def _llm_dense_block_elem_decode(
    N_in: int, n_out: int, hidden: int, n_q: int, n_kv: int, head_dim: int,
    ffn: int, *, has_qk_norm: bool, has_qkv_bias: bool, mlp_kind: str = 'swiglu',
) -> int:
    """Elementwise FLOPs for ONE dense LLM block summed over `n_out` decode
    steps (KV cache prefilled to length N_in). Multiply by n_layers in caller."""
    if n_out <= 0:
        return 0
    norms = 2 * rmsnorm_flops(1, hidden) * n_out
    residuals = 2 * residual_flops(1, hidden) * n_out
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = (bias_flops(1, qkv_dim) * n_out) if has_qkv_bias else 0
    qk_norm = ((rmsnorm_flops(1, head_dim) * (n_q + n_kv)) * n_out) if has_qk_norm else 0
    rope = rope_flops_decode(n_out, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_decode(N_in, n_out, n_q)
    if mlp_kind == 'swiglu':
        ffn_elem = silu_flops(n_out, ffn) + n_out * ffn
    else:
        ffn_elem = gelu_tanh_flops(n_out, ffn)
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem


def _patches_for(frame: dict, patch_size: int) -> int:
    """Number of pre-merge ViT patches for a single frame at the caller's
    H/W (no smart-resize: callers using this helper own their snapping)."""
    h = frame["height"]
    w = frame["width"]
    return (h // patch_size) * (w // patch_size)


# --- LongVILA -------------------------------------------------------------- #

# All values cited from the LongVILA-R1-7B HuggingFace repo
# (Efficient-Large-Model/LongVILA-R1-7B). The vision tower is SigLIP-So400M
# (model_args.json -> "vision_tower_name": "google/siglip-so400m-patch14-384"
# overridden to image_size=448; google/siglip-so400m config: hidden_size=1152,
# intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
# patch_size=14). The LLM is Qwen2-7B (config.json -> llm_cfg ->
# hidden_size=3584, num_hidden_layers=28, intermediate_size=18944,
# num_attention_heads=28, num_key_value_heads=4, vocab_size=151651).
LV_VIS_HIDDEN = 1152      # siglip-so400m hidden_size
LV_VIS_FFN = 4304         # siglip-so400m intermediate_size
LV_VIS_LAYERS = 27        # siglip-so400m num_hidden_layers
LV_VIS_HEADS = 16         # siglip-so400m num_attention_heads
LV_VIS_PATCH = 14         # siglip-so400m patch_size
LV_VIS_IMG = 448          # LongVILA siglip image_size override (model_args.json)
LV_PROJ_DOWNSAMPLE = 4    # mlp_downsample_2x2_fix kernel=2 stride=2 -> 4x token reduction
                          # (base_projector.py:153-161 in LongVILA-R1-7B)

LV_LLM_HIDDEN = 3584      # llm_cfg.hidden_size (Qwen2-7B)
LV_LLM_FFN = 18944        # llm_cfg.intermediate_size
LV_LLM_LAYERS = 28        # llm_cfg.num_hidden_layers
LV_LLM_HEADS = 28         # llm_cfg.num_attention_heads
LV_LLM_KV_HEADS = 4       # llm_cfg.num_key_value_heads
LV_LLM_VOCAB = 151651     # llm_cfg.vocab_size


def _longvila_tokens_per_frame(height: int, width: int) -> int:
    """Connector-output tokens per frame for LongVILA (SigLIP-So400M @ 448x448
    + mlp_downsample_2x2_fix).

    The SigLIP forward emits ``(H // patch) * (W // patch)`` patches per frame
    (no CLS); the mlp_downsample_2x2_fix connector then 2x2-spatial-packs them
    so the connector input is ``patches * 4 (channels) -> Linear(4608 -> 3584)``
    and the connector OUTPUT is ``patches // 4`` tokens per frame.

    For 448x448 this returns 256 (= (32*32)//4 = LV_VIS_TOKENS_RAW//4).
    For 224x336 this returns (16*24)//4 = 96.
    For 896x448 this returns (64*32)//4 = 512.

    Source: ``base_projector.py:153-161`` in LongVILA-R1-7B (the
    DownSampleBlock Conv2d kernel=2 stride=2 followed by 2-layer MLP) and
    SigLIP patch grid sized at H/14 x W/14.
    """
    raw_patches = (height // LV_VIS_PATCH) * (width // LV_VIS_PATCH)
    return raw_patches // LV_PROJ_DOWNSAMPLE


def _longvila_video_pool(num_video_frames: int) -> int:
    """Temporal pool size used by LongVILA-R1-7B's TSPVideoEncoder.

    AUTHORITATIVE: ``modeling_vila.py`` per-batch override (NOT the
    constructor default). The constructor sets ``pool_sizes = [[8, 1, 1]]``,
    but at every forward pass the per-batch override fires before the
    encoder runs.

    Verbatim from the LongVILA-R1-7B HuggingFace repo:

      # media_encoder.py (TSPVideoEncoder.__init__):
        self.pool_sizes = [[8, 1, 1]]

      # modeling_vila.py (__embed_media_tokens, lines 743-748):
        def round_up_to_bucket(x: int) -> int:
            bucket = 1
            total = 8
            while bucket < total:
                if x <= bucket:
                    return bucket
                bucket *= 2
            return total
        if "video" in name:
            num_video_frames = max([video.shape[0] for video in media[name]])
            if isinstance(self.encoders[name], TSPVideoEncoder):
                self.encoders[name].pool_sizes[0][0] = 4 * round_up_to_bucket(num_video_frames / 256)

    The override sits OUTSIDE the ``if self.training:`` block, so it fires
    in every video forward (training and inference). ``round_up_to_bucket``
    returns the smallest power-of-two bucket in {1, 2, 4, 8} that is >=
    the input, with 8 as the unconditional ceiling. So:
        N <=  256 -> bucket=1, pool_t = 4
        N <=  512 -> bucket=2, pool_t = 8
        N <= 1024 -> bucket=4, pool_t = 16
        N >  1024 -> bucket=8, pool_t = 32

    For N > 512 the same code path additionally splits the video into
    512-frame chunks before encoding, but each chunk is still encoded with
    the same per-batch pool_t set above, and 512 % pool_t == 0 for every
    bucket -- so the temporal-token count after concat is unchanged from
    ``N // pool_t`` (e.g. N=1024 -> 2 chunks of 512 / pool_t=16 -> 64 total).
    """
    if num_video_frames <= 0:
        return 4
    if num_video_frames <= 256:
        bucket = 1
    elif num_video_frames <= 512:
        bucket = 2
    elif num_video_frames <= 1024:
        bucket = 4
    else:
        bucket = 8
    return 4 * bucket


def flops_longvila(
    frames: list[dict],
    n_in_text_tokens: int,
    n_out_text_tokens: int,
    is_video: bool = True,
) -> dict:
    """LongVILA (7B) FLOPs — SigLIP-So400M + mlp_downsample_2x2_fix + Qwen2-7B.

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function does NOT internally snap (LongVILA's
    SigLIP override resizes inputs to a fixed image_size=448 inside the
    preprocessor; arbitrary H, W simply scale the patch grid via integer
    division). For best fidelity pass H, W as multiples of LV_VIS_PATCH=14;
    otherwise the function silently truncates fractional patches via floor
    division. At the canonical 448x448 inputs the ViT runs on 32*32=1024
    patches/frame and the connector emits 256 LLM tokens/frame.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (SigLIP-So400M, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads). No GQA.
    2. Attention scope: FULL N^2 (1024 patches @ 448x448).
    3. Positional embedding: LEARNED absolute (SigLIP standard).
    4. FFN: 2-matmul GELU, intermediate=4304 -> 4*N*H*I.
    5. CLS token: ABSENT (SigLIP avg-pool style; no CLS).
    6. Variable-length packing: NO (each frame is its own ViT forward, since
       LongVILA uses BasicImageEncoder per-frame even from TSPVideoEncoder —
       see media_encoder.py:144-154 where features are computed via
       parent.encode_images on the cat'd frames; conceptually the ViT runs
       on each frame independently with a fixed 1024-patch sequence).
    Connector: spatial 2x2 token pack -> Linear(4608 -> 3584) -> GELU ->
        Linear(3584 -> 3584). (base_projector.py:153-161)
    Video path: TSPVideoEncoder applies a dynamic temporal pool AFTER the
    projector; the ViT and connector still see every frame.
    """
    n_frames = len(frames)

    # ---- Vision encoder: SigLIP-So400M, run per frame regardless of pooling ----
    f0 = frames[0] if frames else {"height": LV_VIS_IMG, "width": LV_VIS_IMG}
    raw_patches_per_frame = _patches_for(f0, LV_VIS_PATCH)
    vision_per_frame = (
        _mha_layer_flops(
            seq_len=raw_patches_per_frame,
            hidden=LV_VIS_HIDDEN,
            n_heads=LV_VIS_HEADS,
            n_kv_heads=LV_VIS_HEADS,
            ffn=LV_VIS_FFN,
            mlp_kind="gelu",
        )
        * LV_VIS_LAYERS
    )
    vision_flops = vision_per_frame * n_frames

    # ---- Connector (mlp_downsample_2x2_fix) ----
    # Spatial 2x2 patch pack -> 4*hidden, then 2-layer MLP (4608 -> 3584 -> 3584).
    # Connector runs BEFORE temporal pool. Token count = patches_per_frame//4
    # per frame (NOT a fixed 256 -- caller H, W are honoured here too).
    proj_in = 4 * LV_VIS_HIDDEN  # 4608
    tokens_per_frame_post_conn = _longvila_tokens_per_frame(
        f0["height"], f0["width"])  # raw_patches_per_frame // 4
    pre_pool_tokens = tokens_per_frame_post_conn * n_frames
    connector_flops = (
        2 * pre_pool_tokens * proj_in * LV_LLM_HIDDEN
        + 2 * pre_pool_tokens * LV_LLM_HIDDEN * LV_LLM_HIDDEN
    )

    # ---- Apply TSP temporal pool (post-projection) ----
    if is_video and n_frames > 0:
        pool_t = _longvila_video_pool(n_frames)  # 4/8/16/32 (bucketed in N)
        # Pool reduces the temporal axis by `pool_t` via mean. The code's
        # `view(-1, pool_t,...)` requires N % pool_t == 0; for N < pool_t this
        # would error at inference, so we apply a defensive max(1, N//pool_t)
        # clamp to keep FLOPs accounting sane at edge N values.
        # E.g. N=8 @ 448 -> pool_t=4 -> n_temporal=2 -> 2*256=512 LLM tokens.
        #      N=1024 @ 448 -> pool_t=16 -> n_temporal=64 -> 64*256=16384 LLM tokens.
        n_temporal = max(1, n_frames // pool_t)
        n_visual_llm_tokens = n_temporal * tokens_per_frame_post_conn
    else:
        n_visual_llm_tokens = pre_pool_tokens

    # ---- LLM prefill + decode ----
    n_prefill = n_visual_llm_tokens + n_in_text_tokens
    prefill_flops = _llm_prefill_flops(
        seq_len=n_prefill,
        hidden=LV_LLM_HIDDEN,
        n_heads=LV_LLM_HEADS,
        n_kv_heads=LV_LLM_KV_HEADS,
        ffn=LV_LLM_FFN,
        n_layers=LV_LLM_LAYERS,
        vocab_size=LV_LLM_VOCAB,
        mlp_kind="swiglu",
    )
    decode_flops = _llm_decode_flops(
        n_prefill=n_prefill,
        n_decode=n_out_text_tokens,
        hidden=LV_LLM_HIDDEN,
        n_heads=LV_LLM_HEADS,
        n_kv_heads=LV_LLM_KV_HEADS,
        ffn=LV_LLM_FFN,
        n_layers=LV_LLM_LAYERS,
        vocab_size=LV_LLM_VOCAB,
        tied_embeddings=False,
        mlp_kind="swiglu",
    )

    # ----- Elementwise -----
    vis_per_frame_elem = LV_VIS_LAYERS * _siglip_block_elem(
        raw_patches_per_frame, LV_VIS_HIDDEN, LV_VIS_HEADS, LV_VIS_FFN)
    vision_elem = vis_per_frame_elem * n_frames
    # Connector: 2x2 token pack -> Linear(4608->3584) -> GELU -> Linear(3584->3584)
    # Biases on Linears; act between them. Applied to pre-pool tokens.
    conn_elem = (bias_flops(pre_pool_tokens, LV_LLM_HIDDEN)
                 + gelu_exact_flops(pre_pool_tokens, LV_LLM_HIDDEN)
                 + bias_flops(pre_pool_tokens, LV_LLM_HIDDEN))
    # LLM (Qwen2-7B): qkv_bias=True, no qk_norm.
    llm_pre_elem = LV_LLM_LAYERS * _llm_dense_block_elem_prefill(
        n_prefill, LV_LLM_HIDDEN, LV_LLM_HEADS, LV_LLM_KV_HEADS,
        LV_LLM_HIDDEN // LV_LLM_HEADS, LV_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem = LV_LLM_LAYERS * _llm_dense_block_elem_decode(
        n_prefill, n_out_text_tokens, LV_LLM_HIDDEN, LV_LLM_HEADS, LV_LLM_KV_HEADS,
        LV_LLM_HIDDEN // LV_LLM_HEADS, LV_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, LV_LLM_HIDDEN)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, LV_LLM_VOCAB)
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill_flops + decode_flops
    return {
        "vision": vision_flops,
        "connector": connector_flops,
        "llm_prefill": prefill_flops,
        "llm_decode": decode_flops,
        "total_flops": total,
        "vision_elementwise": vision_elem,
        "connector_elementwise": conn_elem,
        "llm_prefill_elementwise": llm_pre_elem,
        "llm_decode_elementwise": llm_dec_elem,
        "elementwise_total": elementwise_total,
        "total_with_elementwise": total + elementwise_total,
    }


# --- MiMo-VL --------------------------------------------------------------- #

# All values cited from MiMo-VL-7B-RL/config.json
# (XiaomiMiMo/MiMo-VL-7B-RL).
# vision_config: depth=32, hidden_size=1280, intermediate_size=3456,
# num_heads=16, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
# window_size=112, fullatt_block_indexes=[7,15,23,31], out_hidden_size=4096.
# (MiMo-VL reuses Qwen2_5_VLVisionTransformer.)
# text_config: hidden_size=4096, num_hidden_layers=36, intermediate_size=11008,
# num_attention_heads=32, num_key_value_heads=8, vocab_size=151680.
MM_VIS_HIDDEN = 1280              # vision_config.hidden_size
MM_VIS_FFN = 3456                 # vision_config.intermediate_size
MM_VIS_LAYERS = 32                # vision_config.depth
MM_VIS_HEADS = 16                 # vision_config.num_heads
MM_VIS_PATCH = 14                 # vision_config.patch_size
MM_VIS_SPATIAL_MERGE = 2          # vision_config.spatial_merge_size
MM_VIS_TEMPORAL_PATCH = 2         # vision_config.temporal_patch_size
MM_VIS_WINDOW_SIZE = 112          # vision_config.window_size (pixel-equivalent)
MM_VIS_FULL_BLOCKS = (7, 15, 23, 31)  # vision_config.fullatt_block_indexes
MM_VIS_OUT_HIDDEN = 4096          # vision_config.out_hidden_size

MM_LLM_HIDDEN = 4096              # text_config.hidden_size
MM_LLM_FFN = 11008                # text_config.intermediate_size
MM_LLM_LAYERS = 36                # text_config.num_hidden_layers
MM_LLM_HEADS = 32                 # text_config.num_attention_heads
MM_LLM_KV_HEADS = 8               # text_config.num_key_value_heads
MM_LLM_VOCAB = 151680             # text_config.vocab_size


def flops_mimo_vl(
    frames: list[dict],
    n_in_text_tokens: int,
    n_out_text_tokens: int,
) -> dict:
    """MIMO-VL (7B) FLOPs — Qwen2.5-VL ViT backbone + Qwen2.5 7B LLM.

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function snaps frame[0]'s (H, W) to
    multiples of ``patch_size * spatial_merge_size = 14 * 2 = 28`` INSIDE via
    ``smart_resize`` (Qwen2.5-VL's ``image_processing_qwen2_vl.py:smart_resize``,
    `factor=28`). Caller may pass non-multiple-of-28 input; the FLOPs equation
    uses the snapped (H, W) for both ViT seq length and merger token count.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (Qwen2.5-VL ViT, depth=32, hidden=1280, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads). QKV is one fused Linear
       (qkv: dim -> dim*3). Output proj: dim -> dim. (modeling_qwen2_5_vl.py:222-269)
    2. Attention scope: WINDOWED in 28/32 blocks, FULL in blocks
       {7, 15, 23, 31}. Window size in merged-token units =
           window_size // spatial_merge // patch = 112 // 2 // 14 = 4
       so 4x4 merger tokens = 8x8 = 64 raw patches per window.
       (modeling_qwen2_5_vl.py:438-477,520-531)
    3. Positional embedding: 2D MRoPE (matmul-free).
    4. FFN: SwiGLU (gate, up, down). intermediate=3456 -> 6*N*H*I.
       (modeling_qwen2_5_vl.py:63-76)
    5. CLS token: ABSENT.
    6. Variable-length packing: ViT runs ONCE on the FULL packed sequence of
       length grid_t * grid_h * grid_w. cu_seqlens decides full vs windowed
       attention masking; QKV/O/MLP costs scale with the full packed length.
    PatchMerger: Linear(5120, 5120) -> GELU -> Linear(5120, 4096).
       (modeling_qwen2_5_vl.py:135-148; context_dim=hidden=1280, dim=out_hidden_size=4096)
    Image preprocessor pads odd frame counts by repeating the last frame
    (image_processing_qwen2_vl.py:276-280) -> n_pairs = ceil(n_frames / 2).
    """
    n_frames = len(frames)
    if n_frames == 0:
        n_pairs = 0
    else:
        n_pairs = (n_frames + MM_VIS_TEMPORAL_PATCH - 1) // MM_VIS_TEMPORAL_PATCH

    vision_flops = 0
    n_visual_llm_tokens = 0
    connector_flops = 0
    if n_frames > 0:
        f0 = frames[0]
        # Snap (h, w) to multiples of patch * spatial_merge so the integer
        # divides below match the real Qwen2.5-VL preprocessor's smart_resize
        # (image_processing_qwen2_vl.py:smart_resize, factor=14*2=28).
        h_s, w_s = smart_resize(int(f0["height"]), int(f0["width"]),
                                factor=MM_VIS_PATCH * MM_VIS_SPATIAL_MERGE)
        gh = h_s // MM_VIS_PATCH       # 32 for 448
        gw = w_s // MM_VIS_PATCH        # 32 for 448
        # Total flattened ViT sequence length across all temporal pairs.
        total_seq = n_pairs * gh * gw

        # --- Window geometry (in raw-patch units) ---
        # vit_merger_window_size in merged-token units:
        merger_win = MM_VIS_WINDOW_SIZE // MM_VIS_SPATIAL_MERGE // MM_VIS_PATCH  # 4
        # Each merged token packs spatial_merge_size**2 raw patches.
        spatial_merge_unit = MM_VIS_SPATIAL_MERGE ** 2  # 4
        window_seq = merger_win * merger_win * spatial_merge_unit  # 64
        # llm-side grid (after merge) per pair:
        llm_gh = gh // MM_VIS_SPATIAL_MERGE  # 16
        llm_gw = gw // MM_VIS_SPATIAL_MERGE  # 16
        # number of windows per pair (assumes alignment; for 448x448 + window=4 -> 4x4 = 16 windows/pair)
        # Code pads when not aligned, but our test point divides exactly.
        windows_per_pair = ((llm_gh + merger_win - 1) // merger_win) * (
            (llm_gw + merger_win - 1) // merger_win
        )
        n_windows = windows_per_pair * n_pairs  # 4*4*4 = 64 for 8 frames @ 448

        n_full = sum(1 for li in MM_VIS_FULL_BLOCKS if li < MM_VIS_LAYERS)
        n_win = MM_VIS_LAYERS - n_full

        full_layer = _mha_layer_flops(
            seq_len=total_seq,
            hidden=MM_VIS_HIDDEN,
            n_heads=MM_VIS_HEADS,
            n_kv_heads=MM_VIS_HEADS,
            ffn=MM_VIS_FFN,
            mlp_kind="swiglu",
        )
        win_layer = _windowed_attn_layer_flops(
            total_seq=total_seq,
            window_seq=window_seq,
            n_windows=n_windows,
            hidden=MM_VIS_HIDDEN,
            n_heads=MM_VIS_HEADS,
            n_kv_heads=MM_VIS_HEADS,
            ffn=MM_VIS_FFN,
            mlp_kind="swiglu",
        )
        vision_flops = full_layer * n_full + win_layer * n_win

        # ---- PatchMerger: Linear(5120 -> 5120) -> GELU -> Linear(5120 -> 4096) ----
        merged_tokens = n_pairs * llm_gh * llm_gw  # 4*16*16 = 1024 for 8 frames
        merger_in = MM_VIS_HIDDEN * spatial_merge_unit  # 5120
        connector_flops = (
            2 * merged_tokens * merger_in * merger_in
            + 2 * merged_tokens * merger_in * MM_VIS_OUT_HIDDEN
        )
        n_visual_llm_tokens = merged_tokens

    # ---- LLM prefill + decode ----
    n_prefill = n_visual_llm_tokens + n_in_text_tokens
    prefill_flops = _llm_prefill_flops(
        seq_len=n_prefill,
        hidden=MM_LLM_HIDDEN,
        n_heads=MM_LLM_HEADS,
        n_kv_heads=MM_LLM_KV_HEADS,
        ffn=MM_LLM_FFN,
        n_layers=MM_LLM_LAYERS,
        vocab_size=MM_LLM_VOCAB,
        mlp_kind="swiglu",
    )
    decode_flops = _llm_decode_flops(
        n_prefill=n_prefill,
        n_decode=n_out_text_tokens,
        hidden=MM_LLM_HIDDEN,
        n_heads=MM_LLM_HEADS,
        n_kv_heads=MM_LLM_KV_HEADS,
        ffn=MM_LLM_FFN,
        n_layers=MM_LLM_LAYERS,
        vocab_size=MM_LLM_VOCAB,
        tied_embeddings=False,
        mlp_kind="swiglu",
    )

    # ----- Elementwise (MiMo-VL: Qwen2.5-VL ViT + Qwen2.5 7B LLM) -----
    if n_frames > 0:
        full_elem_per = _qwen25_vit_block_elem(
            total_seq, MM_VIS_HIDDEN, MM_VIS_HEADS, MM_VIS_FFN, window_tokens=None)
        win_elem_per = _qwen25_vit_block_elem(
            total_seq, MM_VIS_HIDDEN, MM_VIS_HEADS, MM_VIS_FFN, window_tokens=window_seq)
        vision_elem = n_full * full_elem_per + n_win * win_elem_per
        # PatchMerger: RMSNorm pre-merger + GELU + biases on Linears
        conn_elem = (rmsnorm_flops(merged_tokens, merger_in)
                     + gelu_tanh_flops(merged_tokens, merger_in)
                     + bias_flops(merged_tokens, merger_in)
                     + bias_flops(merged_tokens, MM_VIS_OUT_HIDDEN))
    else:
        vision_elem = 0
        conn_elem = 0
    # Qwen2.5 LLM: qkv_bias=True, no qk_norm.
    llm_pre_elem = MM_LLM_LAYERS * _llm_dense_block_elem_prefill(
        n_prefill, MM_LLM_HIDDEN, MM_LLM_HEADS, MM_LLM_KV_HEADS,
        MM_LLM_HIDDEN // MM_LLM_HEADS, MM_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem = MM_LLM_LAYERS * _llm_dense_block_elem_decode(
        n_prefill, n_out_text_tokens, MM_LLM_HIDDEN, MM_LLM_HEADS, MM_LLM_KV_HEADS,
        MM_LLM_HIDDEN // MM_LLM_HEADS, MM_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, MM_LLM_HIDDEN)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, MM_LLM_VOCAB)
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill_flops + decode_flops
    return {
        "vision": vision_flops,
        "connector": connector_flops,
        "llm_prefill": prefill_flops,
        "llm_decode": decode_flops,
        "total_flops": total,
        "vision_elementwise": vision_elem,
        "connector_elementwise": conn_elem,
        "llm_prefill_elementwise": llm_pre_elem,
        "llm_decode_elementwise": llm_dec_elem,
        "elementwise_total": elementwise_total,
        "total_with_elementwise": total + elementwise_total,
    }


# --- Phi-4-MM -------------------------------------------------------------- #

# All values cited from microsoft/Phi-4-multimodal-instruct/config.json and
# the bundled vision_siglip_navit.py / processing_phi4mm.py / modeling_phi4mm.py.
# vision (vision_lora_config / SigLIPVisionConfig fields):
#   hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
#   num_attention_heads=16, patch_size=14, image_size=448 (= dyhd_base_resolution).
# embd_layer.image_token_compression_cls='avg_pool_2d' kernel=2 stride=2 -> 4x.
# llm (config.json: text fields):
#   hidden_size=3072, intermediate_size=8192, num_hidden_layers=32,
#   num_attention_heads=24, num_key_value_heads=8, head_dim=128,
#   vocab_size=200064, tie_word_embeddings=true.
# vision LoRA (config.json embd_layer.image_embd_layer.vision_lora):
#   r=256, alpha=512, target_modules=qkv_proj/o_proj/gate_up_proj/down_proj.
PH_VIS_HIDDEN = 1152                 # SigLIPVisionConfig.hidden_size
PH_VIS_FFN = 4304                    # SigLIPVisionConfig.intermediate_size
PH_VIS_LAYERS = 27                   # SigLIPVisionConfig.num_hidden_layers
PH_VIS_HEADS = 16                    # SigLIPVisionConfig.num_attention_heads
PH_VIS_PATCH = 14                    # SigLIPVisionConfig.patch_size
PH_VIS_IMG = 448                     # config.json embd_layer.dyhd_base_resolution
PH_VIS_TOKEN_REDUCTION = 4           # avg_pool_2d kernel=2 stride=2 -> 4x reduction
                                     # (modeling_phi4mm.py:119-122)

PH_LLM_HIDDEN = 3072                 # config.json hidden_size
PH_LLM_FFN = 8192                    # config.json intermediate_size
PH_LLM_LAYERS = 32                   # config.json num_hidden_layers
PH_LLM_HEADS = 24                    # config.json num_attention_heads
PH_LLM_KV_HEADS = 8                  # config.json num_key_value_heads
PH_LLM_VOCAB = 200064                # config.json vocab_size

# Vision LoRA (config.json embd_layer.image_embd_layer.vision_lora):
PH_LORA_R = 256                      # vision_lora.r
PH_LLM_HEAD_DIM = PH_LLM_HIDDEN // PH_LLM_HEADS  # 128 (config: head_dim)
# Phi-4 fuses qkv_proj into one Linear with output dim (n_q + 2*n_kv) * head_dim:
PH_QKV_OUT = (PH_LLM_HEADS + 2 * PH_LLM_KV_HEADS) * PH_LLM_HEAD_DIM  # 5120
# Phi-4 fuses gate+up: one Linear with output dim 2*ffn:
PH_GATE_UP_OUT = 2 * PH_LLM_FFN  # 16384


def _phi4_hd_geometry(height: int, width: int) -> tuple[int, int, int, int]:
    """Replicates dynamic_preprocess for image_size=448.

    Returns (n_sigip_crops, n_sub_crops, n_sub_tokens_total, llm_tokens_per_frame).
      - n_siglip_crops: number of SigLIP forward passes (1 global + sub_crops).
      - n_sub_crops: number of sub-crops (excludes global thumbnail).
      - n_sub_tokens_total: total sub-crop tokens summed across sub-crops AFTER
        avg_pool_2d (each crop contributes 256 tokens for 448-base).
      - llm_tokens_per_frame: tokens placed into the LLM stream by
        modeling_phi4mm.py for this frame, including separators.

    For 448x448 input the dyhd grid is 1x1 (sub_crops = 1) and the function
    builds:
      glb_img: 16 rows of 16 tokens + 16 row-separators = 272 tokens
      sub_img: 16 rows of 16 tokens + 16 row-separators = 272 tokens
      glb_GN : 1 token
      total  : 545 tokens per frame.
    Citation: processing_phi4mm.py:188-244, modeling_phi4mm.py:320-406.
    """
    import math

    base_resolution = PH_VIS_IMG  # 448
    w_crop_num = math.ceil(width / float(base_resolution))
    h_crop_num = math.ceil(height / float(base_resolution))
    n_sub_crops = w_crop_num * h_crop_num
    n_siglip_crops = n_sub_crops + 1  # +1 global thumbnail

    # Each crop yields (image_size/patch)^2 patches -> /4 via avg_pool_2d.
    tokens_per_crop = (base_resolution // PH_VIS_PATCH) ** 2 // PH_VIS_TOKEN_REDUCTION
    # 256 for 448x448.

    # llm_tokens_per_frame: glb (16 rows, 17 cols incl row-sep) + glb_GN (1)
    #                     + sub (h_rows, w_cols of 16x17 tiles per crop).
    H = base_resolution // PH_VIS_PATCH // 2  # 16: post-pool side length
    glb_tokens = H * (H + 1)  # 16*17 = 272
    # For sub_img: a (h*H) x (w*H) grid + (h*H) row separators in the last col
    # + 0 (no extra glb_GN inside sub). Code: useful_height = h*H,
    # temp_sub_GN repeats useful_height rows, sub flat = useful_height*(useful_width+1).
    sub_useful_h = h_crop_num * H
    sub_useful_w = w_crop_num * H
    sub_tokens = sub_useful_h * (sub_useful_w + 1)
    llm_tokens_per_frame = glb_tokens + 1 + sub_tokens
    n_sub_tokens_total = n_sub_crops * tokens_per_crop  # purely informational

    return n_siglip_crops, n_sub_crops, n_sub_tokens_total, llm_tokens_per_frame


def _phi4_lora_flops(seq_total: int) -> int:
    """LoRA cost across all LLM layers for `seq_total` token-positions
    (sum of prefill + per-decode-step counts).

    LoRA target modules per layer: qkv_proj, o_proj, gate_up_proj, down_proj.
    """
    per_token_per_layer = (
        _lora_flops_per_token(PH_LLM_HIDDEN, PH_QKV_OUT, PH_LORA_R)
        + _lora_flops_per_token(PH_LLM_HIDDEN, PH_LLM_HIDDEN, PH_LORA_R)
        + _lora_flops_per_token(PH_LLM_HIDDEN, PH_GATE_UP_OUT, PH_LORA_R)
        + _lora_flops_per_token(PH_LLM_FFN, PH_LLM_HIDDEN, PH_LORA_R)
    )
    return per_token_per_layer * PH_LLM_LAYERS * seq_total


def flops_phi4_mm(
    frames: list[dict],
    n_in_text_tokens: int,
    n_out_text_tokens: int,
) -> dict:
    """Phi-4-MM (5.6B) FLOPs — SigLIP-So400M NaViT + Phi-4 LLM (with vision LoRA).

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function uses ``_phi4_hd_geometry`` to
    compute the dyhd grid as ``(ceil(H/448), ceil(W/448))`` sub-crops + 1 global
    thumbnail. Each crop is INTERNALLY resized by ``dynamic_preprocess`` to
    ``base_resolution=448`` before the SigLIP forward; per-crop ViT sequence
    length is therefore fixed at (448/14)^2 = 1024 regardless of caller H/W
    (citation: ``processing_phi4mm.py:dynamic_preprocess``). Caller H/W only
    affects the NUMBER of crops. Note: this implements naive ``ceil`` rather
    than the upstream ``find_closest_aspect_ratio`` candidate enumeration —
    they agree at 448-multiples and at integer aspect ratios but can diverge
    by 1 sub-crop at irregular aspects (e.g. 1280x720).
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (SigLIP-So400M NaViT, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads). No GQA.
       (vision_siglip_navit.py:641; SiglipAttention is dense MHA.)
    2. Attention scope: FULL N^2 per crop (1024 patches @ 448x448).
    3. Positional embedding: LEARNED absolute (NaViT uses fractional bucketing
       to interpolate the SAME embedding table for variable resolutions, but
       PE is matmul-free either way). (vision_siglip_navit.py:566-603)
    4. FFN: 2-matmul GELU (fc1 -> activation -> fc2). intermediate=4304.
       (vision_siglip_navit.py:905-918)
    5. CLS token: ABSENT (no CLS in SiglipVisionEmbeddings).
    6. Variable-length packing: NO at this test point (each crop is one
       independent forward; HD-transform creates 2 SigLIP crops for 448x448 =
       global thumbnail + 1 sub-crop).
    Token compression: nn.AvgPool2d(2,2) on patch features -> base_feat_height
       halved (32->16). (modeling_phi4mm.py:119-122)
    Projector (image_token_compression=avg_pool_2d -> base_feat_height_reduction=1):
       Linear(1152, 3072) -> GELU -> Linear(3072, 3072).
       (modeling_phi4mm.py:138-148)
    LLM-visible visual tokens for 448x448: 545 per frame (sub 272 + glb_GN 1
       + glb 272), all of which pass through img_projection.
    Vision LoRA (r=256) is always active when an image is present; counted
       explicitly in the LLM cost.
    """
    n_frames = len(frames)

    # ---- Vision encoder: SigLIP-So400M, run per HD-crop per frame ----
    if n_frames > 0:
        f0 = frames[0]
        n_siglip_crops, n_sub_crops, _n_sub_tokens, llm_tokens_per_frame = (
            _phi4_hd_geometry(f0["height"], f0["width"])
        )
        # Per-crop ViT input is ALWAYS the dyhd base_resolution (448) regardless
        # of caller H/W — `dynamic_preprocess` resizes each sub-crop and the
        # global thumbnail to base_resolution * base_resolution before passing
        # to the SigLIP tower. Caller H/W only changes the *number* of crops,
        # not the per-crop sequence length. Citation: processing_phi4mm.py
        # `dynamic_preprocess` resizes to `(base_resolution * w_crop, base_resolution * h_crop)`
        # then splits into base_resolution^2 sub-crops.
        n_patches_per_crop = (PH_VIS_IMG // PH_VIS_PATCH) * (
            PH_VIS_IMG // PH_VIS_PATCH
        )
        vision_per_crop = (
            _mha_layer_flops(
                seq_len=n_patches_per_crop,
                hidden=PH_VIS_HIDDEN,
                n_heads=PH_VIS_HEADS,
                n_kv_heads=PH_VIS_HEADS,
                ffn=PH_VIS_FFN,
                mlp_kind="gelu",
            )
            * PH_VIS_LAYERS
        )
        vision_flops = vision_per_crop * n_siglip_crops * n_frames
    else:
        vision_flops = 0
        n_siglip_crops = 0
        n_sub_crops = 0
        llm_tokens_per_frame = 0

    # ---- Connector: avg_pool_2d (kernel=2,stride=2; no FLOPs) then MLP ----
    # base_feat_height_reduction=1 with avg_pool_2d, so projector input is
    # image_dim_out=1152 (per-token), NOT 4*1152.
    # All llm_tokens_per_frame go through img_projection (sub + glb_GN + glb).
    if n_frames > 0:
        proj_tokens = llm_tokens_per_frame * n_frames
        connector_flops = (
            2 * proj_tokens * PH_VIS_HIDDEN * PH_LLM_HIDDEN
            + 2 * proj_tokens * PH_LLM_HIDDEN * PH_LLM_HIDDEN
        )
        n_visual_llm_tokens = llm_tokens_per_frame * n_frames
    else:
        connector_flops = 0
        n_visual_llm_tokens = 0

    # ---- LLM prefill + decode ----
    n_prefill = n_visual_llm_tokens + n_in_text_tokens
    prefill_flops = _llm_prefill_flops(
        seq_len=n_prefill,
        hidden=PH_LLM_HIDDEN,
        n_heads=PH_LLM_HEADS,
        n_kv_heads=PH_LLM_KV_HEADS,
        ffn=PH_LLM_FFN,
        n_layers=PH_LLM_LAYERS,
        vocab_size=PH_LLM_VOCAB,
        mlp_kind="swiglu",
    )
    decode_flops = _llm_decode_flops(
        n_prefill=n_prefill,
        n_decode=n_out_text_tokens,
        hidden=PH_LLM_HIDDEN,
        n_heads=PH_LLM_HEADS,
        n_kv_heads=PH_LLM_KV_HEADS,
        ffn=PH_LLM_FFN,
        n_layers=PH_LLM_LAYERS,
        vocab_size=PH_LLM_VOCAB,
        tied_embeddings=True,
        mlp_kind="swiglu",
    )

    # ---- Vision LoRA (active when image is present) ----
    # Per-token cost applies to every prefill token AND every decode step's new
    # token (single-token forward). Sum:
    #   prefill: n_prefill tokens
    #   decode : n_out_text_tokens tokens (1 token per step)
    if n_frames > 0:
        lora_flops = _phi4_lora_flops(seq_total=n_prefill + n_out_text_tokens)
    else:
        lora_flops = 0

    # ----- Elementwise (Phi-4-MM: SigLIP NaViT + Phi-4 LLM) -----
    if n_frames > 0:
        vis_per_crop_elem = PH_VIS_LAYERS * _siglip_block_elem(
            n_patches_per_crop, PH_VIS_HIDDEN, PH_VIS_HEADS, PH_VIS_FFN)
        vision_elem = vis_per_crop_elem * n_siglip_crops * n_frames
        # Connector: avg_pool_2d (no FLOPs counted, just memory) + Linear -> GELU -> Linear
        proj_tokens = llm_tokens_per_frame * n_frames
        conn_elem = (bias_flops(proj_tokens, PH_LLM_HIDDEN)
                     + gelu_exact_flops(proj_tokens, PH_LLM_HIDDEN)
                     + bias_flops(proj_tokens, PH_LLM_HIDDEN))
    else:
        vision_elem = 0
        conn_elem = 0
    # Phi-4 LLM: qkv_bias=False (Phi-4 has no qkv bias); no qk_norm.
    llm_pre_elem = PH_LLM_LAYERS * _llm_dense_block_elem_prefill(
        n_prefill, PH_LLM_HIDDEN, PH_LLM_HEADS, PH_LLM_KV_HEADS,
        PH_LLM_HEAD_DIM, PH_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=False)
    llm_dec_elem = PH_LLM_LAYERS * _llm_dense_block_elem_decode(
        n_prefill, n_out_text_tokens, PH_LLM_HIDDEN, PH_LLM_HEADS, PH_LLM_KV_HEADS,
        PH_LLM_HEAD_DIM, PH_LLM_FFN,
        has_qk_norm=False, has_qkv_bias=False)
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, PH_LLM_HIDDEN)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, PH_LLM_VOCAB)
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill_flops + decode_flops + lora_flops
    return {
        "vision": vision_flops,
        "connector": connector_flops,
        "llm_prefill": prefill_flops,
        "llm_decode": decode_flops,
        "vision_lora": lora_flops,
        "total_flops": total,
        "vision_elementwise": vision_elem,
        "connector_elementwise": conn_elem,
        "llm_prefill_elementwise": llm_pre_elem,
        "llm_decode_elementwise": llm_dec_elem,
        "elementwise_total": elementwise_total,
        "total_with_elementwise": total + elementwise_total,
    }


# --- validation ----------------------------------------------------------- #

PFLOP = 1e15


def _fmt(d: dict) -> str:
    """Format a per-component FLOPs dict as a one-liner (used by __main__)."""
    parts = [
        f"vision={d['vision']/PFLOP:.4f} PF",
        f"conn={d['connector']/PFLOP:.4f} PF",
        f"prefill={d['llm_prefill']/PFLOP:.4f} PF",
        f"decode={d['llm_decode']/PFLOP:.4f} PF",
    ]
    if "vision_lora" in d:
        parts.append(f"lora={d['vision_lora']/PFLOP:.4f} PF")
    parts.append(f"TOTAL={d['total_flops']/PFLOP:.4f} PF")
    return " | ".join(parts)


if __name__ == "__main__":
    frames = [{"height": 448, "width": 448}] * 8
    n_in_text = 128
    n_out_text = 64

    print("== Validation: 8 frames at 448x448, 128 in-text, 64 out-text ==")

    lv_img = flops_longvila(frames, n_in_text, n_out_text, is_video=False)
    print(f"LongVILA (image path, no pool) | {_fmt(lv_img)}")

    lv_vid = flops_longvila(frames, n_in_text, n_out_text, is_video=True)
    print(f"LongVILA (video, TSP pool=8)   | {_fmt(lv_vid)}")

    mm = flops_mimo_vl(frames, n_in_text, n_out_text)
    print(f"MiMo-VL  (7B)                  | {_fmt(mm)}")

    ph = flops_phi4_mm(frames, n_in_text, n_out_text)
    print(f"Phi-4-MM (5.6B)                | {_fmt(ph)}")
