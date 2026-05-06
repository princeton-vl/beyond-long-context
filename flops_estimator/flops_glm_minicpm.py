"""
First-principles FLOPs equations for:
  - GLM-4.5V       (zai-org/GLM-4.5V)            -- MoE, 104B-A12B
  - MiniCPM-V 4.5  (openbmb/MiniCPM-V-4_5)       -- 9B, SigLIP + 3D resampler + Qwen3-8B
  - MiniCPM-V 2.6  (openbmb/MiniCPM-V-2_6)       -- 8B, SigLIP + 2D resampler + Qwen2-7B

Methodology
-----------
Counts MATMUL FLOPs only, with the 2*a*b*c convention for an (a x b) @ (b x c)
matmul. Activations / softmax / norms / RoPE phase ops / embedding lookups are
excluded -- they're <1% of the matmul total at these scales and not what people
quote when comparing transformers.

For each model we return a dict:
  {
    "vision":       FLOPs in the ViT (per-frame ViT * num frames),
    "connector":    FLOPs in the projector / resampler / merger,
    "llm_prefill":  FLOPs to encode (vision_tokens_to_LLM + n_in_text_tokens),
    "llm_decode":   FLOPs for n_out_text_tokens autoregressive steps
                    (each step over the growing KV cache, summed),
    "total_flops":  sum of the above.
  }

References:
  GLM-4.5V config:      https://huggingface.co/zai-org/GLM-4.5V/raw/main/config.json
  GLM-4.5V modeling:    transformers/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py
  MiniCPM-V 4.5 config: https://huggingface.co/openbmb/MiniCPM-V-4_5/raw/main/config.json
  MiniCPM-V 4.5 model:  modeling_minicpmv.py + resampler.py from that repo
  MiniCPM-V 2.6:        same files in openbmb/MiniCPM-V-2_6 (gated; 2.6 architecture
                        is documented as identical to 4.5 except the LLM is Qwen2-7B
                        and resampler is 2D = one cross-attn pack per frame).

================================================================================
CHANGES from previous pass (each cites the verifying source):
================================================================================

[1] GLM-4.5V ViT FFN: SwiGLU, NOT 2-matrix GELU.
    Source: modeling_glm4v_moe.py:Glm4vMoeisionMlp (lines 499-510). Has
    gate_proj + up_proj + down_proj with `act_fn(gate) * up`.  Vision config
    `hidden_act: "silu"` confirms SiLU/SwiGLU.
    Effect: vision FFN cost goes 2*2*N*H*F  ->  2*3*N*H*F  (+50%).
    `vit_ffn_kind` flipped from "gelu_2mat" to "swiglu".

[2] MiniCPM Resampler is a SINGLE cross-attn block with NO FFN.
    Source: openbmb/MiniCPM-V-4_5/resampler.py:Resampler (lines 83-241).
    Architecture per call:
      kv_proj: Linear(kv_dim=1152 -> embed_dim) on L kv-tokens
      ln_q, ln_kv (norms, ignored)
      attn = nn.MultiheadAttention(embed_dim, num_heads)
        -> in_proj_weight does Q,K,V projections (3 x embed_dim^2)
           Cross-attn (q!=k): Q proj on n_q tokens; K,V proj on L tokens each.
        -> attention QK^T and (.)V on n_q queries vs L kv-tokens.
        -> out_proj: embed_dim -> embed_dim on n_q tokens.
      proj: parameter (embed_dim x embed_dim), x @ proj on n_q tokens.
    NO MLP. Previous "FFN dim 4x" was wrong.
    For 4.5 (`batch_3d_resampler=true`), KV is concatenated across pack frames;
    one MHA call per pack with L = pack_size * patches_per_frame.

[3] GLM-4.5V vision -> LLM connector is a Conv2d downsample + SwiGLU MLP, NOT
    a (4*1536 -> 4096 -> 4096) 2-layer GELU MLP.
    Source: modeling_glm4v_moe.py:Glm4vMoeVisionModel.__init__ (lines 778-788)
    and forward (lines 870-878):
      post_layernorm
      view -> permute -> downsample = nn.Conv2d(in=1536, out=4096,
                                                kernel=2, stride=2)
        Input: (#patches/4, 1536, 2, 2). Output: (#patches/4, 4096).
        FLOPs per output token: 2 * 4096 * (1536*2*2) = 2 * 4096 * 6144.
      merger = Glm4vMoeVisionPatchMerger(dim=4096, context_dim=10944) :
        proj:        Linear(4096 -> 4096)
        post_norm + GELU (free)
        gate_proj:   Linear(4096 -> 10944)
        up_proj:     Linear(4096 -> 10944)
        down_proj:   Linear(10944 -> 4096)
    Applied per merged token (count = total_patches / spatial_merge^2).

[4] MiniCPM-V 4.5 batched 3D resampler: confirmed 6-frames-per-pack -> 64
    LLM tokens per pack.
    Source: config.json `batch_3d_resampler: true`, `query_num: 64`.
    Code path: resampler.py:batch_attn_forward (lines 248-280) merges k,v
    across temporal_ids groups; merge size = len(tp) frames per group. The
    physical pack size at inference is set by the processing pipeline: video
    is segmented into 6-frame packs (the standard MiniCPM-V 4.5 video pipeline,
    consistent with the public 3D-resampler description). One MHA call per pack
    with L = pack_size * patches_per_frame.

[5] GLM-4.5V LLM block structure (verified):
    Source: modeling_glm4v_moe.py:Glm4vMoeTextDecoderLayer (lines 352-365)
       if layer_idx >= config.first_k_dense_replace:  -> MoE
       else:                                           -> dense MLP
    Config: first_k_dense_replace=1, num_hidden_layers=46  =>  1 dense + 45 MoE.
    Glm4vMoeTextMoE (lines 259-312): top-k experts + ONE shared expert.
       shared_experts = Glm4vMoeTextMLP(intermediate_size =
                          moe_intermediate_size * n_shared_experts)
       => with n_shared_experts=1, shared FFN dim = moe_intermediate_size = 1408.
    Glm4vMoeTextNaiveMoe (lines 220-256): each routed expert is SwiGLU
       (gate_up_proj packs gate+up; chunk(2); act(gate)*up; down_proj).
    Glm4vMoeTextMLP (dense, lines 315-328): SwiGLU, intermediate=10944.
    Attention has NO Q/K-norm (config `use_qk_norm: false`; modeling code does
       not call any q_norm/k_norm).

[6] GLM ViT spatial+temporal merge (verified):
    Source: modeling_glm4v_moe.py:Glm4vMoeVisionPatchEmbed (lines 513-530):
       Conv3d kernel=[temporal_patch_size, patch_size, patch_size]
       => temporal_patch_size=2 collapses pairs of frames at the patch-embed
          stage (the ViT processes N_frames/2 temporal patches; processor
          duplicates a single frame to keep parity).
    Vision model.forward (lines 872-876): hidden_states is reshaped into
       (N_patches/spatial_merge^2, hidden, sm, sm) and downsampled by
       spatial_merge^2 in one Conv2d step.
    LLM-side tokens per video = (N_frames/temporal_patch_size) *
                                 (h/patch/spatial_merge) *
                                 (w/patch/spatial_merge).
    For an 8-frame video at 448x448:
       = (8/2) * (448/14/2)^2 = 4 * 16^2 = 1024 tokens to the LLM.
"""

from __future__ import annotations

from .elementwise import (
    rmsnorm_flops, layernorm_flops, residual_flops, bias_flops,
    rope_flops, rope_flops_decode,
    softmax_flops_attention, softmax_flops_attention_chunks,
    softmax_flops_decode,
    silu_flops, gelu_exact_flops,
    moe_router_flops, moe_combine_flops,
    lm_head_softmax_decode,
)
from ._resize_helpers import smart_resize, minicpm_slice_geometry


# ============================================================================
# Helpers
# ============================================================================

def _vit_block_flops(seq_len: int, hidden: int, n_heads: int, ffn: int,
                     ffn_kind: str = "swiglu",
                     attn_seqlens: list[int] | None = None) -> float:
    """ViT block matmul FLOPs at sequence length N.

    Standard ViT block (per-token projections, FFN, output proj):
      QKV proj (fused): 2 * N * H * (3 * H)
      Out proj:         2 * N * H * H
      FFN swiglu:       3 * 2 * N * H * F
      FFN gelu_2mat:    2 * 2 * N * H * F

    Attention core: sum over independent chunks (cu_seqlens), each contributing
    full N_chunk^2 * 2H per matmul (QK^T and (.)V). When `attn_seqlens` is None
    the whole sequence is one block (vanilla ViT), otherwise each segment is
    attended independently (var-len packed attention; e.g. GLM-4.5V).
    """
    qkv = 2.0 * seq_len * hidden * (3 * hidden)
    o = 2.0 * seq_len * hidden * hidden
    if attn_seqlens is None:
        attn = 2.0 * (2.0 * hidden * seq_len * seq_len)
    else:
        # 2 matmuls (QK^T, AV) summed over independent chunks; cost per chunk
        # is 2 * H * Lc^2.  Same heads/head_dim convention as a flat block.
        attn = 0.0
        for Lc in attn_seqlens:
            attn += 2.0 * (2.0 * hidden * Lc * Lc)
    if ffn_kind == "swiglu":
        ffn_f = 2.0 * 3.0 * seq_len * hidden * ffn
    else:  # GELU 2-matrix
        ffn_f = 2.0 * 2.0 * seq_len * hidden * ffn
    return qkv + o + attn + ffn_f


def _llm_step_flops_dense(cache_len: int,
                          hidden: int,
                          n_heads: int,
                          n_kv_heads: int,
                          head_dim: int,
                          ffn: int,
                          ffn_kind: str = "swiglu") -> float:
    """Per-token decode FLOPs for ONE dense transformer layer (GQA-aware)."""
    H = hidden
    nh = n_heads
    nkv = n_kv_heads
    dh = head_dim
    q = 2.0 * H * nh * dh
    kv = 2.0 * H * 2 * nkv * dh
    o = 2.0 * (nh * dh) * H
    attn = 2.0 * (2.0 * nh * dh * cache_len)
    if ffn_kind == "swiglu":
        ffn_f = 2.0 * 3.0 * H * ffn
    else:
        ffn_f = 2.0 * 2.0 * H * ffn
    return q + kv + o + attn + ffn_f


def _llm_step_flops_moe(cache_len: int,
                        hidden: int,
                        n_heads: int,
                        n_kv_heads: int,
                        head_dim: int,
                        n_routed_experts: int,
                        n_active_experts: int,
                        moe_inter: int,
                        n_shared_experts: int,
                        shared_inter: int) -> float:
    """Per-token decode FLOPs for ONE MoE layer (SwiGLU experts).

    Shared experts always-active; routed top-k active.
    """
    H = hidden
    nh = n_heads
    nkv = n_kv_heads
    dh = head_dim
    q = 2.0 * H * nh * dh
    kv = 2.0 * H * 2 * nkv * dh
    o = 2.0 * (nh * dh) * H
    attn = 2.0 * (2.0 * nh * dh * cache_len)
    router = 2.0 * H * n_routed_experts
    experts = n_active_experts * (2.0 * 3.0 * H * moe_inter)
    shared = (2.0 * 3.0 * H * shared_inter) if n_shared_experts > 0 else 0.0
    return q + kv + o + attn + router + experts + shared


def _sum_attn_prefill(N: int) -> float:
    """Sum_{q=1..N} q  (causal attention work over a length-N prefill;
    each query q attends to its own row plus q-1 prior keys)."""
    return 0.5 * N * (N + 1)


def _sum_attn_decode(N: int, S: int) -> float:
    """Sum_{t=0..S-1} (N + t + 1)  (causal attention work over S decode
    steps with a KV cache of length N at step 0)."""
    return S * N + 0.5 * S * (S + 1)


# ---------------------------------------------------------------------------
# Elementwise per-block helpers (see flops_estimator/elementwise.py)
# ---------------------------------------------------------------------------

def _glm_vit_block_elem(N: int, attn_seqlens, hidden: int, n_heads: int,
                        ffn: int) -> float:
    """GLM-4.5V ViT block elementwise: RMSNorm pre-attn/pre-FFN, 2D RoPE on
    Q/K, var-len softmax via cu_seqlens, SwiGLU SiLU + gate*up.

    QKV/proj BIASES: the GLM-4.5V vision config sets ``attention_bias: false``
    and ``Glm4vMoeVisionAttention`` instantiates ``self.proj = nn.Linear(
    .., bias=False)`` unconditionally. So neither the fused QKV nor the
    output projection has a bias term — they are not counted here.
    Source: ``transformers/.../modeling_glm4v_moe.py:Glm4vMoeVisionAttention``;
    ``zai-org/GLM-4.5V/config.json`` ``vision_config.attention_bias=false``.
    SwiGLU MLP biases are also zero (``Glm4vMoeisionMlp`` Linear bias=False).
    """
    head_dim = hidden // n_heads
    norms = 2 * rmsnorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    rope = rope_flops(N, head_dim, n_heads, n_heads)
    attn_sm = softmax_flops_attention_chunks(attn_seqlens, n_heads)
    act = silu_flops(N, ffn)
    gateup_mul = N * ffn
    return norms + residuals + rope + attn_sm + act + gateup_mul


def _siglip_vit_block_elem(N: int, hidden: int, n_heads: int, ffn: int) -> float:
    """SigLIP-style ViT block: LayerNorm, qkv_bias=True, learned PE (no RoPE),
    2-mat GELU exact (SigLIP uses gelu, not gelu_pytorch_tanh)."""
    norms = 2 * layernorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_bias = bias_flops(N, 3 * hidden)
    o_bias = bias_flops(N, hidden)
    attn_sm = softmax_flops_attention(N, n_heads)
    act = gelu_exact_flops(N, ffn)
    ffn_bias = bias_flops(N, ffn) + bias_flops(N, hidden)
    return norms + residuals + qkv_bias + o_bias + attn_sm + act + ffn_bias


def _llm_block_elem_prefill_dense(N: int, hidden: int, n_q: int, n_kv: int,
                                  head_dim: int, ffn: int,
                                  *, has_qk_norm: bool, has_qkv_bias: bool) -> float:
    """Dense LLM block elementwise (RMSNorm Qwen-style)."""
    norms = 2 * rmsnorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    qk_norm = (rmsnorm_flops(N, head_dim) * (n_q + n_kv)) if has_qk_norm else 0
    rope = rope_flops(N, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_attention(N, n_q)
    ffn_elem = silu_flops(N, ffn) + N * ffn
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem


def _llm_block_elem_prefill_moe(N: int, hidden: int, n_q: int, n_kv: int,
                                head_dim: int, n_routed: int, k_active: int,
                                moe_inter: int, has_shared: bool,
                                shared_inter: int,
                                *, has_qk_norm: bool, has_qkv_bias: bool) -> float:
    """MoE LLM block elementwise (GLM-4.5V style: RMSNorm + RoPE + softmax +
    router + top-k SwiGLU experts + optional shared expert)."""
    norms = 2 * rmsnorm_flops(N, hidden)
    residuals = 2 * residual_flops(N, hidden)
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    qk_norm = (rmsnorm_flops(N, head_dim) * (n_q + n_kv)) if has_qk_norm else 0
    rope = rope_flops(N, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_attention(N, n_q)
    router = moe_router_flops(N, n_routed, k_active)
    per_expert = silu_flops(N, moe_inter) + N * moe_inter
    experts = k_active * per_expert
    combine = moe_combine_flops(N, hidden, k_active)
    shared = (silu_flops(N, shared_inter) + N * shared_inter) if has_shared else 0
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + router + experts + combine + shared


def _llm_block_elem_decode_dense(N_in: int, n_out: int, hidden: int, n_q: int,
                                 n_kv: int, head_dim: int, ffn: int,
                                 *, has_qk_norm: bool, has_qkv_bias: bool) -> float:
    """Dense LLM block elementwise summed over `n_out` decode steps with
    KV cache prefilled to length N_in."""
    if n_out <= 0:
        return 0
    norms = 2 * rmsnorm_flops(1, hidden) * n_out
    residuals = 2 * residual_flops(1, hidden) * n_out
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = (bias_flops(1, qkv_dim) * n_out) if has_qkv_bias else 0
    qk_norm = ((rmsnorm_flops(1, head_dim) * (n_q + n_kv)) * n_out) if has_qk_norm else 0
    rope = rope_flops_decode(n_out, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_decode(N_in, n_out, n_q)
    ffn_elem = silu_flops(n_out, ffn) + n_out * ffn
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem


def _llm_block_elem_decode_moe(N_in: int, n_out: int, hidden: int, n_q: int,
                               n_kv: int, head_dim: int, n_routed: int,
                               k_active: int, moe_inter: int,
                               has_shared: bool, shared_inter: int,
                               *, has_qk_norm: bool, has_qkv_bias: bool) -> float:
    """MoE LLM block elementwise summed over `n_out` decode steps with
    KV cache prefilled to length N_in."""
    if n_out <= 0:
        return 0
    norms = 2 * rmsnorm_flops(1, hidden) * n_out
    residuals = 2 * residual_flops(1, hidden) * n_out
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    qkv_bias = (bias_flops(1, qkv_dim) * n_out) if has_qkv_bias else 0
    qk_norm = ((rmsnorm_flops(1, head_dim) * (n_q + n_kv)) * n_out) if has_qk_norm else 0
    rope = rope_flops_decode(n_out, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_decode(N_in, n_out, n_q)
    router = moe_router_flops(n_out, n_routed, k_active)
    per_expert = silu_flops(1, moe_inter) + moe_inter
    experts = k_active * per_expert * n_out
    combine = moe_combine_flops(n_out, hidden, k_active)
    shared = ((silu_flops(1, shared_inter) + shared_inter) * n_out) if has_shared else 0
    return norms + residuals + qkv_bias + qk_norm + rope + attn_sm + router + experts + combine + shared


# ============================================================================
# 1) GLM-4.5V (104B-A12B MoE)
# ============================================================================

GLM45V = dict(
    # Vision (ViT)
    vit_layers=24,
    vit_hidden=1536,
    vit_heads=12,
    vit_ffn=10944,
    vit_patch=14,
    vit_tile=336,
    spatial_merge=2,
    temporal_patch=2,
    vit_ffn_kind="swiglu",   # CHANGED [1]: SiLU/SwiGLU per Glm4vMoeisionMlp
    out_hidden=4096,
    # Text LLM
    n_layers=46,
    first_k_dense=1,
    hidden=4096,
    n_heads=96,
    n_kv=8,
    head_dim=128,
    dense_ffn=10944,
    moe_inter=1408,
    n_routed=128,
    k_active=8,
    n_shared=1,
    vocab=151552,
)


def _glm_snap(h: int, w: int, cfg: dict) -> tuple[int, int]:
    """Snap (h, w) to multiples of (vit_patch * spatial_merge) per
    transformers' smart_resize. GLM-4.5V's preprocessor (Glm4vMoeImageProcessor
    in transformers) reuses the same smart_resize body as the Qwen2/3-VL
    family with factor=patch_size*spatial_merge_size=28."""
    factor = cfg["vit_patch"] * cfg["spatial_merge"]
    return smart_resize(h, w, factor=factor)


def _glm_vit_seq_len_per_frame(h: int, w: int, cfg: dict) -> int:
    """Spatial patches per ViT temporal-patch (one slot in Conv3d output).

    Caller-provided (h, w) is first snapped via smart_resize so we never
    silently truncate fractional patches when the input isn't a multiple of
    patch * spatial_merge.
    """
    h_s, w_s = _glm_snap(h, w, cfg)
    p = cfg["vit_patch"]
    return (h_s // p) * (w_s // p)


def _glm_llm_tokens(n_frames: int, h: int, w: int, cfg: dict) -> int:
    """LLM-side vision tokens after temporal+spatial merge.

    temporal_patch_size collapses pairs of frames at the Conv3d patch embed;
    spatial_merge^2 downsamples spatial dims after the ViT.
    """
    h_s, w_s = _glm_snap(h, w, cfg)
    p = cfg["vit_patch"]
    sm = cfg["spatial_merge"]
    tp = cfg["temporal_patch"]
    nh = (h_s // p) // sm
    nw = (w_s // p) // sm
    n_temporal = max(1, (n_frames + tp - 1) // tp)  # processor pads to even
    return n_temporal * nh * nw


def _glm_temporal_patches(n_frames: int, cfg: dict) -> int:
    """Number of temporal patches for `n_frames` input frames given the GLM-4.5V
    Conv3d patch embed (`temporal_patch=2`). Equivalent to ceil(n_frames / tp);
    `max(1, …)` handles the zero-frame text-only path."""
    tp = cfg["temporal_patch"]
    return max(1, (n_frames + tp - 1) // tp)


def flops_glm45v(frames: list[dict],
                 n_in_text_tokens: int,
                 n_out_text_tokens: int) -> dict:
    """GLM-4.5V (104B, A12B) FLOPs.

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function snaps each frame's (H, W) to
    multiples of ``vit_patch * spatial_merge = 14 * 2 = 28`` INSIDE via
    ``_glm_snap`` -> ``smart_resize`` (transformers'
    ``Glm4vMoeImageProcessor.smart_resize`` reuses the Qwen-family body
    verbatim; ``zai-org/GLM-4.5V/preprocessor_config.json`` inherits
    ``min_pixels=3136``, ``max_pixels=12845056`` — slightly looser than the
    transformers default but post-snap pixel counts at our test points
    (448x448) are far inside both envelopes).
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (GLM4V-MoE ViT, depth=24, hidden=1536, heads=12)
    -------------------------------------------------------------------
    1. Attention type: MHA (12 Q heads, 12 KV heads, head_dim=128). No GQA.
       Source: Glm4vMoeVisionAttention.__init__ — single Linear(H -> 3H) for QKV.
    2. Attention scope: FULL within-chunk, but CHUNKED via cu_seqlens. Each
       temporal-patch slice (h*w tokens) is its own attention block; chunks do
       NOT attend to each other. For 8 frames @ 448x448 (temporal_patch=2,
       patch=14), grid_thw=(t=4, h=32, w=32) -> 4 chunks of 1024 patches each,
       so attention compute = 4 * (2 * H * 1024^2) per matmul (NOT one
       4096^2 block). This is what the [GLM-varlen] CHANGE fixes.
       Source: Glm4vMoeVisionAttention.forward (cu_seq_lens path) +
       Glm4vMoeVisionModel.forward (cu_seqlens construction).
    3. Positional embedding: 2D RoPE on visual Q/K (apply_rotary_pos_emb_vision);
       matmul-free, not counted.
    4. FFN: SwiGLU (Glm4vMoeisionMlp: gate_proj + up_proj + down_proj),
       intermediate=10944 -> 6*N*H*I.
    5. CLS token: ABSENT (no class_embedding parameter; only patch tokens).
    6. Variable-length packing: YES (cu_seqlens, var-len attention per chunk).
    """
    cfg = GLM45V

    # ---------- Vision ViT ----------
    # ViT processes (n_temporal * patches_per_frame) tokens. We assume one
    # uniform frame size; `frames` length = N_input frames; ViT sees ceil(N/2)
    # temporal patches.
    # Text-only short-circuit: matches HF inference (Glm4vMoeModel.forward gates
    # the vision tower on `pixel_values is not None`) so frames=[] runs no ViT.
    if not frames:
        h0 = w0 = cfg["vit_tile"]
        spatial_per_tp = 0
        n_tp = 0
        vit_seq = 0
    else:
        h0 = int(frames[0].get("height", cfg["vit_tile"]))
        w0 = int(frames[0].get("width", cfg["vit_tile"]))
        spatial_per_tp = _glm_vit_seq_len_per_frame(h0, w0, cfg)
        n_tp = _glm_temporal_patches(len(frames), cfg)
        vit_seq = n_tp * spatial_per_tp
    # CHANGED [GLM-varlen]: GLM-4.5V Glm4vMoeVisionAttention is variable-length
    # packed via cu_seqlens. The cu_seqlens are constructed from grid_thw as
    #   cu_seqlens = repeat_interleave(h*w, t).cumsum()
    # i.e. each temporal-patch block is its OWN (h*w)-token attention chunk.
    # For 8 frames @ 448x448 (patch=14, temporal_patch=2), grid_thw=(t=4,h=32,w=32),
    # giving 4 chunks of 1024 patches each. Attention compute is therefore
    #   sum_t  2 * n_layers * 2 * (h*w)^2 * vit_hidden
    # not    2 * n_layers * 2 * (t*h*w)^2 * vit_hidden.
    # Source: modeling_glm4v_moe.py:Glm4vMoeVisionAttention.forward (var-len
    # packing path) and Glm4vMoeVisionModel.forward (cu_seqlens construction).
    attn_seqlens = [spatial_per_tp] * n_tp
    vis_flops = cfg["vit_layers"] * _vit_block_flops(
        seq_len=vit_seq,
        hidden=cfg["vit_hidden"],
        n_heads=cfg["vit_heads"],
        ffn=cfg["vit_ffn"],
        ffn_kind=cfg["vit_ffn_kind"],
        attn_seqlens=attn_seqlens,
    )

    # ---------- Connector: Conv2d downsample + SwiGLU PatchMerger ----------
    # CHANGED [3]: replace previous (4*1536 -> 4096 -> 4096) GELU 2-layer.
    # Number of merged tokens = vit_seq / spatial_merge^2.
    sm = cfg["spatial_merge"]
    H_v = cfg["vit_hidden"]
    H_o = cfg["out_hidden"]
    F_o = cfg["vit_ffn"]  # 10944, used as merger context_dim (= intermediate_size)
    n_merged = vit_seq // (sm * sm)
    # Conv2d kernel=2, stride=2: per output token does 2 * H_o * (H_v * sm * sm)
    conv_flops = n_merged * (2.0 * H_o * (H_v * sm * sm))
    # Merger MLP per merged token:
    #   proj:      Linear(H_o -> H_o)        : 2 * H_o * H_o
    #   gate_proj: Linear(H_o -> F_o)        : 2 * H_o * F_o
    #   up_proj:   Linear(H_o -> F_o)        : 2 * H_o * F_o
    #   down_proj: Linear(F_o -> H_o)        : 2 * F_o * H_o
    merger_per_tok = (2.0 * H_o * H_o
                      + 2.0 * 3.0 * H_o * F_o)
    merger_flops = n_merged * merger_per_tok
    connector_flops = conv_flops + merger_flops
    llm_vis_tokens = n_merged

    # ---------- LLM ----------
    N = llm_vis_tokens + int(n_in_text_tokens)
    S = int(n_out_text_tokens)
    H = cfg["hidden"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv"]
    dh = cfg["head_dim"]

    def per_token_dense_const() -> float:
        return _llm_step_flops_dense(0, H, nh, nkv, dh, cfg["dense_ffn"], "swiglu")

    def per_token_moe_const() -> float:
        return _llm_step_flops_moe(0, H, nh, nkv, dh,
                                   cfg["n_routed"], cfg["k_active"],
                                   cfg["moe_inter"], cfg["n_shared"],
                                   cfg["moe_inter"])  # shared_inter = moe_inter * n_shared = moe_inter

    n_dense = cfg["first_k_dense"]
    n_moe = cfg["n_layers"] - cfg["first_k_dense"]

    # Prefill
    pre_const = N * (n_dense * per_token_dense_const() + n_moe * per_token_moe_const())
    pre_attn = (n_dense + n_moe) * 4.0 * nh * dh * _sum_attn_prefill(N)
    llm_prefill = pre_const + pre_attn + 2.0 * H * cfg["vocab"]

    # Decode
    dec_const = S * (n_dense * per_token_dense_const() + n_moe * per_token_moe_const())
    dec_attn = (n_dense + n_moe) * 4.0 * nh * dh * _sum_attn_decode(N, S)
    llm_decode = dec_const + dec_attn + S * 2.0 * H * cfg["vocab"]

    # ----- Elementwise -----
    vis_elem_per = _glm_vit_block_elem(vit_seq, attn_seqlens, cfg["vit_hidden"],
                                       cfg["vit_heads"], cfg["vit_ffn"])
    vision_elem = cfg["vit_layers"] * vis_elem_per
    # Connector: post_layernorm + Conv2d (with bias) + merger (RMSNorm post + GELU + biases)
    conn_elem = (rmsnorm_flops(vit_seq, H_v)
                 + bias_flops(n_merged, H_o)            # Conv2d bias
                 + rmsnorm_flops(n_merged, H_o)         # merger post_norm
                 + silu_flops(n_merged, F_o) + n_merged * F_o)  # SwiGLU element-wise
    # LLM elementwise: 1 dense + 45 MoE; qk_norm=False; qkv_bias=False; shared=1.
    llm_pre_dense = _llm_block_elem_prefill_dense(
        N, H, nh, nkv, dh, cfg["dense_ffn"],
        has_qk_norm=False, has_qkv_bias=False)
    llm_pre_moe = _llm_block_elem_prefill_moe(
        N, H, nh, nkv, dh,
        cfg["n_routed"], cfg["k_active"], cfg["moe_inter"],
        cfg["n_shared"] > 0, cfg["moe_inter"] * max(1, cfg["n_shared"]),
        has_qk_norm=False, has_qkv_bias=False)
    llm_pre_elem = n_dense * llm_pre_dense + n_moe * llm_pre_moe
    llm_dec_dense = _llm_block_elem_decode_dense(
        N, S, H, nh, nkv, dh, cfg["dense_ffn"],
        has_qk_norm=False, has_qkv_bias=False)
    llm_dec_moe = _llm_block_elem_decode_moe(
        N, S, H, nh, nkv, dh,
        cfg["n_routed"], cfg["k_active"], cfg["moe_inter"],
        cfg["n_shared"] > 0, cfg["moe_inter"] * max(1, cfg["n_shared"]),
        has_qk_norm=False, has_qkv_bias=False)
    llm_dec_elem = n_dense * llm_dec_dense + n_moe * llm_dec_moe
    llm_dec_elem += rmsnorm_flops(S, H)
    llm_dec_elem += lm_head_softmax_decode(S, cfg["vocab"])
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vis_flops + connector_flops + llm_prefill + llm_decode
    return dict(
        vision=vis_flops,
        connector=connector_flops,
        llm_prefill=llm_prefill,
        llm_decode=llm_decode,
        total_flops=total,
        vision_elementwise=vision_elem,
        connector_elementwise=conn_elem,
        llm_prefill_elementwise=llm_pre_elem,
        llm_decode_elementwise=llm_dec_elem,
        elementwise_total=elementwise_total,
        total_with_elementwise=total + elementwise_total,
        meta=dict(
            vit_seq=vit_seq,
            n_temporal_patches=n_tp,
            llm_vision_tokens=llm_vis_tokens,
            prefill_seq_len=N,
            decode_steps=S,
        ),
    )


# ============================================================================
# 2) MiniCPM-V 4.5 (9B): SigLIP + 3D resampler + Qwen3-class 8B LLM
# ============================================================================

MCPM45 = dict(
    # SigLIP ViT  (config.json vision_config: hidden_size=1152,
    # num_hidden_layers=27, num_attention_heads=16, intermediate_size=4304,
    # patch_size=14, image_size=448, hidden_act='gelu_pytorch_tanh' (SigLIP).)
    vit_layers=27,
    vit_hidden=1152,
    vit_heads=16,
    vit_ffn=4304,
    vit_patch=14,
    vit_input_size=448,         # SigLIP scale_resolution; per-slice fallback only
    vit_ffn_kind="gelu_2mat",   # SigLIP standard
    # Multi-slice preprocessing  (image_processing_minicpmv.py __init__:
    # max_slice_nums=9, scale_resolution=448, patch_size=14, slice_mode=True.)
    slice_mode=True,
    max_slice_nums=9,
    scale_resolution=448,
    # Resampler (single cross-attn, NO FFN -- CHANGED [2])
    # resampler.py:Resampler.__init__: num_heads=16, embed_dim=hidden_size,
    # query_num=64, kv_dim=vision_dim. batch_3d_resampler=true -> 6 frames/pack.
    resampler_queries=64,
    resampler_kv_dim=1152,
    resampler_embed=4096,       # = LLM hidden
    resampler_num_heads=16,     # resampler.py constructor (embed/16=256 head_dim)
    resampler_pack=6,           # 6 frames per pack -> 64 LLM tokens (CHANGED [4])
    # LLM (Qwen3-class 8B dense)  (config.json llm_config: hidden_size=4096,
    # num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
    # head_dim=128, intermediate_size=12288, vocab_size=151748.)
    n_layers=36,
    hidden=4096,
    n_heads=32,
    n_kv=8,
    head_dim=128,
    ffn=12288,
    vocab=151748,
)


def _siglip_vit_flops_for_seq(N_vit: int, cfg: dict) -> float:
    """SigLIP ViT matmul FLOPs for a single forward of seq length N_vit."""
    return cfg["vit_layers"] * _vit_block_flops(
        seq_len=N_vit,
        hidden=cfg["vit_hidden"],
        n_heads=cfg["vit_heads"],
        ffn=cfg["vit_ffn"],
        ffn_kind=cfg["vit_ffn_kind"],
    )


def _siglip_per_frame_slices(h: int, w: int, cfg: dict) -> list[tuple[int, int]]:
    """Per-frame list of (h_slice, w_slice) for each SigLIP ViT forward.

    Uses MiniCPM's slice algorithm (image_processing_minicpmv.py:
    get_sliced_grid + slice_image + get_sliced_images, lines 209-280) to decide
    the number of crops and their resolutions. With the canonical 448x448
    input the grid is None (multiple<=1) so the function returns a single
    448x448 SigLIP forward -- matching the previous behaviour. With high-res
    inputs the function returns 1 thumbnail + grid_x*grid_y sub-crops, each
    SigLIP-sized.
    """
    if not cfg.get("slice_mode", True):
        return [(cfg["vit_input_size"], cfg["vit_input_size"])]
    return minicpm_slice_geometry(
        height=int(h), width=int(w),
        scale_resolution=cfg["scale_resolution"],
        patch_size=cfg["vit_patch"],
        max_slice_nums=cfg["max_slice_nums"],
    )


def _siglip_per_frame_flops(cfg: dict, h: int | None = None,
                            w: int | None = None) -> tuple[float, int, int]:
    """SigLIP ViT FLOPs for a single frame at (h, w) using MiniCPM slicing.

    Returns
    -------
    (flops_total, total_patches_summed_over_slices, n_slices)
        flops_total                  -- sum of ViT FLOPs over all slices.
        total_patches_summed_over_slices -- sum of (slice_h//p)*(slice_w//p)
                                            across slices, used by the
                                            elementwise / resampler kv_proj.
        n_slices                     -- 1 (no slicing) or 1+grid_x*grid_y.

    For backwards compatibility, when h/w are not given we fall back to
    cfg["vit_input_size"]^2 (single forward, no slicing).
    """
    if h is None or w is None:
        h = cfg["vit_input_size"]
        w = cfg["vit_input_size"]
    slices = _siglip_per_frame_slices(int(h), int(w), cfg)
    p = cfg["vit_patch"]
    flops_total = 0.0
    total_patches = 0
    for (sh, sw) in slices:
        N_vit = (sh // p) * (sw // p)
        flops_total += _siglip_vit_flops_for_seq(N_vit, cfg)
        total_patches += N_vit
    return flops_total, total_patches, len(slices)


def _resampler_pack_flops(kv_per_pack: int, cfg: dict) -> float:
    """One MHA cross-attn call with n_q=64 queries against kv_per_pack tokens.

    Exact components (from resampler.py:Resampler):
      kv_proj:          2 * L * kv_dim * E
      MHA in_proj Q:    2 * n_q * E * E
      MHA in_proj K:    2 * L   * E * E
      MHA in_proj V:    2 * L   * E * E
      QK^T (sum heads): 2 * n_q * E * L
      (.)V (sum heads): 2 * n_q * E * L
      MHA out_proj:     2 * n_q * E * E
      proj parameter:   2 * n_q * E * E
    """
    n_q = cfg["resampler_queries"]
    E = cfg["resampler_embed"]
    H_kv = cfg["resampler_kv_dim"]
    L = kv_per_pack
    f = (
        2.0 * L * H_kv * E                 # kv_proj
        + 2.0 * n_q * E * E                # Q proj
        + 2.0 * 2 * L * E * E              # K + V proj
        + 2.0 * 2 * n_q * E * L            # QK^T + (.)V
        + 2.0 * n_q * E * E                # MHA out_proj
        + 2.0 * n_q * E * E                # final `proj` parameter
    )
    return f


def flops_minicpmv45(frames: list[dict],
                     n_in_text_tokens: int,
                     n_out_text_tokens: int) -> dict:
    """MiniCPM-V 4.5 (9B) FLOPs — SigLIP-So400M + 3D resampler + Qwen3-class 8B LLM.

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function INTERNALLY runs MiniCPM's slice
    algorithm (``minicpm_slice_geometry``, verbatim port of
    ``image_processing_minicpmv.py: get_sliced_grid + slice_image +
    find_best_resize + get_refine_size + ensure_divide``). At <=448x448 input
    the algorithm returns a single thumbnail forward; at higher resolutions it
    returns 1 thumbnail + grid_x*grid_y sub-crops with each crop's (h, w) snapped
    to multiples of patch=14. ``max_slice_nums=9`` is the published default
    (preprocessor_config.json); the chat template can override this at
    inference but we do not honor a per-call override (caller responsibility
    if that matters).
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (SigLIP-So400M, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads, head_dim=72). No GQA.
    2. Attention scope: FULL N^2 (1024 patches @ 448x448, bidirectional).
    3. Positional embedding: LEARNED absolute (SigLIP standard).
    4. FFN: 2-matmul GELU (fc1 + GELU + fc2), intermediate=4304 -> 4*N*H*I.
    5. CLS token: ABSENT (SigLIP uses average pooling on patch tokens, no CLS).
    6. Variable-length packing: NO (each frame is its own ViT forward).
    Connector: 3D resampler (single MHA cross-attention, no FFN) -> 64 LLM
    tokens per pack of 6 frames.
    """
    cfg = MCPM45
    n_frames = len(frames)

    # ViT per frame, with multi-slice preprocessing per
    # image_processing_minicpmv.py:get_sliced_images (1 thumbnail +
    # grid_x*grid_y sub-crops at high-res; 1 thumbnail-only when no slicing).
    # We sum per-slice FLOPs and per-slice patch counts. patches_per_frame
    # is the *total* patches summed over slices, fed to the resampler kv_proj
    # (the resampler concatenates kv tokens across all slices of all pack
    # frames; the SigLIP token count is L per frame after the per-slice forward).
    per_frame_vit = 0.0
    per_frame_patches: list[int] = []  # one int per frame, summed across slices
    for fr in frames:
        flp, npatch, _nslc = _siglip_per_frame_flops(cfg, fr["height"], fr["width"])
        per_frame_vit += flp
        per_frame_patches.append(npatch)
    vis_flops = per_frame_vit

    # 3D resampler: ceil(n_frames / pack); each pack -> 64 LLM tokens.
    pack = cfg["resampler_pack"]
    if n_frames > 0:
        full_packs = n_frames // pack
        rem = n_frames - full_packs * pack
        n_packs = full_packs + (1 if rem else 0)
        # Per-pack KV length = sum of patches across that pack's frames.
        connector_flops = 0.0
        idx = 0
        for _ in range(full_packs):
            kv_len = sum(per_frame_patches[idx: idx + pack])
            connector_flops += _resampler_pack_flops(kv_len, cfg)
            idx += pack
        if rem:
            kv_len = sum(per_frame_patches[idx: idx + rem])
            connector_flops += _resampler_pack_flops(kv_len, cfg)
    else:
        n_packs = 0
        connector_flops = 0.0
    llm_vis_tokens = n_packs * cfg["resampler_queries"]

    # LLM
    N = llm_vis_tokens + int(n_in_text_tokens)
    S = int(n_out_text_tokens)
    H, nh, nkv, dh, F = (cfg["hidden"], cfg["n_heads"], cfg["n_kv"],
                         cfg["head_dim"], cfg["ffn"])

    def per_token_const() -> float:
        return _llm_step_flops_dense(0, H, nh, nkv, dh, F, "swiglu")

    L = cfg["n_layers"]
    pre_const = L * N * per_token_const()
    pre_attn = L * 4.0 * nh * dh * _sum_attn_prefill(N)
    llm_prefill = pre_const + pre_attn + 2.0 * H * cfg["vocab"]

    dec_const = L * S * per_token_const()
    dec_attn = L * 4.0 * nh * dh * _sum_attn_decode(N, S)
    llm_decode = dec_const + dec_attn + S * 2.0 * H * cfg["vocab"]

    # ----- Elementwise (MiniCPM-V 4.5: SigLIP + Resampler + Qwen3-class 8B) -----
    # Per-slice SigLIP elementwise; sum over each frame's slices.
    vision_elem = 0
    for fr in frames:
        slices = _siglip_per_frame_slices(fr["height"], fr["width"], cfg)
        p = cfg["vit_patch"]
        for (sh, sw) in slices:
            N_vit = (sh // p) * (sw // p)
            vision_elem += cfg["vit_layers"] * _siglip_vit_block_elem(
                N_vit, cfg["vit_hidden"], cfg["vit_heads"], cfg["vit_ffn"])
    # Resampler: kv_proj + LN + cross-attn (1 softmax over L) + out_proj.
    # No FFN. Run once per pack.
    n_q_resamp = cfg["resampler_queries"]
    embed = cfg["resampler_embed"]
    n_heads_resamp = cfg["resampler_num_heads"]  # explicit, from resampler.py
    if n_frames > 0:
        # Per-pack: ln_q (RMS) + ln_kv (RMS) + softmax over L per Q-head per query
        conn_elem = 0
        full_packs = n_frames // cfg["resampler_pack"]
        rem = n_frames - full_packs * cfg["resampler_pack"]
        idx = 0
        pack_kvs: list[int] = []
        for _ in range(full_packs):
            pack_kvs.append(sum(per_frame_patches[idx: idx + cfg["resampler_pack"]]))
            idx += cfg["resampler_pack"]
        if rem:
            pack_kvs.append(sum(per_frame_patches[idx: idx + rem]))
        for L_kv in pack_kvs:
            # ln_q on n_q queries; ln_kv on L_kv. Use LayerNorm (modeling code).
            conn_elem += layernorm_flops(n_q_resamp, embed)
            conn_elem += layernorm_flops(L_kv, embed)
            # softmax over L_kv per Q-head per query token: 5 * L_kv per row.
            conn_elem += 5 * n_q_resamp * L_kv * n_heads_resamp
            # biases on Q,K,V and out_proj (MHA in_proj has bias=True).
            conn_elem += bias_flops(n_q_resamp, embed)
            conn_elem += bias_flops(L_kv, 2 * embed)  # K,V biases
            conn_elem += bias_flops(n_q_resamp, embed)
    else:
        conn_elem = 0
    # LLM (Qwen3-style: qk_norm=True, no qkv_bias).
    llm_pre_elem = L * _llm_block_elem_prefill_dense(
        N, H, nh, nkv, dh, F, has_qk_norm=True, has_qkv_bias=False)
    llm_dec_elem = L * _llm_block_elem_decode_dense(
        N, S, H, nh, nkv, dh, F, has_qk_norm=True, has_qkv_bias=False)
    llm_dec_elem += rmsnorm_flops(S, H)
    llm_dec_elem += lm_head_softmax_decode(S, cfg["vocab"])
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vis_flops + connector_flops + llm_prefill + llm_decode
    return dict(
        vision=vis_flops,
        connector=connector_flops,
        llm_prefill=llm_prefill,
        llm_decode=llm_decode,
        total_flops=total,
        vision_elementwise=vision_elem,
        connector_elementwise=conn_elem,
        llm_prefill_elementwise=llm_pre_elem,
        llm_decode_elementwise=llm_dec_elem,
        elementwise_total=elementwise_total,
        total_with_elementwise=total + elementwise_total,
        meta=dict(
            n_frames=n_frames,
            n_packs=n_packs,
            llm_vision_tokens=llm_vis_tokens,
            prefill_seq_len=N,
            decode_steps=S,
        ),
    )


# ============================================================================
# 3) MiniCPM-V 2.6 (8B): SigLIP + 2D resampler + Qwen2-7B
# ============================================================================
# The canonical openbmb/MiniCPM-V-2_6 repo is gated, but every architecture
# field below is verified against the openbmb/MiniCPM-V-2_6-int4 mirror
# (architecture-identical; only adds a quantization_config block) AND the
# openbmb/MiniCPM-o-2_6 omni sister model (shares vision tower, resampler,
# Qwen2-7B LLM). Resampler runs the SAME single cross-attn block PER FRAME
# (no temporal merging; no batch_attn_forward in 2.6's resampler.py).
# See flops_estimator/glm_minicpm_verification.md for citations.

MCPM26 = dict(
    # SigLIP ViT  (config.json vision_config: identical to MCPM45.)
    vit_layers=27,
    vit_hidden=1152,
    vit_heads=16,
    vit_ffn=4304,
    vit_patch=14,
    vit_input_size=448,
    vit_ffn_kind="gelu_2mat",
    # Multi-slice preprocessing (image_processing_minicpmv.py: same defaults
    # as 4.5). The 2.6 architecture is identical to 4.5 except (a) no batch_3d
    # in the resampler -- i.e. one cross-attn per frame, no temporal merging,
    # and (b) Qwen2-7B as LLM backbone.
    slice_mode=True,
    max_slice_nums=9,
    scale_resolution=448,
    # Resampler (single cross-attn per frame, no FFN, no temporal grouping).
    resampler_queries=64,
    resampler_kv_dim=1152,
    resampler_embed=3584,       # = Qwen2-7B hidden
    resampler_num_heads=16,     # resampler.py constructor (embed/16=224 head_dim)
    # LLM (Qwen2-7B)  (config.json: hidden_size=3584, num_hidden_layers=28,
    # num_attention_heads=28, num_key_value_heads=4, head_dim=128,
    # intermediate_size=18944, vocab_size=151666.)
    n_layers=28,
    hidden=3584,
    n_heads=28,
    n_kv=4,
    head_dim=128,
    ffn=18944,
    vocab=151666,
)


def flops_minicpmv26(frames: list[dict],
                     n_in_text_tokens: int,
                     n_out_text_tokens: int) -> dict:
    """MiniCPM-V 2.6 (8B) FLOPs — SigLIP-So400M + 2D resampler + Qwen2-7B.

    CALLER CONTRACT
    ---------------
    Same slicing contract as MiniCPM-V 4.5: pass any H, W; the function uses
    ``minicpm_slice_geometry`` INSIDE to compute per-frame sub-crop count and
    each crop's (h, w). The 2.6 architecture differs from 4.5 only in the
    resampler (one cross-attn pack PER FRAME, no temporal grouping) and the
    LLM backbone (Qwen2-7B). The slice algorithm itself is identical.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (SigLIP-So400M, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    Identical ViT to flops_minicpmv45: MHA, full N^2, learned absolute PE,
    2-matmul GELU MLP (intermediate=4304), NO CLS, no varlen.
    Differs only in the connector (2D resampler — one cross-attn pack PER
    FRAME, no temporal merging) and the LLM backbone (Qwen2-7B).
    """
    cfg = MCPM26
    n_frames = len(frames)

    # Per-frame multi-slice ViT (1 thumbnail + grid_x*grid_y sub-crops at
    # high-res, 1 forward at canonical 448x448).
    per_frame_vit = 0.0
    per_frame_patches: list[int] = []
    for fr in frames:
        flp, npatch, _nslc = _siglip_per_frame_flops(cfg, fr["height"], fr["width"])
        per_frame_vit += flp
        per_frame_patches.append(npatch)
    vis_flops = per_frame_vit

    # 2D resampler: ONE cross-attn pack PER FRAME (no temporal grouping).
    # KV length per pack = sum of patches across that frame's slices.
    if n_frames > 0:
        connector_flops = sum(_resampler_pack_flops(L, cfg) for L in per_frame_patches)
    else:
        connector_flops = 0.0
    llm_vis_tokens = n_frames * cfg["resampler_queries"]

    N = llm_vis_tokens + int(n_in_text_tokens)
    S = int(n_out_text_tokens)
    H, nh, nkv, dh, F = (cfg["hidden"], cfg["n_heads"], cfg["n_kv"],
                         cfg["head_dim"], cfg["ffn"])

    def per_token_const() -> float:
        return _llm_step_flops_dense(0, H, nh, nkv, dh, F, "swiglu")

    L = cfg["n_layers"]
    pre_const = L * N * per_token_const()
    pre_attn = L * 4.0 * nh * dh * _sum_attn_prefill(N)
    llm_prefill = pre_const + pre_attn + 2.0 * H * cfg["vocab"]

    dec_const = L * S * per_token_const()
    dec_attn = L * 4.0 * nh * dh * _sum_attn_decode(N, S)
    llm_decode = dec_const + dec_attn + S * 2.0 * H * cfg["vocab"]

    # ----- Elementwise (MiniCPM-V 2.6: SigLIP + 2D resampler + Qwen2-7B) -----
    # Per-slice SigLIP elementwise; sum over each frame's slices.
    vision_elem = 0
    for fr in frames:
        slices = _siglip_per_frame_slices(fr["height"], fr["width"], cfg)
        p = cfg["vit_patch"]
        for (sh, sw) in slices:
            N_vit = (sh // p) * (sw // p)
            vision_elem += cfg["vit_layers"] * _siglip_vit_block_elem(
                N_vit, cfg["vit_hidden"], cfg["vit_heads"], cfg["vit_ffn"])
    n_q_resamp = cfg["resampler_queries"]
    embed = cfg["resampler_embed"]
    n_heads_resamp = cfg["resampler_num_heads"]  # explicit, from resampler.py
    if n_frames > 0:
        conn_elem = 0
        for L_kv in per_frame_patches:
            conn_elem += layernorm_flops(n_q_resamp, embed)
            conn_elem += layernorm_flops(L_kv, embed)
            conn_elem += 5 * n_q_resamp * L_kv * n_heads_resamp
            conn_elem += bias_flops(n_q_resamp, embed)
            conn_elem += bias_flops(L_kv, 2 * embed)
            conn_elem += bias_flops(n_q_resamp, embed)
    else:
        conn_elem = 0
    # Qwen2-7B LLM: has qkv_bias=True in Qwen2; no qk_norm.
    llm_pre_elem = L * _llm_block_elem_prefill_dense(
        N, H, nh, nkv, dh, F, has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem = L * _llm_block_elem_decode_dense(
        N, S, H, nh, nkv, dh, F, has_qk_norm=False, has_qkv_bias=True)
    llm_dec_elem += rmsnorm_flops(S, H)
    llm_dec_elem += lm_head_softmax_decode(S, cfg["vocab"])
    elementwise_total = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vis_flops + connector_flops + llm_prefill + llm_decode
    return dict(
        vision=vis_flops,
        connector=connector_flops,
        llm_prefill=llm_prefill,
        llm_decode=llm_decode,
        total_flops=total,
        vision_elementwise=vision_elem,
        connector_elementwise=conn_elem,
        llm_prefill_elementwise=llm_pre_elem,
        llm_decode_elementwise=llm_dec_elem,
        elementwise_total=elementwise_total,
        total_with_elementwise=total + elementwise_total,
        meta=dict(
            n_frames=n_frames,
            llm_vision_tokens=llm_vis_tokens,
            prefill_seq_len=N,
            decode_steps=S,
        ),
    )


# ============================================================================
# Validation
# ============================================================================

def _pf(x: float) -> str:
    """Format raw FLOPs as a PFLOPs string (used by __main__)."""
    return f"{x / 1e15:.3f} PFLOPs"


def _fmt(d: dict) -> str:
    """Pretty-print a per-component FLOPs dict (used by __main__)."""
    return ("  vision      = " + _pf(d["vision"]) + "\n"
            + "  connector   = " + _pf(d["connector"]) + "\n"
            + "  llm_prefill = " + _pf(d["llm_prefill"]) + "\n"
            + "  llm_decode  = " + _pf(d["llm_decode"]) + "\n"
            + "  TOTAL       = " + _pf(d["total_flops"]) + "\n"
            + "  meta        = " + str(d["meta"]))


if __name__ == "__main__":
    frames = [{"height": 448, "width": 448}] * 8
    n_in = 128
    n_out = 64

    print("=" * 70)
    print(f"Validation: 8 frames @ 448x448, n_in_text={n_in}, n_out_text={n_out}")
    print("=" * 70)

    print("\n[1] GLM-4.5V (104B-A12B MoE)")
    print(_fmt(flops_glm45v(frames, n_in, n_out)))

    print("\n[2] MiniCPM-V 4.5 (9B)")
    print(_fmt(flops_minicpmv45(frames, n_in, n_out)))

    print("\n[3] MiniCPM-V 2.6 (8B)")
    print(_fmt(flops_minicpmv26(frames, n_in, n_out)))
