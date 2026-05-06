"""
First-principles FLOPs equations for the Qwen multimodal family.

Built from architecture only (HuggingFace config.json values), no consultation
of any existing FLOPs code. Counts FLOPs from matmuls only:
    FLOPs(MatMul of (a,b) by (b,c)) = 2 * a * b * c
Softmax / norms / activations / element-wise ops are ignored (the only
estimation allowed by the spec).

CHANGES (post-code-verification pass):
  [Qwen2.5-VL] ViT attention: 28 of 32 blocks use *windowed* attention with
      window_size=112 px (=8x8 patches=64 tokens), only blocks
      fullatt_block_indexes=[7,15,23,31] use full attention.
      Source: transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
              `Qwen2_5_VisionTransformerPretrainedModel.forward`
              ("for layer_num, blk in enumerate(self.blocks): ... if layer_num
                in self.fullatt_block_indexes: cu_seqlens_now = cu_seqlens
                else: cu_seqlens_now = cu_window_seqlens").
              The sequence is *partitioned* (`hidden_states[window_index, :, :]`)
              so attention is O(N*W) not O(N^2 with mask). The 28 windowed
              blocks therefore drop from 4*N^2*d to 4*N*W*d for the attn core.
      Magnitude: with N=2048 (8 frames @ 448x448 -> 4 temporal groups *
          512 patches each) and W=64, attn core in windowed blocks shrinks by
          N/W = 32x. ViT FLOPs drop from ~0.0080 PF -> ~0.0049 PF
          (vision_flops -38%).
  [Qwen2.5-VL] ViT MLP: confirmed SwiGLU (gate+up+down).
      Source: `Qwen2_5_VLMLP.__init__` (gate_proj, up_proj, down_proj).
      No change (was already 'swiglu').
  [Qwen2.5-VL] PatchMerger ("merger"): Linear(5120,5120)+GELU+Linear(5120,3584).
      Source: `Qwen2_5_VLPatchMerger.__init__`
              (mlp = Sequential(Linear(self.hidden_size=context_dim*merge^2,
               self.hidden_size), GELU, Linear(self.hidden_size, dim))).
      Confirmed; no change.
  [Qwen2.5-VL] Token count for 448x448: patch=14 -> 32x32=1024 pre-merge
      tokens per spatial grid; spatial_merge=2 -> 16x16=256 post-merge tokens
      per frame-pair; temporal_patch=2 -> ceil(8/2)=4 groups, so 4*1024=4096
      pre-merge ViT tokens and 4*256=1024 post-merge LLM tokens.
      Source: preprocessor_config.json (patch_size=14, merge_size=2,
              temporal_patch_size=2).
      Confirmed; no change.
  [Qwen3-VL] ViT MLP: 2-linear (linear_fc1 + linear_fc2), NOT SwiGLU.
      Source: `Qwen3VLVisionMLP.__init__` lines 57-62.
      Already counted as 'gelu' (4*N*d*m). No change.
  [Qwen3-VL] PatchMerger: Linear(vit_d*merge^2 -> vit_d*merge^2) +
      Linear(vit_d*merge^2 -> out_hidden). With vit_d=1152, merge=2,
      out_hidden=4096: Linear(4608,4608) + Linear(4608,4096).
      Source: `Qwen3VLVisionPatchMerger.__init__` lines 136-143
              (self.hidden_size = config.hidden_size * spatial_merge_size**2;
               linear_fc1 = Linear(hidden_size, hidden_size);
               linear_fc2 = Linear(hidden_size, config.out_hidden_size)).
      Confirmed; no change.
  [Qwen3-VL] DeepStack adapters: same `Qwen3VLVisionPatchMerger` class as
      main merger, only differing by `use_postshuffle_norm=True` (LayerNorm
      placement). Linear dims are IDENTICAL.
      Source: `Qwen3VLVisionModel.__init__` deepstack_merger_list =
              ModuleList([Qwen3VLVisionPatchMerger(config,
                          use_postshuffle_norm=True)
                          for _ in range(len(config.deepstack_visual_indexes))]).
      So 1 main + 3 deepstack = 4 connector projections. Confirmed; no change.
  [Qwen3-Omni Thinker] Pure MoE every layer (decoder_sparse_step=1,
      mlp_only_layers=[]), no shared expert (shared_expert_intermediate_size=0).
      Source: `Qwen3OmniMoeThinkerTextDecoderLayer.__init__`
              ("if (layer_idx not in config.mlp_only_layers) and ...
                self.mlp = Qwen3OmniMoeThinkerTextSparseMoeBlock else
                Qwen3OmniMoeThinkerTextMLP"). With mlp_only_layers=[] and
              decoder_sparse_step=1, every layer hits the MoE branch.
              `Qwen3OmniMoeThinkerTextSparseMoeBlock.__init__` instantiates
              `experts` and `gate` only (no shared_expert attribute; nor is
              one invoked in `forward`).
      Confirmed; no change.
  [Qwen3-Omni Thinker] Expert structure: gate+up fused (gate_up_proj of
      shape (E, 2*moe_ffn, d)) + down (down_proj of shape (E, d, moe_ffn)).
      FLOPs per active expert per token = 2*d*(2*moe_ffn) + 2*moe_ffn*d
                                        = 6*d*moe_ffn (same as SwiGLU).
      Source: `Qwen3OmniMoeThinkerTextExperts.__init__`
              (gate_up_proj: (num_experts, 2*intermediate_dim, hidden_dim);
               down_proj:   (num_experts, hidden_dim, intermediate_dim)).
      Confirmed; no change.
  [Qwen3-Omni Thinker] Router: nn.Parameter((num_experts, hidden_dim)) used
      as a single linear (hidden -> num_experts). FLOPs per token =
      2 * hidden * num_experts.
      Source: `Qwen3OmniMoeThinkerTextTopKRouter.__init__`
              (self.weight = nn.Parameter(zeros(num_experts, hidden_dim))).
      Confirmed; no change.
  [Causal attention convention] We keep full N^2 matmul accounting for the
      attention core, ignoring the upper-triangular zeros from the causal
      mask. This matches the standard 6*N*P (Chinchilla, Hoffmann et al.
      2022) and the attention term in Kaplan/PaLM scaling-law accounting,
      where attention is reported as 4*L*d_model*N^2 (forward) for an L-layer
      transformer of width d_model and seq len N. No fix needed.

Each model returns:
    {
        'vision_flops':       ViT encoder over all frames + patch embedding,
        'connector_flops':    merger / projector from vision dim -> LLM dim,
        'llm_prefill_flops':  one forward pass over all input tokens,
        'llm_decode_flops':   sum over n_out_text generated tokens with KV cache,
        'total_flops':        sum of the four,
    }

Convention used everywhere:
    QKV projection:      input (N, d) -> (N, (n_q + 2*n_kv) * head_dim)
    Causal self-attn:    sum over query positions q=1..N of 2 * q * d_head per head
                         -> total ~ 2 * N^2 * d  (approximating q over N)
                         For prefill we use exactly: n_heads * (N*(N+1)) * head_dim
                                 ~= n_heads * N^2 * head_dim  (forward, two matmuls qk + av)
                         Actually the two attn matmuls each contribute, see below.
    Output projection:   (N, d) -> (N, d)            => 2 * N * d * d
    SwiGLU FFN:          three (d, m) matmuls per token (gate, up, down)
                         => 3 * 2 * N * d * m = 6 * N * d * m
    Standard 2-matmul FFN (used for ViT MLPs that are GELU-style):
                         => 2 * 2 * N * d * m = 4 * N * d * m

Self-attention FLOPs (causal, prefill of length N):
    QK^T per head: queries (N, h_d) x keys^T (h_d, N) = (N, N) -> 2 * N * N * h_d
                   (causal mask is just zeroing; matmul work is 2 N^2 h_d)
    softmax(.) V per head: (N, N) x (N, h_d) -> 2 * N * N * h_d
    Total over all heads (Q heads, since softmax is per query head):
        attn_core = n_q_heads * (2 * N^2 * h_d + 2 * N^2 * h_d) = 4 * n_q_heads * N^2 * h_d
    With GQA the V/K are shared across groups but the softmax-V product is still
    computed per query head, so the cost is the same as MHA for the attn core.
    (We follow the standard "matmul-only" convention.)

For decode step t (1-indexed from 1..n_out), the model attends to current_KV
tokens of length L_t = N + t - 1 (the previous tokens) and generates token t.
    QKV proj for 1 token, attention over L_t+1 keys (~L_t for large N),
    output proj for 1 token, FFN for 1 token.
    Per-step attn core (1 query token, L_t key/value tokens):
        2 * 1 * L_t * h_d + 2 * 1 * L_t * h_d = 4 * L_t * h_d, per head
        => 4 * n_q_heads * L_t * h_d
    Summed over t=1..n_out, sum_t L_t = sum_{t=1..n_out} (N + t - 1)
                                      = n_out * N + n_out*(n_out-1)/2
"""

from __future__ import annotations
from math import ceil

from .elementwise import (
    rmsnorm_flops, layernorm_flops, residual_flops, bias_flops,
    rope_flops, rope_flops_decode,
    softmax_flops_attention, softmax_flops_attention_chunks,
    softmax_flops_attention_windowed, softmax_flops_decode,
    silu_flops, gelu_tanh_flops,
    moe_router_flops, moe_combine_flops,
    lm_head_softmax_decode,
)
from ._resize_helpers import smart_resize


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _vit_block_flops(n_tokens: int, d: int, n_heads: int, ffn_dim: int,
                    ffn_style: str, window_tokens: int | None = None) -> int:
    """FLOPs of one ViT-style transformer block.

    QKV proj:    3 * 2 * N * d * d
    Attn core (full):     4 * n_heads * N^2 * (d / n_heads) = 4 * N^2 * d
    Attn core (windowed): 4 * N * W * d  (each query attends only to W keys)
        Used by Qwen2.5-VL non-fullatt blocks. Source: modeling_qwen2_5_vl.py
        forward partitions tokens by window_index and runs attention per
        cu_window_seqlens chunk, so compute is O(N*W) not O(N^2).
    Out proj:        2 * N * d * d
    FFN:
        'gelu':  4 * N * d * ffn_dim   (two matmuls fc1+fc2)
        'swiglu':6 * N * d * ffn_dim   (three matmuls gate+up+down)
    """
    qkv = 3 * 2 * n_tokens * d * d
    if window_tokens is None or window_tokens >= n_tokens:
        attn_core = 4 * n_tokens * n_tokens * d
    else:
        attn_core = 4 * n_tokens * window_tokens * d
    out_proj = 2 * n_tokens * d * d
    if ffn_style == 'gelu':
        ffn = 4 * n_tokens * d * ffn_dim
    elif ffn_style == 'swiglu':
        ffn = 6 * n_tokens * d * ffn_dim
    else:
        raise ValueError(ffn_style)
    return qkv + attn_core + out_proj + ffn


def _llm_prefill_flops(N: int, n_layers: int, d: int, n_q: int, n_kv: int,
                      head_dim: int, ffn_per_token: int) -> int:
    """FLOPs of LLM prefill over N tokens.

    QKV proj per layer:  2 * N * d * (n_q + 2*n_kv) * head_dim
    Attn core per layer: 4 * n_q * N^2 * head_dim
                         (two matmuls QK^T and AV, each costs 2*n_q*N^2*head_dim)
                         Causal-mask convention: we keep the full N^2 matmul
                         work (ignore the upper-triangular zeros). This matches
                         Chinchilla/Kaplan/PaLM scaling-law accounting where
                         the forward attention term is reported as
                         4 * L * d_model * N^2 for an L-layer transformer.
    Out proj per layer:  2 * N * d * (n_q * head_dim) = 2 * N * d * d_q  (d_q = n_q*head_dim, usually = d)
    FFN per layer:       ffn_per_token * N  (caller provides per-token FFN matmul cost)
    """
    d_q = n_q * head_dim
    d_kv = n_kv * head_dim
    qkv = 2 * N * d * (d_q + 2 * d_kv)
    attn_core = 4 * n_q * N * N * head_dim
    out_proj = 2 * N * d_q * d
    ffn = ffn_per_token * N
    per_layer = qkv + attn_core + out_proj + ffn
    return per_layer * n_layers


def _llm_decode_flops(N: int, n_out: int, n_layers: int, d: int,
                     n_q: int, n_kv: int, head_dim: int,
                     ffn_per_token: int) -> int:
    """FLOPs of LLM decode for n_out generated tokens with KV cache.

    For step t (t=1..n_out), the new query attends over L_t = N + t - 1
    cached KV positions (we count keys/values of past tokens; the new token's
    own K/V is added but is essentially the same as L_t for large N).

    Per step:
        QKV proj for 1 token:  2 * 1 * d * (d_q + 2*d_kv)
        Attn core for 1 token over L_t keys:
            QK^T:  2 * 1 * L_t * head_dim per query head -> 2 * n_q * L_t * head_dim
            AV:    2 * 1 * L_t * head_dim per query head -> 2 * n_q * L_t * head_dim
            total: 4 * n_q * L_t * head_dim
        Out proj:  2 * 1 * d_q * d
        FFN:       ffn_per_token

    Summed over t=1..n_out (and over n_layers).
    """
    if n_out <= 0:
        return 0
    d_q = n_q * head_dim
    d_kv = n_kv * head_dim
    # Constant per-step cost (independent of L_t):
    qkv = 2 * d * (d_q + 2 * d_kv)
    out_proj = 2 * d_q * d
    ffn = ffn_per_token
    constant_per_step = qkv + out_proj + ffn

    # L_t = N + t - 1, t=1..n_out
    sum_L = n_out * N + (n_out * (n_out - 1)) // 2
    attn_core_total = 4 * n_q * head_dim * sum_L

    per_layer = constant_per_step * n_out + attn_core_total
    return per_layer * n_layers


# ---------------------------------------------------------------------------
# Elementwise (norms / softmax / RoPE / activations / biases / residuals)
# ---------------------------------------------------------------------------

def _qwen_vit_elem_block(
    N: int, d: int, ffn: int, n_heads: int, ffn_style: str,
    *, has_qkv_bias: bool, use_rope: bool,
    window_tokens: int | None, attn_seqlens: list | None = None,
) -> int:
    """Elementwise FLOPs for one Qwen ViT block.
    Conventions (verified per code citations in module docstring):
      - LayerNorm pre-attn + pre-FFN (Qwen2.5/Qwen3 ViT use LayerNorm).
      - QKV bias: Qwen2.5-VL ViT qkv_bias=True; Qwen3-VL ViT linear_qkv has bias.
      - O proj bias: True for both.
      - 2D MRoPE on Q,K (matmul cost not counted; elementwise is).
      - SwiGLU (Qwen2.5-VL): SiLU + gate*up mul.
      - 2-mat GELU (Qwen3-VL ViT, gelu_pytorch_tanh): tanh-approx GELU.
    """
    head_dim = d // n_heads
    norms = 2 * layernorm_flops(N, d)
    residuals = 2 * residual_flops(N, d)
    qkv_bias = bias_flops(N, 3 * d) if has_qkv_bias else 0
    o_bias = bias_flops(N, d)  # Qwen ViT proj has bias
    rope = rope_flops(N, head_dim, n_heads, n_heads) if use_rope else 0
    if attn_seqlens is not None:
        attn_sm = softmax_flops_attention_chunks(attn_seqlens, n_heads)
    elif window_tokens is not None and window_tokens < N:
        attn_sm = softmax_flops_attention_windowed(N, window_tokens, n_heads)
    else:
        attn_sm = softmax_flops_attention(N, n_heads)
    if ffn_style == 'swiglu':
        # SiLU activation on gate, plus gate*up element-wise mul; ffn linears
        # have biases on Qwen2.5-VL ViT MLP.
        act = silu_flops(N, ffn)
        gateup_mul = N * ffn
        ffn_bias = bias_flops(N, ffn) * 2 + bias_flops(N, d)  # gate, up, down biases
    else:
        # gelu_pytorch_tanh on Qwen3-VL ViT
        act = gelu_tanh_flops(N, ffn)
        gateup_mul = 0
        ffn_bias = bias_flops(N, ffn) + bias_flops(N, d)  # fc1 + fc2 biases
    return norms + residuals + qkv_bias + o_bias + rope + attn_sm + act + gateup_mul + ffn_bias


def _qwen_llm_elem_prefill(
    N: int, d: int, n_q: int, n_kv: int, head_dim: int, ffn: int,
    *, n_layers: int, has_qk_norm: bool, has_qkv_bias: bool,
    is_moe: bool = False, moe_n: int = 0, moe_k: int = 0, moe_ffn: int = 0,
    has_shared: bool = False, shared_ffn: int = 0,
) -> int:
    """Per-layer elementwise FLOPs for LLM prefill, summed over n_layers.
    All Qwen LLMs use RMSNorm. RoPE on Q,K every layer."""
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    norms = 2 * rmsnorm_flops(N, d)
    residuals = 2 * residual_flops(N, d)
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    qk_norm = (rmsnorm_flops(N, head_dim) * (n_q + n_kv)) if has_qk_norm else 0
    rope = rope_flops(N, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_attention(N, n_q)
    if is_moe:
        router = moe_router_flops(N, moe_n, moe_k)
        per_expert = silu_flops(N, moe_ffn) + N * moe_ffn
        experts = moe_k * per_expert
        combine = moe_combine_flops(N, d, moe_k)
        shared = (silu_flops(N, shared_ffn) + N * shared_ffn) if has_shared else 0
        ffn_elem = router + experts + combine + shared
    else:
        ffn_elem = silu_flops(N, ffn) + N * ffn   # SwiGLU SiLU + gate*up
    per_layer = norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem
    return per_layer * n_layers


def _qwen_llm_elem_decode(
    N_in: int, n_out: int, d: int, n_q: int, n_kv: int, head_dim: int, ffn: int,
    *, n_layers: int, has_qk_norm: bool, has_qkv_bias: bool,
    is_moe: bool = False, moe_n: int = 0, moe_k: int = 0, moe_ffn: int = 0,
    has_shared: bool = False, shared_ffn: int = 0,
) -> int:
    """Per-layer elementwise FLOPs for LLM decode (n_out new tokens), summed
    over n_layers. Mirrors `_qwen_llm_elem_prefill` but at sequence length 1
    per step and integrated over the n_out steps; KV cache length is N_in."""
    if n_out <= 0:
        return 0
    qkv_dim = (n_q + 2 * n_kv) * head_dim
    norms = 2 * rmsnorm_flops(1, d) * n_out
    residuals = 2 * residual_flops(1, d) * n_out
    qkv_bias = (bias_flops(1, qkv_dim) * n_out) if has_qkv_bias else 0
    qk_norm = ((rmsnorm_flops(1, head_dim) * (n_q + n_kv)) * n_out) if has_qk_norm else 0
    rope = rope_flops_decode(n_out, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_decode(N_in, n_out, n_q)
    if is_moe:
        router = moe_router_flops(n_out, moe_n, moe_k)
        per_expert = silu_flops(1, moe_ffn) + moe_ffn
        experts = moe_k * per_expert * n_out
        combine = moe_combine_flops(n_out, d, moe_k)
        shared = ((silu_flops(1, shared_ffn) + shared_ffn) * n_out) if has_shared else 0
        ffn_elem = router + experts + combine + shared
    else:
        ffn_elem = silu_flops(n_out, ffn) + n_out * ffn
    per_layer = norms + residuals + qkv_bias + qk_norm + rope + attn_sm + ffn_elem
    return per_layer * n_layers


def _frame_visual_tokens(frames, patch: int, merge: int, temporal_patch: int):
    """Compute (n_pre_merge_tokens_total, n_post_merge_tokens_total).

    Qwen2.5/Qwen3 VL uses a 3D patch embedding with temporal_patch_size=2:
    consecutive pairs of frames are grouped into one temporal patch.
    Each frame contributes (H/patch) * (W/patch) spatial patches. Two
    frames in a temporal group share the same spatial grid.
    Spatial merging by `merge` (=2) reduces tokens by merge^2 *after* the ViT.

    pre-merge tokens (what enters ViT, per temporal group):
        n_pre = (H / patch) * (W / patch)   per spatial grid
        For an isolated frame (no pair), it is duplicated to form a temporal
        patch (the HF processor pads the temporal axis); we still get one
        temporal group per pair, ceil(n_frames/2) groups.
    post-merge tokens (what enters LLM):
        n_post = n_pre / (merge*merge)  per temporal group

    Snapping (H, W) to multiples of (patch * merge) is done via the
    transformers ``smart_resize`` (image_processing_qwen2_vl.py), which not
    only rounds to ``factor=patch*merge`` but also clamps the post-snap pixel
    count into [min_pixels=56*56, max_pixels=14*14*4*1280]. We fall back to
    the second frame in a pair only if the first is missing dims (the HF
    processor pads odd frame counts by repeating the last frame -- so we
    safely take the first frame of each temporal group).
    """
    n_pre_total = 0
    n_post_total = 0
    n_groups = ceil(len(frames) / temporal_patch)
    factor = patch * merge
    for g in range(n_groups):
        f = frames[g * temporal_patch]
        h_p, w_p = smart_resize(int(f['height']), int(f['width']), factor=factor)
        n_h = h_p // patch
        n_w = w_p // patch
        n_pre = n_h * n_w
        n_post = (n_h // merge) * (n_w // merge)
        n_pre_total += n_pre
        n_post_total += n_post
    return n_pre_total, n_post_total


# ---------------------------------------------------------------------------
# 1) Qwen2.5-VL-7B-Instruct
#    Source: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json
#
#    Text (Qwen2.5 LLM, SwiGLU FFN):
#      hidden_size = 3584
#      num_hidden_layers = 28
#      intermediate_size = 18944
#      num_attention_heads = 28
#      num_key_value_heads = 4              -> GQA, head_dim = 3584/28 = 128
#      vocab_size = 152064
#      hidden_act = "silu"  (SwiGLU)
#
#    Vision (Qwen2.5-VL ViT, SwiGLU MLP per Qwen2.5-VL paper / code):
#      depth = 32
#      hidden_size = 1280
#      intermediate_size = 3420             -> SwiGLU (3 matmuls)
#      num_heads = 16                       -> head_dim = 80
#      patch_size = 14
#      spatial_merge_size = 2               -> 2x2 patch merging connector
#      temporal_patch_size = 2              -> pairs of frames -> 1 temporal patch
#      in_chans = 3
#      out_hidden_size = 3584               -> connector output dim = LLM hidden
#
#    Connector ("merger"): concat 2*2 visual tokens, RMSNorm, then a 2-layer MLP:
#        Linear(hidden*4 -> hidden*4)  GELU  Linear(hidden*4 -> out_hidden_size)
#    Per-token (post-merge) cost:
#        2 * (hidden*4) * (hidden*4)  +  2 * (hidden*4) * out_hidden
#      = 2*5120*5120 + 2*5120*3584
#    (Patch embedding: Conv3d(3, 1280, kernel=(2,14,14), stride=(2,14,14)) is
#     applied per pre-merge token; cost per pre-merge token =
#     2 * (3 * 2 * 14 * 14) * hidden = 2 * 1176 * 1280.)
# ---------------------------------------------------------------------------

QWEN25_VL_7B = dict(
    # text  (config.json: hidden_size=3584, num_hidden_layers=28,
    # intermediate_size=18944, num_attention_heads=28, num_key_value_heads=4,
    # head_dim=hidden_size/num_attention_heads=128, vocab_size=152064.)
    text_layers=28, text_d=3584, text_ffn=18944,
    text_n_q=28, text_n_kv=4, text_head_dim=128,
    vocab=152064,                       # config.json: vocab_size
    # vision  (config.json vision_config: depth=32, hidden_size=1280,
    # intermediate_size=3420, num_heads=16, patch_size=14, spatial_merge_size=2,
    # temporal_patch_size=2, in_channels=3, out_hidden_size=3584,
    # window_size=112, fullatt_block_indexes=[7,15,23,31], hidden_act='silu'.)
    vit_depth=32, vit_d=1280, vit_ffn=3420, vit_heads=16,
    patch=14, merge=2, temporal_patch=2, in_chans=3,
    out_hidden=3584,
    vit_ffn_style='swiglu',
    # Windowed attention: 4 of 32 blocks are full-attn, 28 are windowed.
    # window_size=112 px / patch=14 = 8 spatial patches per side -> 64 tokens.
    # Source: config.json vision_config.window_size=112 and
    #   fullatt_block_indexes=[7,15,23,31];
    #   modeling_qwen2_5_vl.py `Qwen2_5_VisionTransformerPretrainedModel.forward`.
    vit_window_tokens=64,
    vit_fullatt_block_indexes=(7, 15, 23, 31),
)


def flops_qwen2_5_vl_7b(frames, n_in_text_tokens, n_out_text_tokens):
    """Qwen2.5-VL (7B) FLOPs.

    CALLER CONTRACT
    ---------------
    Pass `frames` as ``[{height: H, width: W}, ...]`` with ANY H, W. The function
    snaps each frame's (H, W) to multiples of ``patch_size * spatial_merge_size
    = 14 * 2 = 28`` INSIDE via ``smart_resize`` (port of
    ``transformers/.../image_processing_qwen2_vl.py:smart_resize``). The same
    helper clamps post-snap pixel count into ``[min_pixels=56*56,
    max_pixels=14*14*4*1280]``. Default ``min_pixels`` and ``max_pixels`` track
    the transformers source; the released Qwen2.5-VL preprocessor_config.json
    ships ``max_pixels=12845056`` instead — for inputs near or beyond the
    transformers default this is a 12.8x larger budget that the helper does NOT
    apply (caller can pre-resize if it matters).
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (Qwen2.5-VL ViT, depth=32, hidden=1280, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 query heads, 16 KV heads — no GQA at the ViT).
    2. Attention scope: WINDOWED for 28 of 32 blocks. fullatt_block_indexes =
       [7,15,23,31] use full attention; the other 28 blocks split tokens by
       cu_window_seqlens with window_size=112 px = 8x8 patches = 64 tokens.
       Compute is O(N*W) on windowed blocks, O(N^2) on full ones.
    3. Positional embedding: 2D MRoPE applied per-head (matmul-free; not counted).
    4. FFN: SwiGLU (gate_proj + up_proj + down_proj), intermediate=3420 -> 6*N*H*I.
    5. CLS token: ABSENT.
    6. Variable-length packing: YES (single cu_seqlens for all packed image/video
       tokens; multi-image batches are a single packed sequence).
    """
    c = QWEN25_VL_7B
    n_pre, n_post = _frame_visual_tokens(
        frames, c['patch'], c['merge'], c['temporal_patch'])

    # ----- Patch embedding (Conv3d as a matmul over each pre-merge token) -----
    patch_input_dim = c['in_chans'] * c['temporal_patch'] * c['patch'] * c['patch']
    patch_embed_flops = 2 * n_pre * patch_input_dim * c['vit_d']

    # ----- ViT trunk: 32 blocks; 4 full-attn, 28 windowed (W=64 tokens) -----
    n_full = len(c['vit_fullatt_block_indexes'])
    n_win = c['vit_depth'] - n_full
    vit_trunk_flops = (
        n_full * _vit_block_flops(
            n_pre, c['vit_d'], c['vit_heads'], c['vit_ffn'], c['vit_ffn_style'])
        + n_win * _vit_block_flops(
            n_pre, c['vit_d'], c['vit_heads'], c['vit_ffn'], c['vit_ffn_style'],
            window_tokens=c['vit_window_tokens'])
    )
    vision_flops = patch_embed_flops + vit_trunk_flops

    # ----- Connector (merger): 2x2 spatial merge then 2-layer MLP -----
    # merge concat: free (just reshape). Per post-merge token:
    merge_in_dim = c['vit_d'] * c['merge'] * c['merge']  # 5120
    # Layer 1: (merge_in_dim -> merge_in_dim), Layer 2: (merge_in_dim -> out_hidden)
    per_token_connector = (2 * merge_in_dim * merge_in_dim
                          + 2 * merge_in_dim * c['out_hidden'])
    connector_flops = n_post * per_token_connector

    # ----- LLM prefill -----
    N = n_post + n_in_text_tokens
    text_d, n_q, n_kv, hd = c['text_d'], c['text_n_q'], c['text_n_kv'], c['text_head_dim']
    # SwiGLU FFN: 3 matmuls of size (d, ffn) per token -> 6*d*ffn per token
    ffn_per_tok = 6 * text_d * c['text_ffn']
    prefill = _llm_prefill_flops(N, c['text_layers'], text_d, n_q, n_kv, hd, ffn_per_tok)

    # ----- LLM decode -----
    decode = _llm_decode_flops(N, n_out_text_tokens, c['text_layers'],
                               text_d, n_q, n_kv, hd, ffn_per_tok)

    # ----- Elementwise (norms / softmax / RoPE / activations / biases / residuals) -----
    # Vision elementwise: per-block; full-attn 4 blocks, windowed 28 blocks.
    vit_elem_full = _qwen_vit_elem_block(
        n_pre, c['vit_d'], c['vit_ffn'], c['vit_heads'], c['vit_ffn_style'],
        has_qkv_bias=True, use_rope=True, window_tokens=None,
    )
    vit_elem_win = _qwen_vit_elem_block(
        n_pre, c['vit_d'], c['vit_ffn'], c['vit_heads'], c['vit_ffn_style'],
        has_qkv_bias=True, use_rope=True, window_tokens=c['vit_window_tokens'],
    )
    vision_elem = n_full * vit_elem_full + n_win * vit_elem_win
    # Connector elementwise: RMSNorm pre-merger (per Qwen2_5_VLPatchMerger),
    # GELU between Linears; biases on both Linears.
    conn_elem = (rmsnorm_flops(n_post, merge_in_dim)
                 + gelu_tanh_flops(n_post, merge_in_dim)
                 + bias_flops(n_post, merge_in_dim) + bias_flops(n_post, c['out_hidden']))
    # LLM elementwise: Qwen2.5 LLM has q/k bias=True (qkv_bias=True), no qk-norm.
    llm_pre_elem = _qwen_llm_elem_prefill(
        N, text_d, n_q, n_kv, hd, c['text_ffn'],
        n_layers=c['text_layers'], has_qk_norm=False, has_qkv_bias=True,
    )
    llm_dec_elem = _qwen_llm_elem_decode(
        N, n_out_text_tokens, text_d, n_q, n_kv, hd, c['text_ffn'],
        n_layers=c['text_layers'], has_qk_norm=False, has_qkv_bias=True,
    )
    # Final RMSNorm + LM-head softmax for sampling at decode.
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, text_d)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, c['vocab'])
    elementwise = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill + decode
    total_with_elem = total + elementwise
    return dict(vision_flops=vision_flops, connector_flops=connector_flops,
                llm_prefill_flops=prefill, llm_decode_flops=decode,
                total_flops=total,
                vision_elementwise=vision_elem,
                connector_elementwise=conn_elem,
                llm_prefill_elementwise=llm_pre_elem,
                llm_decode_elementwise=llm_dec_elem,
                elementwise_total=elementwise,
                total_with_elementwise=total_with_elem)


# ---------------------------------------------------------------------------
# 2) Qwen3-VL-8B-Instruct
#    Source: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json
#
#    Text:
#      hidden_size = 4096
#      num_hidden_layers = 36
#      intermediate_size = 12288  (SwiGLU)
#      num_attention_heads = 32
#      num_key_value_heads = 8        (GQA)
#      head_dim = 128
#      vocab_size = 151936
#      hidden_act = "silu"
#
#    Vision (SigLIP-style ViT, GELU MLP):
#      depth = 27
#      hidden_size = 1152
#      intermediate_size = 4304     (standard 2-matmul GELU MLP, model card)
#      num_heads = 16               -> head_dim = 72
#      patch_size = 16
#      spatial_merge_size = 2
#      temporal_patch_size = 2
#      out_hidden_size = 4096
#      deepstack_visual_indexes = [8, 16, 24]
#        -> features from layers 8, 16, 24 are also projected by separate
#           "deepstack" adapters and added into the LLM at multiple LLM layers.
#           That is 3 extra connector projections. We model it as 3 extra
#           merge+linear projections of post-merge token count.
#
#    Connector: same 2x2 merge + Linear -> Linear (RMSNorm + GELU between).
#      Linear(hidden*4 -> hidden*4) -> Linear(hidden*4 -> out_hidden)
#      = 2 * 4608*4608 + 2 * 4608*4096 per post-merge token
#    DeepStack adapters: 3 extra such connectors (one per indexed layer).
#
#    Vision uses GELU MLP per Qwen3-VL paper (SigLIP-derived). hidden_act
#    = "gelu_pytorch_tanh".
# ---------------------------------------------------------------------------

QWEN3_VL_8B = dict(
    # text  (config.json: hidden_size=4096, num_hidden_layers=36,
    # intermediate_size=12288, num_attention_heads=32, num_key_value_heads=8,
    # head_dim=128, vocab_size=151936, hidden_act='silu'.)
    text_layers=36, text_d=4096, text_ffn=12288,
    text_n_q=32, text_n_kv=8, text_head_dim=128,
    vocab=151936,                       # config.json: vocab_size
    # vision  (config.json vision_config: depth=27, hidden_size=1152,
    # intermediate_size=4304, num_heads=16, patch_size=16, spatial_merge_size=2,
    # temporal_patch_size=2, in_channels=3, out_hidden_size=4096,
    # deepstack_visual_indexes=[8,16,24], hidden_act='gelu_pytorch_tanh'.)
    vit_depth=27, vit_d=1152, vit_ffn=4304, vit_heads=16,
    patch=16, merge=2, temporal_patch=2, in_chans=3,
    out_hidden=4096,
    vit_ffn_style='gelu',
    deepstack_indexes=[8, 16, 24],
)


def _qwen3_vl_flops(c, frames, n_in_text_tokens, n_out_text_tokens):
    """Shared body for Qwen3-VL (8B / 8B-Thinking / Omni-30B-A3B).

    `c` is one of QWEN3_VL_8B / QWEN3_OMNI_30B_A3B (a dict of arch params).
    The three variants only differ in (a) whether the LLM is dense or MoE
    (handled by their own wrappers, not here) and (b) `n_out_text_tokens`.
    Returns the matmul-only `total_flops` dict; elementwise terms are added
    by the caller wrappers.
    """
    n_pre, n_post = _frame_visual_tokens(
        frames, c['patch'], c['merge'], c['temporal_patch'])

    # Patch embedding: Conv3d(3, vit_d, kernel=(2,16,16))
    patch_input_dim = c['in_chans'] * c['temporal_patch'] * c['patch'] * c['patch']
    patch_embed_flops = 2 * n_pre * patch_input_dim * c['vit_d']

    vit_trunk_flops = c['vit_depth'] * _vit_block_flops(
        n_pre, c['vit_d'], c['vit_heads'], c['vit_ffn'], c['vit_ffn_style'])
    vision_flops = patch_embed_flops + vit_trunk_flops

    # Main connector + 3 deepstack connectors
    merge_in_dim = c['vit_d'] * c['merge'] * c['merge']
    per_token_connector = (2 * merge_in_dim * merge_in_dim
                          + 2 * merge_in_dim * c['out_hidden'])
    n_connectors = 1 + len(c['deepstack_indexes'])  # 1 main + 3 deepstack
    connector_flops = n_post * per_token_connector * n_connectors

    N = n_post + n_in_text_tokens
    text_d, n_q, n_kv, hd = c['text_d'], c['text_n_q'], c['text_n_kv'], c['text_head_dim']
    ffn_per_tok = 6 * text_d * c['text_ffn']  # SwiGLU
    prefill = _llm_prefill_flops(N, c['text_layers'], text_d, n_q, n_kv, hd, ffn_per_tok)
    decode = _llm_decode_flops(N, n_out_text_tokens, c['text_layers'],
                               text_d, n_q, n_kv, hd, ffn_per_tok)
    # ----- Elementwise -----
    # Vision: 27 full-attn blocks, 2-mat GELU (gelu_pytorch_tanh), qkv bias=True.
    vit_elem_per = _qwen_vit_elem_block(
        n_pre, c['vit_d'], c['vit_ffn'], c['vit_heads'], c['vit_ffn_style'],
        has_qkv_bias=True, use_rope=True, window_tokens=None,
    )
    vision_elem = c['vit_depth'] * vit_elem_per
    # Connector: RMSNorm pre-merger, GELU between Linears, biases on Linears.
    conn_elem_per_pack = (rmsnorm_flops(n_post, merge_in_dim)
                          + gelu_tanh_flops(n_post, merge_in_dim)
                          + bias_flops(n_post, merge_in_dim) + bias_flops(n_post, c['out_hidden']))
    conn_elem = conn_elem_per_pack * n_connectors  # main + 3 deepstack
    # LLM: Qwen3-VL LLM has qk_norm=True, no qkv_bias; SwiGLU.
    llm_pre_elem = _qwen_llm_elem_prefill(
        N, text_d, n_q, n_kv, hd, c['text_ffn'],
        n_layers=c['text_layers'], has_qk_norm=True, has_qkv_bias=False,
    )
    llm_dec_elem = _qwen_llm_elem_decode(
        N, n_out_text_tokens, text_d, n_q, n_kv, hd, c['text_ffn'],
        n_layers=c['text_layers'], has_qk_norm=True, has_qkv_bias=False,
    )
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, text_d)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, c['vocab'])
    elementwise = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill + decode
    return dict(vision_flops=vision_flops, connector_flops=connector_flops,
                llm_prefill_flops=prefill, llm_decode_flops=decode,
                total_flops=total,
                vision_elementwise=vision_elem,
                connector_elementwise=conn_elem,
                llm_prefill_elementwise=llm_pre_elem,
                llm_decode_elementwise=llm_dec_elem,
                elementwise_total=elementwise,
                total_with_elementwise=total + elementwise)


def flops_qwen3_vl_8b(frames, n_in_text_tokens, n_out_text_tokens):
    """Qwen3-VL (8B) FLOPs.

    CALLER CONTRACT
    ---------------
    Pass `frames` as ``[{height: H, width: W}, ...]`` with ANY H, W. The function
    snaps each frame's (H, W) to multiples of ``patch_size * spatial_merge_size
    = 16 * 2 = 32`` INSIDE via ``smart_resize`` (same body as Qwen2.5-VL; port
    of ``transformers/.../image_processing_qwen2_vl.py:smart_resize``). The
    pixel-count envelope ``[min_pixels=56*56, max_pixels=14*14*4*1280]`` matches
    transformers defaults.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (Qwen3-VL ViT, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads).
    2. Attention scope: FULL N^2 every block (no windowing).
    3. Positional embedding: 2D MRoPE (matmul-free; not counted).
    4. FFN: 2-matmul GELU (linear_fc1 + linear_fc2), intermediate=4304 -> 4*N*H*I.
    5. CLS token: ABSENT.
    6. Variable-length packing: YES (cu_seqlens used for batched image/video).
    """
    return _qwen3_vl_flops(QWEN3_VL_8B, frames, n_in_text_tokens, n_out_text_tokens)


# ---------------------------------------------------------------------------
# 3) Qwen3-VL-8B-Thinking
#    Source: https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking/raw/main/config.json
#    Architecture is IDENTICAL to Qwen3-VL-8B-Instruct (verified by diffing
#    config.json). Only difference is post-training (RL for reasoning) which
#    yields longer outputs - that is captured by n_out_text_tokens, not by
#    the FLOPs equation itself.
# ---------------------------------------------------------------------------

QWEN3_VL_8B_THINKING = dict(QWEN3_VL_8B)


def flops_qwen3_vl_8b_thinking(frames, n_in_text_tokens, n_out_text_tokens):
    """Qwen3-VL-Thinking (8B) FLOPs.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT
    -------------------------------------------------------------------
    Identical architecture to Qwen3-VL (8B) — only post-training differs. See
    flops_qwen3_vl_8b for the full audit. (MHA, full N^2, 2D MRoPE, 2-matmul
    GELU, no CLS, varlen packing.)
    """
    return _qwen3_vl_flops(QWEN3_VL_8B_THINKING, frames,
                          n_in_text_tokens, n_out_text_tokens)


# ---------------------------------------------------------------------------
# 4) Qwen3-Omni-30B-A3B (Thinker only - the speech codec/Talker is not used
#    for video QA tasks; we model the same component the question/text flows
#    through. The Talker would add cost only when generating speech.)
#    Source: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json
#
#    Thinker text (MoE):
#      hidden_size = 2048
#      num_hidden_layers = 48
#      intermediate_size = 768          (NOTE: equal to moe_intermediate_size;
#                                        this model has NO dense FFN layers)
#      moe_intermediate_size = 768      (per-expert SwiGLU intermediate)
#      num_experts = 128
#      num_experts_per_tok = 8
#      decoder_sparse_step = 1          (every layer is MoE)
#      mlp_only_layers = []             (no dense-FFN-only layers)
#      shared_expert_intermediate_size = 0  (no shared expert)
#      num_attention_heads = 32
#      num_key_value_heads = 4          (GQA)
#      head_dim = 128
#      vocab_size = 152064
#
#    Thinker vision (same SigLIP-style ViT as Qwen3-VL):
#      depth = 27, hidden_size = 1152, intermediate_size = 4304, heads = 16,
#      patch_size = 16, spatial_merge_size = 2, temporal_patch_size = 2,
#      out_hidden_size = 2048,
#      deepstack_visual_indexes = [8, 16, 24]
#
#    MoE FFN cost per token (SwiGLU expert, top-K activation):
#        per_expert_swiglu = 6 * d * moe_intermediate_size
#        active_experts_per_tok = num_experts_per_tok = 8
#        moe_ffn_per_token = 8 * per_expert_swiglu
#                          = 8 * 6 * 2048 * 768
#    Router cost per token:
#        2 * d * num_experts = 2 * 2048 * 128
#    No shared expert (shared_expert_intermediate_size = 0).
# ---------------------------------------------------------------------------

QWEN3_OMNI_30B_A3B = dict(
    # text MoE  (config.json thinker_config.text_config: hidden_size=2048,
    # num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4,
    # head_dim=128, num_experts=128, num_experts_per_tok=8,
    # moe_intermediate_size=768, decoder_sparse_step=1, mlp_only_layers=[],
    # shared_expert_intermediate_size=0, vocab_size=152064.)
    text_layers=48, text_d=2048,
    text_n_q=32, text_n_kv=4, text_head_dim=128,
    moe_n_experts=128, moe_topk=8, moe_ffn=768,
    moe_shared_ffn=0,
    vocab=152064,                       # config.json: vocab_size
    # vision  (config.json thinker_config.vision_config: depth=27,
    # hidden_size=1152, intermediate_size=4304, num_heads=16, patch_size=16,
    # spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
    # out_hidden_size=2048, deepstack_visual_indexes=[8,16,24].)
    vit_depth=27, vit_d=1152, vit_ffn=4304, vit_heads=16,
    patch=16, merge=2, temporal_patch=2, in_chans=3,
    out_hidden=2048,
    vit_ffn_style='gelu',
    deepstack_indexes=[8, 16, 24],
)


def flops_qwen3_omni_30b_a3b(frames, n_in_text_tokens, n_out_text_tokens):
    """Qwen3-Omni (30B, A3B) FLOPs (Thinker only — Talker not exercised by VQA).

    CALLER CONTRACT
    ---------------
    Same as Qwen3-VL (8B): pass any H, W and the function snaps to multiples of
    ``patch_size * spatial_merge_size = 16 * 2 = 32`` INSIDE via ``smart_resize``.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (Qwen3-VL ViT, depth=27, hidden=1152, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads).
    2. Attention scope: FULL N^2 every block.
    3. Positional embedding: 2D MRoPE (matmul-free; not counted).
    4. FFN: 2-matmul GELU, intermediate=4304 -> 4*N*H*I.
    5. CLS token: ABSENT.
    6. Variable-length packing: YES.
    Same ViT as Qwen3-VL; differs only in DeepStack output dim (out_hidden=2048)
    and the LLM backbone (Thinker MoE: 48L, hidden=2048, 128 experts, top-8).
    """
    c = QWEN3_OMNI_30B_A3B
    n_pre, n_post = _frame_visual_tokens(
        frames, c['patch'], c['merge'], c['temporal_patch'])

    patch_input_dim = c['in_chans'] * c['temporal_patch'] * c['patch'] * c['patch']
    patch_embed_flops = 2 * n_pre * patch_input_dim * c['vit_d']

    vit_trunk_flops = c['vit_depth'] * _vit_block_flops(
        n_pre, c['vit_d'], c['vit_heads'], c['vit_ffn'], c['vit_ffn_style'])
    vision_flops = patch_embed_flops + vit_trunk_flops

    merge_in_dim = c['vit_d'] * c['merge'] * c['merge']
    per_token_connector = (2 * merge_in_dim * merge_in_dim
                          + 2 * merge_in_dim * c['out_hidden'])
    n_connectors = 1 + len(c['deepstack_indexes'])
    connector_flops = n_post * per_token_connector * n_connectors

    # ----- LLM prefill (every layer is MoE; topk=8 of 128 experts active) -----
    N = n_post + n_in_text_tokens
    d, n_q, n_kv, hd = c['text_d'], c['text_n_q'], c['text_n_kv'], c['text_head_dim']
    # MoE per-token FFN: (router) + topk * SwiGLU(d, moe_ffn) + shared expert (zero here)
    router_per_tok = 2 * d * c['moe_n_experts']
    expert_swiglu_per_tok = 6 * d * c['moe_ffn']
    moe_ffn_per_tok = router_per_tok + c['moe_topk'] * expert_swiglu_per_tok
    if c['moe_shared_ffn'] > 0:
        moe_ffn_per_tok += 6 * d * c['moe_shared_ffn']

    prefill = _llm_prefill_flops(N, c['text_layers'], d, n_q, n_kv, hd, moe_ffn_per_tok)
    decode = _llm_decode_flops(N, n_out_text_tokens, c['text_layers'],
                              d, n_q, n_kv, hd, moe_ffn_per_tok)

    # ----- Elementwise -----
    vit_elem_per = _qwen_vit_elem_block(
        n_pre, c['vit_d'], c['vit_ffn'], c['vit_heads'], c['vit_ffn_style'],
        has_qkv_bias=True, use_rope=True, window_tokens=None,
    )
    vision_elem = c['vit_depth'] * vit_elem_per
    conn_elem_per_pack = (rmsnorm_flops(n_post, merge_in_dim)
                          + gelu_tanh_flops(n_post, merge_in_dim)
                          + bias_flops(n_post, merge_in_dim) + bias_flops(n_post, c['out_hidden']))
    conn_elem = conn_elem_per_pack * n_connectors
    # Q3-Omni Thinker LLM: pure MoE every layer (no shared expert), top-8 of 128.
    # qk_norm=True (Qwen3-style); no qkv_bias.
    llm_pre_elem = _qwen_llm_elem_prefill(
        N, d, n_q, n_kv, hd, 0,
        n_layers=c['text_layers'], has_qk_norm=True, has_qkv_bias=False,
        is_moe=True, moe_n=c['moe_n_experts'], moe_k=c['moe_topk'],
        moe_ffn=c['moe_ffn'], has_shared=False, shared_ffn=0,
    )
    llm_dec_elem = _qwen_llm_elem_decode(
        N, n_out_text_tokens, d, n_q, n_kv, hd, 0,
        n_layers=c['text_layers'], has_qk_norm=True, has_qkv_bias=False,
        is_moe=True, moe_n=c['moe_n_experts'], moe_k=c['moe_topk'],
        moe_ffn=c['moe_ffn'], has_shared=False, shared_ffn=0,
    )
    llm_dec_elem += rmsnorm_flops(n_out_text_tokens, d)
    llm_dec_elem += lm_head_softmax_decode(n_out_text_tokens, c['vocab'])
    elementwise = vision_elem + conn_elem + llm_pre_elem + llm_dec_elem

    total = vision_flops + connector_flops + prefill + decode
    return dict(vision_flops=vision_flops, connector_flops=connector_flops,
                llm_prefill_flops=prefill, llm_decode_flops=decode,
                total_flops=total,
                vision_elementwise=vision_elem,
                connector_elementwise=conn_elem,
                llm_prefill_elementwise=llm_pre_elem,
                llm_decode_elementwise=llm_dec_elem,
                elementwise_total=elementwise,
                total_with_elementwise=total + elementwise)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _fmt(d):
    """Pretty-print a per-component FLOPs dict in PFLOPs (used by __main__)."""
    PF = 1e15
    keys = ['vision_flops', 'connector_flops', 'llm_prefill_flops',
            'llm_decode_flops', 'total_flops']
    parts = [f"{k}={d[k]/PF:.3f} PF" for k in keys]
    return "  " + "\n  ".join(parts)


if __name__ == '__main__':
    frames = [{'height': 448, 'width': 448}] * 8
    n_in = 128
    n_out = 64

    # Pre-correction totals at full precision. Captured by re-running the
    # *prior* version of this file (windowed-attention fix reverted via
    # /tmp/_capture_before.py shim that used full attention for all 32
    # Qwen2.5-VL ViT blocks). Other three models had no numeric changes
    # (only docstring citations were added) -> their BEFORE == AFTER.
    BEFORE = {
        "Qwen2.5-VL (7B)": dict(
            vision_flops=7921781964800,
            connector_flops=91268055040,
            llm_prefill_flops=15566974746624,
            llm_decode_flops=865641168896,
            total_flops=24445665935360,
        ),
        "Qwen3-VL (8B)": dict(
            vision_flops=3812900732928,
            connector_flops=251557576704,
            llm_prefill_flops=13159662354432,
            llm_decode_flops=924674162688,
            total_flops=18148794826752,
        ),
        "Qwen3-VL-Thinking (8B)": dict(
            vision_flops=3812900732928,
            connector_flops=251557576704,
            llm_prefill_flops=13159662354432,
            llm_decode_flops=924674162688,
            total_flops=18148794826752,
        ),
        "Qwen3-Omni (30B-A3B)": dict(
            vision_flops=3812900732928,
            connector_flops=192367558656,
            llm_prefill_flops=5634527330304,
            llm_decode_flops=396990873600,
            total_flops=10036786495488,
        ),
    }

    print("Validation: 8 frames @ 448x448, n_in_text=128, n_out_text=64")
    print()

    for name, fn in [
        ("Qwen2.5-VL (7B)", flops_qwen2_5_vl_7b),
        ("Qwen3-VL (8B)", flops_qwen3_vl_8b),
        ("Qwen3-VL-Thinking (8B)", flops_qwen3_vl_8b_thinking),
        ("Qwen3-Omni (30B-A3B)", flops_qwen3_omni_30b_a3b),
    ]:
        out = fn(frames, n_in, n_out)
        print(f"=== {name} ===")
        print(_fmt(out))
        before = BEFORE[name]
        delta = out['total_flops'] - before['total_flops']
        pct = 100.0 * delta / before['total_flops'] if before['total_flops'] else 0.0
        print(f"  total before  = {before['total_flops']/1e15:.4f} PF")
        print(f"  total after   = {out['total_flops']/1e15:.4f} PF")
        print(f"  delta         = {delta/1e15:+.4f} PF ({pct:+.1f}%)")
        v_delta = out['vision_flops'] - before['vision_flops']
        v_pct = 100.0 * v_delta / before['vision_flops'] if before['vision_flops'] else 0.0
        print(f"  vision delta  = {v_delta/1e15:+.4f} PF ({v_pct:+.1f}%)")
        print()
