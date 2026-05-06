"""Centralised per-element FLOPs constants and helpers.

Convention (per task spec):
  - RMSNorm  : 4 * N * d                        per (N, d) tensor
  - LayerNorm: 7 * N * d                        per (N, d) tensor
  - Softmax  : 5 * K        per row of K elems   (max + sub + exp + sum + div)
  - RoPE     : 4 * N * head_dim * (H_q + H_kv)   per layer, per N tokens
  - SiLU     : 4 * N * d                         (sigmoid = 3 ops, x*sig = 1)
  - GELU exact:5 * N * d
  - GELU tanh:15 * N * d                         (cube + tanh approximation)
  - Bias add : 1 * N * d_out
  - Residual : 1 * N * d
  - MoE router softmax + top-k: per token cost = 5*E + (E + K*log2(E))
        (softmax: 5*E ops; top-k via heap selection: ~E + K*log2(E) comparisons.
         Source: PyTorch ATen `aten/src/ATen/native/cuda/SortingKthValue.cu`
         and the radix-select fallback in `aten/src/ATen/native/cuda/TensorTopK.cu`,
         which run a single linear scan over E elements followed by K
         heap-sift operations of cost log2(E) each. Earlier versions of this
         module used the algorithmic-complexity bound K*log2(E) only; replaced
         with the runtime-realistic E + K*log2(E) on 2026-04-27.)
  - MoE weighted combine: 2*K*d*N
  - LM-head decode softmax: 5 * V per output token
  - KV cache memory ops: NOT counted (memory bandwidth, not FLOPs)

Each helper returns FLOPs (int / float). All modules in flops_estimator
import from here so the convention is enforced identically across all 16 models.
"""

from __future__ import annotations

from math import log2


# ---------------------------------------------------------------------------
# Per-element constants
# ---------------------------------------------------------------------------

RMSNORM_FLOPS_PER_ELEM    = 4    # square + sum + div_sqrt(amortised) + 2 muls
LAYERNORM_FLOPS_PER_ELEM  = 7    # mean + var(2x) + normalise(2x) + scale + shift
SOFTMAX_FLOPS_PER_ELEM    = 5    # max + sub + exp + sum + div
SILU_FLOPS_PER_ELEM       = 4    # sigmoid (exp + add + div) + x*sigmoid
GELU_EXACT_FLOPS_PER_ELEM = 5    # erf + 4 surrounding ops
GELU_TANH_FLOPS_PER_ELEM  = 15   # cube(2) + chain(5) + tanh(5) + finish(3)
BIAS_FLOPS_PER_ELEM       = 1
RESIDUAL_FLOPS_PER_ELEM   = 1
ROPE_FLOPS_PER_ELEM       = 4    # 2 muls + 2 adds per (head_dim/2) pair = 4/elem
LM_HEAD_DECODE_SOFTMAX_PER_ELEM = 5


# ---------------------------------------------------------------------------
# Helper functions (return FLOPs as a number)
# ---------------------------------------------------------------------------

def rmsnorm_flops(N: int, d: int) -> int:
    """RMSNorm over an (N, d) tensor."""
    return RMSNORM_FLOPS_PER_ELEM * N * d


def layernorm_flops(N: int, d: int) -> int:
    """LayerNorm over an (N, d) tensor."""
    return LAYERNORM_FLOPS_PER_ELEM * N * d


def softmax_flops_attention(N: int, H_q: int) -> int:
    """Softmax over QK^T scores: full N rows of length N per Q-head, full attn.

    For full attention with causal mask, we count the full N*N (consistent with
    matmul-only accounting which also counts full N^2). Per row: 5*N FLOPs;
    per Q-head: 5*N^2; per layer: 5*N^2*H_q.
    """
    return SOFTMAX_FLOPS_PER_ELEM * N * N * H_q


def softmax_flops_attention_chunks(seqlens, H_q: int) -> int:
    """Softmax over var-len packed attention: sum over chunks of 5*L^2*H_q."""
    return SOFTMAX_FLOPS_PER_ELEM * H_q * sum(L * L for L in seqlens)


def softmax_flops_attention_windowed(N: int, W: int, H_q: int) -> int:
    """Softmax over windowed attention: each row of length W, N rows, per head."""
    return SOFTMAX_FLOPS_PER_ELEM * N * W * H_q


def softmax_flops_decode(N_in: int, n_out: int, H_q: int) -> int:
    """Softmax over decode KV-cache: step t attends to L_t = N_in + t keys.

    sum_{t=0..n_out-1} L_t = n_out*N_in + n_out*(n_out-1)/2
    Per head per step: 5 * L_t. Multiply by H_q heads.
    """
    if n_out <= 0:
        return 0
    sum_L = n_out * N_in + (n_out * (n_out - 1)) // 2
    return SOFTMAX_FLOPS_PER_ELEM * H_q * sum_L


def rope_flops(N: int, head_dim: int, H_q: int, H_kv: int) -> int:
    """RoPE applied to Q and K: 4 * N * head_dim * (H_q + H_kv) per layer."""
    return ROPE_FLOPS_PER_ELEM * N * head_dim * (H_q + H_kv)


def rope_flops_decode(n_out: int, head_dim: int, H_q: int, H_kv: int) -> int:
    """RoPE during decode: per output token cost on its 1 new Q,K row."""
    if n_out <= 0:
        return 0
    return ROPE_FLOPS_PER_ELEM * n_out * head_dim * (H_q + H_kv)


def silu_flops(N: int, d: int) -> int:
    """SiLU activation over an N x d tensor: per-element constant times N*d."""
    return SILU_FLOPS_PER_ELEM * N * d


def gelu_exact_flops(N: int, d: int) -> int:
    """GELU (exact, erf-based) activation over N x d."""
    return GELU_EXACT_FLOPS_PER_ELEM * N * d


def gelu_tanh_flops(N: int, d: int) -> int:
    """GELU (tanh approximation) activation over N x d."""
    return GELU_TANH_FLOPS_PER_ELEM * N * d


def bias_flops(N: int, d_out: int) -> int:
    """Per-row bias add (1 add per output element)."""
    return BIAS_FLOPS_PER_ELEM * N * d_out


def residual_flops(N: int, d: int) -> int:
    """Residual add (1 add per element)."""
    return RESIDUAL_FLOPS_PER_ELEM * N * d


def moe_router_flops(N: int, num_experts: int, top_k: int) -> int:
    """Router softmax (5*E) + top-k selection (E + K*log2(E)) per token.

    The top-k cost matches what `torch.topk(x, k)` actually runs: a single
    linear scan over the E-element vector plus K heap-sift insertions each
    costing log2(E) comparisons. Source: PyTorch CUDA top-k kernels
    (`aten/src/ATen/native/cuda/SortingKthValue.cu`,
    `aten/src/ATen/native/cuda/TensorTopK.cu`). 1 op per comparison.
    Magnitude check at E=128, K=8: per-token = 5*128 + (128 + 8*7) = 824 ops.
    """
    if num_experts <= 1:
        return 0
    log2_E = max(1, int(log2(num_experts)))
    topk_per_tok = num_experts + top_k * log2_E
    per_tok = SOFTMAX_FLOPS_PER_ELEM * num_experts + topk_per_tok
    return N * per_tok


def moe_combine_flops(N: int, d: int, top_k: int) -> int:
    """Weighted combine across top-k expert outputs: 2*K*d per token."""
    return 2 * top_k * d * N


def lm_head_softmax_decode(n_out: int, vocab: int) -> int:
    """Softmax over vocab during sampling, counted only at decode."""
    if n_out <= 0:
        return 0
    return LM_HEAD_DECODE_SOFTMAX_PER_ELEM * vocab * n_out


# ---------------------------------------------------------------------------
# Composite helpers (commonly needed combos)
# ---------------------------------------------------------------------------

def vit_block_elementwise(
    N: int,
    d: int,
    n_heads_q: int,
    n_heads_kv: int,
    head_dim: int,
    ffn: int,
    *,
    norm_kind: str,           # 'rmsnorm' or 'layernorm'
    activation: str,          # 'silu' / 'gelu_exact' / 'gelu_tanh'
    ffn_kind: str,            # 'swiglu' / 'gelu_2mat'
    has_qkv_bias: bool,
    has_o_bias: bool,
    has_ffn_bias: bool,
    use_rope: bool,
    attn_seqlens: list | None = None,  # var-len chunks; None = full N
    window_tokens: int | None = None,  # windowed attn; None = full
) -> dict:
    """Return per-component elementwise FLOPs for one ViT block.

    Two pre-norms (one before attn, one before FFN) by default. Two residuals.
    QKV bias (post-Linear) if has_qkv_bias; etc. Activation between MLP layers.
    """
    norm_fn = rmsnorm_flops if norm_kind == 'rmsnorm' else layernorm_flops

    norms = 2 * norm_fn(N, d)
    residuals = 2 * residual_flops(N, d)

    qkv_dim = (n_heads_q + 2 * n_heads_kv) * head_dim
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    o_bias = bias_flops(N, d) if has_o_bias else 0

    if use_rope:
        rope = rope_flops(N, head_dim, n_heads_q, n_heads_kv)
    else:
        rope = 0

    if attn_seqlens is not None:
        attn_softmax = softmax_flops_attention_chunks(attn_seqlens, n_heads_q)
    elif window_tokens is not None and window_tokens < N:
        attn_softmax = softmax_flops_attention_windowed(N, window_tokens, n_heads_q)
    else:
        attn_softmax = softmax_flops_attention(N, n_heads_q)

    if ffn_kind == 'swiglu':
        # gate, up are activated via SiLU (gate * up), down has no act
        # bias adds: gate, up, down each get one (if has_ffn_bias)
        if activation == 'silu':
            act = silu_flops(N, ffn)
        elif activation == 'gelu_exact':
            act = gelu_exact_flops(N, ffn)
        else:
            act = gelu_tanh_flops(N, ffn)
        # element-wise gate*up multiply (after activating gate):
        gate_up_mul = N * ffn  # 1 op per element
        if has_ffn_bias:
            ffn_bias = bias_flops(N, ffn) * 2 + bias_flops(N, d)  # gate+up+down
        else:
            ffn_bias = 0
        ffn_elem = act + gate_up_mul + ffn_bias
    else:  # 'gelu_2mat'
        if activation == 'silu':
            act = silu_flops(N, ffn)
        elif activation == 'gelu_exact':
            act = gelu_exact_flops(N, ffn)
        else:
            act = gelu_tanh_flops(N, ffn)
        if has_ffn_bias:
            ffn_bias = bias_flops(N, ffn) + bias_flops(N, d)  # fc1 + fc2
        else:
            ffn_bias = 0
        ffn_elem = act + ffn_bias

    return {
        'norms': norms,
        'residuals': residuals,
        'qkv_bias': qkv_bias,
        'o_bias': o_bias,
        'rope': rope,
        'attn_softmax': attn_softmax,
        'ffn_elem': ffn_elem,
        'total': norms + residuals + qkv_bias + o_bias + rope + attn_softmax + ffn_elem,
    }


def llm_block_elementwise_prefill(
    N: int,
    d: int,
    n_heads_q: int,
    n_heads_kv: int,
    head_dim: int,
    ffn: int,
    *,
    activation: str = 'silu',         # SiLU for SwiGLU LLMs
    has_qkv_bias: bool = False,
    has_o_bias: bool = False,
    has_ffn_bias: bool = False,
    has_qk_norm: bool = False,
    use_rope: bool = True,
    is_moe: bool = False,
    moe_num_experts: int = 0,
    moe_top_k: int = 0,
    moe_ffn: int = 0,
    has_shared_expert: bool = False,
    shared_ffn: int = 0,
) -> dict:
    """Per-LLM-block elementwise FLOPs for prefill of N tokens (no causal mask
    discount — matches matmul convention)."""
    norms = 2 * rmsnorm_flops(N, d)  # pre-attn + pre-FFN
    residuals = 2 * residual_flops(N, d)
    qkv_dim = (n_heads_q + 2 * n_heads_kv) * head_dim
    qkv_bias = bias_flops(N, qkv_dim) if has_qkv_bias else 0
    o_bias = bias_flops(N, d) if has_o_bias else 0
    qk_norm = (rmsnorm_flops(N, head_dim) * (n_heads_q + n_heads_kv)
               if has_qk_norm else 0)
    rope = rope_flops(N, head_dim, n_heads_q, n_heads_kv) if use_rope else 0
    attn_softmax = softmax_flops_attention(N, n_heads_q)

    if is_moe:
        router = moe_router_flops(N, moe_num_experts, moe_top_k)
        # Each active expert: SwiGLU activation + gate*up + biases (none)
        per_expert_act = silu_flops(N, moe_ffn) + (N * moe_ffn)  # silu + gate*up mul
        experts_act = moe_top_k * per_expert_act
        combine = moe_combine_flops(N, d, moe_top_k)
        if has_shared_expert:
            shared_act = silu_flops(N, shared_ffn) + (N * shared_ffn)
        else:
            shared_act = 0
        ffn_elem = router + experts_act + combine + shared_act
    else:
        if activation == 'silu':
            act = silu_flops(N, ffn)
        elif activation == 'gelu_exact':
            act = gelu_exact_flops(N, ffn)
        else:
            act = gelu_tanh_flops(N, ffn)
        gate_up_mul = N * ffn  # SwiGLU element-wise gate*up
        if has_ffn_bias:
            ffn_bias = bias_flops(N, ffn) * 2 + bias_flops(N, d)
        else:
            ffn_bias = 0
        ffn_elem = act + gate_up_mul + ffn_bias

    return {
        'norms': norms,
        'residuals': residuals,
        'qkv_bias': qkv_bias,
        'o_bias': o_bias,
        'qk_norm': qk_norm,
        'rope': rope,
        'attn_softmax': attn_softmax,
        'ffn_elem': ffn_elem,
        'total': norms + residuals + qkv_bias + o_bias + qk_norm + rope + attn_softmax + ffn_elem,
    }


def llm_block_elementwise_decode(
    N_in: int,
    n_out: int,
    d: int,
    n_heads_q: int,
    n_heads_kv: int,
    head_dim: int,
    ffn: int,
    *,
    activation: str = 'silu',
    has_qkv_bias: bool = False,
    has_o_bias: bool = False,
    has_ffn_bias: bool = False,
    has_qk_norm: bool = False,
    use_rope: bool = True,
    is_moe: bool = False,
    moe_num_experts: int = 0,
    moe_top_k: int = 0,
    moe_ffn: int = 0,
    has_shared_expert: bool = False,
    shared_ffn: int = 0,
) -> dict:
    """Per-LLM-block elementwise FLOPs for decode of n_out tokens after N_in prefill.

    Each step processes 1 new query token; norm/residual/FFN-act over 1 token;
    softmax over L_t = N_in + t cached keys.
    """
    if n_out <= 0:
        return {'total': 0}
    # Constant-per-step bits (each over 1 token)
    norms = 2 * rmsnorm_flops(1, d) * n_out
    residuals = 2 * residual_flops(1, d) * n_out
    qkv_dim = (n_heads_q + 2 * n_heads_kv) * head_dim
    qkv_bias = (bias_flops(1, qkv_dim) * n_out) if has_qkv_bias else 0
    o_bias = (bias_flops(1, d) * n_out) if has_o_bias else 0
    qk_norm = ((rmsnorm_flops(1, head_dim) * (n_heads_q + n_heads_kv)) * n_out
               if has_qk_norm else 0)
    rope = rope_flops_decode(n_out, head_dim, n_heads_q, n_heads_kv) if use_rope else 0
    attn_softmax = softmax_flops_decode(N_in, n_out, n_heads_q)

    if is_moe:
        router = moe_router_flops(n_out, moe_num_experts, moe_top_k)
        per_expert_act = silu_flops(1, moe_ffn) + moe_ffn  # silu + gate*up mul
        experts_act = moe_top_k * per_expert_act * n_out
        combine = moe_combine_flops(n_out, d, moe_top_k)
        if has_shared_expert:
            shared_act = (silu_flops(1, shared_ffn) + shared_ffn) * n_out
        else:
            shared_act = 0
        ffn_elem = router + experts_act + combine + shared_act
    else:
        if activation == 'silu':
            act = silu_flops(n_out, ffn)
        elif activation == 'gelu_exact':
            act = gelu_exact_flops(n_out, ffn)
        else:
            act = gelu_tanh_flops(n_out, ffn)
        gate_up_mul = n_out * ffn
        if has_ffn_bias:
            ffn_bias = bias_flops(n_out, ffn) * 2 + bias_flops(n_out, d)
        else:
            ffn_bias = 0
        ffn_elem = act + gate_up_mul + ffn_bias

    return {
        'norms': norms,
        'residuals': residuals,
        'qkv_bias': qkv_bias,
        'o_bias': o_bias,
        'qk_norm': qk_norm,
        'rope': rope,
        'attn_softmax': attn_softmax,
        'ffn_elem': ffn_elem,
        'total': norms + residuals + qkv_bias + o_bias + qk_norm + rope + attn_softmax + ffn_elem,
    }
