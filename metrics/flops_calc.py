"""
FLOP calculation functions for modern vision-language and language models.

Conventions and baseline assumptions
------------------------------------
1. Dense matrix multiplies cost:    2 * batch * tokens * in_dim * out_dim
   (multiply + add fused as 2 FLOPs).
2. MLPs:
   - Standard GeLU MLP: hidden → (expansion * hidden) → hidden.
   - SwiGLU MLP (Qwen/GLM style): two parallel projections (gate, up) plus
     an elementwise gate and a SiLU on the gating path.
3. Attention:
   - QK^T and Attn@V are counted as pure matmuls.
   - Softmax cost is linear in the number of attention scores.
   - Rotary / RoPE is approximated as 4 FLOPs per component.
4. Normalization:
   - RMSNorm:  (4 * hidden + 2) FLOPs per token.
   - LayerNorm: (7 * hidden + 4) FLOPs per token.
5. Autoregressive generation:
   - Uses KV caching: only the newly generated tokens pay Q/K/V projections,
     QK^T, Attn@V, and the LM head.
6. Training cost:
   - All *building blocks* return forward-only FLOPs.
   - Model-level functions take `do_backward`; if True, they multiply the
     total FLOP count (and per-component breakdowns) by 3 to approximate
     forward + backward + optimizer update.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# ============================================================================
# Core MLP and normalization primitives
# ============================================================================


def mlp_layer_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    expansion: float = 4.0,
    swiglu: bool = False,
) -> int:
    """
    FLOPs for a transformer-style MLP on (batch_size, seq_len, hidden_size).

    Standard GeLU MLP:
        hidden → (expansion * hidden) → hidden

        FLOPs = 2 * B * L * H * (expansion*H)        # up-projection
              + 4 * B * L * (expansion*H)           # GeLU approximation
              + 2 * B * L * (expansion*H) * H       # down-projection

    SwiGLU MLP (gate + up projections, gated, then down):
        intermediate = expansion * hidden

        FLOPs = 2 * B * L * H * inter               # gate_proj
              + 2 * B * L * H * inter               # up_proj
              + 2 * B * L * inter * H               # down_proj
              +     B * L * inter                   # elementwise gate
              + 3 * B * L * inter                   # SiLU (exp + recip + mul)
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0:
        return 0

    B = int(batch_size)
    L = int(seq_len)
    H = int(hidden_size)
    inter = int(expansion * H)

    if swiglu:
        gate_proj = 2 * B * L * H * inter
        up_proj = 2 * B * L * H * inter
        down_proj = 2 * B * L * inter * H
        elementwise_gate = B * L * inter
        silu = 3 * B * L * inter
        return gate_proj + up_proj + down_proj + elementwise_gate + silu
    else:
        up_proj = 2 * B * L * H * inter
        act = 4 * B * L * inter
        down_proj = 2 * B * L * inter * H
        return up_proj + act + down_proj


def moe_mlp_token_flops(
    hidden_size: int,
    expert_hidden_size: int,
    num_experts_per_token: int,
    num_routed_experts: int,
    num_shared_experts: int,
    swiglu: bool = True,
) -> int:
    """
    FLOPs per *token* for a routed MoE block.

    Components:
      1) Router: linear(hidden_size → num_routed_experts)
         FLOPs = 2 * hidden_size * num_routed_experts

      2) Top-k experts: each expert uses an MLP(hidden → expert_hidden → hidden).
         FLOPs = num_experts_per_token * mlp_layer_flops(1, 1, hidden_size, expert_hidden/hidden_size)

      3) Optional shared experts: modeled as another dense MLP path whose
         effective intermediate dim is (num_shared_experts * expert_hidden_size).
    """
    router = 2 * hidden_size * num_routed_experts

    expert_expansion = expert_hidden_size / float(hidden_size)
    expert = num_experts_per_token * mlp_layer_flops(
        batch_size=1,
        seq_len=1,
        hidden_size=hidden_size,
        expansion=expert_expansion,
        swiglu=swiglu,
    )

    shared = 0
    if num_shared_experts > 0:
        shared_expansion = (expert_hidden_size * num_shared_experts) / float(hidden_size)
        shared = mlp_layer_flops(
            batch_size=1,
            seq_len=1,
            hidden_size=hidden_size,
            expansion=shared_expansion,
            swiglu=swiglu,
        )

    return int(router + expert + shared)


def rmsnorm_flops(batch_size: int, seq_len: int, hidden_size: int) -> int:
    """
    FLOPs for RMSNorm on (batch_size, seq_len, hidden_size).

    Per token:
      - hidden_size squares
      - (hidden_size - 1) adds for the mean
      - 1 divide for mean scaling
      - 1 rsqrt
      - hidden_size multiplies to normalize
      - hidden_size multiplies to apply scale

      → (4 * hidden_size + 2) FLOPs per token
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0:
        return 0
    flops_per_token = 4 * hidden_size + 2
    return batch_size * seq_len * flops_per_token


def layernorm_flops(batch_size: int, seq_len: int, hidden_size: int) -> int:
    """
    FLOPs for LayerNorm on (batch_size, seq_len, hidden_size).

    Per token:
      - hidden_size adds (mean)
      - 1 divide
      - hidden_size subtractions
      - hidden_size multiplies (squaring)
      - hidden_size adds (variance)
      - 1 divide
      - 1 add (epsilon)
      - 1 rsqrt
      - hidden_size multiplies (scale)
      - hidden_size multiplies (gamma)
      - hidden_size adds (beta)

      → (7 * hidden_size + 4) FLOPs per token
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0:
        return 0
    flops_per_token = 7 * hidden_size + 4
    return batch_size * seq_len * flops_per_token


# ============================================================================
# Attention primitives
# ============================================================================


def attn_layer_flops(
    batch_size: int,
    context_len: int,
    hidden_size: int,
    num_heads: int,
    gqa: bool = True,
    gqa_groups: int = 8,
    kv_channels: Optional[int] = None,
    cache_len: int = 0,
    head_dim: Optional[int] = None,
    softmax_cost_per_score: int = 1,
) -> int:
    """
    FLOPs for (potentially incremental) self-attention over a sequence.

    Parameters
    ----------
    context_len:
        Total sequence length *after* adding new tokens.
    cache_len:
        Length of existing KV cache before this call.
        - If cache_len == 0: full-context self-attention over context_len tokens.
        - If cache_len > 0: only (context_len - cache_len) *new* tokens pay Q/K/V
          and attention, while old keys/values are reused.
    """
    B = int(batch_size)
    L_total = int(context_len)
    C = int(cache_len)

    if C < 0:
        raise ValueError("cache_len must be >= 0")
    if L_total < C:
        raise ValueError("context_len must be >= cache_len")

    new_tokens = L_total - C
    if new_tokens <= 0 or batch_size <= 0 or hidden_size <= 0 or num_heads <= 0:
        return 0

    H = int(hidden_size)
    Nh = int(num_heads)

    if head_dim is None:
        if H % Nh != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is not provided")
        head_dim = H // Nh
    else:
        head_dim = int(head_dim)

    kv_head_dim = int(kv_channels) if kv_channels is not None else head_dim
    num_kv_heads = int(gqa_groups) if gqa else Nh

    q_proj_out = Nh * head_dim
    kv_proj_out = num_kv_heads * kv_head_dim

    # Projections for NEW tokens only.
    q_proj = 2 * B * new_tokens * H * q_proj_out
    k_proj = 2 * B * new_tokens * H * kv_proj_out
    v_proj = 2 * B * new_tokens * H * kv_proj_out

    # Rotary position embeddings (approx 4 FLOPs per component).
    rope_q = 4 * B * new_tokens * q_proj_out
    rope_k = 4 * B * new_tokens * kv_proj_out

    total_keys = C + new_tokens  # each new query attends to all previous keys + itself
    qk_flops = 2 * B * Nh * new_tokens * total_keys * head_dim
    softmax_flops = B * Nh * new_tokens * total_keys * int(softmax_cost_per_score)
    attn_v_flops = 2 * B * Nh * new_tokens * total_keys * head_dim

    out_proj = 2 * B * new_tokens * q_proj_out * H

    return int(
        q_proj
        + k_proj
        + v_proj
        + rope_q
        + rope_k
        + qk_flops
        + softmax_flops
        + attn_v_flops
        + out_proj
    )


def sliding_window_attention_flops(
    batch_size: int,
    context_len: int,
    hidden_size: int,
    num_heads: int,
    window_size,
    is_2d: bool = False,
    gqa: bool = True,
    gqa_groups: int = 8,
    kv_channels: Optional[int] = None,
    cache_len: int = 0,
    head_dim: Optional[int] = None,
    softmax_cost_per_score: int = 1,
) -> int:
    """
    FLOPs for *causal* sliding-window (local) attention.

    We treat this as a causal sequence with KV caching:
      - context_len: total length after adding new tokens.
      - cache_len:  cached prefix length.
      - new_tokens = context_len - cache_len.

    Each new token attends to up to `win_tokens` keys, but near the start
    it sees fewer than that. We compute the total number of attention
    scores S = sum_s keys_for_token_s in closed form and reuse that
    for QK^T, softmax, and Attn @ V.

    If is_2d=True, window_size should be (h, w) and
      win_tokens = h * w (local patch grid).
    """
    B = int(batch_size)
    L = int(context_len)
    C = int(cache_len)

    if C < 0:
        raise ValueError("cache_len must be >= 0")
    if L < C:
        raise ValueError("context_len must be >= cache_len")

    N = L - C  # number of newly processed tokens
    if N <= 0 or batch_size <= 0 or hidden_size <= 0 or num_heads <= 0:
        return 0

    if is_2d:
        if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
            raise ValueError("For is_2d=True, window_size must be a (height, width) tuple")
        win_tokens = int(window_size[0]) * int(window_size[1])
    else:
        win_tokens = int(window_size)

    H = int(hidden_size)
    Nh = int(num_heads)

    if head_dim is None:
        if H % Nh != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is not provided")
        head_dim = H // Nh
    else:
        head_dim = int(head_dim)

    kv_head_dim = int(kv_channels) if kv_channels is not None else head_dim
    num_kv_heads = int(gqa_groups) if gqa else Nh

    q_proj_out = Nh * head_dim
    kv_proj_out = num_kv_heads * kv_head_dim

    # Projections and RoPE for NEW tokens only.
    q_proj = 2 * B * N * H * q_proj_out
    k_proj = 2 * B * N * H * kv_proj_out
    v_proj = 2 * B * N * H * kv_proj_out

    rope_q = 4 * B * N * q_proj_out
    rope_k = 4 * B * N * kv_proj_out

    # Total number of attention scores across new tokens:
    #   For token s (0-indexed among new tokens), keys_seen = min(win_tokens, C + s + 1)
    #   Summing this piecewise gives:
    #     if win_tokens <= C+1: S = N * win_tokens
    #     else:
    #       let available_base = C+1
    #       tokens_to_saturate = win_tokens - available_base
    #       fill = min(N, tokens_to_saturate)
    #       S_fill = fill * available_base + fill*(fill-1)/2
    #       S = S_fill + (N - fill) * win_tokens
    available_base = C + 1
    if win_tokens <= available_base:
        S = N * win_tokens
    else:
        tokens_to_saturate = max(0, win_tokens - available_base)
        fill = min(N, tokens_to_saturate)
        S_fill = fill * available_base + (fill * (fill - 1)) // 2
        remaining = N - fill
        S = S_fill + remaining * win_tokens

    qk_flops = 2 * B * Nh * head_dim * S
    softmax_flops = B * Nh * S * int(softmax_cost_per_score)
    attn_v_flops = 2 * B * Nh * head_dim * S

    out_proj = 2 * B * N * q_proj_out * H

    return int(
        q_proj
        + k_proj
        + v_proj
        + rope_q
        + rope_k
        + qk_flops
        + softmax_flops
        + attn_v_flops
        + out_proj
    )


def bidirectional_attention_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    kv_channels: Optional[int] = None,
    gqa: bool = False,
    gqa_groups: Optional[int] = None,
) -> int:
    """
    FLOPs for full *bidirectional* self-attention on `seq_len` tokens.

    This is equivalent to a dense QK^T and Attn@V over all pairs (i, j).
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0 or num_heads <= 0:
        return 0

    B = int(batch_size)
    L = int(seq_len)
    H = int(hidden_size)
    Nh = int(num_heads)

    if head_dim is None:
        if H % Nh != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is not provided")
        head_dim = H // Nh
    else:
        head_dim = int(head_dim)

    num_kv_heads = int(gqa_groups) if gqa and gqa_groups is not None else Nh
    kv_head_dim = int(kv_channels) if kv_channels is not None else head_dim

    q_proj_out = Nh * head_dim
    kv_proj_out = num_kv_heads * kv_head_dim

    q_proj = 2 * B * L * H * q_proj_out
    k_proj = 2 * B * L * H * kv_proj_out
    v_proj = 2 * B * L * H * kv_proj_out

    rope_q = 4 * B * L * q_proj_out
    rope_k = 4 * B * L * kv_proj_out

    qk_flops = 2 * B * Nh * L * L * head_dim
    softmax_flops = B * Nh * L * L
    attn_v_flops = 2 * B * Nh * L * L * head_dim

    out_proj = 2 * B * L * q_proj_out * H

    return int(
        q_proj
        + k_proj
        + v_proj
        + rope_q
        + rope_k
        + qk_flops
        + softmax_flops
        + attn_v_flops
        + out_proj
    )


def chunked_bidirectional_attention_flops(
    batch_size: int,
    total_seq_len: int,
    chunk_seq_len: int,
    num_chunks: int,
    hidden_size: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    kv_channels: Optional[int] = None,
    gqa: bool = False,
    gqa_groups: Optional[int] = None,
) -> int:
    """
    Bidirectional attention FLOPs when tokens are processed in independent chunks.

    Qwen2.5-VL computes flash-attention over each temporal slice separately
    (see ``Qwen2_5_VisionTransformerPretrainedModel``), so the expensive
    QK^T/Attn@V terms only cover ``chunk_seq_len`` tokens at a time while the
    projection work still spans ``total_seq_len`` tokens.
    """
    if (
        batch_size <= 0
        or total_seq_len <= 0
        or chunk_seq_len <= 0
        or num_chunks <= 0
        or hidden_size <= 0
        or num_heads <= 0
    ):
        return 0

    B = int(batch_size)
    total_tokens = int(total_seq_len)
    chunk_tokens = int(chunk_seq_len)
    chunks = int(num_chunks)
    H = int(hidden_size)
    Nh = int(num_heads)

    if head_dim is None:
        if H % Nh != 0:
            raise ValueError("hidden_size must be divisible by num_heads when head_dim is not provided")
        head_dim = H // Nh
    else:
        head_dim = int(head_dim)

    kv_head_dim = int(kv_channels) if kv_channels is not None else head_dim
    num_kv_heads = int(gqa_groups) if gqa and gqa_groups is not None else Nh

    # The ViT processes exactly ``chunk_seq_len * num_chunks`` tokens; if the
    # caller passed a smaller total we still need to account for the full load.
    effective_tokens = chunk_tokens * chunks
    total_tokens = max(total_tokens, effective_tokens)

    q_proj_out = Nh * head_dim
    kv_proj_out = num_kv_heads * kv_head_dim

    q_proj = 2 * B * total_tokens * H * q_proj_out
    k_proj = 2 * B * total_tokens * H * kv_proj_out
    v_proj = 2 * B * total_tokens * H * kv_proj_out

    rope_q = 4 * B * total_tokens * q_proj_out
    rope_k = 4 * B * total_tokens * kv_proj_out

    chunk_scores = chunk_tokens * chunk_tokens * chunks
    qk_flops = 2 * B * Nh * chunk_scores * head_dim
    softmax_flops = B * Nh * chunk_scores
    attn_v_flops = 2 * B * Nh * chunk_scores * head_dim

    out_proj = 2 * B * total_tokens * q_proj_out * H

    return int(
        q_proj
        + k_proj
        + v_proj
        + rope_q
        + rope_k
        + qk_flops
        + softmax_flops
        + attn_v_flops
        + out_proj
    )


def causal_attention_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    gqa: bool = True,
    gqa_groups: int = 8,
    kv_channels: Optional[int] = None,
    head_dim: Optional[int] = None,
    cache_len: int = 0,
) -> int:
    """Thin wrapper around attn_layer_flops for causal self-attention."""
    return attn_layer_flops(
        batch_size=batch_size,
        context_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        gqa=gqa,
        gqa_groups=gqa_groups,
        kv_channels=kv_channels,
        cache_len=cache_len,
        head_dim=head_dim,
    )


# ============================================================================
# Mamba / SSM layer primitive
# ============================================================================


def mamba_layer_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    state_dim: int = 16,
    head_dim: int = 64,
    num_groups: int = 1,
    num_heads: int = 128,
) -> int:
    """
    FLOPs for a Mamba / SSM layer (approximation from published selective-scan
    kernels, dominated by dense projections).

    The formula follows the structure:
      - in_proj:   O(B * L * hidden_size * (2*d_in + 2*num_groups*state_dim + nheads))
      - scan:      O(B * L * d_in * state_dim)
      - out_proj:  O(B * L * d_in * hidden_size)
    where d_in = 2 * hidden_size.
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0:
        return 0

    B = int(batch_size)
    L = int(seq_len)
    H = int(hidden_size)
    d_in = 2 * H
    nheads = num_heads if num_heads else d_in // head_dim

    in_proj = 2 * B * L * H * (2 * d_in + 2 * num_groups * state_dim + nheads)
    scan = 7 * B * L * d_in * state_dim
    out_proj = 2 * B * L * d_in * H

    return in_proj + scan + out_proj


# ============================================================================
# Generic hybrid transformer FLOPs
# ============================================================================


def hybrid_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_full_attn_layers: int,
    num_sliding_attn_layers: int,
    num_mamba_layers: int,
    num_mlp_layers: int,
    # sliding-window args
    window_size=None,
    is_2d: bool = False,
    # Mamba args
    mamba_state_dim: int = 128,
    mamba_head_dim: int = 64,
    mamba_num_groups: int = 8,
    mamba_num_heads: int = 128,
    # attention args
    num_attn_heads: int = 32,
    gqa: bool = True,
    gqa_groups: int = 8,
    kv_channels: Optional[int] = None,
    attn_head_dim: Optional[int] = None,
    # MLP args
    mlp_expansion: float = 4.0,
    swiglu: bool = False,
    # LM head
    vocab_size: int = 0,
    # attention type
    attn_mode: str = "causal",
) -> int:
    """
    Forward FLOPs for a stack of transformer-like layers sharing a common
    (batch_size, seq_len, hidden_size).

    Layer composition:
      - Full attention layers: attention + MLP + 2 * RMSNorm
      - Sliding-window attention layers: sliding attention + MLP + 2 * RMSNorm
      - Mamba layers: Mamba + MLP + 2 * RMSNorm
      - Standalone MLP-only layers: MLP + (implicitly) 2 * RMSNorm

    All norms are counted explicitly using rmsnorm_flops, with 2 per
    "transformer block".
    """
    if batch_size <= 0 or seq_len <= 0 or hidden_size <= 0:
        return 0

    attn_mode = attn_mode.lower()
    if attn_mode not in {"causal", "bidirectional"}:
        raise ValueError("attn_mode must be 'causal' or 'bidirectional'")

    B = int(batch_size)
    L = int(seq_len)
    H = int(hidden_size)

    # --- Full attention ---
    flops_full_attn = 0
    if num_full_attn_layers > 0 and num_attn_heads > 0:
        if attn_mode == "causal":
            flops_full_attn = num_full_attn_layers * causal_attention_flops(
                batch_size=B,
                seq_len=L,
                hidden_size=H,
                num_heads=num_attn_heads,
                gqa=gqa,
                gqa_groups=gqa_groups,
                kv_channels=kv_channels,
                head_dim=attn_head_dim,
                cache_len=0,
            )
        else:
            flops_full_attn = num_full_attn_layers * bidirectional_attention_flops(
                batch_size=B,
                seq_len=L,
                hidden_size=H,
                num_heads=num_attn_heads,
                head_dim=attn_head_dim,
                kv_channels=kv_channels,
                gqa=gqa,
                gqa_groups=gqa_groups,
            )

    # --- Sliding-window attention ---
    flops_sliding_attn = 0
    if num_sliding_attn_layers > 0:
        if window_size is None:
            raise ValueError("window_size must be set when num_sliding_attn_layers > 0")
        flops_sliding_attn = num_sliding_attn_layers * sliding_window_attention_flops(
            batch_size=B,
            context_len=L,
            hidden_size=H,
            num_heads=num_attn_heads,
            window_size=window_size,
            is_2d=is_2d,
            gqa=gqa,
            gqa_groups=gqa_groups,
            kv_channels=kv_channels,
            cache_len=0,
            head_dim=attn_head_dim,
        )

    # --- Mamba layers ---
    flops_mamba = 0
    if num_mamba_layers > 0:
        flops_mamba = num_mamba_layers * mamba_layer_flops(
            batch_size=B,
            seq_len=L,
            hidden_size=H,
            state_dim=mamba_state_dim,
            head_dim=mamba_head_dim,
            num_groups=mamba_num_groups,
            num_heads=mamba_num_heads,
        )

    # --- MLP layers ---
    flops_mlp = 0
    if num_mlp_layers > 0:
        flops_mlp = num_mlp_layers * mlp_layer_flops(
            batch_size=B,
            seq_len=L,
            hidden_size=H,
            expansion=mlp_expansion,
            swiglu=swiglu,
        )

    # --- Norm layers (2 per "transformer block") ---
    total_attention_like_layers = num_full_attn_layers + num_sliding_attn_layers + num_mamba_layers
    transformer_blocks = max(num_mlp_layers, total_attention_like_layers)
    norm_layers = 2 * transformer_blocks
    flops_norm = norm_layers * rmsnorm_flops(B, L, H)

    # --- LM head (if present) ---
    flops_logits = 0
    if vocab_size > 0:
        flops_logits = 2 * B * L * H * vocab_size

    return flops_full_attn + flops_sliding_attn + flops_mamba + flops_mlp + flops_norm + flops_logits


# ============================================================================
# Autoregressive generation with KV caching
# ============================================================================


def autoregressive_generation_flops(
    prompt_len: int,
    num_generated: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mlp_expansion: float,
    vocab_size: int,
    gqa_groups: Optional[int] = None,
    head_dim: Optional[int] = None,
    kv_channels: Optional[int] = None,
    num_full_attn_layers: Optional[int] = None,
    num_sliding_attn_layers: Optional[int] = None,
    sliding_window_size: int = 4096,
    sliding_is_2d: bool = False,
    num_mamba_layers: int = 0,
    mamba_state_dim: int = 128,
    mamba_head_dim: int = 64,
    mamba_num_groups: int = 8,
    mamba_num_heads: int = 128,
    swiglu: bool = False,
) -> int:
    """
    Forward-only FLOPs for autoregressive generation with KV caching.

    We assume the *prompt* up to `prompt_len` tokens has already been processed
    in a separate prefill pass. Here we only account for the additional cost
    of generating `num_generated` new tokens, one at a time, reusing cached
    keys and values.

    For each new token at step t:
      - current_seq_len = prompt_len + t + 1
      - cache_len = current_seq_len - 1
      - each layer contributes:
          norm + (attention or Mamba) + norm + MLP
      - final RMSNorm + LM head + softmax
    """
    if num_generated <= 0 or prompt_len < 0 or hidden_size <= 0 or num_layers <= 0:
        return 0

    H = int(hidden_size)
    L_layers = int(num_layers)
    Nh = int(num_heads)
    kv_groups = gqa_groups if gqa_groups is not None else Nh

    if num_full_attn_layers is None and num_sliding_attn_layers is None:
        num_full_attn_layers = L_layers
        num_sliding_attn_layers = 0
    elif num_full_attn_layers is None:
        num_full_attn_layers = L_layers - int(num_sliding_attn_layers or 0)
    elif num_sliding_attn_layers is None:
        num_sliding_attn_layers = L_layers - int(num_full_attn_layers)

    total = 0

    for step in range(num_generated):
        current_seq_len = prompt_len + step + 1
        cache_len = current_seq_len - 1
        step_flops = 0

        # Full-attention blocks
        for _ in range(max(0, int(num_full_attn_layers))):
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += attn_layer_flops(
                batch_size=1,
                context_len=current_seq_len,
                hidden_size=H,
                num_heads=Nh,
                gqa=True,
                gqa_groups=kv_groups,
                kv_channels=kv_channels,
                cache_len=cache_len,
                head_dim=head_dim,
            )
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += mlp_layer_flops(
                batch_size=1,
                seq_len=1,
                hidden_size=H,
                expansion=mlp_expansion,
                swiglu=swiglu,
            )

        # Sliding-window blocks
        for _ in range(max(0, int(num_sliding_attn_layers))):
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += sliding_window_attention_flops(
                batch_size=1,
                context_len=current_seq_len,
                hidden_size=H,
                num_heads=Nh,
                window_size=sliding_window_size,
                is_2d=sliding_is_2d,
                gqa=True,
                gqa_groups=kv_groups,
                kv_channels=kv_channels,
                cache_len=cache_len,
                head_dim=head_dim,
            )
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += mlp_layer_flops(
                batch_size=1,
                seq_len=1,
                hidden_size=H,
                expansion=mlp_expansion,
                swiglu=swiglu,
            )

        # Mamba blocks
        for _ in range(max(0, int(num_mamba_layers))):
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += mamba_layer_flops(
                batch_size=1,
                seq_len=1,
                hidden_size=H,
                state_dim=mamba_state_dim,
                head_dim=mamba_head_dim,
                num_groups=mamba_num_groups,
                num_heads=mamba_num_heads,
            )
            step_flops += rmsnorm_flops(1, 1, H)
            step_flops += mlp_layer_flops(
                batch_size=1,
                seq_len=1,
                hidden_size=H,
                expansion=mlp_expansion,
                swiglu=swiglu,
            )

        # Final norm + LM head + softmax for this token.
        step_flops += rmsnorm_flops(1, 1, H)
        step_flops += 2 * H * vocab_size
        step_flops += 3 * vocab_size

        total += step_flops

    return total


# ============================================================================
# Simple generic image-slicing and SigLIP-style encoder helpers
# ============================================================================


def adaptive_image_slicing(
    width: int,
    height: int,
    target_width: int = 448,
    target_height: int = 448,
    max_slices: int = 9,
) -> int:
    """
    Adaptive image slicing used in MiniCPM-V 2.6 (section 3.2 of the tech report).

    It computes an ideal number of slices N ≈ (WI*HI)/(Wv*Hv), considers
    N-1, N, N+1, evaluates a score that penalizes aspect-ratio mismatch, and
    chooses the best (m, n) tiling. We return the resulting slice count,
    clipped to `max_slices`.

    This implements the exact strategy described in the paper. See:
    "MiniCPM-V: A GPT-4V Level MLLM on Your Phone", §3.2.
    """
    WI, HI = width, height
    Wv, Hv = target_width, target_height

    if WI <= 0 or HI <= 0:
        return 1

    N = math.ceil((WI * HI) / (Wv * Hv))

    candidates = []
    for test_N in [max(1, N - 1), N, N + 1]:
        for m in range(1, int(math.sqrt(test_N)) + 1):
            if test_N % m == 0:
                n = test_N // m
                candidates.append((m, n, test_N))

    def score_fn(m: int, n: int) -> float:
        try:
            # S(m, n) = -|log((WI/m)/(HI/n)) - log(Wv/Hv)|
            return -abs(math.log((WI / m) / (HI / n)) - math.log(Wv / Hv))
        except (ValueError, ZeroDivisionError):
            return float("-inf")

    best = None
    best_score = float("-inf")
    for m, n, test_N in candidates:
        s = score_fn(m, n)
        if s > best_score:
            best_score = s
            best = (m, n, test_N)

    if best is None:
        return 1
    return min(best[2], max_slices)


def vision_encoder_flops(
    patches: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mlp_expansion: float,
) -> int:
    """
    Generic ViT-style encoder FLOPs (e.g., SigLIP, ViT).

    Each transformer block:
        attention (bidirectional) + MLP + 2 * RMSNorm
    """
    if patches <= 0 or hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
        return 0

    total = 0
    for _ in range(num_layers):
        total += bidirectional_attention_flops(
            batch_size=1,
            seq_len=patches,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        total += mlp_layer_flops(
            batch_size=1,
            seq_len=patches,
            hidden_size=hidden_size,
            expansion=mlp_expansion,
            swiglu=False,
        )
        total += 2 * rmsnorm_flops(1, patches, hidden_size)
    return total



# ============================================================================
# Generic VL helper (vision → merger → language → generation)
# ============================================================================


def mlp_merger_general(
    batch_size: int,
    seq_len: int,
    input_size: int,
    output_size: int,
    expansion: float = 4.0,
    swiglu: bool = False,
) -> int:
    """
    FLOPs for a general 2-layer MLP used as a merger:

        input_size → (expansion * input_size) → output_size

    This is used for simple "vision→language" projection layers.
    """
    if batch_size <= 0 or seq_len <= 0 or input_size <= 0 or output_size <= 0:
        return 0

    B = int(batch_size)
    L = int(seq_len)
    d_in = int(input_size)
    d_mid = int(expansion * input_size)
    d_out = int(output_size)

    up_proj = 2 * B * L * d_in * d_mid
    if swiglu:
        # SwiGLU on the intermediate representation.
        act_cost = 3 * B * L * d_mid
    else:
        # GeLU-like activation.
        act_cost = 4 * B * L * d_mid

    down_proj = 2 * B * L * d_mid * d_out
    return up_proj + act_cost + down_proj


def vision_language_flops(
    # Vision encoder params
    vision_frames: int = 1,
    vision_height: int = 224,
    vision_width: int = 224,
    vision_patch_size: int = 14,
    vision_hidden_size: int = 1024,
    num_full_attn_layers_vision: int = 12,
    num_sliding_attn_layers_vision: int = 0,
    num_mamba_layers_vision: int = 0,
    num_mlp_layers_vision: int = 12,
    vision_window_size=None,
    vision_is_2d: bool = False,
    vision_num_attn_heads: int = 32,
    vision_gqa: bool = True,
    vision_gqa_groups: int = 8,
    vision_kv_channels: Optional[int] = None,
    vision_mlp_expansion: float = 4.0,
    vision_head_dim: Optional[int] = None,
    # Merger
    merger: bool = True,
    merger_mlp_expansion: float = 4.0,
    # Language model params
    lang_seq_len: int = 1024,
    lang_hidden_size: int = 2048,
    num_full_attn_layers_lang: int = 24,
    num_sliding_attn_layers_lang: int = 0,
    num_mamba_layers_lang: int = 0,
    num_mlp_layers_lang: int = 24,
    lang_window_size=None,
    lang_is_2d: bool = False,
    lang_num_attn_heads: int = 32,
    lang_gqa: bool = True,
    lang_gqa_groups: int = 8,
    lang_kv_channels: Optional[int] = None,
    lang_mlp_expansion: float = 4.0,
    lang_head_dim: Optional[int] = None,
    lang_sliding_is_2d: bool = False,
    # Universal
    swiglu: bool = False,
    vocab_size: int = 256000,
    # Generation
    num_generated: int = 0,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    Generic VL FLOPs helper:

        vision encoder → (optional) merger MLP → language prompt → generation.

    We assume batch_size = 1. The caller is responsible for supplying
    consistent hyperparameters matching a specific model.
    """
    batch_size = 1

    # --- Vision sequence length from patches ---
    if vision_frames <= 0 or vision_patch_size <= 0:
        vision_seq_len = 0
    else:
        patches_per_frame = (vision_height // vision_patch_size) * (vision_width // vision_patch_size)
        vision_seq_len = vision_frames * patches_per_frame

    flops_vision = 0
    if vision_seq_len > 0 and vision_hidden_size > 0:
        flops_vision = hybrid_flops(
            batch_size=batch_size,
            seq_len=vision_seq_len,
            hidden_size=vision_hidden_size,
            num_full_attn_layers=num_full_attn_layers_vision,
            num_sliding_attn_layers=num_sliding_attn_layers_vision,
            num_mamba_layers=num_mamba_layers_vision,
            num_mlp_layers=num_mlp_layers_vision,
            window_size=vision_window_size,
            is_2d=vision_is_2d,
            num_attn_heads=vision_num_attn_heads,
            gqa=vision_gqa,
            gqa_groups=vision_gqa_groups,
            kv_channels=vision_kv_channels,
            attn_head_dim=vision_head_dim,
            mlp_expansion=vision_mlp_expansion,
            swiglu=swiglu,
            vocab_size=0,
            attn_mode="bidirectional",
        )

    # --- Merger ---
    flops_merger = 0
    if merger and vision_seq_len > 0:
        flops_merger = mlp_merger_general(
            batch_size=batch_size,
            seq_len=vision_seq_len,
            input_size=vision_hidden_size,
            output_size=lang_hidden_size,
            expansion=merger_mlp_expansion,
            swiglu=swiglu,
        )

    # --- Language prompt (prefill) ---
    prompt_seq_len = lang_seq_len
    flops_lang_prompt = hybrid_flops(
        batch_size=batch_size,
        seq_len=prompt_seq_len,
        hidden_size=lang_hidden_size,
        num_full_attn_layers=num_full_attn_layers_lang,
        num_sliding_attn_layers=num_sliding_attn_layers_lang,
        num_mamba_layers=num_mamba_layers_lang,
        num_mlp_layers=num_mlp_layers_lang,
        window_size=lang_window_size,
        is_2d=lang_is_2d,
        num_attn_heads=lang_num_attn_heads,
        gqa=lang_gqa,
        gqa_groups=lang_gqa_groups,
        kv_channels=lang_kv_channels,
        attn_head_dim=lang_head_dim,
        mlp_expansion=lang_mlp_expansion,
        swiglu=swiglu,
        vocab_size=vocab_size,
        attn_mode="causal",
    )

    # --- Generation ---
    flops_gen = 0
    if num_generated > 0:
        flops_gen = autoregressive_generation_flops(
            prompt_len=prompt_seq_len,
            num_generated=num_generated,
            hidden_size=lang_hidden_size,
            num_layers=num_full_attn_layers_lang + num_sliding_attn_layers_lang + num_mamba_layers_lang,
            num_heads=lang_num_attn_heads,
            mlp_expansion=lang_mlp_expansion,
            vocab_size=vocab_size,
            gqa_groups=lang_gqa_groups if lang_gqa else None,
            head_dim=lang_head_dim,
            kv_channels=lang_kv_channels,
            num_full_attn_layers=num_full_attn_layers_lang,
            num_sliding_attn_layers=num_sliding_attn_layers_lang,
            sliding_window_size=lang_window_size if lang_window_size else 4096,
            sliding_is_2d=lang_sliding_is_2d,
            num_mamba_layers=num_mamba_layers_lang,
            swiglu=swiglu,
        )

    total_fwd = flops_vision + flops_merger + flops_lang_prompt + flops_gen
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": flops_vision * multiplier,
        "merger_flops": flops_merger * multiplier,
        "lang_prompt_flops": flops_lang_prompt * multiplier,
        "gen_flops": flops_gen * multiplier,
        "total_flops": total_fwd * multiplier,
    }

def minicpm_resampler_flops(
    input_tokens: int,
    output_tokens: int,
    input_hidden_size: int = 1152,
    output_hidden_size: int = 3584,
) -> int:
    """
    FLOPs for the MiniCPM-V resampler (1024 → 64 tokens via cross-attention),
    modeled as described in the MiniCPM-V 2.6 paper.

    Pipeline:
      1. kv_proj: linear(input_hidden_size → output_hidden_size) on each input token
      2. Cross-attention: Q from learnable queries (output_tokens),
         K,V from projected inputs (input_tokens)
      3. Output projection back to output_hidden_size
      4. Light normalization cost
    """
    if input_tokens <= 0 or output_tokens <= 0:
        return 0

    B = 1
    Hin = input_hidden_size
    Hout = output_hidden_size

    kv_proj = 2 * B * input_tokens * Hin * Hout

    q_proj = 2 * B * output_tokens * Hout * Hout
    k_proj = 2 * B * input_tokens * Hout * Hout
    v_proj = 2 * B * input_tokens * Hout * Hout

    qk = 2 * B * output_tokens * input_tokens * Hout
    softmax_flops = B * output_tokens * input_tokens
    attn_v = 2 * B * output_tokens * input_tokens * Hout

    out_proj = 2 * B * output_tokens * Hout * Hout

    # Approximate norm cost across output + input tokens.
    norm_flops = layernorm_flops(B, output_tokens + input_tokens, Hout)

    return kv_proj + q_proj + k_proj + v_proj + qk + softmax_flops + attn_v + out_proj + norm_flops


def minicpm_hybrid_generation_flops(
    prompt_len: int,
    num_generated: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mlp_expansion: float,
    vocab_size: int,
    vision_tokens: int = 0,
    gqa_groups: Optional[int] = None,
) -> int:
    """
    MiniCPM generation FLOPs with KV caching.

    The public MiniCPM API (streaming_prefill + streaming_generate)
    builds and reuses a KV cache; this helper simply forwards to
    autoregressive_generation_flops with MiniCPM-like hyperparameters.
    """
    return autoregressive_generation_flops(
        prompt_len=prompt_len,
        num_generated=num_generated,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_expansion=mlp_expansion,
        vocab_size=vocab_size,
        gqa_groups=gqa_groups if gqa_groups is not None else num_heads,
        head_dim=hidden_size // num_heads,
        kv_channels=hidden_size // num_heads,
        num_full_attn_layers=num_layers,
        num_sliding_attn_layers=0,
        sliding_window_size=4096,
        sliding_is_2d=False,
        num_mamba_layers=0,
        swiglu=True,
    )




# ============================================================================
# Qwen2.5-VL-7B model-specific FLOPs
# ============================================================================

# These constants come from the official config.json and Qwen2.5-VL tech report.
#   Text: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ/blob/main/config.json
#   Vision: Qwen2.5-VL blog + paper sections 3.1–3.2.

# Text / LLM
QWEN25_VL7B_TEXT_HIDDEN = 3584
QWEN25_VL7B_TEXT_LAYERS = 28
QWEN25_VL7B_TEXT_HEADS = 28
QWEN25_VL7B_TEXT_KV_HEADS = 4
QWEN25_VL7B_TEXT_INTERMEDIATE = 18944
QWEN25_VL7B_TEXT_MLP_EXPANSION = QWEN25_VL7B_TEXT_INTERMEDIATE / QWEN25_VL7B_TEXT_HIDDEN
QWEN25_VL7B_VOCAB = 152064

# Vision encoder (ViT-style with 3D patching)
QWEN25_VL7B_VISION_HIDDEN = 1280
QWEN25_VL7B_VISION_LAYERS = 32
QWEN25_VL7B_VISION_HEADS = 16
# Intermediate from Qwen2.5-VL docs (ViT variant)
QWEN25_VL7B_VISION_INTERMEDIATE = 3420
QWEN25_VL7B_VISION_PATCH = 14           # spatial patch size
QWEN25_VL7B_VISION_TEMPORAL = 2         # frames per temporal patch
QWEN25_VL7B_VISION_MERGE = 2            # 2 x 2 spatial grouping
QWEN25_VL7B_VISION_OUT_HIDDEN = QWEN25_VL7B_TEXT_HIDDEN
QWEN25_VL7B_VISION_FULL_ATTENTION_LAYERS = 4
QWEN25_VL7B_VISION_WINDOW_PIXELS = 112
# Sliding attention operates on tokens *after* the 2×2 spatial merger.
QWEN25_VL7B_VISION_WINDOW_TOKENS_SIDE = (
    QWEN25_VL7B_VISION_WINDOW_PIXELS
    // (QWEN25_VL7B_VISION_PATCH * QWEN25_VL7B_VISION_MERGE)
)
QWEN25_VL7B_VISION_PROMPT_OVERHEAD = 2
QWEN25_VL7B_MIN_PIXELS = 56 * 56
QWEN25_VL7B_MAX_PIXELS = 3584 * 3584


def _qwen25_vision_pipeline_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
) -> Dict[str, int]:
    """Vision FLOPs for Qwen2.5-VL-7B, matching the image processor."""
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return {
            "vision_flops": 0,
            "vision_tokens": QWEN25_VL7B_VISION_PROMPT_OVERHEAD,
            "vision_tokens_droppable": 0,
            "vision_prompt_overhead": QWEN25_VL7B_VISION_PROMPT_OVERHEAD,
            "vision_seq_len": 0,
            "resized_height": 0,
            "resized_width": 0,
            "patches_per_frame": 0,
            "temporal_segments": 0,
        }

    if QWEN25_VL7B_MIN_PIXELS <= 0 or QWEN25_VL7B_MAX_PIXELS <= 0:
        raise ValueError("Invalid dynamic resize bounds for Qwen2.5-VL vision FLOPs.")

    h_resized, w_resized = _qwen2_vl_dynamic_resize(
        height=vision_height,
        width=vision_width,
        patch_size=QWEN25_VL7B_VISION_PATCH,
        merge_size=QWEN25_VL7B_VISION_MERGE,
        min_pixels=QWEN25_VL7B_MIN_PIXELS,
        max_pixels=QWEN25_VL7B_MAX_PIXELS,
    )

    in_channels = 3
    patch = QWEN25_VL7B_VISION_PATCH
    temporal_patch = QWEN25_VL7B_VISION_TEMPORAL

    patches_h = h_resized // patch
    patches_w = w_resized // patch
    patches_per_frame = patches_h * patches_w

    padded_frames = vision_frames
    remainder = vision_frames % temporal_patch
    if remainder:
        padded_frames += temporal_patch - remainder
    temporal_segments = max(1, padded_frames // temporal_patch)
    vision_seq_len = temporal_segments * patches_per_frame

    # Conv3D patch embedding: kernel_volume = temporal_patch * patch^2
    kernel_volume = temporal_patch * (patch ** 2)
    patch_embed_flops = 2 * vision_seq_len * kernel_volume * in_channels * QWEN25_VL7B_VISION_HIDDEN

    # Vision transformer: mix of full and sliding-window attention blocks.
    full_blocks = QWEN25_VL7B_VISION_FULL_ATTENTION_LAYERS
    sliding_blocks = QWEN25_VL7B_VISION_LAYERS - full_blocks
    head_dim = QWEN25_VL7B_VISION_HIDDEN // QWEN25_VL7B_VISION_HEADS

    mlp_expansion = QWEN25_VL7B_VISION_INTERMEDIATE / QWEN25_VL7B_VISION_HIDDEN
    mlp_per_layer = mlp_layer_flops(
        batch_size=1,
        seq_len=vision_seq_len,
        hidden_size=QWEN25_VL7B_VISION_HIDDEN,
        expansion=mlp_expansion,
        swiglu=True,
    )
    mlp_flops = QWEN25_VL7B_VISION_LAYERS * mlp_per_layer

    norm_per_layer = rmsnorm_flops(1, vision_seq_len, QWEN25_VL7B_VISION_HIDDEN)
    norm_flops = 2 * QWEN25_VL7B_VISION_LAYERS * norm_per_layer

    full_attn_flops = 0
    if full_blocks > 0 and patches_per_frame > 0:
        full_attn_per_layer = chunked_bidirectional_attention_flops(
            batch_size=1,
            total_seq_len=vision_seq_len,
            chunk_seq_len=patches_per_frame,
            num_chunks=temporal_segments,
            hidden_size=QWEN25_VL7B_VISION_HIDDEN,
            num_heads=QWEN25_VL7B_VISION_HEADS,
            head_dim=head_dim,
            kv_channels=head_dim,
            gqa=False,
        )
        full_attn_flops = full_blocks * full_attn_per_layer

    sliding_attn_flops = 0
    if sliding_blocks > 0:
        window_side = QWEN25_VL7B_VISION_WINDOW_TOKENS_SIDE
        if window_side <= 0:
            raise ValueError(
                "Qwen2.5-VL sliding attention window resolves to zero tokens. "
                "Check QWEN25_VL7B_VISION_WINDOW_PIXELS / (patch_size * merge_size)."
            )
        window_shape = (window_side, window_side)
        sliding_per_layer = sliding_window_attention_flops(
            batch_size=1,
            context_len=vision_seq_len,
            hidden_size=QWEN25_VL7B_VISION_HIDDEN,
            num_heads=QWEN25_VL7B_VISION_HEADS,
            window_size=window_shape,
            is_2d=True,
            gqa=False,
            gqa_groups=QWEN25_VL7B_VISION_HEADS,
            kv_channels=head_dim,
            cache_len=0,
            head_dim=head_dim,
        )
        sliding_attn_flops = sliding_blocks * sliding_per_layer

    vision_transformer_flops = full_attn_flops + sliding_attn_flops + mlp_flops + norm_flops

    # Patch merger: 2×2 spatial tokens → 1 output token via 2-layer MLP.
    merge_size = QWEN25_VL7B_VISION_MERGE
    merged_tokens = max(1, vision_seq_len // (merge_size ** 2))
    merge_dim = QWEN25_VL7B_VISION_HIDDEN * (merge_size ** 2)

    # LayerNorm over the merge_dim representation, then MLP(merge_dim → merge_dim → out_hidden).
    ln_flops = layernorm_flops(1, merged_tokens, merge_dim)
    up_proj = 2 * merged_tokens * merge_dim * merge_dim
    gelu_cost = 4 * merged_tokens * merge_dim
    down_proj = 2 * merged_tokens * merge_dim * QWEN25_VL7B_VISION_OUT_HIDDEN
    merger_flops = ln_flops + up_proj + gelu_cost + down_proj

    vision_flops = patch_embed_flops + vision_transformer_flops + merger_flops
    return {
        "vision_flops": vision_flops,
        "vision_tokens": merged_tokens + QWEN25_VL7B_VISION_PROMPT_OVERHEAD,
        "vision_tokens_droppable": merged_tokens,
        "vision_prompt_overhead": QWEN25_VL7B_VISION_PROMPT_OVERHEAD,
        "vision_seq_len": vision_seq_len,
        "resized_height": h_resized,
        "resized_width": w_resized,
        "patches_per_frame": patches_per_frame,
        "temporal_segments": temporal_segments,
    }


def _qwen25_lm_flops_for_prompt_and_gen(
    prompt_len: int,
    num_generated: int,
) -> Dict[str, int]:
    """LLM prefill + generation FLOPs for Qwen2.5-VL-7B's text stack."""
    head_dim = QWEN25_VL7B_TEXT_HIDDEN // QWEN25_VL7B_TEXT_HEADS

    # Prefill (prompt)
    prompt_flops = hybrid_flops(
        batch_size=1,
        seq_len=prompt_len,
        hidden_size=QWEN25_VL7B_TEXT_HIDDEN,
        num_full_attn_layers=QWEN25_VL7B_TEXT_LAYERS,
        num_sliding_attn_layers=0,
        num_mamba_layers=0,
        num_mlp_layers=QWEN25_VL7B_TEXT_LAYERS,
        window_size=None,
        is_2d=False,
        num_attn_heads=QWEN25_VL7B_TEXT_HEADS,
        gqa=True,
        gqa_groups=QWEN25_VL7B_TEXT_KV_HEADS,
        kv_channels=head_dim,
        attn_head_dim=head_dim,
        mlp_expansion=QWEN25_VL7B_TEXT_MLP_EXPANSION,
        swiglu=True,
        vocab_size=QWEN25_VL7B_VOCAB,
        attn_mode="causal",
    )

    # Generation
    gen_flops = 0
    if num_generated > 0:
        gen_flops = autoregressive_generation_flops(
            prompt_len=prompt_len,
            num_generated=num_generated,
            hidden_size=QWEN25_VL7B_TEXT_HIDDEN,
            num_layers=QWEN25_VL7B_TEXT_LAYERS,
            num_heads=QWEN25_VL7B_TEXT_HEADS,
            mlp_expansion=QWEN25_VL7B_TEXT_MLP_EXPANSION,
            vocab_size=QWEN25_VL7B_VOCAB,
            gqa_groups=QWEN25_VL7B_TEXT_KV_HEADS,
            head_dim=head_dim,
            kv_channels=head_dim,
            num_full_attn_layers=QWEN25_VL7B_TEXT_LAYERS,
            num_sliding_attn_layers=0,
            sliding_window_size=0,
            sliding_is_2d=False,
            num_mamba_layers=0,
            swiglu=True,
        )

    return {"lang_prompt_flops": prompt_flops, "gen_flops": gen_flops}


def qwen_2_5_vl_7b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for Qwen2.5-VL-7B-Instruct.

    Args
    ----
    vision_frames:
        Number of video frames (1 for a single image, 0 for text-only).
    vision_height, vision_width:
        Input resolution before internal resizing. We assume it is already
        chosen such that integer patching with patch_size = 14 is valid.
    lang_prompt_len:
        Total number of tokens in the LLM prompt *including* the vision
        placeholders that correspond to merged vision tokens.
    num_generated:
        Number of autoregressive tokens generated after the prompt.
    """
    vision_info = _qwen25_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )
    vision_flops = vision_info["vision_flops"]
    vision_tokens = vision_info["vision_tokens"]

    text_tokens = max(lang_prompt_len - vision_tokens, 0)
    prompt_len = lang_prompt_len

    lm_info = _qwen25_lm_flops_for_prompt_and_gen(prompt_len, num_generated)
    prompt_flops = lm_info["lang_prompt_flops"]
    gen_flops = lm_info["gen_flops"]

    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,
    }


# ============================================================================
# TimeChat (Qwen2.5-VL with token dropping) FLOPs
# ============================================================================


def timechat_online_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    tokens_dropped: int = 0,
    tokens_total_before_drop: Optional[int] = None,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for TimeChat-style token-dropping on top of Qwen2.5-VL.

    lang_prompt_len:
        Length of the prompt (vision placeholders + text) *before dropping*.
    tokens_total_before_drop:
        Number of droppable tokens (typically a subset of vision tokens).
        If None, we assume all vision tokens are initially droppable.
    tokens_dropped:
        How many of those droppable tokens the policy actually drops.
    """
    # First compute the base vision pipeline (before any dropping).
    vision_info = _qwen25_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )
    vision_flops = vision_info["vision_flops"]
    vision_tokens = vision_info["vision_tokens"]
    droppable_ref = vision_info.get("vision_tokens_droppable", vision_tokens)
    prompt_overhead = vision_info.get("vision_prompt_overhead", 0)

    # Determine how many vision tokens the dropping logic is allowed to touch.
    droppable_total = droppable_ref if tokens_total_before_drop is None else max(tokens_total_before_drop, 0)
    droppable_total = min(droppable_total, droppable_ref)

    tokens_dropped = max(min(tokens_dropped, droppable_total), 0)
    remaining_droppable = droppable_total - tokens_dropped
    non_droppable_vision = max(droppable_ref - droppable_total, 0) + prompt_overhead
    effective_vision_tokens = remaining_droppable + non_droppable_vision
    effective_vision_tokens_for_prompt = effective_vision_tokens

    # Text tokens in the original prompt are lang_prompt_len - droppable_total.
    baseline_text_tokens = max(lang_prompt_len - (droppable_total + prompt_overhead), 0)
    prompt_len_after_drop_total = baseline_text_tokens + effective_vision_tokens
    prompt_len_after_drop_effective = prompt_len_after_drop_total

    lm_info = _qwen25_lm_flops_for_prompt_and_gen(
        prompt_len=prompt_len_after_drop_effective,
        num_generated=num_generated,
    )
    prompt_flops = lm_info["lang_prompt_flops"]
    gen_flops = lm_info["gen_flops"]

    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": vision_tokens,
        "vision_tokens_kept": effective_vision_tokens,
        "vision_tokens_dropped": tokens_dropped,
        "vision_tokens_droppable": droppable_total,
        "prompt_tokens_total": prompt_len_after_drop_total,
        "prompt_tokens_effective": prompt_len_after_drop_effective,
        "text_tokens": baseline_text_tokens,
    }


## ============================================================================
# MiniCPM-V-2.6 FLOPs (SigLIP + resampler + Qwen2-style LLM)
# ============================================================================
# Sources:
#   - MiniCPM-o-2_6 HF config (openbmb/MiniCPM-o-2_6, config.json):
#       * text: hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
#         num_key_value_heads=4, intermediate_size=18944, vocab_size=151700.
#       * vision_config: hidden_size=1152, num_hidden_layers=27,
#         num_attention_heads=16, patch_size=14.
#   - MiniCPM-V 2.6 tech report:
#       * 448x448 slices, at most 9 slices.
#       * Each slice → 1024 vision tokens (32x32 patches at 14x14).
#       * Resampler: 1024 → 64 tokens per slice via cross-attention into
#         the LLM hidden space.

MINICPM26_VISION_HIDDEN = 1152
MINICPM26_VISION_LAYERS = 27
MINICPM26_VISION_HEADS = 16
# intermediate_size = 4304 → exact expansion = 4304 / 1152 ≈ 3.736
MINICPM26_VISION_MLP_EXPANSION = 4304 / MINICPM26_VISION_HIDDEN
MINICPM26_PATCH_TOKENS_PER_SLICE = 1024  # 32x32 patches for a 448x448 slice
MINICPM26_MAX_SLICES = 9
MINICPM26_RESAMPLER_QUERIES = 64         # query_num in config

MINICPM26_LM_HIDDEN = 3584
MINICPM26_LM_LAYERS = 28
MINICPM26_LM_HEADS = 28
MINICPM26_LM_KV_HEADS = 4
MINICPM26_LM_INTERMEDIATE = 18944
MINICPM26_LM_MLP_EXPANSION = MINICPM26_LM_INTERMEDIATE / MINICPM26_LM_HIDDEN
MINICPM26_VOCAB = 151700


def minicpm_resampler_flops(
    input_tokens: int,
    output_tokens: int,
    input_hidden_size: int = MINICPM26_VISION_HIDDEN,
    output_hidden_size: int = MINICPM26_LM_HIDDEN,
) -> int:
    """
    FLOPs for the MiniCPM-V 2.6 resampler (≈1024 → 64 tokens per slice),
    modeled as a single cross-attention layer.

    Pipeline (per resampler.py and the MiniCPM-o 2.6 paper):
      1. kv_proj: linear(input_hidden_size → output_hidden_size) on each vision token
      2. Cross-attention:
         - Q from learnable queries (output_tokens, in output_hidden_size)
         - K,V from projected inputs (input_tokens, in output_hidden_size)
      3. Output projection: linear(output_hidden_size → output_hidden_size)
      4. LayerNorm over input + output tokens
    """
    if input_tokens <= 0 or output_tokens <= 0:
        return 0

    B = 1
    Hin = int(input_hidden_size)
    Hout = int(output_hidden_size)

    # Step 1: project visual tokens into the LLM-hidden space (K/V base).
    kv_proj = 2 * B * input_tokens * Hin * Hout

    # Step 2: standard cross-attention in output_hidden_size.
    q_proj = 2 * B * output_tokens * Hout * Hout
    k_proj = 2 * B * input_tokens * Hout * Hout
    v_proj = 2 * B * input_tokens * Hout * Hout

    qk = 2 * B * output_tokens * input_tokens * Hout
    softmax_flops = B * output_tokens * input_tokens
    attn_v = 2 * B * output_tokens * input_tokens * Hout

    # Step 3: output projection.
    out_proj = 2 * B * output_tokens * Hout * Hout

    # Step 4: approximate LayerNorm cost over input+output tokens.
    norm_flops = layernorm_flops(B, output_tokens + input_tokens, Hout)

    return kv_proj + q_proj + k_proj + v_proj + qk + softmax_flops + attn_v + out_proj + norm_flops


def minicpm_hybrid_generation_flops(
    prompt_len: int,
    num_generated: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mlp_expansion: float,
    vocab_size: int,
    vision_tokens: int = 0,
    gqa_groups: Optional[int] = None,
) -> int:
    """
    MiniCPM generation FLOPs with KV caching.

    The public MiniCPM-o 2.6 code (streaming_prefill + streaming_generate)
    builds and reuses a KV cache. We model the text stack as a standard
    Qwen-style transformer:
      - all layers are full attention (no sliding window),
      - SwiGLU MLP with expansion = mlp_expansion.
    """
    return autoregressive_generation_flops(
        prompt_len=prompt_len,
        num_generated=num_generated,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_expansion=mlp_expansion,
        vocab_size=vocab_size,
        gqa_groups=gqa_groups if gqa_groups is not None else num_heads,
        head_dim=hidden_size // num_heads,
        kv_channels=hidden_size // num_heads,
        num_full_attn_layers=num_layers,
        num_sliding_attn_layers=0,
        sliding_window_size=4096,
        sliding_is_2d=False,
        num_mamba_layers=0,
        swiglu=True,
    )


def minicpm_v_2_6_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for MiniCPM-V-2.6 using:
      - adaptive slicing → up to 9 slices per image, each resized to 448×448,
      - a SigLIP-style vision encoder (27 layers, 1152 hidden, 16 heads),
      - a cross-attention resampler (1024 → 64 tokens per slice),
      - a Qwen2.5-7B-style LLM (28 layers, 3584 hidden, 28 heads, 4 KV heads).

    Args
    ----
    vision_frames:
        Number of frames (1 for a single image, >1 for video).
    vision_height, vision_width:
        Input spatial resolution before slicing/resizing.
    lang_prompt_len:
        Number of *text* tokens before appending compressed vision tokens.
    num_generated:
        Number of autoregressive tokens generated after the prompt.
    """
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        # Text-only path: use the same MiniCPM LLM spec as below (no vision).
        prompt_len = lang_prompt_len
        head_dim = MINICPM26_LM_HIDDEN // MINICPM26_LM_HEADS
        prompt_flops = hybrid_flops(
            batch_size=1,
            seq_len=prompt_len,
            hidden_size=MINICPM26_LM_HIDDEN,
            num_full_attn_layers=MINICPM26_LM_LAYERS,
            num_sliding_attn_layers=0,
            num_mamba_layers=0,
            num_mlp_layers=MINICPM26_LM_LAYERS,
            window_size=None,
            is_2d=False,
            num_attn_heads=MINICPM26_LM_HEADS,
            gqa=True,
            gqa_groups=MINICPM26_LM_KV_HEADS,
            kv_channels=head_dim,
            attn_head_dim=head_dim,
            mlp_expansion=MINICPM26_LM_MLP_EXPANSION,
            swiglu=True,
            vocab_size=MINICPM26_VOCAB,
            attn_mode="causal",
        )
        generation_flops = 0
        if num_generated > 0:
            generation_flops = minicpm_hybrid_generation_flops(
                prompt_len=prompt_len,
                num_generated=num_generated,
                hidden_size=MINICPM26_LM_HIDDEN,
                num_layers=MINICPM26_LM_LAYERS,
                num_heads=MINICPM26_LM_HEADS,
                mlp_expansion=MINICPM26_LM_MLP_EXPANSION,
                vocab_size=MINICPM26_VOCAB,
                vision_tokens=0,
                gqa_groups=MINICPM26_LM_KV_HEADS,
            )
        vision_encoding_flops = 0
        compression_flops = 0
    else:
        # 1) Adaptive slicing (per frame), each slice resized to 448×448.
        target_size = 448
        slices_per_frame = adaptive_image_slicing(
            width=vision_width,
            height=vision_height,
            target_width=target_size,
            target_height=target_size,
            max_slices=MINICPM26_MAX_SLICES,
        )
        total_slices = slices_per_frame * vision_frames

        patches_per_slice = MINICPM26_PATCH_TOKENS_PER_SLICE
        vision_tokens_total = total_slices * patches_per_slice

        # 2) SigLIP encoder per slice.
        vision_flops_per_slice = vision_encoder_flops(
            patches=patches_per_slice,
            hidden_size=MINICPM26_VISION_HIDDEN,
            num_layers=MINICPM26_VISION_LAYERS,
            num_heads=MINICPM26_VISION_HEADS,
            mlp_expansion=MINICPM26_VISION_MLP_EXPANSION,
        )
        vision_encoding_flops = total_slices * vision_flops_per_slice

        # 3) Resampler compresses 1024 → 64 tokens per slice (all slices together).
        total_output_tokens = total_slices * MINICPM26_RESAMPLER_QUERIES
        compression_flops = minicpm_resampler_flops(
            input_tokens=vision_tokens_total,
            output_tokens=total_output_tokens,
            input_hidden_size=MINICPM26_VISION_HIDDEN,
            output_hidden_size=MINICPM26_LM_HIDDEN,
        )

        # 4) LLM prompt: text tokens + all compressed vision tokens.
        prompt_len = lang_prompt_len + total_output_tokens

        head_dim = MINICPM26_LM_HIDDEN // MINICPM26_LM_HEADS
        prompt_flops = hybrid_flops(
            batch_size=1,
            seq_len=prompt_len,
            hidden_size=MINICPM26_LM_HIDDEN,
            num_full_attn_layers=MINICPM26_LM_LAYERS,
            num_sliding_attn_layers=0,
            num_mamba_layers=0,
            num_mlp_layers=MINICPM26_LM_LAYERS,
            window_size=None,
            is_2d=False,
            num_attn_heads=MINICPM26_LM_HEADS,
            gqa=True,
            gqa_groups=MINICPM26_LM_KV_HEADS,
            kv_channels=head_dim,
            attn_head_dim=head_dim,
            mlp_expansion=MINICPM26_LM_MLP_EXPANSION,
            swiglu=True,
            vocab_size=MINICPM26_VOCAB,
            attn_mode="causal",
        )

        generation_flops = 0
        if num_generated > 0:
            generation_flops = minicpm_hybrid_generation_flops(
                prompt_len=prompt_len,
                num_generated=num_generated,
                hidden_size=MINICPM26_LM_HIDDEN,
                num_layers=MINICPM26_LM_LAYERS,
                num_heads=MINICPM26_LM_HEADS,
                mlp_expansion=MINICPM26_LM_MLP_EXPANSION,
                vocab_size=MINICPM26_VOCAB,
                vision_tokens=total_output_tokens,
                gqa_groups=MINICPM26_LM_KV_HEADS,
            )

    total_fwd = vision_encoding_flops + compression_flops + prompt_flops + generation_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_encoding_flops": vision_encoding_flops * multiplier,
        "compression_flops": compression_flops * multiplier,
        "prompt_flops": prompt_flops * multiplier,
        "generation_flops": generation_flops * multiplier,
        "total_flops": total_fwd * multiplier,
    }


# ============================================================================
# M3-Agent (control & memorization) FLOPs
# ============================================================================

def m3_agent_control_flops(
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for the M3-Agent control stage.

    This stage uses a text-only Qwen-style LLM; the constants below are
    taken from the public M3-Agent config (hidden_size=5120, 64 layers,
    64 heads, 8 KV heads, intermediate_size=25600, vocab≈151936).
    """
    hidden = 5120
    layers = 64
    heads = 64
    kv_heads = 8
    inter = 25600
    mlp_expansion = inter / hidden
    vocab = 151936
    head_dim = hidden // heads

    prompt_flops = hybrid_flops(
        batch_size=1,
        seq_len=lang_prompt_len,
        hidden_size=hidden,
        num_full_attn_layers=layers,
        num_sliding_attn_layers=0,
        num_mamba_layers=0,
        num_mlp_layers=layers,
        window_size=None,
        is_2d=False,
        num_attn_heads=heads,
        gqa=True,
        gqa_groups=kv_heads,
        kv_channels=head_dim,
        attn_head_dim=head_dim,
        mlp_expansion=mlp_expansion,
        swiglu=False,  # SiLU activations, but MLP is not SwiGLU here
        vocab_size=vocab,
        attn_mode="causal",
    )

    gen_flops = 0
    if num_generated > 0:
        gen_flops = autoregressive_generation_flops(
            prompt_len=lang_prompt_len,
            num_generated=num_generated,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            mlp_expansion=mlp_expansion,
            vocab_size=vocab,
            gqa_groups=kv_heads,
            head_dim=head_dim,
            kv_channels=head_dim,
            num_full_attn_layers=layers,
            num_sliding_attn_layers=0,
            swiglu=False,
        )

    total_fwd = prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": 0,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
    }


def m3_agent_memorization_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for the M3-Agent memorization stage, which uses a Qwen2.5-VL-7B-like
    vision-language stack.

    lang_prompt_len counts the total number of tokens seen by the LLM during
    prefill (text + vision placeholders), mirroring the public
    ``qwen_2_5_vl_7b_flops`` helper.
    """
    # Reuse the Qwen2.5-VL vision pipeline.
    vision_info = _qwen25_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )
    vision_flops = vision_info["vision_flops"]
    vision_tokens = vision_info["vision_tokens"]

    text_tokens = max(lang_prompt_len - vision_tokens, 0)
    prompt_len = lang_prompt_len

    lm_info = _qwen25_lm_flops_for_prompt_and_gen(prompt_len, num_generated)
    prompt_flops = lm_info["lang_prompt_flops"]
    gen_flops = lm_info["gen_flops"]

    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,
    }


# ============================================================================
# GLM-4.5V FLOPs (MoE text, ViT vision)
# ============================================================================

# GLM-4.5V constants below are taken from the public GLM-4.5V config and docs.
GLM45V_VISION_HIDDEN = 1536
GLM45V_VISION_DEPTH = 24
GLM45V_VISION_HEADS = 12
GLM45V_VISION_INTERMEDIATE = 10944
GLM45V_VISION_PATCH = 14
GLM45V_VISION_TEMPORAL = 2
GLM45V_VISION_SPATIAL_MERGE = 2
GLM45V_VISION_OUT_HIDDEN = 4096  # text hidden_size
# Vision block MLP expands to out_hidden_size (see HF Glm4vMoeisionMlp).
GLM45V_VISION_BLOCK_INTERMEDIATE = GLM45V_VISION_OUT_HIDDEN

GLM45V_TEXT_HIDDEN = 4096
GLM45V_TEXT_LAYERS = 46
GLM45V_TEXT_HEADS = 96
GLM45V_TEXT_KV_HEADS = 8
GLM45V_TEXT_INTERMEDIATE = 10944
GLM45V_TEXT_MOE_INTERMEDIATE = 1408
GLM45V_TEXT_DENSE_LAYERS = 1
GLM45V_TEXT_NUM_EXPERTS_PER_TOK = 8
GLM45V_TEXT_NUM_ROUTED_EXPERTS = 128
GLM45V_TEXT_NUM_SHARED_EXPERTS = 1
GLM45V_TEXT_VOCAB = 151552


def _glm45_language_prompt_flops(
    seq_len: int,
) -> int:
    """Forward FLOPs for GLM-4.5V MoE language stack on a prompt of length seq_len."""
    if seq_len <= 0:
        return 0

    hidden = GLM45V_TEXT_HIDDEN
    layers = GLM45V_TEXT_LAYERS
    heads = GLM45V_TEXT_HEADS
    kv_heads = GLM45V_TEXT_KV_HEADS
    dense_layers = GLM45V_TEXT_DENSE_LAYERS
    dense_inter = GLM45V_TEXT_INTERMEDIATE
    moe_inter = GLM45V_TEXT_MOE_INTERMEDIATE

    head_dim = hidden // heads
    moe_token_cost = moe_mlp_token_flops(
        hidden_size=hidden,
        expert_hidden_size=moe_inter,
        num_experts_per_token=GLM45V_TEXT_NUM_EXPERTS_PER_TOK,
        num_routed_experts=GLM45V_TEXT_NUM_ROUTED_EXPERTS,
        num_shared_experts=GLM45V_TEXT_NUM_SHARED_EXPERTS,
        swiglu=True,
    )
    dense_expansion = dense_inter / hidden

    total = 0
    for layer_idx in range(layers):
        total += rmsnorm_flops(1, seq_len, hidden)
        total += causal_attention_flops(
            batch_size=1,
            seq_len=seq_len,
            hidden_size=hidden,
            num_heads=heads,
            gqa=True,
            gqa_groups=kv_heads,
            kv_channels=head_dim,
            head_dim=head_dim,
            cache_len=0,
        )
        total += rmsnorm_flops(1, seq_len, hidden)
        if layer_idx < dense_layers:
            total += mlp_layer_flops(
                batch_size=1,
                seq_len=seq_len,
                hidden_size=hidden,
                expansion=dense_expansion,
                swiglu=True,
            )
        else:
            total += seq_len * moe_token_cost

    # Final norm (logits handled elsewhere)
    total += rmsnorm_flops(1, seq_len, hidden)
    return total


def _glm45_generation_flops(
    prompt_len: int,
    num_generated: int,
) -> int:
    if num_generated <= 0:
        return 0

    hidden = GLM45V_TEXT_HIDDEN
    layers = GLM45V_TEXT_LAYERS
    heads = GLM45V_TEXT_HEADS
    kv_heads = GLM45V_TEXT_KV_HEADS
    dense_layers = GLM45V_TEXT_DENSE_LAYERS
    dense_inter = GLM45V_TEXT_INTERMEDIATE
    moe_inter = GLM45V_TEXT_MOE_INTERMEDIATE
    vocab = GLM45V_TEXT_VOCAB

    head_dim = hidden // heads
    moe_token_cost = moe_mlp_token_flops(
        hidden_size=hidden,
        expert_hidden_size=moe_inter,
        num_experts_per_token=GLM45V_TEXT_NUM_EXPERTS_PER_TOK,
        num_routed_experts=GLM45V_TEXT_NUM_ROUTED_EXPERTS,
        num_shared_experts=GLM45V_TEXT_NUM_SHARED_EXPERTS,
        swiglu=True,
    )
    dense_expansion = dense_inter / hidden

    total = 0
    for step in range(num_generated):
        current_len = prompt_len + step + 1
        cache_len = current_len - 1
        for layer_idx in range(layers):
            total += rmsnorm_flops(1, 1, hidden)
            total += attn_layer_flops(
                batch_size=1,
                context_len=current_len,
                hidden_size=hidden,
                num_heads=heads,
                gqa=True,
                gqa_groups=kv_heads,
                kv_channels=head_dim,
                cache_len=cache_len,
                head_dim=head_dim,
            )
            total += rmsnorm_flops(1, 1, hidden)
            if layer_idx < dense_layers:
                total += mlp_layer_flops(
                    batch_size=1,
                    seq_len=1,
                    hidden_size=hidden,
                    expansion=dense_expansion,
                    swiglu=True,
                )
            else:
                total += moe_token_cost

        total += rmsnorm_flops(1, 1, hidden)
        total += 2 * hidden * vocab
        total += 3 * vocab

    return total


def glm45v_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for GLM-4.5V (MoE text + ViT visual encoder).

    lang_prompt_len:
        Length of the prompt (text + vision placeholders) the GLM sees.
    """
    if vision_frames > 0 and vision_height > 0 and vision_width > 0:
        in_channels = 3
        patch = GLM45V_VISION_PATCH
        temporal_patch = GLM45V_VISION_TEMPORAL

        patches_per_frame = (vision_height // patch) * (vision_width // patch)
        temporal_segments = max(1, vision_frames // temporal_patch)
        vision_seq_len = temporal_segments * patches_per_frame

        kernel_volume = temporal_patch * (patch ** 2)
        patch_embed_flops = 2 * vision_seq_len * kernel_volume * in_channels * GLM45V_VISION_HIDDEN

        post_embed_norm = rmsnorm_flops(1, vision_seq_len, GLM45V_VISION_HIDDEN)

        head_dim = GLM45V_VISION_HIDDEN // GLM45V_VISION_HEADS
        transformer_flops = hybrid_flops(
            batch_size=1,
            seq_len=vision_seq_len,
            hidden_size=GLM45V_VISION_HIDDEN,
            num_full_attn_layers=GLM45V_VISION_DEPTH,
            num_sliding_attn_layers=0,
            num_mamba_layers=0,
            num_mlp_layers=GLM45V_VISION_DEPTH,
            window_size=None,
            is_2d=False,
            num_attn_heads=GLM45V_VISION_HEADS,
            gqa=False,
            gqa_groups=GLM45V_VISION_HEADS,
            kv_channels=head_dim,
            attn_head_dim=head_dim,
            mlp_expansion=GLM45V_VISION_BLOCK_INTERMEDIATE / GLM45V_VISION_HIDDEN,
            swiglu=True,
            vocab_size=0,
            attn_mode="bidirectional",
        )

        post_layernorm = rmsnorm_flops(1, vision_seq_len, GLM45V_VISION_HIDDEN)

        merge = GLM45V_VISION_SPATIAL_MERGE
        output_tokens = max(1, vision_seq_len // (merge ** 2))
        downsample_flops = (
            2
            * output_tokens
            * (merge ** 2)
            * GLM45V_VISION_HIDDEN
            * GLM45V_VISION_OUT_HIDDEN
        )

        merger_norm = rmsnorm_flops(1, output_tokens, GLM45V_VISION_OUT_HIDDEN)
        merger_proj = 2 * output_tokens * GLM45V_VISION_OUT_HIDDEN * GLM45V_VISION_OUT_HIDDEN
        merger_activation = 4 * output_tokens * GLM45V_VISION_OUT_HIDDEN

        merger_mlp = mlp_layer_flops(
            batch_size=1,
            seq_len=output_tokens,
            hidden_size=GLM45V_VISION_OUT_HIDDEN,
            expansion=GLM45V_VISION_INTERMEDIATE / GLM45V_VISION_OUT_HIDDEN,
            swiglu=True,
        )

        vision_flops = (
            patch_embed_flops
            + post_embed_norm
            + transformer_flops
            + post_layernorm
            + downsample_flops
            + merger_norm
            + merger_proj
            + merger_activation
            + merger_mlp
        )
        vision_tokens = output_tokens
    else:
        vision_flops = 0
        vision_tokens = 0

    text_tokens = max(lang_prompt_len - vision_tokens, 0)
    prompt_seq_len = text_tokens + vision_tokens

    prompt_flops = _glm45_language_prompt_flops(prompt_seq_len)
    prompt_flops += 2 * prompt_seq_len * GLM45V_TEXT_HIDDEN * GLM45V_TEXT_VOCAB

    gen_flops = _glm45_generation_flops(prompt_seq_len, num_generated)
    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_seq_len,
    }


# ============================================================================
# Qwen3 family FLOPs (vision + text, DeepStack)
# ============================================================================
# Sources:
#   - DeepStack paper: "DeepStack: Deep-stacking features for multi-modal LLMs"
#     (arxiv:2406.04334) – 3D ViT, spatial_merge_size=2, multi-layer visual stacks.
#   - Qwen3-Omni-MoE (30B thinker) config: thinker_config.text_config and
#     thinker_config.vision_config from the JSON you provided.
#   - Qwen3-VL 8B config: text_config & vision_config (HF), plus OLLama metadata
#     showing vision.deepstack_visual_indexes = [8, 16, 24].

# Qwen3-Omni 30B A3B thinker config constants (MoE text, DeepStack vision)
Q3O_VISION_HIDDEN = 1152
Q3O_VISION_DEPTH = 27
Q3O_VISION_HEADS = 16
Q3O_VISION_INTERMEDIATE = 4304
Q3O_VISION_PATCH = 16
Q3O_VISION_TEMPORAL = 2
Q3O_VISION_MERGE = 2
Q3O_VISION_OUT_HIDDEN = 2048
# DeepStack paper clamps T*H*W between [128, 768] * 32^2 before patching.
Q3O_VISION_MIN_PIXELS = 128 * 32 * 32
Q3O_VISION_MAX_PIXELS = 768 * 32 * 32

Q3O_TEXT_HIDDEN = 2048
Q3O_TEXT_LAYERS = 48
Q3O_TEXT_HEADS = 32
Q3O_TEXT_KV_HEADS = 4
Q3O_TEXT_HEAD_DIM = 128
Q3O_TEXT_MOE_INTERMEDIATE = 768        # moe_intermediate_size
Q3O_TEXT_NUM_EXPERTS = 128             # num_experts
Q3O_TEXT_TOP_K = 8                     # num_experts_per_tok
Q3O_TEXT_SHARED_INTERMEDIATE = 0       # shared_expert_intermediate_size
Q3O_TEXT_VOCAB = 152064                # thinker_config.text_config.vocab_size

# Qwen3-VL 8B config constants (dense text, DeepStack vision)
Q3_VL8B_VISION_HIDDEN = 1152
Q3_VL8B_VISION_DEPTH = 27
Q3_VL8B_VISION_HEADS = 16
Q3_VL8B_VISION_INTERMEDIATE = 4304
Q3_VL8B_VISION_PATCH = 16
Q3_VL8B_VISION_TEMPORAL = 2
Q3_VL8B_VISION_MERGE = 2
Q3_VL8B_VISION_OUT_HIDDEN = 4096
# Reuse DeepStack min/max pixels; Qwen3-VL shares the same 3D ViT backbone.
Q3_VL8B_VISION_MIN_PIXELS = Q3O_VISION_MIN_PIXELS
Q3_VL8B_VISION_MAX_PIXELS = Q3O_VISION_MAX_PIXELS

Q3_VL8B_TEXT_HIDDEN = 4096
Q3_VL8B_TEXT_LAYERS = 36
Q3_VL8B_TEXT_HEADS = 32
Q3_VL8B_TEXT_KV_HEADS = 8
Q3_VL8B_TEXT_HEAD_DIM = 128
Q3_VL8B_TEXT_INTERMEDIATE = 12288
Q3_VL8B_TEXT_VOCAB = 151936
Q3_VL8B_TEXT_MLP_EXPANSION = Q3_VL8B_TEXT_INTERMEDIATE / Q3_VL8B_TEXT_HIDDEN


@dataclass(frozen=True)
class Qwen3VisionSpec:
    """
    Vision encoder spec for Qwen3 models.

    Fields:
      - hidden_size: ViT channel dimension.
      - depth: number of transformer blocks.
      - num_heads: attention heads per block.
      - intermediate_size: MLP intermediate size inside vision blocks.
      - patch_size: spatial patch size (e.g., 16 → 16x16 patches).
      - temporal_patch: number of frames per temporal patch (3D conv).
      - merge_size: spatial_merge_size for the patch merger (e.g., 2 → 2x2).
      - out_hidden_size: dimension after projecting to the language hidden size.
      - swiglu: whether the vision MLPs use SwiGLU; Qwen3 uses GELU-like
        (gelu_pytorch_tanh), so this is False.
      - min_pixels, max_pixels: DeepStack resize clamp on T*H*W.
      - deepstack_visual_indexes: layer indices where additional visual
        stacks are tapped (e.g., [8, 16, 24]).
    """
    hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch: int
    merge_size: int
    out_hidden_size: int
    swiglu: bool
    min_pixels: int
    max_pixels: int
    deepstack_visual_indexes: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Qwen3TextSpec:
    """
    Text encoder spec for Qwen3 models.

    If moe_expert_hidden_size is set, we treat all layers as MoE layers with
    num_experts and moe_top_k per token. Otherwise we use a dense MLP with
    expansion = mlp_expansion.
    """
    hidden_size: int
    layers: int
    num_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    swiglu: bool
    mlp_expansion: Optional[float] = None
    moe_expert_hidden_size: Optional[int] = None
    moe_num_experts: Optional[int] = None
    moe_top_k: Optional[int] = None
    moe_shared_experts: int = 0

    def uses_moe(self) -> bool:
        return self.moe_expert_hidden_size is not None


Q3O_VISION_SPEC = Qwen3VisionSpec(
    hidden_size=Q3O_VISION_HIDDEN,
    depth=Q3O_VISION_DEPTH,
    num_heads=Q3O_VISION_HEADS,
    intermediate_size=Q3O_VISION_INTERMEDIATE,
    patch_size=Q3O_VISION_PATCH,
    temporal_patch=Q3O_VISION_TEMPORAL,
    merge_size=Q3O_VISION_MERGE,
    out_hidden_size=Q3O_VISION_OUT_HIDDEN,
    swiglu=False,
    min_pixels=Q3O_VISION_MIN_PIXELS,
    max_pixels=Q3O_VISION_MAX_PIXELS,
    deepstack_visual_indexes=(8, 16, 24),
)

Q3O_TEXT_SPEC = Qwen3TextSpec(
    hidden_size=Q3O_TEXT_HIDDEN,
    layers=Q3O_TEXT_LAYERS,
    num_heads=Q3O_TEXT_HEADS,
    kv_heads=Q3O_TEXT_KV_HEADS,
    head_dim=Q3O_TEXT_HEAD_DIM,
    vocab_size=Q3O_TEXT_VOCAB,
    swiglu=True,
    moe_expert_hidden_size=Q3O_TEXT_MOE_INTERMEDIATE,
    moe_num_experts=Q3O_TEXT_NUM_EXPERTS,
    moe_top_k=Q3O_TEXT_TOP_K,
    moe_shared_experts=Q3O_TEXT_SHARED_INTERMEDIATE,
)

Q3_VL8B_VISION_SPEC = Qwen3VisionSpec(
    hidden_size=Q3_VL8B_VISION_HIDDEN,
    depth=Q3_VL8B_VISION_DEPTH,
    num_heads=Q3_VL8B_VISION_HEADS,
    intermediate_size=Q3_VL8B_VISION_INTERMEDIATE,
    patch_size=Q3_VL8B_VISION_PATCH,
    temporal_patch=Q3_VL8B_VISION_TEMPORAL,
    merge_size=Q3_VL8B_VISION_MERGE,
    out_hidden_size=Q3_VL8B_VISION_OUT_HIDDEN,
    swiglu=False,
    min_pixels=Q3_VL8B_VISION_MIN_PIXELS,
    max_pixels=Q3_VL8B_VISION_MAX_PIXELS,
    # DeepStack taps; confirmed by vision.deepstack_visual_indexes metadata.
    deepstack_visual_indexes=(8, 16, 24),
)

Q3_VL8B_TEXT_SPEC = Qwen3TextSpec(
    hidden_size=Q3_VL8B_TEXT_HIDDEN,
    layers=Q3_VL8B_TEXT_LAYERS,
    num_heads=Q3_VL8B_TEXT_HEADS,
    kv_heads=Q3_VL8B_TEXT_KV_HEADS,
    head_dim=Q3_VL8B_TEXT_HEAD_DIM,
    vocab_size=Q3_VL8B_TEXT_VOCAB,
    swiglu=True,
    mlp_expansion=Q3_VL8B_TEXT_MLP_EXPANSION,
)


def _qwen3_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """
    Qwen3-Omni / Qwen3-VL adaptive resizing.

    - Snap height/width to multiples of `factor = patch_size * merge_size`.
    - Snap frames to multiples of temporal_factor.
    - Clamp the effective volume T*H*W between [min_pixels, max_pixels]
      as described in the DeepStack paper, then resnap to the nearest factor.
    """
    if num_frames < temporal_factor:
        num_frames = temporal_factor
    if height < factor:
        height = factor
    if width < factor:
        width = factor

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    volume = t_bar * h_bar * w_bar
    source_volume = num_frames * height * width

    if volume > max_pixels and source_volume > 0:
        beta = math.sqrt(source_volume / max_pixels)
        scaled_h = height / beta
        scaled_w = width / beta
        h_bar = max(factor, math.floor(scaled_h / factor) * factor)
        w_bar = max(factor, math.floor(scaled_w / factor) * factor)
    elif volume < min_pixels and source_volume > 0:
        beta = math.sqrt(min_pixels / source_volume)
        scaled_h = height * beta
        scaled_w = width * beta
        h_bar = max(factor, math.ceil(scaled_h / factor) * factor)
        w_bar = max(factor, math.ceil(scaled_w / factor) * factor)

    return int(h_bar), int(w_bar)


def _qwen3_patch_merger_flops(
    num_input_tokens: int,
    hidden_size: int,
    merge_size: int,
    out_hidden_size: int,
    layernorm_dim: int,
) -> int:
    """
    Patch-merger FLOPs: merge_size^2 tokens → 1 token via LN + 2-layer MLP.

    We model the DeepStack patch merger as:
      - LayerNorm over the concatenated merge-dim,
      - Linear(merge_dim → merge_dim),
      - GeLU-like activation,
      - Linear(merge_dim → out_hidden_size).

    This matches the structure in DeepStack-style vision heads for Qwen3.
    """
    if num_input_tokens <= 0:
        return 0

    merged_tokens = num_input_tokens // (merge_size ** 2)
    if merged_tokens <= 0:
        return 0

    merge_dim = hidden_size * (merge_size ** 2)
    ln_tokens = merged_tokens if layernorm_dim == merge_dim else num_input_tokens
    ln_flops = layernorm_flops(1, ln_tokens, layernorm_dim)

    proj1 = 2 * merged_tokens * merge_dim * merge_dim
    gelu_cost = 4 * merged_tokens * merge_dim
    proj2 = 2 * merged_tokens * merge_dim * out_hidden_size
    return ln_flops + proj1 + gelu_cost + proj2


def _qwen3_vision_pipeline_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    spec: Qwen3VisionSpec,
) -> Tuple[int, int]:
    """
    Return (vision_flops, vision_tokens) for a Qwen3-style vision encoder.

    Modeling assumptions (DeepStack-style):
      - Vision encoder is a 3D ViT with depth = spec.depth.
      - Patching uses temporal_patch × patch_size × patch_size.
      - Spatial patch merger uses merge_size × merge_size groups.
      - Each entry in spec.deepstack_visual_indexes corresponds to an
        additional "visual head" that:
          * takes the same vision_seq_len tokens,
          * runs the same patch merger, and
          * produces the same number of merged tokens as the final head.

    vision_tokens is the total number of merged vision tokens projected into
    the language hidden size (i.e., tokens that can appear in the text context).
    """
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return 0, 0

    temporal_factor = spec.temporal_patch
    factor = spec.patch_size * spec.merge_size

    resized_h, resized_w = _qwen3_smart_resize(
        num_frames=vision_frames,
        height=vision_height,
        width=vision_width,
        temporal_factor=temporal_factor,
        factor=factor,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )

    grid_h = max(1, resized_h // spec.patch_size)
    grid_w = max(1, resized_w // spec.patch_size)
    grid_t = max(1, math.ceil(vision_frames / temporal_factor))
    vision_seq_len = grid_t * grid_h * grid_w

    # 3D patch embedding conv: [T, H, W, 3] → [tokens, hidden_size]
    kernel_volume = temporal_factor * (spec.patch_size ** 2)
    patch_embed_flops = 2 * vision_seq_len * kernel_volume * 3 * spec.hidden_size

    # ViT encoder over all vision tokens.
    head_dim = spec.hidden_size // spec.num_heads
    transformer_flops = hybrid_flops(
        batch_size=1,
        seq_len=vision_seq_len,
        hidden_size=spec.hidden_size,
        num_full_attn_layers=spec.depth,
        num_sliding_attn_layers=0,
        num_mamba_layers=0,
        num_mlp_layers=spec.depth,
        window_size=None,
        is_2d=False,
        num_attn_heads=spec.num_heads,
        gqa=False,
        gqa_groups=spec.num_heads,
        kv_channels=head_dim,
        attn_head_dim=head_dim,
        mlp_expansion=spec.intermediate_size / spec.hidden_size,
        swiglu=spec.swiglu,
        vocab_size=0,
        attn_mode="bidirectional",
    )

    # Base vision merger (final head).
    base_merger_flops = _qwen3_patch_merger_flops(
        num_input_tokens=vision_seq_len,
        hidden_size=spec.hidden_size,
        merge_size=spec.merge_size,
        out_hidden_size=spec.out_hidden_size,
        layernorm_dim=spec.hidden_size,
    )

    # DeepStack heads: one extra merger at each specified block index.
    # We assume each head uses an independent patch merger with the same
    # FLOPs and token count as the final head.
    num_deepstack_heads = len(spec.deepstack_visual_indexes)
    deepstack_flops = 0
    if num_deepstack_heads > 0:
        deepstack_flops = num_deepstack_heads * _qwen3_patch_merger_flops(
            num_input_tokens=vision_seq_len,
            hidden_size=spec.hidden_size,
            merge_size=spec.merge_size,
            out_hidden_size=spec.out_hidden_size,
            layernorm_dim=spec.hidden_size,
        )

    vision_merger_flops = base_merger_flops + deepstack_flops

    merged_tokens_per_head = max(1, vision_seq_len // (spec.merge_size ** 2))
    vision_tokens = merged_tokens_per_head * max(1, 1 + num_deepstack_heads)

    total_flops = patch_embed_flops + transformer_flops + vision_merger_flops
    return total_flops, vision_tokens


def _qwen3_text_mlp_cost(seq_len: int, text_spec: Qwen3TextSpec) -> int:
    """Return the per-layer MLP FLOPs for a Qwen3 text stack."""
    if text_spec.uses_moe():
        token_cost = moe_mlp_token_flops(
            hidden_size=text_spec.hidden_size,
            expert_hidden_size=text_spec.moe_expert_hidden_size or 0,
            num_experts_per_token=text_spec.moe_top_k or 0,
            num_routed_experts=text_spec.moe_num_experts or 0,
            num_shared_experts=text_spec.moe_shared_experts,
            swiglu=True,
        )
        return seq_len * token_cost

    if text_spec.mlp_expansion is None:
        raise ValueError("Dense Qwen3 text specs must provide mlp_expansion.")

    return mlp_layer_flops(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=text_spec.hidden_size,
        expansion=text_spec.mlp_expansion,
        swiglu=text_spec.swiglu,
    )


def _qwen3_prompt_flops(seq_len: int, text_spec: Qwen3TextSpec) -> int:
    """Forward FLOPs for a Qwen3 text stack over an input prefix."""
    if seq_len <= 0:
        return 0

    hidden = text_spec.hidden_size
    per_layer_mlp = _qwen3_text_mlp_cost(seq_len, text_spec)
    total = 0
    for _ in range(text_spec.layers):
        total += rmsnorm_flops(1, seq_len, hidden)
        total += attn_layer_flops(
            batch_size=1,
            context_len=seq_len,
            hidden_size=hidden,
            num_heads=text_spec.num_heads,
            gqa=True,
            gqa_groups=text_spec.kv_heads,
            kv_channels=text_spec.head_dim,
            cache_len=0,
            head_dim=text_spec.head_dim,
        )
        total += rmsnorm_flops(1, seq_len, hidden)
        total += per_layer_mlp

    total += rmsnorm_flops(1, seq_len, hidden)
    total += 2 * seq_len * hidden * text_spec.vocab_size
    return total


def _qwen3_generation_flops(
    prompt_len: int,
    num_generated: int,
    text_spec: Qwen3TextSpec,
) -> int:
    """Autoregressive FLOPs with KV caching for a Qwen3 text stack."""
    if num_generated <= 0:
        return 0

    hidden = text_spec.hidden_size
    per_layer_mlp = _qwen3_text_mlp_cost(1, text_spec)
    total = 0
    for step in range(num_generated):
        current_len = prompt_len + step + 1
        cache_len = current_len - 1
        for _ in range(text_spec.layers):
            total += rmsnorm_flops(1, 1, hidden)
            total += attn_layer_flops(
                batch_size=1,
                context_len=current_len,
                hidden_size=hidden,
                num_heads=text_spec.num_heads,
                gqa=True,
                gqa_groups=text_spec.kv_heads,
                kv_channels=text_spec.head_dim,
                cache_len=cache_len,
                head_dim=text_spec.head_dim,
            )
            total += rmsnorm_flops(1, 1, hidden)
            total += per_layer_mlp

        total += rmsnorm_flops(1, 1, hidden)
        total += 2 * hidden * text_spec.vocab_size
        total += 3 * text_spec.vocab_size

    return total


def _qwen3_model_flops(
    *,
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool,
    vision_spec: Qwen3VisionSpec,
    text_spec: Qwen3TextSpec,
) -> Dict[str, int]:
    """
    Shared FLOPs helper for Qwen3-Omni 30B and Qwen3-VL 8B.

    lang_prompt_len:
        Total prompt length seen by the text model (vision placeholders
        + text tokens). We infer the split between text and vision tokens
        based on the number of merged vision tokens produced.
    """
    vision_flops, vision_tokens = _qwen3_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
        spec=vision_spec,
    )

    prompt_len = int(lang_prompt_len)
    text_tokens = max(prompt_len - vision_tokens, 0)
    prompt_seq_len = text_tokens + vision_tokens

    prompt_flops = _qwen3_prompt_flops(prompt_seq_len, text_spec)
    gen_flops = _qwen3_generation_flops(prompt_seq_len, int(num_generated), text_spec)
    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_seq_len,
    }


def qwen3_omni_30b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for Qwen3-Omni 30B thinker (text MoE + DeepStack vision).
    """
    return _qwen3_model_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
        lang_prompt_len=lang_prompt_len,
        num_generated=num_generated,
        do_backward=do_backward,
        vision_spec=Q3O_VISION_SPEC,
        text_spec=Q3O_TEXT_SPEC,
    )


def qwen3_vl_8b_thinking_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for Qwen3-VL-8B-Thinking (DeepStack SigLIP-like vision + dense text).
    """
    return _qwen3_model_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
        lang_prompt_len=lang_prompt_len,
        num_generated=num_generated,
        do_backward=do_backward,
        vision_spec=Q3_VL8B_VISION_SPEC,
        text_spec=Q3_VL8B_TEXT_SPEC,
    )



# ============================================================================
# MiMo-VL-7B FLOPs (Qwen2.5-ViT + MiMo-7B LLM)
# ============================================================================
# Sources:
#   - MiMo-VL-7B-RL config.json (model_type="qwen2_5_vl", vocab_size=151680,
#     hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
#     num_key_value_heads=8, intermediate_size=11008, sliding_window=8192,
#     use_sliding_window=false, vision_config=...).
#     (Provided in the prompt and on the HF hub: XiaomiMiMo/MiMo-VL-7B-RL.)
#   - preprocessor_config.json: min_pixels=3136 (56^2), max_pixels=12845056
#     (3584^2), patch_size=14, temporal_patch_size=2, merge_size=2,
#     image_processor_type="Qwen2VLImageProcessor". 
#   - Qwen2-VL / Qwen2.5-VL docs: Qwen2VLImageProcessor, dynamic resizing,
#     patch_size / temporal_patch_size / merge_size semantics. 
#   - MiMo-VL Technical Report, §2: architecture design: Qwen2.5-ViT vision
#     encoder + projector + MiMo-7B LLM. 

# ---- Text / LLM (MiMo-7B) -----------------------------------------------

MIMO_VL7B_TEXT_HIDDEN = 4096  # MiMo-VL config: "hidden_size": 4096
MIMO_VL7B_TEXT_LAYERS = 36    # MiMo-VL config: "num_hidden_layers": 36
MIMO_VL7B_TEXT_HEADS = 32     # MiMo-VL config: "num_attention_heads": 32
MIMO_VL7B_TEXT_KV_HEADS = 8   # MiMo-VL config: "num_key_value_heads": 8
MIMO_VL7B_TEXT_INTERMEDIATE = 11008  # MiMo-VL config: "intermediate_size": 11008
MIMO_VL7B_TEXT_MLP_EXPANSION = MIMO_VL7B_TEXT_INTERMEDIATE / MIMO_VL7B_TEXT_HIDDEN
MIMO_VL7B_TEXT_FULL_ATTENTION_LAYERS = 0
MIMO_VL7B_TEXT_SLIDING_ATTENTION_LAYERS = MIMO_VL7B_TEXT_LAYERS
MIMO_VL7B_TEXT_SLIDING_WINDOW = 4096
# MiMo-VL config: "vocab_size": 151680
MIMO_VL7B_VOCAB = 151680

# ---- Vision encoder (Qwen2.5-ViT as used in MiMo-VL) --------------------

# MiMo-VL vision_config: depth=32, hidden_size=1280, intermediate_size=3456,
# num_heads=16, in_chans=3, patch_size=14, spatial_merge_size=2,
# temporal_patch_size=2, window_size=112, out_hidden_size=4096, fullatt_block_indexes=[7,15,23,31].
MIMO_VL7B_VISION_HIDDEN = 1280
MIMO_VL7B_VISION_LAYERS = 32
MIMO_VL7B_VISION_HEADS = 16
MIMO_VL7B_VISION_INTERMEDIATE = 3456
MIMO_VL7B_VISION_PATCH = 14
MIMO_VL7B_VISION_TEMPORAL = 2
MIMO_VL7B_VISION_MERGE = 2
MIMO_VL7B_VISION_OUT_HIDDEN = MIMO_VL7B_TEXT_HIDDEN
# Number of ViT blocks that use global (full) attention; equal to len(fullatt_block_indexes).
MIMO_VL7B_VISION_FULL_ATTENTION_LAYERS = 4
# window_size is in pixels; convert to #patches per side: 112 / 14 = 8.
MIMO_VL7B_VISION_WINDOW_PIXELS = 112
# Sliding windows operate on the token grid *after* the 2×2 spatial merger, so
# convert the pixel window to tokens by dividing by both the patch size and the
# merge factor. The upstream transformer does the same calculation inside
# Qwen2_5_VisionTransformerPretrainedModel.get_window_index.
MIMO_VL7B_VISION_WINDOW_TOKENS_SIDE = (
    MIMO_VL7B_VISION_WINDOW_PIXELS
    // (MIMO_VL7B_VISION_PATCH * MIMO_VL7B_VISION_MERGE)
)

# Tokens-per-second is *sampling policy* in the video preprocessor; FLOPs only
# see the resulting #frames, so we keep it as a constant for documentation.
MIMO_VL7B_TOKENS_PER_SECOND = 2  # vision_config: "tokens_per_second": 2
MIMO_VL7B_VISION_PROMPT_OVERHEAD = 2
MIMO_VL7B_VISION_PROMPT_OVERHEAD = 2

# Dynamic resize bounds (from preprocessor_config.json, identical to Qwen2-VL docs).
MIMO_VL7B_MIN_PIXELS = 56 * 56
MIMO_VL7B_MAX_PIXELS = 3584 * 3584


def _qwen2_vl_dynamic_resize(
    height: int,
    width: int,
    patch_size: int = MIMO_VL7B_VISION_PATCH,
    merge_size: int = MIMO_VL7B_VISION_MERGE,
    min_pixels: int = MIMO_VL7B_MIN_PIXELS,
    max_pixels: int = MIMO_VL7B_MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Exact clone of Qwen2VLImageProcessor.smart_resize, specialized for MiMo-VL.

    Source: transformers/models/qwen2_vl/image_processing_qwen2_vl.py,
    `smart_resize` definition (also mirrored in zhouyik/DenseLabelDev
    vlm/models/qwen2_vl/image_processing_qwen2_vl.py:L196–242).

    - factor = patch_size * merge_size (ensures both patching and 2x2 merging
      align on integer grids).
    - Enforces:
        * h_bar, w_bar divisible by factor
        * area in [min_pixels, max_pixels]
        * aspect ratio maintained within a tolerance.
    """
    if height <= 0 or width <= 0:
        return 0, 0

    factor = patch_size * merge_size

    if height < factor or width < factor:
        # This is exactly what the HF processor does; if the user passes an
        # impossible resolution, emulate the same failure.
        raise ValueError(f"height:{height} or width:{width} must be >= factor:{factor}")

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )

    # First snap to nearest multiple of factor.
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # If the rounded area is too large, shrink by a uniform scale beta.
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    # If the rounded area is too small, enlarge by a uniform scale beta.
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(float(min_pixels) / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def _mimo_vl_vision_pipeline_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    apply_dynamic_resize: bool = True,
) -> Dict[str, int]:
    """
    Vision-side FLOPs for MiMo-VL-7B (Qwen2.5-ViT + 2x2 spatial merger).

    - Resize is done exactly like Qwen2VLImageProcessor.smart_resize
      (factor = patch_size * merge_size).
      Source: transformers/models/qwen2_vl/image_processing_qwen2_vl.py,
      `smart_resize` helper. :contentReference[oaicite:7]{index=7}

    - Temporal patching pads frames so that num_frames is divisible by
      temporal_patch_size, then groups them into 3D patches; this is the
      behavior mirrored in vLLM / DotsOCR:
        padded_num_frames = num_frames + num_frames % temporal_patch_size
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
      :contentReference[oaicite:8]{index=8}

    - Patch merger FLOPs match Qwen2_5_VLPatchMerger:
        * RMSNorm over context_dim (= vision_hidden_size) on *all* ViT tokens.
        * 2-layer MLP on grouped tokens of size hidden_size = context_dim * 4.
      Source: Qwen2_5_VLPatchMerger & Qwen2_5_VisionTransformerPretrainedModel
      in transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py. :contentReference[oaicite:9]{index=9}
    """
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return {
            "vision_flops": 0,
            "vision_tokens": 0,
            "vision_tokens_droppable": 0,
            "vision_prompt_overhead": MIMO_VL7B_VISION_PROMPT_OVERHEAD,
            "vision_seq_len": 0,
            "resized_height": 0,
            "resized_width": 0,
            "patches_per_frame": 0,
            "temporal_segments": 0,
        }

    # --- 1) Dynamic resize (exact smart_resize logic) -----------------------
    if apply_dynamic_resize:
        h_resized, w_resized = _qwen2_vl_dynamic_resize(
            height=vision_height,
            width=vision_width,
            patch_size=MIMO_VL7B_VISION_PATCH,
            merge_size=MIMO_VL7B_VISION_MERGE,
            min_pixels=MIMO_VL7B_MIN_PIXELS,
            max_pixels=MIMO_VL7B_MAX_PIXELS,
        )
    else:
        # Match HF behavior: dimensions must be multiples of patch_size * merge_size.
        factor = MIMO_VL7B_VISION_PATCH * MIMO_VL7B_VISION_MERGE
        h_resized = max(factor, (vision_height // factor) * factor)
        w_resized = max(factor, (vision_width // factor) * factor)

    patches_h = h_resized // MIMO_VL7B_VISION_PATCH
    patches_w = w_resized // MIMO_VL7B_VISION_PATCH
    patches_per_frame = patches_h * patches_w

    if patches_per_frame <= 0:
        return {
            "vision_flops": 0,
            "vision_tokens": MIMO_VL7B_VISION_PROMPT_OVERHEAD,
            "vision_tokens_droppable": 0,
            "vision_prompt_overhead": MIMO_VL7B_VISION_PROMPT_OVERHEAD,
            "vision_seq_len": 0,
            "resized_height": h_resized,
            "resized_width": w_resized,
            "patches_per_frame": patches_per_frame,
            "temporal_segments": 0,
        }

    # --- 2) Temporal patching and padding ----------------------------------
    # HF pads frames so that num_frames % temporal_patch_size == 0 by
    # repeating the last frame. The effective temporal grid is:
    #   padded_num_frames = frames + frames % temporal_patch_size
    #   grid_t = max(padded_num_frames // temporal_patch_size, 1)
    # (See vLLM DotsOCRProcessingInfo.get_mm_max_tokens_per_item, which
    # references Qwen2VLImageProcessor._preprocess.) :contentReference[oaicite:10]{index=10}
    padded_frames = vision_frames + vision_frames % MIMO_VL7B_VISION_TEMPORAL
    temporal_segments = max(padded_frames // MIMO_VL7B_VISION_TEMPORAL, 1)
    vision_seq_len = temporal_segments * patches_per_frame  # num_patches

    # --- 3) 3D patch embedding (Conv3D) ------------------------------------
    kernel_volume = (
        MIMO_VL7B_VISION_TEMPORAL * (MIMO_VL7B_VISION_PATCH ** 2)
    )  # [Tpatch, P, P]
    in_channels = 3
    patch_embed_flops = (
        2
        * vision_seq_len
        * kernel_volume
        * in_channels
        * MIMO_VL7B_VISION_HIDDEN
    )

    # --- 4) ViT encoder FLOPs (unchanged from your previous version) -------
    full_blocks = MIMO_VL7B_VISION_FULL_ATTENTION_LAYERS
    sliding_blocks = MIMO_VL7B_VISION_LAYERS - full_blocks
    head_dim = MIMO_VL7B_VISION_HIDDEN // MIMO_VL7B_VISION_HEADS

    window_shape = (
        (MIMO_VL7B_VISION_WINDOW_TOKENS_SIDE, MIMO_VL7B_VISION_WINDOW_TOKENS_SIDE)
        if sliding_blocks > 0
        else None
    )

    vision_transformer_flops = hybrid_flops(
        batch_size=1,
        seq_len=vision_seq_len,
        hidden_size=MIMO_VL7B_VISION_HIDDEN,
        num_full_attn_layers=full_blocks,
        num_sliding_attn_layers=sliding_blocks,
        num_mamba_layers=0,
        num_mlp_layers=MIMO_VL7B_VISION_LAYERS,
        window_size=window_shape,
        is_2d=True,
        num_attn_heads=MIMO_VL7B_VISION_HEADS,
        gqa=False,
        gqa_groups=MIMO_VL7B_VISION_HEADS,
        kv_channels=head_dim,
        attn_head_dim=head_dim,
        mlp_expansion=MIMO_VL7B_VISION_INTERMEDIATE / MIMO_VL7B_VISION_HIDDEN,
        swiglu=True,
        vocab_size=0,
        attn_mode="bidirectional",
    )

    # --- 5) Patch merger FLOPs (RMSNorm + 2-layer MLP) ---------------------
    merge_size = MIMO_VL7B_VISION_MERGE
    merge_area = merge_size ** 2
    merged_tokens = max(1, vision_seq_len // merge_area)
    context_dim = MIMO_VL7B_VISION_HIDDEN
    merge_dim = context_dim * merge_area  # e.g. 1280 * 4 = 5120

    # (a) RMSNorm over context_dim on *all* ViT tokens before grouping.
    merger_rms_flops = rmsnorm_flops(
        batch_size=1,
        seq_len=vision_seq_len,
        hidden_size=context_dim,
    )

    # (b) MLP on grouped tokens: merge_dim -> merge_dim -> out_hidden.
    up_proj = 2 * merged_tokens * merge_dim * merge_dim
    act_cost = 4 * merged_tokens * merge_dim  # GELU approx
    down_proj = 2 * merged_tokens * merge_dim * MIMO_VL7B_VISION_OUT_HIDDEN
    merger_mlp_flops = up_proj + act_cost + down_proj

    merger_flops = merger_rms_flops + merger_mlp_flops

    vision_flops = patch_embed_flops + vision_transformer_flops + merger_flops

    return {
        "vision_flops": int(vision_flops),
        "vision_tokens": int(merged_tokens + MIMO_VL7B_VISION_PROMPT_OVERHEAD),
        "vision_tokens_droppable": int(merged_tokens),
        "vision_prompt_overhead": MIMO_VL7B_VISION_PROMPT_OVERHEAD,
        "vision_seq_len": int(vision_seq_len),
        "resized_height": int(h_resized),
        "resized_width": int(w_resized),
        "patches_per_frame": int(patches_per_frame),
        "temporal_segments": int(temporal_segments),
    }

def _mimo_vl_lm_flops_for_prompt_and_gen(
    prompt_len: int,
    num_generated: int,
) -> Dict[str, int]:
    """
    LLM prefill + generation FLOPs for the MiMo-7B text stack used in MiMo-VL.

    Architecture is a Qwen2.5-style decoder:
      - hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=11008, hidden_act="silu",
        vocab_size=151680. (MiMo-VL config.json.)
      - SwiGLU MLP with expansion = intermediate_size / hidden_size.
      - MiMo's public config advertises `sliding_window=8192` and the profiler
        traces show the stack behaves like a pure sliding-window decoder. We
        therefore model all 36 layers with a window of 8192 tokens, which
        matches the runtime mask selection inside `Qwen2_5_VLTextModel` when
        `layer_types` is dominated by "sliding_attention" entries.
    """
    if prompt_len <= 0 and num_generated <= 0:
        return {"lang_prompt_flops": 0, "gen_flops": 0}

    head_dim = MIMO_VL7B_TEXT_HIDDEN // MIMO_VL7B_TEXT_HEADS

    # --- Prefill (prompt) ---------------------------------------------------
    prompt_flops = 0
    sliding_layers = MIMO_VL7B_TEXT_SLIDING_ATTENTION_LAYERS
    sliding_window = MIMO_VL7B_TEXT_SLIDING_WINDOW if sliding_layers > 0 else None

    if prompt_len > 0:
        prompt_flops = hybrid_flops(
            batch_size=1,
            seq_len=prompt_len,
            hidden_size=MIMO_VL7B_TEXT_HIDDEN,
            num_full_attn_layers=MIMO_VL7B_TEXT_FULL_ATTENTION_LAYERS,
            num_sliding_attn_layers=sliding_layers,
            num_mamba_layers=0,
            num_mlp_layers=MIMO_VL7B_TEXT_LAYERS,
            window_size=sliding_window,
            is_2d=False,
            num_attn_heads=MIMO_VL7B_TEXT_HEADS,
            gqa=True,
            gqa_groups=MIMO_VL7B_TEXT_KV_HEADS,
            kv_channels=head_dim,
            attn_head_dim=head_dim,
            mlp_expansion=MIMO_VL7B_TEXT_MLP_EXPANSION,
            swiglu=True,  # MiMo-7B uses SwiGLU MLP with SiLU gate
            vocab_size=MIMO_VL7B_VOCAB,
            attn_mode="causal",
        )

    # --- Generation with KV cache ------------------------------------------
    gen_flops = 0
    if num_generated > 0:
        gen_flops = autoregressive_generation_flops(
            prompt_len=prompt_len,
            num_generated=num_generated,
            hidden_size=MIMO_VL7B_TEXT_HIDDEN,
            num_layers=MIMO_VL7B_TEXT_LAYERS,
            num_heads=MIMO_VL7B_TEXT_HEADS,
            mlp_expansion=MIMO_VL7B_TEXT_MLP_EXPANSION,
            vocab_size=MIMO_VL7B_VOCAB,
            gqa_groups=MIMO_VL7B_TEXT_KV_HEADS,
            head_dim=head_dim,
            kv_channels=head_dim,
            num_full_attn_layers=MIMO_VL7B_TEXT_FULL_ATTENTION_LAYERS,
            num_sliding_attn_layers=sliding_layers,
            sliding_window_size=sliding_window or 0,
            sliding_is_2d=False,
            num_mamba_layers=0,
            swiglu=True,
        )

    return {
        "lang_prompt_flops": int(prompt_flops),
        "gen_flops": int(gen_flops),
    }


def mimo_vl_7b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    apply_dynamic_resize: bool = True,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for MiMo-VL-7B (SFT or RL weights) on a single example.

    Pipeline (per MiMo-VL Technical Report, §2): 
      1) Qwen2.5-ViT visual encoder on images/videos.
      2) 2×2 spatial patch merger + MLP projector to the MiMo-7B hidden space.
      3) MiMo-7B LLM on a mixed prompt (text + vision placeholders).
      4) Autoregressive generation with KV caching.

    Args
    ----
    vision_frames:
        Number of frames actually fed into the ViT (after any FPS subsampling by
        Qwen2VLVideoProcessor). Use 1 for a single image, 0 for text-only.
    vision_height, vision_width:
        Raw frame resolution in pixels. If `apply_dynamic_resize=True`, the
        function applies the same dynamic resize heuristic as Qwen2VLImageProcessor:
          - clamp area to [56^2, 3584^2],
          - preserve aspect ratio,
          - snap to multiples of 14.
        If False, these are assumed to already be the resized, patch-aligned
        dimensions.
    lang_prompt_len:
        Total prompt length *seen by the LLM*, i.e. text tokens + all vision
        placeholder tokens that correspond to merged vision tokens.
        This matches the convention in `qwen_2_5_vl_7b_flops`.
    num_generated:
        Number of autoregressive tokens generated after the prompt.
    apply_dynamic_resize:
        Whether to approximate the Qwen2-VL dynamic resizing step.
    do_backward:
        If True, multiply all FLOP counts by 3 to approximate forward +
        backward + optimizer update (per your global convention).

    Returns
    -------
    dict with keys:
      - "vision_flops": FLOPs in the vision stack (patch embed + ViT + merger).
      - "lang_prompt_flops": FLOPs for LLM prefill (prompt).
      - "gen_flops": FLOPs for autoregressive generation.
      - "total_flops": total FLOPs (forward-only or 3× if do_backward=True).
      - "vision_tokens": #merged vision tokens inserted into the LLM prompt.
      - "text_tokens": #text tokens in the prompt.
      - "prompt_tokens_total": total prompt length before compression.
      - "prompt_tokens_effective": effective length used for the LM FLOPs
        (vision tokens divided by `tokens_per_second`).
      - plus some vision-side bookkeeping entries from `_mimo_vl_vision_pipeline_flops`.
    """
    # --- Vision side --------------------------------------------------------
    vision_info = _mimo_vl_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
        apply_dynamic_resize=apply_dynamic_resize,
    )
    vision_flops = vision_info["vision_flops"]
    vision_tokens = vision_info["vision_tokens"]

    # --- Prompt composition -------------------------------------------------
    # lang_prompt_len is defined as text + vision placeholders. The true text
    # token count is whatever remains after accounting for the vision tokens.
    text_tokens = max(lang_prompt_len - vision_tokens, 0)
    prompt_len = text_tokens + vision_tokens

    # --- LLM side -----------------------------------------------------------
    lm_info = _mimo_vl_lm_flops_for_prompt_and_gen(prompt_len, num_generated)
    prompt_flops = lm_info["lang_prompt_flops"]
    gen_flops = lm_info["gen_flops"]

    total_fwd = vision_flops + prompt_flops + gen_flops
    multiplier = 3 if do_backward else 1

    out: Dict[str, Any] = {
        "vision_flops": vision_flops * multiplier,
        "lang_prompt_flops": prompt_flops * multiplier,
        "gen_flops": gen_flops * multiplier,
        "total_flops": total_fwd * multiplier,
        "vision_tokens": int(vision_tokens),
        "text_tokens": int(text_tokens),
        "prompt_tokens_total": int(prompt_len),
        "prompt_tokens_effective": int(prompt_len),
    }
    # Attach the intermediate vision bookkeeping as well.
    out.update(
        {
            "vision_seq_len": vision_info["vision_seq_len"],
            "resized_height": vision_info["resized_height"],
            "resized_width": vision_info["resized_width"],
            "patches_per_frame": vision_info["patches_per_frame"],
            "temporal_segments": vision_info["temporal_segments"],
        }
    )
    return out

# ============================================================================
# InternVL3.5-8B FLOPs (InternVision-1B + Qwen3-8B)
# ============================================================================

# Vision hyperparameters from vision_config in OpenGVLab/InternVL3_5-8B:
#   hidden_size = 1024
#   num_hidden_layers = 24
#   num_attention_heads = 16
#   intermediate_size = 4096
#   image_size = 448
#   patch_size = 14
#   norm_type = "layer_norm"
#
# See OpenGVLab/InternVL3_5-8B/config.json (vision_config) and
# transformers/models/internvl/configuration_intern_vit.py / InternVLVisionConfig.


INTERNVL35_8B_VISION_HIDDEN = 1024
INTERNVL35_8B_VISION_LAYERS = 24
INTERNVL35_8B_VISION_HEADS = 16
INTERNVL35_8B_VISION_INTERMEDIATE = 4096
INTERNVL35_8B_IMAGE_SIZE = 448
INTERNVL35_8B_PATCH_SIZE = 14

INTERNVL35_DOWNSAMPLE_RATIO = 0.5
INTERNVL35_MAX_DYNAMIC_PATCH = 12
INTERNVL35_MIN_DYNAMIC_PATCH = 1
INTERNVL35_FORCE_IMAGE_SIZE = 448

INTERNVL35_TOKENS_PER_PATCH_RAW = (INTERNVL35_8B_IMAGE_SIZE // INTERNVL35_8B_PATCH_SIZE) ** 2  # 32*32=1024
INTERNVL35_DOWNSAMPLE_RATIO2 = int(1.0 / (INTERNVL35_DOWNSAMPLE_RATIO ** 2))                    # 4
INTERNVL35_TOKENS_PER_PATCH = int(
    INTERNVL35_TOKENS_PER_PATCH_RAW * (INTERNVL35_DOWNSAMPLE_RATIO ** 2)
)  # 1024 * 0.25 = 256
# Keep the legacy name expected by runtime integrations.
INTERNVL35_NUM_IMAGE_TOKENS_PER_PATCH = INTERNVL35_TOKENS_PER_PATCH
INTERNVL35_8B_VISION_DIM_AFTER = INTERNVL35_8B_VISION_HIDDEN * INTERNVL35_DOWNSAMPLE_RATIO2     # 4096

# Text (Qwen3-8B) from llm_config in OpenGVLab/InternVL3_5-8B:
#   hidden_size = 4096
#   num_hidden_layers = 36
#   num_attention_heads = 32
#   num_key_value_heads = 8
#   head_dim = 128
#   intermediate_size = 12288
#   vocab_size = 151936
#
# This matches the dense Qwen3-VL-8B text config.


INTERNVL35_8B_TEXT_SPEC = Qwen3TextSpec(
    hidden_size=4096,
    layers=36,
    num_heads=32,
    kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    swiglu=True,
    mlp_expansion=12288 / 4096,
)


def _internvl35_patches_per_frame(
    height: int,
    width: int,
    max_patches: int = INTERNVL35_MAX_DYNAMIC_PATCH,
    min_patches: int = INTERNVL35_MIN_DYNAMIC_PATCH,
    target_size: int = INTERNVL35_FORCE_IMAGE_SIZE,
) -> int:
    """
    Emulate GotOcr2ImageProcessorFast.get_number_of_image_patches(height, width)
    using adaptive_image_slicing (same behavior described in GOT-OCR2 docs):
      - choose N patches in [min_patches, max_patches]
      - patches are target_size x target_size (448 x 448)
      - grid aspect ratio matches the original image.
    """
    if height <= 0 or width <= 0:
        return 0
    approx = adaptive_image_slicing(
        width=width,
        height=height,
        target_width=target_size,
        target_height=target_size,
        max_slices=max_patches,
    )
    return max(min_patches, min(approx, max_patches))


def _internvl35_8b_vit_flops_per_patch() -> int:
    """
    InternVision-1B FLOPs on a single 448x448 patch.

    Modeled as 24 blocks of:
      LayerNorm(1024) -> bidirectional attention over 1024 tokens
      -> LayerNorm(1024) -> GeLU MLP with expansion 4.0.
    """
    H = INTERNVL35_8B_VISION_HIDDEN
    L = INTERNVL35_TOKENS_PER_PATCH_RAW
    heads = INTERNVL35_8B_VISION_HEADS
    mlp_expansion = INTERNVL35_8B_VISION_INTERMEDIATE / float(H)

    total = 0
    for _ in range(INTERNVL35_8B_VISION_LAYERS):
        total += layernorm_flops(1, L, H)
        total += bidirectional_attention_flops(
            batch_size=1,
            seq_len=L,
            hidden_size=H,
            num_heads=heads,
            head_dim=None,
            kv_channels=None,
            gqa=False,
            gqa_groups=None,
        )
        total += layernorm_flops(1, L, H)
        total += mlp_layer_flops(
            batch_size=1,
            seq_len=L,
            hidden_size=H,
            expansion=mlp_expansion,
            swiglu=False,  # vision_config.hidden_act = "gelu"
        )

    # InternVision typically has a final norm; negligible extra cost.
    return total


def _internvl35_8b_downsample_and_project_flops_per_patch() -> int:
    """
    Pixel-unshuffle + ln_vision + mm_projector on a single patch.

    Shapes per patch:
      - before: 1024 tokens x 1024 dim
      - after pixel-unshuffle (ratio=0.5): 256 tokens x 4096 dim
    We count:
      - LayerNorm(4096) on 256 tokens
      - Linear(4096 -> 4096) on 256 tokens.
    """
    L = INTERNVL35_TOKENS_PER_PATCH
    D = INTERNVL35_8B_VISION_DIM_AFTER

    ln = layernorm_flops(1, L, D)
    mm = 2 * L * D * D
    return ln + mm


def _internvl35_8b_vision_flops_and_tokens(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
) -> (int, int):
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return 0, 0

    patches_per_frame = _internvl35_patches_per_frame(vision_height, vision_width)
    total_patches = vision_frames * patches_per_frame
    if total_patches <= 0:
        return 0, 0

    vit_per = _internvl35_8b_vit_flops_per_patch()
    down_per = _internvl35_8b_downsample_and_project_flops_per_patch()

    vision_flops = total_patches * (vit_per + down_per)
    vision_tokens = total_patches * INTERNVL35_TOKENS_PER_PATCH

    return vision_flops, vision_tokens


def internvl3_5_8b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for OpenGVLab/InternVL3_5-8B.

    lang_prompt_len is the number of *text* tokens (no vision tokens)
    in the prompt. Vision tokens are added on top of this.
    """
    vision_flops, vision_tokens = _internvl35_8b_vision_flops_and_tokens(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )

    text_tokens = max(int(lang_prompt_len), 0)
    prompt_len = text_tokens + vision_tokens

    prompt_flops = _qwen3_prompt_flops(prompt_len, INTERNVL35_8B_TEXT_SPEC)
    gen_flops = _qwen3_generation_flops(prompt_len, int(num_generated), INTERNVL35_8B_TEXT_SPEC)

    total_fwd = vision_flops + prompt_flops + gen_flops
    mult = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * mult,
        "lang_prompt_flops": prompt_flops * mult,
        "gen_flops": gen_flops * mult,
        "total_flops": total_fwd * mult,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,
    }


# ============================================================================
# InternVL3.5-38B FLOPs (InternViT-6B + Qwen3-32B)
# ============================================================================

# Vision config from OpenGVLab/InternVL3_5-38B/config.json (vision_config):
#   hidden_size        = 3200
#   num_hidden_layers  = 45
#   num_attention_heads= 25
#   intermediate_size  = 12800
#   image_size         = 448
#   patch_size         = 14
#   norm_type          = "rms_norm"
#   qk_normalization   = true  (ignored in FLOPs; affects scaling only).


INTERNVL35_38B_VISION_HIDDEN = 3200
INTERNVL35_38B_VISION_LAYERS = 45
INTERNVL35_38B_VISION_HEADS = 25
INTERNVL35_38B_VISION_INTERMEDIATE = 12800
INTERNVL35_38B_IMAGE_SIZE = 448
INTERNVL35_38B_PATCH_SIZE = 14

INTERNVL35_38B_TOKENS_PER_PATCH_RAW = (INTERNVL35_38B_IMAGE_SIZE // INTERNVL35_38B_PATCH_SIZE) ** 2  # 1024
INTERNVL35_38B_TOKENS_PER_PATCH = int(
    INTERNVL35_38B_TOKENS_PER_PATCH_RAW * (INTERNVL35_DOWNSAMPLE_RATIO ** 2)
)  # 256
INTERNVL35_38B_VISION_DIM_AFTER = INTERNVL35_38B_VISION_HIDDEN * INTERNVL35_DOWNSAMPLE_RATIO2  # 3200*4=12800

# Text (Qwen3-32B) from llm_config in OpenGVLab/InternVL3_5-38B:
#   hidden_size        = 5120
#   num_hidden_layers  = 64
#   num_attention_heads= 64
#   num_key_value_heads= 8
#   head_dim           = 128
#   intermediate_size  = 25600
#   vocab_size         = 151936


INTERNVL35_38B_TEXT_SPEC = Qwen3TextSpec(
    hidden_size=5120,
    layers=64,
    num_heads=64,
    kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    swiglu=True,
    mlp_expansion=25600 / 5120,
)


def _internvl35_38b_vit_flops_per_patch() -> int:
    """
    InternViT-6B FLOPs for a single 448x448 patch.

    Modeled as 45 blocks of:
      RMSNorm(3200) -> bidirectional attention (1024 tokens)
      -> RMSNorm(3200) -> GeLU/SwiGLU-ish MLP with expansion 4.0.
    """
    H = INTERNVL35_38B_VISION_HIDDEN
    L = INTERNVL35_38B_TOKENS_PER_PATCH_RAW
    heads = INTERNVL35_38B_VISION_HEADS
    mlp_expansion = INTERNVL35_38B_VISION_INTERMEDIATE / float(H)

    total = 0
    for _ in range(INTERNVL35_38B_VISION_LAYERS):
        total += rmsnorm_flops(1, L, H)  # norm_type = "rms_norm"
        total += bidirectional_attention_flops(
            batch_size=1,
            seq_len=L,
            hidden_size=H,
            num_heads=heads,
            head_dim=None,
            kv_channels=None,
            gqa=False,
            gqa_groups=None,
        )
        total += rmsnorm_flops(1, L, H)
        total += mlp_layer_flops(
            batch_size=1,
            seq_len=L,
            hidden_size=H,
            expansion=mlp_expansion,
            swiglu=False,  # config.hidden_act="gelu"; FLOPs same as GeLU-like
        )
    return total


def _internvl35_38b_downsample_and_project_flops_per_patch() -> int:
    """
    Pixel-unshuffle + ln_vision + mm_projector for 38B.

    After downsampling:
      - 256 tokens x (3200 * 4 = 12800) dim
    mm_projector maps 12800 -> 5120 (Qwen3-32B hidden size).
    """
    L = INTERNVL35_38B_TOKENS_PER_PATCH   # 256
    D_in = INTERNVL35_38B_VISION_DIM_AFTER  # 12800
    D_out = INTERNVL35_38B_TEXT_SPEC.hidden_size  # 5120

    ln = rmsnorm_flops(1, L, D_in)  # InternVL3.5-38B uses RMSNorm here as well
    mm = 2 * L * D_in * D_out
    return ln + mm


def _internvl35_38b_vision_flops_and_tokens(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
) -> (int, int):
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return 0, 0

    patches_per_frame = _internvl35_patches_per_frame(vision_height, vision_width)
    total_patches = vision_frames * patches_per_frame
    if total_patches <= 0:
        return 0, 0

    vit_per = _internvl35_38b_vit_flops_per_patch()
    down_per = _internvl35_38b_downsample_and_project_flops_per_patch()

    vision_flops = total_patches * (vit_per + down_per)
    vision_tokens = total_patches * INTERNVL35_38B_TOKENS_PER_PATCH
    return vision_flops, vision_tokens


def internvl3_5_38b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for OpenGVLab/InternVL3_5-38B (InternViT-6B + Qwen3-32B).
    """
    vision_flops, vision_tokens = _internvl35_38b_vision_flops_and_tokens(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )

    text_tokens = max(int(lang_prompt_len), 0)
    prompt_len = text_tokens + vision_tokens

    prompt_flops = _qwen3_prompt_flops(prompt_len, INTERNVL35_38B_TEXT_SPEC)
    gen_flops = _qwen3_generation_flops(prompt_len, int(num_generated), INTERNVL35_38B_TEXT_SPEC)

    total_fwd = vision_flops + prompt_flops + gen_flops
    mult = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * mult,
        "lang_prompt_flops": prompt_flops * mult,
        "gen_flops": gen_flops * mult,
        "total_flops": total_fwd * mult,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,
    }


# ============================================================================
# InternVL3.5-30B-A3B FLOPs (InternVision-1B + Qwen3-30B-A3B MoE)
# ============================================================================

# Vision is the same as InternVL3.5-8B (InternViT-1B). We reuse those constants
# and the _internvl35_8b_vision_flops_and_tokens helper.


# Text (Qwen3-30B-A3B MoE) from llm_config in OpenGVLab/InternVL3_5-30B-A3B:
#   hidden_size        = 2048
#   num_hidden_layers  = 48
#   num_attention_heads= 32
#   num_key_value_heads= 4
#   head_dim           = 128
#   intermediate_size  = 6144   (dense)
#   moe_intermediate_size = 768
#   num_experts        = 128
#   num_experts_per_tok= 8
#   vocab_size         = 151936


INTERNVL35_30B_MOE_TEXT_SPEC = Qwen3TextSpec(
    hidden_size=2048,
    layers=48,
    num_heads=32,
    kv_heads=4,
    head_dim=128,
    vocab_size=151936,
    swiglu=True,
    mlp_expansion=None,              # we use MoE instead
    moe_expert_hidden_size=768,
    moe_num_experts=128,
    moe_top_k=8,
    moe_shared_experts=0,
)


def _internvl35_30b_prompt_flops(seq_len: int) -> int:
    """
    Prompt FLOPs for Qwen3-30B-A3B MoE on a sequence of length seq_len.
    Reuses the generic Qwen3 MoE machinery via Qwen3TextSpec.
    """
    return _qwen3_prompt_flops(seq_len, INTERNVL35_30B_MOE_TEXT_SPEC)


def _internvl35_30b_generation_flops(prompt_len: int, num_generated: int) -> int:
    return _qwen3_generation_flops(prompt_len, num_generated, INTERNVL35_30B_MOE_TEXT_SPEC)


def internvl3_5_30b_a3b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for OpenGVLab/InternVL3_5-30B-A3B (InternVision-1B + Qwen3-30B-A3B MoE).
    """
    # Vision is identical to the 8B variant.
    vision_flops, vision_tokens = _internvl35_8b_vision_flops_and_tokens(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )

    text_tokens = max(int(lang_prompt_len), 0)
    prompt_len = text_tokens + vision_tokens

    prompt_flops = _internvl35_30b_prompt_flops(prompt_len)
    gen_flops = _internvl35_30b_generation_flops(prompt_len, int(num_generated))

    total_fwd = vision_flops + prompt_flops + gen_flops
    mult = 3 if do_backward else 1

    return {
        "vision_flops": vision_flops * mult,
        "lang_prompt_flops": prompt_flops * mult,
        "gen_flops": gen_flops * mult,
        "total_flops": total_fwd * mult,
        "vision_tokens": vision_tokens,
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,
    }


# ============================================================================
# Phi-4 Mini (text) + Phi-4-Multimodal FLOPs with vision LoRA
# ============================================================================
#
# Grounding in code / paper:
# --------------------------
# 1. Base text backbone (Phi-4-Mini)
#    • configuration_phi4mm.Phi4MMConfig:
#        hidden_size       = 3072
#        intermediate_size = 8192
#        num_hidden_layers = 32
#        num_attention_heads = 24
#        num_key_value_heads = 8
#        vocab_size        = 200064
#        hidden_act        = "silu"
#        sliding_window    = 262144
#      (See HF config.json for microsoft/Phi-4-multimodal-instruct.)
#
#    • modeling_phi4mm.Phi4MMDecoderLayer:
#        self.self_attn = PHI4MM_ATTENTION_CLASSES[...]
#        self.mlp       = Phi4MMMLP(config)
#      and Phi4MMMLP (not shown here) declares:
#        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=...)
#        self.down_proj    = nn.Linear(config.intermediate_size, config.hidden_size, bias=...)
#      i.e. a fused SwiGLU MLP: gate_up (H → 2M), split into gate/up, SiLU on gate,
#      elementwise gate, and down_proj (M → H). This matches the “gate_up_proj” pattern used
#      by Phi-4 models elsewhere and the regex in config.vision_lora["layer"].
#
# 2. Vision encoder and HD transform
#    • modeling_phi4mm.Phi4MMImageEmbedding.__init__:
#        self.img_processor = get_siglip_vision_model(...)
#        pe_weight = self.img_processor.embeddings.position_embedding.weight
#        L, D = pe_weight.size()  # L = 32*32 for SigLIP 448×448 with 14×14 patches
#        H = int(sqrt(L))         # H = 32
#        self.num_img_tokens     = (H // 2) ** 2    # (32/2)^2 = 16^2 = 256
#        self.base_feat_height_target = H          # 32, overwritten to 16 with avg_pool_2d
#        self.image_token_compression_cls = 'avg_pool_2d' → AvgPool2d(k=2, s=2)
#        self.base_feat_height_reduction = 1
#        self.base_feat_height_target   //= 2      # 16
#      So: each 448×448 crop → 32×32 patch grid → 16×16 pooled grid, 256 tokens per crop.
#
#    • processing_phi4mm.Phi4MMImageProcessor.dynamic_preprocess:
#        – Base resolution = 448
#        – w_crop_num = ceil(orig_width  / 448)
#          h_crop_num = ceil(orig_height / 448)
#        – If w_crop_num * h_crop_num > dynamic_hd (preprocessor_config.json: dynamic_hd = 36),
#          choose a (i, j) ∈ ℕ² with 1 ≤ i*j ≤ 36 that best matches the image aspect ratio.
#          target_width  = 448 * i
#          target_height = 448 * j
#        – Else, target_width  = 448 * w_crop_num
#               target_height = 448 * h_crop_num
#        – Resize + pad to (target_width, target_height) and construct a patch-mask tensor.
#
#      preprocess() then:
#        – Builds hd_images_reshape with shape:
#              (#crops, 3, 448, 448), where #crops = target_aspect_ratio[0] * target_aspect_ratio[1]
#          and prepends a 448×448 global view (so “max_num_crops” = 1 + local_crops).
#        – Builds image_attention_mask and a per-image scalar:
#              num_img_tokens[i] = 256 + 1 + mask.sum() + mask[:,0].sum() + 16
#          which corresponds to the HD-layout in Phi4MMImageEmbedding.forward.
#
#    • modeling_phi4mm.Phi4MMImageEmbedding.forward (HD branch):
#        – Consumes img_embeds: (num_images, max_num_crops, 3, 448, 448)
#        – Uses get_img_features → SigLIP encoder → patch features
#        – Asserts base_feat_height == base_feat_height_target (16) after compression
#        – For each image, with h, w from image_sizes//crop_size and B_ = h * w local crops:
#              temp_len (no mask) = (h*w + 1)*num_img_tokens + 1 + (h+1)*base_feat_height
#          or a slightly smaller value when image_attention_mask prunes padded tiles.
#        – Projects the per-image HD tensor via self.img_projection, which for our config is:
#              projection_cls = "mlp", use_hd_transform = True
#              img_projection = Sequential(
#                   Linear(image_dim_out * base_feat_height_reduction**2, hidden_size),
#                   GELU(),
#                   Linear(hidden_size, hidden_size)
#              )
#          with image_dim_out = 1152, hidden_size = 3072, base_feat_height_reduction = 1.
#
#    In our FLOP model we:
#      • Use the exact dynamic_hd cropping grid (same integer search as dynamic_preprocess)
#        but *ignore the fine-grained patch mask* when counting tokens, i.e. we assume
#        “no pruned tiles” and use the closed-form temp_len formula from the no-mask branch.
#        This slightly overestimates vision tokens for images that require large padding,
#        but is exact for images where padding is ≤ one patch in each dimension.
#
# 3. Mixture-of-LoRAs for modalities
#    • configuration_phi4mm.Phi4MMConfig:
#        vision_lora = {
#          "dp":   0.0,
#          "layer": "layers.*((self_attn\\.(qkv_proj|o_proj))|(mlp\\.(gate_up|down)_proj))",
#          "lora_alpha": 512,
#          "r": 256
#        }
#        speech_lora = { "dp": 0.01, "layer": "...", "r": 320, ... }
#
#    • HF discussions (e.g. /discussions/4):
#        – “Currently we separate base weight, vision lora weights, and speech lora weights,
#           and use set_lora_adapter('vision'/'speech') for weight switching.”
#        – For pure vision-language, they recommend merging LoRAs into base weights for speed.
#
#    • Phi4MMForCausalLM.forward:
#          if input_mode in [VISION_SPEECH, VISION]:
#              self.set_lora_adapter('vision')
#          elif input_mode == SPEECH:
#              self.set_lora_adapter('speech')
#          elif input_mode == LANGUAGE:
#              self.unset_lora_adapter()
#
#      i.e. *at most one* LoRA adapter is active at a time. There is no per-token mixture
#      in the shipped model; mixture-of-LoRAs is *between modalities*, not inside a single
#      forward pass.
#
#    In our FLOP model we:
#      • Add a vision-LoRA overhead term whenever use_vision_lora=True.
#      • Ignore speech LoRA (and the entire audio path) as requested.
#
# ============================================================================

import math
from typing import Dict, Any, Tuple

# ---- Helper for LoRA linear layers -----------------------------------------------------------

def lora_linear_flops_per_token(in_dim: int, out_dim: int, rank: int) -> int:
    """
    FLOPs for a single LoRA-augmented linear layer on ONE token.

    LoRA forward: y = x W + (alpha / r) * (x A) B
      • x: shape (..., in_dim)
      • A: (in_dim, rank)
      • B: (rank, out_dim)

    LoRA extra matmuls per token:
      • x @ A          : 2 * in_dim * rank FLOPs (mul+add)
      • (xA) @ B       : 2 * rank * out_dim FLOPs

    Total extra FLOPs per token = 2 * rank * (in_dim + out_dim)
    """
    if in_dim <= 0 or out_dim <= 0 or rank <= 0:
        return 0
    return 2 * rank * (in_dim + out_dim)


# ---- Text backbone constants (Phi-4-Mini) ----------------------------------------------------

PHI4MM_TEXT_HIDDEN = 3072            # config.hidden_size
PHI4MM_TEXT_LAYERS = 32              # config.num_hidden_layers
PHI4MM_TEXT_HEADS = 24               # config.num_attention_heads (in config.json)
PHI4MM_TEXT_KV_HEADS = 8             # config.num_key_value_heads
PHI4MM_TEXT_INTERMEDIATE = 8192      # config.intermediate_size
PHI4MM_TEXT_MLP_EXPANSION = PHI4MM_TEXT_INTERMEDIATE / PHI4MM_TEXT_HIDDEN
PHI4MM_VOCAB = 200064                # config.vocab_size
PHI4MM_TEXT_HEAD_DIM = PHI4MM_TEXT_HIDDEN // PHI4MM_TEXT_HEADS  # 3072 / 24 = 128

# Vision LoRA (only this adapter is modeled here)
PHI4MM_VISION_LORA_R = 256           # config.vision_lora["r"]
PHI4MM_VISION_LORA_ALPHA = 512       # config.vision_lora["lora_alpha"] (scale, no FLOP impact)


def _phi4_text_lora_flops_per_token(rank: int = PHI4MM_VISION_LORA_R) -> int:
    """
    Extra FLOPs per token, per *decoder layer*, when a single LoRA adapter
    (e.g. the “vision” LoRA) is active on:

      • self_attn.qkv_proj : Linear(H → op_size)
      • self_attn.o_proj   : Linear(H → H)
      • mlp.gate_up_proj   : Linear(H → 2M)  (fused SwiGLU gate+up)
      • mlp.down_proj      : Linear(M → H)

    Shapes from modeling_phi4mm.Phi4MMAttention and Phi4MMMLP:
      • head_dim      = H / num_heads
      • op_size       = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
                      = H + 2 * kv_heads * head_dim
      • gate_up_proj  : in_dim = H, out_dim = 2 * M
      • down_proj     : in_dim = M, out_dim = H
    """
    H = PHI4MM_TEXT_HIDDEN
    M = PHI4MM_TEXT_INTERMEDIATE
    heads = PHI4MM_TEXT_HEADS
    kv_heads = PHI4MM_TEXT_KV_HEADS
    d = PHI4MM_TEXT_HEAD_DIM

    op_size = heads * d + 2 * (kv_heads * d)

    f_qkv = lora_linear_flops_per_token(H, op_size, rank)
    f_o   = lora_linear_flops_per_token(H, H, rank)
    f_gate_up = lora_linear_flops_per_token(H, 2 * M, rank)
    f_down    = lora_linear_flops_per_token(M, H, rank)

    return f_qkv + f_o + f_gate_up + f_down


def _phi4_base_prompt_flops(seq_len: int) -> int:
    """
    Base (no-LoRA) Phi-4-Mini FLOPs for a prompt of length `seq_len`.

    Uses:
      • 32 layers of causal self-attention with GQA (24 heads, 8 KV heads),
      • SwiGLU MLP with expansion = 8192 / 3072,
      • RMSNorm pre/post attention, RMSNorm before logits,
      • LM head over vocab_size = 200064.

    Implemented via the existing `hybrid_flops` helper with:
      swiglu=True and mlp_expansion=PHI4MM_TEXT_MLP_EXPANSION.
    """
    if seq_len <= 0:
        return 0

    return hybrid_flops(
        batch_size=1,
        seq_len=seq_len,
        hidden_size=PHI4MM_TEXT_HIDDEN,
        num_full_attn_layers=PHI4MM_TEXT_LAYERS,
        num_sliding_attn_layers=0,
        num_mamba_layers=0,
        num_mlp_layers=PHI4MM_TEXT_LAYERS,
        window_size=None,
        is_2d=False,
        num_attn_heads=PHI4MM_TEXT_HEADS,
        gqa=True,
        gqa_groups=PHI4MM_TEXT_KV_HEADS,
        kv_channels=PHI4MM_TEXT_HEAD_DIM,
        attn_head_dim=PHI4MM_TEXT_HEAD_DIM,
        mlp_expansion=PHI4MM_TEXT_MLP_EXPANSION,
        swiglu=True,                  # fused gate_up_proj → SwiGLU MLP
        vocab_size=PHI4MM_VOCAB,
        attn_mode="causal",
    )


def _phi4_base_generation_flops(prompt_len: int, num_generated: int) -> int:
    """
    Base (no-LoRA) autoregressive generation FLOPs with KV caching, using
    the generic `autoregressive_generation_flops` helper.

    This counts:
      • Per new token:
          – RMSNorm + attention (KV-cache aware) + RMSNorm + MLP
          – Final RMSNorm + LM head + softmax
        across 32 layers.
    """
    if num_generated <= 0:
        return 0

    return autoregressive_generation_flops(
        prompt_len=prompt_len,
        num_generated=num_generated,
        hidden_size=PHI4MM_TEXT_HIDDEN,
        num_layers=PHI4MM_TEXT_LAYERS,
        num_heads=PHI4MM_TEXT_HEADS,
        mlp_expansion=PHI4MM_TEXT_MLP_EXPANSION,
        vocab_size=PHI4MM_VOCAB,
        gqa_groups=PHI4MM_TEXT_KV_HEADS,
        head_dim=PHI4MM_TEXT_HEAD_DIM,
        kv_channels=PHI4MM_TEXT_HEAD_DIM,
        num_full_attn_layers=PHI4MM_TEXT_LAYERS,
        num_sliding_attn_layers=0,
        sliding_window_size=0,
        sliding_is_2d=False,
        num_mamba_layers=0,
        swiglu=True,
    )


def phi4_mini_flops(
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    Text-only Φ-4-Mini FLOPs (no multimodal inputs, no LoRAs active).

    This corresponds to InputMode.LANGUAGE, where Phi4MMForCausalLM.forward
    calls `unset_lora_adapter()` before running the decoder.
    """
    lang_prompt_len = max(int(lang_prompt_len), 0)
    prompt_flops = _phi4_base_prompt_flops(lang_prompt_len)
    gen_flops = _phi4_base_generation_flops(lang_prompt_len, num_generated)

    total_fwd = prompt_flops + gen_flops
    mult = 3 if do_backward else 1

    return {
        "vision_flops": 0,
        "vision_projector_flops": 0,
        "audio_flops": 0,
        "lang_prompt_flops": prompt_flops * mult,
        "gen_flops": gen_flops * mult,
        "total_flops": total_fwd * mult,
    }


# ---- Vision encoder & dynamic HD cropping (approximate token count) --------------------------

PHI4MM_VISION_HIDDEN = 1152
PHI4MM_VISION_LAYERS = 27
PHI4MM_VISION_HEADS = 16
PHI4MM_VISION_INTERMEDIATE = 4304
PHI4MM_VISION_MLP_EXPANSION = PHI4MM_VISION_INTERMEDIATE / PHI4MM_VISION_HIDDEN

PHI4MM_VISION_IMAGE_SIZE = 448
PHI4MM_VISION_PATCH_SIZE = 14
PHI4MM_VISION_PATCH_TOKENS_PREPOOL = (
    (PHI4MM_VISION_IMAGE_SIZE // PHI4MM_VISION_PATCH_SIZE)
    * (PHI4MM_VISION_IMAGE_SIZE // PHI4MM_VISION_PATCH_SIZE)
)  # 32 × 32 = 1024 patches per 448×448 crop

# After AvgPool2d(k=2, s=2), base_feat_height_target is halved: 32 → 16.
PHI4MM_VISION_FEAT_HEIGHT = (PHI4MM_VISION_IMAGE_SIZE // PHI4MM_VISION_PATCH_SIZE) // 2  # 16
PHI4MM_VISION_PATCH_TOKENS_POSTPOOL = PHI4MM_VISION_FEAT_HEIGHT ** 2  # 16 × 16 = 256

PHI4MM_CROP_SIZE = 448
PHI4MM_DYNAMIC_HD_MAX = 36  # preprocessor_config.json["dynamic_hd"]


def _phi4mm_dynamic_hd_grid(
    height: int,
    width: int,
    max_num: int = PHI4MM_DYNAMIC_HD_MAX,
    image_size: int = PHI4MM_CROP_SIZE,
    min_num: int = 1,
) -> Tuple[int, int]:
    """
    Replicates the aspect-ratio grid logic of Phi4MMImageProcessor.dynamic_preprocess:

      • w_crop_num = ceil(W / image_size)
        h_crop_num = ceil(H / image_size)

      • If w_crop_num * h_crop_num <= max_num:
          target_aspect_ratio = (w_crop_num, h_crop_num)
        else:
          target_aspect_ratio = argmin_{(i,j), min_num ≤ i*j ≤ max_num}
                                  |(W/H) − (i/j)| with a tie-break favoring larger area.

    Returns:
      (h_tiles, w_tiles) = target_aspect_ratio.
    """
    if height <= 0 or width <= 0:
        return 1, 1

    w_crop_num = math.ceil(width / float(image_size))
    h_crop_num = math.ceil(height / float(image_size))

    if w_crop_num * h_crop_num <= max_num:
        return int(h_crop_num), int(w_crop_num)

    aspect_ratio = width / float(height)
    area = width * height

    # Construct the candidate set exactly as in dynamic_preprocess
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best = (1, 1)
    best_diff = float("inf")

    for i, j in target_ratios:
        target_aspect = i / float(j)
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best = (i, j)
        elif diff == best_diff:
            # Tie-break rule: prefer ratios that “cover” more of the original area.
            if area > 0.5 * image_size * image_size * i * j:
                best = (i, j)

    # dynamic_preprocess uses (w_tiles, h_tiles) = target_aspect_ratio;
    # our HD transform uses h, w from image_sizes // crop_size → (h_tiles, w_tiles).
    w_tiles, h_tiles = best
    return int(h_tiles), int(w_tiles)


def _phi4mm_vision_backbone_flops_per_crop() -> int:
    """
    FLOPs for running a single 448×448 crop through the SigLIP vision encoder:

      1. Conv2d patch embedding: 3×14×14 → 1152
      2. 27-layer ViT with hidden=1152, heads=16, MLP intermediate=4304
      3. AvgPool2d(k=2, s=2) on the 32×32 patch grid → 16×16

    The 27-layer encoder is counted via `vision_encoder_flops`, which uses
    bidirectional_attention_flops + mlp_layer_flops + RMSNorms.
    """
    in_channels = 3
    patch = PHI4MM_VISION_PATCH_SIZE
    patches = PHI4MM_VISION_PATCH_TOKENS_PREPOOL  # 1024
    kernel_area = patch * patch                   # 14×14 = 196

    # Conv2d patch embedding: 2 * (#patches) * kernel_area * Cin * Cout
    patch_embed = (
        2
        * patches
        * kernel_area
        * in_channels
        * PHI4MM_VISION_HIDDEN
    )

    vit = vision_encoder_flops(
        patches=patches,
        hidden_size=PHI4MM_VISION_HIDDEN,
        num_layers=PHI4MM_VISION_LAYERS,
        num_heads=PHI4MM_VISION_HEADS,
        mlp_expansion=PHI4MM_VISION_MLP_EXPANSION,
    )

    # AvgPool2d(k=2,s=2) over 32×32→16×16 per channel: approximate as 4 FLOPs per pooled value.
    tokens_out = PHI4MM_VISION_PATCH_TOKENS_POSTPOOL  # 256
    pool = 4 * tokens_out * PHI4MM_VISION_HIDDEN

    return patch_embed + vit + pool


def phi4mm_vision_pipeline_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
) -> Dict[str, int]:
    """
    Vision-side FLOPs and token count for Φ-4-Multimodal, assuming:

      • Each frame is processed independently via the dynamic HD preprocessing.
      • SigLIP encoder runs on:
          – 1 “global” 448×448 crop, plus
          – h*w “local” 448×448 crops, where (h, w) is the dynamic HD grid.
      • HD transform uses the *no-mask* token-length formula:

            temp_len ≈ (h*w + 1) * num_img_tokens + 1 + (h + 1) * base_feat_height

        where:
          num_img_tokens = 256          (# tokens per crop after pooling)
          base_feat_height = 16         (pooled spatial height)

        This matches the else-branch in Phi4MMImageEmbedding.forward when
        image_attention_mask is absent; with masks, the true token count
        is slightly smaller if the image requires heavy padding.
    """
    if vision_frames <= 0 or vision_height <= 0 or vision_width <= 0:
        return {
            "vision_backbone_flops": 0,
            "vision_projector_flops": 0,
            "vision_tokens": 0,
            "tokens_per_frame": 0,
            "crops_per_frame": 0,
            "crop_grid": (0, 0),
        }

    # Dynamic HD grid (h_tiles × w_tiles)
    h_tiles, w_tiles = _phi4mm_dynamic_hd_grid(
        height=vision_height,
        width=vision_width,
        max_num=PHI4MM_DYNAMIC_HD_MAX,
        image_size=PHI4MM_CROP_SIZE,
        min_num=1,
    )
    local_crops = h_tiles * w_tiles
    crops_per_frame = 1 + local_crops   # 1 global + local grid

    # (1) SigLIP encoder FLOPs
    per_crop = _phi4mm_vision_backbone_flops_per_crop()
    vision_backbone = vision_frames * crops_per_frame * per_crop

    # (2) HD-transform token count per frame (approx, ignoring mask pruning)
    num_img_tokens = PHI4MM_VISION_PATCH_TOKENS_POSTPOOL  # 256
    base_feat_height = PHI4MM_VISION_FEAT_HEIGHT          # 16

    tokens_per_frame = (
        (local_crops + 1) * num_img_tokens   # global + locals
        + 1                                  # global GN separator
        + (h_tiles + 1) * base_feat_height   # line separators
    )
    vision_tokens = vision_frames * tokens_per_frame

    # (3) MLP projector FLOPs: 1152 → 3072 → 3072 on all HD tokens
    proj_expansion = PHI4MM_TEXT_HIDDEN / PHI4MM_VISION_HIDDEN  # 3072 / 1152 = 8/3
    projector_per_frame = mlp_merger_general(
        batch_size=1,
        seq_len=tokens_per_frame,
        input_size=PHI4MM_VISION_HIDDEN,
        output_size=PHI4MM_TEXT_HIDDEN,
        expansion=proj_expansion,
        swiglu=False,   # GELU activation in the projector
    )
    vision_projector = vision_frames * projector_per_frame

    return {
        "vision_backbone_flops": vision_backbone,
        "vision_projector_flops": vision_projector,
        "vision_tokens": vision_tokens,
        "tokens_per_frame": tokens_per_frame,
        "crops_per_frame": crops_per_frame,
        "crop_grid": (h_tiles, w_tiles),
    }


# ---- Full Phi-4-Multimodal FLOPs (vision + text, no audio) -----------------------------------

def phi4mm_flops(
    *,
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    text_tokens: int,
    num_generated: int,
    use_vision_lora: bool = True,
    do_backward: bool = False,
) -> Dict[str, Any]:
    """
    FLOPs for the published `microsoft/Phi-4-multimodal-instruct` checkpoint
    (Φ-4-MM) in the **vision+language** regime, ignoring audio and speech LoRA.

    Model flow (HF code path):
      1. Images/frames → Phi4MMImageProcessor (dynamic HD) →
         input_image_embeds, image_sizes, image_attention_mask, num_img_tokens.

      2. Phi4MMImageEmbedding.forward:
           - Runs SigLIP on each crop (global + local)
           - Applies HD transform into a per-image HD feature tensor
           - Projects via a 2-layer MLP into 3072-dim language tokens
           - Injects these tokens into the text sequence at <image> positions

      3. Phi4MMForCausalLM.forward:
           - If input_mode ∈ {VISION, VISION_SPEECH}, calls set_lora_adapter("vision")
             so all decoder layers use the “vision” LoRA on
             qkv_proj, o_proj, gate_up_proj, down_proj.
           - Runs the 32-layer Phi-4-Mini decoder (GQA + SwiGLU MLP)
           - Applies lm_head to get logits.

      4. Generation:
           - Uses KV caching; only new tokens pay attention, MLP, LoRA and LM-head.

    FLOP breakdown:
      • vision_backbone_flops   : SigLIP encoder + AvgPool over all crops
      • vision_projector_flops  : MLP projector from vision to text space
      • lang_prompt_flops       : base decoder FLOPs on (text + vision) prompt
      • lora_prompt_flops       : extra LoRA FLOPs on the same prompt
      • gen_flops               : base decoder FLOPs for `num_generated` tokens
      • lora_gen_flops          : extra LoRA FLOPs for `num_generated` tokens
      • total_flops             : sum of the above (×3 if do_backward=True)

    Args
    ----
    vision_frames:
        Number of frames (1 for image, N for video). Set 0 for text-only.
    vision_height, vision_width:
        Original spatial resolution of each frame.
    text_tokens:
        Number of *text* tokens in the prompt (excluding multimodal special tokens).
    num_generated:
        Number of autoregressive tokens after the prompt.
    use_vision_lora:
        Whether a vision LoRA adapter is active (i.e. InputMode.VISION or VISION_SPEECH).
        If you have merged the LoRA into the base weights, you can set this to False.
    do_backward:
        If True, multiply all counts by 3 to approximate fwd + bwd + optimizer update.
    """
    text_tokens = max(int(text_tokens), 0)
    num_generated = max(int(num_generated), 0)

    # 1) Vision side ----------------------------------------------------------
    vision_info = phi4mm_vision_pipeline_flops(
        vision_frames=vision_frames,
        vision_height=vision_height,
        vision_width=vision_width,
    )
    vision_backbone_flops = vision_info["vision_backbone_flops"]
    vision_projector_flops = vision_info["vision_projector_flops"]
    vision_tokens = vision_info["vision_tokens"]

    # 2) Text side (Phi-4-Mini decoder) --------------------------------------
    prompt_len = text_tokens + vision_tokens
    base_prompt_flops = _phi4_base_prompt_flops(prompt_len)
    base_gen_flops = _phi4_base_generation_flops(prompt_len, num_generated)

    # 3) LoRA overhead (vision adapter only) ---------------------------------
    lora_prompt_flops = 0
    lora_gen_flops = 0
    if use_vision_lora and prompt_len > 0:
        per_token_per_layer = _phi4_text_lora_flops_per_token(PHI4MM_VISION_LORA_R)
        lora_prompt_flops = PHI4MM_TEXT_LAYERS * prompt_len * per_token_per_layer
        lora_gen_flops = PHI4MM_TEXT_LAYERS * num_generated * per_token_per_layer

    lang_prompt_flops = base_prompt_flops + lora_prompt_flops
    gen_flops = base_gen_flops + lora_gen_flops

    # 4) Total ---------------------------------------------------------------
    total_fwd = (
        vision_backbone_flops
        + vision_projector_flops
        + lang_prompt_flops
        + gen_flops
    )
    mult = 3 if do_backward else 1

    return {
        # Vision
        "vision_backbone_flops": vision_backbone_flops * mult,
        "vision_projector_flops": vision_projector_flops * mult,
        "vision_tokens": vision_tokens,
        "tokens_per_frame": vision_info["tokens_per_frame"],
        "crops_per_frame": vision_info["crops_per_frame"],
        "crop_grid": vision_info["crop_grid"],

        # Audio (explicitly ignored here)
        "audio_flops": 0,
        "audio_tokens": 0,

        # Text
        "lang_prompt_flops": lang_prompt_flops * mult,
        "gen_flops": gen_flops * mult,

        # LoRA breakdown (so you can inspect overhead independently)
        "lora_prompt_flops": lora_prompt_flops * mult,
        "lora_gen_flops": lora_gen_flops * mult,

        # Tokens
        "text_tokens": text_tokens,
        "prompt_tokens_total": prompt_len,

        # Total
        "total_flops": total_fwd * mult,
    }


# ============================================================================
# LongVILA-R1-7B FLOPs (SigLIP + Qwen2-7B + mlp_downsample_2x2_fix + TSP)
# ============================================================================
# This section adds *new* helpers only. It does NOT modify any existing
# primitives (mlp_layer_flops, hybrid_flops, autoregressive_generation_flops,
# vision_encoder_flops, etc.), so other model-specific functions continue to
# work unchanged.
#
# Sources:
#   - LongVILA-R1-7B config (your JSON snippet), esp. llm_cfg & vision_tower_cfg.
#   - NVILA / Omnivinci implementation:
#       * modeling_vila.py: encode_images(), encode_video()
#       * base_projector.py: MultimodalProjector, mm_projector_type="mlp_downsample_2x2_fix"
#       * siglip_encoder.py: SiglipVisionTower / VisionTower.num_patches
#       * media_encoder.py: BasicImageEncoder, BasicVideoEncoder, TSPVideoEncoder
# ============================================================================

# ---- LLM (Qwen2-7B) from llm_cfg -------------------------------------------

LONGVILA_R1_TEXT_HIDDEN = 3584          # llm_cfg.hidden_size
LONGVILA_R1_TEXT_LAYERS = 28            # llm_cfg.num_hidden_layers
LONGVILA_R1_TEXT_HEADS = 28             # llm_cfg.num_attention_heads
LONGVILA_R1_TEXT_KV_HEADS = 4           # llm_cfg.num_key_value_heads
LONGVILA_R1_TEXT_INTERMEDIATE = 18944   # llm_cfg.intermediate_size
LONGVILA_R1_TEXT_MLP_EXPANSION = LONGVILA_R1_TEXT_INTERMEDIATE / LONGVILA_R1_TEXT_HIDDEN
LONGVILA_R1_TEXT_VOCAB = 151651         # llm_cfg.vocab_size
LONGVILA_R1_TEXT_HEAD_DIM = LONGVILA_R1_TEXT_HIDDEN // LONGVILA_R1_TEXT_HEADS  # 128


# ---- Vision (SigLIP so400m patch14, resized to 448x448) --------------------
# From vision_tower_cfg in your LongVILA config and NVILA's SiglipVisionTower. 

LONGVILA_R1_VISION_IMAGE_SIZE = 448     # vision_tower_cfg.image_size
LONGVILA_R1_VISION_PATCH = 14           # vision_tower_cfg.patch_size
LONGVILA_R1_VISION_HIDDEN = 1152        # vision_tower_cfg.hidden_size
LONGVILA_R1_VISION_LAYERS = 27          # vision_tower_cfg.num_hidden_layers
LONGVILA_R1_VISION_HEADS = 16           # vision_tower_cfg.num_attention_heads
LONGVILA_R1_VISION_INTERMEDIATE = 4304  # vision_tower_cfg.intermediate_size
LONGVILA_R1_VISION_MLP_EXPANSION = LONGVILA_R1_VISION_INTERMEDIATE / LONGVILA_R1_VISION_HIDDEN

# Number of patch tokens per 448x448 frame (32x32). VisionTower.num_patches. 
LONGVILA_R1_VISION_PATCHES_PER_FRAME = (
    (LONGVILA_R1_VISION_IMAGE_SIZE // LONGVILA_R1_VISION_PATCH)
    * (LONGVILA_R1_VISION_IMAGE_SIZE // LONGVILA_R1_VISION_PATCH)
)  # 32 * 32 = 1024

# SigLIP uses a CLS token; VisionTower.feature_select("cls_patch") keeps it.
# We approximate transformer seq_len = patches + 1 (CLS); patch embedding
# only covers patch tokens (1024), CLS is a learned vector. 
LONGVILA_R1_VISION_SEQ_LEN = LONGVILA_R1_VISION_PATCHES_PER_FRAME + 1  # 1025

# Projector 'mlp_downsample_2x2_fix' details from base_projector.py: 
#   DownSample2x2BlockFix -> 2x2 spatial pooling: 32x32 -> 16x16 tokens.
#   LayerNorm(4 * mm_hidden_size).
#   Linear(4*mm_hidden_size -> hidden_size) + GELU + Linear(hidden_size -> hidden_size).
LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME = LONGVILA_R1_VISION_PATCHES_PER_FRAME // 4  # 256
LONGVILA_R1_PROJECTOR_IN_DIM = LONGVILA_R1_VISION_HIDDEN * 4   # 4 * 1152 = 4608
LONGVILA_R1_PROJECTOR_OUT_DIM = LONGVILA_R1_TEXT_HIDDEN        # 3584

# For TSPVideoEncoder pool_sizes=[[8,1,1]] (temporal, H, W). 
LONGVILA_R1_TSP_TEMPORAL_POOL = 8


# ---------------------------------------------------------------------------
#  SigLIP tower FLOPs per 448x448 frame
# ---------------------------------------------------------------------------

def _longvila_r1_siglip_frame_flops() -> int:
    """
    FLOPs for one 448x448 frame through the SigLIP vision backbone used by
    LongVILA-R1-7B.

    Components:
      1) Conv2d patch embedding: 3x14x14 -> 1152 over 32x32 patches.
      2) 27-layer ViT encoder (bidirectional attention + GeLU MLP) modeled via
         vision_encoder_flops(patches=LONGVILA_R1_VISION_SEQ_LEN,...).

    Notes:
      - We use vision_encoder_flops, which internally uses RMSNorm cost;
        SigLIP uses LayerNorm, but the FLOP difference is small and we keep
        the same convention as the rest of your library.
    """
    # 1) Patch embedding conv FLOPs (no CLS here, only patches).
    patches = LONGVILA_R1_VISION_PATCHES_PER_FRAME  # 1024
    kernel_area = LONGVILA_R1_VISION_PATCH * LONGVILA_R1_VISION_PATCH  # 14 * 14
    in_channels = 3
    out_channels = LONGVILA_R1_VISION_HIDDEN

    # Conv FLOPs: 2 * (#output_positions) * (kernel_area * Cin) * Cout
    patch_embed_flops = 2 * patches * (kernel_area * in_channels) * out_channels

    # 2) Transformer encoder FLOPs over seq_len = patches + CLS.
    vit_flops = vision_encoder_flops(
        patches=LONGVILA_R1_VISION_SEQ_LEN,          # 1025 tokens (1024 patches + 1 CLS)
        hidden_size=LONGVILA_R1_VISION_HIDDEN,       # 1152
        num_layers=LONGVILA_R1_VISION_LAYERS,        # 27
        num_heads=LONGVILA_R1_VISION_HEADS,          # 16
        mlp_expansion=LONGVILA_R1_VISION_MLP_EXPANSION,  # 4304 / 1152
    )

    return int(patch_embed_flops + vit_flops)


# ---------------------------------------------------------------------------
#  mm_projector: mlp_downsample_2x2_fix FLOPs
# ---------------------------------------------------------------------------

def _longvila_r1_projector_flops(num_frames: int) -> int:
    """
    FLOPs for the MultimodalProjector with mm_projector_type="mlp_downsample_2x2_fix"
    over `num_frames` frames.

    From base_projector.py: 
        self.layers = nn.Sequential(
            DownSample2x2BlockFix(),
            nn.LayerNorm(config.mm_hidden_size * 4),
            nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.downsample_rate = 2

    We count:
      - DownSample2x2BlockFix as 0 FLOPs (pure reshape + zero padding).
      - LayerNorm over 4 * mm_hidden_size on 256 tokens per frame.
      - Linear(4C -> H), GELU, Linear(H -> H) on each token.
    """
    if num_frames <= 0:
        return 0

    B = int(num_frames)
    L = LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME      # 256
    C4 = LONGVILA_R1_PROJECTOR_IN_DIM               # 4608
    H = LONGVILA_R1_PROJECTOR_OUT_DIM               # 3584

    # LayerNorm(4C) per token.
    ln_flops = layernorm_flops(batch_size=B, seq_len=L, hidden_size=C4)

    # MLP: 4C -> H -> H with GeLU in between.
    # up_proj: 2 * B * L * (4C) * H
    up_proj = 2 * B * L * C4 * H
    # GeLU ~ 4 FLOPs per hidden unit per token (consistent with mlp_layer_flops doc).
    act = 4 * B * L * H
    # down_proj: 2 * B * L * H * H
    down_proj = 2 * B * L * H * H

    return int(ln_flops + up_proj + act + down_proj)


# ---------------------------------------------------------------------------
#  TSPVideoEncoder temporal pooling FLOPs (pool_sizes=[[8,1,1]])
# ---------------------------------------------------------------------------

def _longvila_r1_tsp_pool_flops(
    num_frames: int,
    tokens_per_frame: int = LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME,
    hidden_size: int = LONGVILA_R1_TEXT_HIDDEN,
    temporal_pool_size: int = LONGVILA_R1_TSP_TEMPORAL_POOL,
) -> int:
    """
    FLOPs for the temporal pooling inside TSPVideoEncoder._process_features()
    when pool_sizes=[[8,1,1]]. 

    In media_encoder.py, pooling is implemented as:

        def pool(x, size, dim):
            # pad if needed, then
            x = x.view(..., -1, size, ...)
            return x.mean(dim+1)

    For temporal pooling with size=8 along dim=0 on features of shape
      [T, H, W, D] (after mm_projector & reshape),
    the number of scalar ops is approximately:

        FLOPs_pool ≈ T_padded * H * W * D

    where T_padded = ceil(T / 8) * 8, assuming 1 FLOP per add / multiply.
    """
    if num_frames <= 0 or temporal_pool_size <= 1:
        return 0

    # Number of scalar values in features: T * tokens_per_frame * hidden_size.
    # Padding makes T_padded a multiple of pool_size.
    T = int(num_frames)
    T_padded = int(math.ceil(T / float(temporal_pool_size)) * temporal_pool_size)
    P = int(tokens_per_frame)
    D = int(hidden_size)

    # Each output element uses `size` inputs; total scalar ops ≈ T_padded * P * D.
    flops = T_padded * P * D
    return flops


# ---------------------------------------------------------------------------
#  Qwen2-7B LLM FLOPs (prefill + generation)
# ---------------------------------------------------------------------------

def _longvila_r1_lm_flops_for_prompt_and_gen(
    prompt_len: int,
    num_generated: int,
) -> Dict[str, int]:
    """
    LLM FLOPs for LongVILA-R1-7B's text stack (Qwen2-7B):

      - hidden_size       = 3584
      - num_hidden_layers = 28
      - num_attention_heads = 28
      - num_key_value_heads = 4
      - intermediate_size = 18944
      - vocab_size        = 151651
      - hidden_act        = "silu" (SwiGLU-style MLP)

    Uses your existing `hybrid_flops` and `autoregressive_generation_flops`
    primitives, with full causal attention and SwiGLU MLP.
    """
    if prompt_len < 0 or num_generated < 0:
        raise ValueError("prompt_len and num_generated must be >= 0")

    head_dim = LONGVILA_R1_TEXT_HEAD_DIM  # 128

    # Prefill over the entire prompt.
    prompt_flops = 0
    if prompt_len > 0:
        prompt_flops = hybrid_flops(
            batch_size=1,
            seq_len=int(prompt_len),
            hidden_size=LONGVILA_R1_TEXT_HIDDEN,
            num_full_attn_layers=LONGVILA_R1_TEXT_LAYERS,
            num_sliding_attn_layers=0,
            num_mamba_layers=0,
            num_mlp_layers=LONGVILA_R1_TEXT_LAYERS,
            window_size=None,
            is_2d=False,
            num_attn_heads=LONGVILA_R1_TEXT_HEADS,
            gqa=True,
            gqa_groups=LONGVILA_R1_TEXT_KV_HEADS,
            kv_channels=head_dim,
            attn_head_dim=head_dim,
            mlp_expansion=LONGVILA_R1_TEXT_MLP_EXPANSION,
            swiglu=True,
            vocab_size=LONGVILA_R1_TEXT_VOCAB,
            attn_mode="causal",
        )

    # Autoregressive generation with KV cache.
    gen_flops = 0
    if num_generated > 0:
        gen_flops = autoregressive_generation_flops(
            prompt_len=int(prompt_len),
            num_generated=int(num_generated),
            hidden_size=LONGVILA_R1_TEXT_HIDDEN,
            num_layers=LONGVILA_R1_TEXT_LAYERS,
            num_heads=LONGVILA_R1_TEXT_HEADS,
            mlp_expansion=LONGVILA_R1_TEXT_MLP_EXPANSION,
            vocab_size=LONGVILA_R1_TEXT_VOCAB,
            gqa_groups=LONGVILA_R1_TEXT_KV_HEADS,
            head_dim=head_dim,
            kv_channels=head_dim,
            num_full_attn_layers=LONGVILA_R1_TEXT_LAYERS,
            num_sliding_attn_layers=0,
            sliding_window_size=0,
            sliding_is_2d=False,
            num_mamba_layers=0,
            swiglu=True,
        )

    return {
        "lang_prompt_flops": int(prompt_flops),
        "gen_flops": int(gen_flops),
    }


# ---------------------------------------------------------------------------
#  Public entry point: LongVILA-R1-7B FLOPs
# ---------------------------------------------------------------------------

def longvila_r1_7b_flops(
    vision_frames: int,
    vision_height: int,
    vision_width: int,
    lang_prompt_len: int,
    num_generated: int,
    do_backward: bool = False,
) -> Dict[str, int]:
    """
    FLOPs for Efficient-Large-Model/LongVILA-R1-7B on a single example.

    Args
    ----
    vision_frames:
        - 0  → text-only (no vision).
        - 1  → single image (BasicImageEncoder path, no TSP).
        - >1 → single video with this many frames (TSPVideoEncoder path).

        In ALL cases, images/frames are effectively processed at 448x448 by
        SiglipImageProcessor + SiglipVisionModel, regardless of the original
        `vision_height` and `vision_width`. 

    vision_height, vision_width:
        Accepted for API symmetry but *ignored* in FLOP counting, since the
        HF preprocessor resizes to the fixed 448x448 resolution for this model.

    lang_prompt_len:
        Number of **language tokens only** in the prompt (system + user text).
        This must NOT include any vision tokens (patch embeddings, start/end
        tokens, etc.). Those are computed internally here and then added.

    num_generated:
        Number of autoregressive tokens generated after the prompt.

    do_backward:
        If True, multiplies all FLOP counts by 3 to approximate forward +
        backward + optimizer update (consistent with your other model helpers).

    Returns
    -------
    dict with keys:
      - vision_flops        : SigLIP + projector + (if video) TSP pooling.
      - lang_prompt_flops   : LLM prefill FLOPs.
      - gen_flops           : LLM generation FLOPs.
      - total_flops         : Sum of all components (×3 if do_backward=True).
      - vision_tokens       : #visual tokens inserted into the LLM.
      - text_tokens         : lang_prompt_len (for convenience).
      - prompt_tokens_total : text_tokens + vision_tokens.
      - mode                : "text", "image", or "video".
    """
    if vision_frames < 0 or lang_prompt_len < 0 or num_generated < 0:
        raise ValueError("vision_frames, lang_prompt_len, and num_generated must be >= 0")

    text_tokens = int(lang_prompt_len)
    mode = "text"
    vision_tokens = 0
    vision_flops = 0

    # --- Case 1: no vision --------------------------------------------------
    if vision_frames == 0:
        # text-only: vision_flops=0, vision_tokens=0
        pass

    # --- Case 2: single image (BasicImageEncoder path, no TSP) -------------
    elif vision_frames == 1:
        mode = "image"

        # Vision backbone + projector for 1 frame.
        siglip_flops = _longvila_r1_siglip_frame_flops()
        projector_flops = _longvila_r1_projector_flops(num_frames=1)

        # BasicImageEncoder adds one end token (newline) by default, no start.
        # So visual tokens into LLM: 256 projector tokens + 1 end token.
        vision_tokens = LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME + 1  # 257

        vision_flops = siglip_flops + projector_flops

    # --- Case 3: video with TSPVideoEncoder --------------------------------
    else:
        mode = "video"
        T = int(vision_frames)

        # SigLIP backbone + projector per frame (encode_video -> encode_images).
        siglip_flops = _longvila_r1_siglip_frame_flops() * T
        projector_flops = _longvila_r1_projector_flops(num_frames=T)

        # TSP temporal pooling over time (pool_sizes=[[8,1,1]]).
        tsp_flops = _longvila_r1_tsp_pool_flops(
            num_frames=T,
            tokens_per_frame=LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME,
            hidden_size=LONGVILA_R1_TEXT_HIDDEN,
            temporal_pool_size=LONGVILA_R1_TSP_TEMPORAL_POOL,
        )

        # After pooling, we have nt_pooled time steps, each with 256 projector
        # tokens. BasicVideoEncoder _process_features adds one end token per
        # pooled time step, then flattens (no sep_tokens by default). 
        nt_pooled = int(math.ceil(T / float(LONGVILA_R1_TSP_TEMPORAL_POOL)))
        vision_tokens = nt_pooled * (LONGVILA_R1_PROJECTOR_TOKENS_PER_FRAME + 1)

        vision_flops = siglip_flops + projector_flops + tsp_flops

    # --- LLM FLOPs ----------------------------------------------------------
    prompt_len = text_tokens + vision_tokens
    lm_info = _longvila_r1_lm_flops_for_prompt_and_gen(
        prompt_len=prompt_len,
        num_generated=num_generated,
    )
    prompt_flops = lm_info["lang_prompt_flops"]
    gen_flops = lm_info["gen_flops"]

    total_fwd = vision_flops + prompt_flops + gen_flops
    mult = 3 if do_backward else 1

    return {
        "vision_flops": int(vision_flops * mult),
        "lang_prompt_flops": int(prompt_flops * mult),
        "gen_flops": int(gen_flops * mult),
        "total_flops": int(total_fwd * mult),
        "vision_tokens": int(vision_tokens),
        "text_tokens": int(text_tokens),
        "prompt_tokens_total": int(prompt_len),
        "mode": mode,
    }



# ============================================================================
# Embedding models
# ============================================================================

def qwen3_embedding_0p6b_flops(
    batch_size: int,
    seq_len_tokens: int,
    do_backward: bool = False,
) -> float:
    """
    FLOPs for a single forward (or forward+backward) pass of Qwen3-Embedding-0.6B.

    Architecture (from Qwen/Qwen3-Embedding-0.6B config.json):

        model_type           = "qwen3"
        architectures        = ["Qwen3ForCausalLM"]
        hidden_size          = 1024
        num_hidden_layers    = 28
        num_attention_heads  = 16
        num_key_value_heads  = 8
        head_dim             = 128
        intermediate_size    = 3072
        hidden_act           = "silu"
        max_position_embeddings = 32768
        vocab_size           = 151669

    Embedding usage (from the Qwen3 Embedding docs/model card):

        model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        out = model(**inputs)
        emb = last_token_pool(out.last_hidden_state, attention_mask)
        emb = F.normalize(emb, p=2, dim=1)

    That is: we use the *base transformer* with a causal mask (decoder-style),
    but we do NOT apply the LM head / logits / softmax when computing
    embeddings. This helper therefore excludes LM-head FLOPs.

    Args
    ----
    batch_size:
        Number of sequences processed together.
    seq_len_tokens:
        Number of tokens per sequence (must be ≤ 32768 for this model).
    do_backward:
        If True, multiply FLOPs by ≈3 to account for forward + backward +
        optimizer update.

    Returns
    -------
    float
        Estimated FLOPs for the embedding computation on the batch.
    """
    if batch_size <= 0 or seq_len_tokens <= 0:
        return 0.0

    # Exact architectural constants from the HF config.
    num_layers = 28
    hidden_size = 1024
    num_heads = 16
    kv_heads = 8
    intermediate_size = 3072
    # Dense SiLU MLP (not SwiGLU): expansion = intermediate / hidden
    mlp_expansion = intermediate_size / hidden_size  # = 3.0
    head_dim = hidden_size // num_heads

    # We use the base transformer only (no LM head) because the recommended
    # embedding usage calls AutoModel (Qwen3Model), not Qwen3ForCausalLM.
    fwd = hybrid_flops(
        batch_size=batch_size,
        seq_len=seq_len_tokens,
        hidden_size=hidden_size,
        num_full_attn_layers=num_layers,
        num_sliding_attn_layers=0,
        num_mamba_layers=0,
        num_mlp_layers=num_layers,
        window_size=None,
        is_2d=False,
        num_attn_heads=num_heads,
        # Qwen3 uses grouped-query attention: 16 query heads, 8 KV heads.
        gqa=True,
        gqa_groups=kv_heads,
        kv_channels=head_dim,
        attn_head_dim=head_dim,
        mlp_expansion=mlp_expansion,
        swiglu=False,      # dense SiLU MLP, not SwiGLU
        vocab_size=0,      # LM head not used in embedding pipeline
        attn_mode="causal",
    )

    multiplier = 3 if do_backward else 1
    return float(fwd * multiplier)
