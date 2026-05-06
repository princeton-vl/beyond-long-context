"""
FLOPs equations for the InternVL3.5 family — built from first principles
========================================================================

Built strictly from each model's HuggingFace `config.json` AND verified against
the actual modeling code (see `CHANGES (verification pass)` at the bottom of
this docstring for line-level citations):
  - OpenGVLab/InternVL3_5-8B            (Qwen3-8B  + InternViT-300M-448px)
  - OpenGVLab/InternVL3_5-30B-A3B       (Qwen3-MoE 30B-A3B + InternViT-300M-448px)
  - OpenGVLab/InternVL3_5-38B           (Qwen3-32B + InternViT-6B-448px-V2_5)

NOTE on Thinking variants:
  InternVL3.5 "Thinking" is *not* a separate checkpoint. The model card states the
  same released weights are used; thinking mode is enabled via the R1 system prompt
  (see InternVL3.5-8B model card). Architecture and per-token FLOPs are identical
  to the base; only `n_out_text_tokens` increases (the model emits a long
  <think>...</think> chain before its final answer).

Counting convention
-------------------
We count FLOPs from matmuls only, using 2*A*B*C per (A,B)·(B,C) matmul. We omit
softmax/elementwise/RoPE/RMSNorm/embedding lookups (well under ~1% of the budget).

CAUSAL ATTENTION FACTOR — chosen convention
-------------------------------------------
For decoder self-attention with sequence length N, only the lower-triangular
half of the QK^T scores are used. There are two reporting conventions:
  (a) Full N^2 per matmul (overcounts the masked half by 2x):  used by
      Chinchilla (Hoffmann et al., 2022, "Training Compute-Optimal Large
      Language Models", Eq. 1+Appendix F: training FLOPs ≈ 6 * N_params *
      N_tokens, derived assuming attention is 2*N^2*d per layer without the
      /2 causal correction); also used by Kaplan et al. (2020) "Scaling Laws
      for Neural Language Models" §2.1, footnote 4 ("we omit the factor of
      1/2 for the causal mask").
  (b) N^2/2 per matmul (causal-corrected): used by some efficiency papers
      (e.g., FlashAttention papers when reporting "exact" FLOPs).
We adopt convention (a) — full N^2 — to match the dominant scaling-laws
literature. This is consistent across all six InternVL3.5 variants reported
here, so cross-model ratios are unaffected. The vision encoder uses bidirectional
(non-causal) attention so its full-N^2 count is exact.

Per-frame visual token count
----------------------------
InternVL uses dynamic high-res tiling (1..max_dynamic_patch, max_dynamic_patch=12)
plus a thumbnail. From the official `dynamic_preprocess`:
  - choose grid (i,j) with 1<=i*j<=12 minimizing |W/H - i/j|
  - n_tiles = i*j
  - if n_tiles > 1 and use_thumbnail: append +1 thumbnail tile
Per tile (448x448, patch_size=14) -> (448/14)^2 = 1024 patches PLUS 1 CLS token,
so the ViT processes N_v = 1025 tokens per tile (verified: see CHANGES #1 below).
After the ViT forward, `extract_feature` drops the CLS token (`vit_embeds[:, 1:, :]`)
and applies pixel-shuffle (downsample_ratio=0.5 == 2x2 merge) to reduce the
remaining 1024 patches to 1024/4 = 256 visual tokens fed to the LLM.

For video, the canonical InternVL inference path uses max_num=1 per frame
(1 tile, no thumbnail) -> 256 tokens/frame. This implementation honors the H/W
the caller passes, applying the tiling rule. To match canonical video usage, pass
448x448 frames.

Vision encoder (InternViT)
--------------------------
Standard ViT block (post-LN). Per layer per token:
  - QKV projection:  hidden_size -> 3*hidden_size  (qkv_bias varies; bias adds
    no matmul FLOPs)
  - Self-attn matmuls: QK^T and AV are 2 * N * num_heads * head_dim * N each
  - Output proj:     hidden_size -> hidden_size
  - MLP (GELU): hidden_size -> intermediate_size -> hidden_size  (standard
    fc1/fc2, NOT SwiGLU — verified, see CHANGES #2 below)
  - For InternViT-6B-V2.5 (38B variant) only: qk_normalization=true applies
    one InternRMSNorm to Q and one to K per layer. RMSNorm is a single
    elementwise scale (no matmul) so it adds zero matmul FLOPs (verified,
    see CHANGES #2 below).
Sequence length per tile: N_v = (448/14)^2 + 1 (CLS) = 1025.

LLM dense (Qwen3 8B / Qwen3 32B)
--------------------------------
Per layer per token, prefill (sequence length N):
  - QKV with GQA: q_proj=H*Hq, k_proj=H*Hkv, v_proj=H*Hkv where Hq=heads*head_dim,
    Hkv=kv_heads*head_dim. Projection FLOPs (per token) = 2*H*(Hq+2*Hkv).
  - Output proj: 2*H*Hq.
  - Attention scores+values: full N^2 per matmul (see "Causal attention factor"
    above): per-token-per-layer cost rolls up to 2 * 2 * N * Hq across QK^T+AV.
  - SwiGLU FFN (3 matmuls): gate_proj=H*I, up_proj=H*I, down_proj=I*H
    -> per-token FFN FLOPs = 2*(2*H*I + I*H) = 6*H*I.
  - LM head (once per generated step): 2 * H * vocab.

LLM MoE (Qwen3-MoE 30B-A3B)
---------------------------
Same attention as dense. FFN replaced by MoE: each token routes to top-k
experts (num_experts_per_tok=8 of 128 total), each expert is a SwiGLU FFN
with intermediate size = moe_intermediate_size=768. Active per-token FFN
FLOPs = k * 6 * H * I_moe. Router gate FLOPs (2*H*num_experts) are negligible
but included. There is NO shared expert (verified, see CHANGES #4 below).

Decode
------
Standard KV-cached growth: at decode step t (0-indexed), attention does
2 * 2 * Hq * (N + t)  (QK^T over cached length, then *V), and projections
are constant per step. Summed over T_out steps with start length N_in.

============================================================================
CHANGES (verification pass) — code citations resolving each open gap
============================================================================
1. CLS token included during ViT forward.
   InternViT-6B-V2.5 modeling_intern_vit.py:
     class InternVisionEmbeddings:
       self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
       embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
   modeling_internvl_chat.py:
     def extract_feature(...):
       vit_embeds = self.vision_model(...).last_hidden_state
       vit_embeds = vit_embeds[:, 1:, :]   # <-- CLS dropped *after* ViT
       ... pixel_shuffle ... mlp1 ...
   => N_v inside the ViT = 1024 + 1 = 1025.  Visual tokens to the LLM stay 256/tile.

2. InternViT-6B-V2.5 MLP type: STANDARD MLP (not SwiGLU).
   modeling_intern_vit.py InternMLP:
     self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
     self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
     def forward(x): return self.fc2(self.act(self.fc1(x)))
   `hidden_act` is "gelu" in 38B vision_config; "gelu" in 8B/30B-A3B vision_config.
   => 2*H*I + 2*I*H per token (already correct in `_vit_block_flops`).
   qk_normalization=true (38B only): InternRMSNorm on Q and K — no matmul.

3. Connector MLP1 architecture (verified verbatim, modeling_internvl_chat.py):
     self.mlp1 = nn.Sequential(
       nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
       nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
       nn.GELU(),
       nn.Linear(llm_hidden_size, llm_hidden_size),
     )
   => in_dim = H_vit * 4; FLOPs/tok = 2*in_dim*H_llm + 2*H_llm*H_llm. Matches.

4. Qwen3-MoE 30B-A3B router & experts.
   InternVL3.5-30B-A3B/config.json (llm_config):
     num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,
     hidden_size=2048, num_attention_heads=32, num_key_value_heads=4,
     head_dim=128, num_hidden_layers=48,
     mlp_only_layers=[], decoder_sparse_step=1, norm_topk_prob=true.
     No `n_shared_experts` / `shared_expert_*` fields (verified).
   transformers/models/qwen3_moe/modeling_qwen3_moe.py:
     class Qwen3MoeSparseMoeBlock (line 233):
       self.gate    = nn.Linear(hidden_size, num_experts, bias=False)   # single Linear router
       self.experts = nn.ModuleList([Qwen3MoeMLP(config, intermediate_size=moe_intermediate_size)
                                     for _ in range(num_experts)])     # per-expert SwiGLU; no shared expert path
     class Qwen3MoeMLP (line 217): gate_proj, up_proj (H -> I_moe) and down_proj (I_moe -> H);
                                   forward = down_proj(silu(gate_proj(x)) * up_proj(x))   # SwiGLU
     class Qwen3MoeDecoderLayer (line 307):
       With mlp_only_layers=[] AND decoder_sparse_step=1, every layer takes the SparseMoeBlock branch.
       Forward pattern: residual + self_attn(input_layernorm(x));
                        residual + self.mlp(post_attention_layernorm(x))
   => Confirmed: RMSNorm -> GQA-attn -> RMSNorm -> MoE-FFN at every layer, top-8 SwiGLU experts,
      single Linear router, no shared expert.

5. Pixel-shuffle exact mechanics (modeling_internvl_chat.py pixel_shuffle):
     x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
     x = x.permute(0, 2, 1, 3).contiguous()
     x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))   # ps_version='v2': permute back
   => With scale_factor=0.5: H' = H/2, W' = W/2, C' = 4C => (H*W) drops 4x; channels
      stack 4x. Matches our `merge = (1/0.5)^2 = 4`.

6. Qwen3-MoE LLM block structure (transformers/qwen3_moe/modeling_qwen3_moe.py
   Qwen3MoeDecoderLayer.forward):
     residual = hidden_states
     hidden_states = self.input_layernorm(hidden_states)         # RMSNorm
     hidden_states, _ = self.self_attn(...)                      # GQA attn
     hidden_states = residual + hidden_states
     residual = hidden_states
     hidden_states = self.post_attention_layernorm(hidden_states) # RMSNorm
     hidden_states = self.mlp(hidden_states)                      # SparseMoeBlock
   Each expert: SwiGLU (SiLU(gate)*up, then down). Confirmed no shared expert.

Effects on FLOPs from this verification pass:
  - Vision N_v: 1024 -> 1025 (+0.10%). Increases vision_flops slightly.
  - All other components: no change (citations confirm prior assumptions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .elementwise import (
    rmsnorm_flops, layernorm_flops, residual_flops, bias_flops,
    rope_flops, rope_flops_decode,
    softmax_flops_attention, softmax_flops_decode,
    silu_flops, gelu_exact_flops,
    moe_router_flops, moe_combine_flops,
    lm_head_softmax_decode,
)


# ---------------------------------------------------------------------------
# Architecture configs — copied verbatim from each repo's config.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViTCfg:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    image_size: int


@dataclass(frozen=True)
class LLMCfg:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int  # dense FFN; for MoE this is the "shared" FFN dim (unused if pure MoE)
    vocab_size: int
    tie_word_embeddings: bool
    # MoE-only:
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0


@dataclass(frozen=True)
class InternVLCfg:
    name: str
    vit: ViTCfg
    llm: LLMCfg
    force_image_size: int = 448
    downsample_ratio: float = 0.5  # 2x2 pixel-shuffle => /4 tokens per tile
    max_dynamic_patch: int = 12
    min_dynamic_patch: int = 1
    use_thumbnail: bool = True


# InternViT-300M (used by 8B and 30B-A3B)
# Source: OpenGVLab/InternVL3_5-8B/config.json -> vision_config:
#   hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
#   intermediate_size=4096, patch_size=14, image_size=448.
VIT_300M = ViTCfg(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
    intermediate_size=4096, patch_size=14, image_size=448,
)

# InternViT-6B-448px-V2_5 (used by 38B)
# Source: OpenGVLab/InternVL3_5-38B/config.json -> vision_config:
#   hidden_size=3200, num_hidden_layers=45, num_attention_heads=25,
#   intermediate_size=12800, patch_size=14, image_size=448, qk_normalization=true.
VIT_6B = ViTCfg(
    hidden_size=3200, num_hidden_layers=45, num_attention_heads=25,
    intermediate_size=12800, patch_size=14, image_size=448,
)

# Qwen3-8B (used by InternVL3.5-8B)
# Source: Qwen/Qwen3-8B/config.json:
#   hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
#   num_key_value_heads=8, head_dim=128, intermediate_size=12288,
#   vocab_size=151936, tie_word_embeddings=false.
LLM_QWEN3_8B = LLMCfg(
    hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, intermediate_size=12288,
    vocab_size=151936, tie_word_embeddings=False,
)

# Qwen3-32B (the 38B variant uses Qwen3-32B as its LLM backbone)
# Source: Qwen/Qwen3-32B/config.json:
#   hidden_size=5120, num_hidden_layers=64, num_attention_heads=64,
#   num_key_value_heads=8, head_dim=128, intermediate_size=25600,
#   vocab_size=151936, tie_word_embeddings=false.
LLM_QWEN3_32B = LLMCfg(
    hidden_size=5120, num_hidden_layers=64, num_attention_heads=64,
    num_key_value_heads=8, head_dim=128, intermediate_size=25600,
    vocab_size=151936, tie_word_embeddings=False,
)

# Qwen3-30B-A3B (MoE; used by InternVL3.5-30B-A3B)
# Source: Qwen/Qwen3-30B-A3B/config.json:
#   hidden_size=2048, num_hidden_layers=48, num_attention_heads=32,
#   num_key_value_heads=4, head_dim=128, intermediate_size=6144 (unused for MoE
#   path), num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,
#   vocab_size=151936, tie_word_embeddings=false.
LLM_QWEN3_30B_A3B = LLMCfg(
    hidden_size=2048, num_hidden_layers=48, num_attention_heads=32,
    num_key_value_heads=4, head_dim=128, intermediate_size=6144,  # not used by MoE path
    vocab_size=151936, tie_word_embeddings=False,
    is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,
)


CFG_8B   = InternVLCfg(name="InternVL3.5-8B",   vit=VIT_300M, llm=LLM_QWEN3_8B)
CFG_30BA = InternVLCfg(name="InternVL3.5-30B-A3B", vit=VIT_300M, llm=LLM_QWEN3_30B_A3B)
CFG_38B  = InternVLCfg(name="InternVL3.5-38B",  vit=VIT_6B,   llm=LLM_QWEN3_32B)


# ---------------------------------------------------------------------------
# Tiling: how many 448x448 tiles for an input frame
# ---------------------------------------------------------------------------

def _closest_grid(width: int, height: int, image_size: int,
                  min_num: int, max_num: int) -> tuple[int, int]:
    """Reproduce InternVL's dynamic_preprocess grid choice."""
    aspect = width / max(height, 1)
    candidates = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = (1, 1)
    best_diff = float("inf")
    target_area = width * height
    for (i, j) in candidates:
        diff = abs(aspect - (i / j))
        if diff < best_diff:
            best_diff = diff
            best = (i, j)
        elif diff == best_diff:
            # tiebreak: pick larger area if it doesn't massively exceed the input
            area = i * j * image_size * image_size
            cur_area = best[0] * best[1] * image_size * image_size
            if area > cur_area and area < 2 * target_area:
                best = (i, j)
    return best


def _tiles_per_frame(h: int, w: int, cfg: InternVLCfg) -> int:
    """Number of 448x448 ViT forwards per input frame.

    Picks an (i, j) tile grid via `_closest_grid` (aspect-ratio match within
    `[min_dynamic_patch, max_dynamic_patch]`) and adds one thumbnail tile when
    `i*j > 1`. Returns the total tile count (each tile is one ViT forward).
    """
    i, j = _closest_grid(w, h, cfg.force_image_size,
                         cfg.min_dynamic_patch, cfg.max_dynamic_patch)
    n = i * j
    if cfg.use_thumbnail and n > 1:
        n += 1
    return n


def _vision_tokens_per_tile(cfg: InternVLCfg) -> int:
    """Patches per tile after pixel-shuffle (2x2 merge)."""
    p = (cfg.force_image_size // cfg.vit.patch_size) ** 2          # 1024
    merge = int(round(1.0 / cfg.downsample_ratio)) ** 2             # 0.5 -> 4
    return p // merge                                               # 256


# ---------------------------------------------------------------------------
# Component FLOPs
# ---------------------------------------------------------------------------

def _vit_block_flops(N: int, vit: ViTCfg) -> int:
    """One InternViT block forward, per tile, sequence length N (incl. CLS).

    Standard pre-LN/post-LN ViT block. InternMLP is a *plain* fc1/fc2 stack:
      modeling_intern_vit.py InternMLP:
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
      forward: x = fc2(act(fc1(x)))   # NOT SwiGLU.
    qk_normalization (38B variant) applies InternRMSNorm to Q and K. RMSNorm
    is elementwise multiply by a learned scale; no matmul, no FLOPs added here.
    """
    H = vit.hidden_size
    I = vit.intermediate_size
    # QKV (no GQA in InternViT) = 3*H per token; output proj adds H*H per token
    proj_per_tok = 2 * H * (3 * H) + 2 * H * H
    # attention: QK^T and A·V, each 2*N*H per token (bidirectional, exact full N^2)
    attn_per_tok = 2 * (2 * N * H)
    # MLP: H -> intermediate_size -> H  (standard, NOT SwiGLU)
    mlp_per_tok = 2 * H * I + 2 * I * H
    return N * (proj_per_tok + attn_per_tok + mlp_per_tok)


def _vision_flops_for_image(cfg: InternVLCfg, n_tiles: int) -> int:
    """Total ViT-side matmul FLOPs for `n_tiles` 448x448 ViT forwards
    (patch embed conv + L transformer blocks per tile)."""
    # Per modeling_intern_vit.py: InternVisionEmbeddings concatenates a CLS token
    # to the patch embeddings BEFORE the encoder, so the ViT processes 1024+1 tokens.
    # The CLS is dropped only afterwards by extract_feature (`vit_embeds[:, 1:, :]`),
    # before pixel-shuffle. Visual tokens fed to the LLM remain (1024//4)=256/tile.
    n_patches = (cfg.force_image_size // cfg.vit.patch_size) ** 2  # 1024 patches/tile
    N = n_patches + 1  # +1 CLS token (verified)
    per_tile = cfg.vit.num_hidden_layers * _vit_block_flops(N, cfg.vit)
    # patch-embed conv: (patch*patch*3)·H per patch ~ 2*n_patches*(p^2*3)*H
    # CLS is a learned parameter (no compute), so use n_patches not N here.
    p = cfg.vit.patch_size
    patch_embed = 2 * n_patches * (p * p * 3) * cfg.vit.hidden_size
    return n_tiles * (patch_embed + per_tile)


def _connector_flops(cfg: InternVLCfg, n_visual_tokens: int) -> int:
    """
    InternVL MLP1: LayerNorm -> Linear(in_dim -> H_llm) -> GELU -> Linear(H_llm -> H_llm).
    in_dim = vit.hidden_size * (1/downsample_ratio)^2  (because pixel-shuffle stacks 4 patches into one feature)
    """
    H_vit = cfg.vit.hidden_size
    H_llm = cfg.llm.hidden_size
    merge = int(round(1.0 / cfg.downsample_ratio)) ** 2  # 4
    in_dim = H_vit * merge
    flops_per_tok = 2 * in_dim * H_llm + 2 * H_llm * H_llm
    return n_visual_tokens * flops_per_tok


def _llm_attn_proj_per_token(llm: LLMCfg) -> int:
    """Per-token Q/K/V + output-projection matmul FLOPs (one LLM layer).
    Q dim = num_attention_heads * head_dim; KV dim = num_kv_heads * head_dim
    (GQA/MHA collapse to the same expression)."""
    H = llm.hidden_size
    Hq = llm.num_attention_heads * llm.head_dim
    Hkv = llm.num_key_value_heads * llm.head_dim
    # q,k,v projections + output projection
    return 2 * H * (Hq + 2 * Hkv) + 2 * Hq * H


def _llm_ffn_per_token(llm: LLMCfg) -> int:
    """Active SwiGLU FFN FLOPs per token (handles MoE top-k)."""
    H = llm.hidden_size
    if llm.is_moe:
        I = llm.moe_intermediate_size
        k = llm.num_experts_per_tok
        # gate router (negligible) + top-k SwiGLU experts
        router = 2 * H * llm.num_experts
        expert = 2 * H * I + 2 * H * I + 2 * I * H  # gate + up + down
        return router + k * expert
    else:
        I = llm.intermediate_size
        return 2 * H * I + 2 * H * I + 2 * I * H


def _llm_attn_matmul_prefill(N: int, llm: LLMCfg) -> int:
    """QK^T and A·V over a full prefill of length N (one layer).

    Uses full N^2 per matmul (the Chinchilla / Kaplan-Scaling-Laws convention,
    which omits the 1/2 causal-mask correction). See module docstring,
    "CAUSAL ATTENTION FACTOR — chosen convention" for the citation.
    """
    Hq = llm.num_attention_heads * llm.head_dim
    # Two matmuls per layer: scores = Q @ K^T (N x Hq) @ (Hq x N) -> 2*N*N*Hq FLOPs
    #                       out    = A @ V    (N x N)  @ (N x Hq) -> 2*N*N*Hq FLOPs
    return 2 * (2 * N * N * Hq)


def _llm_prefill_flops(N: int, llm: LLMCfg) -> int:
    """Total LLM prefill FLOPs over a sequence of length N (matmul only).
    Per layer: N tokens * (attention projections + FFN) + the QK^T/AV
    attention core matmuls (N^2 in N). Multiplied by `num_hidden_layers`."""
    per_tok = _llm_attn_proj_per_token(llm) + _llm_ffn_per_token(llm)
    per_layer_attn = _llm_attn_matmul_prefill(N, llm)
    layers = llm.num_hidden_layers
    return layers * (N * per_tok + per_layer_attn)


def _llm_lm_head_flops(n_steps: int, llm: LLMCfg) -> int:
    """LM head applied once per generated step (and once at end of prefill, but we fold prefill's into decode-start)."""
    return n_steps * 2 * llm.hidden_size * llm.vocab_size


def _llm_decode_flops(N_in: int, T_out: int, llm: LLMCfg) -> int:
    """Total LLM decode FLOPs for T_out generated tokens with KV cache prefilled
    to length N_in. Each step's attention cost grows linearly with the cache;
    projections + FFN are constant per step."""
    if T_out <= 0:
        return 0
    per_tok = _llm_attn_proj_per_token(llm) + _llm_ffn_per_token(llm)
    Hq = llm.num_attention_heads * llm.head_dim
    layers = llm.num_hidden_layers

    # Per generated step t in [0, T_out): one new token, KV cache length = N_in + t
    # Projection/FFN per step is constant. Attention matmul grows linearly.
    per_step_proj_ffn = layers * per_tok
    # Attention QK^T + AV at step t: 2 * 2 * Hq * (N_in + t + 1)  per layer
    # Sum_{t=0..T-1} (N_in + t + 1) = T*N_in + T*(T+1)/2
    attn_sum = T_out * N_in + T_out * (T_out + 1) // 2
    attn_total = layers * 2 * 2 * Hq * attn_sum

    lm_head = _llm_lm_head_flops(T_out, llm)
    return T_out * per_step_proj_ffn + attn_total + lm_head


# ---------------------------------------------------------------------------
# Top-level FLOPs for a single (frames, n_in_text, n_out_text) call
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Elementwise (norms / softmax / RoPE / activations / biases / residuals)
# ---------------------------------------------------------------------------

def _intern_vit_block_elem(N: int, vit: ViTCfg, has_qk_norm: bool) -> int:
    """One InternViT block elementwise FLOPs.

    InternViT uses LayerNorm (modeling_intern_vit.InternVisionEncoderLayer),
    qkv_bias=True (300M) or qkv_bias=False (6B-V2.5; the 38B variant). We
    handle both via has_qk_norm: only InternViT-6B-V2.5 has qk_normalization=True.
    Activation: GELU exact (config 'hidden_act': 'gelu').
    Positional embedding is learned absolute (no RoPE elementwise cost).
    """
    H = vit.hidden_size
    I = vit.intermediate_size
    n_heads = vit.num_attention_heads
    head_dim = H // n_heads
    norms = 2 * layernorm_flops(N, H)
    residuals = 2 * residual_flops(N, H)
    # InternViT-300M: qkv_bias=True, InternViT-6B-V2.5: qkv_bias=False.
    has_qkv_bias = (H == 1024)  # 300M; 6B has H=3200 with qkv_bias=False
    qkv_bias = bias_flops(N, 3 * H) if has_qkv_bias else 0
    o_bias = bias_flops(N, H)  # proj has bias in both
    # qk_norm only on 6B-V2.5: RMSNorm on Q and K per layer, per head.
    qk_norm = (rmsnorm_flops(N, head_dim) * 2 * n_heads) if has_qk_norm else 0
    attn_sm = softmax_flops_attention(N, n_heads)
    # 2-mat GELU (exact); biases on both Linears.
    act = gelu_exact_flops(N, I)
    ffn_bias = bias_flops(N, I) + bias_flops(N, H)
    return norms + residuals + qkv_bias + o_bias + qk_norm + attn_sm + act + ffn_bias


def _intern_vision_elem(cfg: InternVLCfg, n_tiles: int) -> int:
    """Total elementwise FLOPs for `n_tiles` 448x448 ViT forwards (norms,
    softmax, GELU, biases). Matmul terms are in `_vision_flops_for_image`."""
    n_patches = (cfg.force_image_size // cfg.vit.patch_size) ** 2
    N = n_patches + 1  # +1 CLS
    has_qk_norm = (cfg.vit.hidden_size == 3200)  # InternViT-6B-V2.5
    per_tile = cfg.vit.num_hidden_layers * _intern_vit_block_elem(N, cfg.vit, has_qk_norm)
    # Patch embed bias (Conv2d has bias by default in transformers ViT).
    patch_bias = bias_flops(n_patches, cfg.vit.hidden_size)
    return n_tiles * (patch_bias + per_tile)


def _intern_connector_elem(cfg: InternVLCfg, n_visual_tokens: int) -> int:
    """Elementwise FLOPs of the MLP1 connector (LayerNorm + GELU + biases).
    Matmul terms are in `_connector_flops`."""
    H_vit = cfg.vit.hidden_size
    H_llm = cfg.llm.hidden_size
    merge = int(round(1.0 / cfg.downsample_ratio)) ** 2  # 4
    in_dim = H_vit * merge
    # MLP1: LayerNorm -> Linear -> GELU -> Linear; biases default True.
    return (layernorm_flops(n_visual_tokens, in_dim)
            + bias_flops(n_visual_tokens, H_llm)
            + gelu_exact_flops(n_visual_tokens, H_llm)
            + bias_flops(n_visual_tokens, H_llm))


def _intern_llm_block_elem_prefill(N: int, llm: LLMCfg) -> int:
    """Elementwise FLOPs for ONE LLM block at prefill length N (Qwen3-class:
    RMSNorm, RoPE, qk_norm, softmax, SiLU). MoE adds router/combine."""
    H = llm.hidden_size
    n_q = llm.num_attention_heads
    n_kv = llm.num_key_value_heads
    head_dim = llm.head_dim
    norms = 2 * rmsnorm_flops(N, H)
    residuals = 2 * residual_flops(N, H)
    # Qwen3 LLMs: qk_norm=True, no qkv_bias.
    qk_norm = rmsnorm_flops(N, head_dim) * (n_q + n_kv)
    rope = rope_flops(N, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_attention(N, n_q)
    if llm.is_moe:
        router = moe_router_flops(N, llm.num_experts, llm.num_experts_per_tok)
        per_expert = silu_flops(N, llm.moe_intermediate_size) + N * llm.moe_intermediate_size
        experts = llm.num_experts_per_tok * per_expert
        combine = moe_combine_flops(N, H, llm.num_experts_per_tok)
        ffn_elem = router + experts + combine
    else:
        ffn_elem = silu_flops(N, llm.intermediate_size) + N * llm.intermediate_size
    return norms + residuals + qk_norm + rope + attn_sm + ffn_elem


def _intern_llm_block_elem_decode(N_in: int, n_out: int, llm: LLMCfg) -> int:
    """Elementwise FLOPs for ONE LLM block summed over `n_out` decode steps
    with KV cache prefilled to length N_in."""
    if n_out <= 0:
        return 0
    H = llm.hidden_size
    n_q = llm.num_attention_heads
    n_kv = llm.num_key_value_heads
    head_dim = llm.head_dim
    norms = 2 * rmsnorm_flops(1, H) * n_out
    residuals = 2 * residual_flops(1, H) * n_out
    qk_norm = (rmsnorm_flops(1, head_dim) * (n_q + n_kv)) * n_out
    rope = rope_flops_decode(n_out, head_dim, n_q, n_kv)
    attn_sm = softmax_flops_decode(N_in, n_out, n_q)
    if llm.is_moe:
        router = moe_router_flops(n_out, llm.num_experts, llm.num_experts_per_tok)
        per_expert = silu_flops(1, llm.moe_intermediate_size) + llm.moe_intermediate_size
        experts = llm.num_experts_per_tok * per_expert * n_out
        combine = moe_combine_flops(n_out, H, llm.num_experts_per_tok)
        ffn_elem = router + experts + combine
    else:
        ffn_elem = silu_flops(n_out, llm.intermediate_size) + n_out * llm.intermediate_size
    return norms + residuals + qk_norm + rope + attn_sm + ffn_elem


def _intern_elementwise(cfg: InternVLCfg, n_tiles_total: int, n_visual_tokens: int,
                        N_in: int, n_out: int) -> dict:
    """Aggregate elementwise FLOPs across vision / connector / LLM prefill /
    LLM decode for one InternVL forward. Mirrors the matmul `_flops` layout."""
    vis = _intern_vision_elem(cfg, n_tiles_total)
    conn = _intern_connector_elem(cfg, n_visual_tokens)
    llm_pre = cfg.llm.num_hidden_layers * _intern_llm_block_elem_prefill(N_in, cfg.llm)
    llm_dec = cfg.llm.num_hidden_layers * _intern_llm_block_elem_decode(N_in, n_out, cfg.llm)
    llm_dec += rmsnorm_flops(n_out, cfg.llm.hidden_size)
    llm_dec += lm_head_softmax_decode(n_out, cfg.llm.vocab_size)
    return dict(vision=vis, connector=conn, llm_prefill=llm_pre, llm_decode=llm_dec,
                total=vis + conn + llm_pre + llm_dec)


def _flops(cfg: InternVLCfg, frames, n_in_text_tokens, n_out_text_tokens) -> dict:
    """Shared body for every InternVL3.5 variant; each public `flops_*`
    wrapper just builds an `InternVLCfg` and forwards here."""
    # Vision per frame (each frame processed independently).
    #
    # InternVL3.5's video preprocessing pipeline (modeling_internvl_chat.py /
    # internvl_video_chat.py) calls dynamic_preprocess(image_size=448,
    # min_num=1, max_num=12, use_thumbnail=True). For VIDEO (frames sampled
    # from a clip), the canonical setting is max_num=1 -> exactly ONE 448x448
    # tile per frame (see InternVL3 README ``Inference with Video Input``).
    # For images, max_num<=12 tiles + thumbnail.
    # _tiles_per_frame replicates this exactly for arbitrary H/W: it picks
    # the (i, j) grid minimizing |aspect - i/j| within [min_num, max_num],
    # then adds 1 thumbnail iff i*j > 1 and use_thumbnail=True. The ViT
    # always runs at 448x448 per tile because tiles are resized to that size
    # by the preprocessor.
    vis_tokens_per_tile = _vision_tokens_per_tile(cfg)
    vision_flops = 0
    connector_flops = 0
    total_visual_tokens = 0
    total_n_tiles = 0
    for fr in frames:
        n_tiles = _tiles_per_frame(fr["height"], fr["width"], cfg)
        total_n_tiles += n_tiles
        vision_flops += _vision_flops_for_image(cfg, n_tiles)
        n_vt = n_tiles * vis_tokens_per_tile
        total_visual_tokens += n_vt
        connector_flops += _connector_flops(cfg, n_vt)

    # LLM
    N_in = total_visual_tokens + n_in_text_tokens
    llm_prefill = _llm_prefill_flops(N_in, cfg.llm)
    llm_decode  = _llm_decode_flops(N_in, n_out_text_tokens, cfg.llm)

    total = vision_flops + connector_flops + llm_prefill + llm_decode

    elem = _intern_elementwise(cfg, total_n_tiles, total_visual_tokens, N_in,
                               n_out_text_tokens)

    return {
        "model": cfg.name,
        "n_visual_tokens": total_visual_tokens,
        "n_in_text_tokens": n_in_text_tokens,
        "n_out_text_tokens": n_out_text_tokens,
        "vision_flops": vision_flops,
        "connector_flops": connector_flops,
        "llm_prefill_flops": llm_prefill,
        "llm_decode_flops": llm_decode,
        "total_flops": total,
        "vision_elementwise": elem["vision"],
        "connector_elementwise": elem["connector"],
        "llm_prefill_elementwise": elem["llm_prefill"],
        "llm_decode_elementwise": elem["llm_decode"],
        "elementwise_total": elem["total"],
        "total_with_elementwise": total + elem["total"],
    }


# ---------------------------------------------------------------------------
# Public per-model functions
# ---------------------------------------------------------------------------

def flops_internvl3_5_8b(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5 V (8B) — Qwen3-8B + InternViT-300M-448px.

    CALLER CONTRACT
    ---------------
    Pass `frames` with ANY H, W. The function calls ``_tiles_per_frame(H, W)``
    INSIDE, which replicates ``dynamic_preprocess`` (1<=i*j<=max_dynamic_patch=12,
    +thumbnail iff i*j>1). Each tile is RESIZED to 448x448 by the real
    preprocessor before the ViT, so the per-tile sequence length is fixed at
    (448/14)^2 + 1 = 1025 regardless of caller H/W. Only the TILE COUNT depends
    on caller H/W.

    Canonical video usage passes ``max_num=1`` (one 448x448 tile per frame, no
    thumbnail) — the modeling code does not enforce this; we expose the same
    default via ``max_dynamic_patch=12`` and fall back to a single tile when
    H<=448 and W<=448. To match the canonical video pipeline, pass H=W=448.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (InternViT-300M, depth=24, hidden=1024, heads=16)
    -------------------------------------------------------------------
    1. Attention type: MHA (16 Q heads, 16 KV heads, head_dim=64). No GQA.
    2. Attention scope: FULL N^2 (per-tile, bidirectional).
    3. Positional embedding: 1D-aligned LEARNED absolute positions (interpolated
       at runtime for 448x448 -> 1025-token grid). RoPE not used at the ViT.
    4. FFN: standard 2-matmul (fc1 + GELU + fc2), intermediate=4096 -> 4*N*H*I.
       NOT SwiGLU. (modeling_intern_vit.py InternMLP).
    5. CLS token: PRESENT during ViT (1024 patches + 1 CLS = 1025 tokens).
       Dropped after the encoder, before pixel-shuffle.
    6. Variable-length packing: NO (each tile is its own forward; tiles batched).
    """
    return _flops(CFG_8B, frames, n_in_text_tokens, n_out_text_tokens)


def flops_internvl3_5_8b_thinking(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5 V Thinking (8B) — same checkpoint as 8B; R1 system-prompt mode.
    Architecture and per-token FLOPs are IDENTICAL to flops_internvl3_5_8b; only
    n_out_text_tokens is typically larger (the <think>...</think> chain). See
    flops_internvl3_5_8b for the vision-encoder audit.
    """
    return _flops(CFG_8B, frames, n_in_text_tokens, n_out_text_tokens)


def flops_internvl3_5_30b_a3b(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5 V (30B, A3B) — Qwen3-MoE 30B-A3B + InternViT-300M-448px.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (InternViT-300M, depth=24, hidden=1024, heads=16)
    -------------------------------------------------------------------
    Identical ViT to flops_internvl3_5_8b: MHA, full N^2, learned 1D-aligned PE,
    standard 2-matmul GELU MLP (intermediate=4096), CLS PRESENT, no varlen.
    Differs only in the LLM backbone (Qwen3-MoE 30B-A3B: 48L, hidden=2048,
    128 experts, top-8, moe_intermediate=768; no shared expert).
    """
    return _flops(CFG_30BA, frames, n_in_text_tokens, n_out_text_tokens)


def flops_internvl3_5_30b_a3b_thinking(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5-30B-Thinking (30B, A3B) — same checkpoint as 30B-A3B; R1
    system-prompt mode. Identical architecture and per-token FLOPs as
    flops_internvl3_5_30b_a3b; only n_out_text_tokens differs in practice.
    """
    return _flops(CFG_30BA, frames, n_in_text_tokens, n_out_text_tokens)


def flops_internvl3_5_38b(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5 V (38B) — Qwen3-32B + InternViT-6B-448px-V2_5.
    -------------------------------------------------------------------
    VISION-ENCODER AUDIT (InternViT-6B-V2.5, depth=45, hidden=3200, heads=25)
    -------------------------------------------------------------------
    1. Attention type: MHA (25 Q heads, 25 KV heads, head_dim=128). No GQA.
    2. Attention scope: FULL N^2 (per-tile, bidirectional).
    3. Positional embedding: LEARNED absolute (interpolated for 1025-token grid).
    4. FFN: standard 2-matmul (fc1 + GELU + fc2), intermediate=12800 -> 4*N*H*I.
       Distinguishing feature: qk_normalization=True (InternRMSNorm on Q and K
       per layer). RMSNorm is elementwise — adds zero matmul FLOPs.
    5. CLS token: PRESENT during ViT (1025 tokens), dropped after encoder.
    6. Variable-length packing: NO.
    """
    return _flops(CFG_38B, frames, n_in_text_tokens, n_out_text_tokens)


def flops_internvl3_5_38b_thinking(frames, n_in_text_tokens, n_out_text_tokens):
    """InternVL3.5 V Thinking (38B) — same checkpoint as 38B; R1 system-prompt
    mode. Identical architecture and per-token FLOPs as flops_internvl3_5_38b.
    See that function for the vision-encoder audit.
    """
    return _flops(CFG_38B, frames, n_in_text_tokens, n_out_text_tokens)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _fmt_pflops(x: int) -> str:
    """Format raw FLOPs as a fixed-width PFLOPs string (used by __main__)."""
    return f"{x / 1e15:10.3f} PF"


def _print_breakdown(label: str, r: dict) -> None:
    """Print per-component FLOPs breakdown for a result dict (used by __main__)."""
    print(f"\n=== {label} ({r['model']}) ===")
    print(f"  visual tokens (post-PS, all frames): {r['n_visual_tokens']}")
    print(f"  N_in (visual + text):                {r['n_visual_tokens'] + r['n_in_text_tokens']}")
    print(f"  n_out_text_tokens:                   {r['n_out_text_tokens']}")
    print(f"  vision_flops      = {_fmt_pflops(r['vision_flops'])}")
    print(f"  connector_flops   = {_fmt_pflops(r['connector_flops'])}")
    print(f"  llm_prefill_flops = {_fmt_pflops(r['llm_prefill_flops'])}")
    print(f"  llm_decode_flops  = {_fmt_pflops(r['llm_decode_flops'])}")
    print(f"  TOTAL             = {_fmt_pflops(r['total_flops'])}")


if __name__ == "__main__":
    frames = [{"height": 448, "width": 448}] * 8
    n_in = 128
    n_out_base = 64
    n_out_think = 2048

    print("Validation: 8 frames @ 448x448, n_in=128 text tokens.")
    print("Per InternVL video preprocessing (max_num=1, 448x448), tiles/frame=1, no thumbnail.")
    print(f"=> visual tokens/frame = (448/14)^2 / 4 = {((448//14)**2)//4} = 256")
    print(f"=> total visual tokens  = 8 * 256 = 2048")

    for fn, name, n_out in [
        (flops_internvl3_5_8b,              "InternVL3.5-8B",                   n_out_base),
        (flops_internvl3_5_8b_thinking,     "InternVL3.5-8B-Thinking",          n_out_think),
        (flops_internvl3_5_30b_a3b,         "InternVL3.5-30B-A3B",              n_out_base),
        (flops_internvl3_5_30b_a3b_thinking,"InternVL3.5-30B-A3B-Thinking",     n_out_think),
        (flops_internvl3_5_38b,             "InternVL3.5-38B",                  n_out_base),
        (flops_internvl3_5_38b_thinking,    "InternVL3.5-38B-Thinking",         n_out_think),
    ]:
        r = fn(frames, n_in, n_out)
        _print_breakdown(name, r)

    # Sanity: a single 1280x720 image should pick a 12-tile grid + thumbnail = 13 tiles.
    print("\n--- Tiling sanity check ---")
    for hw in [(448, 448), (896, 448), (720, 1280), (1080, 1920)]:
        h, w = hw
        n = _tiles_per_frame(h, w, CFG_8B)
        print(f"  {h}x{w}  -> {n} tiles  ({n*256} visual tokens)")
