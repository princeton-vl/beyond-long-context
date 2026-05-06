from __future__ import annotations

import torch
import torch.nn as nn

from .backend_base import MemoryBackend, SequenceInputs
from .modules import AttentionCore, PositionalMode, RMSNorm


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        *,
        positional_mode: PositionalMode,
        rotary_base: float,
        pope_theta_base: float,
        pope_bias_init: str,
        use_flash_attention: bool,
        use_qk_norm: bool,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = AttentionCore(
            dim,
            num_heads,
            attn_dropout=dropout,
            positional_mode=positional_mode,
            rotary_base=rotary_base,
            pope_theta_base=pope_theta_base,
            pope_bias_init=pope_bias_init,
            use_flash_attention=use_flash_attention,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        hidden_dim = mlp_ratio * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden)
        attn_out = self.attn(
            attn_input,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True,
            query_positions=positions,
            key_positions=positions,
        )
        hidden = hidden + self.dropout1(attn_out)
        mlp_out = self.mlp(self.norm2(hidden))
        hidden = hidden + self.dropout2(mlp_out)
        return hidden


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        *,
        positional_mode: PositionalMode,
        rotary_base: float,
        pope_theta_base: float,
        pope_bias_init: str,
        use_flash_attention: bool,
        use_qk_norm: bool,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = AttentionCore(
            dim,
            num_heads,
            attn_dropout=dropout,
            positional_mode=positional_mode,
            rotary_base=rotary_base,
            pope_theta_base=pope_theta_base,
            pope_bias_init=pope_bias_init,
            use_flash_attention=use_flash_attention,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.cross_attn = AttentionCore(
            dim,
            num_heads,
            attn_dropout=dropout,
            positional_mode=positional_mode,
            rotary_base=rotary_base,
            pope_theta_base=pope_theta_base,
            pope_bias_init=pope_bias_init,
            use_flash_attention=use_flash_attention,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        hidden_dim = mlp_ratio * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        *,
        query_padding: torch.Tensor | None,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        q_norm = self.norm1(queries)
        self_out = self.self_attn(
            q_norm,
            attn_mask=attn_mask,
            key_padding_mask=query_padding,
            is_causal=True,
            query_positions=query_positions,
            key_positions=query_positions,
        )
        hidden = queries + self.dropout(self_out)
        cross_norm = self.norm2(hidden)
        cross_out = self.cross_attn(
            cross_norm,
            key=memory,
            value=memory,
            key_padding_mask=memory_padding,
            is_causal=False,
        )
        hidden = hidden + self.dropout(cross_out)
        mlp_out = self.mlp(self.norm3(hidden))
        hidden = hidden + self.dropout(mlp_out)
        return hidden


class TransformerPPBackend(MemoryBackend):
    """Transformer++ backend supporting both decoder and direct modes."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        positional_mode: str = "rope",
        rotary_base: float = 10000.0,
        pope_theta_base: float = 10000.0,
        pope_bias_init: str = "zero",
        use_flash_attention: bool = True,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        super().__init__(
            embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        try:
            mode = PositionalMode(positional_mode.lower())
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                "positional_mode must be one of 'none', 'rope', or 'pope'"
            ) from exc
        block_kwargs = {
            "dim": embed_dim,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "positional_mode": mode,
            "rotary_base": rotary_base,
            "pope_theta_base": pope_theta_base,
            "pope_bias_init": pope_bias_init,
            "use_flash_attention": use_flash_attention,
            "use_qk_norm": use_qk_norm,
            "qk_norm_eps": qk_norm_eps,
        }
        self.memory_blocks = nn.ModuleList(
            [_TransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [_DecoderBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self._positional_mode = mode
        self._requires_causal_mask = mode is PositionalMode.POPE
        self.supports_direct_logits = True
        self._cached_causal_mask: torch.Tensor | None = None
        self._position_cache: dict[int, torch.Tensor] = {}

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("TransformerPPBackend requires decoder token embeddings.")
        padding_mask = self._resolve_padding_mask(sequence)
        encoded = self._run_memory_blocks(token_embeddings, padding_mask)
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return encoded

    def _run_memory_blocks(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.size(1)
        positions = self._position_indices(seq_len, hidden.device)
        causal_mask = (
            self._build_causal_mask(seq_len, hidden.device)
            if self._requires_causal_mask
            else None
        )
        for block in self.memory_blocks:
            hidden = block(
                hidden,
                key_padding_mask=padding_mask,
                attn_mask=causal_mask,
                positions=positions,
            )
        return hidden

    def _build_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self._cached_causal_mask is None or self._cached_causal_mask.size(0) < length:
            full = torch.ones(length, length, dtype=torch.bool)
            self._cached_causal_mask = torch.tril(full)
        return self._cached_causal_mask[:length, :length].to(device=device)

    def _position_indices(
        self, length: int, device: torch.device
    ) -> torch.Tensor:
        cached = self._position_cache.get(length)
        if cached is None or cached.device != torch.device("cpu"):
            cached = torch.arange(length, dtype=torch.long)
            self._position_cache[length] = cached
        return cached.to(device=device)
