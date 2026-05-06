from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backend_base import MemoryBackend, SequenceInputs
from .external.ttt import TTTConfig, TTTLinear, TTTMLP


class TTTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mini_batch_size: int,
        mlp_ratio: int,
        attn_dropout: float,
        resid_dropout: float,
        use_gate: bool,
        layer_idx: int,
        variant: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        if variant not in {"linear", "mlp", "linear_fast", "mlp_fast"}:
            raise ValueError("ttt_variant must be one of 'linear', 'mlp', 'linear_fast', or 'mlp_fast'")
        self.variant = variant
        self.fast_variant = variant.endswith("_fast")
        if self.fast_variant:
            # Import the fast adapter lazily so regular TTT variants do not
            # depend on the optional ThunderKittens / Mamba CUDA stack.
            from .ttt_fast_adapter import FastTTTLayerWrapper

            self.ttt = FastTTTLayerWrapper(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mini_batch_size=mini_batch_size,
                mlp_ratio=mlp_ratio,
                layer_idx=layer_idx,
                variant=variant,
            )
        else:
            base_lr = 1.0 if variant == "linear" else 0.1
            config = TTTConfig(
                vocab_size=1,
                hidden_size=embed_dim,
                intermediate_size=embed_dim * 4,
                num_hidden_layers=1,
                num_attention_heads=num_heads,
                mini_batch_size=mini_batch_size,
                pad_token_id=0,
                bos_token_id=0,
                eos_token_id=0,
                tie_word_embeddings=False,
                use_gate=use_gate,
                share_qk=False,
                ttt_base_lr=base_lr,
                ttt_layer_type=variant,
            )
            ttt_cls = TTTLinear if variant == "linear" else TTTMLP
            self.ttt = ttt_cls(config, layer_idx=layer_idx)
        self.norm3 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(resid_dropout)
        self.window_chunk_size = max(1, int(mini_batch_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        window_size: int,
        collect_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len = hidden_states.size(1)
        residual = hidden_states
        x = self.norm1(hidden_states)
        attn_out = self._apply_attention_window(x, padding_mask, window_size)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        if self.fast_variant and self.training:
            raise RuntimeError(
                "Fast TTT variants are inference-only; switch to eval mode or use the PyTorch variant."
            )
        ttt_result = self.ttt(x, attention_mask=None, position_ids=position_ids, cache_params=None)
        if isinstance(ttt_result, tuple):
            x, params = ttt_result
        else:
            x = ttt_result
            params = None
        x = residual + self.dropout(x)

        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        return residual + self.dropout(x), None

    def _apply_attention_window(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        window_size: int,
    ) -> torch.Tensor:
        if window_size <= 0:
            seq_len = hidden_states.size(1)
            causal = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=hidden_states.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            attn_out, _ = self.attn(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=causal,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            return attn_out

        batch, seq_len, embed_dim = hidden_states.shape
        chunk = min(self.window_chunk_size, seq_len)
        outputs = hidden_states.new_zeros(batch, seq_len, embed_dim)
        positions = torch.arange(seq_len, device=hidden_states.device)
        for start in range(0, seq_len, chunk):
            end = min(seq_len, start + chunk)
            query = hidden_states[:, start:end, :]
            key_start = max(0, start - window_size)
            key_end = end
            key = hidden_states[:, key_start:key_end, :]
            key_mask = None if padding_mask is None else padding_mask[:, key_start:key_end]
            attn_mask = self._chunk_mask(
                positions[start:end],
                positions[key_start:key_end],
                window_size,
            )
            chunk_out, _ = self.attn(
                query,
                key,
                key,
                attn_mask=attn_mask,
                key_padding_mask=key_mask,
                need_weights=False,
            )
            outputs[:, start:end, :] = chunk_out
        return outputs

    @staticmethod
    def _chunk_mask(
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        window_size: int,
    ) -> Optional[torch.Tensor]:
        if query_positions.numel() == 0 or key_positions.numel() == 0:
            return None
        dist = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        if window_size <= 0:
            allowed = dist >= 0
        else:
            allowed = (dist >= 0) & (dist <= window_size)
        if torch.all(allowed):
            return None
        return ~allowed


class TestTimeTrainingBackend(MemoryBackend):
    """Multi-layer backend combining windowed attention, TTT adaptation, and MLP blocks."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mini_batch_size: int = 32,
        window_size: int = 64,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        use_gate: bool = False,
        ttt_variant: str = "linear",
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for the TTT blocks")
        super().__init__(
            embed_dim,
            projects_to_decoder_dim=True,
            requires_token_embeddings=True,
        )
        self.window_size = window_size
        allowed_variants = {"linear", "mlp", "linear_fast", "mlp_fast"}
        if ttt_variant not in allowed_variants:
            raise ValueError(
                "TestTimeTrainingBackend supports variants {linear, mlp, linear_fast, mlp_fast}"
            )
        self.inference_only = ttt_variant.endswith("_fast")
        self.blocks = nn.ModuleList(
            [
                TTTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mini_batch_size=mini_batch_size,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    use_gate=use_gate,
                    layer_idx=i,
                    variant=ttt_variant,
                )
                for i in range(num_layers)
            ]
        )
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del label_mask
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None:
            raise ValueError("TestTimeTrainingBackend requires token embeddings for every token.")
        padding_mask = self._resolve_padding_mask(sequence)
        hidden = self._mask_embeddings(token_embeddings, padding_mask)
        for block in self.blocks:
            hidden, _ = block(
                hidden,
                padding_mask=padding_mask,
                window_size=self.window_size,
                collect_state=False,
            )
        hidden = self._project_hidden(hidden)
        return self._mask_embeddings(hidden, padding_mask)
