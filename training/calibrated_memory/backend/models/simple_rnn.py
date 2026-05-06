from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .backend_base import MemoryBackend, SequenceInputs
from .modules import RMSNorm


class SimpleRNNEncoder(MemoryBackend):
    """GRU encoder that emits decoder-aligned token representations."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        hidden_dim = hidden_dim or embed_dim
        super().__init__(
            hidden_dim,
            projects_to_decoder_dim=False,
            requires_token_embeddings=True,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, int(num_layers))
        self.layers = nn.ModuleList()
        self.input_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        input_size = embed_dim
        for _ in range(self.num_layers):
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                dropout=0.0,
                batch_first=True,
            )
            self.layers.append(gru)
            self.input_norms.append(RMSNorm(input_size))
            if input_size == hidden_dim:
                self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(nn.Linear(input_size, hidden_dim))
            input_size = hidden_dim
        self.layer_dropout = nn.Dropout(dropout if self.num_layers > 1 else 0.0)
        self.supports_direct_logits = True

    def encode_sequence(
        self,
        sequence: SequenceInputs,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = sequence.token_embeddings
        if token_embeddings is None or token_embeddings.size(1) == 0:
            raise ValueError("SimpleRNNEncoder requires token embeddings for every input sequence.")
        padding_mask = self._resolve_padding_mask(sequence)
        masked_inputs = self._mask_embeddings(token_embeddings, padding_mask)
        lengths = torch.sum(~padding_mask, dim=1).clamp(min=1)
        outputs, _ = self._run_stacked_gru(masked_inputs, lengths=lengths)
        outputs = self._project_hidden(outputs)
        return self._mask_embeddings(outputs, padding_mask)

    def _run_stacked_gru(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None):
        outputs = inputs
        hidden_states: list[torch.Tensor] = []
        if lengths is not None:
            residual_input = outputs
            total_len = inputs.size(1)
            for layer_idx, (layer, norm_in, proj) in enumerate(
                zip(self.layers, self.input_norms, self.residual_projs)
            ):
                normed = norm_in(residual_input).contiguous()
                packed = pack_padded_sequence(
                    normed,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed, hidden = layer(packed)
                hidden_states.append(hidden.squeeze(0))
                outputs, _ = pad_packed_sequence(packed, batch_first=True, total_length=total_len)
                residual = residual_input
                if residual.size(-1) != outputs.size(-1):
                    residual = proj(residual)
                outputs = outputs + residual
                if layer_idx < self.num_layers - 1:
                    outputs = self.layer_dropout(outputs)
                residual_input = outputs
            hidden_stack = torch.stack(hidden_states, dim=0)
            return outputs, hidden_stack

        outputs = inputs
        for idx, (layer, norm_in, proj) in enumerate(zip(self.layers, self.input_norms, self.residual_projs)):
            residual = outputs
            normed = norm_in(residual).contiguous()
            outputs, hidden = layer(normed)
            hidden_states.append(hidden.squeeze(0))
            if residual.size(-1) != outputs.size(-1):
                residual = proj(residual)
            outputs = outputs + residual
            if idx < self.num_layers - 1:
                outputs = self.layer_dropout(outputs)
        hidden_stack = torch.stack(hidden_states, dim=0)
        return outputs, hidden_stack
