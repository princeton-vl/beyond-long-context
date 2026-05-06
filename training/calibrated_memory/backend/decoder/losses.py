"""Loss utilities for MemoryBankDecoder."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from calibrated_memory.data.sequences.common import IGNORE_INDEX


class DeepGamblerLoss(nn.Module):
    """Deep Gambler's loss with optional adaptive wager scheduling."""

    def __init__(
        self,
        *,
        option_tokens: List[int],
        reserve_token: int,
        base_o: float,
        mode: str,
        epsilon: float,
        activation_acc: float,
    ) -> None:
        super().__init__()
        if not option_tokens:
            raise ValueError("DeepGamblerLoss requires at least one option token.")
        normalized_mode = mode.lower()
        if normalized_mode not in {"fixed", "adaptive"}:
            raise ValueError(f"Unknown Deep Gambler mode '{mode}'")
        self.num_options = len(option_tokens)
        class_tokens = list(sorted(set(option_tokens)))
        if len(class_tokens) != len(option_tokens):
            raise ValueError("option_tokens must be unique")
        class_tokens.append(int(reserve_token))
        self.register_buffer("class_tokens", torch.tensor(class_tokens, dtype=torch.long), persistent=False)
        self.reserve_index = len(class_tokens) - 1
        self.base_o = float(base_o)
        self.mode = normalized_mode
        if not 0.0 <= activation_acc <= 1.0:
            raise ValueError("activation_acc must be within [0, 1]")
        self.epsilon = float(epsilon)
        self.activation_acc = float(activation_acc)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.dim() != labels.dim() + 1:
            raise ValueError("Logits must have one extra dimension compared to labels")
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        device = flat_labels.device
        mask = flat_labels != IGNORE_INDEX
        if not torch.any(mask):
            zero = flat_logits.new_zeros(())
            return zero, {
                "current_o": zero.new_tensor(self.base_o),
                "supervised": zero.clone(),
                "uncertain": zero.clone(),
            }

        columns = torch.full(flat_labels.shape, -1, dtype=torch.long, device=device)
        for idx, token in enumerate(self.class_tokens):
            columns[flat_labels == token] = idx
        valid = mask & (columns >= 0)
        if not torch.any(valid):
            raise ValueError("No labels matched DeepGamblerLoss option tokens.")
        full_logits = flat_logits[valid]
        full_labels = flat_labels[valid]
        selected_logits = full_logits.index_select(-1, self.class_tokens)
        probs = torch.softmax(selected_logits, dim=-1)
        reserve_probs = probs[:, -1]
        targets = columns[valid]
        uncertain_mask = targets == self.reserve_index
        certain_mask = ~uncertain_mask
        eps = selected_logits.new_tensor(self.epsilon)
        losses = reserve_probs.new_zeros(reserve_probs.size())
        stats: Dict[str, torch.Tensor]

        current_o = self.base_o
        if uncertain_mask.any():
            losses[uncertain_mask] = -torch.log(reserve_probs[uncertain_mask].clamp_min(eps))
        if certain_mask.any():
            wager, certain_losses = self._resolve_losses(
                full_logits[certain_mask],
                full_labels[certain_mask],
                selected_logits[certain_mask],
                targets[certain_mask],
                probs[certain_mask],
                reserve_probs[certain_mask],
            )
            losses[certain_mask] = certain_losses
            current_o = wager
        stats = {
            "current_o": selected_logits.new_tensor(current_o),
            "supervised": selected_logits.new_tensor(float(valid.sum().item())),
            "uncertain": selected_logits.new_tensor(float(uncertain_mask.sum().item())),
        }
        return losses.mean(), stats

    def _resolve_losses(
        self,
        full_logits: torch.Tensor,
        full_labels: torch.Tensor,
        subset_logits: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor,
        reserve_probs: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        if self.mode == "fixed" or targets.numel() == 0:
            wager = self.base_o
            losses = self._gambler_losses(wager, probs, reserve_probs, targets)
            return wager, losses
        class_logits = subset_logits[:, :-1]
        preds = class_logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean().item()
        if accuracy < self.activation_acc:
            ce_loss = F.cross_entropy(
                full_logits,
                full_labels,
                reduction="none",
            )
            return self.base_o, ce_loss
        m = float(self.num_options)
        term = (1.0 - accuracy) * (m / (m + 1.0))
        wager = (m - 1.0) * term + 1.0
        losses = self._gambler_losses(wager, probs, reserve_probs, targets)
        return wager, losses

    def _gambler_losses(
        self,
        wager: float,
        probs: torch.Tensor,
        reserve_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        values = wager * p_true + reserve_probs
        eps = probs.new_tensor(self.epsilon)
        return -torch.log(values.clamp_min(eps))
