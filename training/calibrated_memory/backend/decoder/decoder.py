from __future__ import annotations

from collections import Counter, OrderedDict
import math
from typing import Any, Iterable, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer

from calibrated_memory.data.sequences.common import (
    CANDIDATE_SEPARATOR,
    IGNORE_INDEX,
    LABEL_SEPARATOR,
    LABEL_TOKENS,
    NO_TOKEN,
    QUERY_END_SEPARATOR,
    STREAM_QUERY_SEPARATOR,
    TOKEN_OFFSET,
    UNCERTAIN_TOKEN,
    YES_TOKEN,
)
from calibrated_memory.backend.models.identity import IdentityBackend
from calibrated_memory.backend.models.backend_base import MemoryBackend, SequenceInputs
from calibrated_memory.metrics.video import VideoBucketTracker, sanitize_boundaries
from calibrated_memory.backend.decoder.modules.rotary import RotaryMultiheadAttention
from calibrated_memory.backend.decoder.losses import DeepGamblerLoss
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: int,
        attn_dropout: float,
        resid_dropout: float,
        rotary_base: float,
        memory_backend: MemoryBackend | None = None,
        task: str = "membership",
        log_sample_queries: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RotaryMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            rotary_base=rotary_base,
        )
        self.dropout1 = nn.Dropout(resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = mlp_ratio * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout2 = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        normed = self.ln1(x)
        attn_out = self.attn(
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + self.dropout1(attn_out)
        residual = x
        normed = self.ln2(x)
        x = residual + self.dropout2(self.mlp(normed))
        return x


_VALID_LR_SCHEDULERS = {"cosine_restart", "cosine_epoch", "constant"}

_DYNAMICS_PARAM_NAMES = {"A_log", "D"}
_DYNAMICS_PARAM_KEYWORDS = (
    "dt_",
    "_dt",
    "time_",
    "_time",
    "decay",
    "temperature",
    "logit",
)
_NORM_CLASS_NAMES = {
    "LayerNorm",
    "RMSNorm",
    "FusedRMSNorm",
    "FusedLayerNorm",
    "GroupNorm",
}


class MemoryBankDecoder(pl.LightningModule):
    """Single decoder that attends over backend-provided memory banks and queries."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_seq_len: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        lr: float = 3e-4,
        lr_warmup_epochs: int = 0,
        lr_scheduler_mode: str = "cosine_restart",
        attn_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        rotary_base: float = 10000.0,
        weight_decay: float = 0.1,
        max_epochs: int = 50,
        memory_backend: MemoryBackend | None = None,
        task: str = "membership",
        log_sample_queries: int = 0,
        execution_mode: str = "direct",
        loss_type: str = "cross_entropy",
        deep_gambler_mode: str = "fixed",
        deep_gambler_o: float = 1.5,
        deep_gambler_epsilon: float = 1e-12,
        deep_gambler_activation_acc: float = 0.33,
        grad_component_specs: Sequence[tuple[str, str]] | None = None,
        feature_input_dim: int | None = None,
        warmup_first_epoch_fraction: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["memory_backend"])
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.lr = lr
        self.lr_warmup_epochs = max(0, int(lr_warmup_epochs))
        if lr_scheduler_mode not in _VALID_LR_SCHEDULERS:
            raise ValueError(
                f"Unsupported lr_scheduler_mode={lr_scheduler_mode!r};"
                f" expected one of {sorted(_VALID_LR_SCHEDULERS)}."
            )
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lr_scheduler_mode = lr_scheduler_mode
        self.task = task
        if task not in {"membership", "continuation"}:
            raise ValueError(f"Unsupported decoder task '{task}'")
        self._label_tokens = list(LABEL_TOKENS)
        self._label_to_index = {token: idx for idx, token in enumerate(self._label_tokens)}
        self._num_label_classes = len(self._label_tokens)
        self._yes_index = self._label_to_index[YES_TOKEN]
        self._no_index = self._label_to_index[NO_TOKEN]
        self._uncertain_index = self._label_to_index.get(UNCERTAIN_TOKEN)
        self.sample_log_limit = log_sample_queries
        self._sample_logs_written = 0
        self.dataset_metadata: dict[str, Any] = {}
        self._entropy_boundaries: list[float] | None = None
        self._length_boundaries: list[float] | None = None
        self._bucket_entropy_enabled = True
        self._video_tracker: VideoBucketTracker | None = None
        self._synthetic_val_enabled = False
        self._synthetic_val_has_primary = False
        self._query_projections = nn.ModuleDict()
        self._last_batch_metadata = None
        self._last_batch_stream_lengths = None
        self._debug_stage_stats = {
            "train": Counter(),
            "val": Counter(),
            "synthetic_val": Counter(),
        }
        if feature_input_dim is not None:
            if feature_input_dim <= 0:
                raise ValueError("feature_input_dim must be positive when provided")
            self._feature_input_dim = int(feature_input_dim)
            self._feature_projector = nn.Linear(self._feature_input_dim, d_model)
        else:
            self._feature_input_dim = None
            self._feature_projector = None
        grad_specs: list[tuple[str, str]] = []
        if grad_component_specs:
            for label, prefix in grad_component_specs:
                label = label.strip()
                prefix = prefix.strip()
                if label and prefix:
                    grad_specs.append((label, prefix))
        self._grad_component_specs = grad_specs
        self._last_pre_clip_norm: torch.Tensor | None = None
        normalized_mode = execution_mode.lower()
        if normalized_mode != "direct":
            raise ValueError("Only direct execution mode is supported.")
        self.execution_mode = "direct"

        if warmup_first_epoch_fraction is not None:
            if not (0.0 < warmup_first_epoch_fraction < 1.0):
                raise ValueError(
                    "warmup_first_epoch_fraction must be between 0 and 1 when provided"
                )
            self._fractional_warmup_fraction = float(warmup_first_epoch_fraction)
        else:
            self._fractional_warmup_fraction = None
        self._fractional_warmup_target_steps: int | None = None
        self._fractional_warmup_completed_steps = 0
        self._fractional_warmup_optimizer: Optimizer | None = None
        self._fractional_warmup_base_lrs: list[float] | None = None
        self._fractional_warmup_active = False
        self._fractional_warmup_checkpoint_state: dict[str, Any] | None = None

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.blocks = nn.ModuleList()
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.label_head = nn.Linear(d_model, self._num_label_classes)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )
        normalized_loss = str(loss_type).lower()
        self.loss_type = normalized_loss
        self._deep_gambler_loss: DeepGamblerLoss | None = None
        if normalized_loss == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        elif normalized_loss == "deep_gambler":
            if self._uncertain_index is None:
                raise ValueError("Deep Gambler loss requires an explicit uncertain class.")
            gambler_mode = str(deep_gambler_mode).lower()
            self._deep_gambler_loss = DeepGamblerLoss(
                option_tokens=[self._yes_index, self._no_index],
                reserve_token=self._uncertain_index,
                base_o=float(deep_gambler_o),
                mode=gambler_mode,
                epsilon=float(deep_gambler_epsilon),
                activation_acc=float(deep_gambler_activation_acc),
            )
            self.criterion = None
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'")

        embed_dim = self.token_embedding.embedding_dim
        backend = memory_backend if memory_backend is not None else IdentityBackend(embed_dim)
        self.memory_backend = backend
        if getattr(backend, "supports_direct_logits", False) and hasattr(backend, "set_memory_mode_enabled"):
            backend.set_memory_mode_enabled(False)
        if isinstance(backend, MemoryBackend):
            backend.register_decoder_dim(embed_dim)
        self._backend_dim = int(getattr(backend, "output_dim"))
        backend_projects = bool(getattr(backend, "projects_to_decoder_dim", False))
        if backend_projects and self._backend_dim != embed_dim:
            raise ValueError(
                "Backends that already live in decoder space must match the decoder embedding dim."
            )
        self._backend_projects_to_decoder_dim = backend_projects
        self._backend_requires_stream_embeds = bool(
            getattr(backend, "requires_token_embeddings", True)
        )
        if self._backend_projects_to_decoder_dim:
            self.representation_projection = nn.Identity()
        else:
            self.representation_projection = nn.Linear(self._backend_dim, embed_dim)
        self.representation_dropout = nn.Dropout(embed_dropout)

    def on_before_optimizer_step(
        self,
        optimizer: Optimizer,
        optimizer_idx: int | None = None,
    ) -> None:  # type: ignore[override]
        if optimizer_idx is None:
            super().on_before_optimizer_step(optimizer)  # type: ignore[misc]
        else:
            super().on_before_optimizer_step(optimizer, optimizer_idx)
        self._log_gradient_norms()

    def _log_gradient_norms(self) -> None:
        total_norm, component_norms = self._compute_gradient_norms()
        if total_norm is None:
            return
        self.log(
            "gradients/global_norm",
            total_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        for label, value in component_norms.items():
            self.log(
                f"gradients/{label}",
                value,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None,
        gradient_clip_algorithm: str | None,
    ) -> None:
        if gradient_clip_val is None:
            return
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(gradient_clip_val), norm_type=2.0)
        post_norm, _ = self._compute_gradient_norms()
        if post_norm is None:
            return
        self.log(
            "gradients/global_norm_post_clip",
            post_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def _compute_gradient_norms(self) -> tuple[torch.Tensor | None, OrderedDict[str, torch.Tensor]]:
        component_accumulators: OrderedDict[str, torch.Tensor | None] = OrderedDict(
            (label, None) for label, _ in self._grad_component_specs
        )
        param_squares: dict[int, torch.Tensor] = {}
        accounted_components: set[tuple[int, str]] = set()
        total_sum: torch.Tensor | None = None
        for name, param in self.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            param_id = id(param)
            squared = param_squares.get(param_id)
            if squared is None:
                squared = self._squared_l2_norm(grad)
                param_squares[param_id] = squared
                total_sum = squared if total_sum is None else total_sum + squared
            for label, prefix in self._grad_component_specs:
                if not name.startswith(prefix):
                    continue
                key = (param_id, label)
                if key in accounted_components:
                    continue
                accounted_components.add(key)
                existing = component_accumulators[label]
                component_accumulators[label] = squared if existing is None else existing + squared
        if total_sum is None:
            return None, OrderedDict()
        normalized_components: OrderedDict[str, torch.Tensor] = OrderedDict()
        for label, value in component_accumulators.items():
            if value is None:
                continue
            normalized_components[label] = torch.sqrt(value)
        total_norm = torch.sqrt(total_sum)
        return total_norm, normalized_components

    @staticmethod
    def _squared_l2_norm(grad: torch.Tensor) -> torch.Tensor:
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad.values()
            return values.float().pow(2).sum()
        return grad.detach().float().pow(2).sum()

    def _finalize_step(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        gambler_stats: dict[str, torch.Tensor] | None = None
        if self._deep_gambler_loss is None:
            if self.criterion is None:
                raise RuntimeError("No loss function configured for MemoryBankDecoder.")
            loss = self.criterion(logits.view(-1, self._num_label_classes), labels.view(-1))
        else:
            loss, gambler_stats = self._deep_gambler_loss(logits, labels)
        metrics = self._compute_metrics(logits, labels)
        is_train = stage == "train"
        prog_bar_enabled = stage in {"train", "val"}

        def _log_metric(name: str, value: torch.Tensor, *, prog_bar: bool = False) -> None:
            self.log(
                name,
                value,
                prog_bar=prog_bar,
                batch_size=batch_size,
                add_dataloader_idx=False,
                on_step=is_train,
                on_epoch=True,
            )

        _log_metric(f"{stage}_loss", loss, prog_bar=prog_bar_enabled)
        _log_metric(f"{stage}_acc", metrics["acc"], prog_bar=prog_bar_enabled)
        _log_metric(
            f"{stage}_pred_yes_pct",
            metrics["yes_pct"],
            prog_bar=prog_bar_enabled,
        )
        _log_metric(
            f"{stage}_pred_no_pct",
            metrics["no_pct"],
            prog_bar=prog_bar_enabled,
        )
        if self._uncertain_index is not None:
            _log_metric(
                f"{stage}_pred_uncertain_pct",
                metrics["uncertain_pct"],
                prog_bar=prog_bar_enabled,
            )
            _log_metric(
                f"{stage}_uncertain_truth_error_pct",
                metrics["uncertain_truth_error_pct"],
                prog_bar=False,
            )
            _log_metric(
                f"{stage}_option_truth_uncertain_pct",
                metrics["option_truth_uncertain_pct"],
                prog_bar=False,
            )
        if gambler_stats is not None and stage in {"train", "val"}:
            _log_metric(
                f"{stage}_gambler_o",
                gambler_stats["current_o"].detach(),
                prog_bar=False,
            )
        return loss

    def _compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mask = labels != IGNORE_INDEX
        preds = logits.argmax(dim=-1)
        zero = logits.new_tensor(0.0)
        result = {
            "acc": zero,
            "yes_pct": zero,
            "no_pct": zero,
            "uncertain_pct": zero,
            "uncertain_truth_error_pct": zero,
            "option_truth_uncertain_pct": zero,
        }
        if mask.any():
            masked_preds = preds[mask]
            masked_labels = labels[mask]
            acc = (masked_preds == masked_labels).float().mean()
            result["acc"] = acc.detach()
            total = masked_preds.numel() or 1
            yes_pct = (masked_preds == self._yes_index).float().sum() / float(total) * 100.0
            no_pct = (masked_preds == self._no_index).float().sum() / float(total) * 100.0
            result["yes_pct"] = yes_pct.detach()
            result["no_pct"] = no_pct.detach()
            if self._uncertain_index is not None:
                uncertain_token = self._uncertain_index
                uncertain_pct = (masked_preds == uncertain_token).float().sum() / float(total) * 100.0
                result["uncertain_pct"] = uncertain_pct.detach()
                uncertain_truth_mask = masked_labels == uncertain_token
                uncertain_total = uncertain_truth_mask.sum().float()
                if uncertain_total > 0:
                    uncertain_wrong = (
                        masked_preds[uncertain_truth_mask] != uncertain_token
                    ).float().sum()
                    result["uncertain_truth_error_pct"] = (
                        uncertain_wrong / uncertain_total * 100.0
                    ).detach()
                answerable_mask = (masked_labels == self._yes_index) | (masked_labels == self._no_index)
                answerable_total = answerable_mask.sum().float()
                if answerable_total > 0:
                    abstain = (masked_preds[answerable_mask] == uncertain_token).float().sum()
                    result["option_truth_uncertain_pct"] = (
                        abstain / answerable_total * 100.0
                    ).detach()
        return result

    def _class_index_to_token(self, index: int) -> int:
        if 0 <= index < len(self._label_tokens):
            return self._label_tokens[index]
        return index

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        return self._decode_from_hidden(hidden, padding_mask)

    def forward_with_hidden(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self._decode_from_hidden(hidden, padding_mask)

    def _decode_from_hidden(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.size(1)
        max_seq_len = self.causal_mask.size(0)
        if seq_len > max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum of {max_seq_len}."
            )
        hidden = self.embed_dropout(hidden)
        attn_mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            hidden = block(hidden, attn_mask, padding_mask)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def _build_sequence_inputs(self, batch: dict, metadata: list[dict]) -> SequenceInputs:
        sequence_ids = batch["input_ids"]
        lengths = batch["lengths"]
        padding_mask = batch.get("padding_mask")
        provided_embeddings = batch.get("embeddings")
        embedding_mask = batch.get("embedding_mask")
        batch_size = sequence_ids.size(0)
        stream_lengths = self._extract_stream_lengths(metadata, batch_size, sequence_ids.device)

        token_embeddings = None
        if provided_embeddings is not None:
            provided_embeddings = self._project_video_embeddings(provided_embeddings)
            provided_embeddings = provided_embeddings.to(device=sequence_ids.device)

        need_decoder_embeddings = self._backend_requires_stream_embeds or provided_embeddings is not None
        if need_decoder_embeddings:
            token_embeddings = self.token_embedding(sequence_ids)
            if provided_embeddings is not None:
                if provided_embeddings.size() != token_embeddings.size():
                    raise ValueError(
                        "Video embeddings must match the sequence input id tensor shape after projection."
                    )
                if embedding_mask is None:
                    token_embeddings = provided_embeddings.to(device=token_embeddings.device)
                else:
                    mask = embedding_mask.to(device=token_embeddings.device, dtype=torch.bool).unsqueeze(-1)
                    token_embeddings = torch.where(mask, provided_embeddings.to(token_embeddings.device), token_embeddings)

        return SequenceInputs(
            input_ids=sequence_ids,
            token_embeddings=token_embeddings,
            lengths=lengths,
            padding_mask=padding_mask,
            stream_lengths=stream_lengths,
        )

    @staticmethod
    def _extract_stream_lengths(metadata: list[dict], batch_size: int, device: torch.device) -> torch.Tensor:
        if len(metadata) != batch_size:
            raise ValueError("Metadata entries must align with the batch dimension.")
        lengths: list[int] = []
        for entry in metadata:
            if entry is None:
                raise ValueError("Sequence metadata missing for one or more samples.")
            stream_length = entry.get("stream_length")
            if stream_length is None:
                raise ValueError("metadata['stream_length'] is required for every sample.")
            resolved = int(stream_length)
            if resolved < 0:
                raise ValueError("stream_length must be non-negative.")
            lengths.append(resolved)
        return torch.tensor(lengths, dtype=torch.long, device=device)

    def _align_query_embeddings(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        if embeddings.size(-1) == target_dim:
            return embeddings
        key = str(embeddings.size(-1))
        if key not in self._query_projections:
            projector = nn.Linear(embeddings.size(-1), target_dim)
            self._query_projections[key] = projector
        return self._query_projections[key](embeddings)

    def _project_video_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        projector = self._feature_projector
        if projector is None:
            return embeddings
        if embeddings.size(-1) != projector.in_features:
            raise ValueError(
                "Video embeddings expected dimension {} but got {}".format(
                    projector.in_features,
                    embeddings.size(-1),
                )
            )
        orig_shape = embeddings.shape
        reshaped = embeddings.view(-1, orig_shape[-1])
        projected = projector(reshaped)
        projected = projected.view(*orig_shape[:-1], projector.out_features)
        return projected.to(dtype=self.token_embedding.weight.dtype)

    def _resolve_sequence_padding(self, sequence_inputs: SequenceInputs) -> torch.Tensor:
        padding_mask = sequence_inputs.padding_mask
        if padding_mask is None:
            max_len = sequence_inputs.input_ids.size(1)
            arange = torch.arange(max_len, device=sequence_inputs.input_ids.device).unsqueeze(0)
            padding_mask = arange >= sequence_inputs.lengths.unsqueeze(1)
        return padding_mask.to(dtype=torch.bool)

    def _gather_label_logits(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        decoder_dtype = self.token_embedding.weight.dtype
        if hidden.dtype != decoder_dtype:
            hidden = hidden.to(decoder_dtype)
        hidden = self.embed_dropout(hidden)
        hidden = self.norm(hidden)

        label_mask = labels != IGNORE_INDEX
        if padding_mask is not None:
            label_mask = label_mask & ~padding_mask
        batch_size = hidden.size(0)
        labeled_counts = label_mask.sum(dim=1)
        max_labeled = int(labeled_counts.max().item()) if labeled_counts.numel() > 0 else 0
        if max_labeled == 0:
            empty_logits = hidden.new_zeros(0, self._num_label_classes)
            empty_labels = labels.new_full((0,), IGNORE_INDEX, dtype=torch.long)
            return empty_logits, empty_labels

        hidden_labeled = hidden.new_zeros(batch_size, max_labeled, hidden.size(-1))
        labels_labeled = labels.new_full((batch_size, max_labeled), IGNORE_INDEX, dtype=torch.long)
        for b in range(batch_size):
            valid_positions = torch.nonzero(label_mask[b], as_tuple=False).view(-1)
            num_valid = min(int(valid_positions.numel()), max_labeled)
            if num_valid == 0:
                continue
            hidden_labeled[b, :num_valid] = hidden[b, valid_positions[:num_valid], :]
            labels_labeled[b, :num_valid] = labels[b, valid_positions[:num_valid]]

        class_labels = torch.full_like(labels_labeled, IGNORE_INDEX)
        for idx, token in enumerate(self._label_tokens):
            mask = labels_labeled == token
            if mask.any():
                class_labels.masked_fill_(mask, idx)

        logits = self.label_head(hidden_labeled)
        return logits, class_labels

    def _shared_step(self, batch, stage: str):
        if not isinstance(batch, dict):
            raise TypeError(
                "MemoryBankDecoder expects the collate output with 'sequence', 'labels', and metadata."
            )
        sequence_batch = batch.get("sequence")
        labels = batch.get("labels")
        metadata = batch.get("metadata")
        if sequence_batch is None or labels is None or metadata is None:
            raise ValueError("Batch must include 'sequence', 'labels', and 'metadata' entries.")
        sequence_inputs = self._build_sequence_inputs(sequence_batch, metadata)
        self._capture_batch_metadata(metadata, sequence_inputs)
        logits, class_labels = self._compute_logits_from_inputs(sequence_inputs, labels)
        batch_size = labels.size(0)
        loss = self._finalize_step(logits, class_labels, stage, batch_size=batch_size)
        self._accumulate_debug_metrics(stage, logits, class_labels)
        if torch.isnan(loss):
            raise RuntimeError(
                "Encountered NaN loss during {} step; aborting to avoid corrupting checkpoints.".format(stage)
            )
        if stage == "val":
            self._maybe_log_samples(batch, logits, class_labels)
            self._accumulate_bucket_metrics(logits, class_labels)
        return loss

    def _compute_logits_from_inputs(
        self,
        sequence_inputs: SequenceInputs,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_mask = labels != IGNORE_INDEX
        hidden = self.memory_backend.encode_sequence(sequence_inputs, label_mask)
        padding_mask = self._resolve_sequence_padding(sequence_inputs)
        return self._gather_label_logits(hidden, labels, padding_mask)

    def compute_sequence_logits(
        self,
        sequence_batch: dict,
        labels: torch.Tensor,
        metadata: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_inputs = self._build_sequence_inputs(sequence_batch, metadata)
        return self._compute_logits_from_inputs(sequence_inputs, labels)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        stage = "val"
        if self._synthetic_val_enabled:
            if self._synthetic_val_has_primary:
                if dataloader_idx == 1:
                    stage = "synthetic_val"
            else:
                stage = "synthetic_val"
        self._shared_step(batch, stage)

    def configure_synthetic_val(self, *, enabled: bool, has_primary_val: bool) -> None:
        self._synthetic_val_enabled = bool(enabled)
        self._synthetic_val_has_primary = bool(has_primary_val)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._prepare_fractional_warmup()

    def on_validation_epoch_start(self):
        self._sample_logs_written = 0
        self._reset_bucket_trackers()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_bucket_metrics()
        self._emit_debug_summary("val")
        if self._synthetic_val_enabled:
            self._emit_debug_summary("synthetic_val")

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
        *args,
        **kwargs,
    ):
        if optimizer_closure is None:
            return super().optimizer_step(
                epoch=epoch,
                batch_idx=batch_idx,
                optimizer=optimizer,
                optimizer_closure=optimizer_closure,
            )
        closure_result = optimizer_closure()
        if self._fractional_warmup_active and optimizer is self._fractional_warmup_optimizer:
            self._apply_fractional_warmup_step(optimizer)
        optimizer.step()
        return closure_result

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if self._fractional_warmup_fraction is None:
            return
        checkpoint["fractional_warmup_state"] = {
            "fraction": self._fractional_warmup_fraction,
            "target_steps": self._fractional_warmup_target_steps,
            "completed_steps": self._fractional_warmup_completed_steps,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        state = checkpoint.get("fractional_warmup_state")
        if state:
            self._fractional_warmup_checkpoint_state = {
                "target_steps": int(state.get("target_steps", 0) or 0),
                "completed_steps": int(state.get("completed_steps", 0) or 0),
            }

    def configure_optimizers(self):
        param_groups = self._build_weight_decay_param_groups()
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)
        scheduler = self._build_lr_scheduler(optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _build_lr_scheduler(self, optimizer: Optimizer):
        base_scheduler = self._build_base_scheduler(optimizer)
        if self.lr_warmup_epochs <= 0:
            return base_scheduler
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.lr_warmup_epochs,
        )
        tail_scheduler = base_scheduler
        if tail_scheduler is None:
            tail_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda _epoch: 1.0,
            )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, tail_scheduler],
            milestones=[self.lr_warmup_epochs],
        )

    def _prepare_fractional_warmup(self) -> None:
        if self._fractional_warmup_fraction is None:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None or trainer.sanity_checking:
            return
        if self.current_epoch > 0:
            self._fractional_warmup_active = False
            return
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if not optimizer:
                raise RuntimeError("No optimizer available for fractional warmup.")
            optimizer = optimizer[0]
        if not isinstance(optimizer, Optimizer):
            raise RuntimeError("Fractional warmup requires a valid optimizer instance.")
        steps_per_epoch = self._optimizer_steps_per_epoch()
        self._initialize_fractional_warmup(optimizer, steps_per_epoch)

    def _optimizer_steps_per_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("Trainer must be available to compute fractional warmup steps.")
        total_batches = trainer.num_training_batches
        if total_batches is None or total_batches == float("inf"):
            raise RuntimeError("Cannot apply fractional warmup with infinite training batches.")
        if total_batches <= 0:
            raise RuntimeError("Training dataloader must provide at least one batch for warmup.")
        accumulation = trainer.accumulate_grad_batches
        if isinstance(accumulation, dict):
            accumulation = accumulation.get(0, 1)
        accumulation = int(accumulation or 1)
        accumulation = max(1, accumulation)
        return max(1, math.ceil(int(total_batches) / accumulation))

    def _initialize_fractional_warmup(self, optimizer: Optimizer, optimizer_steps: int) -> None:
        fraction = self._fractional_warmup_fraction
        if fraction is None:
            self._fractional_warmup_active = False
            return
        target_steps = max(1, int(math.ceil(optimizer_steps * fraction)))
        restored = self._fractional_warmup_checkpoint_state or {}
        completed = int(restored.get("completed_steps", 0) or 0)
        if completed >= target_steps:
            self._fractional_warmup_active = False
            self._fractional_warmup_target_steps = target_steps
            self._fractional_warmup_completed_steps = target_steps
            self._fractional_warmup_checkpoint_state = None
            return
        self._fractional_warmup_target_steps = target_steps
        self._fractional_warmup_completed_steps = completed
        self._fractional_warmup_optimizer = optimizer
        self._fractional_warmup_base_lrs = [group["lr"] for group in optimizer.param_groups]
        base_progress = 0.0
        if target_steps > 0:
            base_progress = completed / float(target_steps)
        for group, base_lr in zip(optimizer.param_groups, self._fractional_warmup_base_lrs):
            group["lr"] = base_lr * base_progress
        self._fractional_warmup_active = True
        self._fractional_warmup_checkpoint_state = None

    def _apply_fractional_warmup_step(self, optimizer: Optimizer) -> None:
        if not self._fractional_warmup_active:
            return
        if optimizer is not self._fractional_warmup_optimizer:
            return
        total = self._fractional_warmup_target_steps
        base_lrs = self._fractional_warmup_base_lrs
        if not total or not base_lrs:
            self._fractional_warmup_active = False
            return
        next_step = self._fractional_warmup_completed_steps + 1
        progress = min(1.0, next_step / float(total))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * progress
        self._fractional_warmup_completed_steps = next_step
        if next_step >= total:
            self._fractional_warmup_active = False

    def _build_weight_decay_param_groups(self) -> list[dict[str, Any]]:
        decay_params, no_decay_params = self._split_parameters_for_weight_decay()
        groups: list[dict[str, Any]] = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": self.weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        if not groups:
            raise RuntimeError("No trainable parameters found while building optimizer groups.")
        return groups

    def _split_parameters_for_weight_decay(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        norm_param_names = self._collect_norm_parameter_names()
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            target = (
                no_decay_params
                if self._should_exclude_from_weight_decay(name, param, norm_param_names)
                else decay_params
            )
            target.append(param)
        return decay_params, no_decay_params

    def _collect_norm_parameter_names(self) -> set[str]:
        norm_params: set[str] = set()
        for module_name, module in self.named_modules():
            if not self._is_norm_module(module):
                continue
            prefix = f"{module_name}." if module_name else ""
            for param_name, _ in module.named_parameters(recurse=False):
                norm_params.add(f"{prefix}{param_name}")
        return norm_params

    def _is_norm_module(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            return True
        class_name = module.__class__.__name__
        return class_name in _NORM_CLASS_NAMES or "Norm" in class_name

    def _should_exclude_from_weight_decay(
        self,
        name: str,
        param: torch.nn.Parameter,
        norm_param_names: set[str],
    ) -> bool:
        if getattr(param, "_no_weight_decay", False):
            return True
        if name in norm_param_names:
            return True
        last_token = name.rsplit(".", 1)[-1]
        lowered = last_token.lower()
        if last_token.endswith("bias"):
            return True
        if last_token in _DYNAMICS_PARAM_NAMES:
            return True
        if any(keyword in lowered for keyword in _DYNAMICS_PARAM_KEYWORDS):
            return True
        return False

    def _build_base_scheduler(self, optimizer: Optimizer):
        if self.lr_scheduler_mode == "cosine_restart":
            cosine_epochs = max(1, min(self.max_epochs, 5))
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cosine_epochs,
                T_mult=2,
                eta_min=0.0,
            )
        if self.lr_scheduler_mode == "cosine_epoch":
            cosine_steps = max(1, self.max_epochs - self.lr_warmup_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=0.0,
            )
        if self.lr_scheduler_mode == "constant":
            return None
        raise RuntimeError(
            f"Unexpected lr_scheduler_mode={self.lr_scheduler_mode!r} in optimizer configuration."
        )

    def _maybe_log_samples(self, batch, logits, class_labels) -> None:
        if self.sample_log_limit <= 0 or self._sample_logs_written >= self.sample_log_limit:
            return
        sequence = batch["sequence"]
        metadata = batch.get("metadata") or [{} for _ in range(class_labels.size(0))]
        sequence_ids = sequence["input_ids"]
        lengths = sequence["lengths"]
        label_tokens = batch["labels"]
        preds_compact = logits.argmax(dim=-1)

        for i in range(class_labels.size(0)):
            if self._sample_logs_written >= self.sample_log_limit:
                break
            entry_meta = metadata[i] or {}
            stream_len = int(entry_meta.get("stream_length", 0))
            total_len = int(lengths[i].item())
            if total_len <= stream_len:
                continue
            query_ids = sequence_ids[i, stream_len:total_len]
            query_labels = label_tokens[i, stream_len:total_len]
            pred_full = query_labels.new_full(query_labels.shape, -1)
            valid_label_positions = torch.nonzero(query_labels != IGNORE_INDEX, as_tuple=False).view(-1)
            valid_prediction_slots = torch.nonzero(class_labels[i] != IGNORE_INDEX, as_tuple=False).view(-1)
            count = min(int(valid_label_positions.numel()), int(valid_prediction_slots.numel()))
            if count == 0:
                continue
            for local_idx in range(count):
                target_pos = valid_label_positions[local_idx]
                pred_index = int(preds_compact[i, local_idx].item())
                pred_full[target_pos] = self._class_index_to_token(pred_index)

            stream_tokens = sequence_ids[i, :stream_len]
            if self.task == "continuation":
                parsed = self._parse_continuation_queries(
                    query_ids,
                    query_labels,
                    pred_full,
                )
            else:
                parsed = self._parse_membership_queries(
                    query_ids,
                    query_labels,
                    pred_full,
                )
            if not parsed:
                continue
            print("--- Validation sample", self._sample_logs_written + 1, "---")
            print("Stream:", self._format_tokens(stream_tokens))
            for info in parsed:
                truth = self._describe_label(info["truth_token"])
                pred = self._describe_label(info["pred_token"])
                if self.task == "continuation":
                    prefix = self._format_tokens(torch.tensor(info["prefix"]))
                    candidate = self._format_tokens(torch.tensor(info["candidate"]))
                    print(f"Prefix {prefix} :: Cand {candidate} | truth={truth} | pred={pred}")
                else:
                    query = self._format_tokens(torch.tensor(info["tokens"]))
                    print(f"Query {query} | truth={truth} | pred={pred}")
            self._sample_logs_written += 1

    def _capture_batch_metadata(self, metadata: list[dict], sequence_inputs: SequenceInputs) -> None:
        self._last_batch_metadata = metadata
        self._last_batch_stream_lengths = sequence_inputs.stream_lengths

    def _reset_bucket_trackers(self) -> None:
        self._video_tracker = VideoBucketTracker()

    def _accumulate_bucket_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        tracker = self._video_tracker
        if tracker is None:
            return
        metadata_entries = getattr(self, "_last_batch_metadata", None)
        batch_size = logits.size(0)
        if not metadata_entries:
            metadata_entries = [None] * batch_size

        stream_lengths = self._last_batch_stream_lengths
        if stream_lengths is None:
            return
        mask = labels != IGNORE_INDEX
        preds = logits.argmax(dim=-1)
        for idx in range(batch_size):
            label_positions = torch.nonzero(mask[idx], as_tuple=False).view(-1)
            if label_positions.numel() == 0:
                continue
            payload = metadata_entries[idx] or {}
            stream_len = int(stream_lengths[idx].item())
            video_meta = payload.get("video") or {}
            if not video_meta:
                video_meta = {
                    "stream_length": stream_len,
                    "length_value": float(stream_len),
                }
            length_value = _video_length_value(video_meta, stream_len)
            entropy_value = _video_entropy_value(video_meta)
            correct = 0
            uncertain_count = 0
            uncertain_truth_total = 0
            uncertain_truth_errors = 0
            option_truth_total = 0
            option_truth_uncertain = 0
            for pos in label_positions:
                truth_index = int(labels[idx, pos].item())
                pred_index = int(preds[idx, pos].item())
                truth_token = self._class_index_to_token(truth_index)
                pred_token = self._class_index_to_token(pred_index)
                if pred_index == truth_index:
                    correct += 1
                if truth_token == UNCERTAIN_TOKEN:
                    uncertain_truth_total += 1
                    if pred_token != UNCERTAIN_TOKEN:
                        uncertain_truth_errors += 1
                elif truth_token in {YES_TOKEN, NO_TOKEN}:
                    option_truth_total += 1
                    if pred_token == UNCERTAIN_TOKEN:
                        option_truth_uncertain += 1
                if pred_token == UNCERTAIN_TOKEN:
                    uncertain_count += 1
            tracker.add_record(
                length_value=length_value,
                entropy_value=entropy_value,
                correct=correct,
                total=label_positions.numel(),
                uncertain=uncertain_count,
                uncertain_truth_total=uncertain_truth_total,
                uncertain_truth_errors=uncertain_truth_errors,
                option_truth_total=option_truth_total,
                option_truth_uncertain=option_truth_uncertain,
            )

    def _accumulate_debug_metrics(self, stage: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        stats = self._debug_stage_stats.get(stage)
        if stats is None:
            return
        mask = labels != IGNORE_INDEX
        total = int(mask.sum().item())
        if total <= 0:
            return
        preds = logits.argmax(dim=-1)
        masked_preds = preds[mask]
        masked_labels = labels[mask]
        stats["total"] += total
        stats["pred_yes"] += int((masked_preds == self._yes_index).sum().item())
        stats["pred_no"] += int((masked_preds == self._no_index).sum().item())
        stats["label_yes"] += int((masked_labels == self._yes_index).sum().item())
        stats["label_no"] += int((masked_labels == self._no_index).sum().item())
        stats["correct"] += int((masked_preds == masked_labels).sum().item())

    def _emit_debug_summary(self, stage: str) -> None:
        stats = self._debug_stage_stats.get(stage)
        if not stats or stats.get("total", 0) == 0:
            return
        total = stats["total"]
        pred_yes = stats["pred_yes"] / total
        label_yes = stats["label_yes"] / total
        acc = stats["correct"] / total
        print(
            f"[debug:{stage}] total={total} pred_yes={pred_yes:.3f} label_yes={label_yes:.3f} acc={acc:.3f}",
            flush=True,
        )
        stats.clear()

    def _log_bucket_metrics(self) -> None:
        tracker = self._video_tracker
        if tracker is None or not tracker.has_data():
            return
        entropy_bounds = self._entropy_boundaries
        if self._bucket_entropy_enabled:
            if not entropy_bounds:
                entropy_bounds = tracker.compute_tertiles("entropy")
                if entropy_bounds:
                    self._entropy_boundaries = entropy_bounds
            if entropy_bounds:
                for idx, entry in enumerate(tracker.summary("entropy", entropy_bounds), start=1):
                    self.log(
                        f"val_entropy_bucket_{idx}_acc",
                        entry["accuracy"],
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
                    self.log(
                        f"val_entropy_bucket_{idx}_uncertain_pct",
                        entry["uncertain_pct"],
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
                    self.log(
                        f"val_entropy_bucket_{idx}_uncertain_truth_error_pct",
                        entry["uncertain_truth_error_pct"],
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
                    self.log(
                        f"val_entropy_bucket_{idx}_option_truth_uncertain_pct",
                        entry["option_truth_uncertain_pct"],
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
                    self.log(
                        f"val_entropy_bucket_{idx}_video_count",
                        entry["video_count"],
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
                for idx, bound in enumerate(entropy_bounds, start=1):
                    self.log(
                        f"val_entropy_tertile_{idx}",
                        bound,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        add_dataloader_idx=False,
                    )
        length_bounds = self._length_boundaries
        if not length_bounds:
            length_bounds = tracker.compute_tertiles("length")
            if length_bounds:
                self._length_boundaries = length_bounds
        if length_bounds:
            for idx, entry in enumerate(tracker.summary("length", length_bounds), start=1):
                self.log(
                    f"val_length_bucket_{idx}_acc",
                    entry["accuracy"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"val_length_bucket_{idx}_uncertain_pct",
                    entry["uncertain_pct"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"val_length_bucket_{idx}_uncertain_truth_error_pct",
                    entry["uncertain_truth_error_pct"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"val_length_bucket_{idx}_option_truth_uncertain_pct",
                    entry["option_truth_uncertain_pct"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"val_length_bucket_{idx}_video_count",
                    entry["video_count"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
            for idx, bound in enumerate(length_bounds, start=1):
                self.log(
                    f"val_length_tertile_{idx}",
                    bound,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )

    def _parse_membership_queries(
        self,
        query_ids: torch.Tensor,
        query_labels: torch.Tensor,
        pred_segment: torch.Tensor,
    ) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        current: list[int] = []
        collecting = False
        for idx in range(query_ids.size(0)):
            token = int(query_ids[idx].item())
            if token == self.pad_id:
                break
            label_token = int(query_labels[idx].item())
            pred_token = int(pred_segment[idx].item())
            if token == STREAM_QUERY_SEPARATOR and label_token == IGNORE_INDEX:
                current = []
                collecting = True
                continue
            if token == QUERY_END_SEPARATOR and label_token == IGNORE_INDEX:
                collecting = False
                continue
            if token == LABEL_SEPARATOR:
                if label_token == IGNORE_INDEX:
                    continue
                parsed.append(
                    {
                        "tokens": list(current),
                        "truth_token": label_token,
                        "pred_token": pred_token,
                    }
                )
                current = []
                collecting = False
                continue
            if collecting:
                current.append(token)
        return parsed

    def _parse_continuation_queries(
        self,
        query_ids: torch.Tensor,
        query_labels: torch.Tensor,
        pred_segment: torch.Tensor,
    ) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        prefix: list[int] = []
        candidate: list[int] = []
        section = None
        for idx in range(query_ids.size(0)):
            token = int(query_ids[idx].item())
            if token == self.pad_id:
                break
            label_token = int(query_labels[idx].item())
            pred_token = int(pred_segment[idx].item())
            if token == STREAM_QUERY_SEPARATOR and label_token == IGNORE_INDEX:
                prefix = []
                candidate = []
                section = "prefix"
                continue
            if token == CANDIDATE_SEPARATOR and label_token == IGNORE_INDEX:
                section = "candidate"
                continue
            if token == QUERY_END_SEPARATOR and label_token == IGNORE_INDEX:
                section = None
                continue
            if token == LABEL_SEPARATOR:
                if label_token == IGNORE_INDEX:
                    continue
                parsed.append(
                    {
                        "prefix": list(prefix),
                        "candidate": list(candidate),
                        "truth_token": label_token,
                        "pred_token": pred_token,
                    }
                )
                prefix = []
                candidate = []
                section = None
                continue
                continue
            if section == "prefix":
                prefix.append(token)
            elif section == "candidate":
                candidate.append(token)
        return parsed

    def _describe_label(self, token: int) -> str:
        if token == YES_TOKEN:
            return "YES"
        if token == NO_TOKEN:
            return "NO"
        if token == UNCERTAIN_TOKEN:
            return "UNCERTAIN"
        return str(token)

    def _format_tokens(self, tokens: torch.Tensor) -> str:
        return " ".join(str(int(tok.item()) - TOKEN_OFFSET) for tok in tokens)

    def set_dataset_metadata(self, metadata: dict[str, Any] | None) -> None:
        self.dataset_metadata = metadata or {}
        source = str(self.dataset_metadata.get("source") or "").lower()
        self._bucket_entropy_enabled = source != "synthetic"
        entropy_bounds = self.dataset_metadata.get("entropy_prefix_tertiles")
        length_bounds = (
            self.dataset_metadata.get("stream_length_tertiles")
            or self.dataset_metadata.get("target_length_tertiles")
        )
        self._entropy_boundaries = sanitize_boundaries(entropy_bounds)
        self._length_boundaries = sanitize_boundaries(length_bounds)


def _avg(values: Iterable[int] | Iterable[float] | None) -> float | None:
    if not values:
        return None
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    return sum(data) / len(data)


def _video_length_value(meta: dict[str, Any] | None, fallback: int) -> float:
    if meta is None:
        return float(fallback)
    value = meta.get("length_value")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    stream_len = meta.get("stream_length")
    if stream_len is not None:
        try:
            return float(stream_len)
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _video_entropy_value(meta: dict[str, Any] | None) -> float | None:
    if meta is None:
        return None
    value = meta.get("entropy_value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Backwards-compatibility aliases
ContinuationLM = MemoryBankDecoder
StreamQueryContinuationLM = MemoryBankDecoder
