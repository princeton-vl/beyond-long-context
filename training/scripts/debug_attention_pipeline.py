#!/usr/bin/env python
"""Inspect sequence inputs, masks, and grads for attention-based backends."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from calibrated_memory.training.registries import build_backend, build_dataset
from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.backend.models.backend_base import SequenceInputs
from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.common import IGNORE_INDEX


@dataclass
class SequenceSummary:
    input_ids: torch.Tensor
    padding_mask: torch.Tensor | None
    label_mask: torch.Tensor


class DebugDecoder(MemoryBankDecoder):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.last_summary: SequenceSummary | None = None

    def _build_sequence_inputs(self, batch: dict, metadata: list[dict]) -> SequenceInputs:  # type: ignore[override]
        seq = super()._build_sequence_inputs(batch, metadata)
        padding_mask = seq.padding_mask
        ids = seq.input_ids
        if padding_mask is None:
            pad_info = "None"
        else:
            pad_info = f"shape={tuple(padding_mask.shape)} true={int(padding_mask.sum().item())}"
        print(f"[debug] sequence ids shape={tuple(ids.shape)} pad_mask={pad_info}")
        if padding_mask is not None:
            print(f"[debug] first padding row: {padding_mask[0].tolist()}")
        print(f"[debug] first input ids row: {ids[0].tolist()}")
        return seq

    def _gather_label_logits(self, hidden: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        label_mask = labels != IGNORE_INDEX
        if padding_mask is not None:
            label_mask = label_mask & ~padding_mask
        positions = torch.nonzero(label_mask, as_tuple=False)
        sample_labels = []
        for pos in positions[:8]:
            sample_labels.append(int(labels[pos[0], pos[1]].item()))
        print(
            f"[debug] label_mask true count={int(label_mask.sum().item())} positions={positions.tolist()} label_values={sample_labels}"
        )
        self.last_summary = SequenceSummary(
            input_ids=self._last_batch_input_ids,
            padding_mask=self._last_batch_padding_mask,
            label_mask=label_mask.detach().cpu(),
        )
        return super()._gather_label_logits(hidden, labels, padding_mask)

    def _capture_batch_metadata(self, metadata: list[dict], sequence_inputs: SequenceInputs) -> None:  # type: ignore[override]
        super()._capture_batch_metadata(metadata, sequence_inputs)
        self._last_batch_input_ids = sequence_inputs.input_ids.detach().cpu()
        self._last_batch_padding_mask = (
            None if sequence_inputs.padding_mask is None else sequence_inputs.padding_mask.detach().cpu()
        )


def gradient_report(module: torch.nn.Module, prefix: str) -> None:
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        norm = param.grad.detach().float().norm().item()
        print(f"[grad:{prefix}] {name} norm={norm:.4e}")


def run_case(backend_name: str, backend_overrides: dict[str, Any], *, device: torch.device) -> None:
    dataset_cfg = {
        "num_sequences": 8,
        "unique_sequences": 8,
        "seq_len_min": 32,
        "seq_len_max": 64,
        "vocab_size": 32,
        "task": "membership",
        "cont_len": 4,
        "seed": 0,
    }
    artifacts = build_dataset("synthetic", dataset_cfg)
    collate = build_collate(artifacts.pad_id)
    loader = DataLoader(
        artifacts.dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    backend, backend_config = build_backend(backend_name, backend_overrides)
    decoder = DebugDecoder(
        vocab_size=artifacts.vocab_size,
        pad_id=artifacts.pad_id,
        max_seq_len=artifacts.max_seq_len,
        d_model=64,
        nhead=8,
        num_layers=2,
        mlp_ratio=4,
        lr=1e-3,
        max_epochs=1,
        memory_backend=backend,
        task="membership",
        grad_component_specs=[],
    ).to(device)
    for key in ["sequence", "labels"]:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for subkey, tensor in batch[key].items():
                if tensor is not None:
                    batch[key][subkey] = tensor.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss = decoder.training_step(batch, 0)
    optimizer.zero_grad()
    loss.backward()
    gradient_report(backend, f"backend-{backend_name}")
    gradient_report(decoder, f"decoder-{backend_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug transformer attention inputs")
    parser.add_argument("backend", choices=["transformer_pp", "simple_rnn"], help="Backend to inspect")
    parser.add_argument(
        "--positional-mode",
        default="rope",
        choices=["rope", "pope"],
        help="Positional mode for transformer backend (ignored for others).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {"embed_dim": 64, "num_layers": 3}
    if args.backend == "transformer_pp":
        overrides.update(
            {
                "num_heads": 4,
                "mlp_ratio": 2,
                "dropout": 0.0,
                "positional_mode": args.positional_mode,
                "use_flash_attention": False,
                "use_qk_norm": True,
                "qk_norm_eps": 1e-6,
            }
        )
    elif args.backend == "simple_rnn":
        overrides.update({"hidden_dim": 64, "dropout": 0.0})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running backend={args.backend} positional={args.positional_mode} on {device}...")
    run_case(args.backend, overrides, device=device)


if __name__ == "__main__":
    main()
