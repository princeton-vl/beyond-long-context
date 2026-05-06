#!/usr/bin/env python
"""Generate a synthetic dataset and fine-tune a checkpoint with gambler's loss."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.evaluation.checkpoint import load_run_metadata
from calibrated_memory.training.data import create_dataloaders
from calibrated_memory.training.registries import build_backend, build_dataset
from calibrated_memory.valset.generation import ValGenerationConfig, generate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Base training run directory (provides config + checkpoint).")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--task", choices=["membership", "continuation"], required=True)
    parser.add_argument("--token-offset", type=int, default=32)
    parser.add_argument("--train-manifest", type=Path, default=None, help="Reuse an existing manifest instead of generating one.")
    parser.add_argument("--train-output", type=Path, default=None, help="Directory to store generated training manifest.")
    parser.add_argument("--train-num-sequences", type=int, default=2000)
    parser.add_argument("--queries-per-sequence", type=int, default=15)
    parser.add_argument("--cont-len", type=int, default=4, help="Continuation length when task=continuation.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deep-gambler-o", type=float, default=1.5)
    parser.add_argument("--deep-gambler-eps", type=float, default=1e-12)
    parser.add_argument("--deep-gambler-activation-acc", type=float, default=0.33)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.run_dir)
    base_args = metadata.get("args", {})
    backend_name = metadata["backend"]
    backend_overrides = metadata.get("backend_overrides", {})
    checkpoint_path = args.run_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    manifest_path = args.train_manifest or _generate_manifest(args)
    dataset_overrides = {
        "path": str(manifest_path),
        "task": args.task,
        "token_offset": args.token_offset,
    }
    if args.task == "continuation":
        dataset_overrides["cont_len"] = args.cont_len
    dataset_artifacts = build_dataset("file", dataset_overrides)
    batch_size = args.batch_size or int(base_args.get("batch_size", 32))
    dataloaders = create_dataloaders(
        dataset_artifacts,
        batch_size=batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
    )
    backend, _ = build_backend(backend_name, backend_overrides)
    model = MemoryBankDecoder(
        vocab_size=dataset_artifacts.vocab_size,
        pad_id=dataset_artifacts.pad_id,
        max_seq_len=dataset_artifacts.max_seq_len,
        d_model=int(base_args.get("decoder_d_model", 256)),
        nhead=int(base_args.get("decoder_nhead", 8)),
        num_layers=int(base_args.get("decoder_num_layers", 4)),
        mlp_ratio=int(base_args.get("decoder_mlp_ratio", 4)),
        lr=float(base_args.get("learning_rate", 3e-4)),
        attn_dropout=float(base_args.get("decoder_attn_dropout", 0.0)),
        embed_dropout=float(base_args.get("decoder_embed_dropout", 0.1)),
        resid_dropout=float(base_args.get("decoder_resid_dropout", 0.1)),
        rotary_base=float(base_args.get("decoder_rotary_base", 10000.0)),
        weight_decay=float(base_args.get("weight_decay", 0.0)),
        max_epochs=args.max_epochs,
        memory_backend=backend,
        task=args.task,
        loss_type="deep_gambler",
        deep_gambler_mode=str(base_args.get("deep_gambler_mode", "fixed")),
        deep_gambler_o=args.deep_gambler_o,
        deep_gambler_epsilon=args.deep_gambler_eps,
        deep_gambler_activation_acc=args.deep_gambler_activation_acc,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state.get("state_dict", state)
    model.load_state_dict(weights)
    save_dir = Path(args.save_dir or _default_save_dir(args.run_dir))
    trainer = pl.Trainer(
        default_root_dir=str(save_dir),
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        precision=args.precision or base_args.get("precision", "32-true"),
        enable_checkpointing=True,
        log_every_n_steps=int(base_args.get("log_every_n_steps", 10)),
    )
    trainer.fit(model, dataloaders.train, dataloaders.val)
    checkpoint_out = save_dir / "gambler_last.ckpt"
    trainer.save_checkpoint(checkpoint_out)
    print(f"Fine-tuned checkpoint saved to {checkpoint_out}")


def _generate_manifest(args: argparse.Namespace) -> Path:
    cfg = ValGenerationConfig(
        task=args.task,
        num_sequences=args.train_num_sequences,
        queries_per_sequence=args.queries_per_sequence,
        vocab_size=16,
        seed=args.seed,
        token_offset=args.token_offset,
        cont_len=args.cont_len,
    )
    result = generate_manifest(cfg)
    output_dir = args.train_output or _default_manifest_dir(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "questions.json"
    manifest_path.write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    return manifest_path


def _default_manifest_dir(run_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = run_dir / "finetune_manifests" / f"run-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_save_dir(run_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = run_dir / "finetune" / f"run-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
