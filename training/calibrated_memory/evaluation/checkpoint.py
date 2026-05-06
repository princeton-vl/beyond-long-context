"""Helpers for loading trained checkpoints for evaluation or interactive flows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Tuple

import torch

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.training.data import DatasetArtifacts


def _estimate_direct_mode_context(artifacts: DatasetArtifacts, sample_limit: int = 2048) -> int:
    """Approximate the combined stream+query length for direct backends."""

    dataset = getattr(artifacts, "dataset", None)
    if dataset is None:
        return 0
    try:
        length = len(dataset)
    except Exception:  # noqa: BLE001
        return 0
    limit = min(sample_limit, length)
    max_required = 0
    for idx in range(limit):
        try:
            sample = dataset[idx]
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            continue
        seq, _, stream_len_tensor = sample[:3]
        try:
            stream_len = int(stream_len_tensor.item())
        except Exception:  # noqa: BLE001
            try:
                stream_len = int(stream_len_tensor)
            except Exception:  # noqa: BLE001
                continue
        total_len = int(getattr(seq, "numel", lambda: len(seq))())
        query_len = max(0, total_len - stream_len)
        max_required = max(max_required, stream_len + query_len)
    return max_required
from calibrated_memory.training.registries import BACKEND_SPECS, build_backend, build_dataset


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Load the serialized config.json emitted at training time."""

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config metadata at {config_path}")
    return _read_json(config_path)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def instantiate_model_from_run(
    run_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    metadata: dict[str, Any] | None = None,
) -> Tuple[MemoryBankDecoder, dict[str, Any], DatasetArtifacts, dict[str, Any]]:
    """Recreate a MemoryBankDecoder and dataset artifacts from a saved run."""

    if metadata is None:
        print(f"[checkpoint] Loading config from {run_dir}", flush=True)
        metadata = load_run_metadata(run_dir)
    backend_overrides = dict(metadata.get("backend_overrides", {}))
    dataset_overrides = dict(metadata.get("dataset_overrides", {}))

    # Build dataset artifacts first so we can expand backend block sizes if needed.
    dataset_build_overrides = dict(dataset_overrides)
    original_max_seq = (
        dataset_build_overrides.get("seq_len_max")
        or dataset_build_overrides.get("seq_len")
        or dataset_overrides.get("seq_len_max")
        or dataset_overrides.get("seq_len")
    )
    if isinstance(original_max_seq, str):
        original_max_seq = int(original_max_seq)
    downscaled = None
    if metadata["dataset"] == "synthetic":
        downscaled = _downscale_synthetic_overrides(dataset_build_overrides)
    dataset_artifacts = build_dataset(metadata["dataset"], downscaled or dataset_build_overrides)
    print("[checkpoint] Dataset artifacts ready", flush=True)

    required_block = int(dataset_artifacts.max_seq_len)
    direct_requirement = _estimate_direct_mode_context(dataset_artifacts)
    if direct_requirement:
        required_block = max(required_block, direct_requirement)
    seq_override = dataset_overrides.get("seq_len_max") or dataset_overrides.get("seq_len")
    if seq_override is not None:
        try:
            required_block = max(required_block, int(seq_override))
        except (TypeError, ValueError):
            pass
    decoder_cap = int(metadata.get("args", {}).get("decoder_context_cap", 0) or 0)
    if decoder_cap > 0:
        required_block = max(required_block, decoder_cap)
    spec = BACKEND_SPECS.get(metadata["backend"])
    supports_block_size = bool(spec and "block_size" in spec.defaults)
    if required_block > 0 and supports_block_size:
        configured_block = int(backend_overrides.get("block_size", 0) or 0)
        if required_block > configured_block:
            backend_overrides["block_size"] = required_block

    if metadata["backend"] == "simple_rnn":
        backend_overrides.pop("block_size", None)
    print(
        f"[checkpoint] Building backend={metadata['backend']} with overrides={backend_overrides}",
        flush=True,
    )
    backend, _ = build_backend(metadata["backend"], backend_overrides)
    print("[checkpoint] Backend constructed", flush=True)

    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")
    print(f"[checkpoint] Loading weights from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    expected_context = None
    causal_mask = state.get("causal_mask")
    if causal_mask is not None and hasattr(causal_mask, "size"):
        expected_context = int(causal_mask.size(0))

    args = metadata.get("args", {})
    context_cap = int(args.get("decoder_context_cap", 0) or 0)
    max_seq_len = dataset_artifacts.max_seq_len
    if context_cap > 0:
        max_seq_len = max(max_seq_len, context_cap)
    if expected_context:
        max_seq_len = max(max_seq_len, expected_context)

    task = str(dataset_overrides.get("task", "membership"))

    model = MemoryBankDecoder(
        vocab_size=dataset_artifacts.vocab_size,
        pad_id=dataset_artifacts.pad_id,
        max_seq_len=max_seq_len,
        d_model=int(args.get("decoder_d_model", 256)),
        nhead=int(args.get("decoder_nhead", 8)),
        num_layers=int(args.get("decoder_num_layers", 4)),
        mlp_ratio=int(args.get("decoder_mlp_ratio", 4)),
        lr=float(args.get("learning_rate", 3e-4)),
        attn_dropout=float(args.get("decoder_attn_dropout", 0.0)),
        embed_dropout=float(args.get("decoder_embed_dropout", 0.1)),
        resid_dropout=float(args.get("decoder_resid_dropout", 0.1)),
        rotary_base=float(args.get("decoder_rotary_base", 10000.0)),
        weight_decay=float(args.get("weight_decay", 0.1)),
        max_epochs=int(args.get("max_epochs", 1)),
        memory_backend=backend,
        task=task,
        loss_type=str(args.get("loss_type", "cross_entropy")),
        deep_gambler_mode=str(args.get("deep_gambler_mode", "fixed")),
        deep_gambler_o=float(args.get("deep_gambler_o", 1.5)),
        deep_gambler_epsilon=float(args.get("deep_gambler_eps", 1e-12)),
        deep_gambler_activation_acc=float(args.get("deep_gambler_activation_acc", 0.33)),
    )
    state = checkpoint.get("state_dict", checkpoint)
    loaded_mask = state.get("causal_mask")
    if loaded_mask is not None and loaded_mask.shape != model.causal_mask.shape:
        print(
            "[checkpoint] Resizing causal_mask from",
            tuple(loaded_mask.shape),
            "to",
            tuple(model.causal_mask.shape),
            flush=True,
        )
        state = dict(state)
        state.pop("causal_mask", None)
        missing_mask = True
    else:
        missing_mask = False
    load_result = model.load_state_dict(state, strict=not missing_mask)
    if isinstance(load_result, tuple):  # older torch returns tuple of missing/unexpected
        missing, unexpected = load_result
        if missing:
            print(f"[checkpoint] Missing parameters after load: {missing}", flush=True)
        if unexpected:
            print(f"[checkpoint] Unexpected parameters after load: {unexpected}", flush=True)
    model.to(device)
    model.eval()
    print("[checkpoint] Model ready for evaluation", flush=True)
    if downscaled is not None and original_max_seq:
        dataset_artifacts = replace(dataset_artifacts, max_seq_len=int(original_max_seq))
    return model, dataset_overrides, dataset_artifacts, metadata


def _downscale_synthetic_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Reduce synthetic dataset size to avoid rebuilding hundreds of thousands of samples."""

    adjusted = dict(overrides)
    num_sequences = int(adjusted.get("num_sequences", 64))
    if num_sequences > 64:
        adjusted["num_sequences"] = 64
    return adjusted
