from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import torch

from calibrated_memory.backend.models.backend_base import MemoryBackend
from calibrated_memory.backend.models.identity import IdentityBackend
from calibrated_memory.backend.models.lt_ct import CompressiveTransformerBackend
from calibrated_memory.backend.models.mann import DNCBackend, STMBackend
from calibrated_memory.backend.models.simple_rnn import SimpleRNNEncoder
from calibrated_memory.backend.models.transformer_pp_backend import TransformerPPBackend
from calibrated_memory.backend.models.titans import TitansEncoder
from calibrated_memory.backend.models.titans_external import TitansExternalMAC
from calibrated_memory.backend.models.rwkv_backend import RWKVBackend
from calibrated_memory.backend.models.memory_mosaic_backend import MemoryMosaicBackend
from calibrated_memory.backend.models.deltanet_backend import DeltaNetBackend
from calibrated_memory.backend.models.gated_deltanet_backend import GatedDeltaNetBackend
from calibrated_memory.backend.models.deltaformer_backend import DeltaFormerBackend
from calibrated_memory.backend.models.mom_backend import MoMBackend
from calibrated_memory.backend.models.retnet_backend import RetNetBackend
from calibrated_memory.backend.models.gla_backend import GLABackend
from calibrated_memory.data.sequences.common import TOKEN_OFFSET
from calibrated_memory.data.sequences.sequence_generator import SequenceDataset, SyntheticSequenceDataset
from calibrated_memory.data.sequences.sources import BucketManifestSource
from calibrated_memory.data.video_features.dataset import VideoFeatureDataset

from calibrated_memory.training.data import DatasetArtifacts


BackendBuilder = Callable[[Dict[str, Any], Dict[str, Any]], MemoryBackend]
DatasetBuilder = Callable[[Dict[str, Any]], DatasetArtifacts]


@dataclass(frozen=True)
class BackendSpec:
    name: str
    description: str
    defaults: Dict[str, Any]
    builder: BackendBuilder
    required_keys: Tuple[str, ...] = ()

    def materialize(self, overrides: Dict[str, Any]) -> tuple[MemoryBackend, Dict[str, Any]]:
        config = self._merge(overrides)
        backend = self.builder(config, overrides)
        return backend, config

    def _merge(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(overrides) - set(self.defaults) - set(self.required_keys)
        if unknown:
            raise KeyError(f"Unknown backend options for {self.name}: {sorted(unknown)}")
        config = dict(self.defaults)
        config.update(overrides)
        missing = [key for key in self.required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required backend options {missing} for {self.name}")
        if "autocast_dtype" in config:
            config["autocast_dtype"] = _resolve_autocast_dtype(config["autocast_dtype"])
        return config


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    defaults: Dict[str, Any]
    builder: Callable[[Dict[str, Any]], DatasetArtifacts]
    required_keys: Tuple[str, ...] = ()

    def materialize(self, overrides: Dict[str, Any]) -> DatasetArtifacts:
        config = self._merge(overrides)
        return self.builder(config)

    def _merge(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(overrides) - set(self.defaults) - set(self.required_keys)
        if unknown:
            raise KeyError(f"Unknown dataset options for {self.name}: {sorted(unknown)}")
        config = dict(self.defaults)
        config.update(overrides)
        missing = [key for key in self.required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required dataset options {missing} for {self.name}")
        return config


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _parse_sequence_keys(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        resolved: list[str] = []
        for entry in value:
            resolved.extend(_parse_sequence_keys(entry) or [])
        return resolved or None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed or trimmed.lower() == "auto":
            return None
        if "," in trimmed:
            parts = [segment.strip() for segment in trimmed.split(",") if segment.strip()]
            return parts or None
        return [trimmed]
    return None


_AUTODTYPE_ALIASES: dict[str, torch.dtype | None] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
    "torch.half": torch.float16,
    "fp32": None,
    "float32": None,
    "torch.float32": None,
    "f32": None,
    "none": None,
    "": None,
    "null": None,
    "auto": None,
}


def _resolve_autocast_dtype(value: Any) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if normalized in _AUTODTYPE_ALIASES:
        return _AUTODTYPE_ALIASES[normalized]
    raise ValueError(
        "autocast_dtype must be one of bf16, fp16, or none/fp32 (case-insensitive); "
        f"got {value!r}."
    )


def _build_mamba_backend(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> MemoryBackend:
    try:
        from calibrated_memory.backend.models.mamba_backend import MambaBackend
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extensions
        raise RuntimeError(
            "Backend 'mamba' is unavailable because its CUDA extension stack failed to import. "
            "Install a compatible mamba-ssm/causal-conv1d build for the active Torch version."
        ) from exc
    return MambaBackend(
        embed_dim=int(cfg["embed_dim"]),
        num_layers=int(cfg["num_layers"]),
        d_state=int(cfg["d_state"]),
        d_conv=int(cfg["d_conv"]),
        expand=int(cfg["expand"]),
        dropout=float(cfg["dropout"]),
        headdim=int(cfg["headdim"]),
        headdim_was_overridden="headdim" in overrides,
    )


def _build_log_linear_mamba_backend(cfg: Dict[str, Any], _: Dict[str, Any]) -> MemoryBackend:
    try:
        from calibrated_memory.backend.models.log_linear_mamba_backend import LogLinearMambaBackend
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extensions
        raise RuntimeError(
            "Backend 'log_linear_mamba' is unavailable because its CUDA extension stack failed to import."
        ) from exc
    return LogLinearMambaBackend(
        embed_dim=int(cfg["embed_dim"]),
        num_layers=int(cfg["num_layers"]),
        ctx_len=int(cfg["ctx_len"]),
        head_dim=int(cfg["head_dim"]),
        expand=int(cfg["expand"]),
        n_groups=int(cfg["n_groups"]),
        conv_kernel=int(cfg["conv_kernel"]),
        chunk_size=int(cfg["chunk_size"]),
        use_bias=bool(cfg["use_bias"]),
        use_conv_bias=bool(cfg["use_conv_bias"]),
        vocab_size=int(cfg["vocab_size"]),
        allow_unstable_fused_kernel=_to_bool(cfg["allow_unstable_fused_kernel"]),
        attn_mode=None if cfg.get("attn_mode") in {None, "None", ""} else str(cfg["attn_mode"]),
        **(
            {}
            if cfg.get("state_size") in {None, "None"}
            else {"state_size": int(cfg["state_size"])}
        ),
    )


def _build_ttt_backend(cfg: Dict[str, Any], _: Dict[str, Any]) -> MemoryBackend:
    try:
        from calibrated_memory.backend.models.ttt_backend import TestTimeTrainingBackend
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extensions
        raise RuntimeError(
            "Backend 'ttt' is unavailable because its CUDA extension stack failed to import."
        ) from exc
    return TestTimeTrainingBackend(
        embed_dim=int(cfg["embed_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        mini_batch_size=int(cfg["mini_batch_size"]),
        window_size=int(cfg["window_size"]),
        mlp_ratio=int(cfg["mlp_ratio"]),
        attn_dropout=float(cfg["attn_dropout"]),
        resid_dropout=float(cfg["resid_dropout"]),
        use_gate=bool(cfg["use_gate"]),
        ttt_variant=str(cfg["ttt_variant"]),
    )


def _build_sequence_dataset(config: Dict[str, Any]) -> DatasetArtifacts:
    json_path = Path(config["path"])
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    vocab_override = config.get("vocab_size")
    vocab_value = int(vocab_override) if vocab_override is not None else None
    sequence_keys = _parse_sequence_keys(config.get("sequence_key"))
    token_offset = int(config["token_offset"])
    num_videos = int(config["num_videos"])
    manifest_root = config.get("manifest_root")
    if isinstance(manifest_root, str) and not manifest_root.strip():
        manifest_root = None
    root_path = Path(manifest_root) if manifest_root else None
    task = str(config["task"])
    if task not in {"membership", "continuation"}:
        raise ValueError("file dataset only supports membership/continuation tasks")

    max_stream_len = int(config.get("max_seq_len", 0) or 0)
    source = BucketManifestSource(
        json_path,
        token_offset=token_offset,
        num_videos=num_videos,
        truncate_len=0,
        sequence_keys=sequence_keys,
        root_path=root_path,
        include_questions=True,
        max_stream_len=max_stream_len,
    )
    dataset = SequenceDataset(
        records=source.records,
        unique_sequences=int(config["unique_sequences"]),
        token_offset=token_offset,
        task=task,
        cont_len=int(config["cont_len"]),
        vocab_size=vocab_value,
        metadata_summary=source.summary,
        sequence_keys=sequence_keys,
        manifest_root=root_path,
        max_stream_len=max_stream_len if max_stream_len > 0 else None,
    )
    seq_vocab = source.summary.get("sequence_vocab") if isinstance(source.summary, dict) else None
    if vocab_value is not None and seq_vocab:
        max_manifest = max(int(val) for val in seq_vocab.values() if val is not None)
        if max_manifest > vocab_value:
            raise SystemExit(
                f"Manifest tokens require vocabulary >= {max_manifest}, but vocab_size={vocab_value}" \
                "; either drop the override or increase vocab_size."
            )
    return DatasetArtifacts(
        dataset=dataset,
        pad_id=dataset.pad_id,
        vocab_size=dataset.vocab_size,
        max_seq_len=dataset.max_input_len,
        metadata=dataset.metadata_summary,
    )


def _build_synthetic_dataset(config: Dict[str, Any]) -> DatasetArtifacts:
    seq_len_value = config.get("seq_len")
    seq_len_min = config.get("seq_len_min")
    seq_len_max = config.get("seq_len_max")
    dataset_kwargs = {
        "num_sequences": int(config["num_sequences"]),
        "unique_sequences": int(config["unique_sequences"]),
        "vocab_size": int(config["vocab_size"]),
        "seed": int(config["seed"]),
        "task": str(config["task"]),
        "cont_len": int(config["cont_len"]),
    }
    if seq_len_value is not None:
        dataset_kwargs["seq_len"] = int(seq_len_value)
    else:
        if seq_len_min is None or seq_len_max is None:
            raise ValueError(
                "Synthetic dataset requires seq_len or both seq_len_min and seq_len_max overrides."
            )
        dataset_kwargs["seq_len_min"] = int(seq_len_min)
        dataset_kwargs["seq_len_max"] = int(seq_len_max)
    dataset = SyntheticSequenceDataset(
        **dataset_kwargs,
    )
    seq_bounds = getattr(dataset, "seq_len_bounds", (dataset.max_input_len, dataset.max_input_len))
    seq_min, seq_max = seq_bounds
    metadata = {
        "source": "synthetic",
        "num_sequences": int(config["num_sequences"]),
        "seq_len_min": seq_min,
        "seq_len_max": seq_max,
        "vocab_size": int(config["vocab_size"]),
        "task": str(config["task"]),
        "seed": int(config["seed"]),
    }
    if seq_min == seq_max:
        metadata["seq_len"] = seq_min
    return DatasetArtifacts(
        dataset=dataset,
        pad_id=dataset.pad_id,
        vocab_size=dataset.vocab_size,
        max_seq_len=dataset.max_input_len,
        metadata=metadata,
    )


def _build_feature_sequence_dataset(config: Dict[str, Any]) -> DatasetArtifacts:
    manifest = Path(config["manifest"])
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    dataset = VideoFeatureDataset(
        manifest_path=manifest,
        max_videos=int(config.get("max_videos", -1)),
        task=str(config.get("task", "membership")),
        cont_len=int(config.get("cont_len", 0) or 0),
        max_seq_len=int(config.get("max_seq_len", 0) or 0),
    )
    return DatasetArtifacts(
        dataset=dataset,
        pad_id=dataset.pad_id,
        vocab_size=dataset.vocab_size,
        max_seq_len=dataset.max_input_len,
        metadata=dataset.metadata_summary,
    )


_CONTEXT_CAPACITY_KEYS: Dict[str, str] = {
    "deltaformer": "ctx_len",
    "deltanet": "ctx_len",
    "gated_deltanet": "ctx_len",
    "mom": "ctx_len",
    "log_linear_mamba": "ctx_len",
    "retnet": "ctx_len",
    "rwkv": "ctx_len",
}


def ensure_backend_context_capacity(
    backend_name: str,
    overrides: Dict[str, Any],
    min_context: int | None,
) -> Dict[str, Any]:
    """Return overrides capable of covering the dataset's longest sequence."""

    if min_context is None or min_context <= 0:
        return dict(overrides)
    adjusted = dict(overrides)
    context_key = _CONTEXT_CAPACITY_KEYS.get(backend_name)
    if context_key is not None:
        current = adjusted.get(context_key)
        if current is None or int(current) < int(min_context):
            adjusted[context_key] = int(min_context)
        return adjusted
    if backend_name == "memory_mosaic":
        current = adjusted.get("block_size")
        if current is None or int(current) < int(min_context):
            adjusted["block_size"] = int(min_context)
    return adjusted


BACKEND_SPECS: Dict[str, BackendSpec] = {
    "identity": BackendSpec(
        name="identity",
        description="Forward embedded tokens without modification.",
        defaults={"embed_dim": 256},
        builder=lambda cfg, _: IdentityBackend(embed_dim=int(cfg["embed_dim"])),
    ),
    "simple_rnn": BackendSpec(
        name="simple_rnn",
        description="Token-aligned GRU encoder emitting decoder-compatible representations.",
        defaults={
            "embed_dim": 64,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.0,
        },
        builder=lambda cfg, _: SimpleRNNEncoder(
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ),
    ),
    "compressive_transformer": BackendSpec(
        name="compressive_transformer",
        description="Compressive Transformer summarizing the prefix stream.",
        defaults={
            "embed_dim": 64,
            "hidden_dim": 128,
            "num_layers": 3,
            "heads": 4,
            "block_length": 32,
            "mem_length": 64,
            "compression_factors": 4,
            "compression_lengths": None,
            "dropout": 0.1,
        },
        builder=lambda cfg, _: CompressiveTransformerBackend(
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            heads=int(cfg["heads"]),
            block_length=int(cfg["block_length"]),
            mem_length=int(cfg["mem_length"]),
            compression_factors=cfg["compression_factors"],
            compression_lengths=cfg.get("compression_lengths"),
            dropout=float(cfg["dropout"]),
        ),
    ),
    "dnc": BackendSpec(
        name="dnc",
        description="Differentiable Neural Computer backend with slot outputs.",
        defaults={
            "embed_dim": 64,
            "segment_length": 8,
            "mem_input_dim": 32,
            "segmentation_method": "flat",
            "hidden_size": 128,
            "num_layers": 1,
            "nr_cells": 8,
            "read_heads": 4,
            "cell_size": 32,
        },
        builder=lambda cfg, _: DNCBackend(
            embed_dim=int(cfg["embed_dim"]),
            segment_length=int(cfg["segment_length"]),
            mem_input_dim=int(cfg["mem_input_dim"]),
            segmentation_method=str(cfg["segmentation_method"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            nr_cells=int(cfg["nr_cells"]),
            read_heads=int(cfg["read_heads"]),
            cell_size=int(cfg["cell_size"]),
        ),
    ),
    "stm": BackendSpec(
        name="stm",
        description="Slot-based Short-Term Memory backend.",
        defaults={
            "embed_dim": 64,
            "segment_length": 8,
            "stm_input_dim": 32,
            "segmentation_method": "avg",
        },
        builder=lambda cfg, _: STMBackend(
            embed_dim=int(cfg["embed_dim"]),
            segment_length=int(cfg["segment_length"]),
            stm_input_dim=int(cfg["stm_input_dim"]),
            segmentation_method=str(cfg["segmentation_method"]),
        ),
    ),
    "ttt": BackendSpec(
        name="ttt",
        description="Test-Time-Training backend with adaptive slots.",
        defaults={
            "embed_dim": 64,
            "num_layers": 3,
            "num_heads": 4,
            "mini_batch_size": 16,
            "window_size": 32,
            "mlp_ratio": 2,
            "attn_dropout": 0.1,
            "resid_dropout": 0.1,
            "use_gate": False,
            "ttt_variant": "linear",
        },
        builder=_build_ttt_backend,
    ),
    "transformer_pp": BackendSpec(
        name="transformer_pp",
        description="Transformer++ encoder that can act as a direct-mode backend.",
        defaults={
            "embed_dim": 96,
            "num_layers": 3,
            "num_heads": 4,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "positional_mode": "rope",
            "rotary_base": 10000.0,
            "pope_theta_base": 10000.0,
            "pope_bias_init": "zero",
            "use_flash_attention": True,
            "use_qk_norm": False,
            "qk_norm_eps": 1e-6,
        },
        builder=lambda cfg, _: TransformerPPBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]),
            mlp_ratio=int(cfg["mlp_ratio"]),
            dropout=float(cfg["dropout"]),
            positional_mode=str(cfg["positional_mode"]),
            rotary_base=float(cfg["rotary_base"]),
            pope_theta_base=float(cfg["pope_theta_base"]),
            pope_bias_init=str(cfg["pope_bias_init"]),
            use_flash_attention=_to_bool(cfg["use_flash_attention"]),
            use_qk_norm=_to_bool(cfg["use_qk_norm"]),
            qk_norm_eps=float(cfg["qk_norm_eps"]),
        ),
    ),
    "titans": BackendSpec(
        name="titans",
        description="Titans-inspired encoder supporting MAC, gated, MAL, or LMM paths.",
        defaults={
            "embed_dim": 256,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "memory_incorporation": "mal",
            "local_window_size": 64,
            "local_window_heads": 4,
            "longterm_mem_tokens": 2,
            "chunk_size": 64,
            "memory_chunk_size": None,
        },
        builder=lambda cfg, _: TitansEncoder(
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", cfg["embed_dim"])),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            memory_incorporation=str(cfg["memory_incorporation"]),
            local_window_size=int(cfg["local_window_size"]),
            local_window_heads=int(cfg["local_window_heads"]),
            longterm_mem_tokens=int(cfg["longterm_mem_tokens"]),
            chunk_size=int(cfg["chunk_size"]),
            memory_chunk_size=None
            if cfg.get("memory_chunk_size") in {None, "None", ""}
            else int(cfg["memory_chunk_size"]),
        ),
    ),
    "ttt_fast": BackendSpec(
        name="ttt_fast",
        description="Inference-only TTT stack backed by ThunderKittens/Triton kernels.",
        defaults={
            "embed_dim": 256,
            "num_layers": 2,
            "num_heads": 8,
            "mini_batch_size": 128,
            "window_size": 64,
            "mlp_ratio": 4,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "use_gate": False,
            "ttt_variant": "linear_fast",
        },
        builder=lambda cfg, _: TestTimeTrainingBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]),
            mini_batch_size=int(cfg["mini_batch_size"]),
            window_size=int(cfg["window_size"]),
            mlp_ratio=int(cfg["mlp_ratio"]),
            attn_dropout=float(cfg["attn_dropout"]),
            resid_dropout=float(cfg["resid_dropout"]),
            use_gate=bool(cfg["use_gate"]),
            ttt_variant=str(cfg["ttt_variant"]),
        ).eval(),
    ),
    "titans_external": BackendSpec(
        name="titans_external",
        description="Wrapper around lucidrains' Titans Memory-as-Context transformer (direct mode only).",
        defaults={
            "embed_dim": 256,
            "hidden_dim": 256,
            "num_layers": 2,
            "titans_num_slots": 4,
            "vocab_size": 32000,
            "pad_id": 0,
            "longterm_mem_tokens": 2,
            "chunk_size": 64,
            "local_window_heads": 4,
            "neural_memory_chunk_size": 64,
            "neural_memory_model_depth": 2,
            "neural_memory_model_expansion": 4.0,
            "dropout": 0.0,
            "ff_mult": 4.0,
            "dim_head": None,
            "persist_mem_tokens": None,
            "use_flex_attention": False,
            "sliding_window_attn": False,
        },
        builder=lambda cfg, _: TitansExternalMAC(
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", cfg["embed_dim"])),
            num_layers=int(cfg["num_layers"]),
            num_slots=int(cfg["titans_num_slots"]),
            vocab_size=int(cfg["vocab_size"]),
            pad_id=int(cfg["pad_id"]),
            longterm_mem_tokens=int(cfg["longterm_mem_tokens"]),
            chunk_size=int(cfg["chunk_size"]),
            local_window_heads=int(cfg["local_window_heads"]),
            neural_memory_chunk_size=int(cfg["neural_memory_chunk_size"]),
            neural_memory_model_depth=int(cfg["neural_memory_model_depth"]),
            neural_memory_model_expansion=float(cfg["neural_memory_model_expansion"]),
            dropout=float(cfg["dropout"]),
            ff_mult=float(cfg["ff_mult"]),
            dim_head=None if cfg.get("dim_head") in {None, "None", ""} else int(cfg["dim_head"]),
            persist_mem_tokens=None if cfg.get("persist_mem_tokens") in {None, "None", ""} else int(cfg["persist_mem_tokens"]),
            use_flex_attention=bool(cfg["use_flex_attention"]),
            sliding_window_attn=bool(cfg["sliding_window_attn"]),
        ),
    ),
    "mamba": BackendSpec(
        name="mamba",
        description="Mamba2 stack producing decoder-aligned token states.",
        defaults={
            "embed_dim": 64,
            "num_layers": 3,
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.1,
            "headdim": 64,
        },
        builder=_build_mamba_backend,
    ),
    "memory_mosaic": BackendSpec(
        name="memory_mosaic",
        description="Memory Mosaic encoder with persistent memories and slot projection.",
        defaults={
            "embed_dim": 128,
            "n_layer": 6,
            "n_head": 8,
            "pmem_size": 2048,
            "pmem_count": 2,
            "dropout": 0.1,
            "block_size": 512,
            "leaky_cuda": False,
        },
        builder=lambda cfg, _: MemoryMosaicBackend(
            embed_dim=int(cfg["embed_dim"]),
            n_layer=int(cfg["n_layer"]),
            n_head=int(cfg["n_head"]),
            pmem_size=int(cfg["pmem_size"]),
            pmem_count=int(cfg["pmem_count"]),
            dropout=float(cfg["dropout"]),
            block_size=int(cfg["block_size"]),
            leaky_cuda=bool(cfg["leaky_cuda"]),
        ),
    ),
    "rwkv": BackendSpec(
        name="rwkv",
        description="Flash-linear RWKV7 backend emitting hidden-state slots.",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ffn_mult": 4,
            "ctx_len": 256,
            "num_heads": None,
            "head_dim": None,
            "autocast_dtype": torch.bfloat16,
            "vocab_size": 32000,
        },
        builder=lambda cfg, _: RWKVBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ffn_mult=int(cfg["ffn_mult"]),
            ctx_len=int(cfg["ctx_len"]),
            num_heads=None
            if cfg.get("num_heads") in {None, "None", ""}
            else int(cfg["num_heads"]),
            head_dim=None
            if cfg.get("head_dim") in {None, "None", ""}
            else int(cfg["head_dim"]),
            autocast_dtype=cfg.get("autocast_dtype"),
        ),
    ),
    "log_linear_mamba": BackendSpec(
        name="log_linear_mamba",
        description="Log-linear Mamba backend (LogLinear Mamba2).",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "head_dim": 64,
            "expand": 2,
            "n_groups": 1,
            "conv_kernel": 4,
            "chunk_size": 64,
            "state_size": 32,
            "use_bias": False,
            "use_conv_bias": True,
            "vocab_size": 32000,
            "allow_unstable_fused_kernel": False,
            "attn_mode": None,
        },
        builder=_build_log_linear_mamba_backend,
    ),
    "deltanet": BackendSpec(
        name="deltanet",
        description="DeltaNet backend from flash-linear-attention.",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "hidden_ratio": 4.0,
            "num_heads": 4,
            "num_kv_heads": None,
            "qk_norm": "l2",
            "qkv_bias": False,
            "rope_theta": 10000.0,
            "attn_mode": "chunk",
            "vocab_size": 32000,
            "expand_k": 1.0,
            "expand_v": 1.0,
            "use_gate": True,
            "use_beta": True,
            "use_short_conv": True,
            "use_output_norm": True,
            "conv_size": 4,
            "qk_activation": "silu",
            "autocast_dtype": "bf16",
        },
        builder=lambda cfg, _: DeltaNetBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            hidden_ratio=float(cfg["hidden_ratio"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=None if cfg.get("num_kv_heads") in {None, "None"} else int(cfg["num_kv_heads"]),
            qk_norm=str(cfg["qk_norm"]),
            qkv_bias=bool(cfg["qkv_bias"]),
            rope_theta=float(cfg["rope_theta"]),
            attn_mode=str(cfg["attn_mode"]),
            vocab_size=int(cfg["vocab_size"]),
            expand_k=float(cfg["expand_k"]),
            expand_v=float(cfg["expand_v"]),
            use_gate=bool(cfg["use_gate"]),
            use_beta=bool(cfg["use_beta"]),
            use_short_conv=bool(cfg["use_short_conv"]),
            use_output_norm=bool(cfg["use_output_norm"]),
            conv_size=int(cfg["conv_size"]),
            qk_activation=str(cfg["qk_activation"]),
            autocast_dtype=cfg.get("autocast_dtype"),
        ),
    ),
    "gated_deltanet": BackendSpec(
        name="gated_deltanet",
        description="Gated DeltaNet backend (FLA implementation).",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "num_heads": 4,
            "head_dim": 64,
            "num_v_heads": None,
            "expand_v": 2.0,
            "use_gate": True,
            "use_short_conv": True,
            "allow_neg_eigval": False,
            "conv_size": 4,
            "attn_mode": "chunk",
            "vocab_size": 32000,
            "autocast_dtype": None,
        },
        builder=lambda cfg, _: GatedDeltaNetBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            num_heads=int(cfg["num_heads"]),
            head_dim=int(cfg["head_dim"]),
            num_v_heads=None if cfg.get("num_v_heads") in {None, "None"} else int(cfg["num_v_heads"]),
            expand_v=float(cfg["expand_v"]),
            use_gate=bool(cfg["use_gate"]),
            use_short_conv=bool(cfg["use_short_conv"]),
            allow_neg_eigval=bool(cfg["allow_neg_eigval"]),
            conv_size=int(cfg["conv_size"]),
            attn_mode=str(cfg["attn_mode"]),
            vocab_size=int(cfg["vocab_size"]),
            autocast_dtype=cfg.get("autocast_dtype"),
        ),
    ),
    "deltaformer": BackendSpec(
        name="deltaformer",
        description="DeltaFormer backend (flash-linear-attention).",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "hidden_ratio": 4,
            "num_heads": 4,
            "num_kv_heads": None,
            "attn_mode": "chunk",
            "qkv_bias": False,
            "qk_norm": False,
            "rope_theta": 10000.0,
            "vocab_size": 32000,
            "autocast_dtype": None,
        },
        builder=lambda cfg, _: DeltaFormerBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            hidden_ratio=float(cfg["hidden_ratio"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=None
            if cfg.get("num_kv_heads") in {None, "None"}
            else int(cfg["num_kv_heads"]),
            attn_mode=str(cfg["attn_mode"]),
            qkv_bias=bool(cfg["qkv_bias"]),
            qk_norm=bool(cfg["qk_norm"]),
            rope_theta=float(cfg["rope_theta"]),
            vocab_size=int(cfg["vocab_size"]),
            autocast_dtype=None if cfg.get("autocast_dtype") in {None, "None", ""} else cfg["autocast_dtype"],
        ),
    ),
    "retnet": BackendSpec(
        name="retnet",
        description="RetNet backend (flash-linear-attention retention model).",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "expand_k": 1.0,
            "expand_v": 2.0,
            "hidden_ratio": 4.0,
            "intermediate_size": None,
            "num_heads": 4,
            "num_kv_heads": None,
            "feature_map": None,
            "hidden_act": "swish",
            "use_short_conv": False,
            "conv_size": 4,
            "use_output_gate": True,
            "attn_mode": "chunk",
            "elementwise_affine": True,
            "norm_eps": 1e-6,
            "fuse_norm": True,
            "fuse_swiglu": True,
            "fuse_cross_entropy": True,
            "fuse_linear_cross_entropy": False,
            "use_l2warp": False,
            "vocab_size": 32000,
            "autocast_dtype": None,
        },
        builder=lambda cfg, _: RetNetBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            expand_k=float(cfg["expand_k"]),
            expand_v=float(cfg["expand_v"]),
            hidden_ratio=float(cfg["hidden_ratio"]),
            **(
                {}
                if cfg.get("intermediate_size") in {None, "None"}
                else {"intermediate_size": int(cfg["intermediate_size"])}
            ),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=None
            if cfg.get("num_kv_heads") in {None, "None"}
            else int(cfg["num_kv_heads"]),
            feature_map=None
            if cfg.get("feature_map") in {None, "None", ""}
            else str(cfg["feature_map"]),
            hidden_act=str(cfg["hidden_act"]),
            use_short_conv=_to_bool(cfg["use_short_conv"]),
            conv_size=int(cfg["conv_size"]),
            use_output_gate=_to_bool(cfg["use_output_gate"]),
            attn_mode=str(cfg["attn_mode"]),
            elementwise_affine=None
            if cfg.get("elementwise_affine") in {None, "None"}
            else _to_bool(cfg["elementwise_affine"]),
            norm_eps=float(cfg["norm_eps"]),
            fuse_norm=_to_bool(cfg["fuse_norm"]),
            fuse_swiglu=_to_bool(cfg["fuse_swiglu"]),
            fuse_cross_entropy=_to_bool(cfg["fuse_cross_entropy"]),
            fuse_linear_cross_entropy=_to_bool(cfg["fuse_linear_cross_entropy"]),
            use_l2warp=_to_bool(cfg["use_l2warp"]),
            vocab_size=int(cfg["vocab_size"]),
        ),
    ),
    "gla": BackendSpec(
        name="gla",
        description="Flash-linear Gated Linear Attention encoder.",
        defaults={
            "embed_dim": 128,
            "num_layers": 6,
            "ctx_len": 24000,
            "num_heads": 8,
            "num_kv_heads": None,
            "hidden_ratio": 1.0,
            "feature_map": None,
            "attn_mode": "chunk",
            "autocast_dtype": None,
        },
        builder=lambda cfg, _: GLABackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=None
            if cfg.get("num_kv_heads") in {None, "None"}
            else int(cfg["num_kv_heads"]),
            hidden_ratio=float(cfg["hidden_ratio"]),
            feature_map=None
            if cfg.get("feature_map") in {None, "None", ""}
            else str(cfg["feature_map"]),
            attn_mode=str(cfg["attn_mode"]),
            autocast_dtype=cfg.get("autocast_dtype"),
        ),
    ),
    "mom": BackendSpec(
        name="mom",
        description="Mixture-of-Memory backend (flash-linear-attention MoM).",
        defaults={
            "embed_dim": 128,
            "num_layers": 4,
            "ctx_len": 512,
            "num_heads": 4,
            "head_dim": 64,
            "num_memories": 4,
            "topk": 2,
            "capacity": 1.0,
            "use_layer_wise_balance": True,
            "aux_loss_scale": 0.01,
            "shared_mem": True,
            "single_kv_proj": False,
            "mom_backend": "gated_deltanet",
            "attn_mode": "chunk",
            "mode": "chunk",
            "vocab_size": 32000,
            "autocast_dtype": None,
        },
        builder=lambda cfg, _: MoMBackend(
            embed_dim=int(cfg["embed_dim"]),
            num_layers=int(cfg["num_layers"]),
            ctx_len=int(cfg["ctx_len"]),
            num_heads=int(cfg["num_heads"]),
            head_dim=int(cfg["head_dim"]),
            num_memories=int(cfg["num_memories"]),
            topk=int(cfg["topk"]),
            capacity=float(cfg["capacity"]),
            use_layer_wise_balance=bool(cfg["use_layer_wise_balance"]),
            aux_loss_scale=float(cfg["aux_loss_scale"]),
            shared_mem=bool(cfg["shared_mem"]),
            single_kv_proj=bool(cfg["single_kv_proj"]),
            mom_backend=str(cfg["mom_backend"]),
            attn_mode=str(cfg["attn_mode"]),
            vocab_size=int(cfg["vocab_size"]),
            mode=str(cfg["mode"]),
            autocast_dtype=cfg.get("autocast_dtype"),
        ),
    ),
}


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "file": DatasetSpec(
        name="file",
        description="Load queries by parsing a JSON manifest of videos.",
        defaults={
            "num_videos": -1,
            "unique_sequences": 5,
            "token_offset": TOKEN_OFFSET,
            "task": "membership",
            "cont_len": 3,
            "sequence_key": "",
            "manifest_root": "",
            "max_seq_len": 0,
            "vocab_size": 16,
        },
        builder=_build_sequence_dataset,
        required_keys=("path",),
    ),
    "synthetic": DatasetSpec(
        name="synthetic",
        description="Generate purely synthetic continuation queries.",
        defaults={
            "num_sequences": 64,
            "seq_len": None,
            "seq_len_min": 32,
            "seq_len_max": 32,
            "unique_sequences": 50,
            "vocab_size": 32,
            "seed": 0,
            "task": "membership",
            "cont_len": 3,
        },
        builder=_build_synthetic_dataset,
    ),
    "video_features": DatasetSpec(
        name="video_features",
        description="Consume precomputed frame embeddings via an embedding manifest.",
        defaults={
            "task": "membership",
            "max_videos": -1,
            "cont_len": 6,
            "max_seq_len": 0,
            "options_per_query": 0,
        },
        builder=_build_feature_sequence_dataset,
        required_keys=("manifest",),
    ),
}


def backend_choices() -> Iterable[str]:
    return BACKEND_SPECS.keys()


def dataset_choices() -> Iterable[str]:
    return DATASET_SPECS.keys()


def build_backend(name: str, overrides: Dict[str, Any]) -> tuple[MemoryBackend, Dict[str, Any]]:
    if name not in BACKEND_SPECS:
        raise KeyError(f"Unknown backend {name}")
    return BACKEND_SPECS[name].materialize(overrides)


def build_dataset(name: str, overrides: Dict[str, Any]) -> DatasetArtifacts:
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset {name}")
    return DATASET_SPECS[name].materialize(overrides)
