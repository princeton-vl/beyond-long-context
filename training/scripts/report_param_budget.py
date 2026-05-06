from __future__ import annotations

import torch

from calibrated_memory.training.registries import build_backend

CONFIGS = {
    "simple_rnn": {
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.05,
    },
    "compressive_transformer": {
        "embed_dim": 64,
        "hidden_dim": 64,
        "num_layers": 3,
        "heads": 4,
        "block_length": 64,
        "mem_length": 128,
        "compression_factors": 2,
        "dropout": 0.05,
        "memory_mode": "hidden_state",
        "num_slots": 64,
    },
    "dnc": {
        "embed_dim": 64,
        "segment_length": 8,
        "mem_input_dim": 64,
        "segmentation_method": "flat",
        "hidden_size": 64,
        "num_layers": 2,
        "nr_cells": 8,
        "read_heads": 2,
        "cell_size": 16,
    },
    "stm": {
        "embed_dim": 64,
        "segment_length": 4,
        "stm_input_dim": 32,
        "segmentation_method": "avg",
        "num_slots": 4,
    },
    "ttt": {
        "embed_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "mlp_ratio": 1.0,
        "attn_dropout": 0.05,
        "resid_dropout": 0.05,
        "mini_batch_size": 32,
        "window_size": 64,
        "use_gate": True,
        "ttt_variant": "linear",
    },
    "mamba": {
        "embed_dim": 64,
        "num_layers": 3,
        "d_state": 128,
        "d_conv": 2,
        "expand": 1,
        "dropout": 0.05,
    },
    "memory_mosaic": {
        "embed_dim": 64,
        "n_layer": 3,
        "n_head": 4,
        "pmem_size": 256,
        "pmem_count": 1,
        "dropout": 0.05,
        "block_size": 128,
    },
    "log_linear_mamba": {
        "embed_dim": 64,
        "num_layers": 3,
        "head_dim": 16,
        "state_size": 96,
        "expand": 1,
        "ctx_len": 4096,
        "vocab_size": 16,
    },
    "deltanet": {
        "embed_dim": 64,
        "num_layers": 3,
        "expand_k": 1.0,
        "expand_v": 1.0,
        "use_gate": True,
        "num_heads": 4,
        "vocab_size": 16,
    },
    "gated_deltanet": {
        "embed_dim": 64,
        "num_layers": 3,
        "head_dim": 16,
        "num_heads": 4,
        "expand_v": 1.0,
        "use_gate": True,
        "use_short_conv": True,
        "conv_size": 4,
        "vocab_size": 16,
    },
    "deltaformer": {
        "embed_dim": 64,
        "num_layers": 3,
        "hidden_ratio": 1.0,
        "num_heads": 4,
        "vocab_size": 16,
    },
    "mom": {
        "embed_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "head_dim": 16,
        "num_memories": 2,
        "topk": 1,
        "vocab_size": 16,
    },
    "rwkv": {
        "embed_dim": 64,
        "num_layers": 3,
        "ffn_mult": 2,
        "ctx_len": 4096,
    },
    "transformer_pp": {
        "embed_dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "mlp_ratio": 1,
        "dropout": 0.05,
        "num_slots": 64,
        "positional_mode": "rope",
        "rotary_base": 10000.0,
        "pope_theta_base": 10000.0,
        "pope_bias_init": "zero",
        "use_flash_attention": True,
        "use_qk_norm": False,
        "qk_norm_eps": 1e-6,
    },
}


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    for name, overrides in sorted(CONFIGS.items()):
        backend, _ = build_backend(name, overrides)
        params = count_params(backend)
        print(f"{name:24s}: {params / 1e3:.1f}k params")


if __name__ == "__main__":
    main()
