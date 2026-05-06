#!/usr/bin/env bash
set -euo pipefail
ATTN_DEBUG=${ATTN_DEBUG:-0}
uv run python main.py \
  --backend mamba \
    --backend-option embed_dim=64 \
    --backend-option num_layers=3 \
    --backend-option d_state=64 \
    --backend-option d_conv=4 \
    --backend-option expand=2 \
    --backend-option dropout=0.0 \
    --backend-option headdim=64 \
  --dataset synthetic \
    --dataset-option num_sequences=64 \
    --dataset-option unique_sequences=16 \
    --dataset-option seq_len_min=16 \
    --dataset-option seq_len_max=512 \
    --dataset-option task=membership \
    --dataset-option vocab_size=16 \
  --batch-size 8 \
  --learning-rate 5e-3 \
  --val-fraction 0.2 \
  --max-epochs 200 \
  --num-workers 0 \
  --pin-memory \
  --precision bf16-mixed \
  --deterministic \
  --enable-checkpoints \
  --checkpoint-dir artifacts/checkpoints/debug-overfit/mamba \
  --log-dir artifacts/logs/debug-overfit/mamba \
  --run-name debug-mamba-overfit \
  --wandb-project debug \
  --wandb-run-name debug-mamba-overfit \
  --wandb-log-note overfit_smoke \
  --seed 0 \
  --log-every-n-steps 5
