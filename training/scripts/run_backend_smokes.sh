#!/usr/bin/env bash
# Run short membership + continuation smokes for every registered backend.
# Usage: ./scripts/run_backend_smokes.sh [--timeout SECONDS] [--backends comma,list] [--tasks comma,list]
#
# Environment activation:
#   The script will source $CONDA_ACTIVATE and `conda activate $CONDA_ENV`
#   if $CONDA_ACTIVATE points to an existing file. Otherwise it assumes the
#   user has already activated an env that provides `uv`. Example:
#     export CONDA_ACTIVATE=/path/to/miniforge3/bin/activate
#     export CONDA_ENV=uv
#     ./scripts/run_backend_smokes.sh

set -euo pipefail

TIMEOUT=180
SELECTED_BACKENDS=""
SELECTED_TASKS="membership,continuation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      TIMEOUT=${2:-180}
      shift 2 || true
      ;;
    --backends)
      SELECTED_BACKENDS=${2:-}
      shift 2 || true
      ;;
    --tasks)
      SELECTED_TASKS=${2:-}
      shift 2 || true
      ;;
    *)
      echo "Unknown flag $1" >&2
      exit 1
      ;;
  esac
done

BACKENDS=(
  log_linear_mamba
  deltaformer
  mom
)

if [[ -n "$SELECTED_BACKENDS" ]]; then
  IFS=',' read -r -a CUSTOM <<<"$SELECTED_BACKENDS"
  BACKENDS=("${CUSTOM[@]}")
fi

IFS=',' read -r -a TASKS <<<"$SELECTED_TASKS"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="$REPO_ROOT/logs/backend_smokes"
mkdir -p "$LOG_DIR"

if [[ -n "${CONDA_ACTIVATE:-}" && -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}"
  conda activate "${CONDA_ENV:-uv}"
else
  echo "[run_backend_smokes] CONDA_ACTIVATE not set or missing; assuming caller already activated an env with 'uv' on PATH." >&2
fi

declare -i failures=0

run_case() {
  local backend=$1
  local task=$2
  local run_name="${backend}_${task}"
  local log_file="$LOG_DIR/${run_name}.log"
  local precision_value="32-true"
  if [[ "$backend" == "deltaformer" ]]; then
    precision_value="bf16-mixed"
  fi

  local cmd=(uv run python main.py
            --mode train
            --backend "$backend"
            --dataset synthetic
            --dataset-option num_sequences=1
            --dataset-option seq_len=16
            --dataset-option unique_sequences=2
            --dataset-option vocab_size=8
            --dataset-option task=membership
            --batch-size 2
            --max-epochs 1
            --learning-rate 1e-3
            --val-check-interval 1.0
            --precision "$precision_value"
            --num-workers 0
            --wandb-mode disabled
            --log-every-n-steps 1)

  if [[ "$task" == "continuation" ]]; then
    cmd+=(--dataset-option task=continuation --dataset-option cont_len=3)
  fi

  if [[ "$backend" == "mom" ]]; then
    cmd+=(--backend-option mode=chunk --dataset-option seq_len=96)
  fi

  printf '\n=== [%s] backend=%s task=%s (timeout=%ss) ===\n' "$(date -Is)" "$backend" "$task" | tee "$log_file"
  if timeout "${TIMEOUT}s" "${cmd[@]}" >>"$log_file" 2>&1; then
    echo "[${run_name}] SUCCESS" | tee -a "$log_file"
  else
    echo "[${run_name}] FAILED (see $log_file)" | tee -a "$log_file"
    failures+=1
  fi
}

for backend in "${BACKENDS[@]}"; do
  for task in "${TASKS[@]}"; do
    run_case "$backend" "$task"
  done
done

echo "\nCompleted with ${failures} failures (logs in $LOG_DIR)."
exit $failures
