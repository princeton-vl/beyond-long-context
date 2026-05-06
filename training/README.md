# QA Ego Memory Backends

End-to-end training playground for decoder-only QA models backed by a variety of memory modules. The repo now exposes a single configurable entrypoint (`main.py`) that can spin up PyTorch Lightning runs with synthetic or file-backed datasets, while `tests/` contains the smoke suites that cover every backend and the sequence-processing utilities.

> *FLOPs counts for evaluation are produced by the closed-form predictor in the eval suite at `../flops_estimator/` (see the root README); training-time FLOP measurement utilities have been removed from this README's recipes.*

## Environment

This README assumes you have already provisioned a Python env with the dependencies
listed in `pyproject.toml`. The author's setup uses a uv-managed conda environment;
any environment with the listed packages should work. Cluster-specific activation
paths in launchers/scripts have been replaced with environment variable hooks:

- `CONDA_ACTIVATE` – path to a `conda` activate script (e.g. `/path/to/miniforge3/bin/activate`).
- `CONDA_ENV` – name of the conda env that exposes `uv` (defaults to `uv` when sourced).
- `QA_TEMP_ROOT` – base directory where per-run temp folders (TMPDIR) are created;
  if unset, defaults to `<system-tmp>/calibrated-temp` (e.g. `/tmp/calibrated-temp`).

Typical activation pattern:

```bash
# Activate your training env (e.g., a uv-managed conda env).
# See pyproject.toml for the dependency set.
export CONDA_ACTIVATE=/path/to/miniforge3/bin/activate
export CONDA_ENV=uv
source "$CONDA_ACTIVATE"
conda activate "$CONDA_ENV"
uv run <COMMAND>
```

Do not install additional packages or alter environment variables—surface configuration issues instead of muting them.

Temporary scratch space is isolated per run via `--temp-root`. The default is
`$QA_TEMP_ROOT/calibrated-temp` if `QA_TEMP_ROOT` is set, otherwise
`<system-tmp>/calibrated-temp` (e.g. `/tmp/calibrated-temp`).
The CLI verifies that location can host UNIX sockets and falls back to a short path
under the system temp directory when the requested base lives on an unsupported
filesystem. Override `--temp-root` directly (or set `QA_TEMP_ROOT`) if you need a
different scratch volume.
All SLURM launchers under `slurm_scripts/low_entropy/` and `slurm_scripts/pure_synthetic/` now pin
`--backend-option num_layers=3` so continuation and membership sweeps match the current three-layer
baseline. Update any custom variants in those folders if you forked them before this change so they
continue to request the correct depth when resubmitting jobs.
Those launchers also renamed their WandB surfaces from the legacy `*-seq32-1024[-fixed]-v3` pattern
to `*-3layer-32-1024[-fixed]` so dashboards reflect the active configuration; adjust downstream
filters if you bookmarked the old tags or project names.
Low-entropy launchers additionally standardize on `--weight-decay 0.01`, and the Titans variants in
both the low-entropy and pure-synthetic trees run with `--gradient-clip-val 0.5` plus learning rates
set to 50% of the other backends at each model size for stability. The legacy
`transformer_pp_pope.sh` jobs have been retired entirely in favor of `gla.sh` wrappers, so every
tier now exposes a flash-linear-attention GLA template alongside the other FLA stacks.
All SLURM launchers now source `slurm_scripts/common_env.sh`, which provisions a per-job runtime
root under `/tmp/qaego4dv2/job-<id>` and, if the node's `/tmp` is exhausted or unwritable,
automatically falls back to `$QA_FALLBACK_TMP/qaego4dv2/tmp/job-<id>` (set
`QA_FALLBACK_TMP` to a writable scratch volume on your cluster). The helper wires `TMPDIR`
and every cache-heavy variable (UV, Triton, TorchInductor, WandB's `XDG_CACHE_HOME`) into that
runtime root before invoking `main.py --temp-root "$TMPDIR"`, so multiprocessing sockets,
`pymp-*` folders, and compiler artifacts are scoped to the job and swept on exit. `/tmp` is
typically saturated on these clusters, so the helper now prefers the repo-local scratch first;
export `QA_FORCE_LOCAL_TMP=0` before sourcing the script if you explicitly want it to try `/tmp`
before falling back.

## Video Embedding Pipeline

The legacy monolithic `scripts/build_video_embeddings.py` flow has been replaced with a sharded pipeline that keeps
GPUs saturated while allowing independent resubmission per shard. The stages are:

1. **Plan** – normalize the manifest, resolve video paths, and write a deterministic `plan/plan.jsonl`:
   ```bash
   uv run python scripts/embed_plan.py \
     --questions <your-data-root>/questions.json \
     --output-dir <your-data-root>/embeddings/run42 \
     --shard-size 2000 --max-seq-len 6000
   ```
   The plan stores shard membership, token statistics, and the resolved root so future stages can resume without
   re-reading the raw manifest.
2. **Mean** – sample a subset of videos/frames once to build a cached zero-mean tensor (run on a single GPU):
   ```bash
   uv run python scripts/embed_mean.py \
     --output-dir <your-data-root>/embeddings/run42 --backbone dinov2-base \
     --fps 1 --batch-size 32 --device cuda:0 --sample-videos 16 --sample-frames 64 --force
   ```
3. **Shard Embedding** – launch one SLURM job per shard (or per GPU) and keep decode workers sized to the allocated
   CPUs. Jobs write `shards/shard_<idx>.json` as they progress so preempted runs can resume via `--resume`.
   ```bash
   uv run python scripts/embed_shard.py \
     --output-dir <your-data-root>/embeddings/run42 --shard-index 3 \
     --backbone dinov2-base --fps 1 --batch-size 32 --dtype bfloat16 \
     --device cuda:0 --cpu-workers 6 --prefetch-limit 8 --mean-path <your-data-root>/zero_mean.pt --resume
   ```
4. **Merge** – once every shard manifest exists, merge them plus the cached mean into the final `embedding_manifest.json`:
   ```bash
   uv run python scripts/embed_merge.py \
     --output-dir <your-data-root>/embeddings/run42 \
     --backbone dinov2-base --fps 1 --dtype bfloat16 --device cuda
   ```

GPU utilization and decode backlog are logged every minute, and `slurm_scripts/embedding_pipeline/` contains ready-made
templates for the plan/mean job, shard arrays targeting RTX 2080/3090/A6000 nodes, and the final merge step. Adjust the
array range to match the number of shards emitted by `embed_plan.py` so each job stays within the 6–8 hour window on the
weakest cards.

## Training CLI (`main.py`)

The CLI wires together dataset builders, memory backends, and the `MemoryBankDecoder`. It optimizes with AdamW and, by default, applies the same cosine-annealing-with-restarts scheduler (linear warmup + per-epoch restarts) while logging metrics via PyTorch Lightning's CSV logger. Override the scheduler with `--lr-scheduler` if you need alternative decay modes.

Basic usage (synthetic data + simple RNN backend):

```bash
uv run python main.py \
  --backend simple_rnn --backend-option num_layers=2 \
  --dataset synthetic --dataset-option num_sequences=128 \
  --dataset-option seq_len_min=16 --dataset-option seq_len_max=32 \
  --batch-size 16 --max-epochs 3 --learning-rate 3e-4
```

To monitor runs in WandB with a slower-refreshing TQDM bar:

```bash
uv run python main.py \
  --backend mamba --dataset file --dataset-option path=<your-data-root>/manifest.json \
  --batch-size 8 --wandb-project qa-ego-memory --wandb-run-name mamba-demo \
  --wandb-tag smoke --progress-refresh-rate 5

Need a local-only run? Append `--disable-wandb` and the CLI will skip initializing the WandB logger.

Gradient health tracking is now built-in: every optimizer step logs `gradients/global_norm` to both the CSV logger and WandB so you can follow exploding/vanishing patterns without adding callbacks. To dive deeper into specific subsystems, pass `--log-grad-component [LABEL=]PREFIX` (repeatable) and the CLI will emit extra metrics such as `gradients/token_embedding` or `gradients/backend` based on the prefixes you supply. The optional `LABEL=` portion controls the suffix shown in WandB; omit it to reuse the module prefix. Prefixes map directly to `MemoryBankDecoder.named_parameters()` (e.g. `token_embedding`, `norm`, `blocks.0`, `memory_backend`). Video-feature datasets can now advertise backbone dimensions that differ from `decoder_d_model`: pass `--feature-input-dim <backbone_dim>` (or let the CLI auto-detect the manifest’s `embed_dim`) and the decoder inserts a trainable projection so every backend sees decoder-sized embeddings.

Flash-linear backends (DeltaNet, GatedDeltaNet, DeltaFormer, MoM, RetNet, RWKV) now normalize `--backend-option autocast_dtype=` overrides so job scripts can pass human-readable tokens such as `bf16`, `bfloat16`, `fp16`, `torch.float16`, `fp32`, or `none`. The CLI still infers a safe default from `--precision`, but explicit overrides no longer have to reference `torch.dtype` objects.

MoM’s default template (`configs.json`) has also been tuned for faster experiments: every preset now routes at most three experts per token (`topk=3`) and enables `single_kv_proj=True` so the per-layer compute stays close to Gated DeltaNet. Override those keys per run if you need the full multi-expert capacity.

Binary runs log per-bucket coverage metrics so drops in accuracy are easy to diagnose. `val_uncertain_truth_error_pct` captures how often the model answers "yes"/"no" when the manifest labels the question as `UNCERTAIN`, while `val_option_truth_uncertain_pct` tracks false abstentions. Synthetic validation (`synthetic_val_*`) mirrors the same signals so WandB charts stay aligned across datasets.
```

### Loss Functions

`--loss-type` now toggles between the original cross-entropy objective and the Deep Gambler loss. When `deep_gambler` is selected the decoder only supervises the YES/NO logits plus the abstain token and minimizes `-log(o * p_true + p_res)` for known answers while supervising abstentions via `-log(p_res)`. `--deep-gambler-mode=adaptive` recomputes the wager per batch using `o = (m-1)*(1-acc)*(m/(m+1)) + 1` once the non-UNK batch accuracy exceeds `--deep-gambler-activation-acc` (default `0.33`); before that point it sticks with plain cross-entropy so the model has to earn the right to abstain. Tune `--deep-gambler-o` (fixed wager or adaptive baseline), `--deep-gambler-eps` (log epsilon), and the activation threshold to control how quickly the model leaves the abstain-heavy regime. Cross-entropy remains the default so existing runs keep the previous behavior unless the new flag set is provided explicitly.

### Evaluation Mode

Trained checkpoints can be scored against a questions manifest without re-running the Lightning trainer.
Switch the CLI into evaluation mode and provide the checkpoint run directory plus the manifest to audit:

```bash
uv run python main.py \
  --mode eval \
  --eval-run-dir artifacts/checkpoints/transformer_pp-video_features-membership-direct-seq1478-v33-d768-seed0 \
  --eval-checkpoint-name best.ckpt \
  --eval-manifest <your-data-root>/pattern/questions_binary.json \
  --eval-manifest-root <your-data-root>/patternvideos_gen \
  --eval-task membership \
  --eval-batch-size 64
```

Evaluation loads the saved config (backend + decoder), rebuilds the trained model on the requested device,
and iterates over every manifest question while logging per-question metadata (video index, question index,
entropy, prefix length), predictions, and correctness. A JSON summary plus the per-question JSONL live under
`artifacts/logs/eval/<run-name>/<timestamp>/`, making it easy to recompute custom metrics such as correctness by entropy
bucket or how often an uncertain answer is missed. Use `--eval-sequence-key`, `--eval-cont-len`, or
`--eval-token-offset` to override manifest defaults, and reuse `--num-workers` / `--pin-memory` to control the
evaluation DataLoader. `--eval-sequence-key` matches the dataset flag: pass a comma-separated list (e.g.,
`--eval-sequence-key S1,S2`) to concatenate multiple `sequences_used` entries, and the loader will allocate a dedicated
vocabulary band plus offset for every key in the requested order. Per-question metadata now exposes the resolved
`sequence_keys` list alongside `sequence_option_lengths` (one dict per option) so downstream aggregation can track
per-sequence lengths, prefix entropy, and accuracy within the same WandB panels.

#### Low-entropy validation sanity check

To inspect the “synthetic vs. manifest” gap for the `lowentropy-membership-64-transformer_pp_pope-s0-20260120-144625`
checkpoint, re-evaluate it on the bucketed validation manifest:

```bash
# Activate your training env first (see Environment section above).
source "$CONDA_ACTIVATE"
conda activate "$CONDA_ENV"
uv run python main.py \
  --mode eval \
  --eval-run-dir artifacts/checkpoints/low_entropy/exists/main/exist_synth_medium64_seq16-512/transformer_pp_pope/seed0/lowentropy-membership-64-transformer_pp_pope-s0-20260120-144625 \
  --eval-checkpoint-name best-overall.ckpt \
  --eval-manifest val/membership/seq2000_q32/questions.json \
  --eval-output-dir artifacts/logs/eval_lowentropy \
  --eval-batch-size 64 \
  --num-workers 8 \
  --pin-memory
```

The latest run writes `summary.json` under
`artifacts/logs/eval_lowentropy/lowentropy-membership-64-transformer_pp_pope-s0-20260120-144625/<timestamp>/` with
77.81% accuracy across 64 000 questions. Membership manifests reuse the same metadata slots as continuation manifests,
so the evaluator now downgrades the resulting prefix-length mismatches to a single warning per job instead of aborting.

#### Interactive low-entropy probe

For ad-hoc poking, use the interactive helper that samples synthetic low-entropy streams (default length 64) and lets
you toss in arbitrary membership queries:

```bash
uv run python -m scripts.interactive_low_entropy \
  --run-dir artifacts/checkpoints/low_entropy/exists/main/exist_synth_medium64_seq16-512/transformer_pp_pope/seed0/lowentropy-membership-64-transformer_pp_pope-s0-20260120-144625 \
  --checkpoint-name best-overall.ckpt
```

Inside the REPL, `n` swaps in a fresh stream, `s` prints the zero-based tokens, and `q 4 12 6 8` checks whether that
subsequence appears. The tool reports the model’s yes/no prediction with its probability plus the exact ground-truth
answer computed from the sampled stream, so you can see immediately whether a “low-entropy” failure is due to ambiguity
in the stream or a true modeling miss.

Every evaluation run now emits the coverage/abstention metrics needed for calibration studies:

- `Acc(T,h)` (top-level `accuracy`) – fraction of questions answered correctly.
- `Cov(T,h)` (`coverage`) – share of examples where the model produced a non-`bot` answer.
- `Acc_ans(T,h)` (`accuracy_when_answering`) – conditional accuracy on answered questions.
- `UA(T,h)` (`useful_answer_rate`) – fraction of prompts that received a correct non-`bot` answer (`Cov × Acc_ans`).
- `TAR` (`tar`) – `P[\hat{a}=bot | a*=bot]`, i.e. abstaining when the ground truth is “uncertain”.
- `FAR` (`far`) – `P[\hat{a}=bot | a*!=bot]`, i.e. false abstentions on answerable questions.

When probing the synthetic control stream via `--eval-random-synthetic`, add `--eval-synthetic-single-query` to force the generator to emit *one* query per sampled stream. This option treats the `--eval-random-synthetic` count as “number of questions” rather than “number of streams”, which makes it easy to sanity-check per-query accuracy without the default 32-query bundles.

`summary.json` appends a `question_bucket_metrics` block that slices those metrics along entropy tertiles, prefix-length
tertiles, and the joint `(T,h)` grid. Every row includes the question/answer counts, coverage, TAR/FAR, and the
formatted bucket label (the joint entries also list any explicit manifest `bucket_id` values). The `compression_summary`
section reports `T_alpha(h) = max{T : Acc(T,h) >= alpha}` for `alpha in {0.50, 0.70, 0.80, 0.90}` so accuracy/coverage
comparisons only need to scan the JSON—`T_0.70(h)` is the primary operating point, while the stricter thresholds are
guaranteed to be monotone non-increasing. Because the upstream manifests assume a single prefix per video, the evaluation loader now
verifies that all questions referencing the same video advertise identical `stream_prefix_length`/`entropy_prefix`
values and raises immediately if the manifest is inconsistent; prefix tertiles are therefore safe to compare against
the per-video boundaries logged during training.

### Membership bucket evaluation

The membership bucket evaluator (`scripts/run_membership_bucket_eval.py`) now probes each checkpoint twice by default: a
full batched pass across every sequential question, followed by four stratified single-question passes. The independent
mode samples 10% of the available questions for each entropy tier (low/med/high) plus a separate overall 10% slice and
re-encodes that smaller manifest with `--eval-batch-size 1`. Use `--question-modes batched,single` (or any subset) to
control which passes run. Each result row in `artifacts/eval/bucket_eval_L*.json` and the generated
`membership_bucket_accuracy.txt` carries its `mode` label (e.g., `single_low`) plus the effective evaluation batch size
and manifest provenance, and the SLURM log lines emit `mode=...` markers for every bucket stat so downstream tools can
distinguish the new subsets.

Entropy tiers are now derived by rebalancing the complete sequential question pool for each length so their mean
entropies follow a strict 1:2:4 ratio (low : medium : high) while keeping the tier cardinalities as even as possible.
This keeps the L1024/L2048/L4096 comparisons aligned—each tier carries comparable entropic content regardless of the
sequence length—so coverage/accuracy deltas can be attributed to temporal context instead of wildly different entropy
mixes. Set `FORCE_ENTROPY_TERTILES=low_boundary,high_boundary` if you need to override the automatic search for a
reproduction study.

## Synthetic Validation Suite

Synthetic builders now emit dedicated control tokens for every query boundary so backends can reliably separate the stream prefix, the query payload, and the label decision. `STREAM_QUERY_SEPARATOR` (token id `0`) appears before each query, the dedicated `LABEL_SEPARATOR` token (id `1`) is the only position where the decoder must output YES/NO/UNCERTAIN (the class id lives solely in the label tensor), and `QUERY_END_SEPARATOR` (id `2`) closes the query before the next one begins. Continuation tasks still rely on `CANDIDATE_SEPARATOR` for the prefix/candidate split, but the decoder never sees the actual YES/NO tokens in the input sequence anymore—only the control markers remain.

Query sampling for the synthetic builders now mirrors the PatternVideos generator: every question picks a span purely by position, flips a 50/50 coin to determine whether it shows the real slice or a fabricated distractor, and allows repeats. Negative membership n-grams are resampled until they are absent from the stream, while continuation negatives are drawn until they are distinct from every observed follower, so "no" answers always reflect true absences.

Membership sweeps log to the `membership-seq32-512` WandB project while continuation sweeps log to `continuation-seq32-512`; both rely on the dynamic `membership|continuation-<config>-<variant>-s<seed>` run names emitted by the launcher template so dashboards stay searchable. Log-linear Mamba launchers currently fix `state_size=64` because A40/A6000 class GPUs only expose enough shared memory for that tile size—the backend now raises immediately if the active device cannot honor the requested state size.

The repository now bundles a reproducible validation harness so every checkpoint can be benchmarked on a held-out
synthetic distribution that mirrors the `pure_synthetic` SLURM sweeps. The workflow is:

1. **Generate a manifest** – spread `num_sequences` evenly across the power-of-two buckets `16-32` ... `1024-2048`
   with exactly `queries_per_sequence` prompts per stream. Each prompt tracks the contiguous token ranges it
   “concerns” so downstream analysis can reason about individual eighths of the context window.
   ```bash
   uv run python scripts/generate_val_manifest.py \
     --task membership --num-sequences 5000 --queries-per-sequence 15 \
     --output-dir val/membership/latest
   ```
   The folder contains `questions.json`, a `bucket_summary.csv`, and a `question_spans.csv` that lists the bucket,
   stream length, and coverage spans for every query.

2. **Evaluate checkpoints** – point the evaluator at any run directory (the folder with `config.json`) and provide the
   manifest from step 1. The runner loads the model, enforces GPU execution, and writes bucket CSVs plus SVG plots.
   ```bash
   uv run python scripts/run_validation.py \
     --run-dir /path/to/run --checkpoint-name best.ckpt \
     --manifest val/membership/latest/questions.json \
     --task membership --output-dir logs/val/runs/2026-01-14
   ```
   Outputs include `bucket_metrics.csv`, `bucket_eighth_metrics.csv`, per-question predictions, and three SVGs
   (accuracy per bucket, abstention per bucket, and an accuracy heatmap over the eight context slices).

3. **Optional gambler fine-tune** – create a fresh synthetic training manifest and continue a checkpoint for a single
   epoch under Deep Gambler. The helper script reuses the stored backend hyperparameters, emits the new manifest under
   the run directory, and saves the fine-tuned checkpoint under `run_dir/finetune/run-*/gambler_last.ckpt` so it can be
   re-evaluated via step 2.
   ```bash
   uv run python scripts/finetune_with_gambler.py \
     --run-dir /path/to/run --checkpoint-name best.ckpt \
     --task membership --train-num-sequences 2000
   ```

All commands retain the repository layout: manifests live under `val/`, evaluation artifacts land in `logs/val/`, and
fine-tune outputs stay scoped to the original run directory. The manifest format is compatible with the existing
evaluation dataset loader thanks to the `extra_metadata` field that records which eighths of the stream a query covers.

Need structured slices for notebooks? Add `--eval-write-csv` and the evaluator will drop a `per_question.csv` next to the
JSON outputs. Each row captures the video/question IDs, scenario (with a `is_spatial` flag), prefix length, prefix entropy,
per-video entropy stats (empirical/LZ and analytic), predicted vs. true labels, correctness, softmax probability of the
`uncertain` token, and option-level metadata (slot, manifest index, presence flag, length). This CSV mirrors the
`per_question.jsonl` content but stays analysis-friendly for pandas, spreadsheets, or SQL imports.


Key ideas:

- `--backend` / `--dataset` pick the registry entry. Names currently include:
  - Backends: `identity`, `simple_rnn`, `compressive_transformer`, `dnc`, `stm`, `ttt`, `mamba`, `transformer_pp`, `memory_mosaic`, `rwkv`.
- Datasets: `synthetic` (randomly generated tokens), `file` (tokenized PatternVideos manifests), and `video_features` (precomputed frame embeddings). All datasets now emit binary membership or continuation queries that terminate in `YES`/`NO` labels, with the `UNCERTAIN` class reserved for Gambler fine-tunes.
- File-backed datasets now reuse the same token vocabulary as their synthetic counterparts: every manifest sequence is re-based with the global token offset so mixing data sources no longer inflates the vocab size, and continuation manifests keep the exact prefix span supplied by the manifest when constructing labels.
  - Every backend consumes the unified `SequenceInputs` dataclass emitted by the collator (`batch["sequence"]`). The legacy `split_sequence_batch` helper and the `StreamInputs`/`QueryInputs` views have been deleted—downstream tools and tests should slice stream/query regions by using `metadata[i]["stream_length"]`, the padding mask, and the label mask (`labels != IGNORE_INDEX`). The Titans family, STM, DNC, and the lucidrains Titans wrapper were the last holdouts and now follow this contract, so direct-mode/logit heads always see decoder-aligned `[batch, seq, hidden]` tensors.
  - Backends now also include `memory_mosaic`, which wraps the facebookresearch/MemoryMosaics blocks and exposes knobs such as `n_layer`, `n_head`, `pmem_size`, `pmem_count`, and `block_size`. The trainer automatically bumps `block_size` to cover the longest stream-plus-query span in each run, so manual overrides are rarely needed unless you intentionally want to shrink the receptive field.
  - Memory Mosaic blocks no longer rely on fixed `(block_size, block_size)` buffers. The leaky average kernel now materializes decay matrices dynamically and exposes a recurrent mode that matches the GPU-friendly parallel matmul, the value featurizer mirrors that interface so keys and values can stream together, and the context attention swaps the registered bias tensor for `scaled_dot_product_attention` with KV caching so direct-mode inference can operate past the training block size without recompilation.
  - `transformer_pp` (Transformer++) ships with RMSNorm layers, FlashAttention-powered multi-head attention, a configurable positional stack (`--backend-option positional_mode=rope|pope|none`), and optional query/key normalization. PoPE-specific knobs such as `pope_theta_base`, `pope_bias_init`, and the QK-normalization epsilon can be overridden via `--backend-option` as needed. When `positional_mode=pope` and FlashAttention is enabled the backend now projects each head into explicit cos/sin components before calling SDPA so PoPE benefits from the flash kernels too; this doubles the per-head width seen by FlashAttention, so keep `embed_dim / num_heads <= 128` if you want to stay on the flash fast path because larger heads automatically fall back to the dense logits implementation.
    Slot mode has been fully removed, so requesting `--backend-option memory_mode=output_slots` now raises `Slot mode is deprecated`; every backend always uses the hidden-state path and learns slots through the shared projection head.
    The CLI automatically aligns `decoder_d_model` with the backend's `embed_dim` whenever the backend consumes token embeddings, so you no longer need to pass `--decoder-d-model` manually. For projection-free backends the default (64) still applies unless you override it.
  - `ttt` now distinguishes between the trainable PyTorch variants (`linear`/`mlp`) and the ThunderKittens/Triton fast path. The new `ttt_fast` configs wire up `ttt_variant=linear_fast` for inference (see `configs.json`), while training will automatically error if you attempt to step the fast blocks. These kernels live under `calibrated_memory/backend/models/external/ttt_fast/` and expect every sequence to be padded to a multiple of `mini_batch_size` before dispatch. Build them once per machine:

    ```bash
    export THUNDERKITTENS_ROOT="$PWD/calibrated_memory/backend/models/external/ttt_fast/ThunderKittens"
    source "$CONDA_ACTIVATE" && conda activate "$CONDA_ENV"
    uv run bash -lc 'cd calibrated_memory/backend/models/external/ttt_fast/ThunderKittens/examples/ttt_linear_prefill && python setup.py install'
    uv run bash -lc 'cd calibrated_memory/backend/models/external/ttt_fast/ThunderKittens/examples/ttt_mlp_prefill && python setup.py install'
    ```

    The build expects CUDA ≥ 12.5 plus sm80+/sm89 NVCC support. Once `tk_ttt_linear_prefill` / `tk_ttt_mlp_prefill` import cleanly, the `ttt_fast` backend becomes available to the benchmark runner while the standard `ttt` backend keeps training safe by staying on the pure-PyTorch path.
  - Titans experiments now exclusively use the upstream MAC implementation (`--backend titans_external`). All configs and launchers source their overrides from `configs.json['titans_external'][embed_dim][tier]`, which expose the slot count, neural-memory depth/expansion, FF multiplier, and FlexAttention toggles required to reproduce the paper defaults. The internal MAL-only backend has been retired, so any legacy scripts that referenced `--backend titans` should be switched to `titans_external`. The lucidrains update also removed the deprecated `local_window_size` knob—launchers should omit that backend option and rely on `chunk_size` + `local_window_heads` (or `sliding_window_attn=true`) when controlling the receptive field.
  - `compressive_transformer` now exposes explicit `layers3` tiers in `configs.json` for every width so the medium benchmark mode (three decoder layers) uses the same overrides as the synthetic training jobs without relying on CLI overrides.
  - `rwkv` now wraps the flash-linear-attention RWKV7 kernels (CUDA-only). It keeps the original knobs (`embed_dim`, `num_layers`, `ffn_mult`, `ctx_len`) and also accepts `--backend-option num_heads=<heads>`, `--backend-option head_dim=<dimension>`, and `--backend-option autocast_dtype=bf16|fp16|none`. We default to bfloat16 autocast so the FLA kernels stay stable; override the dtype if you need fp16 or full precision. RWKV remains a hidden-state backend so decoder and direct execution both work without extra flags, but cross-check accuracy against the official RWKV repo if you suspect numerical differences (the upstream package emits a warning about potential deviations).
  - `log_linear_mamba` runs the LogLinearMamba2 fast path and therefore *requires* `chunk_size=64`; the CLI now raises immediately if you attempt to override it because the upstream Triton kernels only implement that tile size. The registry defaults already stay on the supported configuration so backend smokes and Hydra runs no longer fall through the `NotImplementedError` path.
  - `mom` kernels only allow chunk mode during training and silently fallback to `fused_recurrent` for short prefixes. Keep synthetic smokes and any curriculum seeds above 64 tokens (we pin the smoke script to `seq_len=96`) so the kernel keeps chunk mode active.
  - `stm` now mirrors the original SAM baseline: it emits logits directly from the STM head and only runs in direct mode, so there is no slot-output path or hidden-state projection overhead. Slots were standardized so the 64/96/128 configs respectively use (segment length, stm input dim, num slots) pairs of (48, 12, 8), (48, 16, 10), and (48, 20, 12).
  - `ttt` defaults were tightened for stream-aligned adaptation: every preset now runs with `mini_batch_size=32`, `window_size=32`, and `ttt_variant="mlp"` so short-context tuning behaves consistently across embedding widths. The windowed attention inside `TTTBlock` is causal—positive `window_size` values cap how far into the past each chunk can look while `window_size<=0` reverts to a full-history causal mask.
  - Attention-heavy backends hand their `key_padding_mask` tensors back to the SDPA kernels again instead of zeroing the projected Q/K/V vectors. That fully removes padded keys from the softmax rather than relying on zero-valued logits, so Transformer++, Titans, and TTT regain the exact masking behavior from the pre-FlashAttention migration (Flash may fall back to the math kernel when a padding mask is present, which is fine—the outputs are now correct again).
  - Slot-based `memory_mode='output_slots'` has been retired across the project (IdentityBackend is the lone debugging exception). Passing that override now exits immediately with `Slot mode is deprecated`; leave the flag unset and rely on direct-mode backends instead. Likewise, `memory_mode='hidden_state'` is now deprecated—expect a warning when legacy configs request it so you can migrate to direct-query execution before the code path disappears entirely.
  - Decoder execution mode has been removed entirely. Backends always emit query-aligned states directly, and inference surfaces `prepare_stream_state` / `answer_queries_from_state` helpers so you can encode the stream once and service many questions while the training loop keeps using the combined fast path.
- `--augment-with-synthetic=<ratio>` expands the training dataset with additional synthetic streams (ratio expresses "synthetic per real": `0.5` adds ~0.5 synthetic rows per real row, `1.0` yields a 50/50 mix, `0` disables augmentation). The helper reuses the synthetic dataset builder so the injected queries share the same task, number of questions (copied from the first manifest entry), and vocab span as the real data. Stream lengths are sampled uniformly between the observed min/max of the manifest, and the loader shuffle mixes the synthetic rows with the originals so batches stay homogenous. Augmentation is training-only, keeps the primary validation loader pure, and adds a second `synthetic_val_*` curve whenever a validation split exists so you can compare real vs. synthetic performance in WandB. (It's disabled for `video_features` datasets and for the dual spatial synthetic mode, because those require bespoke generators.)
- Validation/benchmark dashboards: keep the metric keys simple (`val_*` and `synthetic_val_*`) and build multi-line WandB charts by grouping related names. Recommended panels: (a) `val_entropy_bucket_{1,2,3}_acc` (accuracy) plus a second chart for `val_entropy_bucket_{1,2,3}_uncertain_pct`; (b) matching charts for the length buckets; (c) stacked counts for `val_entropy_bucket_{1,2,3}_video_count` and `val_length_bucket_{1,2,3}_video_count`; and (d) dual-line plots for the shared boundaries (`val_entropy_tertile_{1,2}` and `val_length_tertile_{1,2}`). Plot `synthetic_val_*` on the same accuracy/loss charts so both datasets share the validation tab without Lightning’s `/dataloader_idx_*` suffixes.
- Decoder/backprop execution no longer mix: direct mode is always active, and the CLI aligns `decoder_d_model` to the backend embed size whenever needed. Decoder-specific flags remain for compatibility but have no effect on the current implementation unless you explicitly override them for advanced use cases.
- `--val-fraction` keeps reserving 10% of each dataset for validation by default; alternatively pass `--val-set-percent` to specify the same split in percent (the flag overrides `--val-fraction`). Pair it with `--val-check-interval <fraction>` to run validation multiple times per epoch (e.g., `0.25` validates every quarter epoch).
- `--backend-option KEY=VALUE` (repeatable) overrides constructor kwargs. The CLI automatically infers `embed_dim` for every backend and mirrors that value into the decoder so you never have to juggle `--decoder-d-model`. The temp directory probe now prefers local storage (e.g. `/tmp`) when `--temp-root` points to a remote filesystem unless you explicitly opt into remote scratch with `QA_ALLOW_REMOTE_TEMP=1`.
- `--init-from <run-dir>` bootstraps a new run from an existing checkpoint directory. The CLI loads the saved `config.json`, enforces that structural knobs (backend, dataset, decoder widths, backend/dataset overrides) match the checkpointed architecture, and copies those overrides into the new launch so you don't retype them. By default it only transfers the model weights—optimizers/schedulers reinitialize unless you add `--init-load-optimizer`, in which case the run resumes from the original trainer state via Lightning's `ckpt_path`. Pick the checkpoint via `--init-checkpoint` (`best.ckpt` by default). Every initialization run writes into a fresh `<run-name>-init-<timestamp>` folder and records the parent directory/checkpoint inside `config.json.provenance`, so old checkpoints are never overwritten.
- `--lr-warmup <value>` inserts a linear warmup ahead of the decay schedule. Values between `0` and `1` now express the fraction of the **first** epoch used for a smooth ramp (e.g., `0.1` warms for the first 10% of epoch 0). Values `>=1` keep the legacy behavior of warming up for that many full epochs (clipped so they never exceed `max_epochs - 1`).
- `--warmup-first-epoch` is a convenience flag that forces a 25% fractional warmup on epoch 0 and overrides `--lr-warmup`.
- `--lr-scheduler {cosine_restart,cosine_epoch,constant}` toggles which scheduler follows warmup. `cosine_restart` preserves the legacy per-epoch restart behavior, `cosine_epoch` runs a single cosine decay over the remaining epochs, and `constant` disables decay once warmup (if any) completes.
- Continuation datasets **currently supervise the `cont_len` placeholder zeros directly with the ground-truth continuation tokens** (see `sequences/question_generator.py:162-188`). Change that behavior before launching experiments that assume blank/ignore slots, otherwise the decoder will treat those positions as normal vocab targets.
  - `--dataset-option KEY=VALUE` (repeatable) applies the same pattern for dataset constructors. For file-backed datasets you must provide `--dataset-option path=/path/to/questions_dataset.json`. By default datasets generate yes/no membership queries; pass `--dataset-option task=continuation --dataset-option cont_len=4` to switch to continuation prompts. The synthetic/question-importer code now writes manifests that follow the binary schema (`STREAM | SEP | prefix? | CANDIDATE | SEP<label>`) and propagates per-question metadata (entropy, concerned ranges, bucket id) so evaluation and logging remain aligned.
- File datasets now consume manifest-defined questions whenever they exist, so membership/continuation sweeps replay the exact prefixes/candidates captured in the PatternVideos manifests instead of sampling new subsequences. Continuation runs automatically filter to `question_type=sequential` questions and expose the per-question metadata through `__getitem__`, making it easier to line up WandB metrics with concrete video/question ids. Spatial manifest questions are now dropped for *both* membership and continuation tasks so we never mix lane-dependent labels into low-entropy sweeps unless an experiment explicitly asks for the dual-spatial synthetic pipeline.
- Additional dataset wiring:
  - `--dataset-option manifest_root=<your-data-root>/patternvideos_gen` resolves relative `video_path` / `clip_path` entries against the provided base. Leave it unset to fall back to the manifest’s directory.
  - `--dataset-option sequence_key=S1,S2` now accepts a comma-separated list so manifests with multiple `sequences_used` entries can be stacked. Each key receives its own vocabulary band + token offset and the loader concatenates the encoded streams in the provided order. Omit the flag to auto-select the single sequence with the highest-valued tokens (the prior behavior).
- `--dataset video_features --dataset-option manifest=/path/embedding_manifest.json` consumes the safetensor-based manifests emitted by the embedding builder described below. Add `--dataset-option task=continuation --dataset-option cont_len=6` when you want binary continuation prompts over frame embeddings. The loader uses each manifest question’s prefix/candidate metadata directly (including the `stream_cutoff` markers emitted by the builder).
- Synthetic query generation now enumerates candidate subsequences once (lengths remain randomized) and caches stream substrings when drawing negative options. Building a 10k×40-query membership set dropped from ~10.6s to ~6.7s, and continuation runs fell from ~38s to ~32s on the shared A40 nodes, so cranking `--dataset-option unique_sequences` no longer stalls dataset construction. The synthetic dataset defaults also emit 50 queries per stream now (was 5), so even the out-of-the-box sweeps cover the longer query budgets without extra flags.
- `--dataset-option max_seq_len=<N>` filters manifest streams whose combined token count (file datasets) or computed token budget (video_features datasets) exceeds `N`. Pair this with the embedding builder’s `--max-seq-len` flag to avoid embedding videos that would be dropped later; the embedding manifest records the original/filtered counts under `filters` so you can confirm the bound was honored downstream.
- Metadata from any dataset is summarized once and recorded in `artifacts/logs/<run-name>/dataset_metadata.json`; when WandB logging is active the same payload is attached to the run's summary so entropy stats, sequence lengths, and question distributions remain visible alongside losses.
- Synthetic datasets now record stream-length tertiles in that summary, so the `val_length_*` bucket metrics remain populated (and update dynamically when curriculum stages expand the active length range).
- Validation now reports entropy and length tertile accuracy buckets (`val/entropy_bucket_*`, `val/length_bucket_*`) along with the shared tertile boundaries (`val/entropy_tertile_*`, `val/length_tertile_*`). Buckets are computed per-video, so every question inherits the bucket of the stream it came from. Evaluation mode prints the same bucket boundaries before exiting so audit logs stay comparable to training diagnostics.
- Video embeddings: `scripts/build_video_embeddings.py` converts PatternVideos question manifests into an embedding manifest by sampling frames (default 8 FPS) and encoding them with a HuggingFace backbone (`videomae-base` by default, a ViT-style video encoder analogous to SlowFast). Example:

```bash
uv run python scripts/build_video_embeddings.py \
  --questions <your-data-root>/patternvideos_gen/runs_seq/len500/videos/questions.json \
  --output-dir <your-data-root>/qaego4dv2/embeddings/len500 \
  --backbone videomae-base --fps 8 --batch-size 4 \
  --zero-mean-frames --zero-mean-sample-videos 32 --zero-mean-sample-frames 512
```

Add `--max-seq-len <N>` when you only want to embed the manifest entries whose combined stream token count is `<= N`; the builder drops longer rows before scheduling work and records the checked/skipped counts under a `filters` key inside the resulting embedding manifest so downstream jobs can confirm the same bound.

The script stores per-video stream embeddings plus per-option clip embeddings under the output directory and writes `embedding_manifest.json` describing each query. Tensor filenames now include the `video_index`, bucket identifier, and variant (e.g., `video00001_l1000-stepf-e2-n2-det_v00.safetensors`) so every manifest row maps to a unique file. Runs will exit with `FileExistsError` if a tensor path already exists—pass `--resume` to reuse the existing tensors (the builder will rebuild the manifest in-place) or delete/move the directory before re-embedding to avoid silent overwrites. Use `--backbone dinov2-small` for frame-level ViT features or `--backbone-option hf_repo=<repo-id>` to point at a custom HuggingFace checkpoint.

Multi-GPU runs are now coordinated inside a single CLI invocation: supply `--devices cuda:0,cuda:1,cuda:2` (or `cpu,cpu` for local stress tests) and the builder spawns one worker per entry unless you override `--num-workers`. `--dispatch-mode queue` (default) feeds videos to the next available worker via a shared queue so long clips do not create tail latency, while `--dispatch-mode stride` pre-assigns every *n*-th video to worker `n` when you prefer deterministic sharding. Resume mode works across workers because only unfinished videos enter the queue, and the parent process rebuilds the manifest in-order once results stream back.

Add `--cpu-workers <N>` to dedicate CPU-only decode workers that run ffmpeg, keep the sampled frames in memory, and stream the tensors directly to the GPU embedding pool; this avoids extra disk writes when you have spare CPUs/DRAM. Bump `--prefetch-limit` to control how many decoded videos can wait in RAM at once (default 2). When the CPU pool is active the decode and embed stages overlap, letting a 32-CPU node keep multiple GPUs busy without re-reading the videos from disk.

Set `--log-interval <seconds>` (e.g., `30`) when using CPU decode workers to emit periodic throughput summaries that include the average per-video decode/embedding time and the current decoded buffer backlog. These logs make it obvious when GPUs are starving so you can rebalance worker counts or storage before a long run stalls.

Pass `--manifest-root <your-data-root>/patternvideos_gen` (or another base directory) when the manifest stores relative `video_path` / `clip_path` entries.
Pass `--zero-mean-frames` to subtract a dataset-wide mean embedding (estimated from a configurable number of sampled videos/frames via `--zero-mean-sample-videos` / `--zero-mean-sample-frames`) before writing tensors, which keeps each embedding dimension centered. Only one worker performs this estimation—pick the device explicitly via `--zero-mean-device cuda:0` if you want the sampling pass to avoid your busiest GPU, and the resulting mean is broadcast to every worker before embedding begins.
Add `--zero-mean-cache /path/to/cache.pt` (defaults to `<output-dir>/zero_mean_cache.pt`) so subsequent resume runs load the cached tensor instead of re-sampling thousands of frames. Pass `--zero-mean-cache-refresh` when you deliberately want to recompute the cache (for example, after changing the manifest or backbone). The preprocessing metadata in `embedding_manifest.json` now records whether the cache was loaded.
`question_index` entries in the manifest now indicate the earliest sequence/token position where the question can be asked; the embedding builder no longer assumes they are unique identifiers. Safetensor keys are derived from the manifest order (`question_0_opt0`, `question_1_opt0`, …) while the original `question_index` metadata is preserved for downstream datasets. Until we add split-by-index batching, every question attached to a video must report the same `question_index`; mixed values raise a `NotImplementedError` so we do not silently drop or reorder options.
Once the manifest is ready, point the CLI at it with `--dataset video_features --dataset-option manifest=/path/to/embedding_manifest.json` and the training loop will load the safetensors on demand—no additional preprocessing required.
- Dinov2 backbones now request the HuggingFace fast image processor by default; add `--backbone-option use_fast_processor=false` if you need to match old preprocessing. Sampling relies on ffmpeg/ffprobe piping, so make sure both binaries are on `PATH` when embedding large manifests.
- Checkpointing is opt-in: add `--enable-checkpoints` (and optionally `--checkpoint-monitor`, `--checkpoint-mode`, `--checkpoint-dir`) to persist the best and last checkpoints for a run. Each run writes its metadata + checkpoints under `<checkpoint-dir>/<run-name>` for later reuse. When you want to spawn a new experiment from those weights, point `--init-from` at the directory and optionally set `--init-checkpoint`/`--init-load-optimizer`—the CLI will copy the saved config, append a suffix to the new run name, and record the parent/provenance so older checkpoints stay untouched. When both the primary validation split and the synthetic validation split are active, training now emits three checkpoint files automatically: `best-val.ckpt` (best manifest validation accuracy), `best-synthetic.ckpt` (best synthetic validation accuracy), and `best-overall.ckpt` (the weighted average of the two). The `last.ckpt` artifact continues to track the latest epoch just like before.
- Decoder knobs (`--decoder-d-model`, `--decoder-num-layers`, `--decoder-nhead`, etc.) flow directly into `MemoryBankDecoder`. Use `--decoder-context-cap` if you need to raise the maximum decoder context length beyond what the dataset reports.
- MemoryBankDecoder always runs in direct mode now: backends emit query-aligned states straight into the LM head, and the CLI keeps their embedding dimension synchronized with the decoder width automatically.
- Decoder-specific knobs (layers, attention heads, rotary base, context cap, etc.) remain for compatibility but do not change the current implementation. Hidden-state mode is forced internally and slot overrides are rejected.
- Training knobs include batch sizing, validation split (`--val-fraction`), dataloader workers, gradient accumulation, precision, and acceleration settings. Set `--val-fraction 0.0` to disable validation splits.
- Pass `--deterministic` when you need bit-for-bit reproducibility across runs. By default we favor the fastest CUDA kernels, so repeated launches with the same seed may diverge slightly due to nondeterministic reductions. Every SLURM script in `slurm_scripts/sweep_sequences/` pins `--seed 0` so you always know which RNG stream produced a result.
- Logs land under `artifacts/logs/runs/` by default; pass `--log-dir` to change the root. WandB logging (project/name/tags + full CLI args in the run config) is enabled by default—use `--disable-wandb` if you need a local-only run. Metrics stream to both WandB and the on-screen TQDM bar, which now highlights train/val loss + accuracy along with yes/no prediction percentages; adjust its cadence with `--progress-refresh-rate`. Early stopping is available via `--early-stop-acc` (requires `--val-fraction > 0`); patience is optional and disabled by default so runs continue until the threshold or `--max-epochs` is reached.
- Need to append a breadcrumb to the WandB log/summary? Pass `--wandb-log-note "short description"`; the CLI records it exactly once at startup and appends a sanitized copy to the WandB run name so long-running sweeps stay searchable.
- Multiprocessing scratch sockets now live under `--temp-root` (default `<system-tmp>/calibrated-temp`, override via `QA_TEMP_ROOT` or `--temp-root`) with per-run hashed folders that auto-clean on exit, so the repo root no longer accumulates `pymp-*` artifacts from dataloader workers.
- Run names (and the default WandB run names) now follow `<backend>-<dataset>-seq<SEQ>-v<VOCAB>-d<DMODEL>-seed<SEED>`. Dataset sequence length and vocabulary always appear, while batch size and learning rate no longer clutter the label—set `--wandb-run-name` manually if you need those knobs in the title.

### Backends and Options

Each backend exposes the following common overrides (all start from small, CPU-friendly defaults):

| Backend | Purpose | Important Options |
| --- | --- | --- |
| `identity` | Pass-through embeddings for debugging | `embed_dim` |
| `simple_rnn` | GRU encoder emitting token-aligned representations | `embed_dim`, `hidden_dim`, `num_layers` |
| `compressive_transformer` | Lightweight compressive transformer | `embed_dim`, `hidden_dim`, `num_layers`, `block_length`, `mem_length`, `num_slots` |
| `dnc` | Differentiable Neural Computer | `segment_length`, `mem_input_dim`, `nr_cells`, `read_heads`, `num_slots` |
| `stm` | SAM-style short-term memory (direct mode only) | `segment_length`, `stm_input_dim`, `num_slots`, `segmentation_method` |
| `transformer_pp` | Transformer++ encoder that returns decoder-ready token states | `embed_dim`, `num_layers`, `num_heads`, `mlp_ratio`, `dropout`, `positional_mode`, `use_flash_attention`, `use_qk_norm` |
| `ttt` | Test-time training backend with attention + adapters | `num_heads`, `mini_batch_size`, `window_size`, `ttt_variant`, `num_slots` |
| `mamba` | Mamba2 stack with fused kernels | `num_layers`, `d_state`, `d_conv`, `expand`, `headdim` (requires CUDA)
| `retnet` | Flash-linear RetNet retention model with optional hybrid attention layers | `embed_dim`, `num_layers`, `num_heads`, `expand_k`, `expand_v`, `hidden_ratio`, `intermediate_size`, `conv_size`, `use_short_conv`, `use_output_gate`, `attn_mode` |

Every backend that emits a hidden width different from `decoder_d_model` now registers its projection layers as proper submodules. Those adapters are therefore optimized, checkpointed, and moved alongside the rest of the model instead of staying as one-off tensors on the CPU.

The TTT backend now enforces `window_size` via chunked attention windows instead of pre-building a dense mask, so its memory footprint scales with the local neighborhood rather than `seq_len²`. RWKV backends also accept `--backend-option vocab_size=...` so SLURM sweeps can forward dataset vocabulary sizes without tripping registry validation (the kernels continue to operate solely on decoder embeddings).

When a backend runs in `hidden_state` mode we now project hidden vectors into decoder space via multi-head slotting: each vector of width `D` spawns `ceil(max(D / decoder_dim, 1))` learned heads, ordered by the original vector order. The allocator trims from the most recent vectors when `num_slots` is smaller than the available heads, and it simply returns fewer rows when the hidden state cannot fill the requested slots. The Mamba backend also picks a compatible `headdim` automatically when the default (64) is not a clean divisor of `expand * embed_dim`; pass `--backend-option headdim=<value>` to override it manually (the value must divide `expand * embed_dim`).

### Dataset Options

| Dataset | Description | Required Overrides |
| --- | --- | --- |
| `synthetic` | Fully synthetic streams that can emit membership or continuation (prefix) queries | None (tune with `num_sequences`, `seq_len_min`, `seq_len_max`, `unique_sequences`, `vocab_size`, `seed`) |
| `file` | Parses the JSON manifests produced by QA Ego pipelines | `path=/n/.../manifest.json` (optional: `num_videos`, `unique_sequences`, `token_offset`, `max_seq_len`) |
| `video_features` | Precomputed frame embeddings loaded from safetensors manifests | `manifest=/path/embedding_manifest.json` (optional: `max_videos`, `task`, `cont_len`, `max_seq_len`) |

### Membership data regeneration pipeline

Our membership datasets now follow the Reverso preprocessing recipe:

1. **Window extraction (`scripts/prep_membership_sequences.py`)** – runs per dataset with linear NaN interpolation, non-overlapping windows whose lengths are sampled between 32 and 1024 tokens, and per-subset min–max normalization (each source listed in the pipeline is scaled individually). We typically stop once 50k–100k windows are available under `datasets/membership/<dataset>/Lmixed`.
2. **Tokenization (`scripts/tokenize_membership_sequences.py`)** – learns a frequency-aware 16-token codebook per dataset (weighted 1-D k-means over the normalized values) and emits compressed integer shards in `datasets/membership_tokens/<dataset>/Lmixed`.
3. **Question building (`scripts/build_membership_questions.py`)** – consumes the token shards, samples 32 questions per window with candidate prefixes of length 3–6, and writes per-dataset JSONL manifests.

### GiftEval tokenized-to-manifest conversion

GiftEval tokenization outputs under `<gifteval-root>/tokenized_k128_streamcap300k` are not directly loadable by the file/evaluation datasets because they only contain stream tokens plus tokenizer metadata. Use `scripts/build_gifteval_questions_manifest.py` to convert those parquets into the canonical `videos[] + questions[]` manifest format that `calibrated_memory.evaluation.dataset.build_evaluation_dataset()` expects.

By default the converter keeps each retained GiftEval series as one manifest stream. If you want the older membership-style regime, pass `--window-min-len` and `--window-max-len` to slice each series into non-overlapping token windows before question generation. That is the recommended mode when you want manifests made of 32-1024 token sequences instead of full-length GiftEval series.

If you also need to cap the final number of manifest sequences per subset, pass `--max-windows-per-subset <N>`. That cap is applied after windowing, so `N` refers to output windows/videos, not raw GiftEval series.

Recommended pipeline:

1. Tokenize GiftEval subsets into `*_k128.parquet` files (the existing `gifteval` pipeline).
2. Build a manifest for a concrete evaluation slice, usually one subset or a sampled subset of streams:

   ```bash
   python scripts/build_gifteval_questions_manifest.py \
     --input-root <gifteval-root>/tokenized_k128_streamcap300k \
     --subset PEMS04 \
      --task membership \
     --max-streams-per-subset 1024 \
     --questions-per-sequence 32 \
     --min-query-len 3 \
     --max-query-len 6 \
     --output <gifteval-root>/manifests/pems04_membership_q32.json
   ```

   Windowed 32-1024-token variant:

   ```bash
   python scripts/build_gifteval_questions_manifest.py \
     --input-root <gifteval-root>/tokenized_k128_streamcap300k \
     --subset PEMS04 \
     --task membership \
     --questions-per-sequence 32 \
     --min-query-len 3 \
     --max-query-len 6 \
     --window-min-len 32 \
     --window-max-len 1024 \
     --max-windows-per-subset 350000 \
     --output <gifteval-root>/manifests/pems04_membership_windows32_1024_q32.json
   ```

   Bulk SLURM launcher for all tokenized subsets:

   ```bash
   sbatch <gifteval-root>/run_tokenize_streamcap300k.sh
   # wait for that tokenize job to finish successfully, then:
   sbatch --dependency=afterok:<tokenize_job_id> \
     <gifteval-root>/run_build_manifests_windows32_1024_all.sh
   ```

   The intended handoff is:

   1. Submit `<gifteval-root>/run_tokenize_streamcap300k.sh`.
   2. Wait for the tokenization job to complete successfully and note its SLURM job id.
   3. Submit `<gifteval-root>/run_build_manifests_windows32_1024_all.sh` with `--dependency=afterok:<tokenize_job_id>` so manifest generation only starts after tokenization finishes.

   The manifest launcher scans all tokenized subsets under `<gifteval-root>/tokenized_k128_streamcap300k`, builds one manifest per subset, and writes them under `<gifteval-root>/manifests/windows32_1024_q32_max350k` by default. Each subset is windowed to 32-1024 tokens and capped at `350000` output windows/videos after windowing.

   The final product is that directory of per-subset manifest JSONs, for example `<gifteval-root>/manifests/windows32_1024_q32_max350k/PEMS04_membership_windows32_1024_q32_max350k.json`, which can be loaded directly by the `qaego4dv2` file-backed evaluation pipeline.

   The SLURM logs for that launcher land in `<gifteval-root>/logs/build_manifests_<jobid>.out` and `<gifteval-root>/logs/build_manifests_<jobid>.err`. The builder now logs each subset's planning phase, window reservoir phase, question-emission phase, periodic emitted-video counts, and the final output size so stalls are visible in the log instead of looking silent.

   The GiftEval manifest launchers use the existing `qaego4dv2` project uv environment at `<qaego4dv2-root>/.venv`. The lightweight Conda `uv` environment is only a bootstrap for the `uv` executable; it is not the runtime environment for the builder itself. If the preflight says `pyarrow` is missing, refresh the project environment once with:

   ```bash
   source "$CONDA_ACTIVATE"
   conda activate "$CONDA_ENV"
   cd <qaego4dv2-root>
   uv sync
   ```

   To write the manifests directly into `<gifteval-root>/manifests` instead of the default subdirectory:

   ```bash
   OUTPUT_ROOT=<gifteval-root>/manifests \
   sbatch --dependency=afterok:<tokenize_job_id> \
     <gifteval-root>/run_build_manifests_windows32_1024_all.sh
   ```

   The bulk launcher also accepts overrides such as:

   ```bash
   QUESTIONS_PER_SEQUENCE=16 MAX_WINDOWS_PER_SUBSET=200000 \
   sbatch --dependency=afterok:<tokenize_job_id> \
     <gifteval-root>/run_build_manifests_windows32_1024_all.sh
   ```

   If you want a smaller variety-first wave instead of every retained subset, use `<gifteval-root>/run_build_manifests_wave1_variety.sh`. That launcher builds only `PEMS_BAY`, `SHMETRO`, `favorita_sales`, `project_tycho`, `weather`, and `wind_power`, writing them into the same shared directory as the bulk q32 run by default: `<gifteval-root>/manifests/windows32_1024_q32_max350k`. It skips any subset whose final JSON already exists.

3. Point evaluation or file-backed training at the generated manifest:

   ```bash
   python -m calibrated_memory.valset.evaluator \
     --eval-manifest <gifteval-root>/manifests/pems04_membership_q32.json
   ```

Question semantics:

- `--task membership` asks binary subsequence-membership questions. Each candidate is a contiguous token slice. Positive questions use a slice that really appears somewhere in the stream; negative questions sample a same-length token sequence from the GiftEval codebook until it is absent from the full stream. The label therefore means “does this exact token pattern occur anywhere in the retained stream?”
- `--task continuation` asks binary next-chunk questions. Each query samples a contiguous prefix from the stream plus a fixed-length continuation candidate. Positive questions use the true continuation that followed one sampled prefix occurrence. Negative questions keep the prefix fixed but sample a continuation that is not one of the observed followers of that prefix anywhere in the stream. The label therefore means “for this exact prefix, is this candidate a real continuation observed in the stream?”
- When `--window-min-len/--window-max-len` are set, those questions are asked over each sampled non-overlapping window instead of the entire source series. The output videos then include `source_window_index` and `source_window_start` so you can map a manifest window back to the original GiftEval series span.
- The converter stores `concerned_ranges` for positive questions, per-stream empirical entropy, per-prefix empirical entropy, and a `scenario` payload carrying the GiftEval subset and original `series_index` so downstream analyses can trace each question back to its source stream.

Submit the entire chain with a single job:

```
sbatch slurm/new_membership/run_pipeline.sh electricity artifacts/manifests/membership_electricity.jsonl
sbatch slurm/new_membership/run_pipeline.sh weather artifacts/manifests/membership_weather.jsonl
```

Each pipeline job runs extraction, tokenization, and manifest generation sequentially inside one SLURM allocation and logs progress under `slurm/new_membership/logs/run_pipeline_<dataset>_<jobid>.{out,err}`.

Float shards are preserved alongside the tokenized shards unless a dataset exceeds ~20 GB of floats; after verifying the tokenizer metadata (`tokenizer.json`), you may delete the oversized float shards to save space.

Synthetic datasets now expect `--dataset-option seq_len_min=<min>` and `--dataset-option seq_len_max=<max>` so you can draw streams from a bounded range. Set the two values equal to recreate the original fixed-length behavior; the deprecated `seq_len` flag still maps to the same contract for backward compatibility. You can also enable a dual spatial variant via `--dataset-option enable_spatial_dual=true`, which fabricates two synchronized channels (default `S_tokens,S_lanes`) with per-key vocab sizes (`dual_vocab_sizes`, default `16,5`). Dual runs split their questions 50/50 between plain sequential queries and spatial ones that keep the real semantic prefix but vary the lane/position sequence—exactly one option preserves the true `(main sequence, spatial sequence)` combination, while the other options pair the same main snippet with lane tokens that never co-occurred with it anywhere in the stream. Adjust the mix with `dual_spatial_fraction` if needed.

Binary datasets always emit a single candidate per question. Membership manifests append the subsequence directly, while continuation manifests append the prefix + candidate pair followed by a YES/NO label. The optional `UNCERTAIN` logit stays reserved for Deep Gambler fine-tuning and evaluation summaries but no longer requires extra CLI knobs.

`token_offset` shifts every stream token above the reserved control tokens so YES/NO/UNCERTAIN labels and separators never collide with the manifest vocabulary. The CLI keeps the generous default (32) for membership/continuation tasks; raise it only if your manifest already consumes those ids. Manifest datasets also ignore any `truncate_len` overrides now—the full stream is always made available so you can revisit longer prefixes without retraining.

The dataset builder returns `pad_id`, `vocab_size`, and `max_seq_len` so the decoder config always stays consistent. Validation splits default to 10% of the samples (minimum one item) and use the same collator as training.

## Tests

All sanity checks now live under `tests/`:

```bash
uv run pytest
```

- `tests/test_backends.py` instantiates every registered backend and runs a short forward pass (the `mamba` backend is only exercised when CUDA kernels are available).
- `tests/test_sequences.py` covers the collator helpers that stitch/flatten queries.
- `tests/test_dataset_registry.py` validates both dataset builders and ensures the placeholder path raises loudly.

Keep exploratory outputs under `tests/output/` if you need fixtures, and never delete/disable tests to chase green runs.

## Interactive Demo

After training a run (preferably with `--enable-checkpoints`), launch the manual playground:

```bash
uv run python scripts/interactive.py --run-dir artifacts/checkpoints/<run-name> --checkpoint-name best.ckpt
```

The script loads the saved config + weights, creates random (or dataset-sampled) streams, and lets you type queries interactively. Commands such as `next`, `len <N>`, or `stream 1,2,3` change the stream; any other comma-separated input is treated as a query. For membership tasks the script prints YES/NO probabilities (along with the ground-truth answer), and for continuation tasks it prints the predicted continuation tokens.

## SLURM Scripts

Seven ready-to-run launchers live under `slurm_scripts/` (one per backend). Each script requests a single RTX 3090 for 48 hours and trains on synthetic membership streams with `seq_len_min=seq_len_max=512`, `unique_sequences=10`, checkpoints, and early stopping at 98% validation accuracy. Submit the script matching your backend (e.g. `sbatch slurm_scripts/train_identity.sh`) or tweak the headers to match your partition.

The legacy uncertain-membership sweep sets (`slurm_scripts/exist_*` and `slurm_scripts/cont_*`) now live under `old/legacy_slurm/`. Use them only when you need to reproduce the pre-refactor experiments; every new membership/continuation sweep runs out of `slurm_scripts/pure_synthetic/`. Those launchers share a consistent flag set (`seq_len_min=32`, `seq_len_max=512`, `unique_sequences=32`, `vocab_size=16`, cosine LR schedules, and timestamped checkpoint dirs) and tag every run with the backend + embedding width so WandB dashboards stay filterable.
RetNet membership jobs in that suite now bump weight decay to `1e-3` (64d) and `2.5e-3` (128d) so the heavier retention stacks stay regularized without resorting to custom schedulers. Continuation launchers in `slurm_scripts/pure_synthetic/continue/**` now advertise themselves as `simple-continue-*` in both their `#SBATCH --job-name` headers and runtime scratch dirs so queued jobs are distinguishable from membership (`simple-exist-*`) runs at a glance.

Sequence sweeps continue to log with dedicated WandB tags: `slurm_scripts/sweep_sequences/` uses `sequence-study`, and the video-feature sweeps under `slurm_scripts/sweep_videos/` carry the `video-sweep` tag. All video scripts share `--val-check-interval 1.0` so validation runs once per epoch regardless of the clip bucket.

Curriculum support is available across synthetic, file, and video datasets via `--curriculum-start <len>` and `--curriculum-target-acc <threshold>`. When both flags are set the trainer begins with sequences whose stream length is at most `curriculum-start`, watches the **training** accuracy, and doubles the active length bound every time the metric exceeds the requested threshold. Validation splits and WandB logs track the current curriculum stage (`curriculum_stage`, `curriculum_max_len`, and dataset sizes) so you can audit how each expansion affected learning. Leave the flags unset to run the traditional fixed-length sweeps.
- Those same scripts now build their command lines through a `run_args` bash array before invoking `uv run python main.py "${run_args[@]}"`. It eliminates the fragile trailing backslash style that yielded `main.py: error: unrecognized arguments: \` whenever SLURM rewrapped the line or someone edited an option mid-block, and it keeps every option/value pair grouped for quick diffing.
- Every synthetic sweep launcher requests at least 16 hours of walltime via `#SBATCH --time=16:00:00`, so medium/large backends have enough headroom to finish or trip the early-stop checks even on the slowest nodes. Shorten the limit only after verifying the backend's convergence window on your cluster.

Video-feature sweeps now follow the A40-friendly resource profile: each script in `slurm_scripts/sweep_videos/` requests 72G of RAM and caps `--batch-size` at 16 so multiple jobs can share a node without thrashing caches. When RTX 2080 nodes are the only option, launch from `slurm_scripts/sweep_videos_64_2080/` instead—those wrappers mirror the video sweep arguments but log under `_2080` suffixed folders, target `gpu:rtx_2080:1`, and force `--batch-size 4` to keep the smaller cards stable.

All video sweep launchers—including the RTX 2080 and `*_tokens` variants—now pass `--enable-checkpoints` so every run writes its best/last checkpoints alongside the logs under their existing `--checkpoint-dir` roots. Manual CLI launches remain opt-in, but the canned sweeps always keep restartable weights from now on.

## Repository Layout

```
main.py                # Training CLI
benchmark/            # Benchmark configuration / runners (was qaego4dv2.benchmark)
models/                # Backend implementations (now a proper Python package)
sequences/             # Dataset builders, collators, and question generators
training/              # Backend/dataset registries + dataloader helpers
decoder/               # MemoryBankDecoder LightningModule
metrics/               # Video metrics utilities (was qaego4dv2.metrics)
utils/                 # Shared helpers (paths, logging, etc.)
tests/                 # Pytest suite
artifacts/             # Local logs, checkpoints, and run metadata (created on demand)
```

## Notes

- The optimizer is intentionally fixed to AdamW + cosine scheduler inside `MemoryBankDecoder.configure_optimizers`. Adjust the CLI learning rate/weight decay knobs instead of swapping optimizers unless you edit the module.
- Heavy backends (`mamba`, `compressive_transformer`) default to CPU-safe dimensions; feel free to bump them via `--backend-option` when running on GPUs.
- Flash-linear-attention backends (`deltaformer`, `deltanet`, `gated_deltanet`, `log_linear_mamba`, `mom`, `retnet`) rely on Triton + FlashAttention kernels that assume every attention head is at least 16 wide. Keep `embed_dim` divisible by `num_heads` (or pick `head_dim >= 16` for the log-linear family) whenever you pass overrides; the CLI now enforces this up front, automatically mirrors the dataset’s `vocab_size` into those configs so their unused word-embedding tables don’t dominate the parameter counts, and now respects the global `--precision` flag when deciding whether their internal stacks should autocast (set `--precision 32-true` to keep the backends in fp32, or `bf16-mixed`/`16-mixed` to force bfloat16/float16). DeltaFormer now wraps the HuggingFace `DeltaFormerModel` directly in parallel flash-attention mode and only supports direct decoding—`encode_stream` intentionally raises—so keep `ctx_len` large enough for concatenated stream/query tokens and avoid mismatched `num_kv_heads`. The log-linear backend still disables the upstream fused training kernel by default (it would intermittently segfault on long RTX A6000 runs); pass `--backend-option allow_unstable_fused_kernel=true` if you explicitly need the fused implementation and are willing to risk the Triton crash. RetNet inherits the same constraints, adds retention-specific toggles (`expand_k`, `expand_v`, `use_short_conv`, `conv_size`, `use_output_gate`, `feature_map`) plus hybrid-attention overrides (`attn_mode`, `num_kv_heads`), and exposes the normalization/fusion switches (`elementwise_affine`, `norm_eps`, `fuse_norm`, `fuse_swiglu`, `use_l2warp`). Direct-mode decoding now casts backend query states back into the decoder’s dtype to avoid layer-norm mismatches when you swap precisions mid-run.
- These flash-linear backends are GPU-only: `main.py` now rewrites `--accelerator auto` to `--accelerator gpu` and aborts immediately if `torch.cuda.is_available()` is `False`. If you see the guard raise, your SLURM submission never landed on a GPU node (check the `#SBATCH --gres/--constraint` settings and make sure the script wasn’t submitted with stripped directives).
- Export `LOG_LINEAR_DEBUG=1` (and optionally `LOG_LINEAR_EPS` / `LOG_LINEAR_MAX_EXP`) to enable the same finite-value guards on the log-linear kernels that we added for DeltaFormer. With debug mode enabled the kernels raise immediately with the offending row/head indices and tensor norms, making it much easier to root-cause late-epoch NaNs. `LOG_LINEAR_MAX_EXP` (default 40.0) clamps the forget-gate exponentials in both the forward and backward loops so oversized gates don’t overflow before the softmax-equivalent reductions run, and `LOG_LINEAR_EPS` (default `1e-9`) lower-bounds every exponential so long negative-gate streaks cannot underflow to exact zero. The backend now keeps its HuggingFace encoder in fp32 even when Lightning is running bf16, and the Triton backward kernels accumulate their recurrent buffers (`dh`) in fp32 as well—both changes were required to stop the bf16 chunk kernels from returning NaNs on 4k-token runs.
- The log-linear chunk kernels also mask the hierarchical lookup table whenever the final chunk is partially filled so we never fetch level scales beyond the current sequence. Without this guard Triton would occasionally throw `cudaErrorIllegalAddress` when the `ctx_len` wasn’t a multiple of 64; the BF16 forward and backward passes now zero those rows instead and continue safely on arbitrarily long clips.
- Export `DELTAFORMER_DEBUG=1` before calling `uv run ...` whenever you need Triton-level diagnostics for these flash-linear backends. With the flag enabled the kernels in `fla/ops/deltaformer/parallel.py` inject finite-value checks around every forward/backward chunk and raise immediately with the batch/chunk metadata plus tensor stats. Leaving the flag unset keeps the default fast path unchanged. You can also widen the numerical guardrails by setting `DELTAFORMER_EPS` (default `1e-9`) to a larger value like `1e-6` so every normalization in the DeltaFormer Triton kernels adds that epsilon to the divisor before computing `log2`/`1 / rowsum`. Both flags apply globally—the same kernels are invoked inside `main.py`, PyTorch Lightning (`pl_train.py`), and the sanity scripts—so a single export covers every training/eval entrypoint.
- The current Triton kernels mirror the upstream flash-linear DeltaFormer implementation but include two local patches so we can debug long contexts: (1) the backward sweep now iterates chunks by index instead of stepping strictly by `ctx_len`, which prevents the leading remainder chunk from being skipped when sequences aren’t multiples of `C`; and (2) `_ensure_finite`/`DELTAFORMER_DEBUG` emit structured logs (row/head, tensor norms, `lse`, `beta`) any time a chunk returns NaNs/Infs. If you need to reproduce these changes elsewhere, copy `fla/ops/deltaformer/parallel.py` from this repo and ensure the same environment variables are wired in before re-running `uv run ...`.
- A verbatim copy of the patched Triton kernel lives at `models/external/deltaformer_parallel_debug.py`; keep that file synced with `.venv/.../fla/ops/deltaformer/parallel.py` whenever you upgrade the dependency so you can diff/patch offline without digging into the virtualenv. The reference (naive) implementation is still bundled upstream, but our automated equivalence test (`tests/test_deltaformer_kernel_equiv.py`) now loads both paths by importing `fla.models.deltaformer` first to resolve the circular dependency and verifies the Triton output (promoted to fp32) matches the naive PyTorch path within `1e-2` on random bf16 batches.
- Passing `--backend-option attn_mode=chunk` now forces the backend to reuse that local kernel copy at runtime, so edits under `calibrated_memory/backend/models/external/deltaformer_parallel_debug.py` immediately apply to the live Triton implementation without touching `.venv`. The chunk path also promotes the gate weights to fp32 before writing each chunk so the bf16 chunk kernels stay numerically stable.
- Long-running jobs belong in `pl_train.py` or managed Hydra configs; do **not** submit SLURM jobs from this repo.
- Temp/log directories are now managed centrally via `utils.paths`. Call `utils.paths.configure_temp_directory(...)` (as `main.py` does) whenever you spin up background workers so everything lands under `tmp/` or the automatic `/tmp/calibrated-temp` fallback instead of ad-hoc folders.
