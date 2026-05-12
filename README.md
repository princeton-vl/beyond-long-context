# Substream Recollection

Code for running the **Substream Recollection** evaluation suite: a controlled
benchmark that asks video- and text-streaming LLMs to recall facts about
sub-streams embedded inside a longer carrier sequence (or video). This repo
runs the evaluations end-to-end against open-weights checkpoints. **It does
not reproduce the paper figures** — the figure-generation code lives outside
the public release.

What's here:

- `main.py` — driver for video and sequence evaluations.
- `models/` — per-model wrappers (one per checkpoint family).
- `flops_estimator/` — closed-form, matmul-only FLOPs predictor for every
  model in the cohort.
- `extras/` — auxiliary experiment drivers (multi-question, anomaly).
- `tests/` — pytest suite covering state-resume + processor invariants.
- `training/` — vendored token-stream training playground for the
  decoder-only QA models with memory backends (see *Training the backends*).
- `data_gen/` — vendored patternvideos_gen synthetic video generator (see
  *Generating the synthetic videos*).

---

## Model cohort

Headline cohort (16 models, fully supported):

| Official name                       | `--model` flag                          | Wrapper                        |
| ----------------------------------- | --------------------------------------- | ------------------------------ |
| Qwen3-Omni (30B-A3B)                | `qwen3_omni`                            | `models/qwen3_omni.py`         |
| MiniCPM-V 4.5 (9B)                  | `minicpm-4-5`                           | `models/minicpm_v_4_5.py`        |
| MiniCPM-V 2.6 (8B)                  | `minicpm`                               | `models/minicpm_v_2_6.py`            |
| LongVILA (7B)                       | `longvila`                              | `models/longvila.py`           |
| Qwen2.5-VL (7B)                     | `qwen_full`                             | `models/qwen2_5_vl.py`    |
| Qwen3-VL (8B)                       | `qwen3_full`                            | `models/qwen3_vl.py`        |
| Qwen3-VL-Thinking (8B)              | `qwen3_full --qwen3-thinking`           | `models/qwen3_vl.py`        |
| GLM-4.5V (104B-A12B)                | `glm45v`                                | `models/glm45v.py`           |
| InternVL3.5 (8B)                    | `internvl-3-5`                          | `models/internvl_3_5.py`       |
| InternVL3.5-Thinking (8B)           | `internvl-3-5-thinking`                 | `models/internvl_3_5.py`       |
| Phi-4-MM (6B)                       | `phi_multimodal`                        | `models/phi_4_mm.py`     |
| InternVL3.5 (30B-A3B)               | `internvl-3-5-30b-a3b`                  | `models/internvl_3_5.py`       |
| InternVL3.5-30B-Thinking (30B-A3B)  | `internvl-3-5-30b-a3b-thinking`         | `models/internvl_3_5.py`       |
| MIMO-VL (7B)                        | `mimo-vl`                               | `models/mimo_vl.py`      |
| InternVL3.5 (38B)                   | `internvl-3-5-38b`                      | `models/internvl_3_5.py`       |
| InternVL3.5-Thinking (38B)          | `internvl-3-5-38b-thinking`             | `models/internvl_3_5.py`       |

### Partially supported

The wrappers below are present but were not part of the headline cohort. They
load and run, but we do not guarantee correctness or stability and have not
re-tested them after the recent cleanup. Use at your own risk.

| Official name      | `--model` flag | Wrapper                |
| ------------------ | -------------- | ---------------------- |
| M3-Agent           | `m3_agent`     | `models/m3_agent/`     |
| TimeChat-Online    | `timechat`     | `models/timechat.py`   |

---

## Setup

This repo uses two virtualenvs because two of the model families pin an older
Transformers/torch combination that conflicts with the rest of the cohort.

### Create the envs

```bash
git clone https://github.com/<your-fork>/streaming-mem-public.git
cd streaming-mem-public

# Env 1: full-stack — used for every model except MiniCPM-V 2.6.
python3.11 -m venv .venv-full-stack
source .venv-full-stack/bin/activate
pip install -e ".[full-stack]"
deactivate

# Env 2: hf450 — pinned Transformers 4.50; used for MiniCPM-V 2.6 (--model
# minicpm) only.
python3.11 -m venv .venv-hf450
source .venv-hf450/bin/activate
pip install -e ".[hf450]"
deactivate
```

The two extras are defined in `pyproject.toml` under
`[project.optional-dependencies]`. `flash-attn` is pulled in as a prebuilt
wheel; if your CUDA / torch combination doesn't match the pinned wheel, edit
the URL in `pyproject.toml` before installing.

### Which env runs which model

| Env                | Models                                           |
| ------------------ | ------------------------------------------------ |
| `.venv-hf450`      | `--model minicpm` (MiniCPM-V 2.6 only)           |
| `.venv-full-stack` | every other `--model` flag listed above          |

Activate the appropriate venv before invoking `main.py`. The model wrapper
will fail loudly at import time if the wrong env is active.

### A word on GPU placement

`main.py` and the model wrappers use HuggingFace `accelerate` device-map logic
internally. **Do not set `CUDA_VISIBLE_DEVICES` manually** — the wrappers
expect to see every visible GPU and will partition the model across them.
Restrict GPU visibility at the SLURM / container level instead (`--gres=gpu:N`
or equivalent).

---

## Dataset

The evaluation manifest and pre-extracted videos live on HuggingFace:

> `https://huggingface.co/datasets/anonstreammem/substream-recollection`

The dataset ships four configs, one per evaluation family. Each is a
self-contained directory with:

- `manifest.json` — nested `{"videos": [{"video_path": ..., "questions": [...]}]}`
  consumable by `main.py` directly. Questions are grouped into buckets keyed
  by carrier length and entropy regime
  (e.g. `UNIFORM_EVAL_L008_ELOW`, `UNIFORM_EVAL_L032_EHIGH`) for the synthetic
  families; natural-video uses `nat_<N>_frames` keys (see mapping below).
- `questions.parquet` and `questions.json` — flat copies of the same data,
  for `datasets.load_dataset(...)` users who only want to inspect rows.
- `videos/...mp4` — clips referenced from the manifest as relative paths;
  resolve them with `--asset-root <config-dir>`.

The four config directories on HuggingFace:

| Local config dir | Contents |
|---|---|
| `text/` | Text Substream Recollection (token-stream sequences, L=8…L=4096) |
| `synthetic_video/` | Synthetic-video Substream Recollection (vidgeom-rendered) |
| `natural_video/` | Natural-video benchmark (EPIC-Kitchens + SoccerNet; see `extras/NATURAL_DATASET.md`) |
| `easyhuman/` | EasyHuman diagnostic subset (text + video) |

### Fetch it

```bash
# Full local mirror (recommended). The manifest references clips by relative
# path and main.py opens them with --asset-root pointing at the config dir.
huggingface-cli download \
    anonstreammem/substream-recollection \
    --repo-type dataset \
    --local-dir ./data
```

For full eval runs use the `huggingface-cli download` path above — the
streaming `datasets` API does not give you the on-disk video files that the
model wrappers open via `decord` / `torchvision`.

---

## Quick smoke test

Three questions on the smallest carrier-length bucket with InternVL3.5 (8B).
Expected wall time on 2x A100 / H100: about 2 minutes once the checkpoint is
cached.

```bash
source .venv-full-stack/bin/activate

DATASET=./data/synthetic_video/manifest.json
ASSETS=./data/synthetic_video

mkdir -p ./logs/smoke

python main.py "$DATASET" \
    --asset-root "$ASSETS" \
    --eval-mode spatial \
    --input-mode video \
    --model internvl-3-5 \
    --enable_metrics \
    --verbose \
    --question-log-csv ./logs/smoke/iv8b_L008_ELOW.csv \
    --state-file       ./logs/smoke/iv8b_L008_ELOW.state \
    --max_frames 4100 \
    --max_tokens 4096 \
    --limit 1 \
    --limit_questions 3 \
    --bucket-filter UNIFORM_EVAL_L008_ELOW \
    --resume-state
```

---

## Running a full evaluation

A run is anchored on three things: the manifest (`json_path`), the model
flag, and the bucket filter. State and CSV files are written incrementally so
you can interrupt and resume any time with `--resume-state`.

### Video mode, single bucket, single model

```bash
python main.py "$DATASET" \
    --asset-root "$ASSETS" \
    --input-mode video --eval-mode spatial \
    --model glm45v \
    --question-log-csv ./logs/glm45v/L032_ELOW.csv \
    --state-file       ./logs/glm45v/L032_ELOW.state \
    --bucket-filter UNIFORM_EVAL_L032_ELOW \
    --max_frames 4100 --max_tokens 4096 \
    --resume-state
```

### Sequence mode (text-token streams instead of video)

```bash
python main.py "$DATASET" \
    --asset-root "$ASSETS" \
    --input-mode sequence --eval-mode sequential \
    --sequence-format comma \
    --model qwen3_full \
    --question-log-csv ./logs/qwen3vl8b/seq_L256.csv \
    --state-file       ./logs/qwen3vl8b/seq_L256.state \
    --bucket-filter UNIFORM_EVAL_L256_ELOW \
    --max_tokens 4096 \
    --resume-state
```

### Multiple buckets in one invocation

`--bucket-filter` is repeatable. A video matches if any filter is a substring
of its manifest path:

```bash
python main.py "$DATASET" --asset-root "$ASSETS" --model phi_multimodal \
    --bucket-filter UNIFORM_EVAL_L008_ELOW \
    --bucket-filter UNIFORM_EVAL_L032_ELOW \
    --bucket-filter UNIFORM_EVAL_L128_ELOW \
    --question-log-csv ./logs/phi/multibucket.csv \
    --state-file       ./logs/phi/multibucket.state \
    --resume-state
```

### Output

- `--question-log-csv` writes one row per `(video, question)` with
  `video_id, question_id, video_entropy, correct_answer, model_answer` and a
  few derived columns. Append-only, safe to tail.
- `--state-file` is a JSON document keyed by `bucket:video_index`, with a
  per-video `questions` dict and a flat `question_results` list. On
  `--resume-state`, completed `(video, question)` pairs are skipped.

Run `python main.py --help` for the full flag surface. Non-obvious flags:

- `--qwen3-thinking` — switch Qwen3-VL (`qwen3_full`) to its Thinking variant.
- `--qwen3-omni-fast-processor` — use the fast image processor for Qwen3-Omni
  if your Transformers build ships it.
- `--restart_on_oom` / `--max_oom_retries` — re-init the model and retry the
  current video on a CUDA OOM (rather than aborting the whole run).
- `--max_gpu_mem` — per-GPU checkpoint memory budget (GiB). Defaults to a
  size-based heuristic.

---

## FLOPs estimation

`flops_estimator/` ships a closed-form, matmul-only FLOPs predictor for
every model in the cohort. Constants are sourced from each model's HuggingFace
`config.json` and verified against the reference modeling code (see the
per-function `VISION-ENCODER AUDIT` blocks and `AUDITS.md`). **There are no
polynomial fits** — every number is computed analytically from the architecture.

```python
from flops_estimator.flops_all_models import MODEL_FUNCTIONS, DISPLAY_NAMES

frames = [{"height": 448, "width": 448}] * 8
fn = MODEL_FUNCTIONS["glm45v"]                  # state_key -> function
result = fn(frames, n_in_text_tokens=128, n_out_text_tokens=64)
print(DISPLAY_NAMES["glm45v"], result["total"])
```

Returned dict keys: `vision_flops`, `connector_flops`, `llm_prefill_flops`,
`llm_decode_flops`, `total` (matmul only), plus `*_elementwise` mirrors and
`total_with_elementwise`. See `flops_estimator/README.md` for the full
counting convention and per-model caveats (silent input downsampling in
Qwen3-VL / GLM-4.5V at high frame counts is the most important one).

---

## Repo layout

| Path                       | Contents                                                       |
| -------------------------- | -------------------------------------------------------------- |
| `main.py`                  | Eval driver: CLI, state manager, per-mode dispatch.            |
| `models/`                  | One file per checkpoint family; thin adapters over HF / vLLM.  |
| `frame_samplers/`          | Per-model frame samplers consumed by the wrappers.             |
| `processors/`              | Question / sequence / video preprocessing.                     |
| `datasets/`                | Manifest loader (`patternvideos_manifest.py`).                 |
| `metrics/`                 | Curve fitting + FLOPs validation helpers used by `main.py`.    |
| `flops_estimator/`     | Closed-form FLOPs predictor for the 16-model cohort.           |
| `extras/`                  | Auxiliary drivers: multi-question, anomaly eval.               |
| `tests/`                   | Pytest suite (state restore, processors, OOM restart, ...).    |
| `external/`                | Vendored bits of LongVILA / TimeChat used by their wrappers.   |
| `utils/`                   | GPU monitor, memory utils, CSV logger, video frame helpers.    |
| `training/`                | Token-stream training playground for the QA backends.          |
| `data_gen/`                | patternvideos_gen video generator that produced the dataset.   |
| `pyproject.toml`           | Defines the `full-stack` and `hf450` extras.                   |

---

## Citation

The accompanying paper is currently under double-blind review. Anonymous
placeholder bibtex:

```bibtex
@inproceedings{anonymous2026substream,
  title     = {Substream Recollection: A Controlled Benchmark for Long-Context
               Video and Sequence Streaming},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

This will be de-anonymized after acceptance.

---

## Training the backends

The `training/` subdirectory vendors the **token-stream** repo: an end-to-end
training playground for the decoder-only QA models with the memory backends
evaluated in this benchmark. It exposes a single configurable `main.py`
entrypoint that drives PyTorch Lightning runs against synthetic and
file-backed datasets, and includes helper scripts under `training/scripts/`.

The upstream training README documents a private-cluster `uv` environment;
on any other machine install the dependencies from
`training/pyproject.toml` into your own venv before running `main.py`. FLOPs
counting was removed from the training recipes — use `flops_estimator/` above
for evaluation FLOPs. The upstream README also references `tests/` and
`slurm_scripts/` directories that are *not* shipped in this snapshot.

See `training/README.md` for dataset preparation, backend registry, and CLI
flag documentation.

---

## Generating the synthetic videos

The `data_gen/` subdirectory vendors **patternvideos_gen**: a geometry-only
sequence-to-video harness used to render the synthetic videos that anchor the
benchmark's controlled-entropy buckets. It produces RGB tensors or MP4s from
template YAML scene descriptors and rule-based vocabularies, and ships a
`main.py` driver plus a `seq2vid` Python package.

See `data_gen/README.md` for the demo (`data_gen/examples/demo.py`), the
programmatic API, and the bucket plans under `data_gen/configs/` and
`data_gen/bucket_plan.txt`.

---

## License

MIT — see `LICENSE`.
