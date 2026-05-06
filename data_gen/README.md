# vidgeom — geometry-only sequence-to-video harness

This is a **geometry-only** (no textures) video generation harness.

- Template YAML describes the **video type** (scene), timing, vocab → procedural geometry mapping, rules, and variant generation.
- Runtime provides a **VideoJob** with 1+ sequences (`dict[str, list[str]]`).
- Output:
  - stream **RGB frames as torch tensors** (batched), or
  - write **MP4** via ffmpeg (streaming raw RGB frames to encoder).

## Quick start

### 1) Install locally (no PYTHONPATH tweaks needed)
```bash
pip install -e .
```

### 2) Run the demo
```bash
python examples/demo.py
```

It will render:
- `examples/out_conveyor_v0.mp4`, `examples/out_conveyor_v1.mp4`
- `examples/out_locker.mp4`
- `examples/out_sorting_hub.mp4`

## Programmatic use

```python
from vidgeom import load_template
from vidgeom.engine import VideoJob, instantiate
from vidgeom.sinks import render_video_to_mp4, render_video_to_tensors

tpl = load_template("examples/template_conveyor.yaml")
job = VideoJob(id="job1", sequences={"S1": list("ABBCBBABCBCBCCCCBCA")})

# Variants per job are controlled in the template YAML (variants.per_job)
instances = instantiate(tpl, job)

# Save mp4
render_video_to_mp4(instances[0], "out.mp4")

# Stream to tensors (batches)
for batch, meta in render_video_to_tensors(instances[0], batch_size=32):
    # batch: [B,3,H,W] uint8 by default
    ...
```

## Scenes included

- `conveyor_v2`
- `locker_room_v1_geom`
- `sorting_hub_v1_geom`
- `shape_flash_v1` — a minimalist scene that flashes centered sculptures on a dark background.

Conveyor renders now flow through a simplified, letter-only pipeline: every token spawns a single centered glyph on its own frame while the belts and lighting stay static. Lane sequences still control horizontal placement, but there is no scrolling conveyor to memorize—one token equals one frame everywhere (main videos, prefixes, options, and continuations).

A lane-controlled conveyor variant is available in `examples/template_conveyor_spatial.yaml`. It pairs the visible token stream with a dedicated lane-control sequence so you can ask spatial (belt-order) questions without changing the rendered token order.

For single-stream experiments, `examples/template_shape_flash.yaml` drives the `shape_flash_v1` scene and maps each token to a predefined sculpture template plus a unique palette entry. The helper scripts under `scripts/bucket_shape_flash/` (`make_bucket_sequences.sh`, `make_bucket_videos.sh`, and `make_bucket_dataset.sh`) mirror the spatial bucket workflow so you can batch-generate entire datasets for this mode.

A letter-based conveyor template lives at `examples/template_conveyor_letters.yaml`. Instead of sprites, each token renders as a crisp white glyph, keeping the rest of the conveyor settings aligned with the spatial buckets. The evaluation harness auto-regenerates two additional datasets that rely on these templates:

- `easy_test_shape_dark_backg/` – 20 short shape-flash videos with per-option clips.
- `easy_test_letters/` – 20 conveyor videos that use the letter glyph palette.

The large-scale membership buckets (`scripts/membership_bucket/*`) now produce 25k training videos split across the six entropy/length buckets in `configs/bucket_spatial_uniform_{tokens,lanes}.yaml`. Each rendered video carries 64 binary questions with a 50/50 split between spatial and sequential prompts. The evaluation counterpart (`scripts/membership_bucket_eval/*`) keeps the same per-bucket video counts as before but emits 32 questions per video so eval runs finish quickly while still covering both spatial and sequential modes.

Both datasets reuse the existing `questions.json` schema so downstream tooling (Hydra configs, evaluation scripts, etc.) needs no adjustments.

### Membership bucket golden rules

When generating membership (and continuation) buckets, every stage now enforces the following non-negotiable constraints:

1. **Negative options are novel** – any candidate labeled `answer: "no"` must be absent from every rendered sequence. Sequential distractors never repeat a token slice from the parent video, while spatial distractors reuse the same letters but reshuffle their lanes/order so the observed `(token, lane)` pair has never occurred.
2. **Clips stand alone** – every slice is rerendered from scratch with the static background, so the only thing that changes between true and false options is the requested letter sequence. Without the main video, annotators cannot tell which option is correct based on belt motion or backlog context.
3. **Metadata matches disk** – if `questions.json` references a video or clip, that file must exist and its SHA-256 fingerprint and size are recorded in `validation_manifest.json`. Missing or mismatched files raise immediately instead of silently corrupting the dataset.
4. **Fail fast** – if the renderer cannot construct a valid negative option after exhausting its retries, it raises a `RuntimeError` so the batch job stops instead of emitting broken data.
5. **Deterministic audit trail** – pass `--debug-frame-metadata` to `seq2vid.render_cli` (or the bucket scripts) to dump per-frame maps of which tokens/letters/lanes are visible. Each true and false question is re-rendered deterministically (no MP4 splicing), so the per-frame JSON proves that every token owns exactly one frame and that clips map cleanly back to their source slices.

### Questions JSON format

Each entry under `videos[].questions[]` is binary:

```json
{
  "question": "Did this sequence appear anywhere in the video?",
  "question_mode": "exists",
  "question_format": "binary_yes_no",
  "answer": "yes",
  "candidate": {
    "sequence": ["4", "7", "10", "13"],
    "clip_path": "clips/video_1_q0_true_sequential.mp4",
    "clip_start": 9.04,
    "clip_end": 12.89,
    "present": true
  },
  ...
}
```

Half of the questions are answered "yes" (the clip shows a real, in-context slice) and half "no" (the clip is a deterministic rerender of a distractor). No multi-option payloads remain.

The `seq2vid.render_cli` entry point now writes `validation_manifest.json` next to every `questions.json` (and can be disabled with `--skip-validation` for debugging). The membership and evaluation scripts keep validation enabled by default so regressions are caught before packaging.

If you want a fully bucketed workflow for the spatial scene, use the helper configs under `configs/buckets_spatial_*.yaml` together with the scripts in `scripts/spatial/` (see below). They generate the token (`S_tokens`) and lane (`S_lanes`) sequences separately, merge them into the combined layout expected by `render_cli`, and render questions with a 50% spatial mix by default.

## Sequence → video pipeline (seq2vid)

The `seq2vid` package wraps discovery-generated sequences into rendered videos with optional Q&A metadata.

Install locally (no PYTHONPATH needed):
```bash
pip install -e .
```

1) Generate sequences (JSON only):

```bash
python -m seq2vid.gen_cli \
  --out-dir runs_seq \
  --num-seqs 10 \
  --seq-lens 120 \
  --vocab-sizes 12 \
  --entropy-mins 0.15 --entropy-maxs 1.10 --ngram-max 6 --top-k 500 \
  --max-attempts 30 \
  --gen-workers 2 \
  --discover-len-mult 6 --max-rules 60 \
  --rule-mode probabilistic \
  --disable-entropy-drop-guard \
  --seed 123
```
For low-entropy experiments you can append `--disable-entropy-drop-guard` to bypass only the per-iteration entropy drop guard; the explicit `[entropy-min, entropy-max)` bounds remain enforced (and the upward guard stays active). Bucket configs expose the same switch via `disable_entropy_drop_guard: true`, which is enabled for every `_E1_` bucket.
Sequence names are auto-assigned (`SEQ_1`, `SEQ_2`, ...); `--num-seqs` is the total pool size.

2) Render videos + questions from a sequences JSON:

```bash
python -m seq2vid.render_cli \
  --template examples/template_conveyor_multi.yaml \
  --sequences-file runs_seq/sequences.json \
  --out-dir runs_render \
  --num-questions 8 \
  --num-videos 5 \
  --clip-options \
  --target-seq-lens 120,120,120 \
  --assignment-seed 42 \
  --question-min-len 3 \
  --ffmpeg-crf 30 --ffmpeg-preset slow \
  --ffmpeg-codec libx265 \
  --fps 24 \
  --question-mode continuation \
  # add --questions-only to skip rendering and only write questions.json
  # add --questions-at-end to ask all questions after each video plays
  --log-progress
```

- Templates declare required sequence IDs in `sequences: [...]`. The renderer shuffles the sequence pool (controlled by `--assignment-seed`) and consumes one entry per sequence ID per video; each sequence is used at most once. If lengths differ, an error is raised.
- `--target-seq-lens` (comma-separated; single value broadcasts) trims longer sequences to the requested length; shorter sequences raise an error.
- `--clip-options` adds MP4 clips for each candidate sequence: "yes" clips are trimmed from the main video; "no" clips are rendered by splicing the candidate into a matching context window.
- `--questions-at-end` forces all questions to reference the full video (default interspersed).
- `--ffmpeg-crf` / `--ffmpeg-preset` control libx264 compression (higher CRF or slower preset → smaller files).
- `--ffmpeg-codec` enables other encoders such as `libx265` for better compression.
- In generation, `--num-seqs` is the total pool size; names are auto-assigned.
- Entropy bounds are checked analytically in bits/symbol using the rule automaton (no LZ re-simulation); a uniform base with no rules always reports `log2(vocab_size)` bits. Generation fails fast if the bounds cannot be met within `--max-attempts`; sequence lengths/vocab sizes/entropy bounds are validated (no silent fallbacks). Top-n-grams per sequence are saved up to `--ngram-max` (configurable) with cumulative mass per n in `top_ngram_mass`. `--disable-entropy-drop-guard` relaxes only the downward guard (allowing larger entropy drops) while still enforcing the upward guard.
- Rendered `questions.json` files now expose *causal* empirical entropy (`entropy_overall.empirical_bits`) measured via a Lempel–Ziv estimator that only looks at prefix context, plus the analytic bits taken from generation (`entropy_overall.analytic_bits`). Every question’s `entropy_prefix` reports the same causal metric evaluated at `question_index + 1`. If you need to backfill older renders, run `python scripts/migrate_entropy_bits.py --render-root buckets/runs_render --sequences-root buckets/runs_seq` (add `--dry-run` first to inspect changes). The migration script recomputes causal entropy and copies analytic bits from the original sequences.
- The renderer supports `--question-mode exists` ("Did this appear?") and `--question-mode continuation` ("Does this follow the prefix?"). Every question is now binary (`question_format: "binary_yes_no"`) and provides a single `candidate` entry with the rendered clip, the token sequences, and an `answer` field (`"yes"`/`"no"`). Continuation questions still include the sampled `prefix` so downstream consumers know the context.
- Rendering question lengths are bounded by available n-gram stats and `--question-min-len`. Fakes are rendered with matching background context and clipped; full fake videos are deleted. Templates must declare `sequences: [...]`.
- Each question entry now carries `question_variant` so you can filter sequential vs spatial prompts when building downstream datasets.
- `--render-workers` runs per-video jobs in a process pool. On the test box, 10 conveyor videos (len=60) took ~62s with 1 worker vs ~13s with 8 workers.
- `--questions-only` skips video/clip rendering and only writes `questions.json` (useful for offline QA generation); `--clip-options` is ignored in that mode.
- `--sequence-source seq_id=path/to/sequences.json` lets you provide dedicated pools for each template sequence (e.g., a token stream plus a lane-control stream). When provided, seq2vid zips those pools together after the assignment shuffle, and the resulting `questions.json` includes both `sequences_file` (if any) and a `sequence_sources` map for provenance.
- `--spatial-question-fraction` (0–1) controls how often to ask spatial questions. Spatial candidates now keep the visible token slice identical *and* sample lane traces from other context windows, guaranteeing that "no" clips still show a real letter stream but with a belt ordering that has never co-occurred. If no unused pairing exists after 100 attempts, the renderer gracefully falls back to a sequential prompt rather than failing the video. Each question reports `question_variant: sequential|spatial` to make downstream splits easy.

### Spatial bucket helpers

```
scripts/spatial/make_bucket_sequences.sh   # generate tokens + lane sequences (writes S_tokens/S_lanes per bucket)
scripts/spatial/make_bucket_videos.sh      # render videos/questions with --spatial-question-fraction 0.5
scripts/spatial/make_bucket_dataset.sh     # combine manifests for evaluation
scripts/spatial/run_mini_test.sh           # end-to-end smoke test using the *_mini configs
```

By default these scripts target `configs/buckets_spatial_tokens.yaml` / `_lanes.yaml` and produce outputs in `buckets_spatial/`. Override `CONFIG_MAIN`, `CONFIG_LANES`, `OUT_DIR`, etc., to point elsewhere.

## Bucketed dataset workflow

When you want to generate the entire 122-bucket dataset, use the new wrapper scripts that sit on top of the updated CLIs:

> Entropy ranges in `configs/buckets*.yaml` are now stored in bits/symbol (`entropy.units: bits`). The four tiers use round intervals: `[0.0, 0.50)`, `[0.50, 1.10)`, `[1.10, 1.80)`, and `[1.80, 2.60)`. Legacy values from older docs were multiplied by `log2(e)` and snapped to these bounds to match the analytic estimator. Buckets can also set `disable_entropy_drop_guard: true` to bypass the per-iteration drop guard; every `_E1_` bucket does so to avoid over-pruning during extremely low-entropy sweeps.

1. **Generate sequences per bucket** — recommended via `./scripts/make_bucket_sequences.sh` (has SBATCH headers for SLURM). Set env vars like `CONFIG`, `OUT_DIR`, `MAX_ATTEMPTS`, `NO_PROGRESS`, or `INCLUDE` before running, or call the underlying Python entry point directly:

   ```bash
   python -m seq2vid.bucket_cli generate \
     --config configs/buckets.yaml \
     --out-dir runs_seq \
     --bucket-batch-size 32 \
     --gen-workers 8 \
     --bucket-write-combined \
     --log-progress
   ```

   This fills `buckets/runs_seq/<bucket_id>/sequences.json` (plus manifests) for every bucket. Buckets that fail to make progress for `NO_PROGRESS` consecutive batches are marked `status="failed"` in `buckets/runs_seq/bucket_generation_manifest.json` so downstream steps can skip them.

2. **Render videos/questions per bucket** — run `./scripts/make_bucket_videos.sh` (or `python -m seq2vid.bucket_cli render ...`). Buckets marked `status != completed` in the generation manifest are skipped automatically, and the rest produce per-bucket `videos/`, `clips/`, and `questions.json` under `buckets/runs_render/`. By default the renderer also sets `--uniform-uncertain`, meaning the “Uncertain / IDK” option is correct with the same probability as the other choices and, when that happens, every listed option is a fake sequence.

3. **Build a unified manifest** — `./scripts/make_bucket_dataset.sh` (or `python -m seq2vid.bucket_cli manifest ...`) merges the generation and render manifests into `buckets/bucket_dataset.json`, mapping each bucket/video back to its ID for evaluation.

Manifest schemas for every stage (`bucket_generation_manifest.json`, `bucket_render_manifest.json`, `bucket_dataset.json`, and `questions_dataset.json`) are described in `docs/manifest_formats.md`.

For quick smoke tests, `configs/buckets_mini.yaml` defines a handful of buckets (including one intentionally impossible entry) so you can see both successful and failed statuses; run the three scripts with `CONFIG=configs/buckets_mini.yaml` to validate your environment.

## Adding a new rule

Create a class in `vidgeom/rules.py` (or another module you import) and register it:

```python
from vidgeom.rules import Rule, register_rule

@register_rule("my_rule")
class MyRule(Rule):
    def on_token(self, ev, scene, scheduler, rng) -> bool:
        ...
        return False
```

Then add it in template YAML:

```yaml
rules:
  - name: my_rule
    params: { ... }
```

## Adding a new scene (video type)

1) Implement a class with:
- `reset(cfg, asset_store, rng)`
- `on_token(token, seq_id, t, meta)`
- `step(t, dt)`
- `draw(t) -> DrawList`
- optional `pop_events()`

2) Register it in `vidgeom/template.py` `SCENE_REGISTRY`.

## Notes

- Everything uses **normalized coordinates** (0..1), so resolution changes are automatic.
- Motion uses `dt = 1/fps`, so framerate changes stay synchronized.
- No textures are used; backgrounds and objects are procedural geometry.



## Optional: vocab fallback for unseen tokens

If your runtime sequences can contain tokens not listed in `vocab.mapping`, add:

```yaml
vocab:
  fallback:
    type: procedural.box
    params: { bevel: 0.08 }
  mapping:
    A: ...
```

Then any unseen token will be assigned a deterministic procedural shape derived from the token string.

## Letter styling

The simplified renderer derives every glyph from the template’s `vocab.token_letters` block. Customize color, outline, font scale, and stroke thickness directly in YAML:

```yaml
vocab:
  token_letters:
    alphabet: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    color: [245, 245, 245]
    outline_color: [20, 20, 24]
    font_scale: 5.4
    thickness: 14
    outline_thickness: 18
    image_size: 256
    shuffle: true
```

Tokens reuse the same letter and styling across the main video, every prefix, and every candidate clip, so annotators always see a consistent mapping from token ID → glyph.


### Continuation buckets

Continuation questions now emit binary yes/no entries just like the membership buckets, but they include two clips:

- `prefix_clip_path` / `prefix_clip_start` / `prefix_clip_end`: the rerendered footage of the prefix (exactly 4 tokens) taken from the original video.
- `candidate` (with `clip_path` etc.): the proposed continuation clip (also 4 tokens) that is either the real follow-up (`answer: "yes"`) or a rerendered distractor (`answer: "no"`).

Both prefix and continuation slices are always length 4 as enforced by the continuation scripts (see `scripts/continuation_bucket*/make_bucket_videos.sh`).

### TODO

- Add an integration test that renders a tiny continuation bucket and verifies via `*.frames.json` that the prefix clip and the candidate clips obey all golden rules (real prefix frames, 4-token length, binary yes/no answers).
- Port the frame-level validation script into an automated test module so we catch future regressions without manual inspection.


### Validation & Golden Rules

We rely on a mix of automated tests and scripted validators to enforce the “golden rules” for both membership and continuation question sets:

**Core guarantees**
1. **True clips match the source video.** Every frame in `clips/video_*_true*.frames.json` must match the corresponding frame in `videos/video_*_v0.frames.json` exactly (token IDs, letters, lanes). We check this via `_clip_frames_match_video()` in unit tests and via the validator scripts for larger buckets.
2. **Sequential falses never reappear.** Any “no” candidate for sequential questions must be a token sequence that never occurs anywhere in the main `S_tokens`. The sampler enforces this at generation time; the validation script re-checks every clip vs. the raw sequences.
3. **Spatial falses only remix lanes.** Spatial distractors reuse the same tokens as the source slice but assign them to lane/order pairs that never appeared in the video. This rule also applies to continuation questions that request a spatial variant.
4. **One token, one frame.** Main videos, prefixes, and candidate clips all render exactly the requested slice—no context padding, no repeated letters. A slice with `n` tokens always yields `n` frames.
5. **Clips are fully rerendered.** True and false options come from deterministic rerenders (never MP4 splicing), so belt lighting/backgrounds never leak extra information. Debug metadata (`--debug-frame-metadata`) records per-frame token/letter/lane assignments so you can audit any clip down to the frame.
6. **Continuation prefixes and candidates are fixed-length.** Both pieces are always length 4, and the glyph mapping stays identical across the main video, the prefix clip, and the candidate clip.

**Automated tests**
- `tests/test_render_validation.py` exercises the plumbing:
  * frame-debug serialization (`_write_frame_debug`, `_write_clip_frame_debug`)
  * validation manifest hashing (`_write_validation_manifest`)
  * the one-token-one-frame invariant (frame counts, metadata, ffprobe)
  * frame matching (`_clip_frames_match_video`)
  * sequential/spatial subset predicates (`_contains_subsequence`, `_contains_joint_subsequence`)
- We also maintain validator scripts (see `tmp/validation_check.py` in the instructions above) that load `test/large_*` buckets and ensure every clip’s frame JSON obeys the golden rules. These are not yet part of CI, so they’re listed under TODO.

**TODO**
- Promote the validator scripts into automated tests so we catch frame-level regressions without manual runs.
- Add an integration test that renders a tiny continuation bucket and verifies both the prefix clip and the candidate clip (true and false) against the source video frames.
