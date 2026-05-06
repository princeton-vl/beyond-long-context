# Manifest formats

All pipeline stages emit JSON manifests so downstream steps know which buckets
succeeded, where assets live, and how to load questions. This doc summarizes the
shape of each manifest and the key fields to expect.

## 1. bucket_generation_manifest.json

Written to each sequences root (e.g. `buckets/runs_seq/` or
`buckets_spatial/runs_seq/`). Structure:

```json
{
  "buckets": [
    {
      "bucket_id": "L1000_STEPF_E2_N2_DET",
      "status": "completed",
      "skip_reason": null,
      "tokens_status": "completed",
      "lanes_status": "completed",
      "sequences_file": "buckets/.../sequences.json"
    }
  ]
}
```

- `bucket_id`: canonical bucket name from the config.
- `status`: `completed` or `incomplete` (`failed` for non-spatial runs).
- `skip_reason`: optional string describing why generation stopped.
- `tokens_status` / `lanes_status`: present in spatial runs; indicate per-stream
  status (`missing` when a stream never ran).
- `sequences_file`: absolute/relative path to the combined `sequences.json` that
  downstream tools should read.
- Additional per-bucket metadata lives in
  `buckets/.../<bucket_id>/meta.json` (counts, entropy bands, etc.).

## 2. bucket_render_manifest.json

Written to each render root (e.g. `buckets/runs_render/`). Structure:

```json
{
  "buckets": [
    {
      "bucket_id": "L1000_STEPF_E2_N2_DET",
      "sequences_file": "buckets/runs_seq/.../sequences.json",
      "render_dir": "buckets/runs_render/...",
      "step_profile": "STEPF",
      "step_range": [0.6, 1.4],
      "fps": 1,
      "num_questions": 64,
      "target_videos": 60,
      "status": "completed"
    }
  ]
}
```

- `render_dir` contains `videos/`, `clips/`, `questions.json`, and render logs.
- Timing/step fields mirror the config so evaluators can re-create schedules.
- Buckets flagged `status != completed` were skipped or failed.

## 3. bucket_dataset.json

Produced by `scripts/make_bucket_dataset.sh` (or the spatial equivalent). It
joins the generation + render manifests so evaluators can iterate bucket by
bucket:

```json
{
  "buckets": [
    {
      "bucket_id": "SPATIAL_L050_P2_EASY...",
      "sequences_file": ".../runs_seq/.../sequences.json",
      "meta": { "status": "completed", ... },
      "render": {
        "render_dir": ".../runs_render/...",
        "step_profile": "FAST",
        "num_questions": 32,
        "target_videos": 40,
        "status": "completed"
      }
    }
  ]
}
```

- `meta` is the entry pulled from the generation manifest.
- `render` is the entry pulled from the render manifest.
- Consumers can drop buckets where either `meta.status` or `render.status`
  differs from `completed`.

## 4. questions_dataset.json

Built by `scripts/make_bucket_questions.sh` (legacy) or by combining per-bucket
`questions.json` files. Top level:

```json
{
  "videos": [
    {
      "video_index": 1,
      "variant": 0,
      "video_path": ".../videos/video_1_v0.mp4",
      "sequences_used": { "S1": ["13", "1", ...] },
      "entropy_overall": {...},
      "questions": [
        {
          "question": "Did this sequence appear anywhere in the video?",
          "question_mode": "exists",
          "question_format": "binary_yes_no",
          "answer": "yes",
          "candidate": {
            "sequence": ["1", "3", "5", "2"],
            "clip_path": "clips/bucket/video_1_q3.mp4",
            "clip_start": 11.70,
            "clip_end": 16.83
          },
          "scenario": "single_true",
          "question_index": 814,
          "question_time": 816.38,
          "clip_start_time": 11.70,
          "clip_end_time": 16.83,
          "entropy_prefix": {"S1": 0.80},
          "asked_after_video": true,
          "question_variant": "spatial"
        }
      ],
      "questions_at_end": true
    }
  ]
}
```

- `sequences_used`: exact token streams assigned to the render.
- `entropy_overall`: analytic + empirical entropy readings per sequence.
- `questions`: every binary question asked for this video. `question_index` is
  the final token position for the target slice, so identical values across
  videos simply mean the same sequence position was probed—not a bug.
- Clips referenced in `candidate.clip_path` live under the bucket’s `clips/`
  directory whenever `--clip-options` was enabled.

Refer to the corresponding scripts in `scripts/` or `scripts/spatial/` for the
exact CLI arguments that produce each manifest.
