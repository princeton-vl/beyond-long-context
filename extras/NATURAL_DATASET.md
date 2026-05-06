# Natural-Video Benchmark — Exact Construction Pipeline

Step-by-step recipe for the natural-video benchmark underlying §6 of
`paper_neurips2025_v2/paper.tex` (synthetic-to-natural Spearman $\rho = +0.80$).
Each stage lists input, exact parameters, exact command/function, and output.
Video-only — no text component.

A home-activity anomaly set of short video clips + 1 SoccerNet class,
organized into four difficulty buckets by duration × frame budget: `short`,
`B1`, `B2`, `B3`. 11 activity classes, balanced YES/NO per class. Each clip
answers one binary question; mp4 encoded at 1 fps so sampling at fps=1
yields the nominal frame count.

**Output tree.**

```
/workspace-vast/shmublu/data/anomaly_video_datasets/
├── benchmark_short/
│   ├── questions_dataset.json
│   └── videos/<class>_{pos,neg}_NN.mp4      (≈ 13 frames each)
├── benchmark_B1_long/
│   ├── questions_dataset.json
│   └── videos/<class>_{pos,neg}_NN.mp4      (14–36 frames, ≈ 25 nominal)
├── benchmark_B2_long/
│   ├── questions_dataset.json
│   └── videos/<class>_{pos,neg}_NN.mp4      (48–96 frames, ≈ 72 nominal)
└── benchmark_B3_long/
    ├── questions_dataset.json
    └── videos/<class>_{pos,neg}_NN.mp4      (112–256 frames, ≈ 184 nominal)
```

**Source scripts of record.** The implementation is in
`/workspace-vast/shmublu/git/streaming-mem/build_benchmark_long_buckets.py`
(B1/B2/B3 driver) and
`/workspace-vast/shmublu/data/anomaly_video_datasets/build_dataset_v2.py`
(primitives: `find_best_rates`, `extract_frames_variable_rate`, constants).
The short bucket is built per
`/workspace-vast/shmublu/data/anomaly_video_datasets/benchmark_short/RECREATE.md`.

---

## 1. Activity classes and questions

10 EPIC-Kitchens-100 classes + 1 SoccerNet v2 class. The strict-event rule
marks a source narration (EPIC) or label entry (SoccerNet) as a positive.
All EPIC questions are phrased `Did the person <action>?`; the SoccerNet
question is `Was a red card shown in this video?`.

| Class | Question action | Strict-event rule |
|---|---|---|
| `open_fridge` | open a fridge | `verb=="open"` and `"fridge" in all_nouns` |
| `open_cupboard` | open a cupboard | `verb=="open"` and `"cupboard" in all_nouns` |
| `open_drawer` | open a drawer | `verb=="open"` and `"drawer" in all_nouns` |
| `pour_water` | pour water | `verb=="pour"` and `"water" in all_nouns` |
| `cut_nontomato` | cut a vegetable/fruit (not a tomato) | `verb=="cut"` and `"tomato" not in all_nouns` |
| `rinse_X` | rinse something | `verb=="rinse"` |
| `turn_on_tap` | turn on a tap | `verb=="turn-on"` and `"tap" in all_nouns` |
| `wipe_surface` | wipe a surface/counter | `verb=="wipe"` and `all_nouns ∩ {surface,counter,table,worktop}` |
| `clean_counter` | clean a counter/surface | `verb=="clean"` and `all_nouns ∩ {counter,worktop}` |
| `wash_plate` | wash a plate | `verb=="wash"` and `"plate" in all_nouns` |
| `red_card` (SoccerNet) | — | `Labels-v2.json` entry with `"label": "Red card"` |

Per-class sample size: **10 YES + 10 NO per bucket** for the 10 EPIC classes;
**8 YES + 8 NO per bucket** for `red_card` (only 8 red-card events corpus-
wide). Nominal per-bucket count: $10 \cdot 10 + 8 = 108$ positives (same for
negatives). Actual counts in B1/B2/B3 are slightly lower because the
per-class window feasibility test (Stage 3) drops clips when the source
video is too short or the event is too long for the bucket's constraints.

---

## 2. Fetch sources

### 2a. EPIC-Kitchens 100 annotations

```bash
mkdir -p /tmp/epic
curl -L -o /tmp/epic/EPIC_100_train.csv \
    https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv
curl -L -o /tmp/epic/EPIC_100_validation.csv \
    https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_validation.csv
```

Columns used: `participant_id, video_id, start_timestamp, stop_timestamp,
verb, all_nouns`. Timestamps are `HH:MM:SS.mmm`; convert to seconds.

EPIC videos are HTTP-streamable (no download) from:

```
https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/<participant>/videos/<video_id>.MP4
```

Construct in Python as
`f"{EPIC_BASE_EXT}/{video_id.split('_')[0]}/videos/{video_id}.MP4"` (see
`epic_video_url` in `build_benchmark_long_buckets.py:159`).

### 2b. SoccerNet v2

```bash
pip install SoccerNet
python - <<'PY'
from SoccerNet.Downloader import SoccerNetDownloader as SND
d = SND(LocalDirectory="/workspace-vast/shmublu/data/anomaly_video_datasets/soccernet")
d.downloadGames(files=["Labels-v2.json","1_720p.mkv","2_720p.mkv"],
                split=["train","valid","test"])
PY
```

Each match dir contains `Labels-v2.json` + `1_720p.mkv` + `2_720p.mkv`.
Red-card events: scan every `Labels-v2.json` for `"label": "Red card"`. There
are **exactly 8** such events corpus-wide. Each event yields
`gameTime="H - MM:SS"` and `position` (ms within that half's 720p mkv).

---

## 3. Pipeline stage-by-stage

### Stage 1 — Strict-event candidate lists per class

**Input.** EPIC CSVs (train ∪ validation) and SoccerNet `Labels-v2.json`.

**Parameters.**

- Class filter: the strict-event rule from §1.
- For EPIC short bucket: additionally require
  `event_duration ∈ [2.0, 6.0]` s (this is the "strict candidate" band; see
  `RECREATE.md` §2a).
- For B1/B2/B3: no short-bucket duration band; instead the per-bucket feasibility
  test in Stage 6 decides inclusion.
- YES event selection per class per bucket: up to
  **10 distinct source videos** (`red_card`: up to 8), chosen by the
  deterministic order stored in each probe's `questions_dataset.json` (sorted by
  `video_index`); see `build_benchmark_long_buckets.py:673`.

**Call.**

Filter EPIC rows into a dict keyed by class (in `build_probe_epic_*` scripts)
to produce per-class `questions_dataset.json` files under
`/workspace-vast/shmublu/data/anomaly_video_datasets/probe_*`. These probe
files are the long-bucket builder's inputs (`CLASSES[*]["source_probe"]` in
`build_benchmark_long_buckets.py:83`).

**Output.**

Per class, a list of positive events, each carrying:

```json
{
  "video_index": <int>,
  "_probe_meta": {
    "source_video" / "source_video_id": "P02_109",
    "source_event_start_sec": <float>,
    "source_event_end_sec":   <float>
  }
}
```

For `red_card` the metadata uses `source_match`, `source_half`, and
`source_event_pos_sec`; the event window is `[pos−1, pos+1]` (instantaneous
event treated as a 2-second window; see `extract_source_meta` at
`build_benchmark_long_buckets.py:186`).

### Stage 2 — Pre-compute source durations

**Input.** Set of unique source refs (EPIC URLs and SoccerNet mkv paths)
across all classes and all YES picks.

**Parameters.** 16 parallel threads; ffprobe per ref with a 60 s timeout.

**Call.** `precompute_source_durations(workers=16)`
(`build_benchmark_long_buckets.py:243`). Each ref is probed via

```bash
ffprobe -v quiet -show_entries format=duration -of csv=p=0 <src_ref>
```

**Output.** In-process `_DUR_CACHE: dict[str, float]`. Sources that fail to
probe (0.0) are skipped from subsequent YES selection.

### Stage 3 — Feasibility test and YES window planning (per bucket)

**Input.** For one class one bucket: `(src_ref, src_dur, ev_start, ev_end,
D_min, D_max)`.

**Parameters (per bucket).**

| Bucket | `bmin` frames | `bmax` frames | `D_min` (s) | `D_max` (s) |
|---|---|---|---|---|
| short | — | ≤ 13 | — | — (3.5 s cut → stretched, see §4.1) |
| B1 | 14 | 36 | 30.0 | 60.0 |
| B2 | 48 | 96 | 60.0 | 180.0 |
| B3 | 112 | 256 | 180.0 | 600.0 |

Common constraints (all buckets):

- Event position within window: `EVENT_POS_LOW=0.10`, `EVENT_POS_HIGH=0.90`
  (event must start ≥ 10 % and end ≤ 90 % of the window).
- Event-duration cap: `ev_dur ≤ max(0.10·D, 2.0 s)`.
- `D ∈ [max(D_min, 10·ev_dur if ev_dur>2), min(D_max, src_dur)]`.
- Preferred `D` = lower bound of the feasible D-range (short B1 windows), or
  midpoint for B2/B3 (see `plan_yes_window` at
  `build_benchmark_long_buckets.py:273`).

**Call.** `plan_yes_window(ev_start, ev_end, src_dur, D_min, D_max)`. Returns
`{w_start, w_dur, ev_start_in_window, ev_end_in_window}` or `None` when the
constraints are infeasible (dropped from that bucket; recorded under
`failures` in the dataset JSON).

**Output.** For each feasible YES entry, a window `[w_start, w_start + D]`
within the source, plus the event's position inside the window.

### Stage 4 — Rate planning via `find_best_rates`

**Input.** `(D, anom_dur, bmin, bmax)` where `anom_dur = ev_end − ev_start`
computed in window-relative coordinates.

**Parameters (from `build_dataset_v2.py`).**

```python
MAX_ANOMALY_BOOST  = 3.0   # R_anom / R_base ≤ 3.0
MIN_ANOMALY_FRAMES = 5     # as set in build_dataset_v2.py (imported by the
MAX_ANOMALY_FRAMES = 10    # long-bucket builder)
```

Discrepancy: `benchmark_short/RECREATE.md` quotes 3 / 8; the code actually
uses 5 / 10. The sweep is literally `[MIN_ANOMALY_FRAMES, 3, 4,
MAX_ANOMALY_FRAMES]` = `[5, 3, 4, 10]`, but the `MIN ≤ anom_target ≤ MAX`
gate at `build_dataset_v2.py:940` discards the 3/4 candidates. See §7.

**Call.** `find_best_rates(D, anom_dur, bmin, bmax)`
(`build_dataset_v2.py:890`).

Algorithm (paraphrased from source):

```python
# NO clips: anom_dur = 0 → uniform rate targeting midpoint of [bmin, bmax]
if anom_dur <= 0:
    r = ((bmin + bmax) // 2) / D
    total = round(D * r)
    return (r, 0, total, 0, 1.0) if bmin <= total <= bmax else None

# YES clips: sweep anom_target ∈ [5, 3, 4, 10]
for anom_target in [5, 3, 4, 10]:
    R_anom     = anom_target / anom_dur
    R_base_lo  = max(R_anom / 3.0, (bmin - anom_target) / (D - anom_dur))
    R_base_hi  = min(R_anom,       (bmax - anom_target) / (D - anom_dur))
    # pick R_base as high as possible (boost ratio close to 1) subject to
    # total_frames = round((D - anom_dur) * R_base) + anom_target ∈ [bmin, bmax]
    # accept only if MIN_ANOMALY_FRAMES ≤ anom_target ≤ MAX_ANOMALY_FRAMES
    # keep the (R_base, R_anom) that minimizes ratio = R_anom / R_base
return (R_base, R_anom, total_frames, anom_frames, ratio)  # or None
```

**Output.** `(R_base, R_anom, target_frames, anom_frames, ratio)`. If `None`,
the clip is dropped and logged to `failures`.

### Stage 5 — Variable-rate frame extraction (direct HTTP seek, no intermediate)

**Input.** `(src_ref, w_start, w_dur, events_in_window, bmin, bmax)`.

**Parameters.**

- Per-frame ffmpeg seek into the source URL (or local mkv), skipping an
  intermediate local window cut. `build_benchmark_long_buckets.py` does
  *not* write an intermediate `cut_window_local` file — it streams each
  target timestamp directly via per-frame HTTP seek (see comment at
  `build_benchmark_long_buckets.py:581` and `encode_variable_rate_direct`
  at `build_benchmark_long_buckets.py:397`).
- Extraction concurrency: 4 threads per clip (`ThreadPoolExecutor(max_workers=4)`).
- Per-frame ffmpeg timeout: 60 s.
- Minimum success bar: at least `max(5, 0.7 * len(timestamps))` frames must
  land.

**Call (per frame i in the region-aware schedule).**

```bash
ffmpeg -y -loglevel error \
    -ss "<w_start + t_i>" \
    -i "<src_ref>" \
    -frames:v 1 -q:v 3 \
    -vf "scale='min(640,iw)':-2" \
    -update 1 \
    /tmp/bench_long_<uuid>/frames/<i:06d>.png
```

Timestamp schedule (region-aware):

```
interval_base = 1 / R_base
interval_anom = 1 / R_anom            # only used if anom_dur > 0
regions = [(0, ev_start), (ev_start, ev_end), (ev_end, w_dur)]
                # each tagged is_anom=True/False
for (rs, re, is_anom) in regions:
    t = rs
    while t < re - 0.001:
        timestamps.append((t, is_anom))
        t += interval_anom if is_anom else interval_base
# Trim excess non-anomaly frames from the tail until len(timestamps) == target_frames
```

Then encode the dense run of extracted pngs at 1 fps:

```bash
ffmpeg -y -loglevel error \
    -framerate 1 \
    -i /tmp/bench_long_<uuid>/dense/%06d.png \
    -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -r 1 \
    videos/<class>_pos_NN.mp4
```

**Output.** `videos/<class>_pos_NN.mp4` encoded at 1 fps so that
`ffprobe -count_frames` reports frame count exactly equal to the number of
extracted pngs (= integer duration in seconds). Audited by
`/workspace-vast/shmublu/data/anomaly_video_datasets/audit_fps1_plugplay.py`:
`round(ffprobe_duration) == nb_read_frames ∈ [bmin, bmax]`. Verified
example (open_fridge_pos_01): 11 / 36 / 96 / 256 frames for short / B1 / B2 / B3.

### Stage 6 — NO clip selection (Strategy A, same-source clean window)

**Input.** Same `(src_ref, src_dur, ev_start, ev_end)` as the paired YES,
plus the planned YES window duration `D`.

**Parameters.** `NEG_BUFFER_S = 10.0` s: the NO window must not overlap
`[ev_start − 10, ev_end + 10]`. Preference: window **before** the event if
`pre_end ≥ D`; otherwise **after** if `src_dur − post_start ≥ D`; otherwise
drop.

**Call.** `plan_neg_window(ev_start, ev_end, src_dur, D)`
(`build_benchmark_long_buckets.py:338`). Returns `(w_start, w_dur)` or `None`.

**Extraction.** Same Stage 5 pipeline with `events_in_window=[]`;
`find_best_rates` returns a single uniform rate targeting the `[bmin, bmax]`
midpoint, no anomaly boost region.

For SoccerNet the stricter 60 s buffer (vs. Red/Yellow/Yellow→Red events) is
applied at probe-build time inside `build_probe_soccer_*.py`; the long-bucket
builder reuses those already-screened source + event picks. NO is always
same-source (same EPIC session/kitchen or same SoccerNet match/half) — tests
true temporal localization, not cross-domain discrimination.

### Stage 7 — Per-class budgeting and iteration loop

**Input.** Per-class sorted YES entries + `(bmin, bmax, D_min, D_max)`.

**Parameters.**

```python
TARGETS = {"red_card": (8, 8), "default": (10, 10)}  # (N_YES, N_NO)
```

**Loop (per class per bucket).**

```python
for v in sorted(yes_entries, key=lambda v: v["video_index"]):
    meta = extract_source_meta(v, origin)
    src_dur = _DUR_CACHE[meta["src_ref"]]
    if src_dur < D_min:
        continue  # source too short for this bucket
    if yes_count < N_YES:
        queue pos task with output videos/<cls>_pos_{yes_count+1:02d}.mp4
    if neg_count < N_NO:
        queue neg task with output videos/<cls>_neg_{neg_count+1:02d}.mp4
    if yes_count == N_YES and neg_count == N_NO:
        break
```

See `build_benchmark_long_buckets.py:688–735`.

### Stage 8 — Parallel execution

**Parameters.** `--workers 6` default process pool
(`ProcessPoolExecutor(max_workers=6)` at `build_benchmark_long_buckets.py:744`).
Per-clip future timeout: 900 s.

**Call (one clip).** `build_one_clip(task)` (`build_benchmark_long_buckets.py:544`).
Each call writes to a fresh `tempfile.mkdtemp(prefix="bench_long_")` workspace,
runs Stage 5 directly on `src_ref` (no intermediate window cut), and produces
one output mp4 plus a success record with:

```python
{
  ok, class_cls, bucket, kind, seed_idx, out_mp4,
  w_start, w_dur, actual_frames, anomaly_frames, anomaly_boost_ratio,
  base_rate, anom_rate, event_frames,
  ev_start_in_window, ev_end_in_window,
}
```

Failures are recorded as `{ok:false, reason, class_cls, bucket, kind, seed_idx}`
and persisted in `questions_dataset.json["failures"]`.

### Stage 9 — `questions_dataset.json` assembly and validation

**Input.** Sorted list of successful clip records (by `class_cls, kind,
seed_idx`).

**Call.** Build one `videos[]` entry per successful clip. Top-level JSON keys:
`dataset_name, bucket, bucket_range_frames, window_duration_range_sec,
video_count, question_count, per_class_counts, failures, videos`.
Each `videos[i]` entry has:

```
video_index, variant=0, video_path="videos/<class>_{pos,neg}_NN.mp4",
contains_anomaly (bool), source_dataset ("EPIC-Kitchens-100"|"SoccerNet"),
source_class, bucket, duration_seconds, actual_frames,
anomaly_frames, anomaly_boost_ratio,
questions=[{question, question_id="bench<Bucket>_v<i>_q0",
            question_time=actual_frames*1.0, question_mode="exists",
            question_format="binary_yes_no", answer ("yes"|"no"),
            event_type_normalized, candidate={}}],
_probe_meta={source_probe, source_probe_video_index, source_video_id,
             source_event_{start,end}_sec,
             window_{start_in_source,duration}_sec,
             event_{start,end}_in_window_sec,
             base_rate_hz, anomaly_rate_hz, event_frames,
             bucket, bucket_range}
```

See `build_benchmark_long_buckets.py:824`.

**Validation.** After writing, run the integrity audit:

```bash
python /workspace-vast/shmublu/data/anomaly_video_datasets/audit_fps1_plugplay.py
```

Checks per mp4: `round(ffprobe_duration) == nb_read_frames` and
`nb_read_frames ∈ [bmin, bmax]`. Exit code 0 iff every bucket passes.

### Stage 10 — Short bucket (different primitive)

The short bucket is **not** built by `build_benchmark_long_buckets.py`. It
is built by the anomaly-centric pipeline in `benchmark_short/RECREATE.md`
§2c. Summary of differences from Stages 3–9:

- Extra per-class filter: `event_duration ∈ [2.0, 6.0]` s.
- No variable-rate machinery. Three direct ffmpeg sub-pipelines:
  1. **`open_fridge`, `open_cupboard`, `wash_plate`**: direct fps=1 cut of
     11 s centred on the event.

     ```bash
     ffmpeg -y -ss "<t_start - 4>" -i "<epic_url>" -t 11 \
         -vf "fps=1,scale='min(640,iw)':-2" -r 1 \
         -c:v libx264 -crf 22 -an -loglevel error videos/<class>_pos_NN.mp4
     ```
  2. **Other 7 EPIC classes**: event trimmed to middle 3.0 s, cut at fps=4
     over a 3.5 s window (`SHORT_CLIP_BUFFER = 0.5` s split ±0.25 s), then
     time-stretched ×4 to fps=1 playback, then trimmed to 13 frames (strict
     `<14` rule).

     ```bash
     ffmpeg -y -ss "<t_trim_start - 0.25>" -i "<epic_url>" -t 3.5 \
         -vf "fps=4,scale='min(640,iw)':-2" -c:v libx264 -crf 22 -an tmp_fps4.mp4
     ffmpeg -y -i tmp_fps4.mp4 -vf "setpts=4.0*PTS" -r 1 \
         -c:v libx264 -crf 22 -an tmp_stretched.mp4
     ffmpeg -y -i tmp_stretched.mp4 -vframes 13 \
         -c:v libx264 -crf 22 -an videos/<class>_pos_NN.mp4
     ```
  3. **`red_card`**: direct fps=1 cut, 10 s window centred at ±5 s around
     the red-card position (`pos_ms / 1000 − 5` start, 10 s duration).
- NO clip: same Strategy A (same-source clean window, ≥ 10 s buffer).
  60 s buffer for SoccerNet from any Red/Yellow/Yellow→Red event.

Output mp4s have frame counts in `{10, 11, 13}` (all ≤ 13). Short bucket
totals: 108 YES + 108 NO = 216 clips.

---

## 4. Bucket parameter summary

| Bucket | `bmin` | `bmax` | Nominal frames (midpoint used for plots) | `D_min` (s) | `D_max` (s) | Event-pct range | Max event dur | N_YES/N_NO (EPIC) | N_YES/N_NO (`red_card`) |
|---|---|---|---|---|---|---|---|---|---|
| short | — | ≤ 13 | **≈ 13** (build_figures.py uses 12) | — | — | — (anomaly ≈ 80 %) | ≤ 3.0 s (trimmed) | 10 / 10 | 8 / 8 |
| B1 | 14 | 36 | **≈ 25** | 30.0 | 60.0 | [0.10, 0.90] | `max(0.10·D, 2.0 s)` | 10 / 10 | 8 / 8 |
| B2 | 48 | 96 | **≈ 72** | 60.0 | 180.0 | [0.10, 0.90] | `max(0.10·D, 2.0 s)` | 10 / 10 | 8 / 8 |
| B3 | 112 | 256 | **≈ 184** | 180.0 | 600.0 | [0.10, 0.90] | `max(0.10·D, 2.0 s)` | 10 / 10 | 8 / 8 |

Nominal frame counts are the plotting midpoints from
`results/natural_video_eval/build_figures.py:87–92`. Per-bucket nominal
total $108 + 108 = 216$; observed B1 on disk is $88 + 88 = 176$ due to
feasibility drops (logged in `questions_dataset.json["failures"]`).

---

## 5. Bucket frame budgets

Per-bucket nominal frame counts (sampled at 1 fps; the per-clip count varies
because some events run shorter or longer than the bucket nominal):

| Bucket  | Nominal frames | Range                    |
|---------|----------------|--------------------------|
| `short` | ≈ 13           | small windows, ~10–15    |
| `B1`    | ≈ 25           | 14–36                    |
| `B2`    | ≈ 72           | 48–96                    |
| `B3`    | ≈ 184          | 112–256                  |

The on-disk mp4 itself is what the evaluator opens; no auxiliary image assets
are required.

---

## 6. Relation to the paper

Section 6 (`paper_neurips2025_v2/paper.tex:242–253`) computes per-model
$\Delta_\text{nat} = \mathrm{acc}(\text{short}) − \mathrm{acc}(\text{B3})$
and correlates it with $\Delta_\text{synth} = \mathrm{acc}(L{=}16) −
\mathrm{acc}(L{=}128)$ on the synthetic sequential structured ELOW streams.
Result: $\rho_\text{Spearman} = +0.80$ (Fisher 95 % CI $[+0.33, +0.95]$,
$p \approx 0.006$, $n=10$ above floor). The floor is the pooled
majority-class yes-rate $= 50.57\%$; the 4 excluded below-floor models are
InternVL3.5 (8B), InternVL3.5 V Thinking (8B), Phi-4-MM, and LongVILA (7B).
Including them gives $\rho = +0.75$; LOCO over the 11 classes keeps
$\rho \in [+0.76, +0.91]$.

Generators: `regenerate_fig_floor_scatter.py` (scatter at
`paper_neurips2025_v2/images/natural_video_degradation_scatter_floor.png`);
`results/natural_video_eval/compare_degradation_overlay_aligned_v2.py`
(per-model trajectories, uses `BUCKET_FRAMES = {short:12, B1:25, B2:72,
B3:184}`).

---

## 7. Known caveats and inconsistencies

- **`MIN_ANOMALY_FRAMES` / `MAX_ANOMALY_FRAMES` discrepancy.** The standalone
  recipe `benchmark_short/RECREATE.md` quotes 3 / 8; the code imported by
  the long-bucket builder uses 5 / 10. On-disk B1/B2/B3 clips were produced
  with 5 / 10 (`build_dataset_v2.py:64–65, 913, 940`).
- **EPIC narration exhaustiveness.** EPIC-100 narrations are dense but not
  formally exhaustive. NO clips require ≥ 10 s buffer from any target
  verb + noun and from semantic neighbours (e.g. `fridge, freezer,
  refrigerator, close fridge` for `open_fridge`), plus ≥ 2 other narrations
  in the NO window (enforced at probe-build time).
- **`red_card` cap.** Only 8 Red-card events exist corpus-wide in
  SoccerNet v2, so per-bucket `red_card` is 8 + 8.
- **Dropped clips per bucket.** `plan_yes_infeasible` and
  `plan_neg_noD_ref` reduce the headline 108 + 108; e.g. B1 on disk is
  88 + 88 = 176. Full loss ledger in `questions_dataset.json["failures"]`.
- **`probe_breakfast_fps4`.** A separate Breakfast Actions probe
  (30 clips, 3 classes at 5 + 5) referenced in `Planning.md`. It is **not**
  part of the 11-class natural benchmark behind $\rho = +0.80$.
- **Short-bucket borderline classes.** `open_drawer`, `turn_on_tap`,
  `wipe_surface` are retained despite being 2/3-passing at 14 frames and
  0–1/3 at 13 (per `benchmark_short/RECREATE.md` §9).
