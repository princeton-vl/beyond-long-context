# Extras

Experiment-running scripts that are not on the main reproduction path
(`main.py`). Each is a driver for a separate evaluation campaign.

| Script | Purpose |
|---|---|
| `run_multi_question.py` | Multi-question batching eval — asks 2 or 3 questions per prompt instead of 1. Writes its own `multi{N}-*.csv` and `multi{N}-*.state` files; never touches baseline `buckets-*` files. |
| `run_anomaly_eval.py` | Natural-video anomaly eval — feeds video + yes/no question to a VLM and extracts the answer. Designed for the natural-video benchmark below. |

## Reference docs

- `NATURAL_DATASET.md` — exact construction recipe for the natural-video
  benchmark (EPIC-Kitchens 100 + SoccerNet v2, four duration buckets).
  Documents the source-data fetch, per-class strict-event rules, and
  per-bucket sampling parameters.

All scripts here require the same Python environment used for `main.py`.
Run from the repo root unless a specific script's docstring says otherwise.
