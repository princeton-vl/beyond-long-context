import numpy as np

from vidgeom.engine import _build_token_schedule, _Scheduler


def _collect_events(scheduler: _Scheduler, seq_id: str):
    events = sorted([ev for ev in scheduler._token_heap if ev.seq_id == seq_id], key=lambda ev: ev.index)
    return [ev.t for ev in events]


def test_token_and_lane_sequences_share_timeline():
    sequences = {
        "S_tokens": ["0", "1", "2", "3"],
        "S_lanes": ["0", "1", "2", "0"],
    }
    timing = {
        "step_duration_range": [0.8, 1.0],
        "per_sequence_offset": {"S_tokens": 0.0, "S_lanes": 0.25},
    }
    scheduler = _Scheduler()
    rng = np.random.default_rng(123)

    _build_token_schedule(sequences, timing, scheduler, rng)

    token_times = _collect_events(scheduler, "S_tokens")
    lane_times = _collect_events(scheduler, "S_lanes")

    # lane stream should lag token stream by constant offset
    offset_diffs = [lane - token for lane, token in zip(lane_times, token_times)]
    assert all(abs(diff - 0.25) < 1e-6 for diff in offset_diffs)

    # consecutive token events obey the configured spacing window
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    for gap in gaps:
        assert 0.8 <= gap <= 1.0
