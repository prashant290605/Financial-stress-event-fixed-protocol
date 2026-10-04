"""Unit tests for the evaluation helpers added for the JDSA revision (no market data)."""
import math

import numpy as np
import pandas as pd

from src.paper_protocol import event_metrics, windowed_hybrid
from src.revision_analysis import (
    alarm_episodes,
    circular_block_indices,
    common_event_delays,
    cost_components,
    day_metrics,
    dispersion_ratio,
    lag_indicator,
    nearest_points,
    paired_block_bootstrap,
    paired_event_bootstrap,
    per_event_outcomes,
    split_by_cutoff,
    symmetric_confirmation,
    trailing_confirmation,
)


def _random_flags(rng, n, p):
    return (rng.random(n) < p).astype(int)


def test_symmetric_confirmation_equals_windowed_hybrid():
    rng = np.random.default_rng(0)
    for w in [0, 1, 5, 10, 30]:
        vol, inst = _random_flags(rng, 400, 0.2), _random_flags(rng, 400, 0.05)
        assert np.array_equal(symmetric_confirmation(vol, inst, w), windowed_hybrid(vol, inst, w))


def test_trailing_confirmation_reads_only_the_past():
    rng = np.random.default_rng(1)
    vol, inst = _random_flags(rng, 300, 0.3), _random_flags(rng, 300, 0.05)
    expected = np.array([int(v and inst[max(0, t - 10): t + 1].any()) for t, v in enumerate(vol)])
    assert np.array_equal(trailing_confirmation(vol, inst, 10), expected)


def test_lag_indicator_shifts_later_and_pads_with_zeros():
    out = lag_indicator(np.array([1, 0, 1, 1, 0]), 2)
    assert out.tolist() == [0, 0, 1, 0, 1]


def test_per_event_outcomes_agree_with_event_metrics():
    pred = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    spans = pd.DataFrame({"event_id": ["A", "B", "C"], "center": [2, 8, 5], "lo": [0, 6, 4], "hi": [4, 9, 5]})
    per = per_event_outcomes(pred, spans)
    ref = event_metrics(pred, list(spans[["center", "lo", "hi"]].itertuples(index=False, name=None)))
    assert per["detected"].tolist() == [True, True, False]
    assert per["detected"].mean() == ref["event_coverage"]
    assert per.loc[per["detected"], "first_delay"].mean() == ref["event_delay"]
    assert per.loc[per["detected"], "nearest_delay"].abs().mean() == ref["event_alignment_error"]


def test_common_event_delay_separates_speed_from_composition():
    a = pd.DataFrame({"event_id": ["E1", "E2", "E3"], "detected": [True, True, True], "first_delay": [-1.0, -1.0, 20.0]})
    b = pd.DataFrame({"event_id": ["E1", "E2", "E3"], "detected": [True, True, False], "first_delay": [-1.0, -1.0, math.nan]})
    summary, per = common_event_delays(a, b)
    assert summary["own_set_mean_delay_a"] > summary["own_set_mean_delay_b"]
    assert summary["n_common"] == 2
    assert summary["paired_mean_difference"] == 0.0
    assert summary["events_only_a"] == "E3"
    assert per["difference_b_minus_a"].tolist() == [0.0, 0.0]


def test_paired_event_bootstrap_keeps_repeated_draws():
    values = np.array([-1.0, 0.0, 0.0, 0.0])
    boot = paired_event_bootstrap(values, 5000, seed=3)
    assert set(np.round(boot, 2)) <= {0.0, -0.25, -0.5, -0.75, -1.0}
    assert (boot <= -0.5).any()  # the same event drawn twice counts twice
    assert np.array_equal(boot, paired_event_bootstrap(values, 5000, seed=3))


def test_circular_block_indices_are_contiguous_modulo_n():
    rng = np.random.default_rng(4)
    idx = circular_block_indices(103, 20, rng)
    assert len(idx) == 103
    for start in range(0, 100, 20):
        block = idx[start: start + 20]
        assert np.array_equal(np.diff(block) % 103, np.ones(len(block) - 1))


def test_block_bootstrap_signs_are_fixed_under_nesting():
    rng = np.random.default_rng(5)
    y = np.zeros(1000, int)
    for c in range(60, 1000, 200):
        y[c - 20: c + 21] = 1
    a = _random_flags(rng, 1000, 0.15)
    b = a & _random_flags(rng, 1000, 0.5)  # alarms of b are a subset of those of a
    boot = paired_block_bootstrap(y, a, b, 40, 300, seed=6)
    assert (boot["delta_fpr"] <= 0).all()
    assert (boot["delta_recall"] <= 0).all()


def test_nearest_points_reports_both_sides():
    curve = pd.DataFrame({"quantile": [0.90, 0.91, 0.92, 0.93], "threshold": [1.0, 1.1, 1.2, 1.3],
                          "alarm_days": [50, 40, 40, 30], "fpr": [0.05, 0.03, 0.03, 0.01]})
    picks = nearest_points(curve, "fpr", 0.025).set_index("match")
    assert picks.loc["nearest", "fpr"] == 0.03
    assert picks.loc["nearest_at_or_above", "quantile_min"] == 0.91
    assert picks.loc["nearest_at_or_above", "quantile_max"] == 0.92
    assert picks.loc["nearest_at_or_below", "fpr"] == 0.01


def test_alarm_episodes_and_dispersion_ratio():
    assert alarm_episodes(np.array([1, 1, 0, 1, 0, 0, 1])) == [(0, 1), (3, 3), (6, 6)]
    assert math.isnan(dispersion_ratio(np.arange(300.0), 100, 150))
    assert dispersion_ratio(np.arange(300.0), 130, 160) > 0


def test_post_cutoff_metrics_ignore_every_day_before_the_cutoff():
    rng = np.random.default_rng(7)
    n, te = 2000, 1400
    y, pred = _random_flags(rng, n, 0.25), _random_flags(rng, n, 0.1)
    spans = pd.DataFrame({"event_id": ["A", "B", "C"], "center": [700, 1400, 1700], "lo": [640, 1340, 1640], "hi": [760, 1460, 1760]})
    post = split_by_cutoff(spans, te)["post_calibration"]
    base_day = day_metrics(y, pred, te)
    base_cost = cost_components(y, pred, per_event_outcomes(pred, post), te)
    y2, pred2 = y.copy(), pred.copy()
    y2[:te], pred2[:te] = 1 - y2[:te], 1 - pred2[:te]
    assert day_metrics(y2, pred2, te) == base_day
    assert cost_components(y2, pred2, per_event_outcomes(pred2, post), te) == base_cost
    assert post["event_id"].tolist() == ["C"]
    assert split_by_cutoff(spans, te)["straddling"]["event_id"].tolist() == ["B"]
