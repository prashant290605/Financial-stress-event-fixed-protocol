"""The post-calibration evaluation must use no day at or before the calibration cutoff."""
from pathlib import Path

import numpy as np
import pytest

from src.multi_asset_evaluation import _load_asset_events, _load_processed_asset
from src.paper_protocol import detector_outputs, event_mask, load_config
from src.revision_analysis import (
    calibration_cutoff,
    component_scores,
    cost_components,
    day_metrics,
    detector_alarms,
    event_spans,
    per_event_outcomes,
    split_by_cutoff,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")


@pytest.fixture(scope="module")
def spy():
    df, events = _load_processed_asset("SPY"), _load_asset_events("SPY").reset_index(drop=True)
    scores = component_scores(df, CONFIG)
    return df, events, scores, detector_alarms(scores, CONFIG)


def test_cutoff_is_derived_from_the_configured_fraction(spy):
    df, _, _, _ = spy
    cut = calibration_cutoff(df, CONFIG)
    assert cut.train_end == int(round(len(df) * CONFIG["detectors"]["calibration_fraction"]))
    assert cut.last_calibration_date < cut.first_evaluation_date
    assert cut.last_calibration_date.date().isoformat() == "2016-11-08"


def test_post_calibration_event_set_is_e09_to_e13_with_no_straddling_window(spy):
    df, events, _, _ = spy
    cut = calibration_cutoff(df, CONFIG)
    parts = split_by_cutoff(event_spans(df, events, CONFIG["events"]["half_window"]), cut.train_end)
    assert parts["post_calibration"]["event_id"].tolist() == ["E09", "E10", "E11", "E12", "E13"]
    assert parts["straddling"].empty
    assert (df["date"].iloc[parts["post_calibration"]["lo"]] > cut.last_calibration_date).all()


@pytest.mark.parametrize("rule", ["volatility", "retrospective_confirmation", "fully_causal_confirmation"])
def test_post_calibration_results_do_not_change_when_pre_cutoff_days_change(spy, rule):
    df, events, _, alarms = spy
    te = calibration_cutoff(df, CONFIG).train_end
    y, _ = event_mask(df, events, CONFIG["events"]["half_window"])
    post = split_by_cutoff(event_spans(df, events, CONFIG["events"]["half_window"]), te)["post_calibration"]
    pred = alarms[rule]
    base = (day_metrics(y, pred, te), cost_components(y, pred, per_event_outcomes(pred, post), te))
    rng = np.random.default_rng(11)
    y2, pred2 = y.copy(), pred.copy()
    y2[:te] = rng.integers(0, 2, te)
    pred2[:te] = rng.integers(0, 2, te)
    assert (day_metrics(y2, pred2, te), cost_components(y2, pred2, per_event_outcomes(pred2, post), te)) == base
    assert day_metrics(y, pred, te)["n_days"] == len(df) - te


def test_revision_alarms_match_the_submitted_detector(spy):
    df, _, _, alarms = spy
    _, submitted = detector_outputs(df, CONFIG)
    assert np.array_equal(alarms["volatility"], submitted["volatility_percentile"])
    assert np.array_equal(alarms["instability"], submitted["instability_score"])
    assert np.array_equal(alarms["retrospective_confirmation"], submitted["hybrid_w10"])
