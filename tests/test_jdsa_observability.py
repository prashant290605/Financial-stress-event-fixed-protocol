"""Unobservable event-asset pairs must never enter a denominator or a cost as missed events."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cost_analysis import compute_event_aware_cost
from src.event_level_evaluation import _asset_event_rows
from src.multi_asset_evaluation import _load_asset_events, _load_processed_asset, evaluate_asset
from src.observability import observable_event_flags, observable_events, unobservable_events
from src.paper_protocol import load_config
from src.revision_analysis import cost_components, event_spans, per_event_outcomes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")
EXPECTED_UNOBSERVABLE = {"SPY": [], "QQQ": [], "IWM": ["E01"], "HYG": ["E01", "E02", "E03"], "TLT": ["E01", "E02", "E03"]}


def test_observability_rule_on_a_constructed_series():
    dates = pd.bdate_range("2005-01-03", periods=400)
    df = pd.DataFrame({"date": dates, "price": np.linspace(100, 120, 400)})
    events = pd.DataFrame({"event_id": ["early", "inside"], "calendar_date": ["2003-06-02", "2005-06-01"],
                           "mapped_trading_date": [dates[0].date().isoformat(), "2005-06-01"]})
    assert observable_event_flags(df, events).tolist() == [False, True]
    obs = observable_events(df, events)
    per = per_event_outcomes(np.zeros(len(df), int), event_spans(df, obs, 60))
    assert per["event_id"].tolist() == ["inside"]
    assert cost_components(np.zeros(len(df), int), np.zeros(len(df), int), per)["M"] == 1


@pytest.mark.parametrize("ticker", list(EXPECTED_UNOBSERVABLE))
def test_unobservable_events_are_derived_from_the_data(ticker):
    df, events = _load_processed_asset(ticker), _load_asset_events(ticker)
    assert unobservable_events(df, events)["event_id"].tolist() == EXPECTED_UNOBSERVABLE[ticker]
    assert len(observable_events(df, events)) == 13 - len(EXPECTED_UNOBSERVABLE[ticker])


@pytest.fixture(scope="module")
def hyg_rows():
    return _asset_event_rows("HYG", CONFIG), _asset_event_rows("HYG", CONFIG, observable_only=True)


def test_observable_rows_drop_only_unobservable_events(hyg_rows):
    full, obs = hyg_rows
    assert set(full["event_id"]) - set(obs["event_id"]) == {"E01", "E02", "E03"}
    kept = full[full["event_id"].isin(obs["event_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, obs.reset_index(drop=True))
    rate = obs.groupby("method")["detected_event"].mean()
    counts = obs.groupby("method")["detected_event"].agg(["sum", "size"])
    assert (counts["size"] == 10).all()
    assert np.allclose(rate, counts["sum"] / 10)


def test_cost_counts_no_unobservable_event_as_missed():
    # The stored event-level rows still contain the unobservable events; the observable
    # path must drop them before counting missed events.
    stored = pd.read_csv(ROOT / "results" / "event_level" / "event_level_metrics.csv")
    cost = compute_event_aware_cost(CONFIG, observable_only=True)
    for ticker, missing in EXPECTED_UNOBSERVABLE.items():
        for method, group in cost.groupby("method"):
            rows = stored[(stored["ticker"] == ticker) & (stored["method"] == method) & ~stored["event_id"].isin(missing)]
            expected = int((~rows["detected_event"].astype(bool)).sum())
            assert set(group.loc[group["ticker"] == ticker, "missed_events"]) == {expected}


def test_default_path_still_reproduces_the_submitted_rows():
    stored = pd.read_csv(ROOT / "results" / "multi_asset" / "multi_asset_pointwise_metrics.csv")
    new = evaluate_asset("TLT", CONFIG)
    old = stored[stored["ticker"] == "TLT"].reset_index(drop=True)
    pd.testing.assert_frame_equal(new[old.columns].reset_index(drop=True), old, check_exact=False, rtol=1e-12, atol=1e-12)
