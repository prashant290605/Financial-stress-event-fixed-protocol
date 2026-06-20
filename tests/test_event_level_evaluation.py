from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TICKERS = {"SPY", "QQQ", "IWM", "HYG", "TLT"}
METHODS = {
    "Volatility percentile",
    "CUSUM",
    "Instability score",
    "MMD-only confirmation",
    "Kappa-only confirmation",
    "Hybrid volatility + instability confirmation",
}


def detail() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "event_level" / "event_level_metrics.csv")


def test_every_event_method_ticker_combination_appears():
    events = pd.read_csv(ROOT / "data" / "events_spy.csv")
    rows = detail()
    expected = len(TICKERS) * len(METHODS) * len(events)
    assert len(rows) == expected
    assert set(rows["ticker"]) == TICKERS
    assert set(rows["method"]) == METHODS
    assert not rows.duplicated(["ticker", "event_id", "method"]).any()


def test_detected_event_is_boolean():
    rows = detail()
    assert set(rows["detected_event"].dropna().unique()) <= {True, False}
    assert set(rows["pre_event_detection"].dropna().unique()) <= {True, False}
    assert set(rows["post_event_detection"].dropna().unique()) <= {True, False}


def test_event_coverage_is_in_unit_interval():
    rows = detail()
    assert rows["event_coverage"].between(0.0, 1.0).all()
    assert (rows["n_alarm_days_in_window"] >= 0).all()


def test_delays_are_na_only_for_missed_events():
    rows = detail()
    for _, row in rows.iterrows():
        for col in ["first_detection_delay", "closest_detection_delay"]:
            if bool(row["detected_event"]):
                assert not pd.isna(row[col])
                assert math.isfinite(row[col])
            else:
                assert pd.isna(row[col])
