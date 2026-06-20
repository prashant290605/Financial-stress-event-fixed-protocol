from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TICKERS = {"SPY", "QQQ", "IWM"}
REQUIRED_METHODS = {
    "Volatility percentile",
    "Hybrid volatility + instability confirmation",
}
METRIC_COLUMNS = ["precision", "recall", "f1", "fpr"]


def metrics_path() -> Path:
    return ROOT / "results" / "multi_asset" / "multi_asset_pointwise_metrics.csv"


def test_multi_asset_metrics_file_exists():
    assert metrics_path().exists()


def test_required_methods_present_for_required_tickers():
    metrics = pd.read_csv(metrics_path())
    for ticker in REQUIRED_TICKERS:
        methods = set(metrics.loc[metrics["ticker"] == ticker, "method"])
        assert REQUIRED_METHODS <= methods


def test_metrics_are_in_valid_range():
    metrics = pd.read_csv(metrics_path())
    for col in METRIC_COLUMNS:
        assert metrics[col].between(0.0, 1.0).all()
    assert (metrics["n_alarms"] >= 0).all()
    assert (metrics["n_event_positive_days"] > 0).all()
    assert (metrics["n_non_event_days"] > 0).all()


def test_delay_and_alignment_are_finite_or_na():
    metrics = pd.read_csv(metrics_path())
    for col in ["mean_delay", "alignment_error"]:
        assert metrics[col].map(lambda value: pd.isna(value) or math.isfinite(value)).all()
