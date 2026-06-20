from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["SPY", "QQQ", "IWM", "HYG", "TLT"]
REQUIRED_COLUMNS = {
    "date",
    "price",
    "log_return",
    "rolling_volatility",
    "return_zscore",
    "signal",
}


def processed_path(ticker: str) -> Path:
    return ROOT / "data" / "processed" / f"{ticker}_processed.csv"


def test_processed_asset_files_exist_and_have_required_columns():
    for ticker in TICKERS:
        path = processed_path(ticker)
        assert path.exists(), f"missing processed file for {ticker}: {path}"
        df = pd.read_csv(path)
        assert REQUIRED_COLUMNS <= set(df.columns)
        assert len(df) > 0


def test_multi_asset_event_dates_map_to_asset_trading_calendars():
    events = pd.read_csv(ROOT / "data" / "events_multi_asset.csv", parse_dates=["mapped_trading_date"])
    assert set(events["ticker"]) == set(TICKERS)
    for ticker in TICKERS:
        asset_dates = set(pd.read_csv(processed_path(ticker), parse_dates=["date"])["date"])
        mapped = events.loc[events["ticker"] == ticker, "mapped_trading_date"]
        assert mapped.isin(asset_dates).all()


def test_no_duplicate_event_ids_per_ticker():
    events = pd.read_csv(ROOT / "data" / "events_multi_asset.csv")
    assert not events.duplicated(["ticker", "event_id"]).any()
    assert (events.groupby("ticker")["event_id"].nunique() == 13).all()


def test_asset_specific_severity_scores_are_finite():
    events = pd.read_csv(ROOT / "data" / "events_multi_asset.csv")
    assert events["severity_score_asset_specific"].map(math.isfinite).all()
    assert events["severity_label_asset_specific"].isin(["medium", "high", "critical"]).all()


def test_data_availability_table_contains_required_columns_and_assets():
    availability = pd.read_csv(ROOT / "results" / "multi_asset" / "data_availability.csv")
    assert list(availability.columns) == [
        "ticker",
        "asset_name",
        "start_date",
        "end_date",
        "number_of_trading_days",
        "number_of_events_mapped",
        "notes",
    ]
    assert set(availability["ticker"]) == set(TICKERS)
    assert (availability["number_of_events_mapped"] == 13).all()
