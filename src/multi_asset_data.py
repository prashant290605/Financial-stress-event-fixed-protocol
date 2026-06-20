from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paper_protocol import ROOT, load_config, nearest_trading_date


@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    asset_name: str
    required: bool = False


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("SPY", "SPDR S&P 500 ETF Trust", True),
    AssetSpec("QQQ", "Invesco QQQ Trust", True),
    AssetSpec("IWM", "iShares Russell 2000 ETF", True),
    AssetSpec("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", False),
    AssetSpec("TLT", "iShares 20+ Year Treasury Bond ETF", False),
)

REQUIRED_PROCESSED_COLUMNS = {
    "date",
    "price",
    "log_return",
    "rolling_volatility",
    "return_zscore",
    "signal",
}


def _candidate_raw_paths(ticker: str, data_dir: Path) -> list[Path]:
    return [
        data_dir / f"{ticker}.csv",
        data_dir / f"{ticker.lower()}.csv",
    ]


def _read_price_csv(path: Path, price_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain a Date column")
    if price_column not in df.columns:
        raise ValueError(f"{path} must contain an {price_column!r} column")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["date"] = pd.to_datetime(df["Date"])
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")
    df = df.dropna(subset=["date", price_column]).sort_values("date")
    if df.empty:
        raise ValueError(f"{path} contains no usable dated {price_column!r} values")
    return df[["date", price_column]].rename(columns={price_column: "price"})


def _download_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp, out_path: Path) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed") from exc

    data = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(data["Date"]).dt.date.astype(str),
            "Open": data.get("Open"),
            "High": data.get("High"),
            "Low": data.get("Low"),
            "Close": data.get("Close"),
            "Adj Close": data.get("Adj Close", data.get("Close")),
            "Volume": data.get("Volume"),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return _read_price_csv(out_path, "Adj Close")


def load_asset_prices(
    ticker: str,
    price_column: str,
    data_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    for path in _candidate_raw_paths(ticker, data_dir):
        if path.exists():
            return _read_price_csv(path, price_column), f"loaded local CSV {path.as_posix()}"
    raw_path = data_dir / f"{ticker}.csv"
    return _download_yfinance(ticker, start, end, raw_path), f"downloaded with yfinance to {raw_path.as_posix()}"


def preprocess_price_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    spy_cfg = config["spy"]
    out = df.copy()
    out["log_return"] = np.log(out["price"]).diff()
    out["rolling_volatility"] = out["log_return"].rolling(spy_cfg["rolling_vol_window"]).std()
    rolling_mean = out["log_return"].rolling(spy_cfg["zscore_window"]).mean()
    rolling_std = out["log_return"].rolling(spy_cfg["zscore_window"]).std()
    out["return_zscore"] = (out["log_return"] - rolling_mean) / rolling_std
    w = spy_cfg["signal_weights"]
    out["signal"] = (
        w["log_return"] * out["log_return"]
        + w["rolling_volatility"] * out["rolling_volatility"]
        + w["return_zscore"] * out["return_zscore"]
    )
    return out.dropna().reset_index(drop=True)


def asset_drawdown_severity(
    df: pd.DataFrame,
    mapped: pd.Timestamp,
    half_window: int,
) -> float:
    center = int(df.index[df["date"] == mapped][0])
    lo = max(0, center - half_window)
    hi = min(len(df), center + half_window + 1)
    prices = df["price"].iloc[lo:hi].to_numpy(float)
    running_max = np.maximum.accumulate(prices)
    drawdown = prices / running_max - 1.0
    return float(abs(np.min(drawdown)))


def map_events_for_asset(df: pd.DataFrame, ticker: str, events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    half_window = config["events"]["severity_half_window"]
    for _, row in events.iterrows():
        cal = pd.Timestamp(row["calendar_date"])
        mapped = nearest_trading_date(df["date"], cal)
        rows.append(
            {
                "ticker": ticker,
                "event_id": row["event_id"],
                "event_name": row["event_name"],
                "calendar_date": cal.date().isoformat(),
                "mapped_trading_date": mapped.date().isoformat(),
                "event_type": row["event_type"],
                "source_reference": row["source_reference"],
                "severity_score_asset_specific": asset_drawdown_severity(df, mapped, half_window),
                "severity_label_asset_specific": "",
                "severity_rule": config["events"]["severity_rule"],
            }
        )
    out = pd.DataFrame(rows)
    q1, q2 = out["severity_score_asset_specific"].quantile([1 / 3, 2 / 3]).to_numpy()
    out["severity_label_asset_specific"] = np.where(
        out["severity_score_asset_specific"] >= q2,
        "critical",
        np.where(out["severity_score_asset_specific"] >= q1, "high", "medium"),
    )
    return out


def prepare_multi_asset_data(config_path: Path | str = ROOT / "config" / "paper_experiment.json") -> dict[str, Any]:
    config = load_config(config_path)
    data_dir = ROOT / "data" / "raw_market"
    processed_dir = ROOT / "data" / "processed"
    multi_results_dir = ROOT / "results" / "multi_asset"
    processed_dir.mkdir(parents=True, exist_ok=True)
    multi_results_dir.mkdir(parents=True, exist_ok=True)

    spy_raw = _read_price_csv(data_dir / "spy.csv", config["spy"]["price_column"])
    start, end = spy_raw["date"].min(), spy_raw["date"].max()
    seed_events = pd.read_csv(ROOT / "data" / "events_spy.csv")

    event_tables: list[pd.DataFrame] = []
    availability_rows: list[dict[str, Any]] = []
    processed: list[str] = []
    skipped: list[dict[str, str]] = []

    for asset in ASSETS:
        try:
            prices, note = load_asset_prices(asset.ticker, config["spy"]["price_column"], data_dir, start, end)
            prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
            raw_start = prices["date"].min()
            processed_df = preprocess_price_frame(prices, config)
            out_path = processed_dir / f"{asset.ticker}_processed.csv"
            processed_df.to_csv(out_path, index=False)

            mapped = map_events_for_asset(processed_df, asset.ticker, seed_events, config)
            notes = note
            if raw_start > spy_raw["date"].min():
                notes = (
                    f"{notes}; available processed period starts after SPY, "
                    "events before asset inception are mapped to the nearest available asset trading date"
                )
            event_tables.append(mapped)
            availability_rows.append(
                {
                    "ticker": asset.ticker,
                    "asset_name": asset.asset_name,
                    "start_date": processed_df["date"].min().date().isoformat(),
                    "end_date": processed_df["date"].max().date().isoformat(),
                    "number_of_trading_days": int(len(processed_df)),
                    "number_of_events_mapped": int(len(mapped)),
                    "notes": notes,
                }
            )
            processed.append(asset.ticker)
        except Exception as exc:
            msg = str(exc)
            if asset.required:
                raise RuntimeError(
                    f"Required asset {asset.ticker} could not be processed: {msg}. "
                    "Provide a CSV with columns Date and Adj Close, matching the Yahoo Finance export format."
                ) from exc
            availability_rows.append(
                {
                    "ticker": asset.ticker,
                    "asset_name": asset.asset_name,
                    "start_date": "",
                    "end_date": "",
                    "number_of_trading_days": 0,
                    "number_of_events_mapped": 0,
                    "notes": f"skipped: {msg}; provide CSV with columns Date and Adj Close",
                }
            )
            skipped.append({"ticker": asset.ticker, "reason": msg})

    events_multi = pd.concat(event_tables, ignore_index=True)
    availability = pd.DataFrame(availability_rows)
    events_multi.to_csv(ROOT / "data" / "events_multi_asset.csv", index=False)
    availability.to_csv(multi_results_dir / "data_availability.csv", index=False)
    return {"processed": processed, "skipped": skipped, "availability": availability, "events": events_multi}
