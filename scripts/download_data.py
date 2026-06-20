from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yfinance as yf


OUT = PROJECT_ROOT / "data" / "raw_market"
TICKERS = {
    "spy": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "HYG": "HYG",
    "TLT": "TLT",
}
START = "2000-01-01"
END = "2023-12-30"


def _download(ticker: str, out_name: str) -> None:
    data = yf.download(
        ticker,
        start=START,
        end=END,
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
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{out_name}.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


def main() -> None:
    for out_name, ticker in TICKERS.items():
        _download(ticker, out_name)


if __name__ == "__main__":
    main()
