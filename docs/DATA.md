# Data Documentation

This repository uses publicly available financial market data and documented financial-stress event records.

## Market Data

The paper uses daily adjusted-close prices for:

- SPY: SPDR S&P 500 ETF Trust
- QQQ: Invesco QQQ Trust
- IWM: iShares Russell 2000 ETF
- HYG: iShares iBoxx $ High Yield Corporate Bond ETF
- TLT: iShares 20+ Year Treasury Bond ETF

Expected local files:

- `data/raw_market/spy.csv`
- `data/raw_market/QQQ.csv`
- `data/raw_market/IWM.csv`
- `data/raw_market/HYG.csv`
- `data/raw_market/TLT.csv`

Each CSV must contain at least:

- `Date`
- `Adj Close`

Yahoo Finance exports with `Open`, `High`, `Low`, `Close`, and `Volume` columns are also accepted. To download fresh copies using `yfinance`, run:

```bash
python scripts/download_data.py
```

The multi-asset preparation script also attempts to download missing assets with `yfinance` if a local CSV is not present.

## Event Data

The fixed event panel is stored in:

- `data/events_spy_seed.csv`
- `data/events_spy.csv`
- `data/events_multi_asset.csv`

The event panel contains calendar dates, mapped trading dates, event types, source labels, inclusion rules, and realized-drawdown severity labels. Event-source strings are retained in the CSV files to support auditability.

## NAB Auxiliary Validation

The auxiliary NAB validation expects the NAB repository structure under:

- `external/NAB`

The code uses `external/NAB/labels/combined_windows.json` and the corresponding NAB data files. NAB is used only as external qualitative validation under the same pointwise metrics used in the financial-stress protocol; official NAB scoring is not used.

See `docs/NAB_SETUP.md` for setup commands.
