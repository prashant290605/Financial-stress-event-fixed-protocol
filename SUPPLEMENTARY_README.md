# Supplementary Reproducibility Notes

This file documents how to regenerate the tables, figures, and supplementary artifacts for the event-fixed financial stress detector evaluation.

## Data

The experiments use publicly available adjusted-close data for:

- SPY: SPDR S&P 500 ETF Trust
- QQQ: Invesco QQQ Trust
- IWM: iShares Russell 2000 ETF
- HYG: iShares iBoxx $ High Yield Corporate Bond ETF
- TLT: iShares 20+ Year Treasury Bond ETF

Raw market CSVs are expected in `data/raw_market/`. Each file must contain `Date` and `Adj Close` columns. To download fresh copies using `yfinance`, run:

```bash
python scripts/download_data.py
```

## Required Files

- `data/events_spy_seed.csv`: fixed source event panel.
- `data/events_spy.csv`: fixed 13-event panel with event names, calendar dates, mapped SPY trading dates, event types, inclusion rules, source references, severity scores, and severity labels.
- `data/events_multi_asset.csv`: per-asset event-date mappings and asset-specific drawdown scores.
- `config/paper_experiment.json`: preprocessing, detector, bootstrap, synthetic, and NAB settings.
- `scripts/reproduce_paper.py`: primary paper reproduction script.

## Environment

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

## Reproduction Commands

Run the main paper reproduction:

```bash
python scripts/reproduce_paper.py
```

Prepare multi-asset data:

```bash
python scripts/prepare_multi_asset_data.py
```

Run multi-asset detector evaluation:

```bash
python scripts/evaluate_multi_asset.py
```

Run event-level evaluation:

```bash
python scripts/evaluate_event_level.py
```

Run cost-sensitive analyses:

```bash
python scripts/run_cost_analysis.py
```

Generate publication-style final figures:

```bash
python scripts/generate_final_figures.py
```

## Output Directories

- `results/paper/`: main SPY paper tables and intermediate paper artifacts.
- `results/paper/figures/`: generated diagnostic paper figures.
- `results/bootstrap/`: event and block bootstrap samples.
- `results/multi_asset/`: multi-asset availability, pointwise metrics, tables, and interpretation.
- `results/event_level/`: event-level detailed metrics, summaries, tables, figure, and interpretation.
- `results/cost/`: pointwise and event-aware cost tables, figure, and interpretation.
- `figures/final/`: publication-style PDF, SVG, and PNG figures used by the manuscript.

## Expected Runtime

Approximate runtime on the development machine:

- `scripts/reproduce_paper.py`: 15-40 seconds.
- `scripts/prepare_multi_asset_data.py`: a few seconds if data are cached; longer if downloads are needed.
- `scripts/evaluate_multi_asset.py`: under 30 seconds for cached processed data.
- `scripts/evaluate_event_level.py`: under 30 seconds for cached processed data.
- `scripts/run_cost_analysis.py`: under 30 seconds.
- `scripts/generate_final_figures.py`: under 30 seconds.

Network availability can affect `yfinance` downloads.

## Known Limitations

- The event panel is fixed and documented but not exhaustive.
- Primary severity labels are SPY-centered and based on realized drawdown.
- HYG and TLT start later than SPY, so early events are mapped to the nearest available trading day in those asset calendars.
- NAB is used only as a qualitative diagnostic check; official NAB scoring is not used.
- Bootstrap uncertainty is reported for the primary SPY protocol and does not capture full model-selection uncertainty.
- Event-aware cost is intentionally simple and should be replaced by institution-specific cost models for deployment.
- A fresh PDF build requires a local LaTeX installation such as `pdflatex`, `latexmk`, `tectonic`, or `xelatex`.
