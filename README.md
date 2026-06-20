# Financial-stress-event-fixed-protocol

This repository accompanies the research paper:

**A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors**

It provides the code, configuration files, event definitions, reproduction scripts, result tables, and publication figures required to regenerate the main analyses in the paper.

## Repository Purpose

This repository accompanies the research paper and provides the code, configuration, event definitions, figures, tables, and scripts necessary to reproduce the published analyses. It is intended as a public reproducibility package for inspection, reuse, and citation.

## Abstract

Financial stress detection is important for risk monitoring, market surveillance, and decision support, but consistent evaluation is difficult because stress events are rare, event boundaries are ambiguous, and detector outputs are often compared under incompatible labeling rules. The paper presents a reproducible event-fixed evaluation protocol for daily financial stress detectors. The protocol fixes documented stress events before detector inspection, applies deterministic preprocessing, assigns rule-based drawdown severity categories, and reports pointwise, event-level, delay, uncertainty, multi-asset, and cost-sensitive metrics. As an illustrative use case, the paper compares a volatility percentile trigger with a hybrid retrospective confirmation rule that requires local instability evidence. The results show that event-fixed evaluation changes detector interpretation: the same confirmation rule can appear attractive under false-alarm-sensitive metrics while remaining weaker for recall, event coverage, and timeliness.

## Repository Structure

```text
config/                 Experiment configuration
data/                   Event files, processed data, and raw market-data location
docs/                   Data, figure, table, and protocol documentation
figures/final/          Publication figures used by the manuscript
paper/                  Manuscript source and Springer template files
results/                Reproducible result tables and generated outputs
scripts/                Reproduction, data-preparation, and figure-generation scripts
src/                    Core paper implementation
tests/                  Reproducibility and consistency tests
```

## Installation

Use Python 3.11 or newer.

```bash
git clone https://github.com/prashant290605/Financial-stress-event-fixed-protocol.git
cd Financial-stress-event-fixed-protocol
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Requirements

The paper uses daily adjusted-close prices for SPY, QQQ, IWM, HYG, and TLT. Local CSV files are expected in `data/raw_market/` with at least `Date` and `Adj Close` columns.

To download fresh public market data with `yfinance`, run:

```bash
python scripts/download_data.py
```

The fixed event panel is provided in `data/events_spy_seed.csv` and regenerated into `data/events_spy.csv`. Multi-asset event mappings are stored in `data/events_multi_asset.csv`.

See [docs/DATA.md](docs/DATA.md) for data-source and format details.

## NAB Setup

NAB is optional and can be obtained separately using [docs/NAB_SETUP.md](docs/NAB_SETUP.md). It is used only for auxiliary qualitative validation and is not included as a repository directory.

Follow the setup guide before running the full workflow if you want to regenerate the NAB auxiliary outputs.

## Reproduction Workflow

Run the commands below from the repository root.

```bash
python scripts/download_data.py
python scripts/prepare_multi_asset_data.py
python scripts/reproduce_paper.py
python scripts/evaluate_multi_asset.py
python scripts/evaluate_event_level.py
python scripts/run_cost_analysis.py
python scripts/generate_final_figures.py
```

The full workflow regenerates the main result tables, intermediate outputs, and final publication figures.

## Expected Outputs

Primary outputs are written to:

- `results/paper/`
- `results/multi_asset/`
- `results/event_level/`
- `results/cost/`
- `results/bootstrap/`
- `figures/final/`

See [docs/FIGURES_AND_TABLES.md](docs/FIGURES_AND_TABLES.md) for a table-by-table and figure-by-figure reproduction map.

## Verification

Run:

```bash
pytest tests
```

The tests check required reproducibility artifacts, key metric values, generated result files, and consistency between result CSVs and manuscript tables.

## Manuscript

The manuscript files are in `paper/`. The current manuscript source and compiled PDF are:

```text
paper/manuscript.tex
paper/manuscript.pdf
```

The manuscript references the final figure files under `figures/final/`.

## Citation

Citation information will be added after publication. Until then, cite the repository as:

```text
Singh, P., and Singh, P. A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors: reproducibility repository.
GitHub: https://github.com/prashant290605/Financial-stress-event-fixed-protocol
```

## Contact

Prashant Singh  
Department of Mathematics and Computing  
Indian Institute of Technology Ropar  
Rupnagar, Punjab, India  
Email: prashants18488@gmail.com
