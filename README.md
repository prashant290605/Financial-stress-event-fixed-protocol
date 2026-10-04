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

The paper uses daily adjusted-close prices for SPY, QQQ, IWM, HYG, and TLT. The exact CSV snapshots used for every number in the paper are archived in `data/raw_market/` (SPY retrieved from Yahoo Finance via `yfinance` on 30 March 2026; QQQ, IWM, HYG, and TLT on 22 May 2026). Providers revise adjusted-close series over time, so reproducing the reported numbers requires these archived snapshots, not a fresh download.

To download fresh market data instead (results will differ from the paper), run:

```bash
python scripts/download_data.py
```

Note that this overwrites the archived snapshots; restore them with `git checkout -- data/raw_market/` afterwards if you want the paper's exact inputs back.

The fixed event panel is provided in `data/events_spy_seed.csv` and regenerated into `data/events_spy.csv`. Multi-asset event mappings are stored in `data/events_multi_asset.csv`.

See [docs/DATA.md](docs/DATA.md) for data-source and format details.

## NAB Setup

NAB is optional and can be obtained separately using [docs/NAB_SETUP.md](docs/NAB_SETUP.md). It is used only for auxiliary qualitative validation and is not included as a repository directory.

Follow the setup guide before running the full workflow if you want to regenerate the NAB auxiliary outputs.

## Reproduction Workflow

Run the commands below from the repository root.

```bash
python scripts/prepare_multi_asset_data.py
python scripts/reproduce_paper.py
python scripts/evaluate_multi_asset.py
python scripts/evaluate_event_level.py
python scripts/run_cost_analysis.py
python scripts/run_ablations.py
python scripts/run_vix_benchmark.py
python scripts/run_synthetic_mechanism.py
python scripts/generate_final_figures.py
```

The full workflow starts from the archived market-data snapshots in `data/raw_market/` and regenerates the main result tables, the ablation and diagnostic tables in `results/ablations/`, intermediate outputs, and final publication figures. It takes roughly one minute end to end. Do not run `scripts/download_data.py` first unless you deliberately want fresh (non-identical) market data.

If NAB has not been installed (see below), `scripts/reproduce_paper.py` prints a skip notice and omits the NAB table and figure; every other output is unaffected.

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

The tests check required reproducibility artifacts, key metric values, generated result files, and consistency between result CSVs and the manuscript tables in `paper/manuscript.tex`. All tests are expected to pass on a clean checkout once the reproduction workflow above has been run.

## JDSA major revision

The paper is in major revision at the *International Journal of Data Science and Analytics*. The revision corrects one event reference date and adds the analyses requested in peer review. Its inputs, code, and outputs are kept apart from the submitted baseline, which is unchanged:

| | Submitted baseline | Revision |
|---|---|---|
| Event seed | `data/events_spy_seed.csv` | `data/jdsa_revision/events_spy_seed_v2.csv` |
| Scripts | `scripts/*.py` | `scripts/jdsa_revision/` |
| Outputs | `results/` (all folders except `jdsa_revision/`), `figures/final/` | `results/jdsa_revision/` |
| Manuscript | `paper/manuscript.tex`, as submitted | not in this repository |

The corrected seed differs from the submitted one only in E03 (WorldCom): its reference date moves from 2002-07-24 to 2002-07-22, the first trading day after the Chapter 11 filing on Sunday 21 July 2002. The new code is in `src/event_panels.py`, `src/observability.py`, and `src/revision_analysis.py`, with small additions to three existing modules that leave every baseline output unchanged.

Reproduce the revision outputs from the repository root, in this order:

```bash
python scripts/jdsa_revision/run_all.py --panel submitted --out results/jdsa_revision/pre_E03_fix --skip t3 t12
python scripts/jdsa_revision/run_all.py
python scripts/jdsa_revision/compare_runs.py results/jdsa_revision/pre_E03_fix results/jdsa_revision
python scripts/jdsa_revision/t15_freeze.py
```

The first command reruns the revision analyses on the submitted panel for comparison. The second runs them on the corrected panel and ends with the audit of every manuscript number (`results/jdsa_revision/final_audit.csv`); it takes about 25 minutes on a laptop, and `--skip t3` omits the perturbation test of every alarm rule, which takes about 7 of them. The third lists every output value that the E03 correction changes, and the fourth writes `results/jdsa_revision/FROZEN_RESULTS.md`. Start reading with `results/jdsa_revision/README.md` and `FROZEN_RESULTS.md`: the revised manuscript takes every reported value from these frozen outputs.

The revision uses the same environment as the baseline (Python 3.11 or newer and `requirements.txt`). The frozen outputs were generated with Python 3.12.7, numpy 1.26.4, pandas 2.2.2, and matplotlib 3.9.2. The baseline workflow above still regenerates the submitted outputs unchanged (`results/jdsa_revision/t12_audit/baseline_reproduction_check.csv`), and `pytest tests` covers both the baseline and the revision code.

The comparison run, the raw bootstrap resamples, and the bulk comparison of operating curves are regenerated by the commands above and are not committed (see `.gitignore`). No reported number depends on material outside this repository, but three provenance files do: `t10_parameters/development_history_note.md` was written from the authors' local development archive and is left as committed when that archive is absent; the source search in `t11_event_provenance.py` also reads that archive when present, so a rerun without it lists three fewer hits; and the frozen audit read the submitted manuscript from the journal source files, which match `paper/manuscript.tex` except for the author list, so a rerun records `paper/manuscript.tex` as its source with identical values.

## Manuscript

The manuscript files are in `paper/`. They are the version originally submitted to the journal:

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
