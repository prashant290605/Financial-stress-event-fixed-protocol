# Figure and Table Reproduction Map

## Primary Reproduction Commands

Run the full workflow from the repository root, in this order:

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

Table and figure numbers below refer to `paper/manuscript.tex`.

## Figures

Figures are written to `figures/final/` as `.pdf`, `.svg`, and `.png`. The
manuscript includes the `.pdf` versions, copied into `paper/` as `Fig1.pdf`
through `Fig10.pdf`; each file number matches the figure number it is printed
as.

| Manuscript figure | Source file in `figures/final/` | Reproduction script | Source data |
| --- | --- | --- | --- |
| Fig. 1 SPY fixed-event timeline | `timeline_spy.*` | `scripts/generate_final_figures.py` | `results/paper/spy_features_and_scores.csv`, `data/events_spy.csv` |
| Fig. 2 Multi-asset tradeoff | `multi_asset_tradeoff.*` | `scripts/generate_final_figures.py` | `results/multi_asset/multi_asset_pointwise_metrics.csv` |
| Fig. 3 Synthetic scale-mechanism test | `synthetic_mechanism.*` | `scripts/run_synthetic_mechanism.py` | generated in-script from `config/paper_experiment.json` |
| Fig. 4 SPY event-level delay and coverage | `event_level_delay_coverage.*` | `scripts/generate_final_figures.py` | `results/event_level/event_level_metrics.csv` |
| Fig. 5 SPY cost comparison | `cost_comparison.*` | `scripts/generate_final_figures.py` | `results/cost/pointwise_cost_by_asset.csv`, `results/cost/event_aware_cost_by_asset.csv` |
| Fig. 6 Event-window sensitivity | `event_window_sensitivity.*` | `scripts/generate_final_figures.py` | `results/paper/table_event_window_sensitivity.csv` |
| Fig. 7 Confirmation-window sensitivity | `confirmation_window_sensitivity.*` | `scripts/generate_final_figures.py` | `results/paper/table_window_sensitivity.csv` |
| Fig. 8 Bootstrap uncertainty | `bootstrap_delta.*` | `scripts/generate_final_figures.py` | `results/paper/bootstrap_samples.csv` |
| Fig. 9 Synthetic diagnostic series | `synthetic_example.*` | `scripts/generate_final_figures.py` | `results/paper/synthetic_series.csv` |
| Fig. 10 NAB qualitative validation | `nab_subset_results.*` | `scripts/generate_final_figures.py` | `results/paper/table_nab_subset_results.csv` |

## Tables

| Manuscript table | Output file | Reproduction script |
| --- | --- | --- |
| Table 1 Protocol comparison | narrative only; no generated file | — |
| Table 2 Fixed event panel | `data/events_spy.csv` | `scripts/reproduce_paper.py` |
| Table 3 Main SPY evaluation | `results/paper/table_spy_methods.csv` | `scripts/reproduce_paper.py` |
| Table 4 Published volatility-index comparison | `results/ablations/table_vix_benchmark.csv` | `scripts/run_vix_benchmark.py` |
| Table 5 Multi-asset day-level results | `results/multi_asset/table_multi_asset_event_fixed_evaluation.csv` | `scripts/evaluate_multi_asset.py` |
| Table 6 SPY event-level detection | `results/event_level/table_spy_event_level_detection.csv` | `scripts/evaluate_event_level.py` |
| Table 7 Per-event component counts | `results/ablations/table_component_diagnosis.csv` | `scripts/run_ablations.py` |
| Table 8 Multi-asset event-level summary | `results/event_level/table_multi_asset_event_level_summary.csv` | `scripts/evaluate_event_level.py` |
| Table 9 Event-level results by severity | `results/event_level/table_event_level_severity_summary.csv` | `scripts/evaluate_event_level.py` |
| Table 10 Day-level cost comparison | `results/paper/table_cost_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Table 11 Event-level cost comparison | `results/cost/table_spy_event_aware_cost.csv` | `scripts/run_cost_analysis.py` |
| Table 12 Crossover ratio by asset | `results/ablations/table_cost_crossover.csv` | `scripts/run_ablations.py` |
| Table 13 Event-window sensitivity | `results/paper/table_event_window_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Table 14 Composite weight ablation | `results/ablations/table_weight_ablation.csv` | `scripts/run_ablations.py` |
| Table 15 Persistence-statistic ablation | `results/ablations/table_kappa_sign_ablation.csv` | `scripts/run_ablations.py` |
| Table 16 Removing the use of future data | `results/ablations/table_causal_confirmation.csv` | `scripts/run_ablations.py` |
| Table 17 NAB results by subset | `results/paper/table_nab_subset_results.csv` | `scripts/reproduce_paper.py` (requires NAB) |
| Table A1 Usable sample per series | `results/multi_asset/data_availability.csv` | `scripts/prepare_multi_asset_data.py` |
| Table A2 Detector and evaluation settings | `config/paper_experiment.json` | — |
| Table A3 Synthetic generator settings | `results/paper/table_synthetic_config.csv` | `scripts/reproduce_paper.py` |

## In-text quantities

| Manuscript location | Output file | Reproduction script |
| --- | --- | --- |
| Section 3.2, dispersion of the weighted composite terms | `results/ablations/table_scale_diagnostic.csv` | `scripts/run_ablations.py` |
| Section 4.4.2, synthetic mechanism alarm counts | `results/ablations/table_synthetic_mechanism.csv` | `scripts/run_synthetic_mechanism.py` |
| Section 4.6.4, MMD-only confirmation signal | `results/paper/table_score_component_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Section 4.6.5, pointwise results by severity tercile | `results/paper/table_spy_severity.csv` | `scripts/reproduce_paper.py` |
| Section 4.7, bootstrap intervals | `results/paper/table_bootstrap_intervals.csv` | `scripts/reproduce_paper.py` |
| Section 4.8, NAB means across subsets | `results/paper/table_nab_methods.csv` | `scripts/reproduce_paper.py` (requires NAB) |

## Note on NAB

The Numenta Anomaly Benchmark is a third-party corpus and is not part of this
repository. Install it with `docs/NAB_SETUP.md` to regenerate Table 17, Figure
10, and the Section 4.8 means. Without it, `scripts/reproduce_paper.py` prints a
skip notice and produces every other output normally.
