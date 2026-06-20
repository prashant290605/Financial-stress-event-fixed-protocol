# Figure and Table Reproduction Map

## Primary Reproduction Command

Run the full paper workflow:

```bash
python scripts/reproduce_paper.py
```

Then regenerate publication-style figures:

```bash
python scripts/generate_final_figures.py
```

## Figures Used in the Manuscript

All manuscript figures are stored in `figures/final/` as `.pdf`, `.svg`, and `.png` files. The Springer manuscript currently references the `.png` versions.

| Manuscript figure | Output file | Reproduction script | Source data |
| --- | --- | --- | --- |
| SPY fixed-event timeline | `figures/final/timeline_spy.*` | `scripts/generate_final_figures.py` | `results/paper/spy_features_and_scores.csv`, `data/events_spy.csv` |
| Multi-asset tradeoff | `figures/final/multi_asset_tradeoff.*` | `scripts/generate_final_figures.py` | `results/multi_asset/multi_asset_pointwise_metrics.csv` |
| SPY event-level delay and coverage | `figures/final/event_level_delay_coverage.*` | `scripts/generate_final_figures.py` | `results/event_level/event_level_metrics.csv` |
| SPY cost comparison | `figures/final/cost_comparison.*` | `scripts/generate_final_figures.py` | `results/cost/pointwise_cost_by_asset.csv`, `results/cost/event_aware_cost_by_asset.csv` |
| Event-window sensitivity | `figures/final/event_window_sensitivity.*` | `scripts/generate_final_figures.py` | `results/paper/table_event_window_sensitivity.csv` |
| Hybrid confirmation-window sensitivity | `figures/final/confirmation_window_sensitivity.*` | `scripts/generate_final_figures.py` | `results/paper/table_window_sensitivity.csv` |
| Bootstrap uncertainty | `figures/final/bootstrap_delta.*` | `scripts/generate_final_figures.py` | `results/paper/bootstrap_samples.csv` |
| Synthetic diagnostic validation | `figures/final/synthetic_example.*` | `scripts/generate_final_figures.py` | `results/paper/synthetic_series.csv` |
| NAB qualitative validation | `figures/final/nab_subset_results.*` | `scripts/generate_final_figures.py` | `results/paper/table_nab_subset_results.csv` |

## Tables

| Manuscript table family | Output file | Reproduction script |
| --- | --- | --- |
| Main SPY pointwise evaluation | `results/paper/table_spy_methods.csv` | `scripts/reproduce_paper.py` |
| Fixed event panel | `data/events_spy.csv` | `scripts/reproduce_paper.py` |
| Multi-asset pointwise evaluation | `results/multi_asset/table_multi_asset_event_fixed_evaluation.csv` | `scripts/evaluate_multi_asset.py` |
| SPY event-level detection | `results/event_level/table_spy_event_level_detection.csv` | `scripts/evaluate_event_level.py` |
| Multi-asset event-level summary | `results/event_level/table_multi_asset_event_level_summary.csv` | `scripts/evaluate_event_level.py` |
| Severity-stratified summary | `results/event_level/table_event_level_severity_summary.csv` | `scripts/evaluate_event_level.py` |
| Pointwise cost sensitivity | `results/cost/table_spy_pointwise_cost.csv` | `scripts/run_cost_analysis.py` |
| Event-aware cost sensitivity | `results/cost/table_spy_event_aware_cost.csv` | `scripts/run_cost_analysis.py` |
| Event-window sensitivity | `results/paper/table_event_window_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Confirmation-window sensitivity | `results/paper/table_window_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Score-component sensitivity | `results/paper/table_score_component_sensitivity.csv` | `scripts/reproduce_paper.py` |
| Bootstrap intervals | `results/paper/table_bootstrap_intervals.csv` | `scripts/reproduce_paper.py` |
| Synthetic diagnostic summary | `results/paper/table_synthetic_results.csv` | `scripts/reproduce_paper.py` |
| NAB subset summary | `results/paper/table_nab_subset_results.csv` | `scripts/reproduce_paper.py` |
