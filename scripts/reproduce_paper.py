from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_protocol import (
    ROOT,
    block_bootstrap,
    bootstrap_summary,
    build_event_table,
    cost_sensitivity,
    detector_outputs,
    evaluate_nab,
    evaluate_synthetic,
    event_bootstrap,
    event_window_sensitivity,
    generate_synthetic,
    load_config,
    missing_trading_days,
    preprocess_spy,
    save_figures,
    score_ablation,
    seed_everything,
    stratified_summary,
    summarize_methods,
    window_ablation,
)


def main() -> None:
    config = load_config(ROOT / "config" / "paper_experiment.json")
    seed_everything(config["seed"])
    results = ROOT / "results" / "paper"
    figures = results / "figures"
    results.mkdir(parents=True, exist_ok=True)

    df = preprocess_spy(config)
    missing_trading_days(df).to_csv(results / "spy_missing_business_days.csv", index=False)
    events = build_event_table(df, config)
    features, pred = detector_outputs(df, config)
    features.to_csv(results / "spy_features_and_scores.csv", index=False)

    summary, y, _ = summarize_methods(df, events, pred, config)
    summary.to_csv(results / "table_spy_methods.csv", index=False)
    cost_table = cost_sensitivity(summary)
    cost_table.to_csv(results / "table_cost_sensitivity.csv", index=False)
    strata = stratified_summary(df, events, pred, config)
    strata.to_csv(results / "table_spy_severity.csv", index=False)
    windows = window_ablation(df, events, pred, config)
    windows.to_csv(results / "table_window_sensitivity.csv", index=False)
    event_windows = event_window_sensitivity(df, events, pred, config)
    event_windows.to_csv(results / "table_event_window_sensitivity.csv", index=False)
    scores = score_ablation(df, events, pred, config)
    scores.to_csv(results / "table_score_component_sensitivity.csv", index=False)

    boot_event = event_bootstrap(df, events, pred, config)
    boot_block = block_bootstrap(y, pred["hybrid_w10"], pred["volatility_percentile"], config)
    boot = boot_event._append(boot_block, ignore_index=True)
    boot.to_csv(results / "bootstrap_samples.csv", index=False)
    boot_dir = ROOT / "results" / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)
    boot_event[["resample", "delta_f1"]].to_csv(boot_dir / "event_bootstrap_delta_f1.csv", index=False)
    boot_event[["resample", "delta_fpr"]].to_csv(boot_dir / "event_bootstrap_delta_fpr.csv", index=False)
    boot_block[["resample", "delta_f1"]].to_csv(boot_dir / "block_bootstrap_delta_f1.csv", index=False)
    boot_block[["resample", "delta_fpr"]].to_csv(boot_dir / "block_bootstrap_delta_fpr.csv", index=False)
    bootstrap_summary(boot).to_csv(results / "table_bootstrap_intervals.csv", index=False)

    syn = generate_synthetic(config)
    syn.to_csv(results / "synthetic_series.csv", index=False)
    syn_summary, syn_example = evaluate_synthetic(syn)
    syn_summary.to_csv(results / "table_synthetic_results.csv", index=False)
    import pandas as pd
    pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in config["synthetic"].items()]
    ).to_csv(results / "table_synthetic_config.csv", index=False)

    nab_rows, nab_subset = evaluate_nab(config)
    nab_rows.to_csv(results / "nab_series_results.csv", index=False)
    nab_subset.to_csv(results / "table_nab_subset_results.csv", index=False)
    nab_rows.groupby("method")[["precision", "recall", "f1", "fpr"]].mean().reset_index().to_csv(results / "table_nab_methods.csv", index=False)

    save_figures(df, events, pred, windows, event_windows, boot, cost_table, strata, syn_example, summary, figures)
    print(f"Reproduced paper artifacts in {results}")


if __name__ == "__main__":
    main()
