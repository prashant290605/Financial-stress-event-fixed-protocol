"""Task 10: parameter inventory, percentile sensitivity, and what the history shows.

Reviewer 2 minor 3. Every setting is listed with its source and classified as already
sensitivity-tested, newly covered in this revision, or not tested. The volatility
percentile sensitivity is read from the Task 4 curves (run t4 first); the instability
percentile is swept here with every other setting at its submitted value. No grid over
window lengths is run.

The development-history note is regenerated only when the local, git-ignored archive
folder is present; otherwise the committed note is left as it is.

Outputs: results/jdsa_revision/t10_parameters/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, OUT_ROOT, ROOT, load_asset, outdir  # noqa: E402
from src.paper_protocol import event_mask  # noqa: E402
from src.revision_analysis import component_scores, confirmation_curve, event_spans, per_event_outcomes, detector_alarms  # noqa: E402

OUT = outdir("t10_parameters")
GRID = np.round(np.linspace(0.5, 1.0, 5001), 4)
ROUND_Q = [0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99]
COLS = ["quantile", "threshold", "alarm_days", "precision", "recall", "f1", "fpr", "events_detected", "n_events",
        "mean_first_delay", "A_out", "M", "D_plus"]

TESTED = "already sensitivity-tested"
NEW = "newly covered in this revision"
NOT = "not sensitivity-tested"
INVENTORY = [
    ("Rolling volatility window", "20 trading days", "config spy.rolling_vol_window", "sigma_t; volatility trigger; composite", NOT, ""),
    ("Return z-score window", "50 trading days", "config spy.zscore_window", "z_t; composite", NOT, ""),
    ("Composite weights (w_r, w_sigma, w_z)", "(0.5, 0.3, 0.2)", "config spy.signal_weights", "x_t", TESTED,
     "Table 14 (results/ablations/table_weight_ablation.csv); t8_z_dominance"),
    ("Event half-window", "60 trading days", "config events.half_window", "evaluation windows", TESTED,
     "Table 13 / Fig 6 (results/paper/table_event_window_sensitivity.csv)"),
    ("Severity half-window", "20 trading days", "config events.severity_half_window", "severity terciles only", NOT, ""),
    ("AR window for kappa", "36 trading days", "config features.ar_window", "kappa_t", NOT, ""),
    ("MMD sample length", "30 + 30 observations", "config features.mmd_window", "MMD; also the lag of the fully causal variant", NOT, ""),
    ("MMD smoothing", "3-day trailing mean", "code: compute_instability_features", "MMD", NOT, ""),
    ("MMD kernel bandwidth", "median pairwise distance, floor 1e-4", "code: compute_instability_features", "MMD", NOT, ""),
    ("Score smoothing", "5-day trailing mean", "config features.smooth_window", "all thresholded scores", NOT, ""),
    ("Instability score composition", "0.5 z_MMD + 0.5 |z_kappa|", "code: detector_outputs", "instability score", TESTED,
     "results/paper/table_score_component_sensitivity.csv (MMD only, |kappa| only, CIS variants)"),
    ("kappa sign convention", "two-sided |z_kappa|", "code: detector_outputs", "instability score", TESTED,
     "Table 15 (results/ablations/table_kappa_sign_ablation.csv)"),
    ("CIS window", "10 days (ablations only)", "config features.cis_window", "CIS ablation variants", TESTED,
     "results/paper/table_score_component_sensitivity.csv"),
    ("Calibration fraction", "first 70 per cent of each series", "config detectors.calibration_fraction", "all thresholds and z-scores", NOT,
     "post-calibration results reported separately in t2_post_calibration (split not varied)"),
    ("Volatility percentile", "0.90", "config detectors.volatility_threshold_quantile", "volatility trigger; first condition of confirmation", NEW,
     "t4_operating_curves (SPY and four assets); t10 volatility_percentile_sensitivity_spy.csv"),
    ("Instability percentile", "0.90", "config detectors.score_threshold_quantile", "instability indicator i(t)", NEW,
     "t10 instability_percentile_sensitivity_spy.csv"),
    ("CUSUM percentile", "0.90", "config detectors.cusum_threshold_quantile", "CUSUM baseline", NOT, "t7_cusum documents its behaviour"),
    ("Confirmation window W", "10 trading days", "config detectors.default_hybrid_window", "confirmation rule", TESTED,
     "Fig 7 (results/paper/table_window_sensitivity.csv, W = 0..60)"),
    ("Confirmation timing", "symmetric [t-W, t+W]", "code: windowed_hybrid", "confirmation rule", TESTED,
     "Table 16 (results/ablations/table_causal_confirmation.csv); t2_post_calibration; t3_causality"),
    ("Bootstrap resamples", "2000", "config bootstrap.n_resamples", "intervals", NOT, "second-seed check in t6_uncertainty"),
    ("Bootstrap block length", "20 (submitted); 121 primary in revision", "config bootstrap.block_length", "block intervals", NEW,
     "t6_uncertainty (20, 60, 121)"),
    ("Random seed", "42", "config seed", "bootstrap and synthetic data", NOT, ""),
]


def development_history_note() -> str | None:
    legacy = ROOT / "archive" / "legacy_dashboard" / "backend" / "app" / "experiments" / "run_experiments.py"
    outputs = ROOT / "archive" / "legacy_outputs"
    changelog = ROOT / "archive" / "paper_history" / "CHANGELOG_PAPER.md"
    if not (legacy.exists() and (outputs / "threshold_ablation.csv").exists() and changelog.exists()):
        return None
    src = legacy.read_text(encoding="utf-8")
    constants = [line.strip() for line in src.splitlines()
                 if line.split("=")[0].strip() in {"EVENT_WINDOW", "HYBRID_WINDOW", "WINDOW_ABLATION", "THRESHOLD_ABLATION"}]
    constants += [line.strip().rstrip(",") for line in src.splitlines() if line.strip().startswith("collapse_quantile: float =")]
    thr = pd.read_csv(outputs / "threshold_ablation.csv")
    win = pd.read_csv(outputs / "window_ablation.csv")
    best_q = thr.loc[thr["f1_mean"].idxmax(), "collapse_quantile"]
    best_w = win.loc[win["f1_mean"].idxmax(), "window"]
    log = changelog.read_text(encoding="utf-8")
    entry = [line.strip() for line in log.splitlines() if "fixed before detector evaluation" in line]
    lines = [
        "# Parameter and event-panel history: what the repository shows",
        "",
        "Generated by scripts/jdsa_revision/t10_parameters.py from the local archive folder, which is excluded from the public",
        "repository by .gitignore. Statements are limited to what these files show.",
        "",
        "## Public repository",
        "",
        "- The git history starts at the initial public release (2026-06-20). `config/paper_experiment.json` has not changed since that",
        "  commit, so every submitted result uses the same settings.",
        "",
        "## Development code before the public release (local archive)",
        "",
        f"- `{legacy.relative_to(ROOT).as_posix()}` belongs to a different, earlier implementation (a dashboard backend with its own",
        "  regime classifier); its outputs in `archive/legacy_outputs/` are dated 2026-04-30. It already contains a confirmation rule of",
        "  the same form (volatility alarm on day t and an instability alarm within [t-W, t+W]) and these settings:",
        *[f"  - `{c}`" for c in constants],
        "  (`collapse_quantile` is the calibration quantile of the instability score in that code.)",
        "- The same file lists the 13 events with the same reference dates as `data/events_spy.csv`, with manually assigned severities.",
        "- That development pipeline ran outcome-scored ablations of its confirmation rule on the same 13-event panel. The F1 and FPR",
        "  below are day-level values over all days (the files call the columns `f1_mean` and `fpr_mean`):",
        f"  - instability quantile {', '.join(str(q) for q in thr['collapse_quantile'])}: F1 was highest at {best_q}, not at 0.9;",
        f"    the false positive rate fell as the quantile rose ({thr['fpr_mean'].iloc[0]:.4f} at {thr['collapse_quantile'].iloc[0]} to"
        f" {thr['fpr_mean'].iloc[-1]:.4f} at {thr['collapse_quantile'].iloc[-1]});",
        f"  - confirmation windows W = {', '.join(str(w) for w in win['window'])}: F1 was highest at W = {best_w}, not at W = 10;",
        f"    the false positive rate rose with W ({win['fpr_mean'].iloc[0]:.4f} at W = 0 to {win['fpr_mean'].iloc[-1]:.4f} at W = 60).",
        "  These ablations used the development detector, not the submitted one, so their numbers are not comparable with the paper.",
        "- The origin of the volatility percentile 0.90 and of the 70 per cent calibration split was not traced in the development",
        "  files examined.",
        "",
        "## Event panel documentation",
        "",
        "- The paper changelog dated 2026-05-21 records that the inclusion rules and source references were written into",
        "  `data/events_spy_seed.csv` after the development runs:",
        *[f"  > {e}" for e in entry],
        "- No list of candidate or excluded events, and no rule for ordering overlapping candidates, exists anywhere in the repository",
        "  or the archive (see t11_event_provenance).",
        "",
        "## What can and cannot be claimed",
        "",
        "- Can be claimed: the 13 events, their reference dates, the 60-day half-window, W = 10 and a 0.90 instability quantile were",
        "  already in the development code by 2026-04-30, and none was changed for any reported result; W = 10 and the 0.90",
        "  instability quantile were not the F1-maximizing values in the development ablations.",
        "- Cannot be claimed: that the event panel or the settings were fixed before any detector output was examined, or that they",
        "  were chosen independently of outcomes. The repository has no record of when they were first chosen, and outcome-scored",
        "  ablations on the same panel were run during development.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    inv = pd.DataFrame(INVENTORY, columns=["parameter", "value", "defined_in", "used_by", "status", "evidence"])
    inv.to_csv(OUT / "parameter_inventory.csv", index=False)

    vol_curve = pd.read_csv(OUT_ROOT / "t4_operating_curves" / "volatility_curve_spy.csv")
    conf_curve = pd.read_csv(OUT_ROOT / "t4_operating_curves" / "confirmation_curve_spy.csv")
    pick = lambda c: c[np.isin(np.round(c["quantile"], 4), ROUND_Q)][COLS]  # noqa: E731
    vol_tab = pd.concat([pick(vol_curve).assign(detector="volatility trigger"),
                         pick(conf_curve).assign(detector="confirmation rule (volatility percentile swept, instability 0.90, W=10)")])
    vol_tab.to_csv(OUT / "volatility_percentile_sensitivity_spy.csv", index=False)

    df, events = load_asset("SPY")
    scores = component_scores(df, CONFIG)
    spans = event_spans(df, events, HALF_WINDOW)
    y, _ = event_mask(df, events, HALF_WINDOW)
    vol_per = per_event_outcomes(detector_alarms(scores, CONFIG)["volatility"], spans)
    inst_curve = confirmation_curve(scores, CONFIG, y, spans, GRID, reference=vol_per, sweep="instability")
    inst_curve.to_csv(OUT / "instability_percentile_curve_spy.csv", index=False)
    inst_tab = pick(inst_curve).assign(detector="confirmation rule (instability percentile swept, volatility 0.90, W=10)")
    inst_tab.to_csv(OUT / "instability_percentile_sensitivity_spy.csv", index=False)

    note = development_history_note()
    if note is not None:
        (OUT / "development_history_note.md").write_text(note, encoding="utf-8")
    else:
        print("archive/ not present: development_history_note.md left unchanged")

    pd.set_option("display.width", 250)
    print(inv[["parameter", "status"]].to_string(index=False))
    print(vol_tab.round(4).to_string(index=False))
    print(inst_tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
