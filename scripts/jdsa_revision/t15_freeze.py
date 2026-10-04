"""Task 15: write the numerical freeze, FROZEN_RESULTS.md, from the outputs of this run.

Every number in the document is read from a frozen output file; nothing is typed in.
Later manuscript edits take their values from these outputs (and from final_audit.csv),
not from new computations.

Also verifies the nesting A_confirmation within A_volatility on every asset and window.

Outputs: <out root>/FROZEN_RESULTS.md, <out root>/t15_freeze/nesting_check.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import CONF_NAME, CONFIG, OUT_ROOT, PANEL, TICKERS, VOL_NAME, load_asset, outdir  # noqa: E402
from src.paper_protocol import detector_outputs  # noqa: E402
from src.revision_analysis import component_scores, detector_alarms  # noqa: E402

OUT = outdir("t15_freeze")


def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(OUT_ROOT / rel, low_memory=False)


def js(rel: str) -> dict:
    return json.loads((OUT_ROOT / rel).read_text(encoding="utf-8"))


def f3(x) -> str:
    return "NA" if pd.isna(x) else f"{x:.3f}"


def f1(x) -> str:
    return "NA" if pd.isna(x) else f"{x:.1f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def nesting() -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        df, _ = load_asset(ticker)
        _, pred = detector_outputs(df, CONFIG)
        v = pred["volatility_percentile"].astype(bool)
        for w in CONFIG["detectors"]["hybrid_windows"]:
            h = pred[f"hybrid_w{w}"].astype(bool)
            rows.append({"ticker": ticker, "rule": f"symmetric confirmation W={w}", "alarm_days": int(h.sum()), "volatility_alarm_days": int(v.sum()),
                         "days_alarmed_but_not_by_volatility": int((h & ~v).sum())})
        alarms = detector_alarms(component_scores(df, CONFIG), CONFIG)
        for rule in ["trailing_confirmation", "fully_causal_confirmation"]:
            h = alarms[rule].astype(bool)
            rows.append({"ticker": ticker, "rule": rule.replace("_", " ") + " W=10", "alarm_days": int(h.sum()),
                         "volatility_alarm_days": int(v.sum()), "days_alarmed_but_not_by_volatility": int((h & ~v).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    nest = nesting()
    nest.to_csv(OUT / "nesting_check.csv", index=False)
    panel = js("t0_event_panel/panel_version.json")
    cut = js("t2_evaluation_periods/calibration_cutoff.json")
    op = js("t2_evaluation_periods/operational_start.json")
    den = csv("t1_denominators/denominators_by_asset.csv")
    t3 = csv("t13_paper_tables/table3_spy_methods.csv").set_index("method")
    a = csv("t2_evaluation_periods/table_A_post_calibration_retrospective.csv").set_index("detector")
    b = csv("t2_evaluation_periods/table_B_prospective_operational.csv").set_index("detector")
    ac = csv("t2_evaluation_periods/table_A_post_calibration_retrospective_crossover.csv")
    bc = csv("t2_evaluation_periods/table_B_prospective_operational_crossover.csv")
    full_c = csv("t1_denominators/table12_cost_crossover_observable.csv")
    t8 = csv("t1_denominators/table8_multi_asset_event_level_observable.csv")
    match = pd.concat([csv("t4_operating_curves/matched_comparison_spy.csv"), csv("t4_operating_curves/matched_comparison_other_assets.csv")])
    ce = csv("t5_common_event_delay/common_event_delay_summary.csv")
    unc = csv("t6_uncertainty/uncertainty_summary.csv")
    sens = csv("t6_uncertainty/block_length_sensitivity.csv")
    cu = js("t7_cusum/cusum_diagnostic.json")
    zd = csv("t8_z_dominance/z_dominance_alarm_differences.csv")
    zs = csv("t8_z_dominance/z_dominance_scale_summary.csv")
    zev = csv("t8_z_dominance/z_dominance_event_differences.csv")
    mech = csv("t9_mechanism/mechanism_manuscript_comparison.csv").set_index("quantity")
    prov = csv("t11_event_provenance/provenance_summary.csv").iloc[0]
    gaps = csv("t11_event_provenance/adjacent_window_gaps.csv")
    audit = csv("final_audit.csv")
    changed = audit[audit["changed"] & audit["kind"].isin(["table cell", "text statement"])]
    e03 = csv("pre_E03_fix_vs_post_E03_fix_summary.csv")
    e03_cells = csv("pre_E03_fix_vs_post_E03_fix.csv")
    bulk_cells = csv("pre_E03_fix_vs_post_E03_fix_curves_and_samples.csv")

    def period_table(t: pd.DataFrame, rules: list[str]) -> str:
        return md_table(pd.DataFrame([{
            "Detector": r, "Precision": f3(t.loc[r, "day_precision"]), "Recall": f3(t.loc[r, "day_recall"]), "F1": f3(t.loc[r, "day_f1"]),
            "FPR": f"{t.loc[r, 'day_fpr']:.4f}", "Alarm days": int(t.loc[r, "day_alarm_days"]),
            "Events": f"{int(t.loc[r, 'events_detected'])}/{int(t.loc[r, 'n_events'])} ({t.loc[r, 'events_detected_ids'] if isinstance(t.loc[r, 'events_detected_ids'], str) else ''})",
            "A_out": int(t.loc[r, "A_out"]), "M": int(t.loc[r, "M"]), "D_plus": int(t.loc[r, "D_plus"]),
        } for r in rules]))

    m = match[(match["target_column"] == "fpr") & (match["match"] == "nearest")].set_index("ticker")
    d = match[match["operating_point"].str.startswith("confirmation default")].set_index("ticker")
    matched = md_table(pd.DataFrame([{
        "Asset": t, "Matched quantile": f"{m.loc[t, 'quantile_min']:.4f}", "FPR (vol / conf)": f"{m.loc[t, 'fpr']:.4f} / {d.loc[t, 'fpr']:.4f}",
        "Precision": f"{f3(m.loc[t, 'precision'])} / {f3(d.loc[t, 'precision'])}", "Recall": f"{f3(m.loc[t, 'recall'])} / {f3(d.loc[t, 'recall'])}",
        "F1": f"{f3(m.loc[t, 'f1'])} / {f3(d.loc[t, 'f1'])}",
        "Events": f"{int(m.loc[t, 'events_detected'])} / {int(d.loc[t, 'events_detected'])} of {int(d.loc[t, 'n_events'])}",
        "Common-event delay": f"{f1(m.loc[t, 'common_mean_delay_this'])} / {f1(m.loc[t, 'common_mean_delay_default_confirmation'])} "
                              f"(n={int(m.loc[t, 'common_events_with_default_confirmation'])})",
    } for t in TICKERS]))

    base = ce[ce["comparison"] == "volatility vs confirmation (submitted defaults)"].set_index("ticker")
    delays = md_table(pd.DataFrame([{
        "Asset": t, "Common events": int(base.loc[t, "n_common"]),
        "Mean (vol / conf)": f"{base.loc[t, 'common_mean_delay_a']:.2f} / {base.loc[t, 'common_mean_delay_b']:.2f}",
        "Median (vol / conf)": f"{base.loc[t, 'common_median_delay_a']:.1f} / {base.loc[t, 'common_median_delay_b']:.1f}",
        "Paired mean difference": f"{base.loc[t, 'paired_mean_difference']:.2f}",
        "Own-set means (vol / conf)": f"{base.loc[t, 'own_set_mean_delay_a']:.2f} / {base.loc[t, 'own_set_mean_delay_b']:.2f}",
        "Detected by volatility only": base.loc[t, "events_only_a"] if isinstance(base.loc[t, "events_only_a"], str) else "",
    } for t in TICKERS]))
    other = ce[ce["comparison"] != "volatility vs confirmation (submitted defaults)"]
    other_md = md_table(other[["comparison", "event_scope", "n_common", "common_mean_delay_a", "common_mean_delay_b", "paired_mean_difference"]]
                        .assign(common_mean_delay_a=lambda x: x["common_mean_delay_a"].map(f1), common_mean_delay_b=lambda x: x["common_mean_delay_b"].map(f1),
                                paired_mean_difference=lambda x: x["paired_mean_difference"].map(f1)))

    def interval_table(comp: str) -> str:
        sub = unc[unc["comparison"].str.startswith(comp) & ~unc["estimand"].str.contains("second seed")]
        rows = []
        for _, r in sub.iterrows():
            if not (pd.isna(r["block_length"]) or r["block_length"] == 121):
                continue
            rows.append({"Asset": r["ticker"], "Estimand": r["estimand"].replace(" (b - a)", ""), "n": int(r["n_units"]),
                         "Observed": f3(r["observed"]), "95% interval": f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}]",
                         "Sign fixed by nesting": "yes" if r["sign_fixed_by_nesting"] else "no"})
        return md_table(pd.DataFrame(rows))

    conf_rows = zev[zev["detector"] == "confirmation (W=10)"]
    conf_flips = int((conf_rows["detected_composite"] != conf_rows["detected_z_only"]).sum())
    flips = sens[~sens["same_conclusion_at_all_block_lengths"]]
    flip_lines = "\n".join(f"- {r.ticker}, {r.comparison}, {r.estimand}: excludes zero at 20/60/121 = "
                           f"{r.excludes_zero_block20}/{r.excludes_zero_block60}/{r.excludes_zero_block121}" for r in flips.itertuples()) or "- none"

    crossover = md_table(full_c.pivot(index="ticker", columns="delay_cost", values="crossover_ratio").reindex(TICKERS)
                         .apply(lambda col: col.map(f3)).reset_index().rename(columns={0.0: "c_D=0", 0.05: "c_D=0.05", 0.1: "c_D=0.1"}))
    nest_viol = int(nest["days_alarmed_but_not_by_volatility"].sum())
    by_loc = changed.groupby("location").size().reset_index(name="numbers to change")
    e03_files = e03[e03["cells_changed"] > 0][["file", "cells_changed"]]
    lines = [
        "# JDSA revision: frozen numerical results",
        "",
        "Generated by `scripts/jdsa_revision/t15_freeze.py` from the outputs in this folder. Every number below is read from a",
        "frozen output file. Manuscript and response-letter edits must take their values from these outputs and from",
        "`final_audit.csv`, not from new computations.",
        "",
        "## Inputs",
        "",
        f"- Event panel: `{PANEL}` (seed `{panel['seed_file']}`), {panel['events']} events.",
        f"- Events changed from the submitted panel: {', '.join(panel['events_changed']) or 'none'}. "
        f"Severity labels changed: {', '.join(panel['severity_labels_changed']) or 'none'}.",
        *[f"- {k}: {v}" for k, v in panel["revision_notes"].items()],
        f"- Nominal calibration cutoff: last calibration day {cut['last_calibration_date']} (index {cut['train_end_index']} of {cut['n_days']}); "
        f"first post-calibration day {cut['first_post_calibration_date']}.",
        f"- Latest raw close read by any fitted constant or threshold: {op['latest_calibration_close_date']}. "
        f"Operational evaluation start: {op['operational_start_date']} (derived by perturbation search; perturbing every close from that "
        f"day leaves all calibration quantities identical: {op['check_all_closes_from_start_perturbed_leaves_calibration_identical']}).",
        "- Observable-event denominators: " + ", ".join(f"{r.ticker} {r.observable_events}" for r in den.itertuples()) + ".",
        f"- Nesting: confirmation alarms that are not volatility alarms, all assets and windows: {nest_viol}. Under nesting, recall, FPR, "
        "detection rate, window coverage and missed-event counts can only move in one direction, as can the first-alarm delay on any "
        "event both rules detect. Precision, F1, the size of every gap, own-set mean delays, D_plus and costs are empirical.",
        "",
        "## Full-period retrospective result (SPY, Table 3 rows)",
        "",
        md_table(pd.DataFrame([{"Detector": k, "Precision": f3(t3.loc[k, "precision"]), "Recall": f3(t3.loc[k, "recall"]),
                                "F1": f3(t3.loc[k, "f1"]), "FPR": f3(t3.loc[k, "fpr"]), "Detection rate": f3(t3.loc[k, "event_coverage"]),
                                "Mean first delay": f1(t3.loc[k, "event_delay"]), "Alignment error": f1(t3.loc[k, "event_alignment_error"])}
                               for k in ["volatility_percentile", "hybrid_windowed_confirmation", "cusum"]])),
        "",
        "## Table A: post-calibration retrospective evaluation (SPY)",
        "",
        f"Scored days {a.iloc[0]['first_scored_date']} to {a.iloc[0]['last_scored_date']}; events {a.iloc[0]['events_in_scope']}. "
        "Retrospective: the symmetric confirmation rule reads days after each alarm.",
        "",
        period_table(a, ["volatility", "retrospective_confirmation"]),
        "",
        f"Common events: {int(a.iloc[0]['common_event_n_common'])}; mean first delay volatility {f1(a.iloc[0]['common_event_common_mean_delay_a'])}, "
        f"confirmation {f1(a.iloc[0]['common_event_common_mean_delay_b'])}. Crossover r*: "
        + ", ".join(f"c_D={r.delay_cost}: {f3(r.crossover_ratio)}" for r in ac.itertuples()) + ".",
        "",
        "## Table B: prospective operational evaluation (SPY)",
        "",
        f"Scored days {b.iloc[0]['first_scored_date']} to {b.iloc[0]['last_scored_date']}; events {b.iloc[0]['events_in_scope']}. "
        "Only calibration quantities frozen before the first scored day are used.",
        "",
        period_table(b, ["volatility", "fully_causal_confirmation"]),
        "",
        f"Common events: {int(b.iloc[0]['common_event_n_common'])}; mean first delay volatility {f1(b.iloc[0]['common_event_common_mean_delay_a'])}, "
        f"fully causal {f1(b.iloc[0]['common_event_common_mean_delay_b'])}. Crossover r*: "
        + ", ".join(f"c_D={r.delay_cost}: {f3(r.crossover_ratio)}" for r in bc.itertuples()) + ".",
        "",
        "## Matched-FPR comparison (volatility trigger at the confirmation rule's FPR; all five assets)",
        "",
        matched,
        "",
        "Values are volatility at the matched point / confirmation at its default. The matched point is a post-hoc diagnostic "
        "requested by Reviewer 1, not a deployment threshold.",
        "",
        "## Common-event first-alarm delays (submitted defaults, observable events)",
        "",
        delays,
        "",
        other_md,
        "",
        "## Uncertainty (2000 resamples; primary block length 121 trading days; evaluation-sample variability only)",
        "",
        "### Nested default comparison",
        "",
        interval_table("volatility default"),
        "",
        "### Matched-FPR comparison",
        "",
        interval_table("volatility at matched FPR"),
        "",
        "Day-level intervals whose conclusion (excludes zero or not) changes with block length:",
        flip_lines,
        "",
        "## Event-aware cost crossover r* (full period, observable events)",
        "",
        crossover,
        "",
        "## Diagnostics",
        "",
        f"- CUSUM (submitted implementation, no reference value, no reset): calibration alarm fraction {cu['calibration_alarm_fraction']:.3f}, "
        f"post-calibration {cu['post_calibration_alarm_fraction']:.3f}; {cu['alarm_episodes_full_period']} episodes, longest {cu['longest_episode_days']} days; "
        f"persistent alarm from {cu['persistent_alarm_state_first_day']}; {cu['detections_with_first_alarm_at_window_start']} of "
        f"{cu['events_detected']} detections start at the window's first day.",
        f"- z-dominance: corr(x_t, z_t) from {zs['corr_x_z'].min():.5f} to {zs['corr_x_z'].max():.5f}; confirmation alarm days differing "
        "between x_t and z_t: " + ", ".join(f"{r.ticker} {r.differing_days}" for r in zd[zd['detector'] == 'confirmation (W=10)'].itertuples())
        + f"; events whose detection differs for the confirmation rule: {conf_flips}.",
        f"- Mechanism: reference dispersion ratios (confirmed events with a full pre-window: {mech.loc['reference events (confirmed, full pre-window)', 'recomputed']}) "
        f"{float(mech.loc['reference mean ratio raw volatility', 'recomputed']):.2f} for raw volatility and {float(mech.loc['reference mean ratio x_t', 'recomputed']):.2f} for x_t; "
        f"U.S. downgrade {float(mech.loc['E07 ratio raw volatility', 'recomputed']):.2f} and {float(mech.loc['E07 ratio x_t', 'recomputed']):.2f}; "
        f"inflation selloff {float(mech.loc['E12 ratio raw volatility', 'recomputed']):.2f} and {float(mech.loc['E12 ratio x_t', 'recomputed']):.2f}; "
        f"confirmed events with a lower x_t ratio than the downgrade: {mech.loc['confirmed events with x_t ratio below E07', 'recomputed']}.",
        "",
        "## Event-date provenance",
        "",
        f"- Events with an official (tier 1) source: {prov['events_with_official_source_tier1']} of {prov['events']}; resting on contemporaneous "
        f"reporting only: {prov['events_resting_on_contemporaneous_reporting_only_tier4'] if isinstance(prov['events_resting_on_contemporaneous_reporting_only_tier4'], str) else 'none'}.",
        f"- Reference dates changed from the submitted panel: {prov['reference_dates_changed_from_submitted']}.",
        f"- Adjacent SPY event windows: smallest gap {int(gaps['trading_days_between_windows'].min())} trading days; any overlap: {bool(gaps['windows_overlap'].any())}.",
        "- No candidate list and no overlap-priority rule exist in the repository or the local archive.",
        "",
        "## Effect of the E03 correction",
        "",
        f"`pre_E03_fix_vs_post_E03_fix.csv` lists every revision result value that differs between the submitted-panel run and this "
        f"run ({len(e03_cells)} values). The {int(len(bulk_cells))} changed values of the full operating curves and raw bootstrap resamples "
        "are in `pre_E03_fix_vs_post_E03_fix_curves_and_samples.csv`. Changed values per file (results, curves and resamples):",
        "",
        md_table(e03_files),
        "",
        "## Manuscript numbers that must change",
        "",
        f"`final_audit.csv` lists every audited table cell and in-text statement; {len(changed)} printed numbers or statements differ "
        "from the frozen values (`t12_audit/manuscript_numbers_to_change.csv`), by location:",
        "",
        md_table(by_loc),
        "",
    ]
    (OUT_ROOT / "FROZEN_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"nesting violations: {nest_viol}")
    print(f"wrote {OUT_ROOT / 'FROZEN_RESULTS.md'}")


if __name__ == "__main__":
    main()
