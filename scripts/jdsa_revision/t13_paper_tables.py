"""Task 13: every submitted SPY table that depends on the event windows, under the active panel.

The submitted scripts read the stored event files, so they cannot be pointed at a revised
panel without overwriting those files. This script reuses their functions instead, with the
event table passed in: ``src.paper_protocol`` for Tables 3, 10, 13 and the severity,
confirmation-window and score-component results; ``scripts/run_ablations.py`` for Tables
7, 14, 15 and 16; ``scripts/run_vix_benchmark.py`` for Table 4; and
``src.event_level_evaluation`` for Table 6. The submitted bootstrap design is rerun too and
labelled superseded. With the submitted panel every table here reproduces the committed
file (checked by compare_runs.py and the tests).

Outputs: <out root>/t13_paper_tables/
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from _common import CONFIG, PANEL, ROOT, outdir, panel_events, severity_labels, spy_events  # noqa: E402
from src.event_level_evaluation import _asset_event_rows, _write_spy_event_table  # noqa: E402
from src.multi_asset_data import _read_price_csv, preprocess_price_frame  # noqa: E402
from src.paper_protocol import (  # noqa: E402
    block_bootstrap,
    bootstrap_summary,
    cost_sensitivity,
    detector_outputs,
    event_bootstrap,
    event_mask,
    event_metrics,
    event_window_sensitivity,
    point_metrics,
    score_ablation,
    stratified_summary,
    summarize_methods,
    window_ablation,
    windowed_hybrid,
    _rolling_mean,
    _z_from_train,
    compute_instability_features,
)

OUT = outdir("t13_paper_tables")
# table name -> committed file reproduced by the submitted panel
COMMITTED = {
    "table3_spy_methods": "results/paper/table_spy_methods.csv",
    "table10_cost_sensitivity": "results/paper/table_cost_sensitivity.csv",
    "severity_day_level": "results/paper/table_spy_severity.csv",
    "confirmation_window_sensitivity": "results/paper/table_window_sensitivity.csv",
    "table13_event_window_sensitivity": "results/paper/table_event_window_sensitivity.csv",
    "score_component_sensitivity": "results/paper/table_score_component_sensitivity.csv",
    "submitted_design_bootstrap_intervals": "results/paper/table_bootstrap_intervals.csv",
    "table14_weight_ablation": "results/ablations/table_weight_ablation.csv",
    "table15_kappa_sign_ablation": "results/ablations/table_kappa_sign_ablation.csv",
    "table7_component_diagnosis": "results/ablations/table_component_diagnosis.csv",
    "table16_causal_confirmation": "results/ablations/table_causal_confirmation.csv",
    "table4_vix_benchmark": "results/ablations/table_vix_benchmark.csv",
    "table6_spy_event_level_detection": "results/event_level/table_spy_event_level_detection.csv",
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ABL = _script("run_ablations")
VIX = _script("run_vix_benchmark")


def spy_frame() -> pd.DataFrame:
    """In-memory SPY preprocessing, the same arithmetic as reproduce_paper.py (nothing written)."""
    return preprocess_price_frame(_read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]), CONFIG)


def protocol_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = spy_frame()
    _, pred = detector_outputs(df, CONFIG)
    summary, y, _ = summarize_methods(df, events, pred, CONFIG)
    boot = event_bootstrap(df, events, pred, CONFIG)._append(
        block_bootstrap(y, pred["hybrid_w10"], pred["volatility_percentile"], CONFIG), ignore_index=True)
    return {
        "table3_spy_methods": summary,
        "table10_cost_sensitivity": cost_sensitivity(summary),
        "severity_day_level": stratified_summary(df, events, pred, CONFIG),
        "confirmation_window_sensitivity": window_ablation(df, events, pred, CONFIG),
        "table13_event_window_sensitivity": event_window_sensitivity(df, events, pred, CONFIG),
        "score_component_sensitivity": score_ablation(df, events, pred, CONFIG),
        "submitted_design_bootstrap_intervals": bootstrap_summary(boot),
    }


def ablation_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables 7, 14, 15 and 16, computed as scripts/run_ablations.py computes them."""
    grid = [(0.5, 0.3, 0.2), (1 / 3, 1 / 3, 1 / 3), (0.2, 0.3, 0.5), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2), (0.5, 0.2, 0.3),
            (0.2, 0.6, 0.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    weights = []
    for w in grid:
        df, _, _, hybrid, _, _ = ABL.build_outputs(w, "abs")
        weights.append({"w_return": w[0], "w_volatility": w[1], "w_zscore": w[2], **ABL.score(df, hybrid, events)})
    kappa = []
    for mode, label in [("abs", "|z_kappa| two-sided"), ("low", "max(-z_kappa,0) low kappa"), ("high", "max(+z_kappa,0) high kappa")]:
        df, _, _, hybrid, _, _ = ABL.build_outputs((0.5, 0.3, 0.2), mode)
        kappa.append({"confirmation_term": label, **ABL.score(df, hybrid, events)})

    df, volatility, instability, hybrid, mmd_s, _ = ABL.build_outputs((0.5, 0.3, 0.2), "abs")
    y, spans = event_mask(df, events, CONFIG["events"]["half_window"])
    components = []
    for (center, lo, hi), (_, ev) in zip(spans, events.iterrows()):
        window = slice(lo, hi + 1)
        components.append({"event_name": ev["event_name"], "severity_label": ev["severity_label"], "severity_score": ev["severity_score"],
                           "volatility_detected": int(volatility[window].any()), "hybrid_detected": int(hybrid[window].any()),
                           "volatility_alarm_days": int(volatility[window].sum()), "instability_alarm_days": int(instability[window].sum()),
                           "peak_smoothed_z_mmd": float(np.nanmax(mmd_s[window]))})

    def trailing(vol, inst, window):
        out = np.zeros_like(vol)
        for t, flag in enumerate(vol):
            if flag:
                out[t] = int(inst[max(0, t - window): t + 1].any())
        return out

    lag = CONFIG["features"]["mmd_window"]
    lagged = np.zeros_like(instability)
    lagged[lag:] = instability[:-lag]
    variants = {"symmetric_w10": hybrid, "trailing_w10": trailing(volatility, instability, 10),
                "trailing_w20": trailing(volatility, instability, 20), "fully_causal_w10": trailing(volatility, lagged, 10),
                "fully_causal_w40": trailing(volatility, lagged, 40), "volatility_reference": volatility}
    causal, vol_stats = [], None
    for label, pred in variants.items():
        detected = sum(1 for c, lo, hi in spans if pred[lo: hi + 1].any())
        positive = 0.0
        for center, lo, hi in spans:
            idx = np.where(pred[lo: hi + 1] == 1)[0] + lo
            if len(idx):
                positive += max(int(idx[0] - center), 0)
        row = {"variant": label, **ABL.score(df, pred, events), "alarm_days_outside": int(((pred == 1) & (y == 0)).sum()),
               "missed_events": len(spans) - detected, "total_positive_delay": positive}
        if label == "volatility_reference":
            vol_stats = row
        causal.append(row)
    for row in causal:
        if row["variant"] == "volatility_reference":
            row["crossover_ratio_cd005"] = np.nan
            continue
        den = vol_stats["alarm_days_outside"] - row["alarm_days_outside"]
        num = (row["missed_events"] - vol_stats["missed_events"]) + 0.05 * (row["total_positive_delay"] - vol_stats["total_positive_delay"])
        row["crossover_ratio_cd005"] = float(num / den) if den else np.nan
    per_event = []
    for (center, lo, hi), (_, ev) in zip(spans, events.iterrows()):
        entry = {"event_id": ev["event_id"], "event_name": ev["event_name"]}
        for label, pred in variants.items():
            idx = np.where(pred[lo: hi + 1] == 1)[0] + lo
            entry[f"{label}_first_delay"] = float(idx[0] - center) if len(idx) else np.nan
        per_event.append(entry)
    return {"table14_weight_ablation": pd.DataFrame(weights), "table15_kappa_sign_ablation": pd.DataFrame(kappa),
            "table7_component_diagnosis": pd.DataFrame(components), "table16_causal_confirmation": pd.DataFrame(causal),
            "table16_per_event_first_delays": pd.DataFrame(per_event)}


def vix_table(events: pd.DataFrame) -> pd.DataFrame:
    """Table 4, computed as scripts/run_vix_benchmark.py computes it."""
    merged = VIX.spy_frame().merge(VIX.ensure_vix(), on="date", how="left")
    merged["vix"] = merged["vix"].ffill()
    merged = merged.dropna(subset=["vix"]).reset_index(drop=True)
    smooth, q = CONFIG["features"]["smooth_window"], CONFIG["detectors"]["volatility_threshold_quantile"]
    te = int(round(len(merged) * CONFIG["detectors"]["calibration_fraction"]))
    vix_s = _rolling_mean(_z_from_train(merged["vix"].to_numpy(float), te), smooth)
    vix_pred = (vix_s > np.quantile(vix_s[:te], q)).astype(int)
    vol_s = _rolling_mean(_z_from_train(merged["rolling_volatility"].to_numpy(float), te), smooth)
    vol_pred = (vol_s > np.quantile(vol_s[:te], q)).astype(int)
    feat = compute_instability_features(merged["signal"].to_numpy(float), CONFIG)
    score = _rolling_mean(0.5 * _z_from_train(feat["MMD"].to_numpy(float), te) + 0.5 * np.abs(_z_from_train(feat["kappa"].to_numpy(float), te)), smooth)
    inst = (score > np.quantile(score[:te], CONFIG["detectors"]["score_threshold_quantile"])).astype(int)
    w = CONFIG["detectors"]["default_hybrid_window"]
    y, spans = event_mask(merged, events, CONFIG["events"]["half_window"])
    rows = []
    for name, pred in [("VIX percentile", vix_pred), ("Volatility percentile", vol_pred), ("Hybrid (W=10)", windowed_hybrid(vol_pred, inst, w)),
                       ("VIX + instability confirmation", windowed_hybrid(vix_pred, inst, w))]:
        pm, em = point_metrics(y, pred), event_metrics(pred, spans)
        detected = sum(1 for c, lo, hi in spans if pred[lo: hi + 1].any())
        dplus = 0.0
        for center, lo, hi in spans:
            idx = np.where(pred[lo: hi + 1] == 1)[0] + lo
            if len(idx):
                dplus += max(int(idx[0] - center), 0)
        rows.append({"method": name, "precision": pm["precision"], "recall": pm["recall"], "f1": pm["f1"], "fpr": pm["fpr"],
                     "events_detected": detected, "event_coverage": em["event_coverage"], "event_delay": em["event_delay"],
                     "event_alignment_error": em["event_alignment_error"], "alarm_days_outside": int(((pred == 1) & (y == 0)).sum()),
                     "missed_events": len(spans) - detected, "total_positive_delay": dplus})
    out = pd.DataFrame(rows)
    idx = out.set_index("method")

    def crossover(broad, narrow, cd=0.05):
        den = broad["alarm_days_outside"] - narrow["alarm_days_outside"]
        num = (narrow["missed_events"] - broad["missed_events"]) + cd * (narrow["total_positive_delay"] - broad["total_positive_delay"])
        return float(num / den) if den else np.nan

    out["crossover_vs_vix_cd005"] = [np.nan if m == "VIX percentile" else crossover(idx.loc["VIX percentile"], idx.loc[m]) for m in out["method"]]
    return out


def spy_event_level_table(asset_events: pd.DataFrame, labels: dict[str, str], tmp: Path) -> pd.DataFrame:
    detail = _asset_event_rows("SPY", CONFIG, observable_only=True, events=asset_events, severity_labels=labels)
    _write_spy_event_table(detail, tmp)
    return pd.read_csv(tmp / "table_spy_event_level_detection.csv")


def all_tables(spy_table: pd.DataFrame, spy_asset_events: pd.DataFrame, labels: dict[str, str], tmp: Path) -> dict[str, pd.DataFrame]:
    tables = {**protocol_tables(spy_table), **ablation_tables(spy_table), "table4_vix_benchmark": vix_table(spy_table)}
    tables["table6_spy_event_level_detection"] = spy_event_level_table(spy_asset_events, labels, tmp)
    return tables


def main() -> None:
    tables = all_tables(spy_events(), panel_events("SPY"), severity_labels(), OUT)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "table_spy_event_level_detection.csv").unlink(missing_ok=True)
    pd.DataFrame([{"table": k, "file": f"{k}.csv", "submitted_artifact": v, "event_panel": PANEL} for k, v in COMMITTED.items()]).to_csv(
        OUT / "index.csv", index=False)
    pd.set_option("display.width", 220)
    print(tables["table3_spy_methods"][["method", "precision", "recall", "f1", "fpr", "event_coverage", "event_delay", "event_alignment_error"]].round(4).to_string(index=False))
    print(tables["table4_vix_benchmark"].drop(columns=["event_alignment_error"]).round(4).to_string(index=False))
    print(tables["table16_causal_confirmation"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
