"""Regenerate the ablation and diagnostic tables reported in the manuscript.

Produces, under results/ablations/:
  table_weight_ablation.csv        composite weight ablation (manuscript Table 13)
  table_kappa_sign_ablation.csv    AR(1) sign convention ablation (Table 14)
  table_component_diagnosis.csv    per-event component diagnosis (Table 6)
  table_scale_diagnostic.csv       dispersion of each weighted composite term
  table_cost_crossover.csv         event-aware cost crossover ratios (Table 12)
  table_causal_confirmation.csv    trailing-only and fully causal hybrid variants

Run from the repository root:
    python scripts/run_ablations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_protocol import (  # noqa: E402
    _rolling_mean,
    _z_from_train,
    compute_instability_features,
    event_mask,
    event_metrics,
    load_config,
    point_metrics,
    windowed_hybrid,
)

CONFIG = load_config(ROOT / "config" / "paper_experiment.json")
OUTDIR = ROOT / "results" / "ablations"


def preprocess(weights: tuple[float, float, float]) -> pd.DataFrame:
    """Deterministic SPY preprocessing with configurable composite weights."""
    spy = CONFIG["spy"]
    df = pd.read_csv(ROOT / "data" / "raw_market" / "spy.csv")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["date"] = pd.to_datetime(df["Date"])
    col = spy["price_column"]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", col]).sort_values("date")
    df = df[["date", col]].rename(columns={col: "price"})
    df["log_return"] = np.log(df["price"]).diff()
    df["rolling_volatility"] = df["log_return"].rolling(spy["rolling_vol_window"]).std()
    mean = df["log_return"].rolling(spy["zscore_window"]).mean()
    std = df["log_return"].rolling(spy["zscore_window"]).std()
    df["return_zscore"] = (df["log_return"] - mean) / std
    df["signal"] = (
        weights[0] * df["log_return"]
        + weights[1] * df["rolling_volatility"]
        + weights[2] * df["return_zscore"]
    )
    return df.dropna().reset_index(drop=True)


def build_outputs(weights, kappa_mode="abs"):
    """Return the frame, volatility alarms, instability alarms, hybrid alarms and z-scores."""
    df = preprocess(weights)
    feat = compute_instability_features(df["signal"].to_numpy(float), CONFIG)
    train_end = int(round(len(df) * CONFIG["detectors"]["calibration_fraction"]))

    mmd_z = _z_from_train(feat["MMD"].to_numpy(float), train_end)
    kappa_z = _z_from_train(feat["kappa"].to_numpy(float), train_end)
    if kappa_mode == "abs":
        kappa_term = np.abs(kappa_z)
    elif kappa_mode == "low":
        kappa_term = np.maximum(-kappa_z, 0.0)
    elif kappa_mode == "high":
        kappa_term = np.maximum(kappa_z, 0.0)
    else:
        raise ValueError(f"unknown kappa_mode: {kappa_mode}")

    smooth = CONFIG["features"]["smooth_window"]
    quantile = CONFIG["detectors"]["score_threshold_quantile"]
    score = _rolling_mean(0.5 * mmd_z + 0.5 * kappa_term, smooth)
    instability = (score > np.quantile(score[:train_end], quantile)).astype(int)

    vol = _rolling_mean(_z_from_train(df["rolling_volatility"].to_numpy(float), train_end), smooth)
    volatility = (vol > np.quantile(vol[:train_end], CONFIG["detectors"]["volatility_threshold_quantile"])).astype(int)

    hybrid = windowed_hybrid(volatility, instability, CONFIG["detectors"]["default_hybrid_window"])
    return df, volatility, instability, hybrid, _rolling_mean(mmd_z, smooth), kappa_z


def score(df, hybrid, events):
    y, spans = event_mask(df, events, CONFIG["events"]["half_window"])
    detected = sum(1 for center, lo, hi in spans if hybrid[lo : hi + 1].any())
    return {**point_metrics(y, hybrid), **event_metrics(hybrid, spans), "events_detected": detected}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(ROOT / "data" / "events_spy.csv")

    # ---- scale diagnostic -------------------------------------------------
    base = preprocess((0.5, 0.3, 0.2))
    scale = pd.DataFrame([
        {"term": "0.5 * log_return", "sd": float((0.5 * base["log_return"]).std())},
        {"term": "0.3 * rolling_volatility", "sd": float((0.3 * base["rolling_volatility"]).std())},
        {"term": "0.2 * return_zscore", "sd": float((0.2 * base["return_zscore"]).std())},
        {"term": "composite x_t", "sd": float(base["signal"].std())},
    ])
    scale["corr_with_composite"] = [
        float(base["signal"].corr(base["log_return"])),
        float(base["signal"].corr(base["rolling_volatility"])),
        float(base["signal"].corr(base["return_zscore"])),
        1.0,
    ]
    scale.to_csv(OUTDIR / "table_scale_diagnostic.csv", index=False)

    # ---- weight ablation --------------------------------------------------
    grid = [
        (0.5, 0.3, 0.2), (1 / 3, 1 / 3, 1 / 3), (0.2, 0.3, 0.5), (0.6, 0.2, 0.2),
        (0.4, 0.4, 0.2), (0.5, 0.2, 0.3), (0.2, 0.6, 0.2),
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    ]
    rows = []
    for w in grid:
        df, _, _, hybrid, _, _ = build_outputs(w, "abs")
        rows.append({"w_return": w[0], "w_volatility": w[1], "w_zscore": w[2], **score(df, hybrid, events)})
    pd.DataFrame(rows).to_csv(OUTDIR / "table_weight_ablation.csv", index=False)

    # ---- kappa sign ablation ---------------------------------------------
    rows = []
    for mode, label in [("abs", "|z_kappa| two-sided"),
                        ("low", "max(-z_kappa,0) low kappa"),
                        ("high", "max(+z_kappa,0) high kappa")]:
        df, _, _, hybrid, _, _ = build_outputs((0.5, 0.3, 0.2), mode)
        rows.append({"confirmation_term": label, **score(df, hybrid, events)})
    pd.DataFrame(rows).to_csv(OUTDIR / "table_kappa_sign_ablation.csv", index=False)

    # ---- per-event component diagnosis -----------------------------------
    df, volatility, instability, hybrid, mmd_s, kappa_z = build_outputs((0.5, 0.3, 0.2), "abs")
    _, spans = event_mask(df, events, CONFIG["events"]["half_window"])
    rows = []
    for (center, lo, hi), (_, ev) in zip(spans, events.iterrows()):
        window = slice(lo, hi + 1)
        rows.append({
            "event_name": ev["event_name"],
            "severity_label": ev["severity_label"],
            "severity_score": ev["severity_score"],
            "volatility_detected": int(volatility[window].any()),
            "hybrid_detected": int(hybrid[window].any()),
            "volatility_alarm_days": int(volatility[window].sum()),
            "instability_alarm_days": int(instability[window].sum()),
            "peak_smoothed_z_mmd": float(np.nanmax(mmd_s[window])),
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "table_component_diagnosis.csv", index=False)

    # ---- cost crossover ---------------------------------------------------
    cost = pd.read_csv(ROOT / "results" / "cost" / "event_aware_cost_by_asset.csv")
    vol_name = "Volatility percentile"
    hyb_name = "Hybrid volatility + instability confirmation"
    rows = []
    for ticker in cost["ticker"].unique():
        for delay_cost in sorted(cost["delay_cost"].unique()):
            sub = cost[(cost["ticker"] == ticker) & (cost["delay_cost"] == delay_cost)]
            if sub.empty:
                continue
            v = sub[sub["method"] == vol_name].iloc[0]
            h = sub[sub["method"] == hyb_name].iloc[0]
            denominator = v["false_alarm_days"] - h["false_alarm_days"]
            numerator = ((h["missed_events"] - v["missed_events"])
                         + delay_cost * (h["total_positive_delay"] - v["total_positive_delay"]))
            rows.append({
                "ticker": ticker, "delay_cost": delay_cost,
                "alarm_days_outside_volatility": int(v["false_alarm_days"]),
                "alarm_days_outside_hybrid": int(h["false_alarm_days"]),
                "missed_volatility": int(v["missed_events"]),
                "missed_hybrid": int(h["missed_events"]),
                "positive_delay_volatility": float(v["total_positive_delay"]),
                "positive_delay_hybrid": float(h["total_positive_delay"]),
                "crossover_ratio": float(numerator / denominator) if denominator else np.nan,
            })
    pd.DataFrame(rows).to_csv(OUTDIR / "table_cost_crossover.csv", index=False)

    # ---- causal confirmation ablation ------------------------------------
    def trailing_hybrid(volatility, instability, window):
        """Causal variant: confirm only with instability in [t-window, t]."""
        out = np.zeros_like(volatility)
        for t, flag in enumerate(volatility):
            if flag:
                out[t] = int(instability[max(0, t - window) : t + 1].any())
        return out

    df, volatility, instability, hybrid, _, _ = build_outputs((0.5, 0.3, 0.2), "abs")
    y, spans = event_mask(df, events, CONFIG["events"]["half_window"])
    # Lag instability by the MMD post-window so no component uses future data.
    mmd_window = CONFIG["features"]["mmd_window"]
    instability_lagged = np.zeros_like(instability)
    instability_lagged[mmd_window:] = instability[:-mmd_window]
    variants = {
        "symmetric_w10": hybrid,
        "trailing_w10": trailing_hybrid(volatility, instability, 10),
        "trailing_w20": trailing_hybrid(volatility, instability, 20),
        "fully_causal_w10": trailing_hybrid(volatility, instability_lagged, 10),
        "fully_causal_w40": trailing_hybrid(volatility, instability_lagged, 40),
        "volatility_reference": volatility,
    }
    rows = []
    vol_stats = None
    for label, pred in variants.items():
        alarm_days_outside = int(((pred == 1) & (y == 0)).sum())
        missed = 13 - sum(1 for center, lo, hi in spans if pred[lo : hi + 1].any())
        positive_delay = 0.0
        for center, lo, hi in spans:
            idx = np.where(pred[lo : hi + 1] == 1)[0] + lo
            if len(idx):
                positive_delay += max(int(idx[0] - center), 0)
        row = {
            "variant": label, **score(df, pred, events),
            "alarm_days_outside": alarm_days_outside,
            "missed_events": missed,
            "total_positive_delay": positive_delay,
        }
        if label == "volatility_reference":
            vol_stats = row
        rows.append(row)
    # Crossover of each hybrid variant against the volatility trigger at c_D = 0.05.
    for row in rows:
        if row["variant"] == "volatility_reference":
            row["crossover_ratio_cd005"] = np.nan
            continue
        denominator = vol_stats["alarm_days_outside"] - row["alarm_days_outside"]
        numerator = ((row["missed_events"] - vol_stats["missed_events"])
                     + 0.05 * (row["total_positive_delay"] - vol_stats["total_positive_delay"]))
        row["crossover_ratio_cd005"] = float(numerator / denominator) if denominator else np.nan
    pd.DataFrame(rows).to_csv(OUTDIR / "table_causal_confirmation.csv", index=False)

    print(f"Wrote 6 ablation tables to {OUTDIR}")


if __name__ == "__main__":
    main()
