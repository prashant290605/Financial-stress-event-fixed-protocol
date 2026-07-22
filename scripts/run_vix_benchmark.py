"""Evaluate a VIX percentile trigger under the event-fixed protocol.

The volatility and hybrid rules in the paper are constructed for this study.
A referee is entitled to ask how the protocol treats an instrument that
practitioners actually deploy. The Cboe Volatility Index is the market
standard gauge of expected near-term equity volatility, so we run a VIX
percentile trigger through exactly the same pipeline: calibrate on the first
70% of the aligned series, standardize on that segment, smooth with the same
five-point trailing mean, and alarm above the 90th calibration percentile.

Unlike the study's realized-volatility trigger, VIX is forward looking (it is
priced from options) and is a level rather than a self-normalizing quantity,
so it is a genuinely different instrument rather than a reparameterization.

Outputs:
  data/raw_market/VIX.csv                     archived snapshot
  results/ablations/table_vix_benchmark.csv   pointwise + event metrics + cost terms

Run from the repository root:
    python scripts/run_vix_benchmark.py
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.paper_protocol import (
    _rolling_mean,
    _z_from_train,
    compute_instability_features,
    event_mask,
    event_metrics,
    load_config,
    point_metrics,
    windowed_hybrid,
)

CONFIG = load_config(PROJECT_ROOT / "config" / "paper_experiment.json")
RAW = PROJECT_ROOT / "data" / "raw_market"
OUTDIR = PROJECT_ROOT / "results" / "ablations"
VIX_CSV = RAW / "VIX.csv"


def ensure_vix() -> pd.DataFrame:
    """Load the archived VIX snapshot, downloading it once if absent."""
    if not VIX_CSV.exists():
        import yfinance as yf

        data = yf.download("^VIX", start="2000-01-01", end="2023-12-30",
                           auto_adjust=False, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        out = pd.DataFrame({
            "Date": pd.to_datetime(data["Date"]).dt.date.astype(str),
            "Close": data["Close"],
        })
        RAW.mkdir(parents=True, exist_ok=True)
        out.to_csv(VIX_CSV, index=False)
        print(f"Downloaded VIX snapshot to {VIX_CSV}")
    df = pd.read_csv(VIX_CSV)
    df["date"] = pd.to_datetime(df["Date"])
    df["vix"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["date", "vix"])[["date", "vix"]]


def spy_frame() -> pd.DataFrame:
    """The paper's deterministic SPY preprocessing."""
    spy = CONFIG["spy"]
    df = pd.read_csv(RAW / "spy.csv")
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
    w = spy["signal_weights"]
    df["signal"] = (w["log_return"] * df["log_return"]
                    + w["rolling_volatility"] * df["rolling_volatility"]
                    + w["return_zscore"] * df["return_zscore"])
    return df.dropna().reset_index(drop=True)


def main() -> None:
    events = pd.read_csv(PROJECT_ROOT / "data" / "events_spy.csv")
    df = spy_frame()
    vix = ensure_vix()

    # Align VIX onto the SPY trading calendar; the protocol is defined on the
    # asset's own calendar, so SPY dates govern.
    merged = df.merge(vix, on="date", how="left")
    missing = int(merged["vix"].isna().sum())
    merged["vix"] = merged["vix"].ffill()
    merged = merged.dropna(subset=["vix"]).reset_index(drop=True)

    smooth = CONFIG["features"]["smooth_window"]
    q = CONFIG["detectors"]["volatility_threshold_quantile"]
    train_end = int(round(len(merged) * CONFIG["detectors"]["calibration_fraction"]))

    # VIX percentile trigger, identical rule shape to the volatility trigger.
    vix_s = _rolling_mean(_z_from_train(merged["vix"].to_numpy(float), train_end), smooth)
    vix_pred = (vix_s > np.quantile(vix_s[:train_end], q)).astype(int)

    # Study detectors recomputed on the same aligned frame for a fair contrast.
    vol_s = _rolling_mean(_z_from_train(merged["rolling_volatility"].to_numpy(float), train_end), smooth)
    vol_pred = (vol_s > np.quantile(vol_s[:train_end], q)).astype(int)

    feat = compute_instability_features(merged["signal"].to_numpy(float), CONFIG)
    mmd_z = _z_from_train(feat["MMD"].to_numpy(float), train_end)
    kappa_z = _z_from_train(feat["kappa"].to_numpy(float), train_end)
    score = _rolling_mean(0.5 * mmd_z + 0.5 * np.abs(kappa_z), smooth)
    inst = (score > np.quantile(score[:train_end], CONFIG["detectors"]["score_threshold_quantile"])).astype(int)
    hyb_pred = windowed_hybrid(vol_pred, inst, CONFIG["detectors"]["default_hybrid_window"])

    # A VIX-gated hybrid: does confirmation help a deployed instrument too?
    vix_hyb = windowed_hybrid(vix_pred, inst, CONFIG["detectors"]["default_hybrid_window"])

    y, spans = event_mask(merged, events, CONFIG["events"]["half_window"])
    rows = []
    for name, pred in [("VIX percentile", vix_pred),
                       ("Volatility percentile", vol_pred),
                       ("Hybrid (W=10)", hyb_pred),
                       ("VIX + instability confirmation", vix_hyb)]:
        pm = point_metrics(y, pred)
        em = event_metrics(pred, spans)
        detected = sum(1 for c, lo, hi in spans if pred[lo:hi + 1].any())
        a_out = int(((pred == 1) & (y == 0)).sum())
        dplus = 0.0
        for center, lo, hi in spans:
            idx = np.where(pred[lo:hi + 1] == 1)[0] + lo
            if len(idx):
                dplus += max(int(idx[0] - center), 0)
        rows.append({
            "method": name,
            "precision": pm["precision"], "recall": pm["recall"],
            "f1": pm["f1"], "fpr": pm["fpr"],
            "events_detected": detected,
            "event_coverage": em["event_coverage"],
            "event_delay": em["event_delay"],
            "event_alignment_error": em["event_alignment_error"],
            "alarm_days_outside": a_out,
            "missed_events": 13 - detected,
            "total_positive_delay": dplus,
        })
    out = pd.DataFrame(rows)

    # Crossover of each confirmation-style rule against each broad trigger.
    def crossover(broad, narrow, cd=0.05):
        den = broad["alarm_days_outside"] - narrow["alarm_days_outside"]
        num = ((narrow["missed_events"] - broad["missed_events"])
               + cd * (narrow["total_positive_delay"] - broad["total_positive_delay"]))
        return float(num / den) if den else np.nan

    idx = out.set_index("method")
    out["crossover_vs_vix_cd005"] = [
        np.nan if m == "VIX percentile" else crossover(idx.loc["VIX percentile"], idx.loc[m])
        for m in out["method"]
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "table_vix_benchmark.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"aligned trading days: {len(merged)} (VIX values carried forward on {missing} SPY dates)")
    print(out.drop(columns=["event_alignment_error"]).to_string(index=False))

    # Which events does VIX catch that the study detectors miss, and vice versa?
    print("\nper-event detection (1 = detected in its +/-60 window):")
    print(f"{'event':28s} {'VIX':>4s} {'Vol':>4s} {'Hyb':>4s}")
    for (center, lo, hi), name in zip(spans, events["event_name"]):
        f = lambda p: int(p[lo:hi + 1].any())
        print(f"{name:28s} {f(vix_pred):4d} {f(vol_pred):4d} {f(hyb_pred):4d}")


if __name__ == "__main__":
    main()
