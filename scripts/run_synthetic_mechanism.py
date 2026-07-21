"""Controlled synthetic test of the scale-adaptation mechanism.

Builds one series containing two engineered stress episodes and runs the
paper's exact detector pipeline on it:

  Episode A (abrupt):   daily volatility jumps from sigma0 to 3*sigma0
                        overnight, faster than the trailing 50-day
                        standardization scale can adapt.
  Episode B (absorbed): volatility ramps linearly up to the same 3*sigma0
                        level and back down again (a symmetric triangle,
                        with no jump at either edge), slowly enough that
                        the trailing scale tracks it, so the composite
                        signal's distribution barely changes.

The mechanism claim in the manuscript predicts that the volatility trigger
fires in both episodes while the instability confirmation gate opens only
in Episode A. A third abrupt episode inside the calibration segment gives
the 90th-percentile thresholds the same crisis-contaminated training that
the SPY calibration period has.

Outputs:
  results/ablations/table_synthetic_mechanism.csv
  figures/final/synthetic_mechanism.{png,pdf,svg}

Run from the repository root:
    python scripts/run_synthetic_mechanism.py
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_protocol import (
    _rolling_mean,
    _z_from_train,
    compute_instability_features,
    load_config,
    windowed_hybrid,
)

CONFIG = load_config(PROJECT_ROOT / "config" / "paper_experiment.json")
OUTDIR = PROJECT_ROOT / "results" / "ablations"
FIGDIR = PROJECT_ROOT / "figures" / "final"

N = 2000
SIGMA0 = 0.010
LEVEL = 3.0
CALIB_EPISODE = (500, 560)        # abrupt, inside the 70% calibration segment
EPISODE_A = (1500, 1560)          # abrupt, evaluation segment
EPISODE_B = (1650, 1950)          # symmetric triangle ramp, peak at the midpoint


def volatility_path() -> np.ndarray:
    sigma = np.full(N, SIGMA0)
    for a, b in (CALIB_EPISODE, EPISODE_A):
        sigma[a:b] = LEVEL * SIGMA0
    a, b = EPISODE_B
    mid = (a + b) // 2
    sigma[a:mid] = np.linspace(SIGMA0, LEVEL * SIGMA0, mid - a)
    sigma[mid:b] = np.linspace(LEVEL * SIGMA0, SIGMA0, b - mid)
    return sigma


def main() -> None:
    rng = np.random.default_rng(CONFIG["seed"])
    sigma = volatility_path()
    returns = sigma * rng.standard_normal(N)

    # The paper's exact preprocessing on the synthetic return series.
    df = pd.DataFrame({"log_return": returns})
    spy_cfg = CONFIG["spy"]
    df["rolling_volatility"] = df["log_return"].rolling(spy_cfg["rolling_vol_window"]).std()
    mean = df["log_return"].rolling(spy_cfg["zscore_window"]).mean()
    std = df["log_return"].rolling(spy_cfg["zscore_window"]).std()
    df["return_zscore"] = (df["log_return"] - mean) / std
    w = spy_cfg["signal_weights"]
    df["signal"] = (
        w["log_return"] * df["log_return"]
        + w["rolling_volatility"] * df["rolling_volatility"]
        + w["return_zscore"] * df["return_zscore"]
    )
    offset = int(df["signal"].isna().sum())
    df = df.dropna().reset_index(drop=True)

    signal = df["signal"].to_numpy(float)
    feat = compute_instability_features(signal, CONFIG)
    train_end = int(round(len(df) * CONFIG["detectors"]["calibration_fraction"]))
    smooth = CONFIG["features"]["smooth_window"]
    quantile = CONFIG["detectors"]["score_threshold_quantile"]

    mmd_z = _z_from_train(feat["MMD"].to_numpy(float), train_end)
    kappa_z = _z_from_train(feat["kappa"].to_numpy(float), train_end)
    score = _rolling_mean(0.5 * mmd_z + 0.5 * np.abs(kappa_z), smooth)
    instability = (score > np.quantile(score[:train_end], quantile)).astype(int)
    # MMD-only confirmation isolates the kernel gate from the kappa noise term,
    # mirroring the mmd_only_confirmation variant of the main pipeline.
    mmd_only_s = _rolling_mean(mmd_z, smooth)
    mmd_only = (mmd_only_s > np.quantile(mmd_only_s[:train_end], quantile)).astype(int)

    vol = _rolling_mean(_z_from_train(df["rolling_volatility"].to_numpy(float), train_end), smooth)
    volatility = (vol > np.quantile(vol[:train_end], CONFIG["detectors"]["volatility_threshold_quantile"])).astype(int)
    hybrid = windowed_hybrid(volatility, instability, CONFIG["detectors"]["default_hybrid_window"])

    mmd_s = _rolling_mean(mmd_z, smooth)

    def window(a: int, b: int) -> slice:
        return slice(a - offset, b - offset)

    episodes = {
        "A abrupt break": window(*EPISODE_A),
        "B absorbed ramp": window(*EPISODE_B),
    }
    # Calm evaluation background: post-calibration days at baseline volatility,
    # excluding both episodes plus a 30-day buffer around each (the MMD
    # post-window is 30 days, so scores within the buffer can still see an
    # episode).
    calm = np.ones(len(df), bool)
    calm[:train_end] = False
    for a, b in (EPISODE_A, EPISODE_B):
        calm[max(0, a - offset - 30):min(len(df), b - offset + 30)] = False

    rows = []
    for name, sl in episodes.items():
        days = sl.stop - sl.start
        rows.append({
            "episode": name,
            "days": days,
            "volatility_alarm_days": int(volatility[sl].sum()),
            "instability_alarm_days": int(instability[sl].sum()),
            "instability_rate": float(instability[sl].mean()),
            "mmd_only_alarm_days": int(mmd_only[sl].sum()),
            "mmd_only_rate": float(mmd_only[sl].mean()),
            "hybrid_alarm_days": int(hybrid[sl].sum()),
            "peak_smoothed_z_mmd": float(np.nanmax(mmd_s[sl])),
        })
    rows.append({
        "episode": "calm background (eval)",
        "days": int(calm.sum()),
        "volatility_alarm_days": int(volatility[calm].sum()),
        "instability_alarm_days": int(instability[calm].sum()),
        "instability_rate": float(instability[calm].mean()),
        "mmd_only_alarm_days": int(mmd_only[calm].sum()),
        "mmd_only_rate": float(mmd_only[calm].mean()),
        "hybrid_alarm_days": int(hybrid[calm].sum()),
        "peak_smoothed_z_mmd": float(np.nanmax(mmd_s[calm])),
    })
    out = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "table_synthetic_mechanism.csv", index=False)
    print(out.to_string(index=False))

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    t = np.arange(len(df)) + offset
    # Plot from day 150 onward: the first weeks hold rolling-window burn-in
    # artifacts in the standardized scores that would collide with the legend
    # and carry no information about the engineered episodes.
    show = t >= 150
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True)
    axes[0].plot(t[show], df["log_return"].to_numpy()[show], color="#222222", lw=0.5)
    axes[0].set_ylabel("Synthetic return")
    axes[1].plot(t[show], vol[show], color="#1f77b4", lw=0.9, label="Volatility score")
    axes[1].plot(t[show], score[show], color="#d95f02", lw=0.9, label="Instability score")
    top = float(max(np.nanmax(vol[show]), np.nanmax(score[show])))
    axes[1].set_ylim(None, top + 1.6)
    hy = hybrid.astype(bool) & show
    axes[1].scatter(t[hy], np.full(hy.sum(), axes[1].get_ylim()[0]), s=8, marker="|", color="#d62728", label="Hybrid alarms")
    axes[1].set_ylabel("Standardized score")
    axes[1].set_xlabel("Synthetic day")
    for ax in axes:
        for (a, b), color in [(EPISODE_A, "#d95f02"), (EPISODE_B, "#1f77b4"), (CALIB_EPISODE, "#7f7f7f")]:
            ax.axvspan(a, b, color=color, alpha=0.10, lw=0)
        ax.grid(axis="y", alpha=0.18)
    axes[0].text(np.mean(EPISODE_A), axes[0].get_ylim()[1] * 0.86, "A", ha="center", fontsize=10)
    axes[0].text(np.mean(EPISODE_B), axes[0].get_ylim()[1] * 0.86, "B", ha="center", fontsize=10)
    axes[0].text(np.mean(CALIB_EPISODE), axes[0].get_ylim()[1] * 0.86, "calib.", ha="center", fontsize=9, color="#555555")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"synthetic_mechanism.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {FIGDIR / 'synthetic_mechanism.png'}")


if __name__ == "__main__":
    main()
