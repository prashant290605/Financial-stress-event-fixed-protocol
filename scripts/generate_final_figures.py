from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paper_protocol import ROOT, detector_outputs, load_config


OUT = ROOT / "figures" / "final"
COLORS = {
    "Volatility": "#1f77b4",
    "Hybrid": "#d95f02",
    "CUSUM": "#2ca02c",
    "MMD": "#7570b3",
    "Instability": "#6a3d9a",
    "Kappa": "#8c564b",
    "Event": "#7f7f7f",
}


def setup_style() -> None:
    # Springer figure requirements: sans-serif lettering (Helvetica or Arial)
    # at 8-12 pt, consistent sizes, and no titles inside the illustrations
    # (captions belong in the manuscript text).
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def method_label(value: str) -> str:
    mapping = {
        "Volatility percentile": "Volatility",
        "Hybrid W=10": "Hybrid",
        "Hybrid volatility + instability confirmation": "Hybrid",
        "CUSUM": "CUSUM",
        "Instability score": "Instability",
        "MMD-only confirmation": "MMD",
        "Kappa-only confirmation": "Kappa",
        "volatility_percentile": "Volatility",
        "hybrid_windowed_confirmation": "Hybrid",
        "cusum": "CUSUM",
        "instability_score": "Instability",
    }
    return mapping.get(value, value)


def fig_timeline(config: dict) -> None:
    df = pd.read_csv(ROOT / "results" / "paper" / "spy_features_and_scores.csv", parse_dates=["date"])
    events = pd.read_csv(ROOT / "data" / "events_spy.csv", parse_dates=["mapped_trading_date"])
    _, pred = detector_outputs(df, config)
    hybrid = pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"].astype(bool)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(df["date"], df["signal"], color="#222222", lw=0.7, label="Signal")
    ax.scatter(df.loc[hybrid, "date"], df.loc[hybrid, "signal"], s=11, color=COLORS["Hybrid"], label="Hybrid alarms", zorder=4)
    for i, d in enumerate(events["mapped_trading_date"]):
        ax.axvspan(d - pd.tseries.offsets.BDay(60), d + pd.tseries.offsets.BDay(60), color=COLORS["Event"], alpha=0.08, lw=0, label="Event window" if i == 0 else None)
        ax.axvline(d, color=COLORS["Event"], alpha=0.35, lw=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Composite signal")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
    )
    ax.grid(axis="y", alpha=0.18)
    save(fig, "timeline_spy")


def fig_event_window() -> None:
    data = pd.read_csv(ROOT / "results" / "paper" / "table_event_window_sensitivity.csv")
    data["method"] = data["method"].map(method_label)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
    for method, grp in data.groupby("method"):
        color = COLORS[method]
        axes[0].plot(grp["half_window"], grp["f1"], marker="o", color=color, label=method)
        axes[1].plot(grp["half_window"], grp["fpr"], marker="o", color=color, label=method)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("FPR")
    for ax in axes:
        ax.set_xlabel("Event half-window (trading days)")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "event_window_sensitivity")


def fig_confirmation_window() -> None:
    data = pd.read_csv(ROOT / "results" / "paper" / "table_window_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(data["window"], data["recall"], marker="o", color="#4c78a8", label="Recall")
    ax.plot(data["window"], data["f1"], marker="s", color=COLORS["Hybrid"], label="F1")
    ax.plot(data["window"], data["fpr"], marker="^", color="#333333", label="FPR")
    ax.set_xlabel("Confirmation window W (trading days)")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save(fig, "confirmation_window_sensitivity")


def fig_bootstrap() -> None:
    data = pd.read_csv(ROOT / "results" / "paper" / "bootstrap_samples.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True)
    for ax, metric, title in zip(axes, ["delta_fpr", "delta_f1"], [r"$\Delta$FPR", r"$\Delta$F1"]):
        vals = [data.loc[data["bootstrap"] == kind, metric].to_numpy() for kind in ["event", "block"]]
        parts = ax.violinplot(vals, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["Hybrid"])
            body.set_edgecolor(COLORS["Hybrid"])
            body.set_alpha(0.45)
        ax.axhline(0, color="#222222", lw=0.8, ls="--")
        ax.set_xticks([1, 2], ["Event", "Block"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Hybrid - Volatility")
    fig.tight_layout()
    save(fig, "bootstrap_delta")


def fig_synthetic() -> None:
    syn = pd.read_csv(ROOT / "results" / "paper" / "synthetic_series.csv")
    ex = syn[syn["series_id"] == 0].copy()
    s = ex["signal"].to_numpy(float)
    vol = pd.Series(s).rolling(20, min_periods=2).std().fillna(0.0).to_numpy(float)
    pred = vol > np.quantile(vol[: int(0.7 * len(vol))], 0.9)
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.plot(ex["t"], ex["signal"], color="#222222", lw=0.8)
    ymin, ymax = float(ex["signal"].min()), float(ex["signal"].max())
    for label, color in [("drift", "#fee08b"), ("collapse", "#fdae61")]:
        mask = ex["label"].eq(label).to_numpy()
        ax.fill_between(ex["t"], ymin, ymax, where=mask, color=color, alpha=0.22, label=label.capitalize())
    ax.scatter(ex.loc[pred, "t"], ex.loc[pred, "signal"], s=9, color=COLORS["Volatility"], label="Volatility alarms", zorder=4)
    ax.set_xlabel("Synthetic time step")
    ax.set_ylabel("Signal")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    save(fig, "synthetic_example")


def fig_nab() -> None:
    data = pd.read_csv(ROOT / "results" / "paper" / "table_nab_subset_results.csv")
    focus = data[data["method"].isin(["volatility_percentile", "hybrid_windowed_confirmation"])].copy()
    focus["method"] = focus["method"].map(method_label)
    subsets = sorted(focus["subset"].unique())
    x = np.arange(len(subsets))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True)
    for i, metric in enumerate(["f1", "fpr"]):
        for offset, method in [(-width / 2, "Volatility"), (width / 2, "Hybrid")]:
            vals = focus[focus["method"] == method].set_index("subset").loc[subsets, metric]
            axes[i].bar(x + offset, vals, width=width, color=COLORS[method], label=method)
        axes[i].set_ylabel(metric.upper() if metric == "fpr" else "F1")
        axes[i].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x, subsets, rotation=30, ha="right")
    fig.tight_layout(h_pad=1.0)
    save(fig, "nab_subset_results")


def fig_multi_asset() -> None:
    data = pd.read_csv(ROOT / "results" / "multi_asset" / "multi_asset_pointwise_metrics.csv")
    focus = data[data["method"].isin(["Volatility percentile", "Hybrid volatility + instability confirmation", "CUSUM", "MMD-only confirmation"])].copy()
    focus["method"] = focus["method"].map(method_label)
    tickers = list(focus["ticker"].drop_duplicates())
    markers = {"Volatility": "o", "Hybrid": "s", "CUSUM": "^", "MMD": "D"}
    fig, axes = plt.subplots(1, len(tickers), figsize=(8.8, 2.8), sharey=True)
    for ax, ticker in zip(axes, tickers):
        grp = focus[focus["ticker"] == ticker]
        for _, row in grp.iterrows():
            method = row["method"]
            ax.scatter(row["fpr"], row["f1"], s=48, marker=markers[method], color=COLORS[method], edgecolor="white", linewidth=0.5, label=method)
        ax.set_title(ticker)
        ax.set_xlabel("FPR")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    save(fig, "multi_asset_tradeoff")


def fig_event_level() -> None:
    data = pd.read_csv(ROOT / "results" / "event_level" / "event_level_metrics.csv")
    focus = data[(data["ticker"] == "SPY") & (data["method"].isin(["Volatility percentile", "Hybrid volatility + instability confirmation"]))].copy()
    focus["method"] = focus["method"].map(method_label)
    events = focus[["event_id", "event_name"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(events))
    offsets = {"Volatility": -0.17, "Hybrid": 0.17}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for method in ["Volatility", "Hybrid"]:
        rows = focus[focus["method"] == method].set_index("event_id")
        for i, event_id in enumerate(events["event_id"]):
            row = rows.loc[event_id]
            xpos = x[i] + offsets[method]
            if bool(row["detected_event"]):
                ax.scatter(xpos, row["first_detection_delay"], s=35 + 550 * row["event_coverage"], color=COLORS[method], marker="o" if method == "Volatility" else "s", edgecolor="white", linewidth=0.5, label=method if i == 0 else None)
            else:
                ax.scatter(xpos, -68, s=48, color=COLORS[method], marker="x", label=method if i == 0 else None)
    ax.axhline(0, color="#222222", lw=0.8, ls="--")
    ax.axhline(-60, color="#999999", lw=0.6, ls=":")
    ax.axhline(60, color="#999999", lw=0.6, ls=":")
    ax.set_ylim(-74, 66)
    ax.set_xticks(x, events["event_name"], rotation=35, ha="right")
    ax.set_ylabel("First detection delay (trading days)")
    ax.text(-0.35, -68, "x = missed", ha="left", va="center", fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save(fig, "event_level_delay_coverage")


def fig_cost() -> None:
    point = pd.read_csv(ROOT / "results" / "cost" / "pointwise_cost_by_asset.csv")
    event = pd.read_csv(ROOT / "results" / "cost" / "event_aware_cost_by_asset.csv")
    methods = ["Volatility percentile", "Hybrid volatility + instability confirmation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    spy = point[point["ticker"] == "SPY"].copy()
    for method in methods:
        grp = spy[spy["method"] == method]
        label = method_label(method)
        axes[0].plot(grp["ratio_fp_to_fn"], grp["cost"], marker="o", color=COLORS[label], label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("FP/FN cost ratio")
    axes[0].set_ylabel("Pointwise cost")
    axes[0].set_title("Pointwise")
    spy_e = event[(event["ticker"] == "SPY") & (event["delay_cost"] == 0.05)].copy()
    for method in methods:
        grp = spy_e[spy_e["method"] == method]
        label = method_label(method)
        axes[1].plot(grp["ratio_false_alarm_to_missed_event"], grp["cost"], marker="o", color=COLORS[label], label=label)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("False-alarm/missed-event cost ratio")
    axes[1].set_ylabel("Event-aware cost")
    axes[1].set_title("Event-aware")
    for ax in axes:
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "cost_comparison")


def main() -> None:
    setup_style()
    config = load_config(ROOT / "config" / "paper_experiment.json")
    fig_timeline(config)
    fig_event_window()
    fig_confirmation_window()
    fig_bootstrap()
    fig_synthetic()
    fig_nab()
    fig_multi_asset()
    fig_event_level()
    fig_cost()
    print(f"Wrote final figures to {OUT}")


if __name__ == "__main__":
    main()
