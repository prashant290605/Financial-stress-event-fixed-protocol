"""Task 14: paper figures that depend on the event windows, redrawn from the revision outputs.

Same layout and style as scripts/generate_final_figures.py, with every input taken from
this run's outputs (active panel, observable events) instead of the submitted files:
Fig. 1 timeline, Fig. 4 event-level delays, Fig. 5 costs, Fig. 6 event-window sensitivity,
Fig. 7 confirmation-window sensitivity, and a candidate replacement for Fig. 8 drawn from
the redesigned block bootstrap. Fig. 2 is drawn by t1 and the operating-curve figure by t4.
Figs. 3, 9 and 10 use synthetic or NAB data and do not depend on the event panel.

Reads t1, t6 and t13 outputs (run them first).

Outputs: <out root>/t14_figures/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import CONF_NAME, CONFIG, COLORS, OUT_ROOT, ROOT, VOL_NAME, figure_style, outdir, save_figure, spy_events  # noqa: E402
from src.multi_asset_data import _read_price_csv, preprocess_price_frame  # noqa: E402
from src.paper_protocol import detector_outputs  # noqa: E402

OUT = outdir("t14_figures")
EVENT_GREY = "#7f7f7f"
LABEL = {VOL_NAME: "Volatility", CONF_NAME: "Confirmation", "Volatility percentile": "Volatility", "Hybrid W=10": "Confirmation"}


def timeline() -> None:
    df = preprocess_price_frame(_read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]), CONFIG)
    _, pred = detector_outputs(df, CONFIG)
    hybrid = pred[f"hybrid_w{CONFIG['detectors']['default_hybrid_window']}"].astype(bool)
    events = spy_events()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(df["date"], df["signal"], color="#222222", lw=0.7, label="Signal")
    ax.scatter(df.loc[hybrid, "date"], df.loc[hybrid, "signal"], s=11, color=COLORS["Confirmation"], label="Confirmation alarms", zorder=4)
    for i, d in enumerate(pd.to_datetime(events["mapped_trading_date"])):
        ax.axvspan(d - pd.tseries.offsets.BDay(60), d + pd.tseries.offsets.BDay(60), color=EVENT_GREY, alpha=0.08, lw=0,
                   label="Event window" if i == 0 else None)
        ax.axvline(d, color=EVENT_GREY, alpha=0.35, lw=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Composite signal")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, OUT / "fig1_timeline_spy")
    plt.close(fig)


def event_window() -> None:
    data = pd.read_csv(OUT_ROOT / "t13_paper_tables" / "table13_event_window_sensitivity.csv")
    data["method"] = data["method"].map(LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
    for method, grp in data.groupby("method"):
        axes[0].plot(grp["half_window"], grp["f1"], marker="o", color=COLORS[method], label=method)
        axes[1].plot(grp["half_window"], grp["fpr"], marker="o", color=COLORS[method], label=method)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("FPR")
    for ax in axes:
        ax.set_xlabel("Event half-window (trading days)")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT / "fig6_event_window_sensitivity")
    plt.close(fig)


def confirmation_window() -> None:
    data = pd.read_csv(OUT_ROOT / "t13_paper_tables" / "confirmation_window_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(data["window"], data["recall"], marker="o", color="#4c78a8", label="Recall")
    ax.plot(data["window"], data["f1"], marker="s", color=COLORS["Confirmation"], label="F1")
    ax.plot(data["window"], data["fpr"], marker="^", color="#333333", label="FPR")
    ax.set_xlabel("Confirmation window W (trading days)")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save_figure(fig, OUT / "fig7_confirmation_window_sensitivity")
    plt.close(fig)


def event_level() -> None:
    data = pd.read_csv(OUT_ROOT / "t1_denominators" / "event_level_metrics_observable.csv")
    focus = data[(data["ticker"] == "SPY") & data["method"].isin([VOL_NAME, CONF_NAME])].copy()
    focus["method"] = focus["method"].map(LABEL)
    events = focus[["event_id", "event_name"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(events))
    offsets = {"Volatility": -0.17, "Confirmation": 0.17}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for method in ["Volatility", "Confirmation"]:
        rows = focus[focus["method"] == method].set_index("event_id")
        for i, event_id in enumerate(events["event_id"]):
            row = rows.loc[event_id]
            if bool(row["detected_event"]):
                ax.scatter(x[i] + offsets[method], row["first_detection_delay"], s=35 + 550 * row["event_coverage"], color=COLORS[method],
                           marker="o" if method == "Volatility" else "s", edgecolor="white", linewidth=0.5, label=method if i == 0 else None)
            else:
                ax.scatter(x[i] + offsets[method], -68, s=48, color=COLORS[method], marker="x", label=method if i == 0 else None)
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
    save_figure(fig, OUT / "fig4_event_level_delay_coverage")
    plt.close(fig)


def cost() -> None:
    point = pd.read_csv(OUT_ROOT / "t1_denominators" / "pointwise_cost_by_asset_observable.csv")
    event = pd.read_csv(OUT_ROOT / "t1_denominators" / "event_aware_cost_by_asset_observable.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for method in [VOL_NAME, CONF_NAME]:
        grp = point[(point["ticker"] == "SPY") & (point["method"] == method)]
        axes[0].plot(grp["ratio_fp_to_fn"], grp["cost"], marker="o", color=COLORS[LABEL[method]], label=LABEL[method])
        grp = event[(event["ticker"] == "SPY") & (event["delay_cost"] == 0.05) & (event["method"] == method)]
        axes[1].plot(grp["ratio_false_alarm_to_missed_event"], grp["cost"], marker="o", color=COLORS[LABEL[method]], label=LABEL[method])
    axes[0].set_xlabel("FP/FN cost ratio")
    axes[0].set_ylabel("Pointwise cost")
    axes[0].set_title("Pointwise")
    axes[1].set_xlabel("False-alarm/missed-event cost ratio")
    axes[1].set_ylabel("Event-aware cost")
    axes[1].set_title("Event-aware")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT / "fig5_cost_comparison")
    plt.close(fig)


def bootstrap() -> None:
    data = pd.read_csv(OUT_ROOT / "t6_uncertainty" / "bootstrap_samples_spy.csv")
    data = data[data["comparison"].str.startswith("volatility default")]
    panels = [("delta_precision_block121", r"$\Delta$precision"), ("delta_f1_block121", r"$\Delta$F1"), ("delta_fpr_block121", r"$\Delta$FPR")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2))
    for ax, (key, title) in zip(axes, panels):
        vals = data.loc[data["estimand"] == key, "value"].to_numpy()
        parts = ax.violinplot([vals], showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["Confirmation"])
            body.set_edgecolor(COLORS["Confirmation"])
            body.set_alpha(0.45)
        ax.axhline(0, color="#222222", lw=0.8, ls="--")
        ax.set_xticks([1], ["121-day blocks"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Confirmation - Volatility")
    fig.tight_layout()
    save_figure(fig, OUT / "fig8_candidate_block_bootstrap")
    plt.close(fig)


def main() -> None:
    figure_style()
    timeline()
    event_window()
    confirmation_window()
    event_level()
    cost()
    bootstrap()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
