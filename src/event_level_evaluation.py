from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.multi_asset_evaluation import _asset_name_lookup, _load_asset_events, _load_processed_asset, _method_predictions
from src.paper_protocol import ROOT, detector_outputs, load_config


FOCUS_METHODS = {
    "Volatility percentile",
    "Hybrid volatility + instability confirmation",
}


def _method_display_names() -> dict[str, str]:
    return {
        "volatility_percentile": "Volatility percentile",
        "cusum": "CUSUM",
        "instability_score": "Instability score",
        "mmd_only_confirmation": "MMD-only confirmation",
        "kappa_only_confirmation": "Kappa-only confirmation",
        "hybrid_windowed_confirmation": "Hybrid volatility + instability confirmation",
    }


def _available_tickers() -> list[str]:
    availability_path = ROOT / "results" / "multi_asset" / "data_availability.csv"
    if not availability_path.exists():
        raise FileNotFoundError("Run scripts/prepare_multi_asset_data.py before event-level evaluation.")
    availability = pd.read_csv(availability_path)
    return availability.loc[availability["number_of_trading_days"] > 0, "ticker"].tolist()


def _asset_event_rows(ticker: str, config: dict[str, Any]) -> pd.DataFrame:
    df = _load_processed_asset(ticker)
    events = _load_asset_events(ticker)
    main_severity = pd.read_csv(ROOT / "data" / "events_spy.csv").set_index("event_id")["severity_label"].to_dict()
    _, pred = detector_outputs(df, config)
    preds = _method_predictions(pred, config)
    names = _method_display_names()
    half_window = config["events"]["half_window"]
    rows: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        mapped = pd.Timestamp(event["mapped_trading_date"])
        center = int(df.index[df["date"] == mapped][0])
        lo = max(0, center - half_window)
        hi = min(len(df) - 1, center + half_window)
        window_idx = np.arange(lo, hi + 1)
        relative_days = window_idx - center
        pre_mask = relative_days < 0
        post_mask = relative_days >= 0

        for method_key, alarms in preds.items():
            window_alarms = alarms[window_idx].astype(bool)
            alarm_relative_days = relative_days[window_alarms]
            detected = bool(len(alarm_relative_days) > 0)
            first_delay = int(alarm_relative_days[0]) if detected else np.nan
            closest_delay = int(alarm_relative_days[np.argmin(np.abs(alarm_relative_days))]) if detected else np.nan
            rows.append(
                {
                    "ticker": ticker,
                    "event_id": event["event_id"],
                    "event_name": event["event_name"],
                    "severity_label": main_severity[event["event_id"]],
                    "method": names[method_key],
                    "detected_event": detected,
                    "first_detection_delay": first_delay,
                    "closest_detection_delay": closest_delay,
                    "event_coverage": float(np.mean(window_alarms)),
                    "pre_event_detection": bool(np.any(window_alarms[pre_mask])),
                    "post_event_detection": bool(np.any(window_alarms[post_mask])),
                    "n_alarm_days_in_window": int(np.sum(window_alarms)),
                }
            )
    return pd.DataFrame(rows)


def _summary_by_asset(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, method), group in detail.groupby(["ticker", "method"], sort=False):
        rows.append(
            {
                "ticker": ticker,
                "method": method,
                "event_detection_rate": float(group["detected_event"].mean()),
                "median_first_detection_delay": float(group["first_detection_delay"].median(skipna=True))
                if group["first_detection_delay"].notna().any()
                else np.nan,
                "median_closest_detection_delay": float(group["closest_detection_delay"].median(skipna=True))
                if group["closest_detection_delay"].notna().any()
                else np.nan,
                "mean_event_coverage": float(group["event_coverage"].mean()),
                "pre_event_detection_rate": float(group["pre_event_detection"].mean()),
                "post_event_detection_rate": float(group["post_event_detection"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary_by_severity(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, severity, method), group in detail.groupby(["ticker", "severity_label", "method"], sort=False):
        rows.append(
            {
                "ticker": ticker,
                "severity_label": severity,
                "method": method,
                "event_detection_rate": float(group["detected_event"].mean()),
                "median_first_detection_delay": float(group["first_detection_delay"].median(skipna=True))
                if group["first_detection_delay"].notna().any()
                else np.nan,
                "mean_event_coverage": float(group["event_coverage"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_spy_event_table(detail: pd.DataFrame, outdir: Path) -> None:
    spy = detail[(detail["ticker"] == "SPY") & (detail["method"].isin(FOCUS_METHODS))].copy()
    pivot = spy.pivot(index=["event_id", "event_name", "severity_label"], columns="method")
    out = pd.DataFrame(
        {
            "event_name": pivot.index.get_level_values("event_name"),
            "severity_label": pivot.index.get_level_values("severity_label"),
            "volatility_detected": pivot["detected_event"]["Volatility percentile"].to_numpy(bool),
            "hybrid_detected": pivot["detected_event"]["Hybrid volatility + instability confirmation"].to_numpy(bool),
            "volatility_first_delay": pivot["first_detection_delay"]["Volatility percentile"].to_numpy(float),
            "hybrid_first_delay": pivot["first_detection_delay"]["Hybrid volatility + instability confirmation"].to_numpy(float),
            "volatility_coverage": pivot["event_coverage"]["Volatility percentile"].to_numpy(float),
            "hybrid_coverage": pivot["event_coverage"]["Hybrid volatility + instability confirmation"].to_numpy(float),
        }
    )
    out.to_csv(outdir / "table_spy_event_level_detection.csv", index=False)


def _write_paper_tables(detail: pd.DataFrame, by_asset: pd.DataFrame, by_severity: pd.DataFrame, outdir: Path) -> None:
    _write_spy_event_table(detail, outdir)
    severity = by_severity[by_severity["method"].isin(FOCUS_METHODS)].copy()
    severity = (
        severity.groupby(["severity_label", "method"], sort=False)[
            ["event_detection_rate", "median_first_detection_delay", "mean_event_coverage"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"median_first_detection_delay": "median_first_delay"})
    )
    severity.to_csv(outdir / "table_event_level_severity_summary.csv", index=False)
    multi_asset = by_asset[by_asset["method"].isin(FOCUS_METHODS)][
        ["ticker", "method", "event_detection_rate", "median_first_detection_delay", "mean_event_coverage"]
    ].copy()
    multi_asset = multi_asset.rename(columns={"median_first_detection_delay": "median_first_delay"})
    multi_asset.to_csv(outdir / "table_multi_asset_event_level_summary.csv", index=False)


def _write_delay_coverage_figure(detail: pd.DataFrame, outdir: Path) -> None:
    spy = detail[(detail["ticker"] == "SPY") & (detail["method"].isin(FOCUS_METHODS))].copy()
    events = spy[["event_id", "event_name", "severity_label"]].drop_duplicates().reset_index(drop=True)
    xpos = np.arange(len(events))
    offsets = {"Volatility percentile": -0.17, "Hybrid volatility + instability confirmation": 0.17}
    colors = {"Volatility percentile": "#4c78a8", "Hybrid volatility + instability confirmation": "#f58518"}
    markers = {"Volatility percentile": "o", "Hybrid volatility + instability confirmation": "s"}

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for method in ["Volatility percentile", "Hybrid volatility + instability confirmation"]:
        rows = spy[spy["method"] == method].set_index("event_id")
        for i, event_id in enumerate(events["event_id"]):
            row = rows.loc[event_id]
            x = i + offsets[method]
            if bool(row["detected_event"]):
                size = 35 + 700 * float(row["event_coverage"])
                ax.scatter(
                    x,
                    row["first_detection_delay"],
                    s=size,
                    color=colors[method],
                    marker=markers[method],
                    edgecolor="white",
                    linewidth=0.5,
                    label=method if i == 0 else None,
                    alpha=0.9,
                )
            else:
                ax.scatter(
                    x,
                    -68,
                    s=55,
                    facecolors="none",
                    color=colors[method],
                    marker="x",
                    linewidth=1.4,
                    label=method if i == 0 else None,
                )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(-60, color="#888888", linewidth=0.6, linestyle="--")
    ax.axhline(60, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels(events["event_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("First detection delay (trading days)")
    ax.set_title("SPY event-level first detection delay and coverage")
    ax.text(len(events) - 0.2, -68, "x = missed", ha="right", va="center", fontsize=8, color="#555555")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "event_level_delay_coverage.png", dpi=220)
    plt.close(fig)


def _write_interpretation(detail: pd.DataFrame, by_asset: pd.DataFrame, by_severity: pd.DataFrame, outdir: Path) -> None:
    vol = by_asset[by_asset["method"] == "Volatility percentile"].set_index("ticker")
    hyb = by_asset[by_asset["method"] == "Hybrid volatility + instability confirmation"].set_index("ticker")
    tickers = list(hyb.index)
    fewer = [t for t in tickers if hyb.loc[t, "event_detection_rate"] < vol.loc[t, "event_detection_rate"]]
    later = [t for t in tickers if hyb.loc[t, "median_first_detection_delay"] > vol.loc[t, "median_first_detection_delay"]]
    higher_coverage = [t for t in tickers if vol.loc[t, "mean_event_coverage"] > hyb.loc[t, "mean_event_coverage"]]

    hybrid_sev = by_severity[by_severity["method"] == "Hybrid volatility + instability confirmation"]
    sev_order = ["critical", "high", "medium"]
    sev_lines = []
    for sev in sev_order:
        grp = hybrid_sev[hybrid_sev["severity_label"] == sev]
        if not grp.empty:
            sev_lines.append(f"{sev}: detection rate {grp['event_detection_rate'].mean():.3f}, coverage {grp['mean_event_coverage'].mean():.3f}")

    lines = [
        "# Event-Level Interpretation",
        "",
        "This event-level analysis asks whether each fixed stress event is detected at least once inside the existing +/-60 trading-day event window, and how early, late, or densely each method fires.",
        "",
        "## Does hybrid detect fewer events than volatility?",
        "",
        f"Yes for {len(fewer)}/{len(tickers)} assets: {', '.join(fewer) if fewer else 'none'}.",
        "",
        "## Does hybrid detect severe events more reliably than medium events?",
        "",
        "Hybrid severity-stratified averages across processed assets:",
        *[f"- {line}" for line in sev_lines],
        "",
        "## Does hybrid tend to detect after the event center?",
        "",
        f"Hybrid median first-detection delay is later than volatility for {len(later)}/{len(tickers)} assets: {', '.join(later) if later else 'none'}. Positive delay means the first in-window alarm occurs after the mapped event date.",
        "",
        "## Does volatility provide earlier but noisier coverage?",
        "",
        f"Volatility has higher mean event-window coverage than hybrid for {len(higher_coverage)}/{len(tickers)} assets: {', '.join(higher_coverage) if higher_coverage else 'none'}. This is consistent with broader, noisier event-window firing.",
        "",
        "## What does this mean for deployment?",
        "",
        "The hybrid is better treated as a confirmed-stress filter than a broad early-warning system. It reduces alarm burden and event-window coverage, but that conservatism can miss entire events or move the first usable signal later. A deployment that needs high recall or early monitoring should not use the hybrid alone; a two-stage workflow is more defensible, with volatility-like triggers for surveillance and hybrid confirmation for escalation.",
        "",
        "## Asset-Level Volatility vs Hybrid Summary",
        "",
    ]
    for ticker in tickers:
        lines.append(
            f"- {ticker}: volatility detection={vol.loc[ticker, 'event_detection_rate']:.3f}, median delay={vol.loc[ticker, 'median_first_detection_delay']:.1f}, coverage={vol.loc[ticker, 'mean_event_coverage']:.3f}; "
            f"hybrid detection={hyb.loc[ticker, 'event_detection_rate']:.3f}, median delay={hyb.loc[ticker, 'median_first_detection_delay']:.1f}, coverage={hyb.loc[ticker, 'mean_event_coverage']:.3f}."
        )
    (outdir / "event_level_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_event_level_evaluation(config_path: Path | str = ROOT / "config" / "paper_experiment.json") -> pd.DataFrame:
    config = load_config(config_path)
    outdir = ROOT / "results" / "event_level"
    outdir.mkdir(parents=True, exist_ok=True)
    detail = pd.concat([_asset_event_rows(ticker, config) for ticker in _available_tickers()], ignore_index=True)
    by_asset = _summary_by_asset(detail)
    by_severity = _summary_by_severity(detail)

    detail.to_csv(outdir / "event_level_metrics.csv", index=False)
    by_asset.to_csv(outdir / "event_level_summary_by_asset.csv", index=False)
    by_severity.to_csv(outdir / "event_level_summary_by_severity.csv", index=False)
    _write_paper_tables(detail, by_asset, by_severity, outdir)
    _write_delay_coverage_figure(detail, outdir)
    _write_interpretation(detail, by_asset, by_severity, outdir)
    return detail
