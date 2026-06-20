from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.multi_asset_data import ASSETS
from src.paper_protocol import ROOT, detector_outputs, event_mask, event_metrics, load_config, point_metrics


METHODS: tuple[tuple[str, str], ...] = (
    ("volatility_percentile", "Volatility percentile"),
    ("cusum", "CUSUM"),
    ("instability_score", "Instability score"),
    ("mmd_only_confirmation", "MMD-only confirmation"),
    ("kappa_only_confirmation", "Kappa-only confirmation"),
    ("hybrid_windowed_confirmation", "Hybrid volatility + instability confirmation"),
)


def _asset_name_lookup() -> dict[str, str]:
    return {asset.ticker: asset.asset_name for asset in ASSETS}


def _load_processed_asset(ticker: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"{ticker}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed asset file: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_asset_events(ticker: str) -> pd.DataFrame:
    path = ROOT / "data" / "events_multi_asset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-asset event table: {path}")
    events = pd.read_csv(path)
    out = events[events["ticker"] == ticker].copy()
    if out.empty:
        raise ValueError(f"No events found for {ticker} in {path}")
    return out


def _method_predictions(pred: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "volatility_percentile": pred["volatility_percentile"],
        "cusum": pred["cusum"],
        "instability_score": pred["instability_score"],
        "mmd_only_confirmation": pred["mmd_only_confirmation"],
        "kappa_only_confirmation": pred["kappa_only_confirmation"],
        "hybrid_windowed_confirmation": pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"],
    }


def evaluate_asset(ticker: str, config: dict[str, Any]) -> pd.DataFrame:
    df = _load_processed_asset(ticker)
    events = _load_asset_events(ticker)
    _, pred = detector_outputs(df, config)
    y, spans = event_mask(df, events, config["events"]["half_window"])
    names = dict(METHODS)
    rows: list[dict[str, Any]] = []
    for method, arr in _method_predictions(pred, config).items():
        point = point_metrics(y, arr)
        event = event_metrics(arr, spans)
        rows.append(
            {
                "ticker": ticker,
                "asset_name": _asset_name_lookup()[ticker],
                "method": names[method],
                "precision": point["precision"],
                "recall": point["recall"],
                "f1": point["f1"],
                "fpr": point["fpr"],
                "mean_delay": event["event_delay"],
                "alignment_error": event["event_alignment_error"],
                "n_alarms": int(np.sum(arr)),
                "n_event_positive_days": int(np.sum(y)),
                "n_non_event_days": int(len(y) - np.sum(y)),
            }
        )
    return pd.DataFrame(rows)


def _write_paper_ready_table(metrics: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    table = metrics[
        [
            "ticker",
            "method",
            "precision",
            "recall",
            "f1",
            "fpr",
            "mean_delay",
            "alignment_error",
        ]
    ].copy()
    for col in ["precision", "recall", "f1", "fpr", "mean_delay", "alignment_error"]:
        table[col] = table[col].round(3)
    table = table.rename(
        columns={
            "ticker": "Ticker",
            "method": "Method",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "fpr": "FPR",
            "mean_delay": "Mean delay",
            "alignment_error": "Alignment error",
        }
    )
    table.to_csv(outdir / "table_multi_asset_event_fixed_evaluation.csv", index=False)
    return table


def _write_tradeoff_figure(metrics: pd.DataFrame, outdir: Path) -> None:
    focus = metrics[
        metrics["method"].isin(
            [
                "Volatility percentile",
                "Hybrid volatility + instability confirmation",
                "CUSUM",
                "Instability score",
                "MMD-only confirmation",
            ]
        )
    ].copy()
    tickers = list(dict.fromkeys(focus["ticker"]))
    colors = {
        "Volatility percentile": "#4c78a8",
        "Hybrid volatility + instability confirmation": "#f58518",
        "CUSUM": "#54a24b",
        "Instability score": "#b279a2",
        "MMD-only confirmation": "#e45756",
    }
    markers = {
        "Volatility percentile": "o",
        "Hybrid volatility + instability confirmation": "s",
        "CUSUM": "^",
        "Instability score": "D",
        "MMD-only confirmation": "P",
    }
    fig, axes = plt.subplots(1, len(tickers), figsize=(3.2 * len(tickers), 3.4), sharey=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        grp = focus[focus["ticker"] == ticker]
        for method, row in grp.groupby("method", sort=False):
            rec = row.iloc[0]
            ax.scatter(
                rec["fpr"],
                rec["f1"],
                s=54,
                marker=markers.get(method, "o"),
                color=colors.get(method, "black"),
                edgecolor="white",
                linewidth=0.5,
                label=method,
            )
        ax.set_title(ticker)
        ax.set_xlabel("FPR")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Multi-asset event-fixed tradeoff comparison", y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.savefig(outdir / "multi_asset_tradeoff_comparison.png", dpi=220)
    plt.close(fig)


def _direction(curr: float, ref: float, tolerance: float = 1e-12) -> str:
    if curr < ref - tolerance:
        return "lower"
    if curr > ref + tolerance:
        return "higher"
    return "unchanged"


def _write_interpretation(metrics: pd.DataFrame, outdir: Path) -> None:
    vol = metrics[metrics["method"] == "Volatility percentile"].set_index("ticker")
    hyb = metrics[metrics["method"] == "Hybrid volatility + instability confirmation"].set_index("ticker")
    tickers = list(hyb.index)
    fpr_lower = [t for t in tickers if hyb.loc[t, "fpr"] < vol.loc[t, "fpr"]]
    recall_lower = [t for t in tickers if hyb.loc[t, "recall"] < vol.loc[t, "recall"]]
    f1_lower = [t for t in tickers if hyb.loc[t, "f1"] < vol.loc[t, "f1"]]
    delayed = [t for t in tickers if hyb.loc[t, "mean_delay"] > vol.loc[t, "mean_delay"]]
    breaks = [
        t
        for t in tickers
        if not (
            hyb.loc[t, "fpr"] < vol.loc[t, "fpr"]
            and hyb.loc[t, "recall"] < vol.loc[t, "recall"]
            and hyb.loc[t, "f1"] < vol.loc[t, "f1"]
        )
    ]

    lines = [
        "# Multi-Asset Interpretation",
        "",
        "This summary compares the volatility percentile trigger with the hybrid volatility + instability confirmation rule using the same event-fixed protocol and unchanged detector hyperparameters.",
        "",
        "## Does hybrid reduce FPR across assets?",
        "",
        f"Yes for {len(fpr_lower)}/{len(tickers)} assets: {', '.join(fpr_lower) if fpr_lower else 'none'}.",
        "",
        "## Does hybrid reduce recall/F1 across assets?",
        "",
        f"Recall is lower for {len(recall_lower)}/{len(tickers)} assets: {', '.join(recall_lower) if recall_lower else 'none'}.",
        f"F1 is lower for {len(f1_lower)}/{len(tickers)} assets: {', '.join(f1_lower) if f1_lower else 'none'}.",
        "",
        "## Does hybrid delay alarms across assets?",
        "",
        f"Mean first-detection delay is later for {len(delayed)}/{len(tickers)} assets: {', '.join(delayed) if delayed else 'none'}.",
        "",
        "## Are there assets where the pattern breaks?",
        "",
    ]
    if breaks:
        lines.append(f"Yes: {', '.join(breaks)} do not show the full SPY pattern of lower FPR, lower recall, and lower F1 simultaneously.")
    else:
        lines.append("No. Every processed asset shows the same broad SPY pattern: the hybrid has lower FPR, lower recall, and lower F1 than the volatility trigger.")
    lines.extend(
        [
            "",
            "## Are HYG/TLT behavior different from equity ETFs?",
            "",
        ]
    )
    for ticker in ["HYG", "TLT"]:
        if ticker not in tickers:
            continue
        lines.append(
            f"- {ticker}: hybrid FPR is {_direction(hyb.loc[ticker, 'fpr'], vol.loc[ticker, 'fpr'])}, "
            f"recall is {_direction(hyb.loc[ticker, 'recall'], vol.loc[ticker, 'recall'])}, "
            f"F1 is {_direction(hyb.loc[ticker, 'f1'], vol.loc[ticker, 'f1'])}, "
            f"and mean delay is {_direction(hyb.loc[ticker, 'mean_delay'], vol.loc[ticker, 'mean_delay'])} than volatility."
        )
    lines.extend(["", "## Volatility vs Hybrid Details", ""])
    for ticker in tickers:
        lines.append(
            f"- {ticker}: volatility FPR={vol.loc[ticker, 'fpr']:.3f}, recall={vol.loc[ticker, 'recall']:.3f}, F1={vol.loc[ticker, 'f1']:.3f}, delay={vol.loc[ticker, 'mean_delay']:.1f}; "
            f"hybrid FPR={hyb.loc[ticker, 'fpr']:.3f}, recall={hyb.loc[ticker, 'recall']:.3f}, F1={hyb.loc[ticker, 'f1']:.3f}, delay={hyb.loc[ticker, 'mean_delay']:.1f}."
        )
    (outdir / "multi_asset_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multi_asset_evaluation(config_path: Path | str = ROOT / "config" / "paper_experiment.json") -> pd.DataFrame:
    config = load_config(config_path)
    availability_path = ROOT / "results" / "multi_asset" / "data_availability.csv"
    if not availability_path.exists():
        raise FileNotFoundError("Run scripts/prepare_multi_asset_data.py before multi-asset evaluation.")
    availability = pd.read_csv(availability_path)
    tickers = availability.loc[availability["number_of_trading_days"] > 0, "ticker"].tolist()
    outdir = ROOT / "results" / "multi_asset"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([evaluate_asset(ticker, config) for ticker in tickers], ignore_index=True)
    metrics.to_csv(outdir / "multi_asset_pointwise_metrics.csv", index=False)
    _write_paper_ready_table(metrics, outdir)
    _write_tradeoff_figure(metrics, outdir)
    _write_interpretation(metrics, outdir)
    return metrics
