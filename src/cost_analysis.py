from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.multi_asset_evaluation import _load_asset_events, _load_processed_asset, _method_predictions
from src.paper_protocol import ROOT, detector_outputs, event_mask, load_config, point_metrics


POINTWISE_RATIOS = [1, 2, 5, 10, 20, 50]
EVENT_FALSE_ALARM_RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DELAY_COSTS = [0.0, 0.05, 0.1]
FOCUS_METHODS = {
    "volatility_percentile": "Volatility percentile",
    "hybrid_windowed_confirmation": "Hybrid volatility + instability confirmation",
}


def _available_tickers() -> list[str]:
    availability = pd.read_csv(ROOT / "results" / "multi_asset" / "data_availability.csv")
    return availability.loc[availability["number_of_trading_days"] > 0, "ticker"].tolist()


def _focus_predictions(ticker: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    df = _load_processed_asset(ticker)
    events = _load_asset_events(ticker)
    _, pred = detector_outputs(df, config)
    all_preds = _method_predictions(pred, config)
    return df, events, {
        "Volatility percentile": all_preds["volatility_percentile"],
        "Hybrid volatility + instability confirmation": all_preds["hybrid_windowed_confirmation"],
    }


def _preferred(group: pd.DataFrame, ratio_col: str) -> pd.DataFrame:
    out = group.copy()
    for _, idx in out.groupby(["ticker", ratio_col] + (["delay_cost"] if "delay_cost" in out.columns else [])).groups.items():
        costs = out.loc[idx, "cost"]
        min_cost = costs.min()
        winners = out.loc[idx[costs.to_numpy() == min_cost], "method"].tolist()
        preferred = "tie" if len(winners) != 1 else winners[0]
        out.loc[idx, "preferred_method_at_ratio"] = preferred
    return out


def compute_pointwise_cost(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in _available_tickers():
        df, events, preds = _focus_predictions(ticker, config)
        y, _ = event_mask(df, events, config["events"]["half_window"])
        for method, arr in preds.items():
            metrics = point_metrics(y, arr)
            for ratio in POINTWISE_RATIOS:
                rows.append(
                    {
                        "ticker": ticker,
                        "ratio_fp_to_fn": ratio,
                        "method": method,
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                        "cost": float(ratio * metrics["fp"] + metrics["fn"]),
                    }
                )
    out = pd.DataFrame(rows)
    return _preferred(out, "ratio_fp_to_fn")[
        ["ticker", "ratio_fp_to_fn", "method", "fp", "fn", "cost", "preferred_method_at_ratio"]
    ]


def _event_components(ticker: str, method: str, y: np.ndarray, alarms: np.ndarray, event_detail: pd.DataFrame) -> dict[str, float]:
    method_events = event_detail[(event_detail["ticker"] == ticker) & (event_detail["method"] == method)]
    false_alarm_days = int(np.sum((alarms == 1) & (y == 0)))
    missed_events = int((~method_events["detected_event"].astype(bool)).sum())
    delays = pd.to_numeric(method_events.loc[method_events["detected_event"].astype(bool), "first_detection_delay"], errors="coerce")
    total_positive_delay = float(np.maximum(delays.dropna().to_numpy(float), 0.0).sum())
    return {
        "false_alarm_days": false_alarm_days,
        "missed_events": missed_events,
        "total_positive_delay": total_positive_delay,
    }


def compute_event_aware_cost(config: dict[str, Any]) -> pd.DataFrame:
    event_detail_path = ROOT / "results" / "event_level" / "event_level_metrics.csv"
    if not event_detail_path.exists():
        raise FileNotFoundError("Run scripts/evaluate_event_level.py before cost analysis.")
    event_detail = pd.read_csv(event_detail_path)
    rows: list[dict[str, Any]] = []
    for ticker in _available_tickers():
        df, events, preds = _focus_predictions(ticker, config)
        y, _ = event_mask(df, events, config["events"]["half_window"])
        for method, arr in preds.items():
            components = _event_components(ticker, method, y, arr, event_detail)
            for delay_cost in DELAY_COSTS:
                for ratio in EVENT_FALSE_ALARM_RATIOS:
                    cost = (
                        ratio * components["false_alarm_days"]
                        + components["missed_events"]
                        + delay_cost * components["total_positive_delay"]
                    )
                    rows.append(
                        {
                            "ticker": ticker,
                            "ratio_false_alarm_to_missed_event": ratio,
                            "delay_cost": delay_cost,
                            "method": method,
                            **components,
                            "cost": float(cost),
                        }
                    )
    out = pd.DataFrame(rows)
    return _preferred(out, "ratio_false_alarm_to_missed_event")[
        [
            "ticker",
            "ratio_false_alarm_to_missed_event",
            "delay_cost",
            "method",
            "false_alarm_days",
            "missed_events",
            "total_positive_delay",
            "cost",
            "preferred_method_at_ratio",
        ]
    ]


def _write_tables(pointwise: pd.DataFrame, event_aware: pd.DataFrame, outdir: Path) -> None:
    pointwise[pointwise["ticker"] == "SPY"].to_csv(outdir / "table_spy_pointwise_cost.csv", index=False)
    event_aware[(event_aware["ticker"] == "SPY") & (event_aware["delay_cost"] == 0.05)].to_csv(
        outdir / "table_spy_event_aware_cost.csv", index=False
    )
    summary = event_aware[event_aware["delay_cost"] == 0.05].copy()
    summary.to_csv(outdir / "table_multi_asset_event_aware_cost_summary.csv", index=False)


def _write_cost_figure(pointwise: pd.DataFrame, event_aware: pd.DataFrame, outdir: Path) -> None:
    methods = ["Volatility percentile", "Hybrid volatility + instability confirmation"]
    colors = {"Volatility percentile": "#4c78a8", "Hybrid volatility + instability confirmation": "#f58518"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    spy_point = pointwise[pointwise["ticker"] == "SPY"]
    for method in methods:
        grp = spy_point[spy_point["method"] == method]
        axes[0].plot(grp["ratio_fp_to_fn"], grp["cost"], marker="o", color=colors[method], label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Pointwise FP/FN cost ratio")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("SPY pointwise cost")
    axes[0].grid(alpha=0.25, linewidth=0.6)

    spy_event = event_aware[(event_aware["ticker"] == "SPY") & (event_aware["delay_cost"] == 0.05)]
    for method in methods:
        grp = spy_event[spy_event["method"] == method]
        axes[1].plot(
            grp["ratio_false_alarm_to_missed_event"],
            grp["cost"],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Event-aware false-alarm/missed-event cost ratio")
    axes[1].set_ylabel("Cost")
    axes[1].set_title("SPY event-aware cost (delay cost=0.05)")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(outdir / "cost_comparison.png", dpi=220)
    plt.close(fig)


def _write_interpretation(pointwise: pd.DataFrame, event_aware: pd.DataFrame, outdir: Path) -> None:
    spy_point = pointwise[pointwise["ticker"] == "SPY"]
    spy_event = event_aware[(event_aware["ticker"] == "SPY") & (event_aware["delay_cost"] == 0.05)]
    point_pref = spy_point.groupby("ratio_fp_to_fn")["preferred_method_at_ratio"].first()
    event_pref = spy_event.groupby("ratio_false_alarm_to_missed_event")["preferred_method_at_ratio"].first()
    event_all = event_aware[event_aware["delay_cost"] == 0.05]
    hybrid_losses = event_all[event_all["preferred_method_at_ratio"] == "Volatility percentile"]["ticker"].unique().tolist()

    lines = [
        "# Cost Interpretation",
        "",
        "The original pointwise cost is retained: `c_FP * FP + c_FN * FN` with `c_FN = 1`. This cost can favor the hybrid because non-event days dominate pointwise masks, so reducing false positives can outweigh missed event-window days.",
        "",
        "The event-aware cost adds deployment-facing terms: false-alarm days outside event windows, missed events, and positive first-detection delay. This makes missed events and late confirmation visible instead of letting the large number of non-event days dominate the conclusion.",
        "",
        "## SPY Preferences",
        "",
        "Pointwise preferred methods by FP/FN ratio:",
        *[f"- {ratio}: {pref}" for ratio, pref in point_pref.items()],
        "",
        "Event-aware preferred methods for delay cost 0.05 by false-alarm/missed-event ratio:",
        *[f"- {ratio}: {pref}" for ratio, pref in event_pref.items()],
        "",
        "## Multi-Asset Event-Aware Pattern",
        "",
    ]
    if hybrid_losses:
        lines.append(
            "Under event-aware cost with delay cost 0.05, volatility is preferred in at least one cost regime for: "
            + ", ".join(hybrid_losses)
            + "."
        )
    else:
        lines.append("Under event-aware cost with delay cost 0.05, hybrid is not beaten by volatility in the evaluated regimes.")
    lines.extend(
        [
            "",
            "## Deployment Implication",
            "",
            "The correct detector depends on operational cost. If false alarm days are expensive and missed events are tolerable, the hybrid can be attractive. If missed events or late confirmation matter, volatility can be preferable even though it is noisier. This supports the paper's tradeoff framing rather than a claim of universal hybrid superiority.",
        ]
    )
    (outdir / "cost_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cost_analysis(config_path: Path | str = ROOT / "config" / "paper_experiment.json") -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    outdir = ROOT / "results" / "cost"
    outdir.mkdir(parents=True, exist_ok=True)
    pointwise = compute_pointwise_cost(config)
    event_aware = compute_event_aware_cost(config)
    pointwise.to_csv(outdir / "pointwise_cost_by_asset.csv", index=False)
    event_aware.to_csv(outdir / "event_aware_cost_by_asset.csv", index=False)
    _write_tables(pointwise, event_aware, outdir)
    _write_cost_figure(pointwise, event_aware, outdir)
    _write_interpretation(pointwise, event_aware, outdir)
    return pointwise, event_aware
