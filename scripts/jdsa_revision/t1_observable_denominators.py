"""Task 1: score each asset only on the events it can observe.

Reviewer 1 minor 4 and Reviewer 2 major 4(c). An event dated before an asset's first
processed day is mapped to that day by the submitted pipeline, which creates a
synthetic window at the start of the series and scores the event as missed. This
script recomputes every affected table with those events removed, using the
production functions with ``observable_only=True`` and the active event panel (E03
corrected by default), and lists every number that changes against the submitted
result files.

Outputs: results/jdsa_revision/t1_denominators/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import (  # noqa: E402
    CONF_NAME, CONFIG, COLORS, HALF_WINDOW, PANEL, ROOT, TICKERS, VOL_NAME, events_by_ticker, figure_style, outdir,
    panel_events, save_figure, severity_labels,
)
from src.cost_analysis import compute_event_aware_cost, compute_pointwise_cost  # noqa: E402
from src.event_level_evaluation import FOCUS_METHODS, _asset_event_rows, _summary_by_asset, _summary_by_severity  # noqa: E402
from src.multi_asset_evaluation import _load_processed_asset, evaluate_asset  # noqa: E402
from src.observability import observable_event_flags  # noqa: E402
from src.paper_protocol import detector_outputs, event_mask, nearest_trading_date, point_metrics  # noqa: E402
from src.revision_analysis import event_spans  # noqa: E402

OUT = outdir("t1_denominators")
FOCUS = [VOL_NAME, CONF_NAME]


def observability_table() -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        df, events = _load_processed_asset(ticker), panel_events(ticker)
        flags = observable_event_flags(df, events)
        spans = event_spans(df, events, HALF_WINDOW).set_index("event_id")
        for _, ev in events.iterrows():
            obs = bool(flags.loc[ev.name])
            s = spans.loc[ev["event_id"]]
            rows.append(
                {
                    "ticker": ticker,
                    "event_id": ev["event_id"],
                    "event_name": ev["event_name"],
                    "calendar_date": ev["calendar_date"],
                    "mapped_trading_date": ev["mapped_trading_date"],
                    "sample_start": df["date"].iloc[0].date().isoformat(),
                    "observable": obs,
                    "window_start": df["date"].iloc[s["lo"]].date().isoformat() if obs else "",
                    "window_end": df["date"].iloc[s["hi"]].date().isoformat() if obs else "",
                    "days_before_reference": int(s["center"] - s["lo"]) if obs else np.nan,
                    "days_after_reference": int(s["hi"] - s["center"]) if obs else np.nan,
                    "window_days": int(s["hi"] - s["lo"] + 1) if obs else np.nan,
                    "window_clamped": bool(obs and (s["hi"] - s["lo"] + 1) < 2 * HALF_WINDOW + 1),
                }
            )
    return pd.DataFrame(rows)


def event_level_tables(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_asset = _summary_by_asset(detail)
    by_severity = _summary_by_severity(detail)
    counts = (
        detail.groupby(["ticker", "method"], sort=False)
        .agg(n_observable_events=("event_id", "size"), events_detected=("detected_event", "sum"))
        .reset_index()
    )
    table8 = by_asset[by_asset["method"].isin(FOCUS)].merge(counts, on=["ticker", "method"])
    table8 = table8[
        ["ticker", "method", "n_observable_events", "events_detected", "event_detection_rate",
         "median_first_detection_delay", "mean_event_coverage"]
    ].rename(columns={"median_first_detection_delay": "median_first_delay"})

    sev = by_severity[by_severity["method"].isin(FOCUS)]
    table9 = (
        sev.groupby(["severity_label", "method"], sort=False)[
            ["event_detection_rate", "median_first_detection_delay", "mean_event_coverage"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"median_first_detection_delay": "mean_of_per_asset_median_first_delay"})
    )
    focus_detail = detail[detail["method"].isin(FOCUS)]
    pooled = (
        focus_detail.groupby(["severity_label", "method"], sort=False)
        .agg(
            pooled_event_asset_pairs=("event_id", "size"),
            pooled_detected=("detected_event", "sum"),
            pooled_median_first_delay=("first_detection_delay", "median"),
        )
        .reset_index()
    )
    pooled["pooled_detection_rate"] = pooled["pooled_detected"] / pooled["pooled_event_asset_pairs"]
    table9 = table9.merge(pooled, on=["severity_label", "method"])
    return {"by_asset": by_asset, "by_severity": by_severity, "table8": table8, "table9": table9}


def tradeoff_figure(metrics: pd.DataFrame) -> None:
    figure_style()
    label = {VOL_NAME: "Volatility", CONF_NAME: "Confirmation", "CUSUM": "CUSUM", "MMD-only confirmation": "MMD"}
    focus = metrics[metrics["method"].isin(label)].copy()
    focus["method"] = focus["method"].map(label)
    markers = {"Volatility": "o", "Confirmation": "s", "CUSUM": "^", "MMD": "D"}
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(8.8, 2.8), sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        for _, row in focus[focus["ticker"] == ticker].iterrows():
            ax.scatter(row["fpr"], row["f1"], s=48, marker=markers[row["method"]], color=COLORS[row["method"]],
                       edgecolor="white", linewidth=0.5, label=row["method"])
        ax.set_title(ticker)
        ax.set_xlabel("FPR")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("F1")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    save_figure(fig, OUT / "multi_asset_tradeoff_observable")
    plt.close(fig)


def excluded_true_window_sensitivity() -> pd.DataFrame:
    """Alternative day-level mask: drop asset days inside an unobservable event's true window.

    The true window is measured on the SPY trading calendar, because the asset has no
    trading day at the reference date. This is a sensitivity check, not the main mask.
    """
    spy = _load_processed_asset("SPY")
    rows = []
    for ticker in TICKERS:
        df, events_all = _load_processed_asset(ticker), panel_events(ticker)
        flags = observable_event_flags(df, events_all)
        obs, unobs = events_all[flags], events_all[~flags]
        excluded = np.zeros(len(df), dtype=bool)
        for _, ev in unobs.iterrows():
            centre = int(spy.index[spy["date"] == nearest_trading_date(spy["date"], pd.Timestamp(ev["calendar_date"]))][0])
            lo_d = spy["date"].iloc[max(0, centre - HALF_WINDOW)]
            hi_d = spy["date"].iloc[min(len(spy) - 1, centre + HALF_WINDOW)]
            excluded |= ((df["date"] >= lo_d) & (df["date"] <= hi_d)).to_numpy()
        _, pred = detector_outputs(df, CONFIG)
        y_obs, _ = event_mask(df, obs, HALF_WINDOW)
        keep = ~excluded
        for name, key in [(VOL_NAME, "volatility_percentile"), (CONF_NAME, f"hybrid_w{CONFIG['detectors']['default_hybrid_window']}")]:
            a = pred[key]
            main, alt = point_metrics(y_obs, a), point_metrics(y_obs[keep], a[keep])
            rows.append(
                {
                    "ticker": ticker, "method": name, "days_excluded": int(excluded.sum()),
                    "alarms_on_excluded_days": int(a[excluded].sum()),
                    **{f"main_{k}": main[k] for k in ["precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]},
                    **{f"excluded_{k}": alt[k] for k in ["precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]},
                }
            )
    return pd.DataFrame(rows)


def crossover_table(event_aware: pd.DataFrame) -> pd.DataFrame:
    """Table 12 recomputed exactly as in scripts/run_ablations.py."""
    rows = []
    for ticker in event_aware["ticker"].unique():
        for delay_cost in sorted(event_aware["delay_cost"].unique()):
            sub = event_aware[(event_aware["ticker"] == ticker) & (event_aware["delay_cost"] == delay_cost)]
            v = sub[sub["method"] == VOL_NAME].iloc[0]
            h = sub[sub["method"] == CONF_NAME].iloc[0]
            denominator = v["false_alarm_days"] - h["false_alarm_days"]
            numerator = (h["missed_events"] - v["missed_events"]) + delay_cost * (h["total_positive_delay"] - v["total_positive_delay"])
            rows.append(
                {
                    "ticker": ticker, "delay_cost": delay_cost,
                    "alarm_days_outside_volatility": int(v["false_alarm_days"]),
                    "alarm_days_outside_hybrid": int(h["false_alarm_days"]),
                    "missed_volatility": int(v["missed_events"]),
                    "missed_hybrid": int(h["missed_events"]),
                    "positive_delay_volatility": float(v["total_positive_delay"]),
                    "positive_delay_hybrid": float(h["total_positive_delay"]),
                    "crossover_ratio": float(numerator / denominator) if denominator else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare(name: str, old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Long-format comparison of every shared column, keyed rows matched exactly."""
    metrics = [c for c in old.columns if c not in keys and c in new.columns]
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    rows = []
    for _, r in merged.iterrows():
        key = "; ".join(f"{k}={r[k]}" for k in keys)
        if r["_merge"] != "both":
            rows.append({"table": name, "key": key, "metric": "(row)", "old": "present" if r["_merge"] == "left_only" else "absent",
                         "new": "present" if r["_merge"] == "right_only" else "absent", "abs_diff": np.nan, "changed": True})
            continue
        for m in metrics:
            o, n = r[f"{m}_old"], r[f"{m}_new"]
            if isinstance(o, (int, float, np.integer, np.floating, bool, np.bool_)) and isinstance(n, (int, float, np.integer, np.floating, bool, np.bool_)):
                both_nan = pd.isna(o) and pd.isna(n)
                diff = np.nan if both_nan else abs(float(n) - float(o))
                changed = (not both_nan) and (pd.isna(o) != pd.isna(n) or diff > 1e-12)
            else:
                diff, changed = np.nan, str(o) != str(n)
            rows.append({"table": name, "key": key, "metric": m, "old": o, "new": n, "abs_diff": diff, "changed": bool(changed)})
    return pd.DataFrame(rows)


def main() -> None:
    obs_table = observability_table()
    obs_table.to_csv(OUT / "observability_by_asset.csv", index=False)
    denominators = obs_table.groupby("ticker", sort=False)["observable"].agg(["size", "sum"]).rename(
        columns={"size": "panel_events", "sum": "observable_events"}).reset_index()
    denominators.insert(0, "event_panel", PANEL)
    denominators.to_csv(OUT / "denominators_by_asset.csv", index=False)

    events = events_by_ticker()
    detail = pd.concat([_asset_event_rows(t, CONFIG, observable_only=True, events=events[t], severity_labels=severity_labels())
                        for t in TICKERS], ignore_index=True)
    detail.to_csv(OUT / "event_level_metrics_observable.csv", index=False)
    tables = event_level_tables(detail)
    tables["by_asset"].to_csv(OUT / "event_level_summary_by_asset_observable.csv", index=False)
    tables["by_severity"].to_csv(OUT / "event_level_summary_by_severity_observable.csv", index=False)
    tables["table8"].to_csv(OUT / "table8_multi_asset_event_level_observable.csv", index=False)
    tables["table9"].to_csv(OUT / "table9_severity_observable.csv", index=False)

    metrics = pd.concat([evaluate_asset(t, CONFIG, observable_only=True, events=events[t]) for t in TICKERS], ignore_index=True)
    metrics.to_csv(OUT / "multi_asset_pointwise_metrics_observable.csv", index=False)
    tradeoff_figure(metrics)
    excluded_true_window_sensitivity().to_csv(OUT / "day_mask_sensitivity_true_window_excluded.csv", index=False)

    pointwise = compute_pointwise_cost(CONFIG, observable_only=True, events_by_ticker=events)
    event_aware = compute_event_aware_cost(CONFIG, observable_only=True, event_detail=detail, events_by_ticker=events)
    pointwise.to_csv(OUT / "pointwise_cost_by_asset_observable.csv", index=False)
    event_aware.to_csv(OUT / "event_aware_cost_by_asset_observable.csv", index=False)
    crossover = crossover_table(event_aware)
    crossover.to_csv(OUT / "table12_cost_crossover_observable.csv", index=False)

    old = lambda rel: pd.read_csv(ROOT / rel)  # noqa: E731
    old_t8 = old("results/event_level/table_multi_asset_event_level_summary.csv")
    old_t9 = old("results/event_level/table_event_level_severity_summary.csv")
    new_t9_same_estimator = tables["table9"].rename(columns={"mean_of_per_asset_median_first_delay": "median_first_delay"})
    comparisons = pd.concat(
        [
            compare("Table 5 / multi_asset_pointwise_metrics", old("results/multi_asset/multi_asset_pointwise_metrics.csv"), metrics, ["ticker", "method"]),
            compare("event_level_summary_by_asset", old("results/event_level/event_level_summary_by_asset.csv"), tables["by_asset"], ["ticker", "method"]),
            compare("event_level_summary_by_severity", old("results/event_level/event_level_summary_by_severity.csv"), tables["by_severity"], ["ticker", "severity_label", "method"]),
            compare("Table 8 / table_multi_asset_event_level_summary", old_t8, tables["table8"], ["ticker", "method"]),
            compare("Table 9 / table_event_level_severity_summary", old_t9, new_t9_same_estimator, ["severity_label", "method"]),
            compare("event_level_metrics (per event)", old("results/event_level/event_level_metrics.csv"), detail, ["ticker", "event_id", "method"]),
            compare("pointwise_cost_by_asset", old("results/cost/pointwise_cost_by_asset.csv"), pointwise, ["ticker", "ratio_fp_to_fn", "method"]),
            compare("Table 11 / event_aware_cost_by_asset", old("results/cost/event_aware_cost_by_asset.csv"), event_aware, ["ticker", "ratio_false_alarm_to_missed_event", "delay_cost", "method"]),
            compare("Table 12 / table_cost_crossover", old("results/ablations/table_cost_crossover.csv"), crossover, ["ticker", "delay_cost"]),
        ],
        ignore_index=True,
    )
    comparisons.to_csv(OUT / "old_vs_new_all_values.csv", index=False)
    changed = comparisons[comparisons["changed"]]
    changed.to_csv(OUT / "old_vs_new_changed_values.csv", index=False)

    print(denominators.to_string(index=False))
    print(f"compared {len(comparisons)} values; changed {len(changed)}")
    print(changed.groupby("table")["changed"].size().to_string())


if __name__ == "__main__":
    main()
