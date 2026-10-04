"""Task 4: operating curves and the matched-false-positive-rate comparison.

Reviewer 1 main concern and Reviewer 2 minor 3. Settings fixed before any curve was
computed:

* grid: calibration quantiles 0.5000, 0.5001, ..., 1.0000 (5001 points), which covers
  the volatility trigger's false positive rate from above the default confirmation
  rule's down to zero;
* volatility curve: the volatility trigger at each quantile;
* confirmation curve: the confirmation rule with the shared volatility percentile
  swept, the instability percentile at 0.90 and W = 10, as submitted;
* matched point: the volatility operating point whose false positive rate is closest
  to the default confirmation rule's, with the nearest points on either side also
  reported; secondary check by total alarm days.

SPY, full period, retrospective evaluation, as in the submitted Table 3. The same
procedure is applied unchanged to the other four assets (observable events only) as
a secondary table. The matched point is a post-hoc diagnostic, not a deployment
threshold.

Outputs: <out root>/t4_operating_curves/
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import CONFIG, COLORS, HALF_WINDOW, TICKERS, figure_style, load_asset, outdir, save_figure  # noqa: E402
from src.cost_analysis import DELAY_COSTS, EVENT_FALSE_ALARM_RATIOS  # noqa: E402
from src.paper_protocol import event_mask  # noqa: E402
from src.revision_analysis import (  # noqa: E402
    common_event_delays,
    component_scores,
    confirmation_curve,
    cost_components,
    crossover_ratio,
    day_metrics,
    detector_alarms,
    event_cost,
    event_spans,
    event_summary,
    nearest_points,
    per_event_outcomes,
    volatility_curve,
)

OUT = outdir("t4_operating_curves")
GRID = np.round(np.linspace(0.5, 1.0, 5001), 4)


def asset_inputs(ticker: str) -> dict:
    df, events = load_asset(ticker)
    scores = component_scores(df, CONFIG)
    alarms = detector_alarms(scores, CONFIG)
    spans = event_spans(df, events, HALF_WINDOW)
    y, _ = event_mask(df, events, HALF_WINDOW)
    return {"df": df, "events": events, "scores": scores, "alarms": alarms, "spans": spans, "y": y}


def describe(label: str, pred: np.ndarray, inp: dict, reference_per_event: pd.DataFrame, reference_pred: np.ndarray) -> dict:
    per = per_event_outcomes(pred, inp["spans"])
    comp = cost_components(inp["y"], pred, per)
    common, _ = common_event_delays(reference_per_event, per)
    both = int(np.sum((pred == 1) & (reference_pred == 1)))
    row = {
        "operating_point": label, **day_metrics(inp["y"], pred), **event_summary(per), **comp,
        "common_events_with_default_confirmation": common["n_common"],
        "common_mean_delay_default_confirmation": common["common_mean_delay_a"],
        "common_mean_delay_this": common["common_mean_delay_b"],
        "events_detected_ids": ",".join(per.loc[per["detected"], "event_id"]),
        "alarm_days_shared_with_default_confirmation": both,
        "alarm_days_only_this": int(np.sum((pred == 1) & (reference_pred == 0))),
        "alarm_days_only_default_confirmation": int(np.sum((pred == 0) & (reference_pred == 1))),
    }
    for c_d in DELAY_COSTS:
        for ratio in EVENT_FALSE_ALARM_RATIOS:
            row[f"cost_cfa{ratio}_cd{c_d}"] = event_cost(comp, ratio, c_d)
    return row, per, comp


def matched_comparison(ticker: str, inp: dict, vol_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf = inp["alarms"]["retrospective_confirmation"]
    vol = inp["alarms"]["volatility"]
    conf_per = per_event_outcomes(conf, inp["spans"])
    target_fpr = day_metrics(inp["y"], conf)["fpr"]
    target_days = int(conf.sum())
    picks = pd.concat([nearest_points(vol_curve, "fpr", target_fpr), nearest_points(vol_curve, "alarm_days", target_days)],
                      ignore_index=True)
    rows, per_events = [], []
    for name, pred in [("volatility default (q=0.90)", vol), ("confirmation default (q=0.90, W=10)", conf)]:
        row, per, comp = describe(name, pred, inp, conf_per, conf)
        rows.append({"ticker": ticker, "match": "default", "quantile_min": 0.9, "quantile_max": 0.9, **row})
        per_events.append(per.assign(ticker=ticker, operating_point=name))
    conf_comp = cost_components(inp["y"], conf, conf_per)
    for _, pick in picks.iterrows():
        thr = float(pick["threshold"])
        pred = (inp["scores"]["volatility"] > thr).astype(int)
        name = f"volatility matched on {pick['target_column']} ({pick['match']})"
        row, per, comp = describe(name, pred, inp, conf_per, conf)
        for c_d in DELAY_COSTS:
            row[f"crossover_conf_vs_this_cd{c_d}"] = crossover_ratio(comp, conf_comp, c_d)
        rows.append({"ticker": ticker, "match": pick["match"], "target_column": pick["target_column"], "target": pick["target"],
                     "quantile_min": pick["quantile_min"], "quantile_max": pick["quantile_max"], "threshold": thr, **row})
        per_events.append(per.assign(ticker=ticker, operating_point=name))
    return pd.DataFrame(rows), pd.concat(per_events, ignore_index=True)


XMAX = 0.15


def figures(vol_curve: pd.DataFrame, conf_curve: pd.DataFrame, matched: pd.DataFrame) -> None:
    """Detection rate and precision against FPR, drawn on FPR in [0, 0.15].

    That range holds the whole confirmation curve and the volatility curve up to the point
    where it detects every event; beyond it the volatility curve stays at full detection,
    which the annotation states. Operating points with no alarms have no precision and
    are left out of the precision panel.
    """
    figure_style()
    defaults = matched[matched["match"] == "default"].set_index("operating_point")
    fpr_match = matched[(matched["target_column"] == "fpr") & (matched["match"] == "nearest")].iloc[0]
    full = vol_curve[vol_curve["detection_rate"] >= 1.0]
    tail_note = (f"Volatility trigger: full detection from FPR {full['fpr'].min():.3f}\n"
                 f"to {vol_curve['fpr'].max():.3f} (quantile {vol_curve['quantile'].min():.2f})") if not full.empty else ""
    for metric, ylabel, name in [("detection_rate", "Event detection rate", "detection_vs_fpr_spy"),
                                 ("precision", "Precision", "precision_vs_fpr_spy")]:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for curve, label, color in [(vol_curve, "Volatility trigger", COLORS["Volatility"]),
                                    (conf_curve, "Confirmation rule", COLORS["Confirmation"])]:
            c = curve.drop_duplicates("alarm_days").sort_values("fpr")
            if metric == "precision":
                c = c[c["alarm_days"] > 0]
            ax.plot(c["fpr"], c[metric], color=color, lw=1.3, label=label, drawstyle="steps-post" if metric == "detection_rate" else "default")
        v = defaults.loc["volatility default (q=0.90)"]
        h = defaults.loc["confirmation default (q=0.90, W=10)"]
        ax.scatter([v["fpr"]], [v[metric]], color=COLORS["Volatility"], marker="o", s=42, zorder=3, label="Volatility, 90th percentile")
        ax.scatter([h["fpr"]], [h[metric]], color=COLORS["Confirmation"], marker="s", s=42, zorder=3, label="Confirmation, default")
        ax.scatter([fpr_match["fpr"]], [fpr_match[metric]], facecolor="white", edgecolor=COLORS["Volatility"], marker="o", s=52,
                   linewidth=1.4, zorder=4, label="Volatility at matched FPR")
        ax.set_xlim(-0.003, XMAX)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        if metric == "detection_rate" and tail_note:
            ax.text(XMAX * 0.97, 0.60, tail_note, ha="right", va="center", fontsize=7, color="#555555")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right" if metric == "detection_rate" else "lower left")
        fig.tight_layout()
        save_figure(fig, OUT / name)
        plt.close(fig)


def main() -> None:
    if "--figures-only" in sys.argv:
        figures(pd.read_csv(OUT / "volatility_curve_spy.csv"), pd.read_csv(OUT / "confirmation_curve_spy.csv"),
                pd.read_csv(OUT / "matched_comparison_spy.csv"))
        return
    spy = asset_inputs("SPY")
    conf_per = per_event_outcomes(spy["alarms"]["retrospective_confirmation"], spy["spans"])
    vol_per = per_event_outcomes(spy["alarms"]["volatility"], spy["spans"])
    vol_curve = volatility_curve(spy["scores"], spy["y"], spy["spans"], GRID, reference=conf_per)
    conf_curve = confirmation_curve(spy["scores"], CONFIG, spy["y"], spy["spans"], GRID, reference=vol_per, sweep="volatility")
    vol_curve.to_csv(OUT / "volatility_curve_spy.csv", index=False)
    conf_curve.to_csv(OUT / "confirmation_curve_spy.csv", index=False)
    matched, per_events = matched_comparison("SPY", spy, vol_curve)
    matched.to_csv(OUT / "matched_comparison_spy.csv", index=False)
    per_events.to_csv(OUT / "matched_comparison_spy_per_event.csv", index=False)
    figures(vol_curve, conf_curve, matched)

    others, others_per = [], []
    for ticker in TICKERS[1:]:
        inp = asset_inputs(ticker)
        cp = per_event_outcomes(inp["alarms"]["retrospective_confirmation"], inp["spans"])
        curve = volatility_curve(inp["scores"], inp["y"], inp["spans"], GRID, reference=cp)
        curve.to_csv(OUT / f"volatility_curve_{ticker}.csv", index=False)
        m, p = matched_comparison(ticker, inp, curve)
        others.append(m)
        others_per.append(p)
    pd.concat(others, ignore_index=True).to_csv(OUT / "matched_comparison_other_assets.csv", index=False)
    pd.concat(others_per, ignore_index=True).to_csv(OUT / "matched_comparison_other_assets_per_event.csv", index=False)

    cols = ["ticker", "operating_point", "quantile_min", "quantile_max", "fpr", "alarm_days", "precision", "recall", "f1",
            "events_detected", "n_events", "common_events_with_default_confirmation", "common_mean_delay_default_confirmation",
            "common_mean_delay_this", "A_out", "M", "D_plus"]
    pd.set_option("display.width", 250)
    print(matched[cols].to_string(index=False))
    print(pd.concat(others)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
