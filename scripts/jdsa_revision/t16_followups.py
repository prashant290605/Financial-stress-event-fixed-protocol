"""Task 16: the fully causal confirmation rule at a matched false positive rate, and detections by window width.

Reviewer 1 further comment 1 asks that the causal variant carry the main results, and the main concern
asks for comparisons at a common alarm rate. Task 4 matches the volatility trigger to the symmetric
(retrospective) confirmation rule; this task applies the same matching to the fully causal rule, which
reads no close after the alarm day. Nothing is refitted and nothing is tuned:

* rules: the volatility trigger and the fully causal confirmation rule (instability indicator lagged by
  30 trading days, trailing window W = 10) at their submitted defaults;
* grid: the calibration quantiles of Task 4, 0.5000, 0.5001, ..., 1.0000;
* matched point: the volatility operating point whose false positive rate is closest to the fully causal
  rule's, chosen by ``nearest_points`` exactly as in Task 4 (rows with the same alarm-day count collapsed;
  the nearest points at or above and at or below also reported);
* periods: the full period, with all 13 observable events, and the post-calibration causal evaluation of
  Task 2, from the operational start written by Task 2 to the end of the sample, scoring the events whose
  whole window lies after that start (E09 to E13). In the second period the false positive rate, the
  alarm days and the costs are counted from the operational start only.

At its default the fully causal rule is nested in the volatility trigger (it can alarm only when
v(t) = 1); the script checks this, and checks that the matched trigger has exactly the fully causal
rule's false positive count, as the manuscript states. The matched threshold is chosen with the event
labels, so it is a retrospective diagnostic, not a deployment threshold.

The second output counts the events detected by the default volatility trigger and the default symmetric
confirmation rule at the four event-window half-widths of the submitted window sensitivity (20, 30, 45 and
60 trading days); Table 13 of the paper tables reports only day-level metrics for those widths.

Outputs: <out root>/t16_followups/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, OUT_ROOT, load_asset, outdir, write_json  # noqa: E402
from src.cost_analysis import DELAY_COSTS, EVENT_FALSE_ALARM_RATIOS  # noqa: E402
from src.paper_protocol import event_mask  # noqa: E402
from src.revision_analysis import (  # noqa: E402
    common_event_delays,
    component_scores,
    cost_components,
    crossover_ratio,
    day_metrics,
    detector_alarms,
    event_cost,
    event_spans,
    event_summary,
    nearest_points,
    per_event_outcomes,
    quantile_threshold,
)

OUT = outdir("t16_followups")
GRID = np.round(np.linspace(0.5, 1.0, 5001), 4)
CAUSAL = "fully_causal_confirmation"
CAUSAL_LABEL = "fully causal confirmation (default, W=10)"
WIDTHS = [20, 30, 45, 60]


def period_curve(scores: dict, y: np.ndarray, start: int) -> pd.DataFrame:
    """Volatility trigger at every grid quantile, day-level metrics counted from ``start``."""
    te = scores["train_end"]
    rows = []
    for q in GRID:
        thr = quantile_threshold(scores["volatility"], te, q)
        pred = (scores["volatility"] > thr).astype(int)
        rows.append({"quantile": float(q), "threshold": thr, **day_metrics(y, pred, start)})
    return pd.DataFrame(rows)


def describe(period: str, label: str, pred: np.ndarray, y: np.ndarray, scope: pd.DataFrame, start: int,
             ref_per: pd.DataFrame, ref_pred: np.ndarray) -> tuple[dict, pd.DataFrame, dict]:
    per = per_event_outcomes(pred, scope)
    comp = cost_components(y, pred, per, start)
    common, _ = common_event_delays(ref_per, per)
    p, r = np.asarray(pred[start:], int), np.asarray(ref_pred[start:], int)
    row = {
        "period": period, "operating_point": label, **day_metrics(y, pred, start), **event_summary(per), **comp,
        "events_detected_ids": ",".join(per.loc[per["detected"], "event_id"]),
        "common_events_with_fully_causal": common["n_common"],
        "common_mean_delay_fully_causal": common["common_mean_delay_a"],
        "common_mean_delay_this": common["common_mean_delay_b"],
        "alarm_days_shared_with_fully_causal": int(np.sum((p == 1) & (r == 1))),
        "alarm_days_only_this": int(np.sum((p == 1) & (r == 0))),
        "alarm_days_only_fully_causal": int(np.sum((p == 0) & (r == 1))),
    }
    for c_d in DELAY_COSTS:
        for ratio in EVENT_FALSE_ALARM_RATIOS:
            row[f"cost_cfa{ratio}_cd{c_d}"] = event_cost(comp, ratio, c_d)
    return row, per, comp


def compare(period: str, start: int, scores: dict, alarms: dict, y: np.ndarray, spans: pd.DataFrame) -> dict:
    scope = spans[spans["lo"] >= start].reset_index(drop=True)
    causal, vol = alarms[CAUSAL], alarms["volatility"]
    causal_per = per_event_outcomes(causal, scope)
    target = day_metrics(y, causal, start)["fpr"]
    picks = nearest_points(period_curve(scores, y, start), "fpr", target)
    rows, pers, costs = [], [], []
    _, _, causal_comp = describe(period, CAUSAL_LABEL, causal, y, scope, start, causal_per, causal)
    for label, pred in [(CAUSAL_LABEL, causal), ("volatility default (q=0.90)", vol)]:
        row, per, _ = describe(period, label, pred, y, scope, start, causal_per, causal)
        rows.append({**row, "match": "default", "quantile_min": 0.9, "quantile_max": 0.9})
        pers.append(per.assign(period=period, operating_point=label))
    for _, pick in picks.iterrows():
        thr = float(pick["threshold"])
        pred = (scores["volatility"] > thr).astype(int)
        label = f"volatility matched on fpr ({pick['match']})"
        row, per, comp = describe(period, label, pred, y, scope, start, causal_per, causal)
        for c_d in DELAY_COSTS:
            row[f"crossover_causal_vs_this_cd{c_d}"] = crossover_ratio(comp, causal_comp, c_d)
            row[f"cost_difference_causal_minus_this_cd{c_d}_excluding_false_alarm_term"] = (
                (causal_comp["M"] - comp["M"]) + c_d * (causal_comp["D_plus"] - comp["D_plus"]))
        rows.append({**row, "match": pick["match"], "target": target, "quantile_min": pick["quantile_min"],
                     "quantile_max": pick["quantile_max"], "threshold": thr})
        pers.append(per.assign(period=period, operating_point=label))
    for row in rows:
        for c_d in DELAY_COSTS:
            for ratio in EVENT_FALSE_ALARM_RATIOS:
                costs.append({"period": period, "operating_point": row["operating_point"], "match": row["match"],
                              "ratio_false_alarm_to_missed_event": ratio, "delay_cost": c_d,
                              "A_out": row["A_out"], "M": row["M"], "D_plus": row["D_plus"], "cost": row[f"cost_cfa{ratio}_cd{c_d}"]})
    nested = bool(np.all(causal[start:] <= vol[start:]))
    return {"table": pd.DataFrame(rows), "per_event": pd.concat(pers, ignore_index=True), "costs": pd.DataFrame(costs), "nested_default": nested}


def window_width_detections(df: pd.DataFrame, events: pd.DataFrame, alarms: dict) -> pd.DataFrame:
    rows = []
    for width in WIDTHS:
        spans = event_spans(df, events, width)
        for rule in ["volatility", "retrospective_confirmation"]:
            per = per_event_outcomes(alarms[rule], spans)
            rows.append({"half_window": width, "rule": rule, "events_detected": int(per["detected"].sum()), "n_events": len(per),
                         "events_detected_ids": ",".join(per.loc[per["detected"], "event_id"])})
    return pd.DataFrame(rows)


def main() -> None:
    df, events = load_asset("SPY")
    scores = component_scores(df, CONFIG)
    alarms = detector_alarms(scores, CONFIG)
    spans = event_spans(df, events, HALF_WINDOW)
    y, _ = event_mask(df, events, HALF_WINDOW)
    op = json.loads((OUT_ROOT / "t2_evaluation_periods" / "operational_start.json").read_text(encoding="utf-8"))
    start = int(op["operational_start_index"])
    assert df["date"].iloc[start].date().isoformat() == op["operational_start_date"]
    full = compare("full_period", 0, scores, alarms, y, spans)
    post = compare("post_calibration_causal", start, scores, alarms, y, spans)
    if not (full["nested_default"] and post["nested_default"]):
        raise RuntimeError("the fully causal rule is not nested in the volatility trigger at its default")
    table = pd.concat([full["table"], post["table"]], ignore_index=True)
    for period in ["full_period", "post_calibration_causal"]:
        t = table[table["period"] == period].set_index("operating_point")
        if int(t.loc["volatility matched on fpr (nearest)", "fp"]) != int(t.loc[CAUSAL_LABEL, "fp"]):
            raise RuntimeError(f"{period}: the matched trigger does not have the fully causal rule's false positive count")
    table.to_csv(OUT / "causal_matched_comparison.csv", index=False)
    pd.concat([full["per_event"], post["per_event"]], ignore_index=True).merge(
        events[["event_id", "event_name"]], on="event_id").to_csv(OUT / "causal_matched_per_event.csv", index=False)
    pd.concat([full["costs"], post["costs"]], ignore_index=True).to_csv(OUT / "causal_matched_costs.csv", index=False)
    widths = window_width_detections(df, events, alarms)
    widths.to_csv(OUT / "window_width_detections.csv", index=False)

    summary = {"operational_start_date": op["operational_start_date"], "fully_causal_nested_in_volatility_at_default": True,
               "matched_point_has_equal_false_positive_count": True}
    for period in ["full_period", "post_calibration_causal"]:
        t = table[table["period"] == period].set_index("operating_point")
        c, m = t.loc[CAUSAL_LABEL], t.loc["volatility matched on fpr (nearest)"]
        cheaper = sorted({"volatility matched" if m[f"cost_cfa{r}_cd{c_d}"] < c[f"cost_cfa{r}_cd{c_d}"] else
                          "fully causal" if m[f"cost_cfa{r}_cd{c_d}"] > c[f"cost_cfa{r}_cd{c_d}"] else f"tie at c_D={c_d}"
                          for c_d in DELAY_COSTS for r in EVENT_FALSE_ALARM_RATIOS})
        summary[period] = {
            "n_events": int(c["n_events"]),
            "fully_causal": {k: c[k] for k in ["fp", "fpr", "alarm_days", "precision", "recall", "f1", "events_detected",
                                                "events_detected_ids", "A_out", "M", "D_plus"]},
            "volatility_matched": {k: m[k] for k in ["quantile_min", "quantile_max", "fp", "fpr", "alarm_days", "precision", "recall", "f1",
                                                      "events_detected", "events_detected_ids", "A_out", "M", "D_plus",
                                                      "common_events_with_fully_causal", "common_mean_delay_fully_causal",
                                                      "common_mean_delay_this", "alarm_days_shared_with_fully_causal",
                                                      "alarm_days_only_this", "alarm_days_only_fully_causal"]},
            "cost_difference_causal_minus_matched_excluding_false_alarm_term":
                {f"c_D={c_d}": float(m[f"cost_difference_causal_minus_this_cd{c_d}_excluding_false_alarm_term"]) for c_d in DELAY_COSTS},
            "cheaper_rule_over_tested_cost_settings": cheaper,
        }
    summary["window_width_detections"] = {f"+/-{r.half_window} {r.rule}": int(r.events_detected) for r in widths.itertuples()}
    write_json(OUT / "causal_matched_summary.json", summary)
    cols = ["period", "operating_point", "quantile_min", "quantile_max", "fp", "fpr", "alarm_days", "precision", "recall", "f1",
            "events_detected", "n_events", "events_detected_ids", "A_out", "M", "D_plus",
            "common_events_with_fully_causal", "common_mean_delay_fully_causal", "common_mean_delay_this"]
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    print(table[cols].to_string(index=False))
    print(widths.to_string(index=False))
    for period in ["full_period", "post_calibration_causal"]:
        print(period, summary[period]["cost_difference_causal_minus_matched_excluding_false_alarm_term"],
              summary[period]["cheaper_rule_over_tested_cost_settings"])


if __name__ == "__main__":
    main()
