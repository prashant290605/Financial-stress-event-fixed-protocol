"""Task 2: post-calibration retrospective evaluation (Table A) and prospective operational evaluation (Table B).

Reviewer 1 further comments 1-2 and Reviewer 2 majors 2-3. Calibration constants and
thresholds are the submitted ones, fitted on the first 70 per cent of the series; nothing
is refitted.

* Table A starts on the first trading day after the nominal calibration cutoff and compares
  the volatility trigger with the submitted symmetric confirmation rule. It is labelled
  retrospective because the symmetric rule reads days after each alarm.
* Table B starts on the operational start date derived by ``derive_operational_start``: the
  first trading day after the latest raw close read by any fitted constant or threshold.
  It compares the volatility trigger with the fully causal confirmation rule on exactly the
  same scoring days.

Event-level metrics use only events whose whole +/-60-day window lies inside the scored
period. The full-period numbers are kept in a separate reference table.

Outputs: <out root>/t2_evaluation_periods/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, PANEL, ROOT, load_asset, outdir, write_json  # noqa: E402
from src.cost_analysis import DELAY_COSTS, EVENT_FALSE_ALARM_RATIOS  # noqa: E402
from src.multi_asset_data import _read_price_csv  # noqa: E402
from src.paper_protocol import detector_outputs, event_mask  # noqa: E402
from src.revision_analysis import (  # noqa: E402
    calibration_cutoff,
    common_event_delays,
    component_scores,
    cost_components,
    crossover_ratio,
    cusum_components,
    day_metrics,
    derive_operational_start,
    detector_alarms,
    event_cost,
    event_spans,
    event_summary,
    per_event_outcomes,
    pipeline_from_prices,
)

OUT = outdir("t2_evaluation_periods")
LABELS = {
    "volatility": "Volatility percentile trigger",
    "retrospective_confirmation": "Confirmation W=10, symmetric window (submitted; reads days after t)",
    "fully_causal_confirmation": "Confirmation W=10, fully causal (lagged instability, trailing window)",
}


def evaluate(name: str, start: int, rules: list[str], alarms: dict, y: np.ndarray, spans: pd.DataFrame,
             df: pd.DataFrame, events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scope = spans[spans["lo"] >= start]
    rows, per_rows, cost_rows = [], [], []
    comps, pers = {}, {}
    for rule in rules:
        pred = alarms[rule]
        per = per_event_outcomes(pred, scope)
        comp = cost_components(y, pred, per, start)
        comps[rule], pers[rule] = comp, per
        rows.append({
            "table": name, "event_panel": PANEL, "detector": rule, "detector_label": LABELS[rule],
            "first_scored_date": df["date"].iloc[start].date().isoformat(), "last_scored_date": df["date"].iloc[-1].date().isoformat(),
            **{f"day_{k}": v for k, v in day_metrics(y, pred, start).items()}, **event_summary(per), **comp,
            "events_in_scope": ",".join(scope["event_id"]), "events_detected_ids": ",".join(per.loc[per["detected"], "event_id"]),
        })
        per_rows.append(per.assign(table=name, detector=rule))
        for c_d in DELAY_COSTS:
            for ratio in EVENT_FALSE_ALARM_RATIOS:
                cost_rows.append({"table": name, "detector": rule, "ratio_false_alarm_to_missed_event": ratio, "delay_cost": c_d,
                                  **comp, "cost": event_cost(comp, ratio, c_d)})
    other = rules[1]
    summary, common = common_event_delays(pers["volatility"], pers[other])
    cross = pd.DataFrame([{"table": name, "reference": "volatility", "alternative": other, "delay_cost": c_d,
                           "crossover_ratio": crossover_ratio(comps["volatility"], comps[other], c_d)} for c_d in DELAY_COSTS])
    table = pd.DataFrame(rows)
    for k in ["n_common", "common_mean_delay_a", "common_mean_delay_b", "common_median_delay_a", "common_median_delay_b",
              "paired_mean_difference"]:
        table[f"common_event_{k}"] = summary[k]
    per_event = pd.concat(per_rows, ignore_index=True).merge(events[["event_id", "event_name"]], on="event_id")
    return {"table": table, "per_event": per_event, "costs": pd.DataFrame(cost_rows), "crossover": cross,
            "common": common.assign(table=name, method_a="volatility", method_b=other)}


def main() -> None:
    df, events = load_asset("SPY")
    cut = calibration_cutoff(df, CONFIG)
    te = cut.train_end
    scores = component_scores(df, CONFIG)
    alarms = detector_alarms(scores, CONFIG)
    _, submitted = detector_outputs(df, CONFIG)
    assert np.array_equal(alarms["volatility"], submitted["volatility_percentile"])
    assert np.array_equal(alarms["retrospective_confirmation"], submitted["hybrid_w10"])

    prices = _read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]).reset_index(drop=True)
    raw = pipeline_from_prices(prices, CONFIG, te)
    for rule in ["volatility", "retrospective_confirmation", "fully_causal_confirmation"]:
        assert np.array_equal(raw["alarms"][rule], alarms[rule]), rule
    op = derive_operational_start(prices, CONFIG, te, seed=CONFIG["seed"])
    if not (op["check_all_closes_from_start_perturbed_leaves_calibration_identical"] and op["check_single_close_before_start_moves_calibration"]):
        raise RuntimeError("operational start failed its checks")
    start_b = op["operational_start_index"]
    write_json(OUT / "operational_start.json", op)
    cusum = cusum_components(df, CONFIG)
    write_json(OUT / "calibration_cutoff.json", {
        "n_days": len(df), "calibration_fraction": CONFIG["detectors"]["calibration_fraction"], "train_end_index": te,
        "first_date": df["date"].iloc[0], "last_calibration_date": cut.last_calibration_date,
        "first_post_calibration_date": cut.first_evaluation_date, "operational_start_date": op["operational_start_date"],
        "latest_calibration_close_date": op["latest_calibration_close_date"], "last_date": df["date"].iloc[-1],
        "note": "train_end is 70 per cent of the archived sample length, so the cutoff date itself is fixed retrospectively; "
                "no price value after the latest calibration close enters any fitted quantity.",
        "frozen_calibration_quantities": {**op["calibration_quantities"], "cusum_threshold": cusum["threshold"]},
    })

    spans = event_spans(df, events, HALF_WINDOW)
    y, _ = event_mask(df, events, HALF_WINDOW)
    a = evaluate("A_post_calibration_retrospective", te, ["volatility", "retrospective_confirmation"], alarms, y, spans, df, events)
    b = evaluate("B_prospective_operational", start_b, ["volatility", "fully_causal_confirmation"], alarms, y, spans, df, events)
    full = evaluate("full_period_reference", 0, ["volatility", "retrospective_confirmation"], alarms, y, spans, df, events)
    full_causal = evaluate("full_period_reference", 0, ["volatility", "fully_causal_confirmation"], alarms, y, spans, df, events)
    for key, res in [("table_A_post_calibration_retrospective", a), ("table_B_prospective_operational", b)]:
        res["table"].to_csv(OUT / f"{key}.csv", index=False)
        res["per_event"].to_csv(OUT / f"{key}_per_event.csv", index=False)
        res["costs"].to_csv(OUT / f"{key}_costs.csv", index=False)
        res["crossover"].to_csv(OUT / f"{key}_crossover.csv", index=False)
        res["common"].to_csv(OUT / f"{key}_common_events.csv", index=False)
    reference = pd.concat([full["table"], full_causal["table"][full_causal["table"]["detector"] == "fully_causal_confirmation"]], ignore_index=True)
    reference["note"] = np.where(reference["detector"] == "fully_causal_confirmation",
                                 "full-period sensitivity: calibration-period alarms use constants fitted on later data; not prospective",
                                 "full-period retrospective baseline")
    reference.to_csv(OUT / "table_full_period_reference.csv", index=False)
    pd.concat([full["crossover"], full_causal["crossover"]]).to_csv(OUT / "table_full_period_reference_crossover.csv", index=False)

    split = []
    for rule in ["volatility", "retrospective_confirmation", "fully_causal_confirmation"]:
        per = per_event_outcomes(alarms[rule], spans)
        cal = per[per["hi"] < te]
        post = per[per["lo"] >= te]
        split.append({"detector": rule, "calibration_events": len(cal), "calibration_detected": int(cal["detected"].sum()),
                      "post_cutoff_events": len(post), "post_cutoff_detected": int(post["detected"].sum()),
                      "straddling_events": int(((per["lo"] < te) & (per["hi"] >= te)).sum())})
    pd.DataFrame(split).to_csv(OUT / "detections_by_calibration_split.csv", index=False)

    gap = slice(te, start_b)
    write_json(OUT / "alarms_between_cutoff_and_operational_start.json", {
        "first_date": df["date"].iloc[te], "last_date": df["date"].iloc[start_b - 1], "days": start_b - te,
        **{rule: int(alarms[rule][gap].sum()) for rule in LABELS},
    })
    cols = ["table", "detector", "first_scored_date", "day_precision", "day_recall", "day_f1", "day_fpr", "day_alarm_days", "A_out",
            "events_detected", "n_events", "events_in_scope", "M", "D_plus", "common_event_n_common", "common_event_common_mean_delay_a",
            "common_event_common_mean_delay_b"]
    pd.set_option("display.width", 300)
    print({k: op[k] for k in ["latest_calibration_close_date", "operational_start_date"]})
    print(pd.concat([a["table"], b["table"], reference])[cols].to_string(index=False))
    print(pd.concat([a["crossover"], b["crossover"]]).to_string(index=False))


if __name__ == "__main__":
    main()
