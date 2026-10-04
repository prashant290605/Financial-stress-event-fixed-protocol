"""Task 5: first-alarm delay on the events both methods detect.

Reviewer 2 minor 4 and Reviewer 1 further comment 1. Mean delays over each method's
own detected events mix speed with composition when the two methods detect different
events. Every comparison here reports the own-set means beside the means and medians
over the common events, the paired differences per event, and the events detected by
one method only. Differences are the second method minus the first.

Reads the operational start from t2_evaluation_periods and the matched volatility threshold
from t4_operating_curves (run t2 and t4 first).

Outputs: <out root>/t5_common_event_delay/
"""
from __future__ import annotations

import json

import pandas as pd

from _common import CONFIG, HALF_WINDOW, OUT_ROOT, TICKERS, load_asset, outdir  # noqa: E402
from src.revision_analysis import (  # noqa: E402
    calibration_cutoff,
    common_event_delays,
    component_scores,
    detector_alarms,
    event_spans,
    per_event_outcomes,
    split_by_cutoff,
)

OUT = outdir("t5_common_event_delay")


def main() -> None:
    rows, per_rows = [], []

    def add(comparison: str, ticker: str, scope: str, a_label: str, a: pd.DataFrame, b_label: str, b: pd.DataFrame) -> None:
        summary, per = common_event_delays(a, b)
        rows.append({"comparison": comparison, "ticker": ticker, "event_scope": scope, "method_a": a_label, "method_b": b_label, **summary})
        per_rows.append(per.assign(comparison=comparison, ticker=ticker, method_a=a_label, method_b=b_label))

    matched = pd.read_csv(OUT_ROOT / "t4_operating_curves" / "matched_comparison_spy.csv")
    matched_thr = float(matched.loc[(matched["target_column"] == "fpr") & (matched["match"] == "nearest"), "threshold"].iloc[0])

    for ticker in TICKERS:
        df, events = load_asset(ticker)
        scores = component_scores(df, CONFIG)
        alarms = detector_alarms(scores, CONFIG)
        spans = event_spans(df, events, HALF_WINDOW)
        vol = per_event_outcomes(alarms["volatility"], spans)
        conf = per_event_outcomes(alarms["retrospective_confirmation"], spans)
        scope = f"observable events ({len(spans)})"
        add("volatility vs confirmation (submitted defaults)", ticker, scope, "volatility", vol, "retrospective_confirmation", conf)
        if ticker != "SPY":
            continue
        causal = per_event_outcomes(alarms["fully_causal_confirmation"], spans)
        add("volatility vs fully causal confirmation, full period (not prospective)", ticker, scope, "volatility", vol,
            "fully_causal_confirmation", causal)
        cut = calibration_cutoff(df, CONFIG)
        post = split_by_cutoff(spans, cut.train_end)["post_calibration"]
        post_scope = f"post-calibration events ({','.join(post['event_id'])})"
        add("volatility vs retrospective confirmation, post-calibration retrospective (Table A)", ticker, post_scope, "volatility",
            per_event_outcomes(alarms["volatility"], post), "retrospective_confirmation",
            per_event_outcomes(alarms["retrospective_confirmation"], post))
        op = json.loads((OUT_ROOT / "t2_evaluation_periods" / "operational_start.json").read_text(encoding="utf-8"))
        prospective = spans[spans["lo"] >= int(op["operational_start_index"])]
        pro_scope = f"prospective operational events ({','.join(prospective['event_id'])})"
        add("volatility vs fully causal confirmation, prospective operational (Table B)", ticker, pro_scope, "volatility",
            per_event_outcomes(alarms["volatility"], prospective), "fully_causal_confirmation",
            per_event_outcomes(alarms["fully_causal_confirmation"], prospective))
        vol_matched = per_event_outcomes((scores["volatility"] > matched_thr).astype(int), spans)
        add("volatility at matched FPR vs confirmation (submitted default)", ticker, scope, "volatility_matched_fpr", vol_matched,
            "retrospective_confirmation", conf)

    summary = pd.DataFrame(rows)
    per_event = pd.concat(per_rows, ignore_index=True)
    summary.to_csv(OUT / "common_event_delay_summary.csv", index=False)
    per_event.to_csv(OUT / "common_event_delay_per_event.csv", index=False)

    hyg = summary[(summary["ticker"] == "HYG")].iloc[0]
    verdict = {
        "own_set_mean_delay_volatility": hyg["own_set_mean_delay_a"],
        "own_set_mean_delay_confirmation": hyg["own_set_mean_delay_b"],
        "common_events": hyg["n_common"],
        "common_mean_delay_volatility": hyg["common_mean_delay_a"],
        "common_mean_delay_confirmation": hyg["common_mean_delay_b"],
        "events_detected_by_volatility_only": hyg["events_only_a"],
    }
    pd.Series(verdict).to_csv(OUT / "hyg_composition_check.csv", header=["value"])
    cols = ["comparison", "ticker", "detected_a", "detected_b", "n_common", "own_set_mean_delay_a", "own_set_mean_delay_b",
            "common_mean_delay_a", "common_mean_delay_b", "common_median_delay_a", "common_median_delay_b",
            "paired_mean_difference", "events_only_a", "events_only_b"]
    pd.set_option("display.width", 260)
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
