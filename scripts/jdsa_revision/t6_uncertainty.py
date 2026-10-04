"""Task 6: uncertainty intervals redesigned for each estimand.

Reviewer 1 minor 7. The submitted event bootstrap resamples events and takes the union
of their windows, so repeated draws collapse and dropped events' windows become
negative days; it also reports the resample mean as the estimate. It is kept in
results/paper/ for provenance and replaced here by:

* detection difference: paired event bootstrap; each event row is one unit and repeated
  draws stay repeated;
* delay difference: paired bootstrap over the events detected by both methods, conditional
  on that observed common set;
* day-level differences: paired circular block bootstrap of the daily tuples
  (event-window label, alarm a, alarm b) at block lengths 20, 60 and 121 trading days.
  The primary length, 121, is the full span of one +/-60-day labelled window, so blocks
  keep dependence at the scale of the principal evaluation window. It was fixed before any
  interval was computed; 20 and 60 are sensitivity checks and none is chosen by results.

Both comparisons are run on all five assets: the nested default (volatility trigger at
the 90th percentile against the submitted confirmation rule) and the matched-FPR pair
(volatility trigger at the confirmation rule's FPR, from t4). Point estimates are the
observed differences. Detector outputs are never re-estimated, so the intervals describe
variability of the evaluation sample within this panel, not retraining uncertainty and
not a population of possible crises. Under nested alarms the signs of the recall, FPR,
detection and common-event delay differences are fixed by construction; those rows are
flagged and are not evidence.

Reads the matched volatility thresholds from t4_operating_curves (run t4 first).

Outputs: <out root>/t6_uncertainty/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, N_RESAMPLES, OUT_ROOT, ROOT, SEED, TICKERS, load_asset, outdir  # noqa: E402
from src.paper_protocol import event_mask  # noqa: E402
from src.revision_analysis import (  # noqa: E402
    common_event_delays,
    component_scores,
    detector_alarms,
    event_spans,
    interval_summary,
    observed_day_deltas,
    paired_block_bootstrap,
    paired_event_bootstrap,
    per_event_outcomes,
)

OUT = outdir("t6_uncertainty")
BLOCKS = [20, 60, 121]
PRIMARY_BLOCK = 121
SEEDS = {"event_detection": SEED + 100, "common_delay": SEED + 200, "block": SEED + 300, "block_check": SEED + 900}


def matched_thresholds() -> dict[str, float]:
    t4 = OUT_ROOT / "t4_operating_curves"
    m = pd.concat([pd.read_csv(t4 / "matched_comparison_spy.csv"), pd.read_csv(t4 / "matched_comparison_other_assets.csv")])
    pick = m[(m["target_column"] == "fpr") & (m["match"] == "nearest")]
    return dict(zip(pick["ticker"], pick["threshold"].astype(float)))


def main() -> None:
    thresholds = matched_thresholds()
    summary_rows, spy_samples = [], []
    for a_index, ticker in enumerate(TICKERS):
        df, events = load_asset(ticker)
        scores = component_scores(df, CONFIG)
        alarms = detector_alarms(scores, CONFIG)
        spans = event_spans(df, events, HALF_WINDOW)
        y, _ = event_mask(df, events, HALF_WINDOW)
        vol_matched = (scores["volatility"] > thresholds[ticker]).astype(int)
        comparisons = {
            "volatility default vs confirmation default (nested)": (alarms["volatility"], alarms["retrospective_confirmation"], True),
            "volatility at matched FPR vs confirmation default (not nested)": (vol_matched, alarms["retrospective_confirmation"], False),
        }
        offset = 10 * a_index
        for name, (a, b, nested) in comparisons.items():
            base = {"ticker": ticker, "comparison": name, "n_events": len(spans)}
            pa, pb = per_event_outcomes(a, spans), per_event_outcomes(b, spans)
            d = pb["detected"].astype(float).to_numpy() - pa["detected"].astype(float).to_numpy()
            boot = paired_event_bootstrap(d, N_RESAMPLES, SEEDS["event_detection"] + offset)
            summary_rows.append({**base, "estimand": "event detection rate difference (b - a)",
                                 "scheme": "paired event bootstrap, event rows resampled with repeats kept", "block_length": np.nan,
                                 "n_units": len(d), "sign_fixed_by_nesting": nested, **interval_summary(boot, d.mean())})
            if ticker == "SPY":
                spy_samples.append(pd.DataFrame({"comparison": name, "estimand": "detection_difference", "value": boot}))
            _, common = common_event_delays(pa, pb)
            diffs = common["difference_b_minus_a"].to_numpy(float)
            if len(diffs):
                boot = paired_event_bootstrap(diffs, N_RESAMPLES, SEEDS["common_delay"] + offset)
                summary_rows.append({**base, "estimand": "first-alarm delay difference on common events (b - a), trading days",
                                     "scheme": "paired bootstrap over common events, conditional on the observed common set",
                                     "block_length": np.nan, "n_units": len(diffs), "sign_fixed_by_nesting": nested,
                                     **interval_summary(boot, diffs.mean())})
                if ticker == "SPY":
                    spy_samples.append(pd.DataFrame({"comparison": name, "estimand": "common_delay_difference", "value": boot}))
            observed = observed_day_deltas(y, a, b)
            for block in BLOCKS:
                samples = paired_block_bootstrap(y, a, b, block, N_RESAMPLES, SEEDS["block"] + offset)
                for metric in ["precision", "recall", "f1", "fpr"]:
                    key = f"delta_{metric}"
                    summary_rows.append({**base, "estimand": f"day-level {metric} difference (b - a)",
                                         "scheme": "paired circular block bootstrap of daily tuples", "block_length": block,
                                         "primary": block == PRIMARY_BLOCK, "n_units": len(y),
                                         "sign_fixed_by_nesting": bool(nested and metric in ("recall", "fpr")),
                                         **interval_summary(samples[key].to_numpy(), observed[key])})
                    if ticker == "SPY":
                        spy_samples.append(pd.DataFrame({"comparison": name, "estimand": f"{key}_block{block}", "value": samples[key]}))
            if ticker == "SPY":
                check = paired_block_bootstrap(y, a, b, PRIMARY_BLOCK, N_RESAMPLES, SEEDS["block_check"])
                for metric in ["precision", "f1"]:
                    key = f"delta_{metric}"
                    summary_rows.append({**base, "estimand": f"day-level {metric} difference (b - a), second seed",
                                         "scheme": "Monte Carlo check of the primary block length with another seed",
                                         "block_length": PRIMARY_BLOCK, "n_units": len(y), "sign_fixed_by_nesting": False,
                                         **interval_summary(check[key].to_numpy(), observed[key])})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "uncertainty_summary.csv", index=False)
    pd.concat(spy_samples, ignore_index=True).to_csv(OUT / "bootstrap_samples_spy.csv", index=False)

    day = summary[summary["estimand"].str.startswith("day-level") & ~summary["estimand"].str.contains("second seed")]
    excl = day.assign(excludes_zero=(day["ci_low"] > 0) | (day["ci_high"] < 0))
    sens = excl.pivot_table(index=["ticker", "comparison", "estimand"], columns="block_length", values="excludes_zero", aggfunc="first")
    sens.columns = [f"excludes_zero_block{int(c)}" for c in sens.columns]
    sens["same_conclusion_at_all_block_lengths"] = sens.nunique(axis=1) == 1
    sens.reset_index().to_csv(OUT / "block_length_sensitivity.csv", index=False)

    df, events = load_asset("SPY")
    alarms = detector_alarms(component_scores(df, CONFIG), CONFIG)
    y, _ = event_mask(df, events, HALF_WINDOW)
    old = pd.read_csv(ROOT / "results" / "paper" / "table_bootstrap_intervals.csv")
    obs = observed_day_deltas(y, alarms["volatility"], alarms["retrospective_confirmation"])
    old["observed_difference_active_panel"] = old["metric"].map({"delta_f1": obs["delta_f1"], "delta_fpr": obs["delta_fpr"]})
    old.rename(columns={"estimate": "submitted_estimate_is_bootstrap_mean"}).to_csv(OUT / "submitted_bootstrap_vs_observed.csv", index=False)

    pd.set_option("display.width", 280)
    cols = ["ticker", "comparison", "estimand", "block_length", "n_units", "sign_fixed_by_nesting", "observed", "ci_low", "ci_high"]
    print(summary[(summary["block_length"].isna()) | (summary["block_length"] == PRIMARY_BLOCK)][cols].round(4).to_string(index=False))
    print(sens[~sens["same_conclusion_at_all_block_lengths"]].to_string())


if __name__ == "__main__":
    main()
