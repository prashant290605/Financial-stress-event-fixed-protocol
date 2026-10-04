"""Task 3: timestamp semantics of the confirmation variants, checked by perturbation.

For an issue date t, every raw adjusted close dated after t is perturbed and the
whole pipeline is recomputed from the raw file, with the calibration cutoff index held
fixed. Two modes are run:

* frozen: standardization constants and thresholds held at their fitted values, which
  tests the signal path of each rule at any date, calibration period included;
* refit: constants and thresholds refitted on the perturbed data, exactly as the
  submitted code does, which tests the rule end to end. This is the operational
  question and is asked at post-calibration issue dates.

A negative control checks that the same test flags the symmetric default rule, which
reads days after t.

Reads the operational start derived by t2_evaluation_periods (run t2 first).

Outputs: <out root>/t3_causality/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import CONFIG, OUT_ROOT, ROOT, outdir, write_json  # noqa: E402
from src.multi_asset_data import _read_price_csv  # noqa: E402
from src.multi_asset_evaluation import _load_processed_asset  # noqa: E402
from src.paper_protocol import detector_outputs  # noqa: E402
from src.revision_analysis import calibration_train_end, pipeline_from_prices  # noqa: E402

OUT = outdir("t3_causality")
RULES = ["fully_causal_confirmation", "retrospective_confirmation", "trailing_confirmation", "volatility"]


def perturbed(prices: pd.DataFrame, date: pd.Timestamp, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    """Copy of ``prices`` with every close dated after ``date`` changed."""
    p = prices.copy()
    after = (p["date"] > date).to_numpy()
    k = int(after.sum())
    values = p.loc[after, "price"].to_numpy(float)
    if kind == "noise":
        values = values * np.exp(rng.normal(0.0, 0.05, k))
    elif kind == "crash":
        values = values * np.exp(-0.5 * np.minimum(np.arange(1, k + 1), 20) / 20)
    elif kind == "spike":
        values[0] = values[0] * 3.0
    elif kind == "flat":
        values = np.full(k, values[0])
    else:
        raise ValueError(kind)
    p.loc[after, "price"] = values
    return p


def main() -> None:
    prices = _read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]).reset_index(drop=True)
    stored = _load_processed_asset("SPY")
    te = calibration_train_end(len(stored), CONFIG)
    base = pipeline_from_prices(prices, CONFIG, te)
    df = base["df"]
    assert len(df) == len(stored) and np.allclose(df["signal"], stored["signal"], atol=1e-12)
    _, submitted = detector_outputs(stored, CONFIG)
    assert np.array_equal(base["alarms"]["retrospective_confirmation"], submitted["hybrid_w10"])
    constants, thresholds = base["scores"]["calibration_constants"], base["alarms"]["thresholds"]
    dates = df["date"]
    w = CONFIG["features"]["mmd_window"]
    op = json.loads((OUT_ROOT / "t2_evaluation_periods" / "operational_start.json").read_text(encoding="utf-8"))
    horizon = int(op["latest_calibration_close_index"])  # latest close read by any fitted quantity (derived in t2)
    start = int(op["operational_start_index"])
    assert horizon == te - 1 + w - 1 and start == horizon + 1, "derived horizon disagrees with the MMD window arithmetic"

    # ---- forward reach of each quantity (signal path, constants frozen) ----------
    s = 3000
    p1 = prices.copy()
    raw_idx = int(p1.index[p1["date"] == dates.iloc[s]][0])
    p1.loc[raw_idx, "price"] *= 1.10
    probe = pipeline_from_prices(p1, CONFIG, te, constants=constants, thresholds=thresholds)
    quantities = {
        "x_t (composite)": (df["signal"].to_numpy(), probe["df"]["signal"].to_numpy()),
        "sigma_t (20-day volatility)": (df["rolling_volatility"].to_numpy(), probe["df"]["rolling_volatility"].to_numpy()),
        "volatility score (5-day smoothed z of sigma)": (base["scores"]["volatility"], probe["scores"]["volatility"]),
        "kappa_t": (base["scores"]["features"]["kappa"].to_numpy(), probe["scores"]["features"]["kappa"].to_numpy()),
        "MMD_t (3-day smoothed)": (base["scores"]["features"]["MMD"].to_numpy(), probe["scores"]["features"]["MMD"].to_numpy()),
        "instability score s(t)": (base["scores"]["instability"], probe["scores"]["instability"]),
    }
    reach_rows = []
    for name, (a, b) in quantities.items():
        moved = np.flatnonzero(np.abs(a - b) > 0)
        reach_rows.append({"quantity": name, "perturbed_close_index": s, "earliest_index_changed": int(moved.min()),
                           "forward_reach_days": int(s - moved.min())})
    reach = pd.DataFrame(reach_rows)
    reach.to_csv(OUT / "forward_reach_single_close_perturbation.csv", index=False)
    r_s = int(reach.loc[reach["quantity"] == "instability score s(t)", "forward_reach_days"].iloc[0])
    r_v = int(reach.loc[reach["quantity"].str.startswith("volatility score"), "forward_reach_days"].iloc[0])
    W = CONFIG["detectors"]["default_hybrid_window"]
    semantics = pd.DataFrame(
        [
            ["r_t", "log(P_t / P_{t-1})", "P_{t-1}..P_t", 0],
            ["sigma_t", "sd of r_{t-19..t}", "P_{t-20}..P_t", 0],
            ["z_t", "(r_t - mean)/sd over r_{t-49..t}", "P_{t-50}..P_t", 0],
            ["x_t", "0.5 r_t + 0.3 sigma_t + 0.2 z_t", "P_{t-50}..P_t", 0],
            ["kappa_t", "1 - AR(1) slope through the origin on x_{t-35..t}", "P_{t-85}..P_t", 0],
            ["MMD_t", "x_{t-30..t-1} against x_{t..t+29}, then 3-day trailing mean", "P_{t-82}..P_{t+29}", w - 1],
            ["s(t)", "5-day trailing mean of 0.5 z_MMD + 0.5 |z_kappa|", "P_{t-86}..P_{t+29}", r_s],
            ["i(t)", "s(t) above its calibration 90th percentile", "as s(t)", r_s],
            ["v(t)", "5-day trailing mean of standardized sigma above its calibration 90th percentile", "P_{t-24}..P_t", r_v],
            ["h_W(t) symmetric (submitted)", "v(t) and any i(tau), tau in [t-10, t+10]", f"up to P_(t+{r_s + W})", r_s + W],
            ["trailing h_W(t)", "v(t) and any i(tau), tau in [t-10, t]", f"up to P_(t+{r_s})", r_s],
            ["i_lag(t) = i(t-30)", "instability indicator issued 30 days late", f"up to P_(t{r_s - w:+d})", r_s - w],
            ["fully causal h_W(t)", "v(t) and any i_lag(tau), tau in [t-10, t]", "up to P_t", max(r_v, r_s - w)],
        ],
        columns=["quantity", "definition", "price closes read (signal path)", "forward_reach_days"],
    )
    semantics.to_csv(OUT / "timing_semantics.csv", index=False)

    # ---- issue dates -------------------------------------------------------------
    def idx(d: str) -> int:
        return int(np.searchsorted(dates.to_numpy(), np.datetime64(pd.Timestamp(d))))

    causal = base["alarms"]["fully_causal_confirmation"]
    post_alarm_days = np.flatnonzero(causal[te:]) + te
    post_dates = {
        "first post-cutoff day": te, "cutoff+10": te + 10, "day before latest calibration close": horizon - 1,
        "latest calibration close": horizon, "operational start": start, "operational start+10": start + 10, "calm 2017-06-15": idx("2017-06-15"), "E09 window start": idx("2017-11-08"),
        "E09 2018-02-05": idx("2018-02-05"), "E09+8": idx("2018-02-15"), "E10 2018-12-24": idx("2018-12-24"),
        "calm 2019-07-15": idx("2019-07-15"), "E11-20": idx("2020-02-14"), "E11 2020-03-16": idx("2020-03-16"),
        "E12 2022-06-13": idx("2022-06-13"), "E13 2023-03-13": idx("2023-03-13"),
        "first post-cutoff causal alarm": int(post_alarm_days[0]), "middle post-cutoff causal alarm": int(post_alarm_days[len(post_alarm_days) // 2]),
        "last post-cutoff causal alarm": int(post_alarm_days[-1]),
    }
    calibration_dates = {
        "calm 2005-06-15": idx("2005-06-15"), "E05 2008-09-29": idx("2008-09-29"), "E05+33": idx("2008-11-14"),
        "E07 2011-08-08": idx("2011-08-08"), "E08 2015-08-24": idx("2015-08-24"), "last calibration day": te - 1,
    }
    retro, trail = base["alarms"]["retrospective_confirmation"], base["alarms"]["trailing_confirmation"]
    future_only = np.flatnonzero((retro == 1) & (trail == 0))
    control_dates = {f"symmetric alarm confirmed only by days after t #{k}": int(future_only[j]) for k, j in
                     enumerate([0, len(future_only) // 2, len(future_only) - 1])}

    rng = np.random.default_rng(CONFIG["seed"])
    rows = []
    plan = [("frozen", calibration_dates | post_dates | control_dates), ("refit", post_dates | calibration_dates)]
    for mode, chosen in plan:
        for label, t in chosen.items():
            for kind in ["noise", "crash", "spike", "flat"]:
                pert = perturbed(prices, dates.iloc[t], kind, rng)
                kw = {"constants": constants, "thresholds": thresholds} if mode == "frozen" else {}
                out = pipeline_from_prices(pert, CONFIG, te, **kw)
                moved_constants = [k for k, v in out["scores"]["calibration_constants"].items() if v != constants[k]]
                moved_constants += [f"threshold_{k}" for k, v in out["alarms"]["thresholds"].items() if v != thresholds[k]]
                row = {"mode": mode, "label": label, "index": t, "date": dates.iloc[t].date().isoformat(),
                       "period": "post_calibration" if t >= te else "calibration", "perturbation": kind,
                       "constants_changed": ";".join(moved_constants) if mode == "refit" else "frozen"}
                for rule in RULES:
                    a, b = base["alarms"][rule], out["alarms"][rule]
                    row[f"{rule}_base"] = int(a[t])
                    row[f"{rule}_perturbed"] = int(b[t])
                    row[f"{rule}_alarm_unchanged"] = bool(a[t] == b[t])
                    row[f"{rule}_history_unchanged"] = bool(np.array_equal(a[: t + 1], b[: t + 1]))
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "perturbation_test_results.csv", index=False)

    frozen = res[res["mode"] == "frozen"]
    refit_post = res[(res["mode"] == "refit") & (res["period"] == "post_calibration")]
    refit_cal = res[(res["mode"] == "refit") & (res["period"] == "calibration")]
    controls = frozen[frozen["label"].str.startswith("symmetric alarm")]
    gap = slice(te, horizon)
    summary = {
        "train_end_index": te,
        "last_calibration_date": dates.iloc[te - 1],
        "instability_constants_read_data_through": dates.iloc[horizon],
        "volatility_alarms_between_cutoff_and_that_date": int(base["alarms"]["volatility"][gap].sum()),
        "frozen_signal_path_fully_causal_alarm_unchanged_in_all_trials": bool(frozen["fully_causal_confirmation_alarm_unchanged"].all()),
        "frozen_signal_path_fully_causal_history_unchanged_in_all_trials": bool(frozen["fully_causal_confirmation_history_unchanged"].all()),
        "frozen_trials": int(len(frozen)),
        "refit_post_calibration_fully_causal_alarm_unchanged_in_all_trials": bool(refit_post["fully_causal_confirmation_alarm_unchanged"].all()),
        "refit_post_calibration_trials": int(len(refit_post)),
        "refit_post_calibration_history_unchanged_from_first_issuable_day": bool(
            refit_post.loc[refit_post["index"] >= horizon, "fully_causal_confirmation_history_unchanged"].all()),
        "operational_start": dates.iloc[start],
        "refit_trials_from_operational_start": int((refit_post["index"] >= start).sum()),
        "refit_from_operational_start_alarm_and_history_unchanged": bool(
            refit_post.loc[refit_post["index"] >= start, ["fully_causal_confirmation_alarm_unchanged",
                                                          "fully_causal_confirmation_history_unchanged",
                                                          "volatility_alarm_unchanged"]].all().all()),
        "refit_post_calibration_volatility_alarm_unchanged_in_all_trials": bool(refit_post["volatility_alarm_unchanged"].all()),
        "refit_calibration_period_history_changed_in_some_trial": bool((~refit_cal["fully_causal_confirmation_history_unchanged"]).any()),
        "refit_calibration_period_constants_changed_in_all_trials": bool((refit_cal["constants_changed"] != "").all()),
        "negative_control_symmetric_rule_alarm_changed_in_some_trial": bool((~controls["retrospective_confirmation_alarm_unchanged"]).any()),
    }
    write_json(OUT / "causality_summary.json", summary)
    print(reach.to_string(index=False))
    print(semantics[["quantity", "forward_reach_days"]].to_string(index=False))
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
