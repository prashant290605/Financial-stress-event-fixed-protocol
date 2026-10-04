"""Task 9: recompute the dispersion and MMD quantities behind Section 4.4.1 and Appendix B.

Reviewer 1 further comment 4. Every number in that paragraph is recomputed from the
archived data and compared with the manuscript value. No new explanation is added.

Dispersion ratio (Appendix B): sd inside the +/-60 window over sd in the 120 trading
days before the window, for raw 20-day volatility and for x_t; only events with a full
120-day preceding window are used for the reference mean.

Outputs: results/jdsa_revision/t9_mechanism/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, load_asset, outdir, write_json  # noqa: E402
from src.revision_analysis import component_scores, detector_alarms, dispersion_ratio, event_spans  # noqa: E402

OUT = outdir("t9_mechanism")
SEV_HALF = CONFIG["events"]["severity_half_window"]


def peak_to_trough(prices: np.ndarray, lo: int, hi: int) -> int:
    """Trading days from the running peak to the deepest drawdown inside [lo, hi]."""
    p = prices[lo : hi + 1]
    running_max = np.maximum.accumulate(p)
    trough = int(np.argmin(p / running_max))
    peak = int(np.argmax(p[: trough + 1]))
    return trough - peak


def main() -> None:
    df, events = load_asset("SPY")
    scores = component_scores(df, CONFIG)
    alarms = detector_alarms(scores, CONFIG)
    spans = event_spans(df, events, HALF_WINDOW)
    vol_raw = df["rolling_volatility"].to_numpy(float)
    x = df["signal"].to_numpy(float)
    price = df["price"].to_numpy(float)
    r = df["log_return"].to_numpy(float)
    rows = []
    for (_, s), (_, ev) in zip(spans.iterrows(), events.iterrows()):
        w = slice(s["lo"], s["hi"] + 1)
        c = int(s["center"])
        slo, shi = max(0, c - SEV_HALF), min(len(df) - 1, c + SEV_HALF)
        rows.append({
            "event_id": ev["event_id"], "event_name": ev["event_name"],
            "volatility_detected": bool(alarms["volatility"][w].any()),
            "confirmation_detected": bool(alarms["retrospective_confirmation"][w].any()),
            "window_days": int(s["hi"] - s["lo"] + 1),
            "full_120_day_pre_window": bool(s["lo"] - 120 >= 0),
            "dispersion_ratio_raw_volatility": dispersion_ratio(vol_raw, s["lo"], s["hi"]),
            "dispersion_ratio_composite_x": dispersion_ratio(x, s["lo"], s["hi"]),
            "volatility_alarm_days": int(alarms["volatility"][w].sum()),
            "instability_alarm_days": int(alarms["instability"][w].sum()),
            "peak_smoothed_z_mmd": float(np.nanmax(scores["mmd_z_smoothed"][w])),
            "severity": float(ev["severity_score_asset_specific"]),
            "peak_to_trough_days_severity_window": peak_to_trough(price, slo, shi),
            "peak_to_trough_days_event_window": peak_to_trough(price, int(s["lo"]), int(s["hi"])),
            "worst_day_log_return_pct_severity_window": float(-100 * r[slo : shi + 1].min()),
            "worst_day_log_return_pct_event_window": float(-100 * r[w].min()),
        })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "mechanism_per_event_spy.csv", index=False)

    conf = per[per["confirmation_detected"]]
    missed = per[~per["confirmation_detected"]]
    ref = conf[conf["full_120_day_pre_window"]]
    e07 = per.set_index("event_id").loc["E07"]
    e12 = per.set_index("event_id").loc["E12"]
    recomputed = {
        "reference events (confirmed, full pre-window)": ",".join(ref["event_id"]),
        "reference mean ratio raw volatility": ref["dispersion_ratio_raw_volatility"].mean(),
        "reference mean ratio x_t": ref["dispersion_ratio_composite_x"].mean(),
        "E07 ratio raw volatility": e07["dispersion_ratio_raw_volatility"],
        "E07 ratio x_t": e07["dispersion_ratio_composite_x"],
        "E12 ratio raw volatility": e12["dispersion_ratio_raw_volatility"],
        "E12 ratio x_t": e12["dispersion_ratio_composite_x"],
        "confirmed events with x_t ratio below E07": ",".join(ref.loc[ref["dispersion_ratio_composite_x"] < e07["dispersion_ratio_composite_x"], "event_id"]),
        "confirmed x_t ratio min": ref["dispersion_ratio_composite_x"].min(),
        "confirmed x_t ratio max": ref["dispersion_ratio_composite_x"].max(),
        "event with lowest x_t ratio (all with pre-window)": per.loc[per["dispersion_ratio_composite_x"].idxmin(), "event_id"],
        "E07 instability alarm days": int(e07["instability_alarm_days"]),
        "E12 instability alarm days": int(e12["instability_alarm_days"]),
        "confirmed instability alarm days min": int(conf["instability_alarm_days"].min()),
        "confirmed instability alarm days max": int(conf["instability_alarm_days"].max()),
        "confirmed instability alarm days mean": conf["instability_alarm_days"].mean(),
        "E07 volatility alarm days": int(e07["volatility_alarm_days"]),
        "E12 volatility alarm days": int(e12["volatility_alarm_days"]),
        "E07 peak z_MMD": e07["peak_smoothed_z_mmd"],
        "E12 peak z_MMD": e12["peak_smoothed_z_mmd"],
        "confirmed mean peak z_MMD": conf["peak_smoothed_z_mmd"].mean(),
        "missed (by confirmation) mean severity": missed["severity"].mean(),
        "confirmed mean severity": conf["severity"].mean(),
        "missed mean peak-to-trough days, severity window": missed["peak_to_trough_days_severity_window"].mean(),
        "confirmed mean peak-to-trough days, severity window": conf["peak_to_trough_days_severity_window"].mean(),
        "missed mean peak-to-trough days, event window": missed["peak_to_trough_days_event_window"].mean(),
        "confirmed mean peak-to-trough days, event window": conf["peak_to_trough_days_event_window"].mean(),
        "E07 severity": e07["severity"],
        "E07 worst day (log return, per cent), severity window": e07["worst_day_log_return_pct_severity_window"],
        "confirmed events with a smaller worst day than E07, severity window":
            int((conf["worst_day_log_return_pct_severity_window"] < e07["worst_day_log_return_pct_severity_window"]).sum()),
    }
    manuscript = {
        "reference mean ratio raw volatility": 3.03, "reference mean ratio x_t": 1.29, "E07 ratio raw volatility": 3.26,
        "E07 ratio x_t": 1.24, "E12 ratio raw volatility": 1.11, "E12 ratio x_t": 0.85, "E07 instability alarm days": 1,
        "E12 instability alarm days": 2, "confirmed instability alarm days min": 10, "confirmed instability alarm days max": 49,
        "confirmed instability alarm days mean": 28.0, "E07 volatility alarm days": 49, "E12 volatility alarm days": 40,
        "E07 peak z_MMD": 0.96, "E12 peak z_MMD": 1.84, "confirmed mean peak z_MMD": 3.38,
        "missed (by confirmation) mean severity": 0.110, "confirmed mean severity": 0.196,
        "missed mean peak-to-trough days, severity window": 13.2, "confirmed mean peak-to-trough days, severity window": 22.5,
        "E07 severity": 0.166, "E07 worst day (log return, per cent), severity window": 6.73,
        "confirmed events with a smaller worst day than E07, severity window": 6,
    }
    rows = []
    for k, v in recomputed.items():
        m = manuscript.get(k)
        status = ""
        if m is not None and isinstance(v, (int, float, np.integer, np.floating)):
            digits = len(str(m).split(".")[1]) if "." in str(m) else 0
            status = "matches" if round(float(v), digits) == round(m, digits) else "DIFFERS"
        rows.append({"quantity": k, "recomputed": v, "manuscript": m, "status": status})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "mechanism_manuscript_comparison.csv", index=False)
    write_json(OUT / "mechanism_summary.json", {k: (v.item() if hasattr(v, "item") else v) for k, v in recomputed.items()})
    pd.set_option("display.width", 220)
    print(per[["event_id", "confirmation_detected", "dispersion_ratio_raw_volatility", "dispersion_ratio_composite_x",
               "instability_alarm_days", "peak_smoothed_z_mmd", "peak_to_trough_days_severity_window", "peak_to_trough_days_event_window",
               "worst_day_log_return_pct_severity_window"]].round(3).to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
