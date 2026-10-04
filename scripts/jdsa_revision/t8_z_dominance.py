"""Task 8: how far the composite x_t is dominated by z_t, and where that stops.

Reviewer 1 further comment 3 and Reviewer 2 minor 2. The submitted composite is kept.
For each asset the detectors are recomputed with x_t replaced by z_t alone (weights
0, 0, 1, as in the Table 14 row) and every difference in alarms, detected events and
first-alarm delays is listed.

Outputs: results/jdsa_revision/t8_z_dominance/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, TICKERS, load_asset, outdir  # noqa: E402
from src.paper_protocol import detector_outputs, event_mask, event_metrics, point_metrics  # noqa: E402
from src.revision_analysis import event_spans, per_event_outcomes  # noqa: E402

OUT = outdir("t8_z_dominance")
W = CONFIG["spy"]["signal_weights"]
DETECTORS = {
    "confirmation (W=10)": f"hybrid_w{CONFIG['detectors']['default_hybrid_window']}",
    "instability score alone": "instability_score",
    "MMD alone": "mmd_only_confirmation",
    "|kappa| alone": "kappa_only_confirmation",
}


def main() -> None:
    summary, alarm_rows, event_rows, metric_rows = [], [], [], []
    for ticker in TICKERS:
        df, events = load_asset(ticker)
        dz = df.copy()
        dz["signal"] = dz["return_zscore"]
        terms = {"0.5 r_t": W["log_return"] * df["log_return"], "0.3 sigma_t": W["rolling_volatility"] * df["rolling_volatility"],
                 "0.2 z_t": W["return_zscore"] * df["return_zscore"]}
        sd_x = float(df["signal"].std())
        summary.append({
            "ticker": ticker, "corr_x_z": float(df["signal"].corr(df["return_zscore"])),
            "corr_x_r": float(df["signal"].corr(df["log_return"])), "corr_x_sigma": float(df["signal"].corr(df["rolling_volatility"])),
            **{f"sd {k}": float(v.std()) for k, v in terms.items()}, "sd x_t": sd_x,
            **{f"sd {k} / sd x_t": float(v.std()) / sd_x for k, v in terms.items()},
        })
        _, p0 = detector_outputs(df, CONFIG)
        _, p1 = detector_outputs(dz, CONFIG)
        spans = event_spans(df, events, HALF_WINDOW)
        y, raw_spans = event_mask(df, events, HALF_WINDOW)
        for label, key in DETECTORS.items():
            a, b = p0[key], p1[key]
            diff = np.flatnonzero(a != b)
            alarm_rows.append({"ticker": ticker, "detector": label, "alarm_days_composite": int(a.sum()), "alarm_days_z_only": int(b.sum()),
                               "differing_days": int(len(diff)), "composite_only_days": int(np.sum((a == 1) & (b == 0))),
                               "z_only_days": int(np.sum((a == 0) & (b == 1))),
                               "first_differing_date": df["date"].iloc[diff[0]].date().isoformat() if len(diff) else ""})
            ea, eb = per_event_outcomes(a, spans), per_event_outcomes(b, spans)
            for (_, ra), (_, rb) in zip(ea.iterrows(), eb.iterrows()):
                same_delay = (pd.isna(ra["first_delay"]) and pd.isna(rb["first_delay"])) or ra["first_delay"] == rb["first_delay"]
                if ra["detected"] != rb["detected"] or not same_delay or ra["alarm_days"] != rb["alarm_days"]:
                    event_rows.append({"ticker": ticker, "detector": label, "event_id": ra["event_id"],
                                       "detected_composite": bool(ra["detected"]), "detected_z_only": bool(rb["detected"]),
                                       "first_delay_composite": ra["first_delay"], "first_delay_z_only": rb["first_delay"],
                                       "window_alarm_days_composite": ra["alarm_days"], "window_alarm_days_z_only": rb["alarm_days"]})
            if ticker == "SPY":
                for sig, arr in [("composite x_t", a), ("z_t only", b)]:
                    pm, em = point_metrics(y, arr), event_metrics(arr, raw_spans)
                    metric_rows.append({"detector": label, "signal": sig, **{k: pm[k] for k in ["precision", "recall", "f1", "fpr", "tp", "fp", "fn"]},
                                        "detection_rate": em["event_coverage"], "mean_first_delay": em["event_delay"],
                                        "alignment_error": em["event_alignment_error"]})
    pd.DataFrame(summary).to_csv(OUT / "z_dominance_scale_summary.csv", index=False)
    alarms = pd.DataFrame(alarm_rows)
    alarms.to_csv(OUT / "z_dominance_alarm_differences.csv", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / "z_dominance_event_differences.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "spy_metrics_composite_vs_z_only.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary)[["ticker", "corr_x_z", "sd 0.5 r_t", "sd 0.3 sigma_t", "sd 0.2 z_t", "sd x_t"]].to_string(index=False))
    print(alarms.to_string(index=False))
    print(pd.DataFrame(event_rows).to_string(index=False))
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
