"""Task 7: what the submitted CUSUM baseline does on SPY.

Reviewer 1 minor 1 and Reviewer 2 major 5. No new CUSUM is implemented. The submitted
recursion is recomputed step by step, checked against ``detector_outputs``, and its
behaviour inside and after the calibration period is recorded.

Outputs: results/jdsa_revision/t7_cusum/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import CONFIG, HALF_WINDOW, load_asset, outdir, write_json  # noqa: E402
from src.paper_protocol import detector_outputs  # noqa: E402
from src.revision_analysis import alarm_episodes, cusum_components, event_spans, per_event_outcomes  # noqa: E402

OUT = outdir("t7_cusum")


def main() -> None:
    df, events = load_asset("SPY")
    cu = cusum_components(df, CONFIG)
    _, submitted = detector_outputs(df, CONFIG)
    assert np.array_equal(cu["alarms"], submitted["cusum"])
    te, alarms, dates = cu["train_end"], cu["alarms"], df["date"]
    n = len(alarms)

    episodes = alarm_episodes(alarms)
    ep = pd.DataFrame(
        [{"start_index": s, "end_index": e, "start_date": dates.iloc[s].date().isoformat(), "end_date": dates.iloc[e].date().isoformat(),
          "days": e - s + 1, "period": "calibration" if e < te else ("post_calibration" if s >= te else "spans cutoff")}
         for s, e in episodes]
    )
    ep.to_csv(OUT / "cusum_alarm_episodes.csv", index=False)
    longest = ep.loc[ep["days"].idxmax()]
    last_off = int(np.flatnonzero(alarms == 0).max())
    persistent_start = last_off + 1

    spans = event_spans(df, events, HALF_WINDOW)
    per = per_event_outcomes(alarms, spans).merge(events[["event_id", "event_name"]], on="event_id")
    per["alarm_on_at_window_start"] = [bool(alarms[lo]) for lo in per["lo"]]
    per.to_csv(OUT / "cusum_per_event.csv", index=False)
    detected = per[per["detected"]]

    z = cu["standardized_return"]
    side = np.where(cu["c_plus"] >= -cu["c_minus"], "C+", "C-")
    write_json(
        OUT / "cusum_diagnostic.json",
        {
            "statistic": "C_t = max(C+_t, -C-_t), C+_t = max(0, C+_{t-1} + z_t), C-_t = min(0, C-_{t-1} + z_t)",
            "standardized_return_constants": "mean and sd of log returns over the calibration period",
            "reference_value_k": 0.0,
            "reset_rule": "none; C+ and C- are only floored at zero and never restarted after an alarm",
            "smoothing": f"{CONFIG['features']['smooth_window']}-day trailing mean of the calibration-standardized C_t",
            "threshold_rule": f"{CONFIG['detectors']['cusum_threshold_quantile']:.2f} calibration quantile of the smoothed statistic",
            "threshold_value_standardized_units": cu["threshold"],
            "calibration_days": te,
            "calibration_alarm_days": int(alarms[:te].sum()),
            "calibration_alarm_fraction": float(alarms[:te].mean()),
            "post_calibration_days": n - te,
            "post_calibration_alarm_days": int(alarms[te:].sum()),
            "post_calibration_alarm_fraction": float(alarms[te:].mean()),
            "alarm_episodes_full_period": len(episodes),
            "alarm_episodes_calibration": int((ep["period"] == "calibration").sum()),
            "alarm_episodes_post_calibration_or_spanning": int((ep["period"] != "calibration").sum()),
            "longest_episode_days": int(longest["days"]),
            "longest_episode_start": longest["start_date"],
            "longest_episode_end": longest["end_date"],
            "persistent_alarm_state_first_day": dates.iloc[persistent_start],
            "persistent_alarm_state_continues_to_last_day": bool(alarms[persistent_start:].all()),
            "last_non_alarm_day": dates.iloc[last_off],
            "mean_standardized_return_calibration": float(z[:te].mean()),
            "mean_standardized_return_post_calibration": float(z[te:].mean()),
            "share_of_post_calibration_days_dominated_by_C_plus": float(np.mean(side[te:] == "C+")),
            "final_C_plus": float(cu["c_plus"][-1]),
            "final_minus_C_minus": float(-cu["c_minus"][-1]),
            "events_detected": int(len(detected)),
            "events_detected_ids": ",".join(detected["event_id"]),
            "detections_with_first_alarm_at_window_start": int(detected["first_alarm_at_window_start"].sum()),
            "detections_already_alarming_when_window_opens": int(detected["alarm_on_at_window_start"].sum()),
            "full_period_alarm_fraction": float(alarms.mean()),
        },
    )
    print((OUT / "cusum_diagnostic.json").read_text())
    print(ep.to_string(index=False))


if __name__ == "__main__":
    main()
