from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pointwise() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "cost" / "pointwise_cost_by_asset.csv")


def event_aware() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "cost" / "event_aware_cost_by_asset.csv")


def test_costs_are_nonnegative():
    assert (pointwise()["cost"] >= 0).all()
    assert (event_aware()["cost"] >= 0).all()


def test_preferred_method_is_min_cost_or_tie():
    for df, group_cols in [
        (pointwise(), ["ticker", "ratio_fp_to_fn"]),
        (event_aware(), ["ticker", "ratio_false_alarm_to_missed_event", "delay_cost"]),
    ]:
        for _, group in df.groupby(group_cols):
            min_cost = group["cost"].min()
            winners = group.loc[group["cost"] == min_cost, "method"].tolist()
            expected = "tie" if len(winners) != 1 else winners[0]
            assert set(group["preferred_method_at_ratio"]) == {expected}


def test_missed_events_match_event_level_metrics():
    events = pd.read_csv(ROOT / "results" / "event_level" / "event_level_metrics.csv")
    costs = event_aware()
    for (ticker, method), group in costs.groupby(["ticker", "method"]):
        expected = int(
            (~events.loc[(events["ticker"] == ticker) & (events["method"] == method), "detected_event"].astype(bool)).sum()
        )
        assert set(group["missed_events"]) == {expected}


def test_false_alarm_days_are_outside_event_windows():
    point = pointwise()
    costs = event_aware()
    fp_lookup = point.drop_duplicates(["ticker", "method"]).set_index(["ticker", "method"])["fp"]
    for (ticker, method), group in costs.groupby(["ticker", "method"]):
        assert set(group["false_alarm_days"]) == {int(fp_lookup.loc[(ticker, method)])}
