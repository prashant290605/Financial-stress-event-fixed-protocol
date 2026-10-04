"""Which events each asset can observe.

An event is observable for an asset when its documented reference date falls inside
the asset's processed sample. Events dated before an asset's first processed day are
mapped by ``nearest_trading_date`` to that first day, which produces a synthetic
window at the start of the series; such an event is not observable for that asset and
must not enter an event-level denominator, a day-level mask, or a cost as a missed
event.
"""
from __future__ import annotations

import pandas as pd


def observable_event_flags(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Boolean flag per event row: True when the reference date lies inside the sample."""
    first, last = pd.Timestamp(df["date"].min()), pd.Timestamp(df["date"].max())
    calendar = pd.to_datetime(events["calendar_date"])
    return pd.Series(((calendar >= first) & (calendar <= last)).to_numpy(), index=events.index, name="observable")


def observable_events(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return only the events whose reference date lies inside the asset's sample."""
    return events.loc[observable_event_flags(df, events)].copy()


def unobservable_events(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return the events whose reference date lies outside the asset's sample."""
    return events.loc[~observable_event_flags(df, events)].copy()
