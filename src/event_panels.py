"""Versioned event panels.

``submitted`` is the panel behind every submitted result (``data/events_spy_seed.csv``).
``jdsa_revision_v2`` is the same panel with the E03 reference date corrected
(``data/jdsa_revision/events_spy_seed_v2.csv``). The builders return tables in memory
and write nothing, so the submitted event files are never overwritten. With the
``submitted`` panel they reproduce ``data/events_spy.csv`` and
``data/events_multi_asset.csv``, which the tests check.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.multi_asset_data import map_events_for_asset
from src.paper_protocol import ROOT, nearest_trading_date

PANELS = {
    "submitted": ROOT / "data" / "events_spy_seed.csv",
    "jdsa_revision_v2": ROOT / "data" / "jdsa_revision" / "events_spy_seed_v2.csv",
}
REVISION_PANEL = "jdsa_revision_v2"
SPY_COLUMNS = ["event_id", "event_name", "calendar_date", "mapped_trading_date", "event_type", "inclusion_rule",
               "source_reference", "severity_score", "severity_label", "severity_rule"]


def load_seed(panel: str) -> pd.DataFrame:
    if panel not in PANELS:
        raise ValueError(f"unknown event panel {panel!r}; expected one of {sorted(PANELS)}")
    return pd.read_csv(PANELS[panel], keep_default_na=False)


def spy_event_table(panel: str, spy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """SPY event table with drawdown severity terciles, as ``build_event_table`` computes it."""
    seed = load_seed(panel)
    half = config["events"]["severity_half_window"]
    rows = []
    for _, row in seed.iterrows():
        cal = pd.Timestamp(row["calendar_date"])
        mapped = nearest_trading_date(spy["date"], cal)
        center = int(spy.index[spy["date"] == mapped][0])
        lo, hi = max(0, center - half), min(len(spy), center + half + 1)
        prices = spy["price"].iloc[lo:hi].to_numpy()
        drawdown = prices / np.maximum.accumulate(prices) - 1.0
        rows.append({
            "event_id": row["event_id"], "event_name": row["event_name"], "calendar_date": cal.date().isoformat(),
            "mapped_trading_date": mapped.date().isoformat(), "event_type": row["event_type"],
            "inclusion_rule": row["inclusion_rule"], "source_reference": row["source_reference"],
            "severity_score": float(abs(np.min(drawdown))),
        })
    out = pd.DataFrame(rows)
    q1, q2 = out["severity_score"].quantile([1 / 3, 2 / 3]).to_numpy()
    out["severity_label"] = np.where(out["severity_score"] >= q2, "critical",
                                     np.where(out["severity_score"] >= q1, "high", "medium"))
    out["severity_rule"] = config["events"]["severity_rule"]
    out = out[SPY_COLUMNS]
    if "revision_note" in seed.columns:
        out["revision_note"] = seed["revision_note"].to_numpy()
    return out


def asset_event_table(spy_events: pd.DataFrame, ticker: str, asset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Per-asset mapping of a SPY event table, as ``prepare_multi_asset_data`` computes it."""
    return map_events_for_asset(asset, ticker, spy_events, config)
