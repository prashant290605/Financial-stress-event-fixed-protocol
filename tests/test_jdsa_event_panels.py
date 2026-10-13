"""The versioned event panels: the submitted one reproduces the stored event files, and the
revision panel differs from it by the documented E03 correction and by the wording of the
inclusion-rule field."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.event_panels import PANELS, REVISION_PANEL, asset_event_table, load_seed, spy_event_table
from src.multi_asset_evaluation import _load_processed_asset
from src.paper_protocol import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")
TICKERS = ["SPY", "QQQ", "IWM", "HYG", "TLT"]


def _same(a: pd.DataFrame, b: pd.DataFrame, numeric: list[str]) -> None:
    assert list(a.columns[: len(b.columns)]) == list(b.columns)
    a = a[b.columns].reset_index(drop=True)
    b = b.reset_index(drop=True)
    assert (a.drop(columns=numeric).astype(str).values == b.drop(columns=numeric).astype(str).values).all()
    assert np.allclose(a[numeric].to_numpy(float), b[numeric].to_numpy(float), rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def spy():
    return _load_processed_asset("SPY")


def test_submitted_panel_reproduces_the_stored_spy_event_table(spy):
    _same(spy_event_table("submitted", spy, CONFIG), pd.read_csv(ROOT / "data" / "events_spy.csv"), ["severity_score"])


def test_submitted_panel_reproduces_the_stored_per_asset_event_table(spy):
    stored = pd.read_csv(ROOT / "data" / "events_multi_asset.csv")
    table = spy_event_table("submitted", spy, CONFIG)
    for ticker in TICKERS:
        mapped = asset_event_table(table, ticker, _load_processed_asset(ticker), CONFIG)
        _same(mapped, stored[stored["ticker"] == ticker], ["severity_score_asset_specific"])


def test_revision_panel_changes_the_e03_entry_and_the_inclusion_wording():
    old, new = load_seed("submitted").set_index("event_id"), load_seed(REVISION_PANEL).set_index("event_id")
    assert list(old.index) == list(new.index) == [f"E{i:02d}" for i in range(1, 14)]
    for column in ["event_name", "calendar_date", "event_type", "inclusion_rule", "source_reference"]:
        changed = [e for e in old.index if old.loc[e, column] != new.loc[e, column]]
        expected = [] if column == "event_type" else list(old.index) if column == "inclusion_rule" else ["E03"]
        assert changed == expected, column
    assert old.loc["E03", "calendar_date"] == "2002-07-24"
    assert new.loc["E03", "calendar_date"] == "2002-07-22"
    assert "2002-07-21" in new.loc["E03", "source_reference"] and "02-13533" in new.loc["E03", "source_reference"]
    assert "correction" in new.loc["E03", "revision_note"]
    assert all("fixed before detector evaluation" not in r and "held fixed across the reported comparative evaluation" in r
               for r in new["inclusion_rule"])


def test_e03_reference_is_the_first_trading_day_after_the_sunday_filing(spy):
    dates = pd.DatetimeIndex(spy["date"])
    filing = pd.Timestamp("2002-07-21")
    assert filing.day_name() == "Sunday" and filing not in dates
    first_after = dates[dates > filing][0]
    table = spy_event_table(REVISION_PANEL, spy, CONFIG).set_index("event_id")
    assert table.loc["E03", "mapped_trading_date"] == first_after.date().isoformat() == "2002-07-22"


def test_unknown_panel_is_rejected():
    with pytest.raises(ValueError):
        load_seed("no_such_panel")
    assert set(PANELS) == {"submitted", REVISION_PANEL}
