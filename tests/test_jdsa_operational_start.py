"""The prospective operational evaluation starts strictly after every raw close that any
calibration quantity reads, and its alarms read no price after their issue date."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.multi_asset_data import _read_price_csv
from src.multi_asset_evaluation import _load_processed_asset
from src.paper_protocol import load_config
from src.revision_analysis import (
    _perturb_from,
    calibration_quantities,
    calibration_train_end,
    derive_operational_start,
    pipeline_from_prices,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")


@pytest.fixture(scope="module")
def setup():
    prices = _read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]).reset_index(drop=True)
    te = calibration_train_end(len(_load_processed_asset("SPY")), CONFIG)
    base = pipeline_from_prices(prices, CONFIG, te)
    return prices, te, base, derive_operational_start(prices, CONFIG, te, seed=CONFIG["seed"])


def test_start_is_derived_and_strictly_after_the_latest_calibration_close(setup):
    _, te, base, op = setup
    assert op["operational_start_index"] == op["latest_calibration_close_index"] + 1
    assert op["latest_calibration_close_index"] >= te - 1
    assert op["check_all_closes_from_start_perturbed_leaves_calibration_identical"]
    assert op["check_single_close_before_start_moves_calibration"]
    assert op["operational_start_date"] > op["latest_calibration_close_date"] > op["last_calibration_date"] - pd.Timedelta(days=1)
    # Regression expectation for the archived data: the MMD second sample reaches 29 days past the cutoff.
    assert op["latest_calibration_close_date"].date().isoformat() == "2016-12-20"
    assert op["operational_start_date"].date().isoformat() == "2016-12-21"


@pytest.mark.parametrize("seed", [1, 2])
def test_no_close_on_or_after_the_start_enters_any_calibration_quantity(setup, seed):
    prices, te, base, op = setup
    reference = calibration_quantities(base)
    after = pipeline_from_prices(_perturb_from(prices, op["operational_start_date"], seed), CONFIG, te)
    assert calibration_quantities(after) == reference
    before = pipeline_from_prices(_perturb_from(prices, op["latest_calibration_close_date"], seed), CONFIG, te)
    assert calibration_quantities(before) != reference


def test_prospective_alarms_read_nothing_after_their_issue_date(setup):
    prices, te, base, op = setup
    dates = base["df"]["date"]
    start = op["operational_start_index"]
    causal = base["alarms"]["fully_causal_confirmation"]
    picks = [start, start + 5, int(np.flatnonzero(causal[start:])[0] + start)]
    for t in picks:
        pert = prices.copy()
        sel = (pert["date"] > dates.iloc[t]).to_numpy()
        pert.loc[sel, "price"] = pert.loc[sel, "price"].to_numpy() * np.exp(np.random.default_rng(t).normal(0, 0.05, int(sel.sum())))
        out = pipeline_from_prices(pert, CONFIG, te)
        for rule in ["volatility", "fully_causal_confirmation"]:
            assert np.array_equal(out["alarms"][rule][: t + 1], base["alarms"][rule][: t + 1]), (rule, t)
