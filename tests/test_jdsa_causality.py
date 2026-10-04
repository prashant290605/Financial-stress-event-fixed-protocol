"""Perturbation tests of the timestamp semantics of the confirmation variants.

For an issue date t every raw adjusted close dated after t is changed and the pipeline is
recomputed from the raw file with the calibration cutoff index held fixed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.multi_asset_data import _read_price_csv
from src.multi_asset_evaluation import _load_processed_asset
from src.paper_protocol import load_config
from src.revision_analysis import calibration_train_end, pipeline_from_prices

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")
W = CONFIG["features"]["mmd_window"]


def _perturb(prices: pd.DataFrame, date: pd.Timestamp, kind: str) -> pd.DataFrame:
    p = prices.copy()
    after = (p["date"] > date).to_numpy()
    values = p.loc[after, "price"].to_numpy(float)
    if kind == "noise":
        values = values * np.exp(np.random.default_rng(1).normal(0.0, 0.05, len(values)))
    elif kind == "flat":
        values = np.full(len(values), values[0])
    elif kind == "crash":
        values = values * np.exp(-0.5 * np.minimum(np.arange(1, len(values) + 1), 20) / 20)
    elif kind == "spike":
        values[0] *= 3.0
    p.loc[after, "price"] = values
    return p


@pytest.fixture(scope="module")
def base():
    prices = _read_price_csv(ROOT / "data" / "raw_market" / "spy.csv", CONFIG["spy"]["price_column"]).reset_index(drop=True)
    te = calibration_train_end(len(_load_processed_asset("SPY")), CONFIG)
    out = pipeline_from_prices(prices, CONFIG, te)
    return prices, te, out


def _index(base, date: str) -> int:
    dates = base[2]["df"]["date"].to_numpy()
    return int(np.searchsorted(dates, np.datetime64(pd.Timestamp(date))))


def test_fully_causal_signal_path_reads_nothing_after_t(base):
    prices, te, out = base
    frozen = {"constants": out["scores"]["calibration_constants"], "thresholds": out["alarms"]["thresholds"]}
    causal = out["alarms"]["fully_causal_confirmation"]
    first_post_alarm = int(np.flatnonzero(causal[te:])[0] + te)
    for t in [_index(base, "2005-06-15"), _index(base, "2008-09-29"), _index(base, "2020-03-16"), first_post_alarm]:
        for kind in ["noise", "flat"]:
            pert = pipeline_from_prices(_perturb(prices, out["df"]["date"].iloc[t], kind), CONFIG, te, **frozen)
            assert np.array_equal(pert["alarms"]["fully_causal_confirmation"][: t + 1], causal[: t + 1]), (t, kind)


def test_fully_causal_rule_is_causal_end_to_end_after_the_cutoff(base):
    prices, te, out = base
    horizon = te - 1 + W - 1
    causal = out["alarms"]["fully_causal_confirmation"]
    for t in [te, horizon - 1, horizon, _index(base, "2020-03-16")]:
        for kind in ["noise", "spike"]:
            pert = pipeline_from_prices(_perturb(prices, out["df"]["date"].iloc[t], kind), CONFIG, te)
            new = pert["alarms"]["fully_causal_confirmation"]
            assert new[t] == causal[t], (t, kind)
            if t >= horizon:
                assert np.array_equal(new[: t + 1], causal[: t + 1]), (t, kind)


def test_volatility_gate_is_closed_until_the_instability_constants_are_complete(base):
    _, te, out = base
    horizon = te - 1 + W - 1
    assert out["alarms"]["volatility"][te:horizon].sum() == 0


def test_negative_control_symmetric_rule_reads_days_after_t(base):
    prices, te, out = base
    frozen = {"constants": out["scores"]["calibration_constants"], "thresholds": out["alarms"]["thresholds"]}
    retro, trail = out["alarms"]["retrospective_confirmation"], out["alarms"]["trailing_confirmation"]
    t = int(np.flatnonzero((retro == 1) & (trail == 0))[0])
    pert = pipeline_from_prices(_perturb(prices, out["df"]["date"].iloc[t], "crash"), CONFIG, te, **frozen)
    assert pert["alarms"]["retrospective_confirmation"][t] != retro[t]
    assert np.array_equal(pert["alarms"]["fully_causal_confirmation"][: t + 1], out["alarms"]["fully_causal_confirmation"][: t + 1])


def test_forward_reach_of_mmd_and_kappa(base):
    prices, te, out = base
    s = 3000
    p = prices.copy()
    p.loc[int(p.index[p["date"] == out["df"]["date"].iloc[s]][0]), "price"] *= 1.10
    frozen = {"constants": out["scores"]["calibration_constants"], "thresholds": out["alarms"]["thresholds"]}
    pert = pipeline_from_prices(p, CONFIG, te, **frozen)
    mmd_moved = np.flatnonzero(pert["scores"]["features"]["MMD"].to_numpy() != out["scores"]["features"]["MMD"].to_numpy())
    kappa_moved = np.flatnonzero(pert["scores"]["features"]["kappa"].to_numpy() != out["scores"]["features"]["kappa"].to_numpy())
    assert s - mmd_moved.min() == W - 1
    assert kappa_moved.min() == s
