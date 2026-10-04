"""The revision code reproduces the submitted detectors and the submitted Table 16 and Appendix B values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.multi_asset_evaluation import _load_asset_events, _load_processed_asset
from src.paper_protocol import detector_outputs, event_mask, event_metrics, load_config, point_metrics
from src.revision_analysis import component_scores, cusum_components, detector_alarms, dispersion_ratio, event_spans

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "paper_experiment.json")


@pytest.fixture(scope="module")
def spy():
    df, events = _load_processed_asset("SPY"), pd.read_csv(ROOT / "data" / "events_spy.csv")
    scores = component_scores(df, CONFIG)
    return df, events, scores, detector_alarms(scores, CONFIG)


def test_cusum_components_reproduce_the_submitted_cusum(spy):
    df, _, _, _ = spy
    _, submitted = detector_outputs(df, CONFIG)
    assert np.array_equal(cusum_components(df, CONFIG)["alarms"], submitted["cusum"])


@pytest.mark.parametrize("rule,variant", [("retrospective_confirmation", "symmetric_w10"),
                                          ("trailing_confirmation", "trailing_w10"),
                                          ("fully_causal_confirmation", "fully_causal_w10")])
def test_confirmation_variants_reproduce_table_16(spy, rule, variant):
    df, events, _, alarms = spy
    committed = pd.read_csv(ROOT / "results" / "ablations" / "table_causal_confirmation.csv").set_index("variant").loc[variant]
    y, spans = event_mask(df, events, CONFIG["events"]["half_window"])
    pm, em = point_metrics(y, alarms[rule]), event_metrics(alarms[rule], spans)
    assert (pm["tp"], pm["fp"]) == (int(committed["tp"]), int(committed["fp"]))
    assert sum(alarms[rule][lo: hi + 1].any() for _, lo, hi in spans) == int(committed["events_detected"])
    assert em["event_delay"] == pytest.approx(committed["event_delay"], abs=1e-12)


def test_appendix_b_dispersion_ratios(spy):
    df, events, _, alarms = spy
    spans = event_spans(df, events, CONFIG["events"]["half_window"]).set_index("event_id")
    vol, x = df["rolling_volatility"].to_numpy(float), df["signal"].to_numpy(float)
    ratios = {e: (dispersion_ratio(vol, s.lo, s.hi), dispersion_ratio(x, s.lo, s.hi)) for e, s in spans.iterrows()}
    confirmed = [e for e, s in spans.iterrows() if alarms["retrospective_confirmation"][s.lo: s.hi + 1].any() and s.lo >= 120]
    assert len(confirmed) == 7
    assert round(np.mean([ratios[e][0] for e in confirmed]), 2) == 3.03
    assert round(np.mean([ratios[e][1] for e in confirmed]), 2) == 1.29
    assert tuple(round(v, 2) for v in ratios["E07"]) == (3.26, 1.24)
    assert tuple(round(v, 2) for v in ratios["E12"]) == (1.11, 0.85)
    assert sorted(e for e in confirmed if ratios[e][1] < ratios["E07"][1]) == ["E02", "E06", "E08"]
