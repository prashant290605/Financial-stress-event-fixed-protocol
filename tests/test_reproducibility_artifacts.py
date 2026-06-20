from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "data/events_spy.csv",
    "data/events_multi_asset.csv",
    "config/paper_experiment.json",
    "results/multi_asset/multi_asset_pointwise_metrics.csv",
    "results/event_level/event_level_metrics.csv",
    "results/cost/pointwise_cost_by_asset.csv",
    "results/cost/event_aware_cost_by_asset.csv",
    "scripts/reproduce_paper.py",
]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def test_required_reproducibility_files_exist():
    for rel in REQUIRED_FILES:
        assert (ROOT / rel).exists(), f"missing required reproducibility file: {rel}"


def test_event_file_has_required_columns_and_sources():
    events = pd.read_csv(ROOT / "data" / "events_spy.csv")
    required = {
        "event_name",
        "calendar_date",
        "mapped_trading_date",
        "event_type",
        "inclusion_rule",
        "source_reference",
    }
    assert required <= set(events.columns)
    assert events["source_reference"].notna().all()
    assert events["source_reference"].str.strip().ne("").all()
    assert events["inclusion_rule"].notna().all()
    assert events["inclusion_rule"].str.strip().ne("").all()


def test_primary_reproduction_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "Reproduced paper artifacts" in result.stdout


def test_expected_result_files_generated():
    generated = [
        "results/paper/table_spy_methods.csv",
        "results/paper/table_event_window_sensitivity.csv",
        "results/paper/table_bootstrap_intervals.csv",
        "results/multi_asset/multi_asset_pointwise_metrics.csv",
        "results/event_level/event_level_metrics.csv",
        "results/cost/pointwise_cost_by_asset.csv",
        "results/cost/event_aware_cost_by_asset.csv",
    ]
    for rel in generated:
        path = ROOT / rel
        assert path.exists()
        assert path.stat().st_size > 0


def test_main_spy_table_values_match_csv():
    tex = (ROOT / "paper" / "research_paper.tex").read_text(encoding="utf-8")
    metrics = pd.read_csv(ROOT / "results" / "paper" / "table_spy_methods.csv").set_index("method")
    volatility = metrics.loc["volatility_percentile"]
    hybrid = metrics.loc["hybrid_windowed_confirmation"]
    assert (
        f"Volatility percentile & {fmt(volatility['precision'])} & {fmt(volatility['recall'])} & "
        f"{fmt(volatility['f1'])} & {fmt(volatility['fpr'])}"
    ) in tex
    assert (
        f"Hybrid ($W=10$) & {fmt(hybrid['precision'])} & {fmt(hybrid['recall'])} & "
        f"{fmt(hybrid['f1'])} & {fmt(hybrid['fpr'])}"
    ) in tex


def test_multi_asset_table_values_match_csv():
    tex = (ROOT / "paper" / "research_paper.tex").read_text(encoding="utf-8")
    metrics = pd.read_csv(ROOT / "results" / "multi_asset" / "multi_asset_pointwise_metrics.csv")
    row = metrics[
        (metrics["ticker"] == "QQQ")
        & (metrics["method"] == "Hybrid volatility + instability confirmation")
    ].iloc[0]
    expected = (
        f"QQQ & Hybrid & {fmt(row['precision'])} & {fmt(row['recall'])} & {fmt(row['f1'])} & "
        f"{fmt(row['fpr'])} & {row['mean_delay']:.1f} & {row['alignment_error']:.1f}"
    )
    assert expected in tex


def test_event_and_cost_tables_match_csv():
    tex = (ROOT / "paper" / "research_paper.tex").read_text(encoding="utf-8")
    event_summary = pd.read_csv(ROOT / "results" / "event_level" / "table_multi_asset_event_level_summary.csv")
    event_row = event_summary[
        (event_summary["ticker"] == "TLT")
        & (event_summary["method"] == "Hybrid volatility + instability confirmation")
    ].iloc[0]
    assert (
        f"TLT & Hybrid & {fmt(event_row['event_detection_rate'])} & "
        f"{event_row['median_first_delay']:.1f} & {fmt(event_row['mean_event_coverage'])}"
    ) in tex

    cost = pd.read_csv(ROOT / "results" / "cost" / "table_spy_event_aware_cost.csv")
    cost_row = cost[
        (cost["method"] == "Hybrid volatility + instability confirmation")
        & (cost["ratio_false_alarm_to_missed_event"] == 0.05)
    ].iloc[0]
    assert (
        f"0.05 & 14.50 & {cost_row['cost']:.2f} & 3 & {int(cost_row['missed_events'])}"
    ) in tex
