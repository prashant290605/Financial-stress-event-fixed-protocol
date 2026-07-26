from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MANUSCRIPT = ROOT / "paper" / "manuscript.tex"


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


def _numeric_cells(line: str) -> list[float | None]:
    """Parse a LaTeX tabular row into floats, using None for non-numeric cells."""
    cells = line.rstrip().removesuffix(r"\\").split("&")
    out: list[float | None] = []
    for cell in cells:
        token = cell.strip().replace("$", "").replace("−", "-").replace("\\,", "")
        try:
            out.append(float(token))
        except ValueError:
            out.append(None)
    return out


def assert_manuscript_row(tex: str, label: str, expected: list[float], digits: list[int]) -> None:
    """Assert some manuscript table row beginning with `label` matches `expected`.

    The comparison is numeric with a half-unit-in-the-last-place tolerance, so a
    row is accepted under either half-up or half-to-even rounding. Matching by
    value rather than by formatted string keeps the test from breaking when a
    column label is reworded or a tie is rounded the other way.
    """
    candidates = [
        line for line in tex.splitlines() if line.strip().startswith(label)
    ]
    assert candidates, f"no manuscript row starts with {label!r}"

    failures = []
    for line in candidates:
        cells = [c for c in _numeric_cells(line) if c is not None]
        if len(cells) < len(expected):
            failures.append((line, "too few numeric cells"))
            continue
        deltas = [abs(cells[i] - expected[i]) for i in range(len(expected))]
        tolerances = [0.5 * 10 ** (-d) for d in digits]
        if all(delta <= tol for delta, tol in zip(deltas, tolerances)):
            return
        failures.append((line, [round(d, 6) for d in deltas]))
    raise AssertionError(
        f"no row starting with {label!r} matched {expected}; candidates: {failures}"
    )


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


def test_manuscript_source_present():
    assert MANUSCRIPT.exists(), f"missing manuscript source: {MANUSCRIPT}"


def test_main_spy_table_values_match_csv():
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    metrics = pd.read_csv(ROOT / "results" / "paper" / "table_spy_methods.csv").set_index("method")
    volatility = metrics.loc["volatility_percentile"]
    confirmation = metrics.loc["hybrid_windowed_confirmation"]
    assert_manuscript_row(
        tex,
        "Volatility percentile",
        [volatility["precision"], volatility["recall"], volatility["f1"], volatility["fpr"]],
        [3, 3, 3, 3],
    )
    assert_manuscript_row(
        tex,
        "Confirmation ($W=10$)",
        [confirmation["precision"], confirmation["recall"], confirmation["f1"], confirmation["fpr"]],
        [3, 3, 3, 3],
    )


def test_multi_asset_table_values_match_csv():
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    metrics = pd.read_csv(ROOT / "results" / "multi_asset" / "multi_asset_pointwise_metrics.csv")
    row = metrics[
        (metrics["ticker"] == "QQQ")
        & (metrics["method"] == "Hybrid volatility + instability confirmation")
    ].iloc[0]
    assert_manuscript_row(
        tex,
        "QQQ & Confirmation",
        [row["precision"], row["recall"], row["f1"], row["fpr"], row["mean_delay"], row["alignment_error"]],
        [3, 3, 3, 3, 1, 1],
    )


def test_event_and_cost_tables_match_csv():
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    event_summary = pd.read_csv(ROOT / "results" / "event_level" / "table_multi_asset_event_level_summary.csv")
    event_row = event_summary[
        (event_summary["ticker"] == "TLT")
        & (event_summary["method"] == "Hybrid volatility + instability confirmation")
    ].iloc[0]
    assert_manuscript_row(
        tex,
        "TLT & Confirm.",
        [event_row["event_detection_rate"], event_row["median_first_delay"], event_row["mean_event_coverage"]],
        [3, 1, 3],
    )

    cost = pd.read_csv(ROOT / "results" / "cost" / "table_spy_event_aware_cost.csv")
    volatility_row = cost[
        (cost["method"] == "Volatility percentile")
        & (cost["ratio_false_alarm_to_missed_event"] == 0.05)
    ].iloc[0]
    confirmation_row = cost[
        (cost["method"] == "Hybrid volatility + instability confirmation")
        & (cost["ratio_false_alarm_to_missed_event"] == 0.05)
    ].iloc[0]
    assert_manuscript_row(
        tex,
        "0.05 &",
        [
            0.05,
            volatility_row["cost"],
            confirmation_row["cost"],
            volatility_row["missed_events"],
            confirmation_row["missed_events"],
        ],
        [2, 2, 2, 0, 0],
    )
