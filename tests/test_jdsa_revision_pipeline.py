"""The JDSA revision pipeline scores observable events only, and its table regenerator reproduces
the submitted tables when it is given the submitted panel."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "jdsa_revision"
REVISION = ROOT / "results" / "jdsa_revision"
UNOBSERVABLE = {("IWM", "E01"), ("HYG", "E01"), ("HYG", "E02"), ("HYG", "E03"), ("TLT", "E01"), ("TLT", "E02"), ("TLT", "E03")}


def _run(code: str, tmp_path: Path, **env) -> str:
    full = {**os.environ, "JDSA_OUT_ROOT": str(tmp_path), **env}
    out = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, env=full, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("panel", ["jdsa_revision_v2", "submitted"])
def test_load_asset_returns_only_observable_events(tmp_path, panel):
    code = (
        "import inspect, _common as c\n"
        "print(list(inspect.signature(c.load_asset).parameters))\n"
        "for t in c.TICKERS:\n"
        "    df, ev = c.load_asset(t)\n"
        "    print(t, len(ev), ','.join(ev['event_id']))\n"
    )
    lines = _run(code, tmp_path, JDSA_PANEL=panel).strip().splitlines()
    assert lines[0] == "['ticker']"  # no switch that could bring unobservable events back
    counts = {line.split()[0]: int(line.split()[1]) for line in lines[1:]}
    assert counts == {"SPY": 13, "QQQ": 13, "IWM": 12, "HYG": 10, "TLT": 10}
    for line in lines[1:]:
        ticker, _, ids = line.split()
        assert not ({(ticker, e) for e in ids.split(",")} & UNOBSERVABLE)


def test_check_observable_stops_a_table_with_an_unobservable_event(tmp_path):
    code = (
        "import _common as c\n"
        "df = c._load_processed_asset('HYG')\n"
        "try:\n"
        "    c.check_observable('HYG', df, c.panel_events('HYG'))\n"
        "    print('accepted')\n"
        "except RuntimeError as e:\n"
        "    print('stopped', e)\n"
    )
    out = _run(code, tmp_path)
    assert out.startswith("stopped") and "E01" in out


def test_run_all_has_no_option_to_score_unobservable_events():
    out = subprocess.run([sys.executable, str(SCRIPTS / "run_all.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0
    options = {tok for tok in out.stdout.split() if tok.startswith("--")}
    assert options <= {"--help", "--panel", "--out", "--skip"}


def test_frozen_revision_outputs_contain_no_unobservable_event_rows():
    files = [REVISION / "t1_denominators" / "event_level_metrics_observable.csv",
             REVISION / "t4_operating_curves" / "matched_comparison_other_assets_per_event.csv"]
    if not all(f.exists() for f in files):
        pytest.skip("revision outputs not generated")
    for f in files:
        rows = pd.read_csv(f)
        assert not (set(zip(rows["ticker"], rows["event_id"])) & UNOBSERVABLE), f.name
    den = pd.read_csv(REVISION / "t1_denominators" / "denominators_by_asset.csv").set_index("ticker")["observable_events"].to_dict()
    assert den == {"SPY": 13, "QQQ": 13, "IWM": 12, "HYG": 10, "TLT": 10}


def test_table_regenerator_reproduces_the_submitted_tables_on_the_submitted_panel(tmp_path):
    full = {**os.environ, "JDSA_PANEL": "submitted", "JDSA_OUT_ROOT": str(tmp_path)}
    out = subprocess.run([sys.executable, str(SCRIPTS / "t13_paper_tables.py")], cwd=ROOT, env=full, capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr
    index = pd.read_csv(tmp_path / "t13_paper_tables" / "index.csv")
    for row in index.itertuples():
        new, old = pd.read_csv(tmp_path / "t13_paper_tables" / row.file), pd.read_csv(ROOT / row.submitted_artifact)
        assert list(new.columns) == list(old.columns) and new.shape == old.shape, row.table
        numeric = old.select_dtypes("number").columns
        assert (new.drop(columns=numeric).astype(str).values == old.drop(columns=numeric).astype(str).values).all(), row.table
        assert np.allclose(new[numeric].to_numpy(float), old[numeric].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True), row.table
