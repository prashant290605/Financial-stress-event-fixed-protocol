"""Shared paths and helpers for the JDSA revision scripts.

Every script in this folder reads the archived inputs and writes only under the
revision output root. No submitted result file or submitted event file is modified.

Two settings are read from the environment so that the whole pipeline can be rerun on
another event panel for comparison:

* ``JDSA_PANEL``: event panel, default ``jdsa_revision_v2`` (E03 corrected);
* ``JDSA_OUT_ROOT``: output root, default ``results/jdsa_revision``.

The revision pipeline always scores observable events only. ``load_asset`` has no option
to include an unobservable event, and ``check_observable`` stops a script if any event
table it is about to score contains one.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.event_panels import REVISION_PANEL, asset_event_table, spy_event_table  # noqa: E402
from src.multi_asset_evaluation import _load_processed_asset  # noqa: E402
from src.observability import observable_event_flags, observable_events  # noqa: E402
from src.paper_protocol import load_config  # noqa: E402

CONFIG = load_config(ROOT / "config" / "paper_experiment.json")
PANEL = os.environ.get("JDSA_PANEL", REVISION_PANEL)
OUT_ROOT = Path(os.environ.get("JDSA_OUT_ROOT", ROOT / "results" / "jdsa_revision"))
if not OUT_ROOT.is_absolute():
    OUT_ROOT = ROOT / OUT_ROOT
TICKERS = ["SPY", "QQQ", "IWM", "HYG", "TLT"]
HALF_WINDOW = CONFIG["events"]["half_window"]
SEED = CONFIG["seed"]
N_RESAMPLES = CONFIG["bootstrap"]["n_resamples"]
VOL_NAME = "Volatility percentile"
CONF_NAME = "Hybrid volatility + instability confirmation"


def outdir(name: str) -> Path:
    path = OUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=None)
def spy_events() -> pd.DataFrame:
    """SPY event table of the active panel, with severity terciles."""
    return spy_event_table(PANEL, _load_processed_asset("SPY"), CONFIG)


def severity_labels() -> dict[str, str]:
    return spy_events().set_index("event_id")["severity_label"].to_dict()


@lru_cache(maxsize=None)
def panel_events(ticker: str) -> pd.DataFrame:
    """All 13 events of the active panel mapped to one asset, observable or not (for listings only)."""
    return asset_event_table(spy_events(), ticker, _load_processed_asset(ticker), CONFIG).reset_index(drop=True)


def check_observable(ticker: str, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Stop if an event table about to be scored contains an event the asset cannot observe."""
    flags = observable_event_flags(df, events)
    if not flags.all():
        raise RuntimeError(f"{ticker}: unobservable events would be scored: {events.loc[~flags, 'event_id'].tolist()}")
    return events


def load_asset(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processed series and the observable events of the active panel; no other option exists."""
    df = _load_processed_asset(ticker)
    events = observable_events(df, panel_events(ticker)).reset_index(drop=True)
    return df, check_observable(ticker, df, events)


def events_by_ticker() -> dict[str, pd.DataFrame]:
    return {t: load_asset(t)[1] for t in TICKERS}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.date().isoformat()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"not serializable: {type(value)}")

    path.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")


def figure_style() -> None:
    """Same Springer figure settings as scripts/generate_final_figures.py."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "jdsa-revision",
        }
    )


def save_figure(fig, path_without_suffix: Path) -> None:
    """PDF and PNG with fixed metadata so reruns produce identical files."""
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path_without_suffix.with_suffix(".png"), bbox_inches="tight", dpi=300, metadata={"Software": None})


COLORS = {"Volatility": "#1f77b4", "Confirmation": "#d95f02", "CUSUM": "#2ca02c", "MMD": "#7570b3"}
