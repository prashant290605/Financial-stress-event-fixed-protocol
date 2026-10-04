"""Task 12: final audit of the submitted manuscript numbers, and integrity checks.

Every numeric cell of the event-dependent manuscript tables (Tables 2-16) is parsed from the
submitted source and compared, at the printed precision, with the frozen revision value. A
curated list adds the in-text statements that quote those numbers or depend on them. Each
change is attributed by comparing three values: the printed one, the same revision code run
on the submitted panel (``pre_E03_fix``), and the corrected panel:

* printed differs from the submitted-panel run: a correction made on the submitted panel
  (observable denominators, estimator label, redesigned interval, documentation);
* the submitted-panel run differs from the corrected run: the E03 correction.

New results that the submitted paper does not report are listed with an empty submitted
value. Old values are the printed numbers; new values are read from the outputs, never typed.

Optional arguments:
  --manifest PATH   sha256 manifest of the project taken before Phase 1 (sha256sum format)
  --repro-dir PATH  a copy of the repository in which the submitted pipeline was re-run

Outputs: <out root>/final_audit.csv and <out root>/t12_audit/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from _common import CONF_NAME, OUT_ROOT, PANEL, ROOT, VOL_NAME, outdir  # noqa: E402

OUT = outdir("t12_audit")
ZIP = ROOT / "A_Reproducible_Event_Fixed_Protocol_for_Evaluating_Financial_Stress_Detectors.zip"
PRE = OUT_ROOT / "pre_E03_fix"
EVENT_IDS = [f"E{i:02d}" for i in range(1, 14)]


# --------------------------------------------------------------------------- inputs
def manuscript() -> tuple[str, str]:
    if ZIP.exists():
        with zipfile.ZipFile(ZIP) as z:
            name = next(n for n in z.namelist() if n.endswith("manuscript.tex"))
            return z.read(name).decode("utf-8"), f"{ZIP.name}:{name}"
    return (ROOT / "paper" / "manuscript.tex").read_text(encoding="utf-8"), "paper/manuscript.tex"


def clean(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("$", "").replace("\\,", "").replace("\\-", "").replace("\\&", "&")
                  .replace("−", "-").replace("\\pm", "+/-").replace("\\kappa", "kappa")).strip()


def table_rows(tex: str, label: str) -> list[list[str]]:
    i = tex.index(f"\\label{{{label}}}")
    j = tex.index("\\midrule", i)
    k = tex.index("\\bottomrule", j)
    rows = []
    for line in tex[j + len("\\midrule"): k].splitlines():
        line = line.strip()
        if line.endswith("\\\\"):
            rows.append([clean(c) for c in re.split(r"(?<!\\)&", line[:-2])])
    return rows


class Run:
    """Cached access to one run's outputs."""

    def __init__(self, root: Path):
        self.root = root

    @lru_cache(maxsize=None)
    def csv(self, rel: str) -> pd.DataFrame:
        return pd.read_csv(self.root / rel)

    @lru_cache(maxsize=None)
    def js(self, rel: str) -> dict:
        return json.loads((self.root / rel).read_text(encoding="utf-8"))


POST, PRE_RUN = Run(OUT_ROOT), Run(PRE)


# --------------------------------------------------------------------------- comparison helpers
def as_number(s) -> float | None:
    try:
        return float(str(s).replace("−", "-"))
    except ValueError:
        return None


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def fmt_like(printed: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, str):
        return value
    if "/" in printed:
        return f"{int(value)}/{printed.split('/')[1]}"
    return f"{float(value):.{decimals(printed)}f}"


def matches(printed: str, value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return printed.strip().upper() == "NA"
    if "/" in printed and not isinstance(value, str):
        return int(printed.split("/")[0]) == int(value)
    p = as_number(printed)
    if p is None or isinstance(value, str):
        return str(printed).strip().lower() == str(value).strip().lower()
    return abs(p - float(value)) <= 0.5 * 10 ** (-decimals(printed)) + 1e-9


# --------------------------------------------------------------------------- table specifications
def spec_rows():
    """(table, label, source file, row key function, [(printed column, name, field)], correction category)."""
    t3_rows = {"Volatility percentile": "volatility_percentile", "CUSUM": "cusum", "Instability score": "instability_score",
               "MMD only": "mmd_only_confirmation", "|kappa| only": "kappa_only_confirmation",
               "Confirmation (W=10)": "hybrid_windowed_confirmation"}
    vix_rows = {"VIX percentile": "VIX percentile", "Volatility percentile": "Volatility percentile",
                "Confirmation (W=10)": "Hybrid (W=10)", "VIX + instability": "VIX + instability confirmation"}
    method = {"Volatility": VOL_NAME, "Confirmation": CONF_NAME, "Vol.": VOL_NAME, "Confirm.": CONF_NAME}
    return [
        ("Table 3", "tab:spy", "t13_paper_tables/table3_spy_methods.csv", lambda i, c, d: d.set_index("method").loc[t3_rows[c[0]]],
         [(1, "Precision", "precision"), (2, "Recall", "recall"), (3, "F1", "f1"), (4, "FPR", "fpr"), (5, "Detect. rate", "event_coverage"),
          (6, "Delay", "event_delay"), (7, "Align. err.", "event_alignment_error")], "none expected"),
        ("Table 4", "tab:vix", "t13_paper_tables/table4_vix_benchmark.csv", lambda i, c, d: d.set_index("method").loc[vix_rows[c[0]]],
         [(1, "Precision", "precision"), (2, "Recall", "recall"), (3, "F1", "f1"), (4, "FPR", "fpr"), (5, "Events", "events_detected"),
          (6, "Delay", "event_delay"), (7, "A_out", "alarm_days_outside"), (8, "r*", "crossover_vs_vix_cd005")], "none expected"),
        ("Table 5", "tab:multiasset_pointwise", "t1_denominators/multi_asset_pointwise_metrics_observable.csv",
         lambda i, c, d: d.set_index(["ticker", "method"]).loc[(c[0], method[c[1]])],
         [(2, "Precision", "precision"), (3, "Recall", "recall"), (4, "F1", "f1"), (5, "FPR", "fpr"), (6, "Mean delay", "mean_delay"),
          (7, "Align. err.", "alignment_error")], "observable-event denominators"),
        ("Table 6", "tab:spy_event_level", "t13_paper_tables/table6_spy_event_level_detection.csv", lambda i, c, d: d.iloc[i],
         [(0, "Event", "event_name"), (1, "Severity", "severity_label"), (2, "Vol. det.", "volatility_detected"), (3, "Conf. det.", "hybrid_detected"),
          (4, "Vol. delay", "volatility_first_delay"), (5, "Conf. delay", "hybrid_first_delay"), (6, "Vol. cov.", "volatility_coverage"),
          (7, "Conf. cov.", "hybrid_coverage")], "none expected"),
        ("Table 7", "tab:mechanism", "t13_paper_tables/table7_component_diagnosis.csv", lambda i, c, d: d.iloc[i],
         [(0, "Event", "event_name"), (1, "Vol. det.", "volatility_detected"), (2, "Conf. det.", "hybrid_detected"),
          (3, "Vol. alarm days", "volatility_alarm_days"), (4, "Instab. alarm days", "instability_alarm_days"), (5, "Peak z_MMD", "peak_smoothed_z_mmd")],
         "none expected"),
        ("Table 8", "tab:multiasset_event_level", "t1_denominators/table8_multi_asset_event_level_observable.csv",
         lambda i, c, d: d.set_index(["ticker", "method"]).loc[(c[0], method[c[1]])],
         [(2, "Detect. rate", "event_detection_rate"), (3, "Med. delay", "median_first_delay"), (4, "Mean cov.", "mean_event_coverage")],
         "observable-event denominators"),
        ("Table 9", "tab:event_level_severity", "t1_denominators/table9_severity_observable.csv",
         lambda i, c, d: d.set_index(["severity_label", "method"]).loc[(c[0].lower(), method[c[1]])],
         [(2, "Detect. rate", "event_detection_rate"), (3, "Med. delay (mean of per-asset medians)", "mean_of_per_asset_median_first_delay"),
          (4, "Mean cov.", "mean_event_coverage")], "observable-event denominators"),
        ("Table 10", "tab:cost", "t13_paper_tables/table10_cost_sensitivity.csv",
         lambda i, c, d: d.set_index("fp_to_fn_cost_ratio").loc[int(float(c[0]))],
         [(1, "Vol. cost", "volatility_cost"), (2, "Confirm. cost", "hybrid_cost"), (3, "Lower cost", "preferred_method")], "none expected"),
        ("Table 11", "tab:event_cost", "t1_denominators/event_aware_cost_by_asset_observable.csv", None,
         [(1, "Vol. cost", ("cost", VOL_NAME)), (2, "Confirm. cost", ("cost", CONF_NAME)), (3, "Vol. missed", ("missed_events", VOL_NAME)),
          (4, "Confirm. missed", ("missed_events", CONF_NAME))], "none expected"),
        ("Table 12", "tab:crossover", "t1_denominators/table12_cost_crossover_observable.csv", None,
         [(1, "c_D=0", 0.0), (2, "c_D=0.05", 0.05), (3, "c_D=0.1", 0.1)], "observable-event denominators"),
        ("Table 13", "tab:eventwindow", "t13_paper_tables/table13_event_window_sensitivity.csv",
         lambda i, c, d: d.set_index(["half_window", "method"]).loc[(int(c[0].replace("+/-", "")),
                                                                     "Volatility percentile" if c[1] == "Vol." else "Hybrid W=10")],
         [(2, "Prec.", "precision"), (3, "Recall", "recall"), (4, "F1", "f1"), (5, "FPR", "fpr")], "none expected"),
        ("Table 14", "tab:weights", "t13_paper_tables/table14_weight_ablation.csv", lambda i, c, d: d.iloc[i],
         [(1, "Prec.", "precision"), (2, "Recall", "recall"), (3, "F1", "f1"), (4, "FPR", "fpr"), (5, "Delay", "event_delay"),
          (6, "Events", "events_detected")], "none expected"),
        ("Table 15", "tab:kappasign", "t13_paper_tables/table15_kappa_sign_ablation.csv", lambda i, c, d: d.iloc[i],
         [(1, "Prec.", "precision"), (2, "Recall", "recall"), (3, "F1", "f1"), (4, "FPR", "fpr"), (5, "Delay", "event_delay"),
          (6, "Events", "events_detected")], "none expected"),
        ("Table 16", "tab:causal", "t13_paper_tables/table16_causal_confirmation.csv", lambda i, c, d: d.iloc[i],
         [(1, "Prec.", "precision"), (2, "Recall", "recall"), (3, "F1", "f1"), (4, "FPR", "fpr"), (5, "Events", "events_detected"),
          (6, "Delay", "event_delay"), (7, "A_out", "alarm_days_outside"), (8, "r*", "crossover_ratio_cd005")], "none expected"),
    ]


def cell_value(table: str, field, row, cells, run: Run, source: str):
    """Value of one cell for one run, in the form the manuscript prints it."""
    if table == "Table 11":
        d = run.csv(source)
        d = d[(d["ticker"] == "SPY") & (d["delay_cost"] == 0.05) & np.isclose(d["ratio_false_alarm_to_missed_event"], float(cells[0]))]
        return float(d.set_index("method").loc[field[1], field[0]])
    if table == "Table 12":
        d = run.csv(source)
        return float(d[(d["ticker"] == cells[0]) & np.isclose(d["delay_cost"], field)]["crossover_ratio"].iloc[0])
    value = row[field]
    if table == "Table 10" and field == "preferred_method":
        return {"hybrid": "Confirmation", "volatility": "Volatility", "tie": "Tie"}[value]
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        return value
    return float(value) if not pd.isna(value) else np.nan


def audit_tables(tex: str) -> list[dict]:
    rows = []
    for table, label, source, resolver, columns, category in spec_rows():
        printed_rows = table_rows(tex, label)
        for i, cells in enumerate(printed_rows):
            post_row = resolver(i, cells, POST.csv(source)) if resolver else None
            pre_row = resolver(i, cells, PRE_RUN.csv(source)) if resolver else None
            for col, name, field in columns:
                printed = cells[col]
                if printed == "":
                    continue
                post = cell_value(table, field, post_row, cells, POST, source)
                pre = cell_value(table, field, pre_row, cells, PRE_RUN, source)
                if name == "Event":  # labels are abbreviated in print; only a renamed event counts as a change
                    changed = pre != post
                    drivers = ["E03 correction"] if changed else []
                    rows.append({"kind": "table cell", "location": table, "row": cells[0], "column": name, "submitted_value": printed,
                                 "pre_E03_fix_value": pre, "revised_value": post if changed else printed, "changed": changed,
                                 "drivers": ";".join(drivers), "source": f"{source}:{field}"})
                    continue
                pre_s, post_s = fmt_like(printed, pre), fmt_like(printed, post)
                drivers = []
                if not matches(printed, pre):
                    drivers.append(category if category != "none expected" else "UNEXPECTED difference on the submitted panel")
                if pre_s != post_s or not matches(printed, post) and matches(printed, pre):
                    if not (isinstance(pre, float) and isinstance(post, float) and abs(pre - post) < 1e-12):
                        drivers.append("E03 correction")
                changed = not matches(printed, post)
                rows.append({"kind": "table cell", "location": table, "row": cells[0] if table not in ("Table 14", "Table 15", "Table 16")
                             else cells[0][:40], "column": name, "submitted_value": printed, "pre_E03_fix_value": pre_s,
                             "revised_value": post_s, "changed": changed, "drivers": ";".join(drivers) if changed else "",
                             "revised_full_precision": post if not isinstance(post, str) else np.nan, "source": f"{source}:{field}"})
    return rows


def audit_event_table(tex: str) -> list[dict]:
    """Table 2: dates, drawdown severity and label of each event."""
    rows = []
    post = POST.csv("t0_event_panel/events_spy_active_panel.csv").set_index("event_id")
    pre = PRE_RUN.csv("t0_event_panel/events_spy_active_panel.csv").set_index("event_id")
    for cells in table_rows(tex, "tab:events"):
        eid = cells[0]
        for col, name, field in [(2, "Calendar", "calendar_date"), (3, "Trading", "mapped_trading_date"), (6, "DD", "severity_score"),
                                 (7, "Severity", "severity_label")]:
            printed, pv, nv = cells[col], pre.loc[eid, field], post.loc[eid, field]
            changed = not matches(printed, nv)
            rows.append({"kind": "table cell", "location": "Table 2", "row": eid, "column": name, "submitted_value": printed,
                         "pre_E03_fix_value": fmt_like(printed, pv), "revised_value": fmt_like(printed, nv), "changed": changed,
                         "drivers": "E03 correction" if changed else "", "source": f"t0_event_panel/events_spy_active_panel.csv:{field}"})
        if pre.loc[eid, "event_name"] != post.loc[eid, "event_name"]:
            for col, name, field in [(1, "Event", "event_name"), (5, "Source label", "source_reference")]:
                rows.append({"kind": "table cell", "location": "Table 2", "row": eid, "column": name, "submitted_value": cells[col],
                             "pre_E03_fix_value": pre.loc[eid, field], "revised_value": post.loc[eid, field], "changed": True,
                             "drivers": "E03 correction", "source": f"t0_event_panel/events_spy_active_panel.csv:{field}"})
    return rows


# --------------------------------------------------------------------------- in-text statements
def f3(x) -> str:
    return f"{x:.3f}"


def text_rows() -> list[dict]:
    def t3(run, m, f):
        return run.csv("t13_paper_tables/table3_spy_methods.csv").set_index("method").loc[m, f]

    def vix(run, m, f):
        return run.csv("t13_paper_tables/table4_vix_benchmark.csv").set_index("method").loc[m, f]

    def t5(run, t, m, f):
        return run.csv("t1_denominators/multi_asset_pointwise_metrics_observable.csv").set_index(["ticker", "method"]).loc[(t, m), f]

    def t8(run, t, m):
        return run.csv("t1_denominators/table8_multi_asset_event_level_observable.csv").set_index(["ticker", "method"]).loc[(t, m)]

    def t9(run, s, m):
        return run.csv("t1_denominators/table9_severity_observable.csv").set_index(["severity_label", "method"]).loc[(s, m), "event_detection_rate"]

    def cross(run, t, c):
        d = run.csv("t1_denominators/table12_cost_crossover_observable.csv")
        return d[(d["ticker"] == t) & np.isclose(d["delay_cost"], c)].iloc[0]

    def t16(run, v, f):
        return run.csv("t13_paper_tables/table16_causal_confirmation.csv").set_index("variant").loc[v, f]

    def t16e(run, eid, v):
        return run.csv("t13_paper_tables/table16_per_event_first_delays.csv").set_index("event_id").loc[eid, f"{v}_first_delay"]

    def mech(run, q):
        return run.csv("t9_mechanism/mechanism_manuscript_comparison.csv").set_index("quantity").loc[q, "recomputed"]

    def sev(run, s, m, f):
        d = run.csv("t13_paper_tables/severity_day_level.csv")
        return d[(d["severity"] == s) & (d["method"] == m)].iloc[0][f]

    def ew(run, h, m, f):
        return run.csv("t13_paper_tables/table13_event_window_sensitivity.csv").set_index(["half_window", "method"]).loc[(h, m), f]

    def wa(run, f):
        return run.csv("t13_paper_tables/table14_weight_ablation.csv")[f]

    def ks(run, i, f):
        return run.csv("t13_paper_tables/table15_kappa_sign_ablation.csv").iloc[i][f]

    def cw(run, w, f):
        return run.csv("t13_paper_tables/confirmation_window_sensitivity.csv").set_index("window").loc[w, f]

    def ce(run, t):
        d = run.csv("t5_common_event_delay/common_event_delay_summary.csv")
        return d[(d["ticker"] == t) & (d["comparison"] == "volatility vs confirmation (submitted defaults)")].iloc[0]

    def u(run, est, block=121, comp="volatility default"):
        d = run.csv("t6_uncertainty/uncertainty_summary.csv")
        d = d[(d["ticker"] == "SPY") & d["comparison"].str.startswith(comp) & (d["estimand"] == est)]
        if block is not None:
            d = d[d["block_length"] == block]
        r = d.iloc[0]
        return f"{f3(r['observed'])} [{f3(r['ci_low'])}, {f3(r['ci_high'])}]"

    def ev_cost(run, ratio, m):
        d = run.csv("t1_denominators/event_aware_cost_by_asset_observable.csv")
        d = d[(d["ticker"] == "SPY") & (d["delay_cost"] == 0.05) & np.isclose(d["ratio_false_alarm_to_missed_event"], ratio)]
        return d.set_index("method").loc[m]

    def pw(run, ratio, m):
        d = run.csv("t13_paper_tables/table10_cost_sensitivity.csv").set_index("fp_to_fn_cost_ratio")
        return d.loc[ratio, "volatility_cost" if m == "v" else "hybrid_cost"]

    V, C = "volatility_percentile", "hybrid_windowed_confirmation"
    s = [
        ("Contribution 1; Results 4.1; Discussion; Conclusion", "precision, confirmation vs volatility", "0.886 vs 0.640",
         lambda r: f"{f3(t3(r, C, 'precision'))} vs {f3(t3(r, V, 'precision'))}", "E03 correction"),
        ("Contribution 1; Results 4.1; Discussion; Conclusion", "FPR, confirmation vs volatility", "0.005 vs 0.044",
         lambda r: f"{f3(t3(r, C, 'fpr'))} vs {f3(t3(r, V, 'fpr'))}", "E03 correction"),
        ("Results 4.1; Discussion; Conclusion", "recall, confirmation vs volatility", "0.122 vs 0.225",
         lambda r: f"{f3(t3(r, C, 'recall'))} vs {f3(t3(r, V, 'recall'))}", "E03 correction"),
        ("Results 4.1", "F1, confirmation vs volatility", "0.214 vs 0.333", lambda r: f"{f3(t3(r, C, 'f1'))} vs {f3(t3(r, V, 'f1'))}", "E03 correction"),
        ("Results 4.1; Discussion", "mean first alarm, confirmation vs volatility (own detected sets; printed as 6.8 after and 1.2 before)", "6.8 vs -1.2",
         lambda r: f"{t3(r, C, 'event_delay'):.1f} vs {t3(r, V, 'event_delay'):.1f}", "E03 correction"),
        ("Results 4.1; Discussion", "CUSUM F1, FPR, mean delay", "0.364, 0.344, -60",
         lambda r: f"{f3(t3(r, 'cusum', 'f1'))}, {f3(t3(r, 'cusum', 'fpr'))}, {t3(r, 'cusum', 'event_delay'):.0f}", "E03 correction"),
        ("Contribution 2; Results 4.2", "VIX + instability: precision 0.596 to 0.947, alarm days outside 214 to 9, events 10 to 7",
         "0.596->0.947; 214->9; 10->7",
         lambda r: f"{f3(vix(r, 'VIX percentile', 'precision'))}->{f3(vix(r, 'VIX + instability confirmation', 'precision'))}; "
                   f"{int(vix(r, 'VIX percentile', 'alarm_days_outside'))}->{int(vix(r, 'VIX + instability confirmation', 'alarm_days_outside'))}; "
                   f"{int(vix(r, 'VIX percentile', 'events_detected'))}->{int(vix(r, 'VIX + instability confirmation', 'events_detected'))}", "E03 correction"),
        ("Results 4.2", "VIX + instability: FPR 0.048 to 0.002; crossover 0.022 against 0.021", "0.048->0.002; 0.022 vs 0.021",
         lambda r: f"{f3(vix(r, 'VIX percentile', 'fpr'))}->{f3(vix(r, 'VIX + instability confirmation', 'fpr'))}; "
                   f"{f3(vix(r, 'VIX + instability confirmation', 'crossover_vs_vix_cd005'))} vs {f3(cross(r, 'SPY', 0.05)['crossover_ratio'])}", "E03 correction"),
        ("Results 4.2", "VIX mean first alarm against volatility", "-4.4 vs -1.2",
         lambda r: f"{vix(r, 'VIX percentile', 'event_delay'):.1f} vs {t3(r, V, 'event_delay'):.1f}", "E03 correction"),
        ("Contribution 3; Sec. 4.4.1; App. B", "dispersion of raw volatility and of x_t, confirmed events with a full pre-window", "3.03 and 1.29",
         lambda r: f"{float(mech(r, 'reference mean ratio raw volatility')):.2f} and {float(mech(r, 'reference mean ratio x_t')):.2f}", "E03 correction"),
        ("Sec. 4.4.1", "instability alarm days on detected events: range and mean", "10 to 49, mean 28.0",
         lambda r: f"{int(float(mech(r, 'confirmed instability alarm days min')))} to {int(float(mech(r, 'confirmed instability alarm days max')))}, "
                   f"mean {float(mech(r, 'confirmed instability alarm days mean')):.1f}", "E03 correction"),
        ("Sec. 4.4.1", "mean peak z_MMD across detected events", "3.38", lambda r: f"{float(mech(r, 'confirmed mean peak z_MMD')):.2f}", "E03 correction"),
        ("Sec. 4.4.1", "peak-to-trough spans, missed vs detected", "13.2 vs 22.5",
         lambda r: f"{float(mech(r, 'missed mean peak-to-trough days, severity window')):.1f} vs "
                   f"{float(mech(r, 'confirmed mean peak-to-trough days, severity window')):.1f}", "E03 correction"),
        ("Sec. 4.4.1", "mean severity, missed vs detected", "0.110 vs 0.196",
         lambda r: f"{float(mech(r, 'missed (by confirmation) mean severity')):.3f} vs {float(mech(r, 'confirmed mean severity')):.3f}", "E03 correction"),
        ("Contribution 4; Sec. 4.5.1; Discussion; Conclusion", "SPY crossover at c_D = 0.05 and its reciprocal (alarm days per missed event)", "0.021; 48",
         lambda r: f"{f3(cross(r, 'SPY', 0.05)['crossover_ratio'])}; {1 / cross(r, 'SPY', 0.05)['crossover_ratio']:.0f}", "E03 correction"),
        ("Sec. 4.5.1", "crossover arithmetic for SPY", "3.55/171 = 0.021",
         lambda r: f"{(cross(r, 'SPY', 0.05)['missed_hybrid'] - cross(r, 'SPY', 0.05)['missed_volatility']) + 0.05 * (cross(r, 'SPY', 0.05)['positive_delay_hybrid'] - cross(r, 'SPY', 0.05)['positive_delay_volatility']):.2f}/"
                   f"{int(cross(r, 'SPY', 0.05)['alarm_days_outside_volatility'] - cross(r, 'SPY', 0.05)['alarm_days_outside_hybrid'])} = {f3(cross(r, 'SPY', 0.05)['crossover_ratio'])}",
         "E03 correction"),
        ("Sec. 4.5", "event cost at c_FA/c_ME = 0.01, volatility vs confirmation; alarm days outside 195 vs 24",
         "6.70 vs 8.54; 195 vs 24",
         lambda r: f"{ev_cost(r, 0.01, VOL_NAME)['cost']:.2f} vs {ev_cost(r, 0.01, CONF_NAME)['cost']:.2f}; "
                   f"{int(ev_cost(r, 0.01, VOL_NAME)['false_alarm_days'])} vs {int(ev_cost(r, 0.01, CONF_NAME)['false_alarm_days'])}", "E03 correction"),
        ("Sec. 4.5", "day-level cost at ratio 1 and 50, volatility vs confirmation", "1384 vs 1372; 10939 vs 2548",
         lambda r: f"{pw(r, 1, 'v'):.0f} vs {pw(r, 1, 'c'):.0f}; {pw(r, 50, 'v'):.0f} vs {pw(r, 50, 'c'):.0f}", "E03 correction"),
        ("Sec. 4.5.1; Discussion; Conclusion", "range of crossover ratios across assets and delay costs", "0.007 to 0.059",
         lambda r: f"{f3(r.csv('t1_denominators/table12_cost_crossover_observable.csv')['crossover_ratio'].min())} to "
                   f"{f3(r.csv('t1_denominators/table12_cost_crossover_observable.csv')['crossover_ratio'].max())}", "E03 correction"),
        ("Sec. 4.3", "QQQ FPR and F1, volatility to confirmation", "0.039->0.006; 0.280->0.139",
         lambda r: f"{f3(t5(r, 'QQQ', VOL_NAME, 'fpr'))}->{f3(t5(r, 'QQQ', CONF_NAME, 'fpr'))}; {f3(t5(r, 'QQQ', VOL_NAME, 'f1'))}->{f3(t5(r, 'QQQ', CONF_NAME, 'f1'))}",
         "E03 correction"),
        ("Sec. 4.3", "IWM FPR and F1, volatility to confirmation", "0.056->0.028; 0.272->0.159",
         lambda r: f"{f3(t5(r, 'IWM', VOL_NAME, 'fpr'))}->{f3(t5(r, 'IWM', CONF_NAME, 'fpr'))}; {f3(t5(r, 'IWM', VOL_NAME, 'f1'))}->{f3(t5(r, 'IWM', CONF_NAME, 'f1'))}",
         "observable-event denominators"),
        ("Sec. 4.3", "TLT precision, confirmation vs volatility", "0.367 vs 0.507",
         lambda r: f"{f3(t5(r, 'TLT', CONF_NAME, 'precision'))} vs {f3(t5(r, 'TLT', VOL_NAME, 'precision'))}", "observable-event denominators"),
        ("Sec. 4.3; 4.4.3; 4.5.1", "'on HYG the confirmation rule alarms earlier'", "stated",
         lambda r: f"common events {int(ce(r, 'HYG')['n_common'])}: {ce(r, 'HYG')['common_mean_delay_a']:.1f} vs {ce(r, 'HYG')['common_mean_delay_b']:.1f} "
                   f"(own sets {ce(r, 'HYG')['own_set_mean_delay_a']:.1f} vs {ce(r, 'HYG')['own_set_mean_delay_b']:.1f})", "common-event delay"),
        ("Sec. 4.4.3", "detection rates, volatility vs confirmation (SPY, QQQ, IWM, HYG, TLT)",
         "0.769/0.615, 0.462/0.385, 0.538/0.385, 0.385/0.231, 0.538/0.308",
         lambda r: ", ".join(f"{f3(t8(r, t, VOL_NAME)['event_detection_rate'])}/{f3(t8(r, t, CONF_NAME)['event_detection_rate'])}"
                             for t in ["SPY", "QQQ", "IWM", "HYG", "TLT"]), "observable-event denominators"),
        ("Sec. 4.4.4", "detection by severity, confirmation (critical, high, medium) vs volatility", "0.680, 0.300, 0.100 vs 0.840, 0.550, 0.150",
         lambda r: f"{f3(t9(r, 'critical', CONF_NAME))}, {f3(t9(r, 'high', CONF_NAME))}, {f3(t9(r, 'medium', CONF_NAME))} vs "
                   f"{f3(t9(r, 'critical', VOL_NAME))}, {f3(t9(r, 'high', VOL_NAME))}, {f3(t9(r, 'medium', VOL_NAME))}", "observable-event denominators"),
        ("Sec. 4.6.1", "precision gap at +/-20 and +/-60; FPR gap at +/-20 and +/-60", "0.155 to 0.247; 0.043 to 0.038",
         lambda r: f"{f3(ew(r, 20, 'Hybrid W=10', 'precision') - ew(r, 20, 'Volatility percentile', 'precision'))} to "
                   f"{f3(ew(r, 60, 'Hybrid W=10', 'precision') - ew(r, 60, 'Volatility percentile', 'precision'))}; "
                   f"{f3(ew(r, 20, 'Volatility percentile', 'fpr') - ew(r, 20, 'Hybrid W=10', 'fpr'))} to "
                   f"{f3(ew(r, 60, 'Volatility percentile', 'fpr') - ew(r, 60, 'Hybrid W=10', 'fpr'))}", "E03 correction"),
        ("Sec. 4.6.2", "F1 and FPR ranges across the ten weightings", "[0.210, 0.235]; [0.005, 0.009]",
         lambda r: f"[{f3(wa(r, 'f1').min())}, {f3(wa(r, 'f1').max())}]; [{f3(wa(r, 'fpr').min())}, {f3(wa(r, 'fpr').max())}]", "E03 correction"),
        ("Sec. 4.6.2", "events detected under the ten weightings (printed as 8 under nine, 7 under the tenth)", "8 under 9, 7 under 1",
         lambda r: ", ".join(f"{k} under {int((wa(r, 'events_detected') == k).sum())}" for k in sorted(set(wa(r, 'events_detected').astype(int)), reverse=True)),
         "E03 correction"),
        ("Sec. 4.6.3", "F1 of the low-kappa and high-kappa variants; high-kappa mean first alarm", "0.202; 0.242; 1.6",
         lambda r: f"{f3(ks(r, 1, 'f1'))}; {f3(ks(r, 2, 'f1'))}; {ks(r, 2, 'event_delay'):.1f}", "E03 correction"),
        ("Sec. 4.6.4", "trailing W=10: precision, alarm days outside, mean first alarm, crossover", "0.917; 24->14; 10.8; 0.027",
         lambda r: f"{f3(t16(r, 'trailing_w10', 'precision'))}; {int(t16(r, 'symmetric_w10', 'alarm_days_outside'))}->{int(t16(r, 'trailing_w10', 'alarm_days_outside'))}; "
                   f"{t16(r, 'trailing_w10', 'event_delay'):.1f}; {f3(t16(r, 'trailing_w10', 'crossover_ratio_cd005'))}", "E03 correction"),
        ("Sec. 4.6.4", "WorldCom first alarm, symmetric W=10 to trailing W=10 (printed as 4 days before to 6 days after)", "-4 -> 6",
         lambda r: f"{t16e(r, 'E03', 'symmetric_w10'):.0f} -> {t16e(r, 'E03', 'trailing_w10'):.0f}", "E03 correction"),
        ("Sec. 4.6.4", "fully causal W=40: events, alarm days outside, share of the gap given back, max crossover", "7/13; 105; 81 of 171; 0.058",
         lambda r: f"{int(t16(r, 'fully_causal_w40', 'events_detected'))}/13; {int(t16(r, 'fully_causal_w40', 'alarm_days_outside'))}; "
                   f"{int(t16(r, 'fully_causal_w40', 'alarm_days_outside') - t16(r, 'symmetric_w10', 'alarm_days_outside'))} of "
                   f"{int(t16(r, 'volatility_reference', 'alarm_days_outside') - t16(r, 'symmetric_w10', 'alarm_days_outside'))}; "
                   f"{f3(r.csv('t13_paper_tables/table16_causal_confirmation.csv')['crossover_ratio_cd005'].max())}", "E03 correction"),
        ("Sec. 4.6.5", "severity day-level F1 and FPR (critical, high, medium), volatility vs confirmation",
         "F1 0.412/0.316, 0.195/0.134, 0.027/0.040; FPR 0.057/0.015, 0.080/0.030, 0.096/0.036",
         lambda r: "F1 " + ", ".join(f"{f3(sev(r, s_, 'volatility_percentile', 'f1'))}/{f3(sev(r, s_, 'hybrid_windowed_confirmation', 'f1'))}"
                                     for s_ in ["critical", "high", "medium"]) + "; FPR " +
                   ", ".join(f"{f3(sev(r, s_, 'volatility_percentile', 'fpr'))}/{f3(sev(r, s_, 'hybrid_windowed_confirmation', 'fpr'))}"
                             for s_ in ["critical", "high", "medium"]), "E03 correction"),
        ("Sec. 4.6.5", "MMD-only confirmation F1 and FPR against the default", "0.219 vs 0.214; 0.011 vs 0.005",
         lambda r: f"{f3(r.csv('t13_paper_tables/score_component_sensitivity.csv').set_index('variant').loc['MMD only', 'f1'])} vs "
                   f"{f3(r.csv('t13_paper_tables/score_component_sensitivity.csv').set_index('variant').loc['default no-CIS', 'f1'])}; "
                   f"{f3(r.csv('t13_paper_tables/score_component_sensitivity.csv').set_index('variant').loc['MMD only', 'fpr'])} vs "
                   f"{f3(r.csv('t13_paper_tables/score_component_sensitivity.csv').set_index('variant').loc['default no-CIS', 'fpr'])}", "E03 correction"),
        ("Sec. 4.6.5; Discussion", "confirmation window W = 60: F1 and FPR", "0.291; 0.031",
         lambda r: f"{f3(cw(r, 60, 'f1'))}; {f3(cw(r, 60, 'fpr'))}", "E03 correction"),
        ("Discussion", "FPR rise when the inflation selloff is first picked up (W = 30)", "sixfold",
         lambda r: f"{cw(r, 30, 'fpr') / cw(r, 10, 'fpr'):.2f}-fold ({int(cw(r, 30, 'fp'))} vs {int(cw(r, 10, 'fp'))} false-positive days); "
                   f"{cw(r, 60, 'fpr') / cw(r, 10, 'fpr'):.2f}-fold at W = 60", "documentation correction"),
        ("Sec. 4.7", "event-bootstrap F1 difference (submitted design: resample mean)", "-0.091 [-0.166, 0.005]",
         lambda r: "observed and 121-day block: " + u(r, "day-level f1 difference (b - a)"), "uncertainty redesign"),
        ("Sec. 4.7", "event-bootstrap FPR difference (submitted design)", "-0.045 [-0.058, -0.035]",
         lambda r: "observed and 121-day block: " + u(r, "day-level fpr difference (b - a)") + " (sign fixed by nesting)", "uncertainty redesign"),
        ("Sec. 4.7", "20-day block bootstrap F1 and FPR differences (submitted design)", "-0.120 [-0.203, -0.047]; -0.039 [-0.062, -0.019]",
         lambda r: u(r, "day-level f1 difference (b - a)", 20) + "; " + u(r, "day-level fpr difference (b - a)", 20), "uncertainty redesign"),
        ("Sec. 4.1; 4.7; Discussion", "'F1 interval includes zero'; '97.3 per cent of event resamples'", "stated",
         lambda r: "F1 interval " + ("excludes" if all(
             (row["ci_high"] < 0 or row["ci_low"] > 0) for _, row in r.csv("t6_uncertainty/uncertainty_summary.csv").query(
                 "ticker == 'SPY' and estimand == 'day-level f1 difference (b - a)' and comparison.str.startswith('volatility default')", engine="python").iterrows())
             else "does not exclude") + " zero at block lengths 20, 60 and 121", "uncertainty redesign"),
        ("Sec. 4.7 (new interval)", "precision difference, nested default, 121-day block", "no interval",
         lambda r: u(r, "day-level precision difference (b - a)"), "uncertainty redesign"),
        ("Sec. 4.7 (new interval)", "detection-rate difference, nested default", "no interval",
         lambda r: u(r, "event detection rate difference (b - a)", None) + " (sign fixed by nesting)", "uncertainty redesign"),
        ("Sec. 4.7 (new interval)", "common-event delay difference, nested default", "no interval",
         lambda r: u(r, "first-alarm delay difference on common events (b - a), trading days", None) + " (sign fixed by nesting)", "uncertainty redesign"),
    ]
    rows = []
    for location, statement, printed, fn, category in s:
        pre, post = fn(PRE_RUN), fn(POST)
        drivers = []
        if printed != pre:
            drivers.append(category if category != "E03 correction" else "UNEXPECTED difference on the submitted panel")
        if pre != post:
            drivers.append("E03 correction")
        changed = printed != post
        rows.append({"kind": "text statement", "location": location, "row": statement, "column": "", "submitted_value": printed,
                     "pre_E03_fix_value": pre, "revised_value": post, "changed": changed,
                     "drivers": ";".join(dict.fromkeys(drivers)) if changed else "", "source": "computed from frozen outputs"})
    return rows


def documentation_rows() -> list[dict]:
    """Statements whose wording must change although no single number carries the change."""
    op = POST.js("t2_evaluation_periods/operational_start.json")
    cu = POST.js("t7_cusum/cusum_diagnostic.json")
    items = [
        ("Abstract; Contributions 1, 2, 5; Discussion; Conclusion", "ranking reversal presented as a property of the evaluation design",
         "see matched-FPR results (t4) and FROZEN_RESULTS.md", "matched FPR"),
        ("Table 16 caption; Sec. 4.6.4; Limitations", "'so no input uses future data' for the full-period causal variant",
         f"true for the signal path only; fitted constants read prices through {op['latest_calibration_close_date']}; operational claims use "
         f"Table B from {op['operational_start_date']}", "prospective operational evaluation"),
        ("Sec. 4.1; Discussion", "CUSUM 'alarms across much of each window'",
         f"alarms on {cu['post_calibration_alarm_days']} of {cu['post_calibration_days']} post-cutoff days, continuously from "
         f"{cu['persistent_alarm_state_first_day']}; conclusions restricted to this implementation", "documentation correction"),
        ("Sec. 4.4.1; App. B; Contribution 3; Conclusion", "absorption account presented as the cause of the U.S. downgrade miss",
         "holds for the inflation selloff only; see t9_mechanism", "documentation correction"),
        ("Sec. 3.3; Abstract", "'fixed before any detector output is examined'",
         "not supported by the repository history; see t10 development_history_note.md", "documentation correction"),
        ("Table 2 caption; Sec. 3.3", "'Calendar is the documented date'",
         "reference dates now distinguished from underlying event dates; see t11 event_provenance.csv", "documentation correction"),
        ("Sec. 3.4", "'no reported result depends on' the severity groups", "Table 9 depends on them", "documentation correction"),
        ("Tables 8 and 9 captions", "13-event denominators; 'Med. delay' label", "observable denominators; Table 9 delay is a mean of per-asset medians",
         "observable-event denominators"),
        ("Sec. 3.11; Table A2", "bootstrap design: event-union resampling and 20-day blocks", "paired designs of t6; primary block length 121",
         "uncertainty redesign"),
        ("Introduction", "'given a start and an end date'; 'fixed percentile of its own trailing distribution'",
         "single reference date with a +/-60-day window; fixed calibration percentile", "documentation correction"),
    ]
    return [{"kind": "wording", "location": loc, "row": stmt, "column": "", "submitted_value": "stated", "pre_E03_fix_value": "",
             "revised_value": new, "changed": True, "drivers": cat, "source": "see note"} for loc, stmt, new, cat in items]


def new_result_rows() -> list[dict]:
    rows = []
    for key, file in [("Table A", "t2_evaluation_periods/table_A_post_calibration_retrospective.csv"),
                      ("Table B", "t2_evaluation_periods/table_B_prospective_operational.csv")]:
        for _, r in POST.csv(file).iterrows():
            for f in ["day_precision", "day_recall", "day_f1", "day_fpr", "day_alarm_days", "A_out", "events_detected", "n_events", "M", "D_plus",
                      "mean_first_delay", "common_event_n_common", "common_event_common_mean_delay_a", "common_event_common_mean_delay_b"]:
                rows.append({"kind": "new result", "location": key, "row": r["detector"], "column": f, "submitted_value": "", "pre_E03_fix_value": "",
                             "revised_value": r[f], "changed": True, "drivers": "new analysis", "source": f"{file}:{f}"})
    m = pd.concat([POST.csv("t4_operating_curves/matched_comparison_spy.csv"), POST.csv("t4_operating_curves/matched_comparison_other_assets.csv")])
    for _, r in m[(m["match"] == "default") | ((m["target_column"] == "fpr") & (m["match"] == "nearest"))].iterrows():
        for f in ["fpr", "alarm_days", "precision", "recall", "f1", "events_detected", "n_events", "common_events_with_default_confirmation",
                  "common_mean_delay_default_confirmation", "common_mean_delay_this", "A_out", "M", "D_plus"]:
            rows.append({"kind": "new result", "location": "matched FPR", "row": f"{r['ticker']}: {r['operating_point']}", "column": f,
                         "submitted_value": "", "pre_E03_fix_value": "", "revised_value": r[f], "changed": True, "drivers": "new analysis",
                         "source": f"t4_operating_curves:{f}"})
    return rows


# --------------------------------------------------------------------------- integrity
def manifest_check(manifest: Path) -> pd.DataFrame:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        path = ROOT / rel
        if not path.exists():
            rows.append({"file": rel, "status": "missing"})
            continue
        rows.append({"file": rel, "status": "unchanged" if hashlib.sha256(path.read_bytes()).hexdigest() == digest else "modified"})
    return pd.DataFrame(rows)


def repro_check(repro: Path) -> pd.DataFrame:
    untouched = [p for d in ["src", "scripts", "tests", "config"] for p in (repro / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    reference = max(p.stat().st_mtime for p in untouched + [repro / "LICENSE", repro / "README.md"])
    rows = []
    for sub in ["results", "data"]:
        for new in sorted((repro / sub).rglob("*")):
            if not new.is_file() or new.suffix not in {".csv", ".md", ".json"} or "jdsa_revision" in new.parts:
                continue
            rel = new.relative_to(repro).as_posix()
            old = ROOT / rel
            if new.stat().st_mtime <= reference:
                rows.append({"file": rel, "status": "not regenerated by the pipeline"})
            elif not old.exists():
                rows.append({"file": rel, "status": "not in committed results"})
            elif new.read_bytes() == old.read_bytes():
                rows.append({"file": rel, "status": "identical"})
            else:
                rows.append({"file": rel, "status": "DIFFERS"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repro-dir", type=Path)
    args = parser.parse_args()
    tex, source = manuscript()
    audit = pd.DataFrame(audit_event_table(tex) + audit_tables(tex) + text_rows() + documentation_rows() + new_result_rows())
    audit.insert(0, "event_panel", PANEL)
    audit["manuscript_source"] = source
    audit.to_csv(OUT_ROOT / "final_audit.csv", index=False)
    changed = audit[audit["changed"] & audit["kind"].isin(["table cell", "text statement"])]
    changed.to_csv(OUT / "manuscript_numbers_to_change.csv", index=False)
    unexpected = audit[audit["drivers"].astype(str).str.contains("UNEXPECTED")]
    unexpected.to_csv(OUT / "unexpected_differences.csv", index=False)
    print(f"audited {len(audit)} rows; manuscript numbers to change: {len(changed)}; unexpected: {len(unexpected)}")
    print(changed.groupby(["location"]).size().to_string())
    if args.manifest:
        check = manifest_check(args.manifest)
        check.to_csv(OUT / "pre_phase1_files_check.csv", index=False)
        print(check["status"].value_counts().to_string())
        print(check[check["status"] != "unchanged"].to_string(index=False))
    if args.repro_dir:
        rep = repro_check(args.repro_dir)
        rep.to_csv(OUT / "baseline_reproduction_check.csv", index=False)
        print(rep["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
