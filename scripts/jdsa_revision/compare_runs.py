"""Every result that differs between the submitted-panel run and the corrected-panel run.

Both runs use the same revision code and observable denominators; the only input that
differs is the event panel (E03 dated 2002-07-24 in the submitted panel, 2002-07-22 in the
corrected one). Every CSV and JSON output present in both runs is compared cell by cell,
so the list is complete for the revision outputs. Rows are matched by position; every
table here has the same rows in both runs because the panel keeps the same 13 events.

    python scripts/jdsa_revision/compare_runs.py results/jdsa_revision/pre_E03_fix results/jdsa_revision

Outputs: <post root>/pre_E03_fix_vs_post_E03_fix.csv (results), ..._curves_and_samples.csv (full operating
curves and raw bootstrap resamples) and ..._summary.csv (per file).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IGNORE_PARTS = {"pre_E03_fix", "t12_audit", "t15_freeze", "t0_event_panel"}  # t0 is the panel itself, listed in panel_diff_vs_submitted.csv
IGNORE_FILES = {"pre_E03_fix_vs_post_E03_fix.csv", "pre_E03_fix_vs_post_E03_fix_summary.csv", "pre_E03_fix_vs_post_E03_fix_curves_and_samples.csv",
                "final_audit.csv", "old_vs_new_all_values.csv", "old_vs_new_changed_values.csv", "index.csv"}
BULK = ("_curve_", "curve_spy", "bootstrap_samples")  # full operating curves and raw resamples go to a separate file


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def row_ids(df: pd.DataFrame) -> list[str]:
    labels = [c for c in df.columns if df[c].dtype == object][:3]
    if not labels:
        return [str(i) for i in range(len(df))]
    return [f"{i}: " + " | ".join(str(df.iloc[i][c]) for c in labels) for i in range(len(df))]


def differs(a, b) -> tuple[bool, float]:
    na, nb = pd.isna(a), pd.isna(b)
    if na and nb:
        return False, np.nan
    if na or nb:
        return True, np.nan
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) > 1e-12, abs(fa - fb)
    except (TypeError, ValueError):
        return str(a) != str(b), np.nan


def compare(pre_root: Path, post_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summary = [], []
    for post in sorted(post_root.rglob("*")):
        rel = post.relative_to(post_root)
        if not post.is_file() or post.suffix not in {".csv", ".json"} or IGNORE_PARTS & set(rel.parts) or post.name in IGNORE_FILES:
            continue
        pre = pre_root / rel
        if not pre.exists():
            summary.append({"file": rel.as_posix(), "status": "only in post run", "cells_compared": 0, "cells_changed": 0})
            continue
        n = changed = 0
        if post.suffix == ".json":
            a, b = flatten(json.loads(pre.read_text(encoding="utf-8"))), flatten(json.loads(post.read_text(encoding="utf-8")))
            for key in sorted(set(a) | set(b)):
                n += 1
                d, diff = differs(a.get(key), b.get(key))
                if d:
                    changed += 1
                    rows.append({"file": rel.as_posix(), "row": "", "column": key, "pre_E03_fix": a.get(key), "post_E03_fix": b.get(key), "abs_diff": diff})
        else:
            a, b = pd.read_csv(pre, low_memory=False), pd.read_csv(post, low_memory=False)
            if list(a.columns) != list(b.columns):
                summary.append({"file": rel.as_posix(), "status": f"columns differ {a.shape} vs {b.shape}", "cells_compared": 0, "cells_changed": 0})
                continue
            if len(a) != len(b):  # rows keyed by their label columns; added and removed rows are listed
                keys = [c for c in b.columns if b[c].dtype == object]
                ka, kb = a[keys].astype(str).agg(" | ".join, axis=1), b[keys].astype(str).agg(" | ".join, axis=1)
                for k in sorted(set(ka) - set(kb)):
                    rows.append({"file": rel.as_posix(), "row": k, "column": "(row)", "pre_E03_fix": "present", "post_E03_fix": "absent", "abs_diff": np.nan})
                for k in sorted(set(kb) - set(ka)):
                    rows.append({"file": rel.as_posix(), "row": k, "column": "(row)", "pre_E03_fix": "absent", "post_E03_fix": "present", "abs_diff": np.nan})
                common = sorted(set(ka) & set(kb))
                a = a[ka.isin(common)].set_index(ka[ka.isin(common)]).loc[common].reset_index(drop=True)
                b = b[kb.isin(common)].set_index(kb[kb.isin(common)]).loc[common].reset_index(drop=True)
                changed += len(set(ka) ^ set(kb))
            ids = row_ids(b)
            av, bv = a.to_numpy(object), b.to_numpy(object)
            for i in range(len(b)):
                for j, col in enumerate(b.columns):
                    n += 1
                    d, diff = differs(av[i, j], bv[i, j])
                    if d:
                        changed += 1
                        rows.append({"file": rel.as_posix(), "row": ids[i], "column": col, "pre_E03_fix": av[i, j], "post_E03_fix": bv[i, j], "abs_diff": diff})
        summary.append({"file": rel.as_posix(), "status": "compared", "cells_compared": n, "cells_changed": changed})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def main() -> None:
    pre_root, post_root = (Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:3])
    diffs, summary = compare(pre_root, post_root)
    bulk = diffs["file"].str.contains("|".join(BULK)) if len(diffs) else pd.Series(dtype=bool)
    diffs[bulk].to_csv(post_root / "pre_E03_fix_vs_post_E03_fix_curves_and_samples.csv", index=False)
    diffs = diffs[~bulk]
    diffs.to_csv(post_root / "pre_E03_fix_vs_post_E03_fix.csv", index=False)
    summary.to_csv(post_root / "pre_E03_fix_vs_post_E03_fix_summary.csv", index=False)
    print(f"files compared: {(summary['status'] == 'compared').sum()}; result values changed: {len(diffs)}; curve and resample values changed: {int(bulk.sum())}")
    print(summary[summary["cells_changed"] > 0].to_string(index=False))
    print(summary[summary["status"] != "compared"].to_string(index=False))


if __name__ == "__main__":
    main()
