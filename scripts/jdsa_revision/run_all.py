"""Run every JDSA revision analysis in dependency order.

t3 reads the operational start from t2; t5, t6 and t10 read the matched thresholds and
curves written by t4; t12 reads the outputs of all others. t3 (about 250 recomputations
from the raw price file) and t4 (5001-point curves on five assets) take several minutes.

    python scripts/jdsa_revision/run_all.py                    # corrected panel -> results/jdsa_revision
    python scripts/jdsa_revision/run_all.py --skip t3          # skip the slow perturbation audit
    python scripts/jdsa_revision/run_all.py --panel submitted --out results/jdsa_revision/pre_E03_fix --skip t3 t12

The revision pipeline always scores observable events only; there is no option to do
otherwise. --panel and --out are passed to every script through JDSA_PANEL and
JDSA_OUT_ROOT.

t12 also reads the run on the submitted panel (results/jdsa_revision/pre_E03_fix), which is
regenerated rather than committed. On a fresh clone, run the third command above first; if that
run is missing, t12 is skipped with a notice.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = [
    "t0_event_panel", "t1_observable_denominators", "t2_evaluation_periods", "t3_causality", "t4_operating_curves",
    "t5_common_event_delay", "t6_uncertainty", "t7_cusum_diagnostic", "t8_z_dominance", "t9_mechanism", "t10_parameters",
    "t11_event_provenance", "t13_paper_tables", "t14_figures", "t12_audit",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None, help="event panel (default jdsa_revision_v2)")
    parser.add_argument("--out", default=None, help="output root (default results/jdsa_revision)")
    parser.add_argument("--skip", nargs="*", default=[], help="task prefixes to skip, e.g. t3 t12")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.panel:
        env["JDSA_PANEL"] = args.panel
    if args.out:
        env["JDSA_OUT_ROOT"] = args.out
    out_root = Path(env.get("JDSA_OUT_ROOT", HERE.parents[1] / "results" / "jdsa_revision"))
    if not out_root.is_absolute():
        out_root = HERE.parents[1] / out_root
    for name in ORDER:
        if name.split("_")[0] in args.skip:
            print(f"skipping {name}")
            continue
        if name == "t12_audit" and not (out_root / "pre_E03_fix").is_dir():
            # t12 compares against the run on the submitted panel, which is regenerated, not committed.
            print("skipping t12_audit: the comparison run on the submitted panel is missing. Run\n"
                  "  python scripts/jdsa_revision/run_all.py --panel submitted --out results/jdsa_revision/pre_E03_fix --skip t3 t12\n"
                  "and then python scripts/jdsa_revision/t12_audit.py", flush=True)
            continue
        print(f"=== {name}", flush=True)
        result = subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=HERE.parents[1], env=env)
        if result.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
