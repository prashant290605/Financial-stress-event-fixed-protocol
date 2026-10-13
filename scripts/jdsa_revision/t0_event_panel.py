"""Task 0: the event panel used by this run, and how it differs from the submitted panel.

The panel is a versioned input (``src/event_panels.py``). This script writes the active
panel's SPY table (with drawdown severity) and its per-asset mapping, and lists every
field that differs from the submitted panel. With the default panel the changes are the E03
reference date, label and source reference, and the inclusion-rule wording of every event, which no
longer states that the panel was fixed before detector evaluation
(see ``data/jdsa_revision/events_spy_seed_v2.csv``).

Outputs: <out root>/t0_event_panel/
"""
from __future__ import annotations

import pandas as pd

from _common import CONFIG, PANEL, TICKERS, outdir, panel_events, spy_events, write_json  # noqa: E402
from src.event_panels import PANELS, spy_event_table  # noqa: E402
from src.multi_asset_evaluation import _load_processed_asset  # noqa: E402
from src.observability import observable_event_flags  # noqa: E402

OUT = outdir("t0_event_panel")


def main() -> None:
    active = spy_events()
    active.to_csv(OUT / "events_spy_active_panel.csv", index=False)
    mapped = pd.concat([panel_events(t).assign(observable=observable_event_flags(_load_processed_asset(t), panel_events(t)).to_numpy())
                        for t in TICKERS], ignore_index=True)
    mapped.to_csv(OUT / "events_multi_asset_active_panel.csv", index=False)

    submitted = spy_event_table("submitted", _load_processed_asset("SPY"), CONFIG).set_index("event_id")
    rows = []
    for eid, row in active.set_index("event_id").iterrows():
        for field in ["event_name", "calendar_date", "mapped_trading_date", "source_reference", "severity_score", "severity_label"]:
            old, new = submitted.loc[eid, field], row[field]
            if (abs(old - new) > 1e-12) if isinstance(old, float) else old != new:
                rows.append({"event_id": eid, "field": field, "submitted_panel": old, "active_panel": new})
    diff = pd.DataFrame(rows, columns=["event_id", "field", "submitted_panel", "active_panel"])
    diff.to_csv(OUT / "panel_diff_vs_submitted.csv", index=False)
    write_json(OUT / "panel_version.json", {
        "event_panel": PANEL, "seed_file": PANELS[PANEL].relative_to(PANELS["submitted"].parents[1]).as_posix(),
        "events": len(active), "events_changed": sorted(diff["event_id"].unique().tolist()),
        "severity_labels_changed": diff.loc[diff["field"] == "severity_label", "event_id"].tolist(),
        "revision_notes": {r.event_id: r.revision_note for r in active.itertuples() if getattr(r, "revision_note", "")},
    })
    print(f"panel {PANEL}: {len(active)} events")
    print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
