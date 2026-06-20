from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.event_level_evaluation import run_event_level_evaluation


def main() -> None:
    detail = run_event_level_evaluation()
    print(f"Evaluated event-level rows: {len(detail)}")
    print("Wrote results/event_level/event_level_metrics.csv")
    print("Wrote results/event_level/event_level_summary_by_asset.csv")
    print("Wrote results/event_level/event_level_summary_by_severity.csv")
    print("Wrote results/event_level/table_spy_event_level_detection.csv")
    print("Wrote results/event_level/table_event_level_severity_summary.csv")
    print("Wrote results/event_level/table_multi_asset_event_level_summary.csv")
    print("Wrote results/event_level/event_level_delay_coverage.png")
    print("Wrote results/event_level/event_level_interpretation.md")


if __name__ == "__main__":
    main()
