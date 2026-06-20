from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_analysis import run_cost_analysis


def main() -> None:
    pointwise, event_aware = run_cost_analysis()
    print(f"Wrote pointwise cost rows: {len(pointwise)}")
    print(f"Wrote event-aware cost rows: {len(event_aware)}")
    print("Wrote results/cost/pointwise_cost_by_asset.csv")
    print("Wrote results/cost/event_aware_cost_by_asset.csv")
    print("Wrote results/cost/table_spy_pointwise_cost.csv")
    print("Wrote results/cost/table_spy_event_aware_cost.csv")
    print("Wrote results/cost/table_multi_asset_event_aware_cost_summary.csv")
    print("Wrote results/cost/cost_comparison.png")
    print("Wrote results/cost/cost_interpretation.md")


if __name__ == "__main__":
    main()
