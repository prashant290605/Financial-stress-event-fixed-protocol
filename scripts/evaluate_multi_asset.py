from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_asset_evaluation import run_multi_asset_evaluation


def main() -> None:
    metrics = run_multi_asset_evaluation()
    tickers = ", ".join(metrics["ticker"].drop_duplicates())
    methods = ", ".join(metrics["method"].drop_duplicates())
    print(f"Evaluated assets: {tickers}")
    print(f"Evaluated methods: {methods}")
    print("Wrote results/multi_asset/multi_asset_pointwise_metrics.csv")
    print("Wrote results/multi_asset/table_multi_asset_event_fixed_evaluation.csv")
    print("Wrote results/multi_asset/multi_asset_tradeoff_comparison.png")
    print("Wrote results/multi_asset/multi_asset_interpretation.md")


if __name__ == "__main__":
    main()
