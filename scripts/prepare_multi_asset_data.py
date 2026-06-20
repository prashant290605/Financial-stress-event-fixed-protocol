from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_asset_data import prepare_multi_asset_data


def main() -> None:
    result = prepare_multi_asset_data()
    processed = ", ".join(result["processed"])
    print(f"Processed assets: {processed}")
    if result["skipped"]:
        skipped = ", ".join(f"{row['ticker']} ({row['reason']})" for row in result["skipped"])
        print(f"Skipped optional assets: {skipped}")
    print("Wrote data/events_multi_asset.csv")
    print("Wrote results/multi_asset/data_availability.csv")


if __name__ == "__main__":
    main()
