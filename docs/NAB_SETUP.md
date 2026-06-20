# NAB Setup

The paper uses NAB only as an auxiliary qualitative validation setting. Official NAB scores are not used. The workflow converts NAB anomaly windows into binary masks and evaluates the volatility and hybrid outputs with the same pointwise precision, recall, F1, and FPR metrics used in the financial-stress protocol.

## Required Location

Place the NAB repository at:

```text
external/NAB/
```

The reproduction code expects this file to exist:

```text
external/NAB/labels/combined_windows.json
```

and expects the corresponding NAB data files under:

```text
external/NAB/data/
```

## How to Obtain NAB

From the repository root, run:

```bash
mkdir -p external
git clone https://github.com/numenta/NAB.git external/NAB
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force external
git clone https://github.com/numenta/NAB.git external/NAB
```

If `external/NAB` is already present in the repository checkout, no additional setup is required.

## How the Workflow Uses NAB

The main reproduction script calls the NAB auxiliary validation through:

```bash
python scripts/reproduce_paper.py
```

The NAB outputs are written to:

```text
results/paper/nab_series_results.csv
results/paper/table_nab_subset_results.csv
results/paper/table_nab_methods.csv
figures/final/nab_subset_results.*
```

If NAB is unavailable, the NAB-specific part of the workflow cannot be regenerated. The financial-stress protocol, SPY analysis, multi-asset validation, event-level analysis, cost analysis, and non-NAB figures do not depend on NAB.
