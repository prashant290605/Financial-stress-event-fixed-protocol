# Repository for Paper

## Title

A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors

## Repository

GitHub URL: https://github.com/prashant290605/Financial-stress-event-fixed-protocol

## Description

This repository contains the code, configuration, event definitions, data-preparation workflow, generated result tables, and figure-generation scripts required to reproduce the analyses reported in the paper.

## Reproducibility Instructions

```bash
git clone https://github.com/prashant290605/Financial-stress-event-fixed-protocol.git
cd Financial-stress-event-fixed-protocol
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
mkdir -p external
git clone https://github.com/numenta/NAB.git external/NAB
python scripts/prepare_multi_asset_data.py
python scripts/reproduce_paper.py
python scripts/evaluate_multi_asset.py
python scripts/evaluate_event_level.py
python scripts/run_cost_analysis.py
python scripts/generate_final_figures.py
```

The main outputs are written to `results/` and `figures/final/`.
