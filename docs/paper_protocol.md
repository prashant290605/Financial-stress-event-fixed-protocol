# Paper protocol and metric definitions

## Data source and preprocessing

The paper experiment uses the local Yahoo Finance export at `Finance Data/spy.csv` and the **adjusted close** column (`Adj Close`). The deterministic preprocessing pipeline is:

1. `log_return_t = log(price_t / price_{t-1})`
2. `rolling_volatility_t = std(log_return_{t-19:t})`
3. `return_zscore_t = (log_return_t - mean(log_return_{t-49:t})) / std(log_return_{t-49:t})`
4. `signal_t = 0.5 * log_return_t + 0.3 * rolling_volatility_t + 0.2 * return_zscore_t`

The processed series is written to `data/processed/spy_processed.csv` with a date column. Fixed calendar events are mapped to the nearest available trading date. Missing NYSE holidays are not treated as errors; the generated `results/paper/spy_missing_business_days.csv` records missing business dates so that the mapping step is explicit.

## Event protocol

The seed event list is `data/events_spy_seed.csv`. The generated machine-readable event table is `data/events_spy.csv` and has exactly these columns: `event_id`, `event_name`, `calendar_date`, `mapped_trading_date`, `event_type`, `inclusion_rule`, `source_reference`, `severity_score`, `severity_label`, `severity_rule`. Severity is objective in the paper pipeline:

- `severity_score`: absolute maximum drawdown within the mapped event date ±20 trading days
- `critical`: top tercile of severity scores
- `high`: middle tercile
- `medium`: bottom tercile

## Baseline definitions

All thresholds are calibrated on the first 70% of each series and then reused unchanged on the remainder.

| Method | Score formula | Threshold rule | Hyperparameters | Online/offline | Output labels |
| --- | --- | --- | --- | --- | --- |
| Volatility percentile trigger | smoothed z-score of 20-day rolling volatility | > 90th percentile of calibration period | vol window 20, smoothing 5 | offline calibration, online-compatible scoring | binary alarm |
| CUSUM | max positive/negative cumulative sum of return z-scores | smoothed score > 90th percentile of calibration period | smoothing 5 | offline calibration, online-compatible scoring | binary alarm |
| Instability score | `0.5*MMD_z + 0.5*|kappa_z|` | smoothed score > 90th percentile of calibration period | AR window 36, MMD window 30, smoothing 5 | offline calibration, online-compatible scoring | binary alarm |
| MMD-only confirmation | `MMD_z` | same score thresholding rule | MMD window 30, smoothing 5 | offline calibration, online-compatible scoring | binary alarm |
| Kappa-only confirmation | `|kappa_z|` | same score thresholding rule | AR window 36, smoothing 5 | offline calibration, online-compatible scoring | binary alarm |
| Hybrid windowed confirmation | volatility alarm AND at least one instability alarm within ±W days | conjunction rule, default `W=10` | W ∈ {0,5,10,20,30,45,60} | offline calibration, online-compatible scoring if right-looking window is replaced by delayed confirmation | binary alarm |

CIS is computed only as a diagnostic feature in the default decision score. It appears in the ablation table but is not part of the default detector.

## Metric definitions

- **Pointwise precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1**: harmonic mean of precision and recall
- **False positive rate**: `FP / (FP + TN)`
- **Event coverage**: fraction of event windows containing at least one detection
- **Event delay**: signed delay from event center to the first detection inside that event window; negative means early detection, positive means late detection
- **Event alignment error**: absolute distance from event center to the nearest detection inside the event window

If an event window has no detection, it contributes `0` to event coverage and `NaN` to delay/alignment before averaging over detected events only. Overlapping event windows are merged for pointwise confusion-matrix metrics through their union mask; event-level metrics are still computed per event window before averaging.

## Bootstrap uncertainty

The paper reports two uncertainty schemes:

1. **Event bootstrap**: resample the fixed event rows with replacement, rebuild the union event mask, and recompute hybrid-minus-volatility metric differences.
2. **Block bootstrap**: resample contiguous 20-day time blocks with replacement over the full binary sequences, preserving local dependence and resampling non-event days as well.

Because event bootstrap leaves the non-event timeline fixed, its FPR interval can be too narrow. The block bootstrap is included to expose that sensitivity directly.

## NAB note

NAB is an external sanity check only. The NAB path evaluates all 58 official series from `combined_windows.json`, converts official anomaly windows into binary masks, reuses thresholds without tuning, and reports ordinary F1/FPR rather than official NAB scores. For runtime tractability on the full corpus, the NAB instability confirmation uses a deterministic rolling mean-difference proxy for the MMD component rather than the quadratic exact-MMD implementation used in the SPY experiment. Therefore NAB results are supportive context, not a like-for-like finance benchmark.
