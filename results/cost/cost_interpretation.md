# Cost Interpretation

The original pointwise cost is retained: `c_FP * FP + c_FN * FN` with `c_FN = 1`. This cost can favor the hybrid because non-event days dominate pointwise masks, so reducing false positives can outweigh missed event-window days.

The event-aware cost adds deployment-facing terms: false-alarm days outside event windows, missed events, and positive first-detection delay. This makes missed events and late confirmation visible instead of letting the large number of non-event days dominate the conclusion.

## SPY Preferences

Pointwise preferred methods by FP/FN ratio:
- 1: Hybrid volatility + instability confirmation
- 2: Hybrid volatility + instability confirmation
- 5: Hybrid volatility + instability confirmation
- 10: Hybrid volatility + instability confirmation
- 20: Hybrid volatility + instability confirmation
- 50: Hybrid volatility + instability confirmation

Event-aware preferred methods for delay cost 0.05 by false-alarm/missed-event ratio:
- 0.01: Volatility percentile
- 0.05: Hybrid volatility + instability confirmation
- 0.1: Hybrid volatility + instability confirmation
- 0.25: Hybrid volatility + instability confirmation
- 0.5: Hybrid volatility + instability confirmation
- 1.0: Hybrid volatility + instability confirmation

## Multi-Asset Event-Aware Pattern

Under event-aware cost with delay cost 0.05, volatility is preferred in at least one cost regime for: SPY, QQQ, IWM, HYG, TLT.

## Deployment Implication

The correct detector depends on operational cost. If false alarm days are expensive and missed events are tolerable, the hybrid can be attractive. If missed events or late confirmation matter, volatility can be preferable even though it is noisier. This supports the paper's tradeoff framing rather than a claim of universal hybrid superiority.
