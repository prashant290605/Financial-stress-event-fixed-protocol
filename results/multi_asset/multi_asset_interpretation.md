# Multi-Asset Interpretation

This summary compares the volatility percentile trigger with the hybrid volatility + instability confirmation rule using the same event-fixed protocol and unchanged detector hyperparameters.

## Does hybrid reduce FPR across assets?

Yes for 5/5 assets: SPY, QQQ, IWM, HYG, TLT.

## Does hybrid reduce recall/F1 across assets?

Recall is lower for 5/5 assets: SPY, QQQ, IWM, HYG, TLT.
F1 is lower for 5/5 assets: SPY, QQQ, IWM, HYG, TLT.

## Does hybrid delay alarms across assets?

Mean first-detection delay is later for 4/5 assets: SPY, QQQ, IWM, TLT.

## Are there assets where the pattern breaks?

No. Every processed asset shows the same broad SPY pattern: the hybrid has lower FPR, lower recall, and lower F1 than the volatility trigger.

## Are HYG/TLT behavior different from equity ETFs?

- HYG: hybrid FPR is lower, recall is lower, F1 is lower, and mean delay is lower than volatility.
- TLT: hybrid FPR is lower, recall is lower, F1 is lower, and mean delay is higher than volatility.

## Volatility vs Hybrid Details

- SPY: volatility FPR=0.044, recall=0.225, F1=0.333, delay=-1.2; hybrid FPR=0.005, recall=0.122, F1=0.214, delay=6.8.
- QQQ: volatility FPR=0.039, recall=0.181, F1=0.280, delay=-19.3; hybrid FPR=0.006, recall=0.076, F1=0.139, delay=10.4.
- IWM: volatility FPR=0.056, recall=0.183, F1=0.272, delay=-0.3; hybrid FPR=0.028, recall=0.093, F1=0.159, delay=14.2.
- HYG: volatility FPR=0.064, recall=0.163, F1=0.246, delay=0.2; hybrid FPR=0.024, recall=0.110, F1=0.188, delay=-1.0.
- TLT: volatility FPR=0.085, recall=0.281, F1=0.362, delay=-25.3; hybrid FPR=0.051, recall=0.094, F1=0.150, delay=-4.8.
