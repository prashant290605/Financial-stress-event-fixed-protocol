# Event-Level Interpretation

This event-level analysis asks whether each fixed stress event is detected at least once inside the existing +/-60 trading-day event window, and how early, late, or densely each method fires.

## Does hybrid detect fewer events than volatility?

Yes for 5/5 assets: SPY, QQQ, IWM, HYG, TLT.

## Does hybrid detect severe events more reliably than medium events?

Hybrid severity-stratified averages across processed assets:
- critical: detection rate 0.680, coverage 0.176
- high: detection rate 0.300, coverage 0.066
- medium: detection rate 0.100, coverage 0.011

## Does hybrid tend to detect after the event center?

Hybrid median first-detection delay is later than volatility for 4/5 assets: SPY, QQQ, IWM, TLT. Positive delay means the first in-window alarm occurs after the mapped event date.

## Does volatility provide earlier but noisier coverage?

Volatility has higher mean event-window coverage than hybrid for 5/5 assets: SPY, QQQ, IWM, HYG, TLT. This is consistent with broader, noisier event-window firing.

## What does this mean for deployment?

The hybrid is better treated as a confirmed-stress filter than a broad early-warning system. It reduces alarm burden and event-window coverage, but that conservatism can miss entire events or move the first usable signal later. A deployment that needs high recall or early monitoring should not use the hybrid alone; a two-stage workflow is more defensible, with volatility-like triggers for surveillance and hybrid confirmation for escalation.

## Asset-Level Volatility vs Hybrid Summary

- SPY: volatility detection=0.769, median delay=3.0, coverage=0.226; hybrid detection=0.615, median delay=5.0, coverage=0.121.
- QQQ: volatility detection=0.462, median delay=-11.5, coverage=0.198; hybrid detection=0.385, median delay=2.0, coverage=0.086.
- IWM: volatility detection=0.538, median delay=1.0, coverage=0.176; hybrid detection=0.385, median delay=5.0, coverage=0.090.
- HYG: volatility detection=0.385, median delay=2.0, coverage=0.123; hybrid detection=0.231, median delay=-3.0, coverage=0.083.
- TLT: volatility detection=0.538, median delay=-4.0, coverage=0.227; hybrid detection=0.308, median delay=-2.5, coverage=0.076.
