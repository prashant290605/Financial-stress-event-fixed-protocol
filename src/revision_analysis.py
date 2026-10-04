"""Evaluation-side analyses added for the JDSA major revision.

Nothing in this module changes a submitted detector. The volatility trigger, the
instability score, the confirmation rule, the CUSUM baseline, the 70 per cent
calibration split and the 90th-percentile thresholds are computed exactly as in
``src.paper_protocol``; the regression tests check that the default alarms here are
identical to ``detector_outputs``. What is added is the evaluation around them:
post-calibration scoring, the fully causal confirmation rule as a reusable function,
operating curves, common-event delays, paired bootstraps, and diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.paper_protocol import (
    _rolling_mean,
    _z_from_train,
    compute_instability_features,
    event_mask,
    point_metrics,
)


# ---------------------------------------------------------------------------
# Calibration cutoff
# ---------------------------------------------------------------------------
def calibration_train_end(n_days: int, config: dict[str, Any]) -> int:
    """Index of the first day after the calibration period, as in ``detector_outputs``."""
    return int(round(n_days * config["detectors"]["calibration_fraction"]))


@dataclass(frozen=True)
class Cutoff:
    train_end: int
    last_calibration_date: pd.Timestamp
    first_evaluation_date: pd.Timestamp


def calibration_cutoff(df: pd.DataFrame, config: dict[str, Any]) -> Cutoff:
    te = calibration_train_end(len(df), config)
    return Cutoff(te, pd.Timestamp(df["date"].iloc[te - 1]), pd.Timestamp(df["date"].iloc[te]))


# ---------------------------------------------------------------------------
# Detector components with explicit quantiles
# ---------------------------------------------------------------------------
def _z_with_constants(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """``_z_from_train`` with the mean and standard deviation supplied instead of fitted."""
    sd = sd if sd > 1e-12 else 1.0
    return np.nan_to_num((values - mean) / sd, nan=0.0, posinf=6.0, neginf=-6.0)


def component_scores(
    df: pd.DataFrame,
    config: dict[str, Any],
    train_end: int | None = None,
    kappa_mode: str = "abs",
    constants: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Continuous scores behind the submitted detectors.

    ``volatility`` is the smoothed calibration-standardized 20-day volatility and
    ``instability`` the smoothed score 0.5 z_MMD + 0.5 |z_kappa|. ``train_end`` can be
    fixed explicitly so that a perturbed copy of the data is calibrated on the same
    days as the original. ``constants`` freezes the standardization constants at given
    values (the ``calibration_constants`` of an earlier call) instead of refitting them.
    """
    te = calibration_train_end(len(df), config) if train_end is None else int(train_end)
    feat = compute_instability_features(df["signal"].to_numpy(float), config)
    mmd = feat["MMD"].to_numpy(float)
    kappa = feat["kappa"].to_numpy(float)
    vol = df["rolling_volatility"].to_numpy(float)
    fitted = {
        "mmd_mean": float(np.nanmean(mmd[:te])),
        "mmd_sd": float(np.nanstd(mmd[:te])),
        "kappa_mean": float(np.nanmean(kappa[:te])),
        "kappa_sd": float(np.nanstd(kappa[:te])),
        "volatility_mean": float(np.nanmean(vol[:te])),
        "volatility_sd": float(np.nanstd(vol[:te])),
    }
    c = fitted if constants is None else constants
    mmd_z = _z_with_constants(mmd, c["mmd_mean"], c["mmd_sd"])
    kappa_z = _z_with_constants(kappa, c["kappa_mean"], c["kappa_sd"])
    if kappa_mode == "abs":
        kappa_term = np.abs(kappa_z)
    elif kappa_mode == "low":
        kappa_term = np.maximum(-kappa_z, 0.0)
    elif kappa_mode == "high":
        kappa_term = np.maximum(kappa_z, 0.0)
    else:
        raise ValueError(f"unknown kappa_mode: {kappa_mode}")
    smooth = config["features"]["smooth_window"]
    return {
        "train_end": te,
        "volatility": _rolling_mean(_z_with_constants(vol, c["volatility_mean"], c["volatility_sd"]), smooth),
        "instability": _rolling_mean(0.5 * mmd_z + 0.5 * kappa_term, smooth),
        "mmd_z": mmd_z,
        "kappa_z": kappa_z,
        "mmd_z_smoothed": _rolling_mean(mmd_z, smooth),
        "kappa_abs_smoothed": _rolling_mean(np.abs(kappa_z), smooth),
        "features": feat,
        "calibration_constants": fitted,
    }


def quantile_threshold(score: np.ndarray, train_end: int, q: float) -> float:
    return float(np.quantile(score[:train_end], q))


def indicator(score: np.ndarray, train_end: int, q: float) -> np.ndarray:
    return (score > quantile_threshold(score, train_end, q)).astype(int)


def window_any(flags: np.ndarray, back: int, forward: int) -> np.ndarray:
    """out[t] = 1 when any flag is set in [t-back, t+forward], clipped to the series."""
    flags = np.asarray(flags, int)
    n = len(flags)
    csum = np.concatenate([[0], np.cumsum(flags)])
    t = np.arange(n)
    lo = np.maximum(0, t - back)
    hi = np.minimum(n, t + forward + 1)
    return ((csum[hi] - csum[lo]) > 0).astype(int)


def symmetric_confirmation(volatility: np.ndarray, instability: np.ndarray, window: int) -> np.ndarray:
    """Equation (13): v(t) and any i(tau) for tau in [t-W, t+W]. Same as ``windowed_hybrid``."""
    return (np.asarray(volatility, int) & window_any(instability, window, window)).astype(int)


def trailing_confirmation(volatility: np.ndarray, instability: np.ndarray, window: int) -> np.ndarray:
    """v(t) and any i(tau) for tau in [t-W, t], as ``trailing_hybrid`` in run_ablations.py."""
    return (np.asarray(volatility, int) & window_any(instability, window, 0)).astype(int)


def lag_indicator(flags: np.ndarray, lag: int) -> np.ndarray:
    """Shift an indicator ``lag`` days later; the first ``lag`` days are zero."""
    flags = np.asarray(flags, int)
    out = np.zeros_like(flags)
    if lag == 0:
        return flags.copy()
    out[lag:] = flags[:-lag]
    return out


def detector_alarms(
    scores: dict[str, Any],
    config: dict[str, Any],
    q_volatility: float | None = None,
    q_instability: float | None = None,
    window: int | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Alarms of the volatility trigger and the three confirmation variants.

    The fully causal variant lags the instability indicator by the MMD second-sample
    length, exactly as in scripts/run_ablations.py (Table 16). ``thresholds`` freezes
    the two percentile thresholds at given values instead of refitting them.
    """
    det = config["detectors"]
    qv = det["volatility_threshold_quantile"] if q_volatility is None else q_volatility
    qi = det["score_threshold_quantile"] if q_instability is None else q_instability
    w = det["default_hybrid_window"] if window is None else window
    te = scores["train_end"]
    thr = thresholds or {
        "volatility": quantile_threshold(scores["volatility"], te, qv),
        "instability": quantile_threshold(scores["instability"], te, qi),
    }
    vol = (scores["volatility"] > thr["volatility"]).astype(int)
    inst = (scores["instability"] > thr["instability"]).astype(int)
    inst_lagged = lag_indicator(inst, config["features"]["mmd_window"])
    return {
        "volatility": vol,
        "instability": inst,
        "instability_lagged": inst_lagged,
        "retrospective_confirmation": symmetric_confirmation(vol, inst, w),
        "trailing_confirmation": trailing_confirmation(vol, inst, w),
        "fully_causal_confirmation": trailing_confirmation(vol, inst_lagged, w),
        "thresholds": dict(thr),
        "settings": {"q_volatility": qv, "q_instability": qi, "window": w},
    }


def pipeline_from_prices(
    prices: pd.DataFrame,
    config: dict[str, Any],
    train_end: int,
    constants: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Raw adjusted closes to alarms, with no file written.

    Preprocessing is ``preprocess_price_frame`` (the same arithmetic as
    ``preprocess_spy``). ``train_end`` is fixed by the caller so that a perturbed copy is
    calibrated on the same days; ``constants`` and ``thresholds`` optionally freeze the
    fitted calibration quantities of an earlier run.
    """
    from src.multi_asset_data import preprocess_price_frame

    df = preprocess_price_frame(prices, config)
    scores = component_scores(df, config, train_end=train_end, constants=constants)
    return {"df": df, "scores": scores, "alarms": detector_alarms(scores, config, thresholds=thresholds)}


def calibration_quantities(out: dict[str, Any]) -> dict[str, float]:
    """Every fitted quantity the volatility trigger and the confirmation rules use."""
    return {**out["scores"]["calibration_constants"], **{f"threshold_{k}": v for k, v in out["alarms"]["thresholds"].items()}}


def _perturb_from(prices: pd.DataFrame, date: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Copy of ``prices`` with every close dated on or after ``date`` multiplied by random noise."""
    p = prices.copy()
    sel = (p["date"] >= date).to_numpy()
    p.loc[sel, "price"] = p.loc[sel, "price"].to_numpy(float) * np.exp(np.random.default_rng(seed).normal(0.0, 0.05, int(sel.sum())))
    return p


def derive_operational_start(prices: pd.DataFrame, config: dict[str, Any], train_end: int, seed: int = 0) -> dict[str, Any]:
    """Latest raw close read by any calibration quantity, and the first day after it.

    ``moved(k)`` asks whether perturbing every raw close dated on or after processed day k
    changes any fitted constant or threshold. It is true for early k and false once k is
    past every close the calibration reads, so a binary search over k finds the first day
    k* for which it is false. The latest close used is day k*-1 and the operational
    evaluation can start on day k*. Both ends are then checked directly: perturbing all
    closes from day k* on leaves every calibration quantity bit-identical, and perturbing
    the single close of day k*-1 changes at least one.
    """
    base = pipeline_from_prices(prices, config, train_end)
    reference = calibration_quantities(base)
    dates = base["df"]["date"].reset_index(drop=True)

    def moved(k: int, p: pd.DataFrame | None = None) -> bool:
        pert = p if p is not None else _perturb_from(prices, dates.iloc[k], seed + k)
        return calibration_quantities(pipeline_from_prices(pert, config, train_end)) != reference

    lo, hi = max(0, train_end - 10), len(dates) - 1  # moved(lo) must be true, moved(hi) false
    log = [{"k": lo, "date": dates.iloc[lo].date().isoformat(), "moved": moved(lo)},
           {"k": hi, "date": dates.iloc[hi].date().isoformat(), "moved": moved(hi)}]
    if not log[0]["moved"] or log[1]["moved"]:
        raise RuntimeError("calibration horizon is not bracketed by the search range")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        m = moved(mid)
        log.append({"k": mid, "date": dates.iloc[mid].date().isoformat(), "moved": m})
        lo, hi = (mid, hi) if m else (lo, mid)
    start = hi
    single = prices.copy()
    raw = int(single.index[single["date"] == dates.iloc[start - 1]][0])
    single.loc[raw, "price"] *= 1.05
    after = pipeline_from_prices(_perturb_from(prices, dates.iloc[start], seed + 999_999), config, train_end)
    return {
        "train_end_index": train_end,
        "last_calibration_date": dates.iloc[train_end - 1],
        "latest_calibration_close_index": start - 1,
        "latest_calibration_close_date": dates.iloc[start - 1],
        "operational_start_index": start,
        "operational_start_date": dates.iloc[start],
        "check_all_closes_from_start_perturbed_leaves_calibration_identical": calibration_quantities(after) == reference,
        "check_single_close_before_start_moves_calibration": moved(start - 1, single),
        "calibration_quantities": reference,
        "search_log": log,
    }


# ---------------------------------------------------------------------------
# Event-level and day-level evaluation
# ---------------------------------------------------------------------------
def event_spans(df: pd.DataFrame, events: pd.DataFrame, half_window: int) -> pd.DataFrame:
    """One row per event with its centre and clamped window, as built by ``event_mask``."""
    _, spans = event_mask(df, events, half_window)
    out = pd.DataFrame(spans, columns=["center", "lo", "hi"])
    out.insert(0, "event_id", events["event_id"].to_numpy())
    return out


def per_event_outcomes(pred: np.ndarray, spans: pd.DataFrame) -> pd.DataFrame:
    """Detection, first and nearest alarm delay, and coverage for each event window."""
    rows = []
    for row in spans.itertuples(index=False):
        idx = np.flatnonzero(np.asarray(pred[row.lo : row.hi + 1]) == 1) + row.lo
        detected = len(idx) > 0
        rows.append(
            {
                "event_id": row.event_id,
                "center": row.center,
                "lo": row.lo,
                "hi": row.hi,
                "detected": detected,
                "first_delay": float(idx[0] - row.center) if detected else math.nan,
                "nearest_delay": float(idx[np.argmin(np.abs(idx - row.center))] - row.center) if detected else math.nan,
                "coverage": len(idx) / (row.hi - row.lo + 1),
                "alarm_days": int(len(idx)),
                "first_alarm_at_window_start": bool(detected and idx[0] == row.lo),
            }
        )
    return pd.DataFrame(rows)


def event_summary(per_event: pd.DataFrame) -> dict[str, float]:
    det = per_event["detected"].astype(bool)
    delays = per_event.loc[det, "first_delay"]
    nearest = per_event.loc[det, "nearest_delay"]
    return {
        "n_events": int(len(per_event)),
        "events_detected": int(det.sum()),
        "detection_rate": float(det.mean()) if len(per_event) else math.nan,
        "mean_first_delay": float(delays.mean()) if det.any() else math.nan,
        "median_first_delay": float(delays.median()) if det.any() else math.nan,
        "mean_coverage": float(per_event["coverage"].mean()) if len(per_event) else math.nan,
        "alignment_error": float(nearest.abs().mean()) if det.any() else math.nan,
    }


def cost_components(y: np.ndarray, pred: np.ndarray, per_event: pd.DataFrame, start: int = 0) -> dict[str, float]:
    """A_out, M and D_+ of Equation (eventcost), with A_out counted from day ``start``."""
    y = np.asarray(y, int)[start:]
    p = np.asarray(pred, int)[start:]
    det = per_event["detected"].astype(bool)
    return {
        "A_out": int(np.sum((p == 1) & (y == 0))),
        "M": int((~det).sum()),
        "D_plus": float(np.maximum(per_event.loc[det, "first_delay"].to_numpy(float), 0.0).sum()),
    }


def event_cost(components: dict[str, float], ratio: float, delay_cost: float) -> float:
    return ratio * components["A_out"] + components["M"] + delay_cost * components["D_plus"]


def crossover_ratio(reference: dict[str, float], alternative: dict[str, float], delay_cost: float) -> float:
    """Equation (crossover) with the volatility trigger as ``reference``."""
    denominator = reference["A_out"] - alternative["A_out"]
    numerator = (alternative["M"] - reference["M"]) + delay_cost * (alternative["D_plus"] - reference["D_plus"])
    return float(numerator / denominator) if denominator else math.nan


def day_metrics(y: np.ndarray, pred: np.ndarray, start: int = 0) -> dict[str, float]:
    y = np.asarray(y, int)[start:]
    p = np.asarray(pred, int)[start:]
    out = point_metrics(y, p)
    out["alarm_days"] = int(p.sum())
    out["n_days"] = int(len(y))
    out["n_event_days"] = int(y.sum())
    out["n_non_event_days"] = int(len(y) - y.sum())
    return out


def split_by_cutoff(spans: pd.DataFrame, train_end: int) -> dict[str, pd.DataFrame]:
    """Events whose whole window is before, straddling, or after the calibration cutoff."""
    return {
        "calibration": spans[spans["hi"] < train_end],
        "straddling": spans[(spans["lo"] < train_end) & (spans["hi"] >= train_end)],
        "post_calibration": spans[spans["lo"] >= train_end],
    }


# ---------------------------------------------------------------------------
# Common-event delay
# ---------------------------------------------------------------------------
def common_event_delays(a: pd.DataFrame, b: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """First-alarm delays on the events detected by both methods (b minus a)."""
    m = a[["event_id", "detected", "first_delay"]].merge(
        b[["event_id", "detected", "first_delay"]], on="event_id", suffixes=("_a", "_b")
    )
    da, db = m["detected_a"].astype(bool), m["detected_b"].astype(bool)
    common = m[da & db].copy()
    common["difference_b_minus_a"] = common["first_delay_b"] - common["first_delay_a"]
    summary = {
        "n_events": int(len(m)),
        "detected_a": int(da.sum()),
        "detected_b": int(db.sum()),
        "n_common": int(len(common)),
        "own_set_mean_delay_a": float(m.loc[da, "first_delay_a"].mean()) if da.any() else math.nan,
        "own_set_mean_delay_b": float(m.loc[db, "first_delay_b"].mean()) if db.any() else math.nan,
        "common_mean_delay_a": float(common["first_delay_a"].mean()) if len(common) else math.nan,
        "common_mean_delay_b": float(common["first_delay_b"].mean()) if len(common) else math.nan,
        "common_median_delay_a": float(common["first_delay_a"].median()) if len(common) else math.nan,
        "common_median_delay_b": float(common["first_delay_b"].median()) if len(common) else math.nan,
        "paired_mean_difference": float(common["difference_b_minus_a"].mean()) if len(common) else math.nan,
        "paired_median_difference": float(common["difference_b_minus_a"].median()) if len(common) else math.nan,
        "events_only_a": ",".join(m.loc[da & ~db, "event_id"]),
        "events_only_b": ",".join(m.loc[db & ~da, "event_id"]),
    }
    return summary, common[["event_id", "first_delay_a", "first_delay_b", "difference_b_minus_a"]]


# ---------------------------------------------------------------------------
# Operating curves
# ---------------------------------------------------------------------------
def operating_point(y: np.ndarray, pred: np.ndarray, spans: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict[str, Any]:
    """Day-level, event-level, cost and (optionally) common-event delay at one alarm set."""
    per = per_event_outcomes(pred, spans)
    out: dict[str, Any] = {**day_metrics(y, pred), **event_summary(per), **cost_components(y, pred, per)}
    if reference is not None:
        summary, _ = common_event_delays(reference, per)
        out.update(
            {
                "n_common_with_reference": summary["n_common"],
                "common_mean_delay_reference": summary["common_mean_delay_a"],
                "common_mean_delay_this": summary["common_mean_delay_b"],
            }
        )
    return out


def volatility_curve(
    scores: dict[str, Any], y: np.ndarray, spans: pd.DataFrame, quantiles: np.ndarray, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Volatility trigger at each calibration quantile."""
    te = scores["train_end"]
    rows = []
    for q in quantiles:
        thr = quantile_threshold(scores["volatility"], te, q)
        pred = (scores["volatility"] > thr).astype(int)
        rows.append({"quantile": float(q), "threshold": thr, **operating_point(y, pred, spans, reference)})
    return pd.DataFrame(rows)


def confirmation_curve(
    scores: dict[str, Any],
    config: dict[str, Any],
    y: np.ndarray,
    spans: pd.DataFrame,
    quantiles: np.ndarray,
    reference: pd.DataFrame | None = None,
    sweep: str = "volatility",
    variant: str = "retrospective_confirmation",
) -> pd.DataFrame:
    """Confirmation rule with one percentile swept and every other setting at its default."""
    rows = []
    for q in quantiles:
        kwargs = {"q_volatility": q} if sweep == "volatility" else {"q_instability": q}
        alarms = detector_alarms(scores, config, **kwargs)
        rows.append(
            {
                "quantile": float(q),
                "threshold": alarms["thresholds"][sweep],
                **operating_point(y, alarms[variant], spans, reference),
            }
        )
    return pd.DataFrame(rows)


def nearest_points(curve: pd.DataFrame, column: str, target: float) -> pd.DataFrame:
    """Nearest operating point overall, and the nearest at or above and at or below ``target``.

    Rows with identical alarm sets (same alarm-day count along a nested sweep) are
    collapsed first, keeping the range of quantiles that produce them.
    """
    grouped = (
        curve.groupby("alarm_days", sort=False)
        .agg(quantile_min=("quantile", "min"), quantile_max=("quantile", "max"))
        .reset_index()
    )
    first = curve.drop_duplicates("alarm_days").set_index("alarm_days")
    collapsed = grouped.join(first, on="alarm_days").drop(columns=["quantile"])
    gap = collapsed[column] - target
    picks = []
    for label, sub in [
        ("nearest", collapsed),
        ("nearest_at_or_above", collapsed[gap >= 0]),
        ("nearest_at_or_below", collapsed[gap <= 0]),
    ]:
        if sub.empty:
            continue
        row = sub.loc[(sub[column] - target).abs().idxmin()].copy()
        row["match"] = label
        row["target"] = target
        row["target_column"] = column
        picks.append(row)
    return pd.DataFrame(picks)


# ---------------------------------------------------------------------------
# Bootstraps
# ---------------------------------------------------------------------------
def paired_event_bootstrap(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Means of event-level paired differences, resampling event rows with replacement.

    Repeated draws stay repeated: each resample is a mean over ``len(values)`` draws.
    """
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[idx].mean(axis=1)


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def _counts(y: np.ndarray, p: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    return tp, fp, fn, tn


def _rates(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "fpr": fp / max(fp + tn, 1),
    }


def paired_block_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, block: int, n_resamples: int, seed: int
) -> pd.DataFrame:
    """Circular block bootstrap of day-level differences b minus a.

    The daily tuples (label, alarm a, alarm b) are resampled together in blocks, so the
    detector outputs stay fixed and only the evaluation sample varies.
    """
    y, a, b = (np.asarray(v).astype(bool) for v in (y, a, b))
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_resamples):
        idx = circular_block_indices(len(y), block, rng)
        ra, rb = _rates(*_counts(y[idx], a[idx])), _rates(*_counts(y[idx], b[idx]))
        rows.append({"resample": r, **{f"delta_{k}": rb[k] - ra[k] for k in ra}})
    return pd.DataFrame(rows)


def observed_day_deltas(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    y, a, b = (np.asarray(v).astype(bool) for v in (y, a, b))
    ra, rb = _rates(*_counts(y, a)), _rates(*_counts(y, b))
    return {f"delta_{k}": rb[k] - ra[k] for k in ra}


def interval_summary(samples: np.ndarray, observed: float) -> dict[str, float]:
    s = np.asarray(samples, float)
    return {
        "observed": float(observed),
        "bootstrap_mean": float(s.mean()),
        "ci_low": float(np.quantile(s, 0.025)),
        "ci_high": float(np.quantile(s, 0.975)),
        "share_below_zero": float(np.mean(s < 0)),
        "share_zero": float(np.mean(s == 0)),
        "share_above_zero": float(np.mean(s > 0)),
    }


# ---------------------------------------------------------------------------
# CUSUM baseline diagnostics
# ---------------------------------------------------------------------------
def cusum_components(df: pd.DataFrame, config: dict[str, Any], train_end: int | None = None) -> dict[str, Any]:
    """The submitted CUSUM baseline, step by step, as computed in ``detector_outputs``."""
    te = calibration_train_end(len(df), config) if train_end is None else int(train_end)
    ret_z = _z_from_train(df["log_return"].to_numpy(float), te)
    n = len(ret_z)
    c_plus, c_minus, stat = np.zeros(n), np.zeros(n), np.zeros(n)
    pos = neg = 0.0
    for i in range(1, n):
        pos = max(0.0, pos + ret_z[i])
        neg = min(0.0, neg + ret_z[i])
        c_plus[i], c_minus[i], stat[i] = pos, neg, max(pos, -neg)
    stat_s = _rolling_mean(_z_from_train(stat, te), config["features"]["smooth_window"])
    threshold = quantile_threshold(stat_s, te, config["detectors"]["cusum_threshold_quantile"])
    return {
        "train_end": te,
        "standardized_return": ret_z,
        "c_plus": c_plus,
        "c_minus": c_minus,
        "statistic": stat,
        "smoothed_standardized_statistic": stat_s,
        "threshold": threshold,
        "alarms": (stat_s > threshold).astype(int),
    }


def alarm_episodes(alarms: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive alarm days as inclusive (start, end) index pairs."""
    a = np.asarray(alarms, int)
    edges = np.diff(np.concatenate([[0], a, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Mechanism diagnostics
# ---------------------------------------------------------------------------
def dispersion_ratio(values: np.ndarray, lo: int, hi: int, pre_days: int = 120) -> float:
    """sd inside [lo, hi] over sd of the ``pre_days`` days before lo; NaN without a full pre-window."""
    if lo - pre_days < 0:
        return math.nan
    v = np.asarray(values, float)
    return float(np.nanstd(v[lo : hi + 1]) / np.nanstd(v[lo - pre_days : lo]))
