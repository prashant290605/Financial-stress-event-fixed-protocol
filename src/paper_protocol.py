from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_spy(path: Path, price_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["date"] = pd.to_datetime(df["Date"])
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")
    df = df.dropna(subset=["date", price_column]).sort_values("date")
    return df[["date", price_column]].rename(columns={price_column: "price"})


def preprocess_spy(config: dict[str, Any]) -> pd.DataFrame:
    spy_cfg = config["spy"]
    df = _load_spy(ROOT / "data" / "raw_market" / "spy.csv", spy_cfg["price_column"])
    df["log_return"] = np.log(df["price"]).diff()
    df["rolling_volatility"] = df["log_return"].rolling(spy_cfg["rolling_vol_window"]).std()
    rolling_mean = df["log_return"].rolling(spy_cfg["zscore_window"]).mean()
    rolling_std = df["log_return"].rolling(spy_cfg["zscore_window"]).std()
    df["return_zscore"] = (df["log_return"] - rolling_mean) / rolling_std
    w = spy_cfg["signal_weights"]
    df["signal"] = (
        w["log_return"] * df["log_return"]
        + w["rolling_volatility"] * df["rolling_volatility"]
        + w["return_zscore"] * df["return_zscore"]
    )
    df = df.dropna().reset_index(drop=True)
    out = ROOT / spy_cfg["processed_output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def missing_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.bdate_range(df["date"].min(), df["date"].max())
    actual = pd.DatetimeIndex(df["date"])
    missing = full.difference(actual)
    return pd.DataFrame({"missing_business_day": missing})


def nearest_trading_date(dates: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
    idx = int(np.argmin(np.abs((dates - target).dt.days.to_numpy())))
    return pd.Timestamp(dates.iloc[idx])


def build_event_table(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    events_cfg = config["events"]
    seed = pd.read_csv(ROOT / events_cfg["input"])
    rows: list[dict[str, Any]] = []
    for _, row in seed.iterrows():
        cal = pd.Timestamp(row["calendar_date"])
        mapped = nearest_trading_date(df["date"], cal)
        center = int(df.index[df["date"] == mapped][0])
        lo = max(0, center - events_cfg["severity_half_window"])
        hi = min(len(df), center + events_cfg["severity_half_window"] + 1)
        prices = df["price"].iloc[lo:hi].to_numpy()
        running_max = np.maximum.accumulate(prices)
        drawdown = prices / running_max - 1.0
        severity_score = float(abs(np.min(drawdown)))
        rows.append(
            {
                "event_id": row["event_id"],
                "event_name": row["event_name"],
                "calendar_date": cal.date().isoformat(),
                "mapped_trading_date": mapped.date().isoformat(),
                "event_type": row["event_type"],
                "inclusion_rule": row["inclusion_rule"],
                "source_reference": row["source_reference"],
                "severity_score": severity_score,
            }
        )
    out = pd.DataFrame(rows)
    q1, q2 = out["severity_score"].quantile([1 / 3, 2 / 3]).to_numpy()
    out["severity_label"] = np.where(
        out["severity_score"] >= q2,
        "critical",
        np.where(out["severity_score"] >= q1, "high", "medium"),
    )
    out["severity_rule"] = config["events"]["severity_rule"]
    cols = [
        "event_id",
        "event_name",
        "calendar_date",
        "mapped_trading_date",
        "event_type",
        "inclusion_rule",
        "source_reference",
        "severity_score",
        "severity_label",
        "severity_rule",
    ]
    out = out[cols]
    out.to_csv(ROOT / events_cfg["output"], index=False)
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy(float)


def _z_from_train(values: np.ndarray, train_end: int) -> np.ndarray:
    ref = values[:train_end]
    mu = float(np.nanmean(ref))
    sd = float(np.nanstd(ref))
    sd = sd if sd > 1e-12 else 1.0
    return np.nan_to_num((values - mu) / sd, nan=0.0, posinf=6.0, neginf=-6.0)


def compute_instability_features(signal: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    feat = config["features"]
    n = len(signal)
    kappa = np.zeros(n)
    for t in range(n):
        s = max(0, t - feat["ar_window"] + 1)
        if t - s < 2:
            continue
        a = signal[s:t]
        b = signal[s + 1 : t + 1]
        denom = float(np.dot(a, a))
        phi = 0.0 if abs(denom) < 1e-12 else float(np.dot(a, b) / denom)
        kappa[t] = 1.0 - phi
    slope = pd.Series(kappa).rolling(5, min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
    ).fillna(0.0).to_numpy(float)
    cis = pd.Series(np.maximum(-slope, 0.0)).rolling(feat["cis_window"], min_periods=1).sum().to_numpy(float)
    mmd = np.zeros(n)
    w = feat["mmd_window"]
    for t in range(1, n):
        a = signal[max(0, t - w) : t]
        b = signal[t : min(n, t + w)]
        if len(a) < 4 or len(b) < 4:
            continue
        ab = np.abs(a[:, None] - b[None, :])
        sigma = max(float(np.median(ab)), 1e-4)
        denom = 2.0 * sigma * sigma
        kaa = np.exp(-np.minimum(((a[:, None] - a[None, :]) ** 2) / denom, 40)).mean()
        kbb = np.exp(-np.minimum(((b[:, None] - b[None, :]) ** 2) / denom, 40)).mean()
        kab = np.exp(-np.minimum((ab**2) / denom, 40)).mean()
        mmd[t] = max(0.0, float(kaa + kbb - 2.0 * kab))
    return pd.DataFrame({"MMD": _rolling_mean(mmd, 3), "kappa": kappa, "CIS": cis})


def detector_outputs(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    signal = df["signal"].to_numpy(float)
    feat = compute_instability_features(signal, config)
    train_end = int(round(len(df) * config["detectors"]["calibration_fraction"]))
    mmd_z = _z_from_train(feat["MMD"].to_numpy(float), train_end)
    kappa_z = _z_from_train(feat["kappa"].to_numpy(float), train_end)
    cis_z = _z_from_train(feat["CIS"].to_numpy(float), train_end)
    variants = {
        "default_no_cis": 0.5 * mmd_z + 0.5 * np.abs(kappa_z),
        "equal_with_cis": (mmd_z + np.abs(kappa_z) + cis_z) / 3.0,
        "mmd_only": mmd_z,
        "kappa_only": np.abs(kappa_z),
        "cis_only": cis_z,
    }
    smoothed = {k: _rolling_mean(v, config["features"]["smooth_window"]) for k, v in variants.items()}
    q = config["detectors"]["score_threshold_quantile"]
    inst_pred = {k: (v > np.quantile(v[:train_end], q)).astype(int) for k, v in smoothed.items()}
    vol = df["rolling_volatility"].to_numpy(float)
    vol_z = _z_from_train(vol, train_end)
    vol_s = _rolling_mean(vol_z, config["features"]["smooth_window"])
    vol_pred = (vol_s > np.quantile(vol_s[:train_end], config["detectors"]["volatility_threshold_quantile"])).astype(int)
    ret_z = _z_from_train(df["log_return"].to_numpy(float), train_end)
    csum = np.zeros(len(ret_z))
    pos = neg = 0.0
    for i in range(1, len(ret_z)):
        pos = max(0.0, pos + ret_z[i])
        neg = min(0.0, neg + ret_z[i])
        csum[i] = max(pos, -neg)
    csum_z = _z_from_train(csum, train_end)
    csum_s = _rolling_mean(csum_z, config["features"]["smooth_window"])
    cusum_pred = (csum_s > np.quantile(csum_s[:train_end], config["detectors"]["cusum_threshold_quantile"])).astype(int)
    pred = {
        "volatility_percentile": vol_pred,
        "cusum": cusum_pred,
        "instability_score": inst_pred["default_no_cis"],
        "mmd_only_confirmation": inst_pred["mmd_only"],
        "kappa_only_confirmation": inst_pred["kappa_only"],
    }
    for w in config["detectors"]["hybrid_windows"]:
        pred[f"hybrid_w{w}"] = windowed_hybrid(vol_pred, inst_pred["default_no_cis"], w)
    feat = pd.concat([df[["date", "price", "log_return", "rolling_volatility", "return_zscore", "signal"]], feat], axis=1)
    return feat, pred | {f"variant_{k}": v for k, v in inst_pred.items()}


def windowed_hybrid(volatility: np.ndarray, instability: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(volatility, dtype=int)
    for t, flag in enumerate(volatility):
        if not flag:
            continue
        lo, hi = max(0, t - window), min(len(volatility), t + window + 1)
        out[t] = int(np.any(instability[lo:hi]))
    return out


def event_mask(df: pd.DataFrame, events: pd.DataFrame, half_window: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    mask = np.zeros(len(df), dtype=int)
    spans = []
    for _, row in events.iterrows():
        center = int(df.index[df["date"] == pd.Timestamp(row["mapped_trading_date"])][0])
        lo, hi = max(0, center - half_window), min(len(df) - 1, center + half_window)
        mask[lo : hi + 1] = 1
        spans.append((center, lo, hi))
    return mask, spans


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def event_metrics(pred: np.ndarray, spans: Iterable[tuple[int, int, int]]) -> dict[str, float]:
    delays, alignments, covered = [], [], []
    for center, lo, hi in spans:
        idx = np.where(pred[lo : hi + 1] == 1)[0] + lo
        if len(idx) == 0:
            delays.append(np.nan)
            alignments.append(np.nan)
            covered.append(0.0)
        else:
            signed = int(idx[0] - center)
            nearest = int(idx[np.argmin(np.abs(idx - center))] - center)
            delays.append(float(signed))
            alignments.append(float(abs(nearest)))
            covered.append(1.0)
    return {
        "event_coverage": float(np.mean(covered)),
        "event_delay": float(np.nanmean(delays)) if np.any(np.isfinite(delays)) else math.nan,
        "event_alignment_error": float(np.nanmean(alignments)) if np.any(np.isfinite(alignments)) else math.nan,
    }


def summarize_methods(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int, int]]]:
    y, spans = event_mask(df, events, config["events"]["half_window"])
    methods = {
        "volatility_percentile": pred["volatility_percentile"],
        "cusum": pred["cusum"],
        "instability_score": pred["instability_score"],
        "mmd_only_confirmation": pred["mmd_only_confirmation"],
        "kappa_only_confirmation": pred["kappa_only_confirmation"],
        "hybrid_windowed_confirmation": pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"],
    }
    rows = []
    for name, arr in methods.items():
        rows.append({"method": name, **point_metrics(y, arr), **event_metrics(arr, spans)})
    return pd.DataFrame(rows), y, spans


def stratified_summary(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    methods = {
        "volatility_percentile": pred["volatility_percentile"],
        "hybrid_windowed_confirmation": pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"],
    }
    for severity, group in events.groupby("severity_label"):
        y, _ = event_mask(df, group, config["events"]["half_window"])
        for name, arr in methods.items():
            rows.append({"severity": severity, "method": name, **point_metrics(y, arr)})
    return pd.DataFrame(rows)


def score_ablation(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> pd.DataFrame:
    y, _ = event_mask(df, events, config["events"]["half_window"])
    rows = []
    vol = pred["volatility_percentile"]
    for label, key in [
        ("default no-CIS", "variant_default_no_cis"),
        ("equal with CIS", "variant_equal_with_cis"),
        ("MMD only", "variant_mmd_only"),
        ("|kappa| only", "variant_kappa_only"),
        ("CIS only", "variant_cis_only"),
    ]:
        rows.append({"variant": label, **point_metrics(y, windowed_hybrid(vol, pred[key], config["detectors"]["default_hybrid_window"]))})
    return pd.DataFrame(rows)


def window_ablation(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> pd.DataFrame:
    y, _ = event_mask(df, events, config["events"]["half_window"])
    return pd.DataFrame(
        [{"window": w, **point_metrics(y, pred[f"hybrid_w{w}"])} for w in config["detectors"]["hybrid_windows"]]
    )


def event_window_sensitivity(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    methods = {
        "Volatility percentile": pred["volatility_percentile"],
        "Hybrid W=10": pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"],
    }
    for half_window in [20, 30, 45, 60]:
        y, _ = event_mask(df, events, half_window)
        for method, arr in methods.items():
            rows.append({"event_window": f"+/-{half_window}", "half_window": half_window, "method": method, **point_metrics(y, arr)})
    return pd.DataFrame(rows)


def cost_sensitivity(summary: pd.DataFrame, ratios: list[int] | None = None) -> pd.DataFrame:
    """Cost = c_FP * FP + c_FN * FN with c_FN fixed to 1."""
    if ratios is None:
        ratios = [1, 2, 5, 10, 20, 50]
    rows = []
    idx = summary.set_index("method")
    vol = idx.loc["volatility_percentile"]
    hyb = idx.loc["hybrid_windowed_confirmation"]
    for ratio in ratios:
        volatility_cost = float(ratio * vol["fp"] + vol["fn"])
        hybrid_cost = float(ratio * hyb["fp"] + hyb["fn"])
        if abs(volatility_cost - hybrid_cost) < 1e-12:
            preferred = "tie"
        else:
            preferred = "hybrid" if hybrid_cost < volatility_cost else "volatility"
        rows.append(
            {
                "fp_to_fn_cost_ratio": ratio,
                "volatility_cost": volatility_cost,
                "hybrid_cost": hybrid_cost,
                "preferred_method": preferred,
            }
        )
    return pd.DataFrame(rows)


def event_bootstrap(df: pd.DataFrame, events: pd.DataFrame, pred: dict[str, np.ndarray], config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    rows = []
    hybrid = pred[f"hybrid_w{config['detectors']['default_hybrid_window']}"]
    vol = pred["volatility_percentile"]
    for _ in range(config["bootstrap"]["n_resamples"]):
        sample = events.sample(len(events), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        y, _ = event_mask(df, sample, config["events"]["half_window"])
        mh, mv = point_metrics(y, hybrid), point_metrics(y, vol)
        rows.append({"bootstrap": "event", "resample": _, "delta_f1": mh["f1"] - mv["f1"], "delta_fpr": mh["fpr"] - mv["fpr"]})
    return pd.DataFrame(rows)


def block_bootstrap(y: np.ndarray, hybrid: np.ndarray, vol: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"] + 1)
    n, block = len(y), config["bootstrap"]["block_length"]
    rows = []
    starts = np.arange(0, max(1, n - block + 1))
    for _ in range(config["bootstrap"]["n_resamples"]):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, min(s + block, n)))
        idx_arr = np.asarray(idx[:n], int)
        mh, mv = point_metrics(y[idx_arr], hybrid[idx_arr]), point_metrics(y[idx_arr], vol[idx_arr])
        rows.append({"bootstrap": "block", "resample": _, "delta_f1": mh["f1"] - mv["f1"], "delta_fpr": mh["fpr"] - mv["fpr"]})
    return pd.DataFrame(rows)


def bootstrap_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, grp in samples.groupby("bootstrap"):
        for metric in ["delta_f1", "delta_fpr"]:
            rows.append(
                {
                    "bootstrap": kind,
                    "metric": metric,
                    "estimate": float(grp[metric].mean()),
                    "ci_low": float(grp[metric].quantile(0.025)),
                    "ci_high": float(grp[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic(config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    cfg = config["synthetic"]
    rows = []
    for sid in range(cfg["n_series"]):
        n = cfg["length"]
        calm_vol = rng.uniform(*cfg["calm_vol_range"])
        stress_mult = rng.uniform(*cfg["stress_vol_multiplier_range"])
        drift_mag = rng.uniform(*cfg["drift_magnitude_range"])
        beta = rng.uniform(*cfg["nonlinear_beta_range"])
        df_t = rng.uniform(*cfg["student_t_df_range"])
        lap_scale = rng.uniform(*cfg["laplace_scale_range"])
        stress_start = int(rng.integers(int(0.45 * n), int(0.6 * n)))
        collapse_start = int(rng.integers(int(0.72 * n), int(0.82 * n)))
        x = np.zeros(n)
        regime = np.zeros(n, int)
        for t in range(1, n):
            if t >= collapse_start:
                regime[t] = 2
            elif t >= stress_start:
                regime[t] = 1
            vol = calm_vol * (stress_mult if regime[t] else 1.0)
            if t >= collapse_start - 30:
                vol *= 1.0 + (t - (collapse_start - 30)) / 30.0
            innovation = vol * rng.standard_t(df_t) + rng.laplace(0.0, lap_scale)
            localized_drift = drift_mag if stress_start <= t < collapse_start else 0.0
            nonlinear = -beta * (x[t - 1] ** 3)
            x[t] = 0.7 * x[t - 1] + localized_drift + nonlinear + innovation
        rows.extend({"series_id": sid, "t": t, "signal": x[t], "label": ["stable", "drift", "collapse"][regime[t]]} for t in range(n))
    return pd.DataFrame(rows)


def evaluate_synthetic(syn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    example = syn[syn["series_id"] == 0].copy()
    for sid, grp in syn.groupby("series_id"):
        s = grp["signal"].to_numpy(float)
        y = (grp["label"] == "collapse").astype(int).to_numpy()
        vol = pd.Series(s).rolling(20, min_periods=2).std().fillna(0.0).to_numpy(float)
        pred = (vol > np.quantile(vol[: int(0.7 * len(vol))], 0.9)).astype(int)
        rows.append({"series_id": sid, **point_metrics(y, pred)})
        if sid == 0:
            example["volatility_detector"] = pred
    return pd.DataFrame(rows), example


def evaluate_nab(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ROOT / config["nab"]["root"]
    with (root / config["nab"]["windows_file"]).open("r", encoding="utf-8") as f:
        windows = json.load(f)
    rows = []
    for rel, spans in sorted(windows.items()):
        csv = root / "data" / rel
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if not {"timestamp", "value"} <= set(df.columns):
            continue
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        y = np.zeros(len(df), int)
        for lo, hi in spans:
            y[(ts >= pd.Timestamp(lo)) & (ts <= pd.Timestamp(hi))] = 1
        values = pd.to_numeric(df["value"], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
        ret = np.diff(values, prepend=values[0])
        train_end = int(0.7 * len(values))
        vol = pd.Series(values).rolling(20, min_periods=2).std().fillna(0.0).to_numpy(float)
        vol_z = _z_from_train(vol, train_end)
        vol_s = _rolling_mean(vol_z, config["features"]["smooth_window"])
        volatility = (vol_s > np.quantile(vol_s[:train_end], 0.9)).astype(int)

        # NAB is an external sanity check, so use a fast deterministic proxy for the
        # same two confirmation ingredients instead of the quadratic SPY MMD loop.
        before = pd.Series(ret).rolling(30, min_periods=4).mean().shift(1).fillna(0.0).to_numpy(float)
        after = pd.Series(ret[::-1]).rolling(30, min_periods=4).mean().shift(1).fillna(0.0).to_numpy(float)[::-1]
        mmd_proxy = np.abs(after - before)
        kappa_proxy = 1.0 - pd.Series(ret).rolling(36, min_periods=3).corr(pd.Series(ret).shift(1)).fillna(0.0).to_numpy(float)
        mmd_z = _z_from_train(mmd_proxy, train_end)
        kappa_z = _z_from_train(kappa_proxy, train_end)
        instability_score = _rolling_mean(0.5 * mmd_z + 0.5 * np.abs(kappa_z), 5)
        instability = (instability_score > np.quantile(instability_score[:train_end], 0.9)).astype(int)
        csum = np.maximum.accumulate(np.cumsum(ret - np.mean(ret[:train_end])))
        csum_z = _z_from_train(csum, train_end)
        cusum = (_rolling_mean(csum_z, 5) > np.quantile(_rolling_mean(csum_z, 5)[:train_end], 0.9)).astype(int)
        hybrid = windowed_hybrid(volatility, instability, config["detectors"]["default_hybrid_window"])
        for method, pred in {
            "instability_score": instability,
            "hybrid_windowed_confirmation": hybrid,
            "volatility_percentile": volatility,
            "cusum": cusum,
        }.items():
            rows.append({"file": rel, "subset": rel.split("/")[0], "method": method, **point_metrics(y[train_end:], pred[train_end:])})
    all_rows = pd.DataFrame(rows)
    summary = all_rows.groupby(["subset", "method"])[["precision", "recall", "f1", "fpr"]].mean().reset_index()
    return all_rows, summary


def save_figures(
    df: pd.DataFrame,
    events: pd.DataFrame,
    pred: dict[str, np.ndarray],
    windows: pd.DataFrame,
    event_windows: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    cost_table: pd.DataFrame,
    strata: pd.DataFrame,
    synthetic_example: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, df["signal"], lw=0.7, color="black", label="signal")
    hyb = pred["hybrid_w10"].astype(bool)
    ax.scatter(dates[hyb], df.loc[hyb, "signal"], s=8, color="#d62728", label="hybrid alarms")
    for d in pd.to_datetime(events["mapped_trading_date"]):
        ax.axvspan(d - pd.tseries.offsets.BDay(60), d + pd.tseries.offsets.BDay(60), color="#1f77b4", alpha=0.035, lw=0)
        ax.axvline(d, color="#1f77b4", alpha=0.25, lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("SPY event-fixed timeline and hybrid outputs")
    fig.tight_layout()
    fig.savefig(outdir / "spy_timeline.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(windows["window"], windows["recall"], marker="o", label="Recall")
    ax.plot(windows["window"], windows["f1"], marker="o", label="F1")
    ax.plot(windows["window"], windows["fpr"], marker="o", label="FPR")
    ax.set_xlabel("Hybrid confirmation window W")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "window_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in event_windows.groupby("method"):
        ax.plot(grp["half_window"], grp["f1"], marker="o", label=f"{method} F1")
        ax.plot(grp["half_window"], grp["fpr"], marker="s", linestyle="--", label=f"{method} FPR")
    ax.set_xlabel("Event half-window (trading days)")
    ax.set_ylabel("Metric value")
    ax.set_title("Event-window sensitivity")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "event_window_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cost_table["fp_to_fn_cost_ratio"], cost_table["volatility_cost"], marker="o", label="Volatility")
    ax.plot(cost_table["fp_to_fn_cost_ratio"], cost_table["hybrid_cost"], marker="o", label="Hybrid")
    ax.set_xscale("log")
    ax.set_xlabel("False-positive / false-negative cost ratio")
    ax.set_ylabel("Total cost")
    ax.set_title("Cost-sensitive comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "cost_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for ax, metric, title in zip(axes, ["delta_f1", "delta_fpr"], ["Delta F1", "Delta FPR"]):
        data = [bootstrap_samples.loc[bootstrap_samples["bootstrap"] == kind, metric].to_numpy() for kind in ["event", "block"]]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor("#4c78a8")
            body.set_alpha(0.45)
        ax.axhline(0.0, color="black", lw=0.8, linestyle=":")
        ax.set_xticks([1, 2], ["event", "block"])
        ax.set_title(title)
    fig.suptitle("Bootstrap distributions: Hybrid - Volatility")
    fig.tight_layout()
    fig.savefig(outdir / "bootstrap_delta_violin.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary["event_alignment_error"], summary["event_coverage"], s=36)
    for _, row in summary.iterrows():
        ax.annotate(row["method"], (row["event_alignment_error"], row["event_coverage"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Event alignment error (trading days)")
    ax.set_ylabel("Event coverage")
    ax.set_title("Coverage-alignment tradeoff")
    fig.tight_layout()
    fig.savefig(outdir / "coverage_alignment_tradeoff.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in strata.groupby("method"):
        ax.plot(grp["severity"], grp["f1"], marker="o", label=f"{method} F1")
    ax.set_ylabel("F1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "severity_f1.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(synthetic_example["t"], synthetic_example["signal"], color="black", lw=0.8)
    for label, color in [("drift", "#ffbb78"), ("collapse", "#d62728")]:
        mask = synthetic_example["label"] == label
        ax.fill_between(synthetic_example["t"], synthetic_example["signal"].min(), synthetic_example["signal"].max(), where=mask, color=color, alpha=0.12)
    alarms = synthetic_example["volatility_detector"].astype(bool)
    ax.scatter(synthetic_example.loc[alarms, "t"], synthetic_example.loc[alarms, "signal"], s=10, color="#1f77b4", label="detector")
    ax.legend(frameon=False)
    ax.set_title("Synthetic example with labels and detector outputs")
    fig.tight_layout()
    fig.savefig(outdir / "synthetic_example.png", dpi=220)
    plt.close(fig)
