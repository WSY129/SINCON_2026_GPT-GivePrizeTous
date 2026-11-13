"""Forecast metrics and calendar-aware diagnostics for SINCON.

RMSLE is the official metric in the user-provided competition rules. ``smape`` uses
200 * |prediction - target| / (|target| + |prediction|), with 0/0 := 0.
MASE scales are calculated only from the supplied training history using
exact seven-calendar-day matches within station/hour, never adjacent rows.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


SUPPORTED_METRICS = ("mae", "rmse", "smape", "rmsle")
GROUP_COLS = ("station_id", "hour")


def _arrays(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Targets and predictions must be nonempty 1-D arrays of equal length.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Targets and predictions must contain only finite numbers.")
    return y, p


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return unweighted row-level metrics; undefined values are NaN.

    RMSLE is undefined for negative values. No implicit clipping is performed.
    ``rmse_mae_ratio`` is undefined for perfect predictions (MAE == 0).
    MBE is prediction minus truth, so positive means overprediction.
    """
    y, p = _arrays(y_true, y_pred)
    err = p - y
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = np.abs(y) + np.abs(p)
    smape_terms = np.divide(200.0 * abs_err, denom, out=np.zeros_like(err), where=denom > 0)
    rmsle = float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))) if (y >= 0).all() and (p >= 0).all() else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "smape": float(smape_terms.mean()),
        "rmsle": rmsle,
        "mbe": float(err.mean()),
        "rmse_mae_ratio": rmse / mae if mae > 0 else float("nan"),
    }


def score(y_true, y_pred, metric: str = "rmsle") -> float:
    """Select an objective explicitly; default is official RMSLE, all minimize."""
    metric = metric.lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose {SUPPORTED_METRICS}.")
    # Ensemble weight search calls this repeatedly: compute only its objective.
    y, p = _arrays(y_true, y_pred)
    if metric == "mae":
        result = float(np.abs(p - y).mean())
    elif metric == "rmse":
        result = float(np.sqrt(np.mean((p - y) ** 2)))
    elif metric == "smape":
        denominator = np.abs(y) + np.abs(p)
        result = float(np.divide(200.0 * np.abs(p - y), denominator, out=np.zeros_like(y), where=denominator > 0).mean())
    else:
        result = float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))) if (y >= 0).all() and (p >= 0).all() else float("nan")
    if not np.isfinite(result):
        raise ValueError(f"{metric} is undefined for these targets/predictions.")
    return result


def _calendar_frame(frame: pd.DataFrame, group_cols: Sequence[str], target: str) -> pd.DataFrame:
    needed = [*group_cols, "date", target]
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    result = frame[needed].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result[needed].isna().any().any():
        raise ValueError("Calendar keys and target cannot contain missing values.")
    if result.duplicated([*group_cols, "date"]).any():
        raise ValueError("Duplicate station/hour/date keys are not valid for calendar diagnostics.")
    values = pd.to_numeric(result[target], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target values must be finite.")
    result[target] = values
    return result


def seasonal_scales(
    train: pd.DataFrame,
    seasonality_days: int = 7,
    target: str = "boardings",
    group_cols: Sequence[str] = GROUP_COLS,
) -> pd.DataFrame:
    """Training-only MASE denominator and exact seasonal pair counts per series.

    Zero denominators are retained as zero so the caller can distinguish a
    constant series from a series with no historical seasonal pairs (NaN).
    """
    if not isinstance(seasonality_days, int) or seasonality_days <= 0:
        raise ValueError("seasonality_days must be a positive integer.")
    t = _calendar_frame(train, group_cols, target)
    previous = t.rename(columns={target: "seasonal_value"})
    previous["date"] += pd.Timedelta(days=seasonality_days)
    paired = t.merge(previous, on=[*group_cols, "date"], how="left", validate="one_to_one")
    paired["seasonal_abs_difference"] = (paired[target] - paired["seasonal_value"]).abs()
    return paired.groupby(list(group_cols), observed=True, dropna=False).agg(
        mase_scale=("seasonal_abs_difference", "mean"),
        seasonal_pairs=("seasonal_abs_difference", "count"),
    ).reset_index()


def residual_autocorrelation(
    predictions: pd.DataFrame,
    lags: Sequence[int] = (1, 7, 28),
    group_cols: Sequence[str] = GROUP_COLS,
) -> pd.DataFrame:
    """Pearson residual correlation at exact calendar-day lags, per series.

    Input needs date/group columns and residual=prediction-truth. This is a
    pairwise calendar-lag correlation, not an ACF across concatenated stations.
    Fewer than three matched pairs or zero variance produces NaN.
    """
    t = _calendar_frame(predictions, group_cols, "residual")
    records = []
    for lag in lags:
        if not isinstance(lag, int) or lag <= 0:
            raise ValueError("ACF lags must be positive integer calendar days.")
        previous = t.rename(columns={"residual": "residual_lag"})
        previous["date"] += pd.Timedelta(days=lag)
        paired = t.merge(previous, on=[*group_cols, "date"], how="inner", validate="one_to_one")
        values_by_group = {}
        for keys, frame in paired.groupby(list(group_cols), observed=True, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            left = frame["residual"].to_numpy()
            right = frame["residual_lag"].to_numpy()
            corr = float(np.corrcoef(left, right)[0, 1]) if len(frame) >= 3 and left.std() > 0 and right.std() > 0 else float("nan")
            values_by_group[keys] = (len(frame), corr)
        for keys in t[list(group_cols)].drop_duplicates().itertuples(index=False, name=None):
            n, corr = values_by_group.get(keys, (0, float("nan")))
            records.append({**dict(zip(group_cols, keys)), "lag_days": lag, "pairs": n, "acf": corr})
    return pd.DataFrame.from_records(records)


def forecast_diagnostics(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictions,
    origin=None,
    horizon_col: str = "horizon",
    target: str = "boardings",
    group_cols: Sequence[str] = GROUP_COLS,
    seasonality_days: int = 7,
) -> dict:
    """Return summary and DataFrames for one fixed-origin forecast fold.

    ``train`` must end no later than origin; all validation dates must follow
    origin. Predictions must have the same row order as ``valid``. The summary
    contains two MASE aggregations: mean per-series MASE (macro), and the mean
    per-row absolute error divided by that row's training-series scale. Both
    exclude unavailable/zero scales and report their coverage explicitly.
    Per-step metrics use calendar-day horizon, not row position or hour number.
    """
    t = _calendar_frame(train, group_cols, target)
    v = _calendar_frame(valid, group_cols, target)
    if t.empty or v.empty:
        raise ValueError("Training and validation must both be nonempty.")
    cutoff = pd.Timestamp(origin).normalize() if origin is not None else t["date"].max()
    if t["date"].max() > cutoff:
        raise ValueError("Training includes observations after forecast origin (leakage).")
    if v["date"].min() <= cutoff:
        raise ValueError("Validation must lie strictly after forecast origin.")
    y, p = _arrays(v[target], predictions)
    v["prediction"] = p
    v["residual"] = p - y
    v["absolute_error"] = np.abs(p - y)
    actual_horizon = (v["date"] - cutoff).dt.days.to_numpy()
    if horizon_col in valid:
        supplied_horizon = pd.to_numeric(valid[horizon_col], errors="raise").to_numpy()
        if not np.array_equal(supplied_horizon, actual_horizon):
            raise ValueError("Supplied horizon does not equal date minus forecast origin.")
    v["horizon"] = actual_horizon
    scales = seasonal_scales(t, seasonality_days, target, group_cols)
    v = v.merge(scales, on=list(group_cols), how="left", sort=False, validate="many_to_one")
    usable_scale = v["mase_scale"].where(v["mase_scale"] > 0)
    v["scaled_absolute_error"] = v["absolute_error"] / usable_scale
    entity_rows = []
    for keys, frame in v.groupby(list(group_cols), observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        stats = regression_metrics(frame[target], frame["prediction"])
        entity_rows.append({
            **dict(zip(group_cols, keys)),
            "rows": len(frame),
            **stats,
            "mase_scale": frame["mase_scale"].iloc[0],
            "seasonal_pairs": frame["seasonal_pairs"].iloc[0],
            "mase": frame["scaled_absolute_error"].mean(),
        })
    per_entity = pd.DataFrame.from_records(entity_rows)
    per_step = pd.DataFrame.from_records([
        {"horizon": int(h), "rows": len(frame), **regression_metrics(frame[target], frame["prediction"]), "mase": frame["scaled_absolute_error"].mean()}
        for h, frame in v.groupby("horizon", sort=True)
    ])
    acf = residual_autocorrelation(v, group_cols=group_cols)
    summary = {
        **regression_metrics(y, p),
        "mase_macro": float(per_entity["mase"].mean()),
        "mase_row_weighted": float(v["scaled_absolute_error"].mean()),
        "mase_series_coverage": float(per_entity["mase"].notna().mean()),
        "mase_row_coverage": float(v["scaled_absolute_error"].notna().mean()),
        "mase_undefined_series": int(per_entity["mase"].isna().sum()),
        "mase_seasonality_days": seasonality_days,
        "origin": str(cutoff.date()),
        "forecast_rows": len(v),
    }
    for lag, frame in acf.groupby("lag_days"):
        finite = frame["acf"].dropna()
        summary[f"acf_{lag}d_macro"] = float(finite.mean())
        summary[f"acf_{lag}d_abs_gt_0_1_fraction"] = float((finite.abs() > 0.1).mean()) if len(finite) else float("nan")
        summary[f"acf_{lag}d_defined_series"] = int(len(finite))
    return {"summary": summary, "per_entity": per_entity, "residual_acf": acf, "per_step": per_step}
