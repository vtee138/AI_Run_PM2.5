"""Nested feature sets F1..F5 and direct multi-horizon targets.

All features at row t use only information available at or before t (no leakage).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import holidays
from .data import POLLUTANTS, METEO, TARGET

HORIZONS = [1, 6, 12, 24, 48]
LAGS_TARGET = list(range(1, 25)) + [48, 72, 168]
LAGS_EXOG = [1, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 6, 12, 24]

FEATURE_SETS = {
    "F1": dict(time=False, pollutants=False, meteo=False),
    "F2": dict(time=True, pollutants=False, meteo=False),
    "F3": dict(time=True, pollutants=True, meteo=False),
    "F4": dict(time=True, pollutants=False, meteo=True),
    "F5": dict(time=True, pollutants=True, meteo=True),
}


def _lag_block(s: pd.Series, name: str, lags, windows) -> pd.DataFrame:
    out = {f"{name}_lag{l}": s.shift(l) for l in lags}
    for w in windows:
        r = s.rolling(w, min_periods=w)
        out[f"{name}_rmean{w}"] = r.mean()
        out[f"{name}_rstd{w}"] = r.std()
        out[f"{name}_rmin{w}"] = r.min()
        out[f"{name}_rmax{w}"] = r.max()
    return pd.DataFrame(out)


def time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    vn = holidays.country_holidays("VN", years=range(idx.year.min(), idx.year.max() + 1))
    h, dow, doy = idx.hour, idx.dayofweek, idx.dayofyear
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": np.array([d in vn for d in idx.date], dtype=int),
    }, index=idx)


def build_features(df: pd.DataFrame, fset: str) -> pd.DataFrame:
    """Return feature matrix X (aligned on df.index). lag0 = current observation."""
    cfg = FEATURE_SETS[fset]
    blocks = [_lag_block(df[TARGET], TARGET, [0] + LAGS_TARGET, ROLL_WINDOWS)]
    if cfg["time"]:
        blocks.append(time_features(df.index))
    if cfg["pollutants"]:
        blocks += [_lag_block(df[c], c, [0] + LAGS_EXOG, [6, 24]) for c in POLLUTANTS]
    if cfg["meteo"]:
        blocks += [_lag_block(df[c], c, [0] + LAGS_EXOG, [6, 24]) for c in METEO]
    return pd.concat(blocks, axis=1)


def build_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame({f"y_h{h}": df[TARGET].shift(-h) for h in horizons}, index=df.index)


def build_xy(df: pd.DataFrame, fset: str, horizons=HORIZONS):
    X, Y = build_features(df, fset), build_targets(df, horizons)
    mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X[mask], Y[mask]


def raw_sequence_columns(fset: str) -> list[str]:
    """Columns fed as raw sequences to DL models (they build lags via the lookback window)."""
    cfg = FEATURE_SETS[fset]
    cols = [TARGET]
    if cfg["pollutants"]:
        cols += POLLUTANTS
    if cfg["meteo"]:
        cols += METEO
    return cols
