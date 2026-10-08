"""Chronological hold-out split and blocked rolling-origin CV with a gap between train and test."""
from __future__ import annotations
import pandas as pd

GAP_HOURS = 168  # >= max horizon + longest lag, so no test target overlaps train inputs

HOLDOUT = dict(train=("2022-01-01", "2023-12-31 23:00"),
               val=("2024-01-01", "2024-12-31 23:00"),
               test=("2025-01-01", "2025-06-30 23:00"))


def holdout(idx: pd.DatetimeIndex):
    m = {k: (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)) for k, (a, b) in HOLDOUT.items()}
    return m["train"], m["val"], m["test"]


def rolling_origin(idx: pd.DatetimeIndex, n_folds: int = 5, test_months: int = 6,
                   min_train_months: int = 12, gap_hours: int = GAP_HOURS):
    """Expanding-window folds. Test blocks of `test_months` stepping back from the series end."""
    end = idx.max().normalize() + pd.Timedelta(hours=23)
    folds = []
    for k in range(n_folds):
        t_end = end - pd.DateOffset(months=test_months * k)
        t_start = t_end - pd.DateOffset(months=test_months) + pd.Timedelta(hours=1)
        tr_end = t_start - pd.Timedelta(hours=gap_hours)
        if tr_end < idx.min() + pd.DateOffset(months=min_train_months):
            break
        folds.append(((idx <= tr_end), (idx >= t_start) & (idx <= t_end)))
    return folds[::-1]  # chronological order


def describe(idx, folds):
    for i, (tr, te) in enumerate(folds):
        print(f"fold {i}: train {idx[tr].min().date()}..{idx[tr].max().date()} ({tr.sum()})  "
              f"test {idx[te].min().date()}..{idx[te].max().date()} ({te.sum()})")
