"""Leakage / split sanity tests. Run: python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest
from aqi_bench.features import build_features, build_targets, build_xy, HORIZONS, FEATURE_SETS
from aqi_bench.splits import rolling_origin, holdout, GAP_HOURS
from aqi_bench.models import Persistence, SeasonalNaive
from aqi_bench.models.deep import make_sequences
from aqi_bench.data import POLLUTANTS, METEO


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=24 * 400, freq="h")
    d = pd.DataFrame({c: rng.random(len(idx)) * 50 + 1 for c in ["pm25"] + POLLUTANTS + METEO}, index=idx)
    d.index.name = "time"
    return d


@pytest.mark.parametrize("fset", list(FEATURE_SETS))
def test_features_are_causal(df, fset):
    """Perturbing the future must not change any feature at t."""
    t = df.index[5000]
    X0 = build_features(df, fset).loc[t]
    d2 = df.copy(); d2.loc[d2.index > t, :] = 999.0
    X1 = build_features(d2, fset).loc[t]
    pd.testing.assert_series_equal(X0, X1)


def test_targets_are_future(df):
    Y = build_targets(df)
    for h in HORIZONS:
        assert np.allclose(Y[f"y_h{h}"].iloc[:-h].values, df["pm25"].iloc[h:].values)


def test_rolling_folds_have_gap(df):
    for tr, te in rolling_origin(df.index, n_folds=3, test_months=2, min_train_months=3):
        assert df.index[tr].max() + pd.Timedelta(hours=GAP_HOURS) <= df.index[te].min()
        assert not (tr & te).any()


def test_holdout_disjoint(df):
    tr, va, te = holdout(pd.date_range("2022-01-01", "2025-06-30 23:00", freq="h"))
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()


def test_naive_baselines(df):
    X, Y = build_xy(df, "F1")
    p = Persistence().fit(X, Y).predict(X)
    assert np.allclose(p["y_h6"].values, df.loc[X.index, "pm25"].values)
    s = SeasonalNaive().fit(X, Y).predict(X)
    # y_hat(t+6) = y(t-18); y_hat(t+24) = y(t); y_hat(t+48) = y(t)
    assert np.allclose(s["y_h6"].values, df["pm25"].shift(18).loc[X.index].values)
    assert np.allclose(s["y_h24"].values, df.loc[X.index, "pm25"].values)


def test_sequences_are_causal(df):
    X, Y = build_xy(df, "F1")
    static = pd.DataFrame(index=df.index)
    seq, st, y, idx = make_sequences(df, ["pm25"], static, Y.iloc[:200], lookback=24)
    t = idx[10]
    assert np.allclose(seq[10, :, 0], df.loc[:t, "pm25"].iloc[-24:].values)
    assert np.allclose(y[10], Y.loc[t].values)
