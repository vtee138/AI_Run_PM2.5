"""Naive baselines and a linear model. All share fit(X, Y) / predict(X) -> DataFrame[y_h*]."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _horizons(Y: pd.DataFrame):
    return [int(c.split("y_h")[1]) for c in Y.columns]


class Persistence:
    """y_hat(t+h) = y(t)."""
    def __init__(self, **kw): self.cols = None
    def fit(self, X, Y): self.cols = list(Y.columns); return self
    def predict(self, X):
        return pd.DataFrame({c: X["pm25_lag0"].values for c in self.cols}, index=X.index)


class SeasonalNaive:
    """y_hat(t+h) = y(t+h-24k) for the smallest k with h-24k <= 0 (daily seasonality)."""
    def __init__(self, **kw): self.cols = None
    def fit(self, X, Y): self.cols = list(Y.columns); return self
    def predict(self, X):
        out = {}
        for c in self.cols:
            h = int(c.split("y_h")[1])
            lag = (24 - h % 24) % 24 if h % 24 else 0
            out[c] = X[f"pm25_lag{lag}"].values
        return pd.DataFrame(out, index=X.index)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip every feature to its [q, 1-q] training quantiles (protects linear models from spikes)."""
    def __init__(self, q: float = 0.005): self.q = q
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.nanquantile(X, self.q, axis=0), np.nanquantile(X, 1 - self.q, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


class RidgeDirect:
    """One standardized ridge regression per horizon."""
    def __init__(self, alpha: float = 1000.0, **kw): self.alpha, self.models = alpha, {}
    def fit(self, X, Y):
        for c in Y.columns:
            self.models[c] = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, Y[c])
        return self
    def predict(self, X):
        return pd.DataFrame({c: m.predict(X) for c, m in self.models.items()}, index=X.index)
