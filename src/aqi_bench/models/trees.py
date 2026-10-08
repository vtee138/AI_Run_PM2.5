"""Gradient-boosted trees, direct strategy (one model per horizon)."""
from __future__ import annotations
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

LGBM_DEFAULT = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
XGB_DEFAULT = dict(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=5,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist")


class _Direct:
    def __init__(self, seed: int = 0, **params):
        self.seed, self.params, self.models = seed, {**self.DEFAULT, **params}, {}

    def fit(self, X, Y, X_val=None, Y_val=None):
        for c in Y.columns:
            m = self._new()
            if X_val is not None:
                self._fit_es(m, X, Y[c], X_val, Y_val[c])
            else:
                m.fit(X, Y[c])
            self.models[c] = m
        return self

    def predict(self, X):
        return pd.DataFrame({c: m.predict(X) for c, m in self.models.items()}, index=X.index)

    def feature_importance(self, c):
        m = self.models[c]
        return pd.Series(m.feature_importances_, index=m.feature_names_in_ if hasattr(m, "feature_names_in_") else None)


class LightGBMDirect(_Direct):
    DEFAULT = LGBM_DEFAULT
    def _new(self):
        return lgb.LGBMRegressor(random_state=self.seed, verbose=-1, n_jobs=-1, **self.params)
    def _fit_es(self, m, X, y, Xv, yv):
        m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])


class XGBoostDirect(_Direct):
    DEFAULT = XGB_DEFAULT
    def _new(self):
        return xgb.XGBRegressor(random_state=self.seed, n_jobs=-1, early_stopping_rounds=50, **self.params)
    def _fit_es(self, m, X, y, Xv, yv):
        m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    def fit(self, X, Y, X_val=None, Y_val=None):
        if X_val is None:  # xgboost requires eval_set when early_stopping_rounds is set
            for c in Y.columns:
                m = xgb.XGBRegressor(random_state=self.seed, n_jobs=-1, **self.params)
                self.models[c] = m.fit(X, Y[c])
            return self
        return super().fit(X, Y, X_val, Y_val)
