"""Optuna tuning with a fixed trial budget per (model, feature set) on the hold-out validation year.

Objective = mean validation RMSE over all horizons. Best params are saved to configs/best/{model}_{fset}.yaml
and picked up automatically by `aqi_bench.run`. Equal budget for every model keeps the benchmark fair.

Usage: python -m aqi_bench.tune --model lgbm --fset F3 --trials 50
"""
from __future__ import annotations
import argparse, os
import numpy as np
import optuna
import pandas as pd
import yaml

from .data import load
from .features import build_xy, time_features, raw_sequence_columns, HORIZONS, FEATURE_SETS
from .splits import holdout
from .models import REGISTRY, DEEP
from .models.deep import make_sequences
from .metrics import rmse

SPACES = {
    "ridge": lambda t: dict(alpha=t.suggest_float("alpha", 1e-2, 1e5, log=True)),
    "lgbm": lambda t: dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 5, 100, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        n_estimators=2000),
    "xgb": lambda t: dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        n_estimators=2000),
}


def _deep_space(t):
    return dict(lookback=t.suggest_categorical("lookback", [24, 72, 168]),
                hidden=t.suggest_categorical("hidden", [64, 128, 256]),
                d_model=t.suggest_categorical("d_model", [64, 128]),
                layers=t.suggest_int("layers", 1, 3),
                dropout=t.suggest_float("dropout", 0.0, 0.4),
                lr=t.suggest_float("lr", 1e-4, 5e-3, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [128, 256, 512]))


for _m in DEEP:
    SPACES[_m] = _deep_space


def _val_rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([rmse(y[:, j], pred[:, j]) for j in range(y.shape[1])]))


def make_objective(model: str, fset: str, max_epochs: int):
    df = load()
    X, Y = build_xy(df, fset, HORIZONS)
    tr, va, _ = holdout(df.index)
    m_tr, m_va = (pd.Series(m, index=df.index).reindex(X.index).fillna(False).to_numpy(bool) for m in (tr, va))
    if model in DEEP:
        seq_cols = raw_sequence_columns(fset)
        static = time_features(df.index) if FEATURE_SETS[fset]["time"] else pd.DataFrame(index=df.index)

    def objective(trial):
        p = SPACES[model](trial)
        if model in DEEP:
            p["max_epochs"] = max_epochs
            a = make_sequences(df, seq_cols, static, Y[m_tr], p["lookback"])
            b = make_sequences(df, seq_cols, static, Y[m_va], p["lookback"])
            net = REGISTRY[model](seed=0, **p).fit(a[0], a[1], a[2], val=(b[0], b[1], b[2]))
            return _val_rmse(net.predict(b[0], b[1]), b[2])
        mdl = REGISTRY[model](seed=0, **p) if model in ("lgbm", "xgb") else REGISTRY[model](**p)
        if model in ("lgbm", "xgb"):
            mdl.fit(X[m_tr], Y[m_tr], X[m_va], Y[m_va])
            p["n_estimators"] = int(np.mean([getattr(m, "best_iteration_", None) or m.best_iteration or 2000
                                             for m in mdl.models.values()]))
            trial.set_user_attr("n_estimators", p["n_estimators"])
        else:
            mdl.fit(X[m_tr], Y[m_tr])
        return _val_rmse(mdl.predict(X[m_va]).to_numpy(), Y[m_va].to_numpy())
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SPACES))
    ap.add_argument("--fset", required=True, choices=sorted(FEATURE_SETS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max-epochs", type=int, default=40)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(os.path.join("configs", "best"), exist_ok=True)
    os.makedirs(os.path.join("results", "tuning"), exist_ok=True)
    storage = f"sqlite:///results/tuning/optuna.db"
    study = optuna.create_study(study_name=f"{a.model}_{a.fset}", storage=storage, load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(make_objective(a.model, a.fset, a.max_epochs), n_trials=max(0, a.trials - len(study.trials)))
    best = dict(study.best_params)
    if "n_estimators" in study.best_trial.user_attrs:
        best["n_estimators"] = int(study.best_trial.user_attrs["n_estimators"] * 1.1)
    out = os.path.join("configs", "best", f"{a.model}_{a.fset}.yaml")
    with open(out, "w") as fh:
        yaml.safe_dump(best, fh)
    print(f"{a.model} {a.fset}: best val RMSE = {study.best_value:.3f} over {len(study.trials)} trials -> {out}")
    print(best)


if __name__ == "__main__":
    main()
