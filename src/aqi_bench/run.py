"""Run one experiment: (model, feature set, split, seed) -> predictions + metrics on disk.

Usage:
  python -m aqi_bench.run --model lgbm --fset F3 --split holdout --seed 0
  python -m aqi_bench.run --model lstm --fset F5 --split fold2 --seed 1 --params configs/best/lstm_F5.yaml
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import yaml

from .data import load, TARGET
from .features import build_xy, time_features, raw_sequence_columns, HORIZONS, FEATURE_SETS
from .splits import holdout, rolling_origin, GAP_HOURS
from .metrics import all_metrics
from .models import REGISTRY, DEEP, STOCHASTIC
from .models.deep import make_sequences

RESULTS = "results"


def masks_for_split(idx: pd.DatetimeIndex, split: str):
    """Return (train, val, test) boolean masks. Folds carve a chronological val block off train."""
    if split == "holdout":
        return holdout(idx)
    k = int(split.replace("fold", ""))
    tr, te = rolling_origin(idx)[k]
    n_tr = tr.sum()
    tr_idx = np.where(tr)[0]
    cut = tr_idx[int(0.85 * n_tr)]
    val = np.zeros_like(tr); val[tr_idx[tr_idx >= cut]] = True
    tr2 = tr.copy(); tr2[tr_idx[tr_idx >= cut - GAP_HOURS]] = False
    return tr2, val, te


def load_params(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def run(model: str, fset: str, split: str, seed: int = 0, params: dict | None = None,
        horizons=HORIZONS, tag: str = "", save: bool = True) -> pd.DataFrame:
    params = dict(params or {})
    df = load()
    X, Y = build_xy(df, fset, horizons)
    tr, va, te = masks_for_split(df.index, split)
    m_tr, m_va, m_te = (pd.Series(m, index=df.index).reindex(X.index).fillna(False).to_numpy(bool) for m in (tr, va, te))
    t0 = time.time()

    if model in DEEP:
        L = int(params.get("lookback", 72))
        seq_cols = raw_sequence_columns(fset)
        static = time_features(df.index) if FEATURE_SETS[fset]["time"] else pd.DataFrame(index=df.index)
        parts = {k: make_sequences(df, seq_cols, static, Y[m], L) for k, m in
                 dict(tr=m_tr, va=m_va, te=m_te).items()}
        net = REGISTRY[model](seed=seed, **params)
        net.fit(parts["tr"][0], parts["tr"][1], parts["tr"][2],
                val=(parts["va"][0], parts["va"][1], parts["va"][2]))
        pred = pd.DataFrame(net.predict(parts["te"][0], parts["te"][1]), index=parts["te"][3], columns=Y.columns)
        epochs = len(net.history)
    else:
        mdl = REGISTRY[model](seed=seed, **params) if model in STOCHASTIC else REGISTRY[model](**params)
        if model in ("lgbm", "xgb"):
            mdl.fit(X[m_tr], Y[m_tr], X[m_va], Y[m_va])
        else:
            mdl.fit(X[m_tr | m_va], Y[m_tr | m_va])  # deterministic models: no early stopping, use train+val
        pred = mdl.predict(X[m_te])
        epochs = None
    elapsed = time.time() - t0

    y_true = Y.loc[pred.index]
    persist = X.loc[pred.index, "pm25_lag0"].to_numpy()
    rows = []
    for c in Y.columns:
        h = int(c.split("y_h")[1])
        met = all_metrics(y_true[c].to_numpy(), pred[c].to_numpy(), persist)
        rows.append(dict(model=model, fset=fset, split=split, seed=seed, horizon=h, tag=tag,
                         n_test=len(pred), train_sec=round(elapsed, 1), epochs=epochs, **met))
    res = pd.DataFrame(rows)
    if save:
        d = os.path.join(RESULTS, "preds"); os.makedirs(d, exist_ok=True)
        name = f"{model}_{fset}_{split}_s{seed}{('_' + tag) if tag else ''}"
        out = pred.copy(); out.columns = [f"pred_{c}" for c in out.columns]
        pd.concat([y_true, out], axis=1).to_parquet(os.path.join(d, name + ".parquet"))
        os.makedirs(os.path.join(RESULTS, "metrics"), exist_ok=True)
        res.to_csv(os.path.join(RESULTS, "metrics", name + ".csv"), index=False)
        with open(os.path.join(RESULTS, "metrics", name + ".json"), "w") as fh:
            json.dump(dict(params=params, seq_cols=raw_sequence_columns(fset) if model in DEEP else None), fh)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--fset", required=True, choices=sorted(FEATURE_SETS))
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default=None, help="yaml with hyper-parameters (e.g. configs/best/lgbm_F3.yaml)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    p = a.params or os.path.join("configs", "best", f"{a.model}_{a.fset}.yaml")
    res = run(a.model, a.fset, a.split, a.seed, load_params(p), tag=a.tag)
    print(res[["model", "fset", "split", "seed", "horizon", "rmse", "mae", "skill", "f1", "train_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
