"""Aggregate results/metrics/*.csv into paper-ready tables + statistical tests.

Outputs (reports/tables/):
  all_runs.csv                 every run x horizon
  main_holdout.csv             mean +- std over seeds on hold-out test (2025 H1), model x fset x horizon
  main_cv.csv                  mean +- std over folds x seeds (rolling-origin CV)
  feature_ablation.csv         delta RMSE of F2..F5 relative to F1 per model/horizon (CV)
  friedman_nemenyi.txt         Friedman p, avg ranks, Nemenyi p-matrix (models on F5, blocks = folds x horizons)
  dm_tests.csv                 Diebold-Mariano p-values vs. best model per horizon (hold-out, seed 0)
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aqi_bench.stats import diebold_mariano, friedman_nemenyi

OUT = os.path.join("reports", "tables"); os.makedirs(OUT, exist_ok=True)
METRICS = ["rmse", "mae", "smape", "r2", "skill", "f1", "csi"]
ORDER = ["persistence", "snaive", "ridge", "lgbm", "xgb", "lstm", "gru", "tcn", "transformer"]


def fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if pd.notna(std) else f"{mean:.2f}"


def summarize(df, by):
    g = df.groupby(by, observed=True)[METRICS]
    m, s = g.mean(), g.std().fillna(0)
    out = m.copy()
    for c in METRICS:
        out[c] = [fmt(a, b) for a, b in zip(m[c], s[c])]
    out["n_runs"] = g.size()
    return out.reset_index(), m.reset_index()


def main():
    files = glob.glob(os.path.join("results", "metrics", "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.tag.fillna("") == ""]
    df["model"] = pd.Categorical(df.model, [m for m in ORDER if m in set(df.model)], ordered=True)
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)

    hold = df[df.split == "holdout"]
    t_hold, m_hold = summarize(hold, ["model", "fset", "horizon"])
    t_hold.to_csv(os.path.join(OUT, "main_holdout.csv"), index=False)
    cv = df[df.split.str.startswith("fold")]
    t_cv, m_cv = summarize(cv, ["model", "fset", "horizon"])
    t_cv.to_csv(os.path.join(OUT, "main_cv.csv"), index=False)

    # Feature ablation: RMSE(Fk) - RMSE(F1) per model & horizon (CV means), negative = improvement
    base = m_cv[m_cv.fset == "F1"].set_index(["model", "horizon"])["rmse"]
    abl = m_cv.set_index(["model", "horizon", "fset"])["rmse"].unstack("fset")
    for f in abl.columns:
        abl[f] = abl[f] - base.reindex(abl.index).values
    abl["F1_rmse"] = base.reindex(abl.index).values
    abl.round(3).to_csv(os.path.join(OUT, "feature_ablation.csv"))

    # Friedman/Nemenyi across models on their best feature set, blocks = fold x horizon
    lines = []
    for fset in sorted(cv.fset.unique()):
        sub = cv[cv.fset == fset].groupby(["model", "split", "horizon"], observed=True)["rmse"].mean().unstack("model")
        sub = sub.dropna(axis=1)
        if sub.shape[1] < 3: continue
        p, ranks, nem = friedman_nemenyi(sub)
        lines += [f"### {fset}  (blocks = {len(sub)} fold x horizon)", f"Friedman p = {p:.2e}", "avg ranks:",
                  ranks.round(2).to_string(), "Nemenyi p-values:", nem.round(3).to_string(), ""]
    with open(os.path.join(OUT, "friedman_nemenyi.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # DM tests on hold-out predictions (seed 0) vs the best model per (fset, horizon)
    rows = []
    for fset in sorted(hold.fset.unique()):
        preds = {}
        for f in glob.glob(os.path.join("results", "preds", f"*_{fset}_holdout_s0.parquet")):
            preds[os.path.basename(f).split("_")[0]] = pd.read_parquet(f)
        if fset != "F1":
            for m in ("persistence", "snaive"):
                f = os.path.join("results", "preds", f"{m}_F1_holdout_s0.parquet")
                if os.path.exists(f): preds[m] = pd.read_parquet(f)
        if len(preds) < 2: continue
        idx = None
        for p in preds.values(): idx = p.index if idx is None else idx.intersection(p.index)
        for h in sorted(hold.horizon.unique()):
            err = {m: (p.loc[idx, f"y_h{h}"] - p.loc[idx, f"pred_y_h{h}"]).to_numpy() for m, p in preds.items()}
            best = min(err, key=lambda m: np.mean(err[m] ** 2))
            for m, e in err.items():
                if m == best: continue
                stat, pv = diebold_mariano(err[best], e, h=h)
                rows.append(dict(fset=fset, horizon=h, best=best, other=m, dm_stat=stat, p_value=pv))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "dm_tests.csv"), index=False)

    print("hold-out RMSE (mean over seeds):")
    print(m_hold.pivot_table(index=["model", "fset"], columns="horizon", values="rmse", observed=True).round(2).to_string())
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
