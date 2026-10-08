"""Paper figures from results/. Writes PDF+PNG into reports/figs/.

  fig_rmse_vs_horizon.*     RMSE vs horizon, one panel per feature set, one line per model (CV mean +- std)
  fig_skill_vs_horizon.*    Skill score vs persistence
  fig_feature_ablation.*    Delta RMSE vs F1 per feature set, grouped by horizon (best tree model)
  fig_cd_diagram.*          Critical-difference diagram of models (F5, blocks = fold x horizon)
  fig_error_by_month.*      Hold-out MAE by month for the best model at h=24
  fig_error_by_level.*      Hold-out bias by true PM2.5 bin (peaks under-estimation)
  fig_shap_by_horizon.*     Top SHAP features by horizon (LightGBM, F5)
"""
from __future__ import annotations
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aqi_bench.stats import friedman_nemenyi, cd_diagram

FIG = os.path.join("reports", "figs"); os.makedirs(FIG, exist_ok=True)
ORDER = ["persistence", "snaive", "ridge", "lgbm", "xgb", "lstm", "gru", "tcn", "transformer"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def line_panels(cv, metric, fname, ylabel):
    fsets = sorted(cv.fset.unique())
    fig, axes = plt.subplots(1, len(fsets), figsize=(3.2 * len(fsets), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fsets):
        sub = cv[(cv.fset == f) | (cv.model.isin(["persistence", "snaive"]))]
        for m in [m for m in ORDER if m in set(sub.model)]:
            g = sub[sub.model == m].groupby("horizon")[metric]
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), label=m, marker="o", ms=3, capsize=2)
        ax.set_title(f); ax.set_xlabel("horizon (h)"); ax.set_xticks(sorted(cv.horizon.unique()))
    axes[0].set_ylabel(ylabel); axes[-1].legend(fontsize=7, frameon=False)
    save(fig, fname)


def main():
    df = pd.read_csv(os.path.join("reports", "tables", "all_runs.csv"))
    cv, hold = df[df.split.str.startswith("fold")], df[df.split == "holdout"]
    line_panels(cv, "rmse", "fig_rmse_vs_horizon", "RMSE (µg/m³)")
    line_panels(cv, "skill", "fig_skill_vs_horizon", "skill vs persistence")

    # feature ablation for the best non-naive model at h=24 (by CV RMSE)
    m_cv = cv.groupby(["model", "fset", "horizon"], observed=True)["rmse"].mean().reset_index()
    best_model = m_cv[(m_cv.horizon == 24) & ~m_cv.model.isin(["persistence", "snaive"])].sort_values("rmse").model.iloc[0]
    sub = m_cv[m_cv.model == best_model].pivot(index="horizon", columns="fset", values="rmse")
    delta = sub.sub(sub["F1"], axis=0).drop(columns="F1")
    fig, ax = plt.subplots(figsize=(5, 3))
    delta.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("ΔRMSE vs F1 (µg/m³)"); ax.set_xlabel("horizon (h)")
    ax.set_title(f"Feature ablation ({best_model})"); ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_feature_ablation")

    # CD diagram
    for fset in ("F5", "F2"):
        blk = cv[cv.fset == fset].groupby(["model", "split", "horizon"])["rmse"].mean().unstack("model").dropna(axis=1)
        if blk.shape[1] >= 3:
            _, ranks, _ = friedman_nemenyi(blk)
            ax = cd_diagram(ranks, len(blk), title=f"CD diagram, RMSE, {fset}")
            save(ax.figure, f"fig_cd_diagram_{fset}")

    # hold-out diagnostics for best model at h=24 on F5 (seed 0)
    f = os.path.join("results", "preds", f"{best_model}_F5_holdout_s0.parquet")
    if os.path.exists(f):
        p = pd.read_parquet(f); y, yh = p["y_h24"], p["pred_y_h24"]
        err = (yh - y)
        fig, ax = plt.subplots(figsize=(5, 2.8))
        (err.abs().groupby(err.index.month).mean()).plot.bar(ax=ax, color="tab:blue")
        ax.set_xlabel("month (2025)"); ax.set_ylabel("MAE @24h (µg/m³)"); ax.set_title(best_model)
        save(fig, "fig_error_by_month")
        bins = pd.cut(y, [0, 25, 50, 75, 100, 150, 500])
        fig, ax = plt.subplots(figsize=(5, 2.8))
        err.groupby(bins, observed=True).mean().plot.bar(ax=ax, color="tab:red")
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("true PM2.5 (µg/m³)"); ax.set_ylabel("bias @24h (pred − true)")
        save(fig, "fig_error_by_level")

    # SHAP by horizon (LightGBM F5, refit quickly on hold-out train)
    try:
        import shap, yaml
        from aqi_bench.data import load
        from aqi_bench.features import build_xy
        from aqi_bench.splits import holdout
        from aqi_bench.models import LightGBMDirect
        from aqi_bench.run import load_params
        d = load(); X, Y = build_xy(d, "F5"); tr, va, te = holdout(d.index)
        mk = lambda m: pd.Series(m, index=d.index).reindex(X.index).fillna(False).to_numpy(bool)
        mdl = LightGBMDirect(seed=0, **load_params("configs/best/lgbm_F5.yaml")).fit(X[mk(tr)], Y[mk(tr)], X[mk(va)], Y[mk(va)])
        Xs = X[mk(te)].sample(min(2000, mk(te).sum()), random_state=0)
        groups = {"pm25 lags": lambda c: c.startswith("pm25"), "time": lambda c: c.split("_")[0] in ("hour", "dow", "doy", "is"),
                  "pollutants": lambda c: c.split("_")[0] in ("pm10", "o3", "no2", "so2", "co"),
                  "meteo": lambda c: c.split("_")[0] in ("temp", "rh", "wind", "pressure", "precip", "clouds", "uv")}
        rows = {}
        for c, m in mdl.models.items():
            sv = np.abs(shap.TreeExplainer(m).shap_values(Xs)).mean(0)
            s = pd.Series(sv, index=Xs.columns)
            rows[int(c.split("y_h")[1])] = {g: s[[k for k in s.index if fn(k)]].sum() for g, fn in groups.items()}
        share = pd.DataFrame(rows).T; share = share.div(share.sum(1), axis=0)
        share.to_csv(os.path.join("reports", "tables", "shap_group_share.csv"))
        fig, ax = plt.subplots(figsize=(5, 3))
        share.plot.bar(stacked=True, ax=ax, width=0.7)
        ax.set_xlabel("horizon (h)"); ax.set_ylabel("share of mean |SHAP|"); ax.legend(fontsize=7, frameon=False, ncol=2)
        save(fig, "fig_shap_by_horizon")
    except Exception as e:
        print("SHAP figure skipped:", e)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
