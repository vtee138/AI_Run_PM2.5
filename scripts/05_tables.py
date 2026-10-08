"""Generate LaTeX tables for the paper from reports/tables/*.csv into paper/tables/*.tex.

  tab_data.tex        descriptive statistics of the variables
  tab_main_cv.tex     RMSE (mean +- std over folds x seeds) model x feature set at each horizon
  tab_skill.tex       skill score vs persistence, best feature set per model, per horizon
  tab_ablation.tex    Delta RMSE vs F1 for F2..F5 (best tree model + ridge), per horizon
  tab_event.tex       event F1 / CSI at 55.5 ug/m3 for h = 24 h
  tab_dm.tex          Diebold-Mariano p-values vs the best model (hold-out, h = 1/24/48)
  tab_hparams.tex     tuned hyper-parameters (appendix)
Re-run after every 03_aggregate.py; missing models are simply omitted so the paper always compiles.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = os.path.join("paper", "tables"); os.makedirs(OUT, exist_ok=True)
T = os.path.join("reports", "tables")
NAME = {"persistence": "Persistence", "snaive": "Seasonal na\\\"ive", "ridge": "Ridge", "lgbm": "LightGBM",
        "xgb": "XGBoost", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
ORDER = list(NAME)
HS = [1, 6, 12, 24, 48]


def w(name, s):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(s)
    print("wrote", name)


def bold_min(vals, fmt="{:.2f}"):
    """Format a list, bolding the minimum."""
    arr = np.array([np.nan if v is None else v for v in vals], float)
    m = np.nanmin(arr) if np.isfinite(arr).any() else None
    out = []
    for v in arr:
        if np.isnan(v): out.append("--")
        else: out.append(("\\textbf{" + fmt + "}").format(v) if v == m else fmt.format(v))
    return out


def tab_data():
    from aqi_bench.data import load, load_raw
    raw = load_raw()
    cols = [("pm25", "PM$_{2.5}$", "$\\mu$g/m$^3$"), ("pm10", "PM$_{10}$", "$\\mu$g/m$^3$"), ("o3", "O$_3$", "$\\mu$g/m$^3$"),
            ("no2", "NO$_2$", "$\\mu$g/m$^3$"), ("so2", "SO$_2$", "$\\mu$g/m$^3$"), ("co", "CO", "$\\mu$g/m$^3$"),
            ("temp", "Temperature", "$^\\circ$C"), ("rh", "Relative humidity", "\\%"), ("wind", "Wind speed", "m/s"),
            ("pressure", "Pressure", "hPa"), ("precip", "Precipitation", "mm"), ("clouds", "Cloud cover", "\\%"), ("uv", "UV index", "--")]
    rows = []
    for c, n, u in cols:
        s = raw[c]
        rows.append(f"{n} & {u} & {s.mean():.1f} & {s.std():.1f} & {s.min():.1f} & {s.max():.1f} \\\\")
    body = "\n".join(rows)
    w("tab_data.tex", f"""\\begin{{tabular}}{{llrrrr}}
\\toprule
Variable & Unit & Mean & Std & Min & Max \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_main_cv():
    df = pd.read_csv(os.path.join(T, "all_runs.csv"))
    cv = df[df.split.str.startswith("fold")]
    g = cv.groupby(["model", "fset", "horizon"])["rmse"].agg(["mean", "std"]).reset_index()
    models = [m for m in ORDER if m in set(g.model)]
    fsets = ["F1", "F2", "F3", "F4", "F5"]
    lines = []
    for h in HS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{Horizon $h={h}$\\,h}}}} \\\\")
        sub = g[g.horizon == h]
        best = sub[~sub.model.isin(["persistence", "snaive"])]["mean"].min()
        for m in models:
            cells = []
            for f in fsets:
                r = sub[(sub.model == m) & (sub.fset == f)]
                if m in ("persistence", "snaive"):
                    r = sub[(sub.model == m)]
                if len(r) == 0: cells.append("--"); continue
                mu, sd = r["mean"].iloc[0], r["std"].iloc[0]
                txt = f"{mu:.2f}" + (f"$\\pm${sd:.2f}" if pd.notna(sd) and sd > 0 else "")
                if mu == best: txt = "\\textbf{" + txt + "}"
                cells.append(txt)
                if m in ("persistence", "snaive"):  # naive rows span all feature sets
                    cells = [f"\\multicolumn{{5}}{{c}}{{{txt}}}"]; break
            lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    body = "\n".join(lines)
    w("tab_main_cv.tex", f"""\\begin{{tabular}}{{lccccc}}
\\toprule
Model & F1 & F2 & F3 & F4 & F5 \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_skill():
    df = pd.read_csv(os.path.join(T, "all_runs.csv"))
    cv = df[df.split.str.startswith("fold") & ~df.model.isin(["persistence"])]
    g = cv.groupby(["model", "fset", "horizon"])["skill"].mean().reset_index()
    models = [m for m in ORDER if m in set(g.model)]
    rows = []
    for m in models:
        cells, bests = [], []
        for h in HS:
            sub = g[(g.model == m) & (g.horizon == h)]
            if len(sub) == 0: cells.append("--"); bests.append(""); continue
            r = sub.loc[sub.skill.idxmax()]
            cells.append(f"{r.skill:.3f}"); bests.append(r.fset)
        rows.append(f"{NAME[m]} & " + " & ".join(f"{c}$^{{\\mathrm{{{b}}}}}$" if b else c for c, b in zip(cells, bests)) + " \\\\")
    body = "\n".join(rows)
    w("tab_skill.tex", f"""\\begin{{tabular}}{{lccccc}}
\\toprule
Model & $h=1$ & $h=6$ & $h=12$ & $h=24$ & $h=48$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_ablation():
    abl = pd.read_csv(os.path.join(T, "feature_ablation.csv"))
    models = [m for m in ("ridge", "lgbm", "xgb", "lstm", "gru", "tcn", "transformer") if m in set(abl.model)]
    rows = []
    for m in models:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{NAME[m]}}}}} \\\\")
        for f in ["F2", "F3", "F4", "F5"]:
            if f not in abl.columns: continue
            cells = []
            for h in HS:
                r = abl[(abl.model == m) & (abl.horizon == h)]
                if len(r) == 0 or pd.isna(r[f].iloc[0]): cells.append("--"); continue
                v = r[f].iloc[0]
                cells.append(f"{v:+.2f}")
            rows.append(f"\\quad {f} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    w("tab_ablation.tex", f"""\\begin{{tabular}}{{lccccc}}
\\toprule
Feature set & $h=1$ & $h=6$ & $h=12$ & $h=24$ & $h=48$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_event():
    df = pd.read_csv(os.path.join(T, "all_runs.csv"))
    cv = df[df.split.str.startswith("fold") & (df.horizon == 24)]
    g = cv.groupby(["model", "fset"])[["precision", "recall", "f1", "csi"]].mean().reset_index()
    models = [m for m in ORDER if m in set(g.model)]
    rows = []
    for m in models:
        sub = g[g.model == m]
        r = sub.loc[sub.f1.idxmax()]
        fs = "" if m in ("persistence", "snaive") else f" ({r.fset})"
        rows.append(f"{NAME[m]}{fs} & {r.precision:.3f} & {r.recall:.3f} & {r.f1:.3f} & {r.csi:.3f} \\\\")
    body = "\n".join(rows)
    w("tab_event.tex", f"""\\begin{{tabular}}{{lcccc}}
\\toprule
Model (feature set) & Precision & Recall & F1 & CSI \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_dm():
    p = os.path.join(T, "dm_tests.csv")
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        return w("tab_dm.tex", "\\begin{tabular}{l}pending\\end{tabular}")
    dm = pd.read_csv(p)
    dm = dm[dm.fset == "F5"] if "F5" in set(dm.fset) else dm
    rows = []
    for h in (1, 24, 48):
        sub = dm[dm.horizon == h]
        if len(sub) == 0: continue
        best = sub.best.iloc[0]
        cells = [f"{NAME[r.other]}: {r.p_value:.3f}" + ("$^*$" if r.p_value < 0.05 else "") for r in sub.itertuples()]
        rows.append(f"{h} & {NAME[best]} & " + "; ".join(cells) + " \\\\")
    body = "\n".join(rows)
    w("tab_dm.tex", f"""\\begin{{tabular}}{{clp{{5.2cm}}}}
\\toprule
$h$ & Best model & DM $p$-values of the other models vs.\\ best ($^*$: $p<0.05$) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def tab_hparams():
    rows = []
    for f in sorted(glob.glob(os.path.join("configs", "best", "*.yaml"))):
        m, fs = os.path.basename(f)[:-5].split("_")
        with open(f) as fh: p = yaml.safe_load(fh) or {}
        txt = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())
        rows.append(f"{NAME.get(m, m)} & {fs} & {txt} \\\\")
    body = "\n".join(rows) if rows else "-- & -- & tuning pending \\\\"
    w("tab_hparams.tex", f"""\\begin{{tabular}}{{llp{{11cm}}}}
\\toprule
Model & Set & Selected hyper-parameters (Optuna, 2024 validation year) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


if __name__ == "__main__":
    tab_data(); tab_main_cv(); tab_skill(); tab_ablation(); tab_event(); tab_dm(); tab_hparams()
