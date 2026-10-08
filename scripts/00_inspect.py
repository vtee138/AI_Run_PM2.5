"""Step 0: download the Kaggle dataset and dump a schema/EDA report."""
import os, sys, glob
import pandas as pd
import numpy as np
import kagglehub

path = kagglehub.dataset_download("phungdinhdat/aqi-in-hanoi-2022-2025")
print("Downloaded to:", path)
files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
lines = ["# Dataset schema report", "", f"Cache path: `{path}`", "", "## Files", ""]
for f in files:
    if os.path.isfile(f):
        print(f, os.path.getsize(f))
        lines.append(f"- `{os.path.relpath(f, path)}` ({os.path.getsize(f)/1e6:.2f} MB)")

for f in files:
    if not f.lower().endswith((".csv", ".xlsx", ".parquet", ".json")):
        continue
    print("\n=====", f)
    if f.lower().endswith(".csv"):
        df = pd.read_csv(f, low_memory=False)
    elif f.lower().endswith(".xlsx"):
        df = pd.read_excel(f)
    elif f.lower().endswith(".parquet"):
        df = pd.read_parquet(f)
    else:
        df = pd.read_json(f)
    lines += ["", f"## `{os.path.relpath(f, path)}`", "", f"Shape: {df.shape}", "", "### Columns", ""]
    print(df.shape); print(df.dtypes); print(df.head(3).to_string())
    for c in df.columns:
        lines.append(f"- `{c}` dtype={df[c].dtype}, missing={df[c].isna().mean()*100:.1f}%, nunique={df[c].nunique()}")
    # find time column
    tcol = None
    for c in df.columns:
        if any(k in c.lower() for k in ["time", "date", "datetime", "timestamp"]):
            tcol = c; break
    if tcol is not None:
        t = pd.to_datetime(df[tcol], errors="coerce")
        d = t.sort_values().diff().dropna()
        step = d.mode().iloc[0] if len(d) else None
        lines += ["", f"Time column `{tcol}`: {t.min()} → {t.max()}, mode step = {step}, "
                  f"unparsed={t.isna().sum()}, duplicates={t.duplicated().sum()}"]
        print("time:", t.min(), t.max(), "step", step, "dups", t.duplicated().sum())
        if step is not None:
            gaps = d[d > step]
            lines.append(f"Gaps > step: {len(gaps)}; gaps > 6h: {(d > pd.Timedelta('6h')).sum()}; "
                         f"largest gap: {d.max()}")
    for c in df.columns:
        if any(k in c.lower() for k in ["station", "site", "location", "name", "city", "source"]):
            vc = df[c].value_counts().head(20)
            lines += ["", f"Top values of `{c}`:", "", "```", vc.to_string(), "```"]
            print(vc)
    num = df.select_dtypes(include=np.number)
    if len(num.columns):
        lines += ["", "### Numeric summary", "", "```", num.describe().T.round(2).to_string(), "```"]
        print(num.describe().T.round(2).to_string())
    lines += ["", "### Head", "", "```", df.head(5).to_string(), "```"]

os.makedirs("reports", exist_ok=True)
with open("reports/00_schema.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nWrote reports/00_schema.md")
