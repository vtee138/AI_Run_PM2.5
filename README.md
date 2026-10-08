# PM2.5 forecasting benchmark for Hanoi: input features × forecast horizons × models

Code for *An Experimental Study of Input Features and Forecast Horizons for PM2.5 Prediction in Hanoi*.
Dataset: Kaggle `phungdinhdat/aqi-in-hanoi-2022-2025` (hourly, 2022-01-13 → 2025-06-30, one city,
PM2.5/PM10/O3/NO2/SO2/CO + temperature, humidity, wind, pressure, precipitation, clouds, UV).

## Setup

```bash
pip install -e .
# on the GPU box install the CUDA build of torch first, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "kagglehub[pandas-datasets]" lightgbm xgboost statsmodels optuna seaborn holidays scikit-posthocs shap pyyaml pytest
python scripts/00_inspect.py          # downloads data via kagglehub, writes reports/00_schema.md
python -m aqi_bench.data              # builds data/processed/hanoi_hourly.parquet
python -m pytest tests -q             # leakage / split tests
```

## Experimental design

| Axis | Levels |
|---|---|
| Feature set | F1 PM2.5 lags/rolling · F2 = F1 + calendar (sin/cos hour, dow, doy, weekend, VN holidays) · F3 = F2 + other pollutants · F4 = F2 + meteorology · F5 = all |
| Horizon | 1, 6, 12, 24, 48 h (direct strategy; DL models are multi-output) |
| Model | persistence, seasonal naive, ridge, LightGBM, XGBoost, LSTM, GRU, TCN, Transformer |
| Split | hold-out (train 2022–23, val 2024, test 2025 H1) + 4-fold rolling-origin CV (6-month test blocks, 168 h gap) |
| Seeds | 5 for every stochastic model; mean ± std reported |
| Tuning | Optuna, equal 50-trial budget per (model, feature set) on the 2024 validation year |
| Metrics | RMSE, MAE, sMAPE, R², skill score vs persistence, event F1/CSI at 55.5 µg/m³ |
| Statistics | Diebold–Mariano (HAC, HLN-corrected), Friedman + Nemenyi, CD diagrams |

## Running

```bash
# 1. tune (CPU models are quick; DL on the GPU box)
for f in F1 F2 F3 F4 F5; do for m in ridge lgbm xgb; do python -m aqi_bench.tune --model $m --fset $f --trials 50; done; done
for f in F1 F2 F3 F4 F5; do for m in lstm gru tcn transformer; do python -m aqi_bench.tune --model $m --fset $f --trials 30; done; done

# 2. full grid (skips finished runs, safe to re-launch)
python scripts/02_run_grid.py --dry-run
python scripts/02_run_grid.py --models persistence snaive ridge lgbm xgb
python scripts/02_run_grid.py --models lstm gru tcn transformer

# 3. tables + tests, 4. figures, 5. LaTeX tables, 6. static check of the manuscript
python scripts/03_aggregate.py
python scripts/04_figs.py && cp reports/figs/*.pdf paper/figs/
python scripts/05_tables.py
python scripts/06_check_tex.py
```

## Paper

`paper/main.tex` (IEEEtran conference format) + `paper/refs.bib`. Tables in `paper/tables/` and figures in
`paper/figs/` are generated, never edited by hand. Compile on Overleaf (upload the `paper/` folder) or locally with
tectonic: `cd paper && tectonic main.tex`. Text marked `\todo{...}` (red in the PDF) must be rewritten once the
full grid has been run; `scripts/06_check_tex.py` counts the remaining markers.

Single run: `python -m aqi_bench.run --model lgbm --fset F3 --split fold2 --seed 1`.
Results land in `results/metrics/*.csv` (one row per horizon) and `results/preds/*.parquet`.

## Data notes (for the paper)

* 24 missing hours (2023-12-31) and 5 duplicated timestamps; gaps ≤ 3 h are linearly interpolated, longer gaps are masked and samples touching them dropped.
* CO changes measurement scale during 2024 (source/unit change: 2024 mean ≈ 936 vs 148 in 2025). CO is robust-scaled per calendar year (median/IQR) before use; the other columns are used as-is.
* Pollutant columns are non-stationary across years (NO2/SO2/O3 level shifts, PM10 spikes up to 1103). A linear model on F3/F5 needs strong shrinkage (ridge alpha ~1e3–1e4, found by tuning); with alpha=1 it is worse than persistence on some folds. This is itself a result for the paper.
* Test year 2025 H1 is more polluted than the training years (mean PM2.5 61 vs 46–51 µg/m³), which is why both hold-out and rolling-origin CV are reported.

## Layout

```
src/aqi_bench/   data.py features.py splits.py metrics.py stats.py tune.py run.py models/{baselines,trees,deep}.py
scripts/         00_inspect 02_run_grid 03_aggregate 04_figs
configs/best/    tuned hyper-parameters (yaml), written by tune.py
results/         metrics/ preds/ tuning/   (git-ignored)
reports/         00_schema.md tables/ figs/
```
