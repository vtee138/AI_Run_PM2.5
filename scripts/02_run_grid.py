"""Run the full benchmark grid: models x feature sets x splits x seeds. Skips runs already on disk.

Examples:
  python scripts/02_run_grid.py --dry-run
  python scripts/02_run_grid.py --models persistence snaive ridge lgbm xgb          # CPU stage
  python scripts/02_run_grid.py --models lstm gru tcn transformer --seeds 0 1 2 3 4 # GPU stage
"""
from __future__ import annotations
import argparse, os, sys, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aqi_bench.run import run, load_params
from aqi_bench.models import REGISTRY, STOCHASTIC
from aqi_bench.features import FEATURE_SETS

SPLITS = ["holdout", "fold0", "fold1", "fold2", "fold3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--fsets", nargs="+", default=sorted(FEATURE_SETS))
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = []
    for m in a.models:
        for f in a.fsets:
            if m in ("persistence", "snaive") and f != "F1":
                continue  # naive baselines do not depend on the feature set
            for s in a.splits:
                for seed in (a.seeds if m in STOCHASTIC else [0]):
                    jobs.append((m, f, s, seed))
    todo = [j for j in jobs if a.force or not os.path.exists(
        os.path.join("results", "metrics", f"{j[0]}_{j[1]}_{j[2]}_s{j[3]}.csv"))]
    print(f"{len(jobs)} jobs in grid, {len(todo)} to run")
    if a.dry_run:
        for j in todo: print("  ", *j)
        return
    for i, (m, f, s, seed) in enumerate(todo, 1):
        p = load_params(os.path.join("configs", "best", f"{m}_{f}.yaml"))
        try:
            r = run(m, f, s, seed, p)
            print(f"[{i}/{len(todo)}] {m} {f} {s} seed={seed}  rmse@24h={r.loc[r.horizon == 24, 'rmse'].iloc[0]:.2f}  "
                  f"({r.train_sec.iloc[0]:.0f}s)", flush=True)
        except Exception:
            print(f"[{i}/{len(todo)}] FAILED {m} {f} {s} seed={seed}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
