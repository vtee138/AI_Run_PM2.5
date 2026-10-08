"""Statistical comparison: Diebold-Mariano (HAC), Friedman + Nemenyi, critical-difference diagram."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(e1, e2, h: int = 1, power: int = 2):
    """DM test on d = |e1|^p - |e2|^p with Newey-West variance (lags h-1) and the
    Harvey-Leybourne-Newbold small-sample correction. Negative stat => model 1 better."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var) if var > 0 else 0.0
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def friedman_nemenyi(scores: pd.DataFrame):
    """scores: rows = blocks (folds/horizons), cols = models, lower is better.
    Returns (friedman p-value, average ranks, Nemenyi p-value matrix)."""
    import scikit_posthocs as sp
    ranks = scores.rank(axis=1).mean()
    f = stats.friedmanchisquare(*[scores[c].values for c in scores.columns])
    nem = sp.posthoc_nemenyi_friedman(scores.values)
    nem.index = nem.columns = scores.columns
    return float(f.pvalue), ranks.sort_values(), nem


_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def cd_diagram(ranks: pd.Series, n_blocks: int, ax=None, title=""):
    """Critical-difference diagram (Demsar 2006), alpha = 0.05."""
    import matplotlib.pyplot as plt
    k = len(ranks)
    cd = _Q05[k] * np.sqrt(k * (k + 1) / (6 * n_blocks))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 * k + 1.5))
    ranks = ranks.sort_values()
    ax.plot([1, k], [0, 0], "k-", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.1], "k-", lw=1)
        ax.text(r, 0.15, str(r), ha="center", fontsize=8)
    for i, (name, r) in enumerate(ranks.items()):
        y = -0.4 - 0.35 * i
        ax.plot([r, r], [0, y], "k-", lw=0.8)
        ax.text(r, y - 0.08, f"{name} ({r:.2f})", ha="center", va="top", fontsize=8)
    vals = ranks.values
    y, i = 0.45, 0
    while i < k:
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            ax.plot([vals[i] - 0.05, vals[j] + 0.05], [y, y], "k-", lw=3)
            y += 0.12
        i = j + 1
    ax.plot([1, 1 + cd], [0.9, 0.9], "k-", lw=1.5)
    ax.text(1 + cd / 2, 0.95, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5 - 0.35 * k, 1.1)
    ax.axis("off")
    ax.set_title(title)
    return ax
