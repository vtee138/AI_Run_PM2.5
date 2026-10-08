"""Point-forecast metrics, skill score vs persistence and event-detection metrics."""
from __future__ import annotations
import numpy as np

UNHEALTHY = 55.5  # ug/m3, US-EPA PM2.5 'Unhealthy' breakpoint used as the alert threshold


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def smape(y, p): return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-8)))
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def skill(y, p, p_ref): return float(1 - np.mean((y - p) ** 2) / np.mean((y - p_ref) ** 2))


def event_metrics(y, p, thr=UNHEALTHY):
    yt, pt = y >= thr, p >= thr
    tp, fp, fn = np.sum(yt & pt), np.sum(~yt & pt), np.sum(yt & ~pt)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(precision=float(prec), recall=float(rec), f1=float(f1), csi=float(csi))


def all_metrics(y, p, p_persist):
    y, p, p_persist = map(np.asarray, (y, p, p_persist))
    out = dict(rmse=rmse(y, p), mae=mae(y, p), smape=smape(y, p), r2=r2(y, p),
               skill=skill(y, p, p_persist))
    out.update(event_metrics(y, p))
    return out
