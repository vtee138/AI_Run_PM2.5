"""Sequence models (LSTM / GRU / TCN / Transformer) with direct multi-horizon output.

Input per sample: raw sequence of the last `lookback` hours of the selected variables plus the
time features at t. Target: log1p(PM2.5) at all horizons, standardized on the training fold.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT = dict(lookback=72, hidden=128, layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-4,
               batch_size=256, max_epochs=60, patience=8, d_model=128, n_heads=4, kernel=3)


# ----------------------------------------------------------------------------- architectures
class _RNN(nn.Module):
    def __init__(self, n_in, n_static, n_out, hidden, layers, dropout, cell):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))
    def forward(self, seq, static):
        out, _ = self.rnn(seq)
        return self.head(torch.cat([out[:, -1], static], 1))


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dil, dropout):
        super().__init__()
        pad = (k - 1) * dil
        self.pad = pad
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dil))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dil))
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        y = self.drop(torch.relu(self.c1(x)[:, :, : -self.pad]))
        y = self.drop(torch.relu(self.c2(y)[:, :, : -self.pad]))
        return torch.relu(y + self.down(x))


class _TCN(nn.Module):
    def __init__(self, n_in, n_static, n_out, hidden, layers, dropout, kernel):
        super().__init__()
        blocks, c = [], n_in
        for i in range(layers):
            blocks.append(_TCNBlock(c, hidden, kernel, 2 ** i, dropout)); c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))
    def forward(self, seq, static):
        h = self.tcn(seq.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([h, static], 1))


class _Transformer(nn.Module):
    def __init__(self, n_in, n_static, n_out, d_model, n_heads, layers, dropout, lookback):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)
        pe = torch.zeros(lookback, d_model)
        pos = torch.arange(lookback).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d_model + n_static, d_model), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_out))
    def forward(self, seq, static):
        h = self.enc(self.inp(seq) + self.pe[: seq.shape[1]])
        return self.head(torch.cat([h[:, -1], static], 1))


# ----------------------------------------------------------------------------- wrapper
class DeepMultiHorizon:
    """fit(seq, static, Y, val=...) / predict(seq, static). Arrays are built by `make_sequences`."""

    def __init__(self, arch: str = "lstm", seed: int = 0, device: str | None = None, **params):
        self.arch, self.seed = arch, seed
        self.p = {**DEFAULT, **params}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []

    def _build(self, n_in, n_static, n_out):
        p = self.p
        if self.arch in ("lstm", "gru"):
            return _RNN(n_in, n_static, n_out, p["hidden"], p["layers"], p["dropout"], self.arch)
        if self.arch == "tcn":
            return _TCN(n_in, n_static, n_out, p["hidden"], p["layers"], p["dropout"], p["kernel"])
        if self.arch == "transformer":
            return _Transformer(n_in, n_static, n_out, p["d_model"], p["n_heads"], p["layers"],
                                p["dropout"], p["lookback"])
        raise ValueError(self.arch)

    def _to(self, *arrs):
        return [torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrs]

    def fit(self, seq, static, Y, val=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.seq_mu, self.seq_sd = seq.reshape(-1, seq.shape[-1]).mean(0), seq.reshape(-1, seq.shape[-1]).std(0) + 1e-6
        self.st_mu, self.st_sd = static.mean(0), static.std(0) + 1e-6
        Yl = np.log1p(Y); self.y_mu, self.y_sd = Yl.mean(0), Yl.std(0) + 1e-6
        Xs, Xt, Yt = self._to((seq - self.seq_mu) / self.seq_sd, (static - self.st_mu) / self.st_sd, (Yl - self.y_mu) / self.y_sd)
        self.net = self._build(seq.shape[-1], static.shape[-1], Y.shape[-1]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        if val is not None:
            vs, vst, vY = val
            Vs, Vt, VY = self._to((vs - self.seq_mu) / self.seq_sd, (vst - self.st_mu) / self.st_sd, (np.log1p(vY) - self.y_mu) / self.y_sd)
        best, best_state, bad = np.inf, None, 0
        n, bs = len(Xs), self.p["batch_size"]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.p["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs].to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(Xs[b], Xt[b]), Yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); tot += loss.item() * len(b)
            tr_loss = tot / n
            if val is None:
                self.history.append((ep, tr_loss, None)); continue
            self.net.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(self._predict_t(Vs, Vt), VY).item()
            self.history.append((ep, tr_loss, vl)); sched.step(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.p["patience"]: break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _predict_t(self, Xs, Xt, bs=2048):
        outs = [self.net(Xs[i:i + bs], Xt[i:i + bs]) for i in range(0, len(Xs), bs)]
        return torch.cat(outs)

    def predict(self, seq, static):
        self.net.eval()
        Xs, Xt = self._to((seq - self.seq_mu) / self.seq_sd, (static - self.st_mu) / self.st_sd)
        with torch.no_grad():
            z = self._predict_t(Xs, Xt).cpu().numpy()
        return np.expm1(z * self.y_sd + self.y_mu)


# ----------------------------------------------------------------------------- data prep
def make_sequences(df: pd.DataFrame, seq_cols, static_df: pd.DataFrame, Y: pd.DataFrame, lookback: int):
    """Build (seq[N, L, C], static[N, S], Y[N, H], index) for rows in Y.index whose full window is observed."""
    vals = df[seq_cols].to_numpy(np.float32)
    pos = df.index.get_indexer(Y.index)
    ok = pos >= lookback - 1
    windows = np.lib.stride_tricks.sliding_window_view(vals, lookback, axis=0)  # [T-L+1, C, L]
    seq = windows[pos[ok] - lookback + 1].transpose(0, 2, 1)
    keep = ~np.isnan(seq).any(axis=(1, 2))
    idx = Y.index[ok][keep]
    return (seq[keep], static_df.loc[idx].to_numpy(np.float32), Y.loc[idx].to_numpy(np.float32), idx)
