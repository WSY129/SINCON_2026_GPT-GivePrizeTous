"""v10 seasonal index: the v8 module with three corrections.

1. Station-level coefficients.  v8 used one coefficient for every station; here each station gets its own,
   shrunk towards the common one and clipped to [0, 1.5], so a station where the index made things worse in the
   other folds (귤현 after the 2025 line extension) is switched off by the data.
2. Event-month gate.  The operator's monthly totals of 송도달빛축제공원 (3139) in August contain the festival
   crowds of earlier years; the festivals are already added day by day, so the August index is not applied there.
3. Line 7 history.  The operator's file starts in 2022 for the Line 7 stations; Seoul's monthly station x hour
   counts 2015-2021 (team share) extend it for the stations whose role did not change in 2021.
"""
import numpy as np
import pandas as pd
from src.v8 import core as c8, pre

GATE = {(3139, 8)}                                  # (station, month) where the index is not applied
LINE7_OK = [3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760]      # 부평구청·산곡·석남 changed role in 2021-05
SHRINK = 0.25                                       # pseudo-weight of the common coefficient, in units of the median station weight


class History10(pre.History):
    def __init__(self, panel, line7=True):
        super().__init__(panel)
        self.line7 = line7
        if line7:
            d = pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/external/seoul_line7_monthly_hourly_2015_2021.csv')
            d = d[d.station_id.isin(LINE7_OK) & d.hour.between(6, 23)]
            y = d.month.str[:4].astype(int).to_numpy() - 2015; m = d.month.str[5:].astype(int).to_numpy() - 1
            s = np.searchsorted(panel.stations, d.station_id.to_numpy()); h = d.hour.to_numpy() - 6
            fill = np.isnan(self.B[s, h, y, m]) & (d.boardings.to_numpy() > 0)
            self.B[s[fill], h[fill], y[fill], m[fill]] = d.boardings.to_numpy()[fill]
            self.added = int(fill.sum())


def delta(history, block, pred):
    d = pre.seasonal_feature(history, block, pred)
    for sid, month in GATE:
        d[np.asarray(block.dates.month == month), int(np.searchsorted(block.panel.stations, sid))] = 0.0
    return d


def station_coefficients(deltas, resid, shrink=SHRINK):
    """deltas, resid: lists of dense arrays [day, S, H] (the training folds). Returns (lambda per station, common)."""
    num = sum((-r * d).sum(axis=(0, 2)) for d, r in zip(deltas, resid))
    den = sum((d * d).sum(axis=(0, 2)) for d in deltas)
    common = float(num.sum() / max(den.sum(), 1e-12))
    kappa = shrink * float(np.median(den[den > 0])) if (den > 0).any() else 0.0
    lam = (num + kappa * common) / (den + kappa + 1e-12)
    return np.clip(lam, 0.0, 1.5), common


GATE_AT = 0.3          # a station whose own (shrunk) coefficient falls below this is switched off; the rest share one coefficient


def gated(lam, common, mode):
    if mode == 'station':
        return lam
    if mode == 'gate':
        return np.where(lam >= GATE_AT, np.clip(common, 0, 1.5), 0.0)
    return np.full_like(lam, np.clip(common, 0, 1.5))


def apply(folds, cur, history, test=None, tcur=None, by_station='gate'):
    """Nested application.  Returns (new fold forecasts, info, new test forecast or None)."""
    deltas = {k: delta(history, b, cur[k]) for k, b in folds.items()}
    resid = {k: cur[k] - b.truth for k, b in folds.items()}
    new, info = {}, {}
    for k in folds:
        others = [j for j in folds if j != k]
        lam, common = station_coefficients([deltas[j] for j in others], [resid[j] for j in others])
        lam = gated(lam, common, by_station)
        new[k] = cur[k] + lam[None, :, None] * deltas[k]
        info[k] = dict(common=round(common, 4), stations_off=[int(s) for s in folds[k].panel.stations[(lam < 0.05) & ((deltas[k] != 0).any(axis=(0, 2)))]])
    lam, common = station_coefficients(list(deltas.values()), list(resid.values()))
    lam = gated(lam, common, by_station)
    out = dict(nested=info, common_all_folds=round(common, 4), lambda_by_station={int(s): round(float(x), 3) for s, x in zip(next(iter(folds.values())).panel.stations, lam)})
    tnew = None
    if test is not None:
        tnew = tcur + lam[None, :, None] * delta(history, test, tcur)
    return new, out, tnew
