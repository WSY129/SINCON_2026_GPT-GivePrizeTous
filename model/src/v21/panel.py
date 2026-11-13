"""Panel21 = Panel15 + the weekend lunar-holiday label fix (src.v21.kinds) + two drift inputs computed from data up to the cutoff only:
  rec_bias   mean residual of the cutoff anchor over the last 28 ordinary weekdays (weekday rows) or 56 days of ordinary weekends (weekend rows), per station x hour
  rec_trend  weekday mean residual of the last 14 days minus that of days 15-42 before the cutoff, per station x hour
Nothing after the cutoff is read: the residual array r is NaN after the cutoff (src.v11.panel)."""
import warnings
import numpy as np
from src.v8 import core as c8
from src.v11.panel import HORIZON
from src.v14.panel import FEATURES14, N_FIRST
from src.v15.panel import Panel15
from . import kinds

kinds.apply()
NEW = ['rec_bias', 'rec_trend']
FEATURES21 = FEATURES14 + NEW
COLS21 = list(range(N_FIRST)) + [len(FEATURES14) + i for i in range(len(NEW))]
NAMES21 = [FEATURES21[i] for i in COLS21]


def _mean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); return np.nanmean(a, axis=axis)


class Panel21(Panel15):
    def build(self, origin, horizon=HORIZON):
        d = super().build(origin, horizon); O = d['O']; r = d['r'].astype(np.float32).copy(); nd = r.shape[1]
        if d.get('mask') is not None:
            r[np.asarray(d['mask'])[:, :nd]] = np.nan
        r[:, O + 1:] = np.nan                                                    # guard: nothing after the cutoff
        kinds_ = np.asarray(d['kinds']); dow = np.asarray(self.p.dow[:nd]); idx = np.arange(nd)
        ordinary = np.isin(kinds_, c8.ORDINARY) & (idx <= O)
        wd = ordinary & (dow < 5); we = ordinary & (dow >= 5)
        rec_wd = _mean(r[:, wd & (idx > O - 28)], 1); rec_we = _mean(r[:, we & (idx > O - 56)], 1)
        trend = _mean(r[:, wd & (idx > O - 14)], 1) - _mean(r[:, wd & (idx <= O - 14) & (idx > O - 42)], 1)
        rec = np.where((dow < 5)[None, :, None], rec_wd[:, None, :], rec_we[:, None, :])
        new = np.stack([rec, np.broadcast_to(trend[:, None, :], rec.shape)], -1).astype(np.float32)
        assert d['X'].shape[-1] == len(FEATURES14), d['X'].shape
        d['X'] = np.concatenate([d['X'], new], -1)
        return d
