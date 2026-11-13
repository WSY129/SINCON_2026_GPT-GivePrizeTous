"""Panel of v14 with (1) the event inputs rebuilt from the corrected event list of the team share v4 and (2) three known
bad station-days taken out of the training targets (a counting error and two service interruptions).  The validation
truth is not touched: the folds are still scored on every cell."""
import numpy as np
import pandas as pd
from src.v11.panel import HORIZON
from src.v14.panel import Panel14, EXTRA, EV_KINDS, FEATURES14, N_FIRST
from .events import load_events, ANOMALIES


class Panel15(Panel14):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        days = pd.DatetimeIndex(self.days); idx = {str(d)[:10]: i for i, d in enumerate(days)}; sidx = {int(s): j for j, s in enumerate(self.p.stations)}
        ev = load_events(self.p.stations); hours = np.arange(6, 24, dtype=np.float32)
        for name in list(EV_KINDS) + ['ev_start_rel']:
            self.extra[..., EXTRA.index(name)] = np.nan if name == 'ev_start_rel' else 0.0
        for (sid, d), e in ev.items():
            if d in idx:
                for name, kind in EV_KINDS.items():
                    if kind in e['kinds']:
                        self.extra[sidx[sid], idx[d], :, EXTRA.index(name)] = 1.0
                if np.isfinite(e['start']):
                    self.extra[sidx[sid], idx[d], :, EXTRA.index('ev_start_rel')] = hours - e['start']
        self.bad = [(sidx[s], idx[d], lo - 6, hi - 6 + 1) for s, d, lo, hi in ANOMALIES if d in idx]

    def build(self, origin, horizon=HORIZON):
        d = super().build(origin, horizon); O = d['O']
        for s, i, a, b in self.bad:
            if i <= O:                                                       # training rows only
                d['r'][s, i, a:b] = np.nan
        return d
