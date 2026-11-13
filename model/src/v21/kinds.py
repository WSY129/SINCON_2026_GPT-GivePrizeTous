"""v21 calendar fix: a Saturday or Sunday next to Chuseok / Seollal is labelled 'long_sat' / 'long_sun' instead of 'lunar_adj'.

Why: `src.v8.core.day_kinds` gives every holiday-adjacent lunar day the kind 'lunar_adj', whatever its weekday.  The holiday
effect of 'lunar_adj' is learned mostly from weekdays (commuting falls away), and the anchor adds it to the weekend level
of that weekday, so the weekend is reduced twice.  On 2025-10-05 (Sunday before Chuseok, fold 1) the v15 trees were
-0.57 log too low (commute hours -1.54) while v10 was +0.07.  The evaluation period has one such day: 2026-09-26 (Sat).
`apply()` patches the module attribute, which every caller uses (c8.day_kinds, core.day_kinds, the Block class)."""
import numpy as np
import pandas as pd
from src.v8 import core as c8

_ORIG = getattr(c8, '_day_kinds_orig', c8.day_kinds)


def day_kinds(dates, origin):
    out = np.array(_ORIG(dates, origin), dtype=object); dow = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    out[(out == 'lunar_adj') & (dow == 5)] = 'long_sat'
    out[(out == 'lunar_adj') & (dow == 6)] = 'long_sun'
    return out.astype(str)


def apply():
    c8._day_kinds_orig = _ORIG
    c8.day_kinds = day_kinds
