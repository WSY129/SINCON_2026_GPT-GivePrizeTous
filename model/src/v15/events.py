"""Event table of v15 = the v13 table plus a short list of corrections that the training data confirm.

The team share v4 added about forty labelled days.  Putting all of them into the model as generic `festival` days made
the single model worse (Fold 1 0.2048 -> 0.2065): several of them leave no trace at the stations at night (a two-week
flower festival, a daytime science fair), and the model answers every `festival` flag with a night-time lift.  Only
the rows of `outputs/ensemble_v15/external/event_corrections.csv` are applied: two labels of days on which the event did
not take place at that station are removed, one three-day concert with a clear exit peak is added.
"""
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v13 import run as R

EXT15 = c8.ROOT / 'outputs/ensemble_v15/external'
ANOMALIES = [(3125, '2026-03-01', 6, 23), (3222, '2026-01-14', 19, 23), (3112, '2025-08-13', 11, 14)]     # counting error, two service interruptions: (station, day, first hour, last hour)


def corrections():
    return pd.read_csv(EXT15 / 'event_corrections.csv', dtype=dict(station_id=int, date=str))


def load_events(stations):
    ev = R.load_events(stations); known = {int(s) for s in stations}
    for r in corrections().itertuples():
        key = (int(r.station_id), r.date)
        if int(r.station_id) not in known:
            continue
        if r.action == 'remove':
            if key in ev:
                ev[key]['kinds'].discard(r.kind)
                if not ev[key]['kinds']:
                    del ev[key]
        else:
            e = ev.setdefault(key, dict(kinds=set(), start=np.nan)); e['kinds'].add(r.kind)
            if isinstance(r.start_time, str):
                e['start'] = R.hhmm(r.start_time)
    return ev
