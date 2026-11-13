"""Scheduled-event overlay for stations whose ordinary demand the base ensemble already models well.

Covers concerts at Incheon Asiad Main Stadium (stations 3209 and 3210), Bucheon FC home matches
(3754) and the Pentaport festival (3139). Everything is computed from observations up to the origin
and from event records whose ``known_since`` is not later than the origin.

Two operations, in this order:

1. De-spike. A one-off event observed last year lifts the year-ago features of the same weekend this
   year. For a verified past concert or festival on day e, the base prediction at e + 364 days is capped,
   in the hours the event lifted, at the base level of neighbouring same-weekday dates. League football
   is left alone: home matches recur every other week, so a year-ago match is not a one-off.
2. Lift. For an announced future event the prediction becomes ordinary level + shrink * lift, where the
   lift is the mean log excess that comparable past events showed over the ordinary same-weekday level
   around them. Football lifts are aligned on the hour the crowd leaves (kick-off + 2 h); concert and
   festival lifts are aligned on the clock and shifted when the announced start times differ.
   An event record may carry ``lift_scale`` < 1 when no comparable past event exists (see the schedule file).
"""
import numpy as np
import pandas as pd
from ..holiday_calendar import calendar_features

HOURS = np.arange(6, 24)
PARAMS = dict(shrink=0.85, min_lift=0.2, despike_min_lift=0.5, ordinary_window=35, min_ordinary_days=3,
              neighbour_weeks=(-2, -1, 1, 2), min_family_events=3, football_exit_offset=2.0,
              despike_kinds=('concert', 'festival'))


def hour_of(text):
    """'18:30' -> 18.5 ; None stays None."""
    if text is None or (isinstance(text, float) and np.isnan(text)): return None
    hh, mm = str(text).split(':'); return int(hh) + int(mm) / 60


def shift_profile(lift, delta):
    """Move an 18-hour profile ``delta`` hours later (fractional hours interpolate between neighbours)."""
    lo = int(np.floor(delta)); frac = float(delta - lo)
    def moved(k):
        out = np.full(18, np.nan); src = np.arange(18) - k; ok = (src >= 0) & (src < 18); out[ok] = lift[src[ok]]; return out
    a = moved(lo)
    if frac < 1e-9: return a
    b = moved(lo + 1)
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (1 - frac) * a + frac * b))


class EventOverlay:
    def __init__(self, events, **overrides):
        self.p = dict(PARAMS, **overrides)
        self.events = [dict(e, date=pd.Timestamp(e['date']), known=pd.Timestamp(e['known_since']), start_hour=hour_of(e.get('start'))) for e in events]

    # ------------------------------------------------------------ helpers
    def _align(self, event):
        """Integer hour used to align lifts of comparable events (None = align on the clock)."""
        if event['kind'] == 'football' and event['start_hour'] is not None:
            return int(np.floor(event['start_hour'] + self.p['football_exit_offset']))
        return event['start_hour']                                # concerts / festivals: fractional clock hour, may be None

    def fit(self, train, origin):
        """Lift of every observed event at every station it touches: observed log minus ordinary same-weekday median."""
        p = self.p; origin = pd.Timestamp(origin); self.origin = origin
        stations = sorted({s for e in self.events for s in e['stations']})
        h = train.loc[train.station_id.isin(stations) & (train.date <= origin), ['date', 'station_id', 'hour', 'boardings']].copy()
        h['z'] = np.log1p(h.boardings)
        dates = pd.DatetimeIndex(sorted(h.date.unique()))
        holiday = set(dates[calendar_features(pd.Series(dates), as_of=origin).is_holiday.to_numpy(bool)])
        self.lifts = {}
        for s in stations:
            hs = h.loc[h.station_id == s]
            table = hs.pivot(index='date', columns='hour', values='z').reindex(columns=HOURS)
            blocked = {e['date'] for e in self.events if s in e['stations']} | holiday
            for i, e in enumerate(self.events):
                if s not in e['stations'] or e['date'] > origin or e['date'] not in table.index: continue
                window = (table.index >= e['date'] - pd.Timedelta(days=p['ordinary_window'])) & (table.index <= e['date'] + pd.Timedelta(days=p['ordinary_window']))
                same = window & (table.index.dayofweek == e['date'].dayofweek) & ~table.index.isin(list(blocked))
                if same.sum() < p['min_ordinary_days']: continue
                self.lifts[(s, i)] = (table.loc[e['date']] - table.loc[same].median()).to_numpy()
        return self

    def _profile(self, station, target):
        """Mean aligned lift of comparable observed events; same family first, then same kind at this station."""
        p = self.p
        pool = [(i, e) for i, e in enumerate(self.events) if (station, i) in self.lifts and e['kind'] == target['kind']]
        family = [(i, e) for i, e in pool if e.get('family') == target.get('family')]
        need = p['min_family_events'] if target['kind'] == 'football' else 1
        chosen = family if len(family) >= need else pool
        if not chosen: return None, 0, 'none'
        ta = self._align(target); rows = []
        for i, e in chosen:
            lift = self.lifts[(station, i)]; ea = self._align(e)
            rows.append(shift_profile(lift, 0 if (ta is None or ea is None) else ta - ea))
        with np.errstate(all='ignore'):
            profile = np.nanmean(np.array(rows), axis=0)
        return np.nan_to_num(profile), len(chosen), ('family' if chosen is family else 'kind')

    def family_summary(self, station):
        """Observed peak lift per event family at one station: count, mean log lift at the peak hour, and its ratio."""
        rows = {}
        for (s, i), lift in self.lifts.items():
            if s != station: continue
            e = self.events[i]
            if e['kind'] == 'football' and e['start_hour'] is not None:
                k = int(np.floor(e['start_hour'] + self.p['football_exit_offset'])) - int(HOURS[0])
                peak = float(lift[k]) if 0 <= k < 18 else float(np.nanmax(lift))
            else:
                peak = float(np.nanmax(lift))
            rows.setdefault(e.get('family', ''), []).append(peak)
        return {f: dict(events=len(v), mean_peak_log_lift=float(np.mean(v)), ratio=float(np.exp(np.mean(v)))) for f, v in sorted(rows.items())}

    # -------------------------------------------------------------- apply
    def apply(self, query, base_log):
        """Return (corrected log predictions as a full array, report list). ``query`` needs date/station_id/hour."""
        p = self.p; origin = self.origin
        out = np.asarray(base_log, float).copy(); report = []
        stations = sorted({s for e in self.events for s in e['stations']})
        q = query.loc[query.station_id.isin(stations), ['date', 'station_id', 'hour']].copy()
        if q.empty: return out, report
        q['pos'] = query.index.get_indexer(q.index)
        dates = pd.DatetimeIndex(sorted(q.date.unique()))
        holiday = set(dates[calendar_features(pd.Series(dates), as_of=origin).is_holiday.to_numpy(bool)])
        future = [(i, e) for i, e in enumerate(self.events) if e['date'] > origin and e['known'] <= origin]
        for s in stations:
            qs = q.loc[q.station_id == s]
            if qs.empty: continue
            grid = qs.pivot(index='date', columns='hour', values='pos').reindex(columns=HOURS)
            scheduled = {e['date'] for _, e in future if s in e['stations']}
            echo = {}                                             # date -> lift of the event one year (364 days) earlier
            for (st, i), lift in self.lifts.items():
                if st == s and self.events[i]['kind'] in p['despike_kinds']:
                    d = self.events[i]['date'] + pd.Timedelta(days=364)
                    if d in grid.index: echo[d] = np.maximum(echo.get(d, 0), lift)
            flagged = set(echo) | scheduled | holiday

            def neighbour_level(day):
                near = [day + pd.Timedelta(days=7 * w) for w in p['neighbour_weeks']]
                near = [d for d in near if d in grid.index and d not in flagged]
                if not near: return None, []
                return np.mean([out[grid.loc[d].to_numpy(int)] for d in near], axis=0), near

            for day, lift in echo.items():                        # 1. de-spike the year-ago echo
                hours = lift > p['despike_min_lift']
                level, near = neighbour_level(day)
                if level is None or not hours.any(): continue
                pos = grid.loc[day].to_numpy(int); before = out[pos].copy()
                out[pos[hours]] = np.minimum(out[pos[hours]], level[hours])
                report.append(dict(station=int(s), date=str(day.date()), action='despike', hours=int(hours.sum()),
                                   mean_change=float((out[pos] - before)[hours].mean()), neighbours=[str(d.date()) for d in near]))
            for i, e in future:                                   # 2. lift announced events
                if s not in e['stations'] or e['date'] not in grid.index: continue
                profile, n, how = self._profile(s, e)
                if profile is None: continue
                hours = profile > p['min_lift']
                pos = grid.loc[e['date']].to_numpy(int); before = out[pos].copy()
                level, _ = neighbour_level(e['date'])                # the lift is defined over the ordinary level, so an
                if level is not None:                               # echo of last year's event in the base is removed first
                    out[pos[hours]] = np.minimum(out[pos[hours]], level[hours])
                out[pos[hours]] = out[pos[hours]] + p['shrink'] * float(e.get('lift_scale', 1.0)) * profile[hours]
                report.append(dict(station=int(s), date=str(e['date'].date()), action='lift', event=e.get('name', ''), family=e.get('family', ''),
                                   comparable_events=int(n), matched_by=how, hours=int(hours.sum()), max_lift=float(profile.max()),
                                   mean_change=float((out[pos] - before)[hours].mean()) if hours.any() else 0.0))
        return out, report
