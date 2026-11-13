"""Inference path of ensemble_v7: base blend -> (v6 specialists with v7 options) -> new-station growth drift.

The v6 classes are imported unchanged; v7 options are passed as overrides or small subclasses so that the same
function serves every validation origin and the final forecast.
"""
import numpy as np
import pandas as pd
from src.v6.munhak import MunhakSpecialist, SID as MUNHAK
from src.v6.events import EventOverlay
from src.v6.schedule import STATE_NAMES, UNANNOUNCED


class MunhakV7(MunhakSpecialist):
    """Unannounced dates: Monday (the league's rest day) gets its own probability inside the weekday group.

    The weekday-group probability of v6 is multiplied by rate(Monday)/rate(all weekdays) or
    rate(Tue-Fri)/rate(all weekdays), both measured on crowd days observed up to the origin and shrunk towards 1,
    so the expected number of crowd days over a week stays the same.
    """
    monday_strength = 6.0

    def fit(self, train, origin):
        super().fit(train, origin)
        hist = self._hist
        dates = pd.DatetimeIndex(hist['dates']); weekday = np.asarray(hist['group']) == 0
        season = (dates.month * 100 + dates.day >= 320) & (dates.month * 100 + dates.day <= 1031)
        use = weekday & season; spike = np.asarray(hist['spike'], bool)
        monday = use & (dates.dayofweek == 0); other = use & (dates.dayofweek != 0)
        rate_all = spike[use].mean() if use.any() else 0.0
        k = self.monday_strength
        rate = lambda m: (spike[m].sum() + k * rate_all) / (m.sum() + k) if rate_all > 0 else 0.0
        self.monday_ratio = float(rate(monday) / rate_all) if rate_all > 0 else 1.0
        self.other_ratio = float(rate(other) / rate_all) if rate_all > 0 else 1.0
        self.audit['monday_split'] = dict(weekday_rate=round(float(rate_all), 4), mondays=int(monday.sum()),
                                          monday_crowd_days=int(spike[monday].sum()), other_weekdays=int(other.sum()),
                                          other_crowd_days=int(spike[other].sum()),
                                          monday_ratio=round(self.monday_ratio, 4), other_ratio=round(self.other_ratio, 4))
        return self

    def _unannounced_probability(self, date, group):
        base = super()._unannounced_probability(date, group)
        if group != 0:
            return base
        return float(np.clip(base * (self.monday_ratio if date.dayofweek == 0 else self.other_ratio), 0, 0.95))


def apply_specialists(train, query, base_log, origin, schedule, use=('munhak', 'events'), munhak_cls=MunhakSpecialist,
                      overlay=None):
    """Same contract as src.v6.pipeline.apply_specialists, with a pluggable Munhak class and overlay overrides."""
    origin = pd.Timestamp(origin)
    query = query.reset_index(drop=True)
    out = np.asarray(base_log, float).copy(); details = {}
    if 'munhak' in use:
        model = munhak_cls(schedule).fit(train, origin)
        dates = pd.DatetimeIndex(sorted(query.loc[query.station_id == MUNHAK, 'date'].unique()))
        if len(dates):
            table = model.predict(dates).stack().rename('z').rename_axis(['date', 'hour']).reset_index()
            rows = query.loc[query.station_id == MUNHAK, ['date', 'hour']].reset_index().merge(table, on=['date', 'hour'], how='left')
            act = rows.z.notna().to_numpy()
            out[rows['index'].to_numpy()[act]] = rows.z.to_numpy()[act]
            details['munhak_states'] = model.last.assign(state_name=model.last.state.map(STATE_NAMES))
            details['munhak_rows_replaced'] = int(act.sum())
        details['munhak_fit'] = model.audit
    if 'events' in use:
        model = EventOverlay(schedule.overlay_events(), **(overlay or {})).fit(train, origin)
        out, report = model.apply(query, out)
        details['event_report'] = report
    return np.maximum(out, 0), details
