"""Shared inference path of ensemble_v6: base blend -> Munhak specialist -> scheduled-event overlay.

The same function is used for the final test forecast and for every validation origin, so validation
exercises exactly the code that produces the submission.
"""
import numpy as np
import pandas as pd
from .munhak import MunhakSpecialist, SID as MUNHAK
from .events import EventOverlay
from .schedule import STATE_NAMES


def apply_specialists(train, query, base_log, origin, schedule, use=('munhak', 'events')):
    """query: frame with date/station_id/hour (any index). Returns (log predictions array, details dict)."""
    origin = pd.Timestamp(origin)
    query = query.reset_index(drop=True)
    out = np.asarray(base_log, float).copy(); details = {}
    if 'munhak' in use:
        model = MunhakSpecialist(schedule).fit(train, origin)
        dates = pd.DatetimeIndex(sorted(query.loc[query.station_id == MUNHAK, 'date'].unique()))
        if len(dates):
            table = model.predict(dates).stack().rename('z').rename_axis(['date', 'hour']).reset_index()
            rows = query.loc[query.station_id == MUNHAK, ['date', 'hour']].reset_index().merge(table, on=['date', 'hour'], how='left')
            act = rows.z.notna().to_numpy()
            out[rows['index'].to_numpy()[act]] = rows.z.to_numpy()[act]
            details['munhak_states'] = model.last.assign(state=model.last.state.map(STATE_NAMES))
            details['munhak_rows_replaced'] = int(act.sum())
        details['munhak_fit'] = model.audit
    if 'events' in use:
        overlay = EventOverlay(schedule.overlay_events()).fit(train, origin)
        out, report = overlay.apply(query, out)
        details['event_report'] = report
    return np.maximum(out, 0), details
