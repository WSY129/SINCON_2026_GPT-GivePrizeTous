"""v3 origin-filtered schedule facts and narrowly specified historical masks.

Schedules describe announced plans, never realized attendance/cancellations.
School/time-of-exit variables are explicit proxies, not measured future facts.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

OUTAGE_KEYS = {
    ('2025-08-13', 3112, 12), ('2025-08-13', 3112, 13),
    *{('2026-01-14', 3222, h) for h in range(19, 24)},
    ('2026-01-18', 3129, 22), ('2026-02-09', 3123, 23),
}

def oneoff_mask(frame):
    dates = pd.to_datetime(frame.date)
    keys = pd.MultiIndex.from_arrays([dates.dt.strftime('%Y-%m-%d'), frame.station_id, frame.hour])
    outage = keys.isin(list(OUTAGE_KEYS))
    opening = frame.station_id.isin([3107, 3108, 3109]) & dates.between('2025-06-28', '2025-06-30')
    return pd.Series(outage | opening, index=frame.index)

@lru_cache(maxsize=1)
def schedules():
    return json.loads(Path(__file__).with_name('domain_schedule_v3.json').read_text())

def add_domain_features(frame, as_of):
    out = frame.copy()
    origin = pd.Timestamp(as_of)
    if origin > pd.Timestamp('2026-06-30'):
        raise ValueError('Forecast origin exceeds competition information cutoff')
    d = pd.to_datetime(out.date); sid = out.station_id; hour = out.hour
    dow = d.dt.dayofweek; month = d.dt.month
    scheduled = np.zeros(len(out), bool); covered = np.zeros(len(out), bool)
    for record in schedules()['kbo']:
        if pd.Timestamp(record['known_since']) <= origin:
            scheduled |= d.isin(pd.to_datetime(record['home_dates'])).to_numpy()
            covered |= d.between(record['coverage_start'], record['coverage_end']).to_numpy()
    munhak = sid.eq(3127)
    out['munhak_kbo_scheduled'] = munhak & scheduled
    out['munhak_schedule_known'] = munhak & covered
    out['munhak_inseason_unannounced'] = munhak & month.between(3, 10) & ~covered
    holiday = out.is_holiday.astype(bool)
    night = ((dow < 5) & ~holiday) | (month.isin([7, 8]) & dow.eq(5))
    out['munhak_exit_peak_proxy'] = np.where(munhak & scheduled,
        np.where(night & hour.between(20, 22), 2,
                 np.where(~night & hour.between(17, 21), 1, 0)), 0)
    for kind, start_hour in [('pentaport', 18), ('beer', 19)]:
        active = np.zeros(len(out), bool)
        for record in schedules()['festivals']:
            if record['kind'] == kind and pd.Timestamp(record['known_since']) <= origin:
                active |= d.between(record['start'], record['end']).to_numpy()
        out[f'songdo_{kind}_scheduled'] = sid.eq(3139) & active
        out[f'songdo_{kind}_evening_proxy'] = sid.eq(3139) & active & hour.ge(start_hour)
    # Lunar flags already pass the holiday calendar's publication filter.
    out['memorial_park_lunar_peak'] = np.where(sid.eq(3122) & hour.between(9, 17),
        np.where(out.lunar_day_0.eq(1), 2, np.where((out.lunar_m1 + out.lunar_p1) > 0, 1, 0)), 0)
    md = d.dt.month * 100 + d.dt.day
    university_break = md.between(622, 831) | (md >= 1222) | (md <= 228)
    highschool_break = md.between(720, 816) | md.between(101, 228)
    weekday = (dow < 5) & ~holiday
    out['university_semester_proxy'] = sid.isin([3136, 3133, 3115]) & ~university_break & weekday
    out['highschool_dismissal_proxy'] = sid.eq(3208) & ~highschool_break & weekday & hour.between(15, 16)
    out['is_external_transfer'] = sid.isin([3110, 3207, 3120, 3218, 3130])
    return out
