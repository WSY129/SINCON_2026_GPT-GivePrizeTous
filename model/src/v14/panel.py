"""Panel of v11 plus exogenous inputs that the v13 residual analysis showed to matter (or that replace noisy ones).

Added per cell [station, day, hour]:
  school    vac            share of middle / high schools on vacation that day (NEIS calendars; Jan-Feb 2025 is outside the
                           collected windows and is filled with the share 364 days later, i.e. the same weekday of winter 2025-26)
  summer    leave_week     weekdays of the week holding 1 August and of the following week (collective summer leave)
  calendar  bridge, block_len, block_pos, before_off, after_off, days_to_mj, mj_day
  events    ev_concert, ev_sports, ev_festival, ev_kovo, ev_art, ev_bac, ev_start_rel (hour minus published start hour)
  weather   k_rain_h, k_rain_prev3, k_rain_day, k_snow, k_temp_h, k_tmax (KMA station 112 observations), w_rain, w_heat (warning level)
  priors    cal_prior      day ratio expected from the calendar position alone, week_prior  seasonal index of the ISO week by day type;
                           both learned from daily card data of other stations in years before 2025 only (src.v14.priors)
None of them is ridership of the 71 stations.  Realised events and observed weather of the evaluation period are allowed by the organisers.
"""
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11.panel import Panel11, FEATURES, HORIZON
from src.v13.run import load_events
from .priors import CalendarPrior

EXT = c8.ROOT / 'outputs/ensemble_v14/external'
EXT13 = c8.ROOT / 'outputs/ensemble_v13/external'
EXTRA = ['vac', 'leave_week', 'bridge', 'block_len', 'block_pos', 'before_off', 'after_off', 'days_to_mj', 'mj_day',
         'ev_concert', 'ev_sports', 'ev_festival', 'ev_kovo', 'ev_art', 'ev_bac', 'ev_start_rel',
         'k_rain_h', 'k_rain_prev3', 'k_rain_day', 'k_snow', 'k_temp_h', 'k_tmax', 'w_rain', 'w_heat', 'cal_prior', 'week_prior']
N_FIRST = len(FEATURES) + 24          # inputs of lgb_c / lgb_d (trained before the two priors were added)
FEATURES14 = FEATURES + EXTRA
EV_KINDS = {'ev_concert': 'concert', 'ev_sports': 'sports', 'ev_festival': 'festival', 'ev_kovo': 'kovo', 'ev_art': 'art', 'ev_bac': 'bac'}


def vacation_share():
    sv = pd.read_csv(EXT13 / 'school_vacation_daily.csv'); sv = sv[(sv.area == '전체') & (sv.level == '전체')].set_index('date').share_vacation
    days = pd.date_range('2025-01-01', '2026-10-31'); s = sv.reindex([str(d)[:10] for d in days]).fillna(0.0); s.index = days
    early = pd.date_range('2025-01-01', '2025-03-03')                      # winter 2024-25 was not collected: same weekday of the next winter
    s.loc[early] = s.reindex(early + pd.Timedelta(days=364)).to_numpy()
    return s


def leave_weeks(days):
    out = np.zeros(len(days))
    for y in sorted(set(days.year)):
        a = pd.Timestamp(f'{y}-08-01'); mon = a - pd.Timedelta(days=a.dayofweek)
        out[(days >= mon) & (days < mon + pd.Timedelta(days=12)) & (days.dayofweek < 5)] = 1.0
    return out


class Panel14(Panel11):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        days = pd.DatetimeIndex(self.days); ds = [str(d)[:10] for d in days]; S, H = self.S, self.H; n = len(days)
        cal = pd.read_csv(EXT / 'calendar.csv', encoding='utf-8-sig').set_index('date').reindex(ds)
        day = dict(vac=vacation_share().reindex(days).fillna(0.0).to_numpy() * (days.dayofweek < 5), leave_week=leave_weeks(days),
                   bridge=cal.is_bridge_day.to_numpy(float), block_len=cal.offday_block_len.to_numpy(float), block_pos=cal.offday_block_pos.to_numpy(float),
                   before_off=cal.is_before_offday.to_numpy(float), after_off=cal.is_after_offday.to_numpy(float),
                   days_to_mj=cal.days_to_mj.clip(-8, 8).to_numpy(float), mj_day=cal.mj_day.to_numpy(float))
        E = {k: np.broadcast_to(v[None, :, None], (S, n, H)).astype(np.float32) for k, v in day.items()}
        wh = pd.read_csv(EXT / 'weather_hourly.csv', encoding='utf-8-sig'); wh = wh[(wh.hour >= 6) & (wh.hour <= 23)]
        for name, col in (('k_rain_h', 'precip'), ('k_rain_prev3', 'precip_prev3h'), ('k_rain_day', 'daily_precip'), ('k_snow', 'snow_depth'), ('k_temp_h', 'temp'), ('k_tmax', 'daily_tmax')):
            E[name] = np.broadcast_to(wh.pivot(index='date', columns='hour', values=col).reindex(ds).to_numpy(np.float32)[None], (S, n, H))
        ww = pd.read_csv(EXT / 'weather_warnings_hourly.csv', encoding='utf-8-sig'); ww = ww[(ww.hour >= 6) & (ww.hour <= 23)]
        region = pd.read_csv(EXT / 'station_info.csv', encoding='utf-8-sig').set_index('station_id').warning_region
        buc = np.array([region.get(int(s), 'incheon') == 'bucheon' for s in self.p.stations])
        for name, col in (('w_rain', 'heavy_rain'), ('w_heat', 'heat')):
            a = ww.pivot(index='date', columns='hour', values=f'incheon_{col}').reindex(ds).to_numpy(np.float32); b = ww.pivot(index='date', columns='hour', values=f'bucheon_{col}').reindex(ds).to_numpy(np.float32)
            E[name] = np.where(buc[:, None, None], b[None], a[None])
        ev = load_events(self.p.stations); idx = {d: i for i, d in enumerate(ds)}; sidx = {int(s): j for j, s in enumerate(self.p.stations)}
        for name in EV_KINDS:
            E[name] = np.zeros((S, n, H), np.float32)
        rel = np.full((S, n, H), np.nan, np.float32); hours = np.arange(6, 24, dtype=np.float32)
        for (sid, d), e in ev.items():
            if d in idx:
                for name, kind in EV_KINDS.items():
                    if kind in e['kinds']:
                        E[name][sidx[sid], idx[d]] = 1.0
                if np.isfinite(e['start']):
                    rel[sidx[sid], idx[d]] = hours - e['start']
        E['ev_start_rel'] = rel
        cp = CalendarPrior(); ok = np.array([d <= pd.Timestamp('2026-10-31') for d in days]); prior = np.zeros(n); prior[ok] = cp.fit('2025-01-01')(days[ok])
        W = cp.weekly_index(2025); dt = np.where(days.dayofweek < 5, 0, np.where(days.dayofweek == 5, 1, 2)); wk = np.minimum(days.isocalendar().week.to_numpy().astype(int), 52)
        E['cal_prior'] = np.broadcast_to(prior[None, :, None], (S, n, H)).astype(np.float32); E['week_prior'] = np.broadcast_to(W[dt, wk][None, :, None], (S, n, H)).astype(np.float32)
        self.extra = np.stack([np.asarray(E[k], np.float32) for k in EXTRA], -1)          # [S, all days, H, extra]

    def build(self, origin, horizon=HORIZON):
        d = super().build(origin, horizon); nd = d['X'].shape[1]
        d['X'] = np.concatenate([d['X'], self.extra[:, :nd]], -1)
        return d
