"""Schedule-conditioned mixture specialist for the Munhak stadium station (3127).

The official metric is RMSLE, so the best point forecast is E[log1p(boardings)]. A stadium day is a
mixture of two very different regimes (a game is played / it is not), therefore

    HOME        : (1-q) * GAME + q * NORMAL            q = past cancellation rate of scheduled games
    NO_HOME     : (1-r) * NORMAL + r * EVENT           r = past rate of unscheduled crowd events
    UNANNOUNCED : p * GAME + (1-p) * NORMAL            p = last year's game-day share around the same dates
    RESERVE     : treated as UNANNOUNCED with a fixed prior
    CONCERT     : (1-c) * max(day mixture, NORMAL + shrink * CONCERT LIFT) + c * day mixture
                  for a concert at the main stadium that was announced before the origin; the lift is the
                  mean log excess over NORMAL on past concert days, aligned on the announced start hour

GAME / NORMAL / EVENT are hour profiles of the mean log boardings estimated from observations up to the
origin only. GAME is keyed by day group x season bucket (which fixes the announced start-time regime) and
shifted by a shrunken opponent effect. Nothing after the origin is read.
"""
import numpy as np
import pandas as pd
from ..holiday_calendar import calendar_features
from .schedule import Schedule, OFF, NO_HOME, UNANNOUNCED, RESERVE, HOME
from .events import shift_profile

SID = 3127
HOURS = np.arange(6, 24)
SPIKE = 1200            # daily max hourly boardings that marks a realised crowd event (history only)
# Season buckets; weekend start times differ by bucket, weekday games always start in the evening.
SPRING, JUNE, SUMMER, FALL, WINTER = 0, 1, 2, 3, 4
COMPATIBLE = {SPRING: (SPRING, FALL), FALL: (FALL, SPRING), JUNE: (JUNE,), SUMMER: (SUMMER,), WINTER: (SPRING, FALL)}
PARAMS = dict(normal_window=180, shrink_cell=3.0, shrink_opp=4.0, cancel_prior=0.06, cancel_strength=10.0,
              event_prior=0.03, event_strength=10.0, unannounced_strength=3.0, reserve_prob=0.4,
              october_taper=0.4, opponent_scale=1.5, concert_shrink=0.85, concert_cancel=0.03)


def _start_hour(text):
    if text is None: return None
    hh, mm = str(text).split(':'); return int(hh) + int(mm) / 60


def day_group(dates, holiday):
    dow = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    return np.where(np.asarray(holiday, bool) | (dow == 6), 2, np.where(dow == 5, 1, 0))


def bucket(dates):
    m = pd.DatetimeIndex(dates).month.to_numpy()
    return np.select([np.isin(m, [3, 4, 5]), m == 6, np.isin(m, [7, 8]), np.isin(m, [9, 10, 11])], [SPRING, JUNE, SUMMER, FALL], WINTER)


class MunhakSpecialist:
    def __init__(self, schedule=None, concerts=None, **overrides):
        self.schedule = schedule or Schedule(); self.p = dict(PARAMS, **overrides)
        self.concerts = [dict(c, date=pd.Timestamp(c['date']), known=pd.Timestamp(c['known_since']), start_hour=_start_hour(c.get('start')))
                         for c in (concerts if concerts is not None else self.schedule.venue_events('munhak_main'))]

    # ------------------------------------------------------------------ fit
    def fit(self, train, origin):
        origin = pd.Timestamp(origin); p = self.p
        h = train.loc[(train.station_id == SID) & (train.date <= origin)]
        Z = np.log1p(h.pivot(index='date', columns='hour', values='boardings').reindex(columns=HOURS)).dropna()
        dates = Z.index; z = Z.to_numpy()
        holiday = calendar_features(pd.Series(dates), as_of=origin).is_holiday.to_numpy(bool)
        g = day_group(dates, holiday); b = bucket(dates)
        state, opp = self.schedule.state(dates, origin)
        spike = np.expm1(z).max(axis=1) >= SPIKE
        game = (state == HOME) & spike
        covered = np.isin(state, [HOME, NO_HOME])
        normal = ~spike & (state != HOME)
        event = spike & (state == NO_HOME)
        self.origin = origin; self.audit = {}

        # NORMAL: ordinary days, recent window first, all history as fallback.
        recent = normal & (dates > origin - pd.Timedelta(days=p['normal_window']))
        self.normal = np.zeros((3, 18)); overall = z[normal].mean(axis=0)
        for k in range(3):
            a, c = recent & (g == k), normal & (g == k)
            self.normal[k] = z[a].mean(axis=0) if a.sum() >= 4 else z[c].mean(axis=0) if c.sum() >= 2 else overall

        # GAME: day group x bucket, shrunk to start-time-compatible buckets, then to the day group.
        self.game = np.zeros((3, 5, 18)); self.game_n = np.zeros((3, 5), int)
        for k in range(3):
            gk = game & (g == k)
            base = z[gk].mean(axis=0) if gk.any() else self.normal[k] + (z[game].mean(axis=0) - z[normal].mean(axis=0) if game.any() else 0)
            for bb in range(5):
                if k == 0:                                      # weekday evening games: one start-time regime all season
                    pooled, n_pool = base, gk.sum()
                else:
                    pool = gk & np.isin(b, COMPATIBLE[bb]); n_pool = pool.sum()
                    pooled = z[pool].mean(axis=0) if n_pool else base
                    pooled = (n_pool * pooled + p['shrink_cell'] * base) / (n_pool + p['shrink_cell']) if gk.any() else pooled
                cell = gk & (b == bb); n = cell.sum(); self.game_n[k, bb] = n
                self.game[k, bb] = (n * z[cell].mean(axis=0) + p['shrink_cell'] * pooled) / (n + p['shrink_cell']) if n else pooled

        # Opponent level effect on the lifted hours (attendance differs by visiting club).
        self.opp = {}
        if game.any():
            lift = np.clip(self.game[g[game], b[game]] - self.normal[g[game]], 0, None)
            w = lift / np.maximum(lift.sum(axis=1, keepdims=True), 1e-9)
            level = ((z[game] - self.game[g[game], b[game]]) * w).sum(axis=1)
            frame = pd.DataFrame({'opp': opp[game], 'lev': level}).groupby('opp')['lev'].agg(['sum', 'count'])
            self.opp = {k: float(r['sum'] / (r['count'] + p['shrink_opp'])) for k, r in frame.iterrows()}

        # Cancellation rate of scheduled games (rainy summer bucket separately) and unscheduled events.
        self.cancel = {}
        for name, mask in [('summer', b == SUMMER), ('other', b != SUMMER)]:
            sched = (state == HOME) & mask; c = ((state == HOME) & ~spike & mask).sum()
            self.cancel[name] = float((c + p['cancel_prior'] * p['cancel_strength']) / (sched.sum() + p['cancel_strength']))
        self.event_rate = np.zeros(3); self.event = np.zeros((3, 18))
        pooled_event = z[event].mean(axis=0) if event.any() else z[game].mean(axis=0) if game.any() else overall
        for k in range(3):
            idle = (state == NO_HOME) & (g == k); e = event & (g == k)
            self.event_rate[k] = (e.sum() + p['event_prior'] * p['event_strength']) / (idle.sum() + p['event_strength'])
            self.event[k] = (e.sum() * z[e].mean(axis=0) + 2 * pooled_event) / (e.sum() + 2) if e.any() else pooled_event

        # Concert lift over NORMAL from concert days already observed, aligned on the announced start hour.
        self._concert_lifts = []
        position = {d: i for i, d in enumerate(dates)}
        for c in self.concerts:
            if c['date'] <= origin and c['date'] in position:
                i = position[c['date']]
                self._concert_lifts.append((c['start_hour'], z[i] - self.normal[g[i]]))

        # Base rate of home games per day group inside announced coverage (prior for unannounced dates).
        self.base_rate = np.array([(((state == HOME) & (g == k)).sum() + 1.0) / ((covered & (g == k)).sum() + 3.0) for k in range(3)])
        self._hist = dict(dates=dates, spike=spike, group=g)
        self._fallback_lift = self.event - self.normal            # crowd days that were not on any schedule
        self.audit = dict(origin=str(origin.date()), history_days=int(len(dates)), scheduled=int((state == HOME).sum()), realised_games=int(game.sum()),
                          cancelled=int(((state == HOME) & ~spike).sum()), unscheduled_events=int(event.sum()), cancel=self.cancel,
                          event_rate=self.event_rate.round(4).tolist(), base_rate=self.base_rate.round(4).tolist(), opponent=self.opp,
                          game_cells=self.game_n.tolist(), observed_concert_days=len(self._concert_lifts))
        return self

    # -------------------------------------------------------------- predict
    def _unannounced_probability(self, date, group):
        p = self.p; hist = self._hist; prior = self.base_rate[group]
        if date.month >= 10: prior = prior * p['october_taper']      # regular season is over; only a few possible games
        centre = date - pd.Timedelta(days=364)
        take = (hist['dates'] >= centre - pd.Timedelta(days=21)) & (hist['dates'] <= centre + pd.Timedelta(days=21)) & (hist['group'] == group)
        n = take.sum()
        if n == 0: return float(prior)
        return float((hist['spike'][take].sum() + prior * p['unannounced_strength']) / (n + p['unannounced_strength']))

    def _concert_lift(self, start_hour, group):
        """Mean past concert lift shifted to the announced start hour; falls back to unscheduled crowd days."""
        if not self._concert_lifts: return np.clip(self._fallback_lift[group], 0, None)
        rows = [shift_profile(lift, 0 if (start_hour is None or past_start is None) else start_hour - past_start)
                for past_start, lift in self._concert_lifts]
        with np.errstate(all='ignore'):
            return np.clip(np.nan_to_num(np.nanmean(np.array(rows), axis=0)), 0, None)

    def predict(self, dates):
        """Return a DataFrame date x hour of expected log1p boardings; NaN where the specialist does not act
        (off-season dates without an announced concert)."""
        dates = pd.DatetimeIndex(dates); p = self.p
        holiday = calendar_features(pd.Series(dates), as_of=self.origin).is_holiday.to_numpy(bool)
        g = day_group(dates, holiday); b = bucket(dates)
        state, opp = self.schedule.state(dates, self.origin)
        out = np.full((len(dates), 18), np.nan); prob = np.full(len(dates), np.nan)
        for i, d in enumerate(dates):
            k, bb, s = g[i], b[i], state[i]
            normal = self.normal[k]; game = self.game[k, bb]
            if s == OFF: continue
            if s == HOME:
                scale = np.clip((game - normal) / p['opponent_scale'], 0, 1)
                game = game + self.opp.get(opp[i], 0.0) * scale
                q = self.cancel['summer' if bb == SUMMER else 'other']; prob[i] = 1 - q
                out[i] = (1 - q) * game + q * normal
            elif s == NO_HOME:
                r = self.event_rate[k]; prob[i] = r
                out[i] = (1 - r) * normal + r * self.event[k]
            else:
                pu = p['reserve_prob'] if s == RESERVE else self._unannounced_probability(d, k); prob[i] = pu
                out[i] = pu * game + (1 - pu) * normal
        concert = np.zeros(len(dates), bool)
        announced = {c['date']: c for c in self.concerts if c['known'] <= self.origin < c['date']}
        for i, d in enumerate(dates):
            if d not in announced: continue
            c = announced[d]; k = g[i]
            day = out[i] if state[i] != OFF else self.normal[k]      # off season: an ordinary day unless a concert is announced
            with_concert = self.normal[k] + p['concert_shrink'] * float(c.get('lift_scale', 1.0)) * self._concert_lift(c['start_hour'], k)
            out[i] = (1 - p['concert_cancel']) * np.maximum(day, with_concert) + p['concert_cancel'] * day
            concert[i] = True
        frame = pd.DataFrame(np.maximum(out, 0), index=dates, columns=HOURS)
        self.last = pd.DataFrame({'date': dates, 'state': state, 'opponent': opp, 'group': g, 'bucket': b, 'event_probability': prob, 'concert': concert})
        return frame


def predict_rows(train, query, origin, schedule=None, **overrides):
    """Convenience: specialist log predictions aligned to ``query`` rows of station 3127 (NaN = keep the base model)."""
    q = query.loc[query.station_id == SID, ['date', 'hour']]
    model = MunhakSpecialist(schedule, **overrides).fit(train, origin)
    table = model.predict(sorted(q.date.unique()))
    values = table.stack().rename('z').rename_axis(['date', 'hour']).reset_index()
    merged = q.merge(values, on=['date', 'hour'], how='left')
    merged.index = q.index
    return merged.z, model
