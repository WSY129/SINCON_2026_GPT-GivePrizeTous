"""Audit of the Munhak (3127) unannounced-date probabilities: quota before/after, weekday table, validation.

    python -m src.v7.munhak_audit     # writes outputs/ensemble_v7/munhak_quota.json

Validation origins: Fold_Summer25 (2025-06-30, where September-October 2025 was not yet announced) and Fold 1
(2025-08-30, where October 2025 was not announced).  Everything is measured on local data only.
"""
import json
import numpy as np
import pandas as pd
from src.v6.munhak import MunhakSpecialist
from src.v6.schedule import Schedule, UNANNOUNCED, STATE_NAMES
from . import core
from .pipeline import MunhakV7

SPIKE = 1200
WEEKDAY = {0: 'Mon', 1: 'Tue-Fri', 2: 'Tue-Fri', 3: 'Tue-Fri', 4: 'Tue-Fri', 5: 'Sat', 6: 'Sun'}


def quota_class(scale):
    class Scaled(MunhakSpecialist):
        def _unannounced_probability(self, date, group):
            return float(np.clip(super()._unannounced_probability(date, group) * scale, 0, 0.95))
    return Scaled


def run(cls, train, schedule, origin, start, end):
    model = cls(schedule).fit(train[train.date <= origin], pd.Timestamp(origin))
    dates = pd.date_range(start, end)
    table = model.predict(dates)
    last = model.last.copy()
    return table, last, model


def evaluate(train, schedule, origin, start, end, classes):
    truth = train[(train.station_id == 3127) & (train.date >= start) & (train.date <= end)].copy()
    truth['z'] = np.log1p(truth.boardings)
    wide = truth.pivot(index='date', columns='hour', values='z')
    crowd = truth.groupby('date').boardings.max() >= SPIKE
    out = {}
    for name, cls in classes.items():
        table, last, model = run(cls, train, schedule, origin, start, end)
        last = last.set_index('date')
        un = last.index[(last.state == UNANNOUNCED) & ~last.concert.astype(bool)]
        un = un[un.isin(wide.index)]
        err = (table.loc[un].to_numpy() - wide.loc[un, table.columns].to_numpy())
        ok = ~np.isnan(err)
        by = {}
        for label in ('Mon', 'Tue-Fri', 'Sat', 'Sun'):
            d = un[[WEEKDAY[x] == label for x in un.dayofweek]]
            if len(d):
                e = table.loc[d].to_numpy() - wide.loc[d, table.columns].to_numpy()
                by[label] = dict(days=int(len(d)), expected_crowd_days=round(float(last.loc[d, 'event_probability'].sum()), 2),
                                 actual_crowd_days=int(crowd.reindex(d).fillna(False).sum()),
                                 mean_probability=round(float(last.loc[d, 'event_probability'].mean()), 3),
                                 rmsle=round(float(np.sqrt(np.nanmean(e ** 2))), 4))
        out[name] = dict(unannounced_days=int(len(un)), expected_crowd_days=round(float(last.loc[un, 'event_probability'].sum()), 2),
                         actual_crowd_days=int(crowd.reindex(un).fillna(False).sum()),
                         rmsle_unannounced_rows=round(float(np.sqrt(np.mean(err[ok] ** 2))), 4), by_weekday=by,
                         monday_split=model.audit.get('monday_split'))
    return out


def main():
    cfg = core.config(); train = core.load_train()
    schedule = Schedule(core.ROOT / cfg['schedule_v6'])
    out = dict(information='local data only; crowd day = daily maximum hourly boardings >= 1200 at station 3127')
    # quota that would give 10 expected crowd days on the test window (the directive's 9-11 target)
    _, last, _ = run(MunhakSpecialist, train, schedule, cfg['final_cutoff'], '2026-07-01', '2026-10-31')
    un = last[(last.state == UNANNOUNCED) & ~last.concert.astype(bool)]
    v6_quota = float(un.event_probability.sum()); scale10 = 10.0 / v6_quota
    classes = {'v6': MunhakSpecialist, 'v7_monday_split': MunhakV7, 'quota_scaled_to_10_games': quota_class(scale10)}
    out['validation_summer25'] = evaluate(train, schedule, '2025-06-30', '2025-07-01', '2025-10-31', classes)
    out['validation_fold1'] = evaluate(train, schedule, '2025-08-30', '2025-08-31', '2025-12-31', classes)
    test = {}
    for name, cls in classes.items():
        _, last, model = run(cls, train, schedule, cfg['final_cutoff'], '2026-07-01', '2026-10-31')
        u = last[(last.state == UNANNOUNCED) & ~last.concert.astype(bool)].copy()
        u['wd'] = [WEEKDAY[x] for x in pd.DatetimeIndex(u.date).dayofweek]
        u['month'] = pd.DatetimeIndex(u.date).month
        test[name] = dict(unannounced_days=int(len(u)), expected_crowd_days=round(float(u.event_probability.sum()), 2),
                          by_weekday={k: dict(days=int(len(g)), expected=round(float(g.event_probability.sum()), 2),
                                              mean_probability=round(float(g.event_probability.mean()), 3)) for k, g in u.groupby('wd')},
                          by_month={int(k): round(float(g.event_probability.sum()), 2) for k, g in u.groupby('month')},
                          monday_split=model.audit.get('monday_split'))
    out['test_window'] = test
    # what the league calendar implies, from facts known at the cutoff
    kbo = json.loads((core.ROOT / cfg['schedule_v6']).read_text())['kbo_munhak']
    out['league_quota_reasoning'] = dict(
        home_games_per_season=72,
        note='2025: the original grid listed 69 home dates and the remaining-games schedule announced 2025-08-19 listed 16 home dates '
             '(3 unscheduled + rained-out games). 2026: the original grid lists 66 home dates, so 6 home games are unscheduled before any '
             'rain-out is added. A target of 9-11 would assume at most 3-5 rained-out home games.')
    (core.OUT / 'munhak_quota.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('validation_summer25', 'validation_fold1', 'test_window')}, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
