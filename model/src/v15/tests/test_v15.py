"""Compliance tests of ensemble_v15.   Run:  .venv/bin/python -m src.v15.tests.test_v15"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v14.panel import Panel14, EXTRA, EV_KINDS, N_FIRST
from src.v15.panel import Panel15
from src.v15.events import ANOMALIES, corrections

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v15'


def test_no_leaderboard_value_in_code_config_or_outputs():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751', '152513', '15251', '150812', '15081', '147741', '14774')]
    paths = list((ROOT / 'src/v15').rglob('*.py')) + [ROOT / 'configs/ensemble_v15.json'] + [OUT / n for n in ('gates_v15.json', 'gates_v15e.json', 'experiments_v15.json', 'base_keep/gates_v12.json', 'base_nokeep/gates_v12.json', 'RESULTS_V15.md')]
    for path in paths:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json',
                 'outputs/ensemble_v10/v9_preserved_before.json', 'outputs/ensemble_v11/v10_preserved_before.json', 'outputs/ensemble_v12/v11_preserved_before.json',
                 'outputs/ensemble_v13/v12_preserved_before.json', 'outputs/ensemble_v14/v13_preserved_before.json', 'outputs/ensemble_v15/v14_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_panel_differs_from_v14_only_by_the_listed_corrections():
    a, b = Panel14(), Panel15(); days = pd.DatetimeIndex(a.days); st = [int(s) for s in a.p.stations]; ev = [EXTRA.index(n) for n in list(EV_KINDS) + ['ev_start_rel']]
    other = [i for i in range(len(EXTRA)) if i not in ev]
    assert np.array_equal(np.nan_to_num(a.extra[..., other], nan=-9), np.nan_to_num(b.extra[..., other], nan=-9))
    changed = {(st[s], str(days[d])[:10]) for s, d in np.argwhere((np.nan_to_num(a.extra[..., ev], nan=-9) != np.nan_to_num(b.extra[..., ev], nan=-9)).any((2, 3)))}
    listed = {(int(r.station_id), r.date) for r in corrections().itertuples()}
    assert changed == listed and all(d <= '2026-06-30' for _, d in listed)      # corrections concern the training period only
    da, db = a.build('2026-06-30'), b.build('2026-06-30'); O = db['O']
    gone = np.isnan(db['r'][:, :O + 1]) & ~np.isnan(da['r'][:, :O + 1])
    assert int(gone.sum()) == sum(hi - lo + 1 for _, _, lo, hi in ANOMALIES) and np.array_equal(np.nan_to_num(da['X'][..., :50], nan=-9), np.nan_to_num(db['X'][..., :50], nan=-9))


def test_panel_never_reads_ridership_after_the_cutoff():
    a = Panel15(); b = Panel15(); origin = '2026-02-27'; O = a.p.day_index(origin)
    b.p.Z[:, O + 1:] = 6.0; b.A[:, O + 1:] = 6.0; b.p.T[:, O + 1:] = 9.0
    da, db = a.build(origin), b.build(origin)
    assert np.array_equal(np.nan_to_num(da['X'], nan=-9), np.nan_to_num(db['X'], nan=-9)) and np.isnan(da['r'][:, O + 1:]).all()


def test_external_tables_hold_no_evaluation_ridership():
    for path in (OUT / 'external').glob('*.csv'):
        cols = set(pd.read_csv(path, nrows=5, encoding='utf-8-sig').columns)
        assert not ({'boardings', 'alightings', 'ride_nope', 'goff_nope'} & cols), path
    g = pd.read_csv(OUT / 'external/ssg_home_games_2025_2026h1.csv', encoding='utf-8-sig'); assert g.date.max() <= '2026-06-30'     # the exit-split curve rests on games before the cutoff
    e = pd.read_csv(OUT / 'external/events_2026_jul_oct_venues.csv', encoding='utf-8-sig')
    assert e.reference_event_train.dropna().str.contains('2025|2026-0[3-6]').all()                                              # reference boardings in that table are from the training period


def test_validation_gates():
    g = json.loads((OUT / 'gates_v15.json').read_text()); a14, a15 = g['generations']['v14']['aligned'], g['generations']['v15']['aligned']
    assert 'without the days that fall inside fold k' in g['protocol'] and a15['all_oof'] < a14['all_oof'] and a15['weighted_4fold'] < a14['weighted_4fold']
    assert set(g['folds_worse_than_v14']) <= {'2', '3'} and all(v < 2e-4 for v in g['folds_worse_than_v14'].values())           # within seed noise; reported in RESULTS_V15.md
    assert all(a15[k] < a14[k] for k in ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4'))                                    # the four validator metrics
    for name, s in g['steps'].items():
        better = s['after']['all_oof'] < s['before']['all_oof'] and all(s['after'][f'fold_{k}'] <= s['before'][f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4))
        assert s['adopted'] == better, name
    assert g['retrained_models']['seeds']['lgb_h'] == dict(folds=3, final=3) and g['retrained_models']['seeds']['xgb_i'] == dict(folds=5, final=5) and not g['steps']['mix']['adopted']
    models = json.loads((OUT / 'models/f4__lgb_h__2026.json').read_text()); assert models['features'] == N_FIRST


def test_submissions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); g = json.loads((OUT / 'gates_v15.json').read_text()); ge = json.loads((OUT / 'gates_v15e.json').read_text())
    a = pd.read_csv(OUT / 'submission_ensemble_v15.csv'); b = pd.read_csv(OUT / 'submission_ensemble_v15e.csv')
    for sub, meta in ((a, g), (b, ge)):
        assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
        assert meta['submission']['sha256'] == hashlib.sha256((OUT / meta['submission']['file']).read_bytes()).hexdigest()
    cfg = json.loads((ROOT / 'configs/ensemble_v15.json').read_text()); L = cfg['summer_leave']
    leave = ((test.date >= L['leave_2026'][0]) & (test.date <= L['leave_2026'][1])).to_numpy(); allowed = leave.copy()
    for e in cfg['extras']:
        hours = {int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})}
        allowed |= ((test.station_id == e['station']) & (test.date == e['date']) & test.hour.isin(hours)).to_numpy()
        assert '2026-07-01' <= e['date'] <= '2026-10-31' and (('scenarios' not in e) or abs(sum(s['p'] for s in e['scenarios']) - 1) < 1e-9)
    d = np.log1p(b.boardings.to_numpy()) - np.log1p(a.boardings.to_numpy()); changed = np.abs(d) > 1e-9
    assert not (changed & ~allowed).any() and changed.sum() == ge['submission']['rows_changed_vs_v15']
    assert (d[leave] <= 1e-9).all() and d[leave].min() >= L['largest_shift'] - 1e-6            # the leave entry only lowers, and never beyond its cap


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok ', t.__name__)
    print(f'{len(tests)} passed')
