"""Compliance tests of ensemble_v14.   Run:  .venv/bin/python -m src.v14.tests.test_v14"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v14.panel import Panel14, FEATURES14, EXTRA, N_FIRST
from src.v14.priors import CalendarPrior

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v14'


def test_no_leaderboard_value_in_code_config_or_outputs():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751', '152513', '15251', '150812', '15081')]
    paths = list((ROOT / 'src/v14').rglob('*.py')) + [ROOT / 'configs/ensemble_v14.json', OUT / 'gates_v14.json', OUT / 'gates_v14e.json', OUT / 'base/gates_v12.json']
    for path in paths:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json',
                 'outputs/ensemble_v10/v9_preserved_before.json', 'outputs/ensemble_v11/v10_preserved_before.json', 'outputs/ensemble_v12/v11_preserved_before.json',
                 'outputs/ensemble_v13/v12_preserved_before.json', 'outputs/ensemble_v14/v13_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_panel_never_reads_ridership_after_the_cutoff():
    a = Panel14(); b = Panel14(); origin = '2026-02-27'; O = a.p.day_index(origin)
    b.p.Z[:, O + 1:] = 6.0; b.A[:, O + 1:] = 6.0; b.p.T[:, O + 1:] = 9.0
    da, db = a.build(origin), b.build(origin)
    assert da['X'].shape[-1] == len(FEATURES14) == N_FIRST + 2 and len(EXTRA) == 26
    assert np.array_equal(np.nan_to_num(da['X'], nan=-9), np.nan_to_num(db['X'], nan=-9)) and np.isnan(da['r'][:, O + 1:]).all()
    assert np.array_equal(np.nan_to_num(a.extra, nan=-9), np.nan_to_num(b.extra, nan=-9))          # the added inputs do not depend on ridership at all


def test_external_tables_and_priors():
    for path in (OUT / 'external').glob('*.csv'):
        cols = set(pd.read_csv(path, nrows=5, encoding='utf-8-sig').columns)
        if path.name != 'seoul_daily_2015_2026h1.csv':
            assert not ({'boardings', 'alightings', 'ride_nope', 'goff_nope'} & cols), path
    d = pd.read_csv(OUT / 'external/seoul_daily_2015_2026h1.csv', usecols=['date'])
    assert d.date.max() <= '2026-06-30'                                         # card data of other lines stops at the cutoff
    cp = CalendarPrior(); assert cp.P.index.max() <= pd.Timestamp('2026-06-30')
    w = cp.weekly_index(2025); assert w.shape == (3, 53) and np.isfinite(w[:, 1:]).all()


def test_validation_gates():
    g = json.loads((OUT / 'gates_v14.json').read_text()); a13, a14 = g['generations']['v13']['aligned'], g['generations']['v14']['aligned']
    assert 'without the days that fall inside fold k' in g['protocol'] and not g['folds_worse_than_v13'] and all(a14[k] <= a13[k] + 1e-5 for k in a13)
    for name, s in g['steps'].items():
        better = s['after']['all_oof'] < s['before']['all_oof'] and all(s['after'][f'fold_{k}'] <= s['before'][f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4))
        assert s['adopted'] == better, name
    assert set(g['retrained_models']['checkpoint']) == {'lgb_c', 'lgb_d'}


def test_submissions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); g = json.loads((OUT / 'gates_v14.json').read_text()); ge = json.loads((OUT / 'gates_v14e.json').read_text())
    a = pd.read_csv(OUT / 'submission_ensemble_v14.csv'); b = pd.read_csv(OUT / 'submission_ensemble_v14e.csv')
    for sub, meta in ((a, g), (b, ge)):
        assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
        assert meta['submission']['sha256'] == hashlib.sha256((OUT / meta['submission']['file']).read_bytes()).hexdigest()
    cfg = json.loads((ROOT / 'configs/ensemble_v14.json').read_text()); L = cfg['summer_leave']
    allowed = ((test.date >= L['leave_2026'][0]) & (test.date <= L['leave_2026'][1])).to_numpy()
    for e in cfg['extras']:
        hours = {int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})}
        allowed |= ((test.station_id == e['station']) & (test.date == e['date']) & test.hour.isin(hours)).to_numpy()
    d = np.log1p(b.boardings.to_numpy()) - np.log1p(a.boardings.to_numpy()); changed = np.abs(d) > 1e-9
    assert not (changed & ~allowed).any() and changed.sum() == ge['submission']['rows_changed_vs_v14']
    leave = ((test.date >= L['leave_2026'][0]) & (test.date <= L['leave_2026'][1])).to_numpy()
    assert (d[leave] <= 1e-9).all() and d[leave].min() >= L['largest_shift'] - 1e-6            # the leave entry only lowers, and never beyond its cap


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok ', t.__name__)
    print(f'{len(tests)} passed')
