"""Compliance tests of ensemble_v13.   Run:  .venv/bin/python -m src.v13.tests.test_v13"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v13 import run as R
from src.v13 import extras

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v13'


def test_no_leaderboard_value_in_code_config_or_outputs():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751', '152513', '15251')]
    for path in list((ROOT / 'src/v13').rglob('*.py')) + [ROOT / 'configs/ensemble_v13.json', OUT / 'gates_v13.json', OUT / 'gates_v13e.json']:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json',
                 'outputs/ensemble_v10/v9_preserved_before.json', 'outputs/ensemble_v11/v10_preserved_before.json', 'outputs/ensemble_v12/v11_preserved_before.json',
                 'outputs/ensemble_v13/v12_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_external_tables_hold_no_ridership():
    for path in (OUT / 'external').glob('*.csv'):
        cols = set(pd.read_csv(path, nrows=5, encoding='utf-8-sig').columns)
        assert not ({'boardings', 'alightings', 'ride_nope', 'goff_nope'} & cols), path
    assert pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['date']).date.max() <= '2026-06-30'


def test_school_slope_ignores_a_level_shift_and_finds_a_school_hour_effect():
    rng = np.random.default_rng(0); ns = 6; delta = np.where(np.arange(80) < 40, 0.9, 0.0); res = rng.normal(0, 0.05, (80, ns, 18))
    res += 0.3 * delta[:, None, None] * (np.arange(ns) == 2)[None, :, None]                   # a station whose whole level drifts with delta
    res[:, :, 0:3] += -0.06 * delta[:, None, None]; res[:, :, 9:12] += -0.04 * delta[:, None, None]   # school hours
    slopes, glob = R.school_fit([(delta, res)], ns, np.zeros(ns, bool))
    assert abs(glob[0] + 0.06) < 0.01 and abs(glob[1] + 0.04) < 0.01
    assert abs(slopes[0, 2] - glob[0]) < 0.02                                                 # the drifting station gets no extra school slope
    only_global, _ = R.school_fit([(delta, res)], ns, np.ones(ns, bool)); assert np.allclose(only_global[0], only_global[0, 0])
    c = R.school_apply(delta, slopes, res.shape); assert (c[:, :, 3:9] == 0).all() and (c[40:] == 0).all()


def test_low_calibration_only_touches_small_forecasts():
    f = np.log1p(np.array([0.2, 1.2, 4.9, 5.0, 40.0])); add = np.full(len(R.LOW_EDGES) - 1, 0.1)
    assert np.allclose(R.low_apply(f, add), [0.1, 0.1, 0.1, 0.0, 0.0])
    assert np.allclose(R.low_fit([(np.log1p(np.full(50, 1.2)), np.full(50, 0.4))])[2], 0.4 * 50 / (50 + R.LOW_SHRINK))


def test_validation_gates():
    g = json.loads((OUT / 'gates_v13.json').read_text()); a12, a13 = g['generations']['v12']['aligned'], g['generations']['v13']['aligned']
    assert 'without the days that fall inside fold k' in g['protocol'] and not g['folds_worse_than_v12']
    assert all(a13[k] <= a12[k] + 1e-5 for k in a12)
    for name, s in g['steps'].items():                                                        # a step is kept only if no fold gets worse
        better = s['after']['all_oof'] < s['before']['all_oof'] and all(s['after'][f'fold_{k}'] <= s['before'][f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4))
        assert s['adopted'] == better, name
    assert g['steps']['events']['test_station_days_changed'] == 0


def test_submissions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); g = json.loads((OUT / 'gates_v13.json').read_text()); ge = json.loads((OUT / 'gates_v13e.json').read_text())
    a = pd.read_csv(OUT / 'submission_ensemble_v13.csv'); b = pd.read_csv(OUT / 'submission_ensemble_v13e.csv')
    for sub, meta in ((a, g), (b, ge)):
        assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
        assert meta['submission']['sha256'] == hashlib.sha256((OUT / meta['submission']['file']).read_bytes()).hexdigest()
    cfg = json.loads((ROOT / 'configs/ensemble_v13.json').read_text()); allowed = np.zeros(len(test), bool)
    for e in cfg['extras']:
        hours = {int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})}
        allowed |= ((test.station_id == e['station']) & (test.date == e['date']) & test.hour.isin(hours)).to_numpy()
    changed = np.abs(a.boardings.to_numpy() - b.boardings.to_numpy()) > 1e-6
    assert changed.sum() == ge['submission']['rows_changed_vs_v13'] and not (changed & ~allowed).any()      # v13e differs from v13 only in the listed cells
    assert abs(extras.scenario_log([dict(p=0.5, hours={'20': 99}), dict(p=0.5, hours={})], 20, np.log1p(9)) - 0.5 * (np.log(100) + np.log(10))) < 1e-9


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok ', t.__name__)
    print(f'{len(tests)} passed')
