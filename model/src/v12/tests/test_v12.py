"""Compliance tests of ensemble_v12.   Run:  .venv/bin/python -m src.v12.tests.test_v12"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v12 import meta

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v12'


def test_no_leaderboard_value_in_code_or_config():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751', '152513', '15251')]
    for path in list((ROOT / 'src/v12').rglob('*.py')) + [ROOT / 'configs/ensemble_v12.json']:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'
    gates = (OUT / 'gates_v12.json').read_text()
    assert not any(token in gates for token in scores) and 'leaderboard value is used' in gates


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json',
                 'outputs/ensemble_v10/v9_preserved_before.json', 'outputs/ensemble_v11/v10_preserved_before.json', 'outputs/ensemble_v12/v11_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_weights_are_convex_and_shrinkage_is_bounded():
    rng = np.random.default_rng(0); E = rng.normal(size=(4000, 5)) * np.array([1.0, 1.2, 0.8, 2.0, 1.5]); seg = rng.integers(0, 5, 4000)
    w1, ws = meta.fit_two_stage(E, seg, 0.002)
    for w in [w1] + list(ws.values()):
        assert (w >= 0).all() and abs(w.sum() - 1) < 1e-9
    far = meta.simplex_ridge(E[seg == 3], 1e6, w1)
    assert np.abs(far - w1).max() < 1e-3                                   # a strong ridge returns the stage-1 weights
    t = rng.normal(size=(30, 6, 18)); b = t + rng.normal(size=t.shape)
    assert np.allclose(meta.eb_station_weights([t], [b], [t]), 1.0)        # perfect stack -> full weight
    a = meta.eb_station_weights([b], [t], [t])                             # perfect base -> shrunk towards the base, never below 0
    assert (a >= 0).all() and (a < 0.5).all()


def test_checkpoint_average_stays_inside_the_stored_range():
    pred = np.arange(8, dtype=float)[:, None] * np.ones((8, 3))
    assert np.allclose(meta.smooth(pred, 3), 3.0) and np.allclose(meta.smooth(pred, 0), 1 / 3) and np.allclose(meta.smooth(pred, 7), 7 - 1 / 3)
    assert abs(sum(meta.KERNEL) - 1) < 1e-12


def test_validation_gates():
    g = json.loads((OUT / 'gates_v12.json').read_text())
    a11, a12 = g['generations']['v11']['aligned'], g['generations']['v12']['aligned']
    assert all(a12[k] <= a11[k] + 1e-5 for k in ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_3', 'fold_4'))        # overall and both later folds
    worse = {k for k in ('1', '2', '3', '4') if a12[f'fold_{k}'] > a11[f'fold_{k}'] + 1e-5}
    assert worse == set(g['folds_worse_than_v11']) and g['no_fold_worse_than_v11'] == (not worse)                   # a worse fold is recorded, never hidden
    assert all(d < 0.0005 for d in g['folds_worse_than_v11'].values())
    own = g['two_stage']['segments_with_own_weights']                      # a segment gets its own weights only if it improves in every fold
    for name, change in g['two_stage']['sse_change_by_segment_and_fold'].items():
        assert (name in own) == all(c < 0 for c in change)
    for sid, v in g['new_stations']['per_station'].items():               # adopted only if better than v10 in both later folds
        assert v['adopted'] == all(v['blended'][k] < v['v10'][k] for k in ('3', '4'))
        assert v['adopted'] or v['weight_test'] == 0.0
    assert not ({'f1s', 'f1c', 'f2s', 'f2c', 'f3s', 'f3c', 'age', 'season', 'growth', 'level', 'yoy'} & set(g['new_stations']['features']))
    assert g['new_stations']['monthly_mean_removed']


def test_submission():
    path = OUT / 'submission_ensemble_v12.csv'
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); sub = pd.read_csv(path)
    assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    g = json.loads((OUT / 'gates_v12.json').read_text())
    assert g['submission']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    v11 = pd.read_csv(ROOT / 'outputs/ensemble_v11/submission_ensemble_v11.csv').boardings.to_numpy()
    cfg = json.loads((ROOT / 'configs/ensemble_v12.json').read_text())
    for e in cfg['one_off_events']:                                         # the one-off event only raises evening hours of its own station-day
        m = ((test.station_id == e['station']) & (test.date == e['date'])).to_numpy()
        d = sub.boardings.to_numpy()[m] - v11[m]; hours = test.hour.to_numpy()[m]
        assert (d[np.isin(hours, [20, 21, 22])] > 0).all() and d.sum() <= e['attendance'] * e['transit_share'] + 30
    b = ((test.station_id == 3754) & (test.date == '2026-07-19') & (test.hour == 21)).to_numpy()
    assert sub.boardings.to_numpy()[b][0] > 2 * v11[b][0]                 # the high-crowd exit hour is no longer truncated


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok ', t.__name__)
    print(f'{len(tests)} passed')
