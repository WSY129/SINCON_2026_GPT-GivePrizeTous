"""Compliance tests of ensemble_v17.   Run:  .venv/bin/python -m src.v17.tests.test_v17"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v17'


def test_no_leaderboard_value_in_code_config_or_outputs():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751', '152513', '15251', '150812', '15081', '147741', '14774', '148621', '14862')]
    paths = list((ROOT / 'src/v17').rglob('*.py')) + [ROOT / 'configs/ensemble_v17.json'] + [OUT / n for n in ('gates_v17.json', 'gates_v17e.json', 'experiments_v17.json', 'RESULTS_V17.md')]
    for path in paths:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'


def test_earlier_versions_unchanged():
    for v in range(7, 18):
        for rel, digest in json.loads((ROOT / f'outputs/ensemble_v{v}/v{v - 1}_preserved_before.json').read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_validation_gates():
    g = json.loads((OUT / 'gates_v17.json').read_text()); a15, a17 = g['generations']['v15']['aligned'], g['generations']['v17']['aligned']
    assert 'without the days that fall inside fold k' in g['protocol'] and not g['folds_worse_than_v15'] and all(a17[k] <= a15[k] + 1e-5 for k in a15) and a17['all_oof'] < a15['all_oof']
    for name, s in g['steps'].items():
        better = s['after']['all_oof'] < s['before']['all_oof'] and all(s['after'][f'fold_{k}'] <= s['before'][f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4))
        assert s['adopted'] == better, name


def test_grave_visit_step_stays_inside_the_observed_range():
    """The holiday itself at station 3122: v16 reached 12,619 boardings at 09-16h; the two observed holidays had 8,285 and 9,968."""
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); a = pd.read_csv(OUT / 'submission_ensemble_v17.csv')
    m = ((test.station_id == 3122) & (test.date == '2026-09-25') & test.hour.between(9, 16)).to_numpy()
    assert 6000 < a.boardings[m].sum() < 10500
    log = json.loads((OUT / 'gates_v17.json').read_text())['steps']['lunar']['test']
    assert all(min(sum(v['before']), sum(v['target'])) - 5 <= sum(v['after']) <= max(sum(v['before']), sum(v['target'])) + 5 for v in log.values())      # a blend never leaves the span of its two ends


def test_submissions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); g = json.loads((OUT / 'gates_v17.json').read_text()); ge = json.loads((OUT / 'gates_v17e.json').read_text())
    a = pd.read_csv(OUT / 'submission_ensemble_v17.csv'); b = pd.read_csv(OUT / 'submission_ensemble_v17e.csv'); v15 = pd.read_csv(ROOT / 'outputs/ensemble_v15/submission_ensemble_v15.csv')
    for sub, meta in ((a, g), (b, ge)):
        assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
        assert meta['submission']['sha256'] == hashlib.sha256((OUT / meta['submission']['file']).read_bytes()).hexdigest()
    d = np.abs(np.log1p(a.boardings.to_numpy()) - np.log1p(v15.boardings.to_numpy())) > 1e-9; dow = pd.to_datetime(test.date).dt.dayofweek
    ok = ((test.station_id == 3127) & test.date.isin(['2026-09-12', '2026-09-13']) & test.hour.between(20, 23)) | ((test.station_id == 3122) & test.date.between('2026-09-22', '2026-09-28') & test.hour.between(9, 18)) \
        | (test.station_id.isin([3131, 3206]) & (dow == 6) & test.hour.between(11, 18)) | ((test.station_id == 3139) & (test.hour == 22))
    assert not (d & ~ok.to_numpy()).any() and d.sum() == g['submission']['rows_changed_vs_v15']
    cfg = json.loads((ROOT / 'configs/ensemble_v17.json').read_text()); L = cfg['summer_leave']
    leave = ((test.date >= L['leave_2026'][0]) & (test.date <= L['leave_2026'][1])).to_numpy(); allowed = leave.copy()
    for e in cfg['extras']:
        hours = {int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})}
        allowed |= ((test.station_id == e['station']) & (test.date == e['date']) & test.hour.isin(hours)).to_numpy()
        assert ('scenarios' not in e) or abs(sum(s['p'] for s in e['scenarios']) - 1) < 1e-9
    x = np.log1p(b.boardings.to_numpy()) - np.log1p(a.boardings.to_numpy()); changed = np.abs(x) > 1e-9
    assert not (changed & ~allowed).any() and changed.sum() == ge['submission']['rows_changed_vs_v17'] and (x[leave] <= 1e-9).all()


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok ', t.__name__)
    print(f'{len(tests)} passed')
