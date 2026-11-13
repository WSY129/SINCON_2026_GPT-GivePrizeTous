"""Compliance tests of ensemble_v10.   Run:  .venv/bin/python -m src.v10.tests.test_v10"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v10'


def test_no_leaderboard_value_in_code_or_config():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751')]
    for path in list((ROOT / 'src/v10').rglob('*.py')) + [ROOT / 'configs/ensemble_v10.json']:
        for token in scores:
            assert token not in path.read_text(), f'{token} in {path}'


def test_earlier_versions_and_insurance_file_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json',
                 'outputs/ensemble_v9/v8_preserved_before.json', 'outputs/ensemble_v10/v9_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    meta = json.loads((ROOT / 'outputs/ensemble_v8/gates_v8.json').read_text())['chains']['precutoff']['submission']
    assert hashlib.sha256((ROOT / 'outputs/ensemble_v8/submission_precutoff.csv').read_bytes()).hexdigest() == meta['sha256']


def test_no_evaluation_period_ridership():
    assert pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['date']).date.max() <= '2026-06-30'
    assert pd.read_csv(OUT / 'external/seoul_line7_monthly_hourly_2015_2021.csv', usecols=['month']).month.max() <= '2021-12'
    assert pd.read_csv(OUT / 'external/spike_days_labeled.csv', usecols=['date']).date.max() <= '2026-06-30'
    cfg = json.loads((ROOT / 'configs/ensemble_v10.json').read_text())
    assert 'boardings' not in json.dumps(cfg) and 'alightings' not in json.dumps(cfg)


def test_every_bucheon_match_is_applied():
    g = json.loads((OUT / 'gates_v10.json').read_text())
    step = next(s for s in g['steps'] if s['step'].startswith('A2'))
    assert step['details']['test']['skipped'] == [] and step['details']['test']['match_days_applied'] >= 10


def test_festival_month_gate_and_submission():
    test = pd.read_csv(ROOT / 'data/raw/test.csv', parse_dates=['date'])
    sub = pd.read_csv(OUT / 'submission_ensemble_v10.csv')
    assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    meta = json.loads((OUT / 'gates_v10.json').read_text())['submission']
    assert meta['sha256'] == hashlib.sha256((OUT / 'submission_ensemble_v10.csv').read_bytes()).hexdigest()
    v8 = pd.read_csv(ROOT / 'outputs/ensemble_v8/submission_ensemble_v8.csv')
    sel = ((test.station_id == 3139) & (test.date.dt.month == 8) & (test.date.dt.dayofweek < 5) & (test.date < '2026-08-21')).to_numpy()
    assert sub.boardings[sel].mean() < v8.boardings[sel].mean()            # the August index is no longer added at 3139


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print('PASS', name)
        except Exception as exc:                                    # noqa: BLE001
            failed += 1; print('FAIL', name, repr(exc)[:300])
    print(f'{len(tests) - failed} passed, {failed} failed')
    raise SystemExit(1 if failed else 0)
