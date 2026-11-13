"""Compliance tests of ensemble_v9.   Run:  .venv/bin/python -m src.v9.tests.test_v9"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v6.schedule import Schedule
from src.v8 import core as c8
from src.v9.panel import Panel9, FEATURES, FI

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v9'


def test_no_leaderboard_value_in_code():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751')]
    for path in (ROOT / 'src/v9').rglob('*.py'):
        for token in scores:
            assert token not in path.read_text(), f'{token} in {path}'


def test_panel_ignores_ridership_after_the_origin():
    a = Panel9(ROOT, Schedule()); b = Panel9(ROOT, Schedule())
    origin = '2026-02-27'; O = a.day_index(origin)
    b.Z[:, O + 1:] = 6.5; b.T[:, O + 1:] = 9.0
    Xa, anchor_a, _, _ = a.eval_set(origin); Xb, anchor_b, _, _ = b.eval_set(origin)
    assert np.array_equal(np.nan_to_num(Xa, nan=-9), np.nan_to_num(Xb, nan=-9)) and np.array_equal(anchor_a, anchor_b)


def test_precutoff_panel_holds_no_realised_fact():
    p = Panel9(ROOT, Schedule(), actual=False)
    assert p.games.day.max() < p.D and float(p.rain_h[p.D:].sum()) == 0.0 and np.isnan(p.temp_h[p.D:]).all()
    X, _, _, _ = p.eval_set('2026-06-30')
    for name in ('wet', 'rain_mm', 'rain_h', 'rain_prev2', 'game_att'):
        assert float(np.nan_to_num(X[..., FI[name]]).sum()) == 0.0, name


def test_submission_and_protected_rows():
    meta = json.loads((OUT / 'blend_v9.json').read_text())['chains']['main']
    sub = pd.read_csv(OUT / 'submission_ensemble_v9.csv'); test = pd.read_csv(ROOT / 'data/raw/test.csv')
    assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    assert meta['submission']['sha256'] == hashlib.sha256((OUT / 'submission_ensemble_v9.csv').read_bytes()).hexdigest()
    v8 = pd.read_csv(ROOT / 'outputs/ensemble_v8/submission_ensemble_v8.csv')
    munhak = (test.station_id == 3127).to_numpy()
    assert np.allclose(sub.boardings[munhak], v8.boardings[munhak], rtol=1e-6, atol=1e-6)      # specialist rows are kept
    for chain in ('precutoff',):
        assert not (OUT / 'submission_precutoff_v9.csv').exists() or json.loads((OUT / 'blend_v9.json').read_text())['chains'][chain]['adopted']


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


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
