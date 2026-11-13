"""Compliance tests of ensemble_v11.   Run:  .venv/bin/python -m src.v11.tests.test_v11"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11.panel import Panel11, FEATURES

ROOT = c8.ROOT
OUT = ROOT / 'outputs/ensemble_v11'


def test_no_leaderboard_value_in_code_or_config():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751')]
    for path in list((ROOT / 'src/v11').rglob('*.py')) + [ROOT / 'configs/ensemble_v11.json']:
        for token in scores:
            assert token not in path.read_text(), f'{token} in {path}'


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json', 'outputs/ensemble_v9/v8_preserved_before.json',
                 'outputs/ensemble_v10/v9_preserved_before.json', 'outputs/ensemble_v11/v10_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_panel_never_reads_ridership_after_the_cutoff():
    a = Panel11(); b = Panel11()
    origin = '2026-02-27'; O = a.p.day_index(origin)
    b.p.Z[:, O + 1:] = 6.0; b.A[:, O + 1:] = 6.0; b.p.T[:, O + 1:] = 9.0
    da, db = a.build(origin), b.build(origin)
    assert np.array_equal(np.nan_to_num(da['X'], nan=-9), np.nan_to_num(db['X'], nan=-9))
    assert np.array_equal(np.nan_to_num(da['anchor'], nan=-9), np.nan_to_num(db['anchor'], nan=-9))
    assert np.isnan(da['r'][:, O + 1:]).all()                              # no target after the cutoff


def test_external_files_hold_no_evaluation_period_ridership():
    assert pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['date']).date.max() <= '2026-06-30'
    scores = pd.read_csv(OUT / 'external/games_ssg_scores.csv')
    assert not ({'boardings', 'alightings'} & set(scores.columns))
    weather = json.loads((OUT / 'external/open_meteo_incheon_hourly_ext_20250101_20261031.json').read_text())['hourly']
    assert set(weather) == {'time', 'precipitation', 'temperature_2m', 'apparent_temperature', 'shortwave_radiation', 'sunshine_duration', 'cloud_cover',
                            'wind_speed_10m', 'relative_humidity_2m'}
    assert 'boardings' not in ' '.join(FEATURES) and 'alight' not in ' '.join(f for f in FEATURES if f != 'ab_ratio')


def test_submission():
    path = OUT / 'submission_ensemble_v11.csv'
    if not path.exists():
        print('   (skipped: submission not built yet)'); return
    test = pd.read_csv(ROOT / 'data/raw/test.csv'); sub = pd.read_csv(path)
    assert len(sub) == 157194 and (sub.example_id == test.example_id).all() and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    meta = json.loads((OUT / 'gates_v11.json').read_text())
    assert meta['submission']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(v >= 2 for v in meta['test']['final_seeds'].values())


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
