"""Compliance tests of ensemble_v8.   Run:  .venv/bin/python -m src.v8.tests.test_v8"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core, pre, actual, run

ROOT, OUT = core.ROOT, core.OUT


def test_no_leaderboard_value_in_code():
    scores = ['0.' + s for s in ('17274', '17246', '18265', '18678', '18849', '21751')]
    for path in (ROOT / 'src/v8').rglob('*.py'):
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} in {path}'


def test_earlier_versions_unchanged():
    for name in ('outputs/ensemble_v7/v6_preserved_before.json', 'outputs/ensemble_v8/v7_preserved_before.json'):
        for rel, digest in json.loads((ROOT / name).read_text()).items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_no_evaluation_period_ridership_anywhere():
    """Subway ridership after 2026-06-30 is forbidden from every source."""
    assert pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['date']).date.max() <= '2026-06-30'
    for name in ('ictr_hourly_monthly_2015_2024.csv', 'ictr_hourly_monthly_2025_2026h1.csv', 'train_vs_ictr_monthly_by_station.csv'):
        assert pd.read_csv(core.SHARE / name, usecols=['data_month']).data_month.max() <= '2026-06', name
    weather = pd.read_csv(core.EXT / 'weather_incheon_daily_20250101_20261031.csv')
    assert set(weather.columns) == {'date', 'service_mm', 'tmax', 'tmean', 'wet'}
    games = pd.read_csv(core.SHARE / 'games_ssg_munhak.csv')
    assert 'boardings' not in games.columns and 'alightings' not in games.columns


def test_insurance_chain_never_reads_realised_facts(monkeypatch=None):
    """The pre-cutoff chain must run, and give the same file, with the realised-fact loaders disabled."""
    def boom(*a, **k):
        raise AssertionError('the insurance chain touched realised facts')
    keep = actual.load_games, actual.Weather
    actual.load_games, actual.Weather = boom, boom
    try:
        before = hashlib.sha256((OUT / 'submission_precutoff.csv').read_bytes()).hexdigest()
        gates = (OUT / 'gates_v8.json').read_bytes(); table = (OUT / 'VALIDATION_V8.csv').read_bytes(); oof = (OUT / 'oof_v8.npz').read_bytes()
        run.main('pre')
        assert hashlib.sha256((OUT / 'submission_precutoff.csv').read_bytes()).hexdigest() == before
    finally:
        actual.load_games, actual.Weather = keep
        (OUT / 'gates_v8.json').write_bytes(gates); (OUT / 'VALIDATION_V8.csv').write_bytes(table); (OUT / 'oof_v8.npz').write_bytes(oof)
    hist = pd.read_csv(core.SHARE / 'ictr_hourly_monthly_2015_2024.csv', usecols=['data_month'])
    assert hist.data_month.max() == '2024-12'


def test_modules_ignore_observations_after_the_origin():
    panel, folds, _ = core.load_blocks()
    b = folds[2]
    history = pre.History(panel)
    a1 = pre.seasonal_feature(history, b, b.pred); a2 = pre.holiday_features(b, b.pred)
    saved = panel.Z[:, b.O + 1:].copy()
    panel.Z[:, b.O + 1:] = 7.0
    try:
        b1 = pre.seasonal_feature(history, b, b.pred); b2 = pre.holiday_features(b, b.pred)
    finally:
        panel.Z[:, b.O + 1:] = saved
    assert np.array_equal(a1, b1) and all(np.array_equal(x, y) for x, y in zip(a2, b2))


def test_day_kinds_respect_announcement_dates():
    d = pd.to_datetime(['2026-07-17', '2026-05-01'])
    assert list(core.day_kinds(d, '2026-01-31')) == ['wd', 'labor']          # 제헌절 was designated on 2026-02-03
    assert list(core.day_kinds(d, '2026-06-30')) == ['hol_wd', 'hol_wd']


def test_submissions():
    test = pd.read_csv(ROOT / 'data/raw/test.csv')
    meta = json.loads((OUT / 'gates_v8.json').read_text())['chains']
    for chain, name in (('precutoff', 'submission_precutoff.csv'), ('main', 'submission_ensemble_v8.csv')):
        sub = pd.read_csv(OUT / name)
        assert len(sub) == 157194 and list(sub.columns) == ['example_id', 'boardings'] and (sub.example_id == test.example_id).all()
        assert np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
        assert meta[chain]['submission']['sha256'] == hashlib.sha256((OUT / name).read_bytes()).hexdigest()


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
