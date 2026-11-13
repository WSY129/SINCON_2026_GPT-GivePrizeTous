"""Compliance and leak tests of ensemble_v7.

Run:  .venv/bin/python -m src.v7.tests.test_v7      (plain functions; also collectable by pytest)
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from functools import lru_cache
from src.v6.schedule import Schedule
from src.v7 import core, growth, modules
from src.v7.panel import Panel, FI

ROOT = core.ROOT
OUT = core.OUT


@lru_cache(maxsize=1)
def _panel():
    return Panel(ROOT, Schedule(ROOT / 'configs/domain_schedule_v6.json'))


def test_no_leaderboard_value_in_code_or_config():
    """No public-leaderboard score may appear in v7 code or configuration (they are never inputs)."""
    scores = ['0.' + s for s in ('17274', '18265', '18678', '18849', '21751')]
    files = [p for p in (ROOT / 'src/v7').rglob('*.py')] + [ROOT / 'configs/ensemble_v7.json']
    for path in files:
        text = path.read_text()
        for token in scores:
            assert token not in text, f'{token} found in {path}'
        if path.suffix == '.json':
            assert 'public' not in text.lower()


def test_v6_files_unchanged():
    before = json.loads((OUT / 'v6_preserved_before.json').read_text())
    files = before.get('files', before)
    assert len(files) >= 20
    for rel, digest in files.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_panel_ignores_everything_after_the_origin():
    panel = _panel()
    origin = '2026-02-27'; O = panel.day_index(origin)
    X1, mu1, _, _ = panel.eval_set(origin)
    other = Panel(ROOT, panel.schedule)
    rng = np.random.default_rng(0)
    other.Z[:, O + 1:] = rng.normal(5, 2, other.Z[:, O + 1:].shape).astype(np.float32)
    other.T[:, O + 1:] = 9.0
    X2, mu2, _, _ = other.eval_set(origin)
    assert np.array_equal(np.nan_to_num(X1, nan=-9), np.nan_to_num(X2, nan=-9))
    assert np.array_equal(mu1, mu2)
    assert growth.slopes(panel, origin) == growth.slopes(other, origin)


def test_last_day_argument_equals_truncated_history():
    full = Panel(ROOT); cut = Panel(ROOT, last_day='2025-10-30')
    a = full.eval_set('2025-10-30')[0]; b = cut.eval_set('2025-10-30')[0]
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


def test_event_invisible_before_its_announcement():
    panel = _panel()
    cfg = core.config()
    schedule = core.schedule_v7(cfg)
    p = Panel(ROOT, schedule)
    s = int(np.searchsorted(p.stations, 3139)); day = p.day_index('2026-08-22')
    assert p.cal(p.day_index('2026-06-22'))['event'][s, day] == 0          # announced 2026-06-23
    assert p.cal(p.day_index('2026-06-30'))['event'][s, day] == 1
    for e in cfg['extra_events']:
        assert e['known_since'] <= cfg['final_cutoff']


def test_rain_features_ignore_future_weather():
    panel = _panel()
    ext = OUT / 'external'
    rain = modules.Rain(ext / 'weather_incheon_daily.csv', ext / 'climate_incheon_daily_2015_2024.csv')
    origin = '2025-12-30'
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-01-05', '2026-03-01']), 'station_id': [3110, 3226], 'hour': [8, 14]})
    a = rain.features(panel, frame, origin)[:2]
    rain.wet = rain.wet.copy(); rain.wet[rain.wet.index > pd.Timestamp(origin)] = True
    b = rain.features(panel, frame, origin)[:2]
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert pd.read_csv(ext / 'weather_incheon_daily.csv').date.max() <= '2026-06-30'


def test_growth_abstains_for_young_stations():
    panel = _panel()
    assert growth.slopes(panel, '2025-08-30') == {}                         # 63 days after opening
    late = growth.slopes(panel, '2026-06-30')
    assert set(late) == {3107, 3108, 3109} and len(set(late.values())) == 1  # common slope
    frame = pd.DataFrame({'station_id': [3107, 3110]})
    assert growth.slope_column(panel, frame, '2026-06-30')[1] == 0


def test_convex_weights_are_a_distribution():
    rng = np.random.default_rng(1)
    w = core.convex(rng.normal(size=(500, 4)) + np.array([0.0, 0.1, 0.3, 0.0]))
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_submission_integrity():
    path = OUT / 'submission_ensemble_v7.csv'
    if not path.exists():
        print('   (skipped: submission not built yet)'); return
    sub = pd.read_csv(path); test = pd.read_csv(ROOT / 'data/raw/test.csv')
    assert len(sub) == 157194 and list(sub.columns) == ['example_id', 'boardings']
    assert (sub.example_id == test.example_id).all() and sub.example_id.is_unique
    assert np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    meta = json.loads((OUT / 'submission_validation.json').read_text())
    assert meta['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    v6 = pd.read_csv(ROOT / 'outputs/ensemble_v6/submission_ensemble_v6.csv')
    if 'eq' not in meta['adopted_modules']:
        # rows of stations that no adopted module touches must equal v6
        touched = {3127, 3139, 3209, 3210, 3754, 3107, 3108, 3109}
        keep = ~test.station_id.isin(touched)
        assert np.allclose(sub.boardings[keep], v6.boardings[keep], rtol=1e-6, atol=1e-6)


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
