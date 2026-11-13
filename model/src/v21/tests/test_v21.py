"""Checks of the v21 deliverables (python -m src.v21.tests.test_v21)."""
import json
import numpy as np, pandas as pd
from src.v8 import core as c8

F = c8.ROOT / 'outputs/ensemble_v21/final'


def test_submissions():
    te = pd.read_csv(c8.ROOT / 'data/raw/test.csv')
    for name in ('submission_ensemble_v21.csv', 'submission_ensemble_v21_bus.csv', 'submission_ensemble_v21e.csv'):
        s = pd.read_csv(F / name)
        assert len(s) == 157194 and (s.example_id.to_numpy() == te.example_id.to_numpy()).all(), name
        assert np.isfinite(s.boardings).all() and (s.boardings >= 0).all(), name


def test_oof_shapes_and_gate():
    g = json.loads((F / 'gates_v21.json').read_text())
    assert not g['folds_worse_than_v20e'], g['folds_worse_than_v20e']
    z = np.load(F / 'oof_v21.npz'); assert all(z[f'fold{k}'].shape == (123, 71, 18) for k in (1, 2, 3, 4))


def test_label_fix():
    from src.v21 import kinds
    before = c8.day_kinds(['2025-10-05', '2026-09-26', '2026-09-24'], '2026-06-30'); kinds.apply()
    after = c8.day_kinds(['2025-10-05', '2026-09-26', '2026-09-24'], '2026-06-30')
    assert list(after) == ['long_sun', 'long_sat', 'lunar_adj'], after
    c8.day_kinds = c8._day_kinds_orig


def test_no_test_period_subway():
    tr = pd.read_csv(c8.ROOT / 'data/raw/train.csv', usecols=['date'])
    assert tr.date.max() <= '2026-06-30'
    for p in (c8.ROOT / 'data/external_bus/stcis_20261113').glob('*long*.csv'):
        b = pd.read_csv(p, dtype={'ars': str}, usecols=['site_label'])
        assert not b.site_label.fillna('').str.contains('지하철').any(), p


if __name__ == '__main__':
    for f in (test_submissions, test_oof_shapes_and_gate, test_label_fix, test_no_test_period_subway):
        f(); print('ok', f.__name__)
