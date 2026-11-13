"""Leak, schedule-visibility and rule-compliance tests for ensemble_v6."""
import json, re, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.v6.schedule import Schedule, HOME, NO_HOME, UNANNOUNCED, RESERVE, OFF, INFORMATION_CUTOFF
from src.v6.munhak import MunhakSpecialist
from src.v6.events import EventOverlay
from src.v6.pipeline import apply_specialists
from src.v6 import blend

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/ensemble_v6.json').read_text())


class V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(ROOT / 'data/raw/train.csv', parse_dates=['date'])
        cls.test = pd.read_csv(ROOT / 'data/raw/test.csv', parse_dates=['date'])
        cls.schedule = Schedule()

    # ---------------------------------------------------------- compliance
    def test_no_leaderboard_score_in_code_or_config(self):
        """Blend weights must come from local OOF only: no leaderboard value may appear in v6 code or config."""
        forbidden = re.compile(r'0\.18265|0\.18678|0\.18849|0\.21751|public_score|public_rmsle', re.IGNORECASE)
        files = list((ROOT / 'src/v6').glob('*.py')) + [ROOT / 'configs/ensemble_v6.json', ROOT / 'configs/domain_schedule_v6.json']
        for path in files:
            self.assertIsNone(forbidden.search(path.read_text()), msg=str(path))
        self.assertNotIn('base_blend_log_weights', CFG)           # weights are computed, never typed in

    def test_blend_weights_are_reproducible_from_oof(self):
        weights, report, _ = blend.fit(ROOT, CFG)
        for seg in ('ordinary', 'holiday'):
            self.assertAlmostEqual(sum(weights[seg].values()), 1.0, places=9)
            self.assertTrue(all(v >= 0 for v in weights[seg].values()))
        saved = ROOT / CFG['output'] / 'blend_weights.json'
        if saved.exists():
            self.assertEqual(json.loads(saved.read_text())['weights'], weights)

    # ------------------------------------------------------------ schedule
    def test_nothing_announced_after_information_cutoff(self):
        doc = self.schedule.doc
        for rec in doc['kbo_munhak'] + doc['festivals'] + doc.get('events', []):
            self.assertLessEqual(pd.Timestamp(rec['known_since']), INFORMATION_CUTOFF, msg=str(rec.get('name', rec.get('stage'))))
        self.assertEqual(max(pd.Timestamp(r['coverage_end']) for r in doc['kbo_munhak']), pd.Timestamp('2026-09-06'))
        for e in doc.get('events', []):
            if pd.Timestamp(e['date']) > INFORMATION_CUTOFF:       # a test-window event needs a real advance announcement and a source
                self.assertLess(pd.Timestamp(e['known_since']), pd.Timestamp(e['date']), msg=e['name'])
                self.assertTrue(e.get('source_url'), msg=e['name'])

    def test_kbo_known_since_filter(self):
        dates = pd.to_datetime(['2026-07-03', '2026-07-06', '2026-09-20', '2026-01-15', '2025-09-05', '2025-09-22'])
        before, _ = self.schedule.state(dates, '2025-12-18')
        after, opp = self.schedule.state(dates, '2025-12-19')
        self.assertEqual(before[:3].tolist(), [UNANNOUNCED] * 3)
        self.assertEqual(after[:4].tolist(), [HOME, NO_HOME, UNANNOUNCED, OFF]); self.assertEqual(opp[0], '삼성')
        early, _ = self.schedule.state(dates, '2025-08-18'); late, _ = self.schedule.state(dates, '2025-08-19')
        self.assertEqual(early[4:].tolist(), [UNANNOUNCED, UNANNOUNCED]); self.assertEqual(late[4:].tolist(), [HOME, RESERVE])

    def test_cancelled_games_stay_scheduled(self):
        state, _ = self.schedule.state(pd.to_datetime(['2025-07-17', '2025-08-13', '2026-04-09']), '2026-06-30')
        self.assertEqual(state.tolist(), [HOME, HOME, HOME])

    def test_event_invisible_before_its_announcement(self):
        for e in self.schedule.events:
            known = pd.Timestamp(e['known_since']); date = pd.Timestamp(e['date'])
            if known >= date or known <= pd.Timestamp('2025-01-10'): continue
            origin = known - pd.Timedelta(days=1)
            overlay = EventOverlay(self.schedule.overlay_events()).fit(self.train[self.train.date <= origin], origin)
            same = lambda x: x['date'] == date and x.get('name') == e['name']      # two venues can share a date
            self.assertFalse(any(same(x) and x['known'] <= origin for x in overlay.events), msg=e['name'])
            model = MunhakSpecialist(self.schedule).fit(self.train, origin)
            self.assertFalse(any(same(c) and c['known'] <= origin for c in model.concerts), msg=e['name'])
            # and the pipeline must not produce a lift or concert day for it at that origin
            q = self.test.loc[self.test.date == date, ['date', 'station_id', 'hour']] if date > pd.Timestamp('2026-06-30') else \
                self.train.loc[self.train.date == date, ['date', 'station_id', 'hour']]
            _, details = apply_specialists(self.train, q, np.full(len(q), 5.0), origin, self.schedule)
            self.assertFalse(any(r['action'] == 'lift' and r.get('event') == e['name'] for r in details.get('event_report', [])), msg=e['name'])

    # ---------------------------------------------------------------- leak
    def test_pipeline_ignores_future_observations(self):
        """Tampering with every observation after the origin must not change a single prediction."""
        origin = pd.Timestamp('2026-02-27')
        q = self.train.loc[self.train.date.between('2026-02-28', '2026-06-30'), ['date', 'station_id', 'hour']].reset_index(drop=True)
        base = np.full(len(q), 5.0)
        a, _ = apply_specialists(self.train, q, base, origin, self.schedule)
        tampered = self.train.copy(); future = tampered.date > origin
        tampered.loc[future, 'boardings'] = tampered.loc[future, 'boardings'] * 7 + 13; tampered.loc[future, 'alightings'] = 0
        b, _ = apply_specialists(tampered, q, base, origin, self.schedule)
        self.assertEqual(float(np.max(np.abs(a - b))), 0.0)

    def test_specialists_touch_only_their_stations(self):
        base = np.full(len(self.test), 5.0)
        out, details = apply_specialists(self.train, self.test, base, '2026-06-30', self.schedule)
        changed = set(self.test.loc[np.abs(out - base) > 1e-12, 'station_id'].unique())
        allowed = {3127} | {s for e in self.schedule.overlay_events() for s in e['stations']}
        self.assertTrue(changed <= allowed, msg=str(changed - allowed))
        self.assertTrue(np.isfinite(out).all() and (out >= 0).all())


if __name__ == '__main__':
    unittest.main()
