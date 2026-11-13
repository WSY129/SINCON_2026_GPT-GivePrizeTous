"""Inning-by-inning scores of SSG home games at Munhak (2025-2026) from the KBO GameCenter scoreboard.

    python -m src.v11.fetch_kbo      # writes outputs/ensemble_v11/external/kbo_box/*.json and games_ssg_scores.csv

Only game facts are stored (scores, times, attendance); no ridership.  Requests carry a generic browser agent.
"""
import json
import subprocess
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'outputs/ensemble_v11/external'
RAW = EXT / 'kbo_box'
URL = 'https://www.koreabaseball.com/ws/Schedule.asmx/GetScoreBoardScroll'


def fetch(season, gid):
    path = RAW / f'{gid}.json'
    if path.exists() and path.stat().st_size > 200:
        return json.loads(path.read_text())
    for sr in (0, 1, 3, 4, 5, 7, 9, 6):
        out = subprocess.run(['curl', '-s', '-m', '30', '-X', 'POST', URL, '-H', 'User-Agent: Mozilla/5.0',
                              '-H', 'Referer: https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx', '-H', 'X-Requested-With: XMLHttpRequest',
                              '-H', 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8',
                              '--data', f'leId=1&srId={sr}&seasonId={season}&gameId={gid}'], capture_output=True, text=True).stdout
        time.sleep(0.25)
        try:
            d = json.loads(out)
        except Exception:                                   # noqa: BLE001
            continue
        if d.get('G_ID'):
            path.write_text(json.dumps(d, ensure_ascii=False)); return d
    return {}


def innings(d):
    """(away runs per inning, home runs per inning) as lists of ints; '-' (not played) is dropped."""
    rows = json.loads(d['table2'])['rows']
    out = []
    for r in rows[:2]:
        vals = [c['Text'] for c in r['row']]
        out.append([int(v) for v in vals if str(v).strip().isdigit()])
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(ROOT / 'outputs/ensemble_v8/external/team_share/games_ssg_munhak.csv', parse_dates=['date'])
    g = g[(g.date >= '2025-01-01') & (g.status == 'played') & g.game_id.notna()]
    recs = []
    for r in g.itertuples():
        d = fetch(r.date.year, r.game_id)
        if not d:
            continue
        try:
            away, home = innings(d)
        except Exception:                                   # noqa: BLE001
            away, home = [], []
        cum = lambda x, n: sum(x[:n])
        recs.append(dict(date=r.date.date(), game_id=r.game_id, away_runs=int(d.get('T_SCORE_CN') or 0), home_runs=int(d.get('B_SCORE_CN') or 0),
                         innings_played=max(len(away), len(home)),
                         margin_after_6=cum(home, 6) - cum(away, 6), margin_after_7=cum(home, 7) - cum(away, 7), margin_after_8=cum(home, 8) - cum(away, 8)))
    out = pd.DataFrame(recs)
    out['margin_final'] = out.home_runs - out.away_runs
    out.to_csv(EXT / 'games_ssg_scores.csv', index=False)
    print(len(out), 'games;', 'test-period games', int((pd.to_datetime(out.date) > '2026-06-30').sum()))
    print(out.describe().T[['mean', 'min', 'max']].round(2).to_string())


if __name__ == '__main__':
    main()
