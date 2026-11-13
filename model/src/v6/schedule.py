"""Origin-filtered access to the pre-announced KBO home schedule at Munhak (station 3127).

A record is visible only when ``known_since <= origin``. The schedule is the grid as first
announced: cancelled games stay in it and later make-up games are never added.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'configs/domain_schedule_v6.json'
INFORMATION_CUTOFF = pd.Timestamp('2026-06-30')

OFF, NO_HOME, UNANNOUNCED, RESERVE, HOME = 0, 1, 2, 3, 4
STATE_NAMES = {OFF: 'OFF', NO_HOME: 'NO_HOME', UNANNOUNCED: 'UNANNOUNCED', RESERVE: 'RESERVE', HOME: 'HOME'}
# Months in which baseball can be played; outside them the specialist does not act.
SEASON_START_MD, SEASON_END_MD = 320, 1031


class Schedule:
    def __init__(self, path=DEFAULT):
        self.doc = json.loads(Path(path).read_text())
        self.records = []
        for r in self.doc['kbo_munhak']:
            self.records.append(dict(
                known=pd.Timestamp(r['known_since']), start=pd.Timestamp(r['coverage_start']), end=pd.Timestamp(r['coverage_end']),
                home={pd.Timestamp(g['date']): g['matchup'] for g in r['home']},
                reserve={pd.Timestamp(g['date']) for g in r['reserve']}, season=r['season'], stage=r['stage']))
        if any(r['known'] > INFORMATION_CUTOFF for r in self.records):
            raise ValueError('schedule record announced after the competition information cutoff')
        self.events = list(self.doc.get('events', []))
        for e in self.events:
            # An event may be used ahead of time only through a pre-cutoff announcement; an event observed
            # before the cutoff may always serve as history.
            if pd.Timestamp(e['known_since']) > INFORMATION_CUTOFF or (pd.Timestamp(e['date']) > INFORMATION_CUTOFF and pd.Timestamp(e['known_since']) > INFORMATION_CUTOFF):
                raise ValueError(f"event announced after the information cutoff: {e.get('name')}")

    def venue_events(self, venue):
        return [e for e in self.events if e['venue'] == venue]

    def overlay_events(self):
        """Events handled by the lift overlay (every venue except the Munhak main stadium) plus festival days."""
        out = [e for e in self.events if e['venue'] != 'munhak_main']
        for f in self.doc.get('festivals', []):
            for k, day in enumerate(pd.date_range(f['start'], f['end'])):
                out.append(dict(name=f"{f['kind']} day {k + 1}", venue='songdo_festival_park', stations=[3139], kind='festival',
                                family=f"{f['kind']}_d{k + 1}", date=str(day.date()), start=None, known_since=f['known_since']))
        return out

    def visible(self, origin):
        origin = pd.Timestamp(origin)
        if origin > INFORMATION_CUTOFF:
            raise ValueError('origin exceeds the competition information cutoff')
        return sorted((r for r in self.records if r['known'] <= origin), key=lambda r: r['known'])

    def state(self, dates, origin):
        """Return (state, opponent) per date as known at ``origin``."""
        dates = pd.DatetimeIndex(dates)
        state = np.full(len(dates), OFF, dtype=np.int8); opponent = np.array([''] * len(dates), dtype=object)
        md = dates.month * 100 + dates.day
        state[(md >= SEASON_START_MD) & (md <= SEASON_END_MD)] = UNANNOUNCED
        for rec in self.visible(origin):                      # before a season's announced opening day the specialist stays off
            if rec['stage'] == 'original':
                state[(dates.year == rec['start'].year) & (dates < rec['start'])] = OFF
        for rec in self.visible(origin):                      # later announcements override earlier ones
            inside = (dates >= rec['start']) & (dates <= rec['end'])
            for i in np.flatnonzero(inside):
                d = dates[i]
                if d in rec['home']:
                    state[i] = HOME; opponent[i] = rec['home'][d].split('-')[0]
                elif d in rec['reserve']:
                    state[i] = RESERVE; opponent[i] = ''
                else:
                    state[i] = NO_HOME; opponent[i] = ''
        return state, opponent
