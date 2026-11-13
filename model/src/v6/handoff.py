"""Write OFFICIAL_SOURCES_V6.md and the single hand-off document HANDOFF_V6.md from the saved artefacts.

Usage (from the project root):  python -m src.v6.handoff
Run after validate, build, report and pipeline_doc. Tables are generated from the schedule file and
the JSON/CSV artefacts so the hand-off cannot drift from what was actually run.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOW = '월화수목금토일'
VENUE = {'munhak_main': '인천문학경기장 주경기장', 'asiad_main': '인천아시아드주경기장', 'bucheon': '부천종합운동장'}
STATION = {3127: '문학경기장', 3754: '부천종합운동장', 3209: '아시아드경기장', 3210: '서구청', 3139: '송도달빛축제공원'}


def day(d):
    t = pd.Timestamp(d); return f"{t.date()} ({DOW[t.dayofweek]})"


def sources(cfg, doc, out):
    L = ['# ensemble_v6 외부 일정 출처', '',
         '조회일 2026-11-12. 정보 마감 2026-06-30. 모든 일정은 **공개일(known_since) ≤ 기준일**일 때만 그 기준일의 예측에 보인다. 기계 판독본은 `configs/domain_schedule_v6.json`이다.', '',
         '확인 수준은 비고에 적었다. "직접 확인"은 기사·공지 본문을 열어 날짜·장소·게시일을 읽었다는 뜻이고, 그 밖은 조사 보고나 검색 요약에 근거한다. 과거 행사 중 사전 발표일을 확인하지 못한 것은 공개일을 행사일로 두어, 이력(과거 표본)으로만 쓰이게 했다.', '',
         '## 1. KBO 원편성 (문학 홈경기, 역 3127)', '',
         '| 시즌·단계 | 공개일 | 범위 | 홈 편성일 | 예비일 | 출처 | 별첨 SHA-256 |', '|---|---|---|---:|---:|---|---|']
    for r in doc['kbo_munhak']:
        L.append(f"| {r['season']} {r['stage']} | {r['known_since']} | {r['coverage_start']} ~ {r['coverage_end']} | {len(r['home'])} | {len(r['reserve'])} | [{r['attachment']}]({r['source_url']}) | `{r['attachment_sha256']}` |")
    k26 = next(r for r in doc['kbo_munhak'] if r['season'] == 2026 and r['stage'] == 'original')
    home = [g for g in k26['home'] if g['date'] > cfg['final_cutoff']]
    L += ['', '- 별첨 표의 `문학` 열에 적힌 경기만 날짜와 대진으로 옮겼다. 표에 개시 시각이 없어 시각은 넣지 않았고, 시간대 형상은 과거 관측에서만 추정한다.',
          '- 우천 등으로 취소된 경기를 지우지 않았다(예: 2025-07-17, 2025-08-13, 2026-04-09는 홈 편성 그대로).',
          '- 2026-08-25에 발표된 2026 잔여 경기 일정과 포스트시즌은 넣지 않았다. 2026-09-07 이후는 미발표로 처리한다.',
          f"- 테스트 구간의 문학 홈 편성 {len(home)}일: " + ', '.join(f"{g['date'][5:]}({g['matchup'].split('-')[0]})" for g in home), '']
    n = 2
    for venue, title in (('munhak_main', '인천문학경기장 주경기장 콘서트 (역 3127)'), ('asiad_main', '인천아시아드주경기장 콘서트 (역 3209·3210)'), ('bucheon', '부천FC1995 홈경기 (역 3754)')):
        ev = sorted((e for e in doc.get('events', []) if e['venue'] == venue), key=lambda e: e['date'])
        if not ev: continue
        L += [f'## {n}. {title}', '', '| 날짜 | 시작 | 행사 | 공개일 | 테스트 구간 | 출처 | 비고 |', '|---|---|---|---|:-:|---|---|']; n += 1
        for e in ev:
            mark = '예' if e['date'] > cfg['final_cutoff'] else ''
            L.append(f"| {day(e['date'])} | {e.get('start') or ''} | {e['name']} | {e['known_since']} | {mark} | [링크]({e['source_url']}) | {e.get('note', '')} |")
        L.append('')
    L += [f'## {n}. 펜타포트 락 페스티벌 (역 3139)', '', '| 기간 | 공개일 | 출처 | 이용 조건 |', '|---|---|---|---|']
    for f in doc['festivals']:
        L.append(f"| {f['start']} ~ {f['end']} | {f['known_since']} | [{f.get('attribution', '출처')}]({f['source_url']}) | {f.get('license', '')} |")
    L += ['', f'## {n + 1}. 넣지 않은 것', ''] + [f'- {x}' for x in doc.get('exclusions_ko', doc.get('exclusions', []))]
    L += ['', f'## {n + 2}. 이용 조건 메모', '',
          'v3 기록에 남아 있듯 KBO 이용약관의 가공·재사용 조항에 대한 별도 허가는 확인되지 않았다. 일정 반영은 사용자가 대회 규정(6월 30일 이전 발표된 경기·행사 일정 사용 가능)에 근거해 승인한 결정에 따른 것이며, 날짜·대진·장소·시각이라는 사실 정보만 사용하고 원문과 표는 재배포하지 않는다. 콘서트와 축구 일정도 같은 원칙이다.', '']
    (out / 'OFFICIAL_SOURCES_V6.md').write_text('\n'.join(L))


def hour_table(rows, station, date):
    g = rows[(rows.station_id == station) & (rows.date == pd.Timestamp(date))].sort_values('hour')
    if g.empty: return None
    return ' | '.join(f"{int(r.hour)}시 {r.base_boardings:.0f}→{r.final_boardings:.0f}" for r in g.itertuples() if abs(r.final_boardings - r.base_boardings) >= 0.5)


def handoff(cfg, doc, out):
    val = json.loads((out / 'validation_summary.json').read_text()); audit = json.loads((out / 'schedule_audit.json').read_text())
    sub = json.loads((out / 'submission_validation.json').read_text()); bw = json.loads((out / 'blend_weights.json').read_text())
    rows = pd.read_csv(out / 'specialist_rows.csv', parse_dates=['date'])
    states = pd.read_csv(out / 'munhak_test_states.csv', parse_dates=['date'])
    lv = val['local_validator']['saved_ensemble_columns']; lo = val['local_validator']['leave_one_fold_out_columns']
    L = ['# SINCON ensemble_v6 전달 문서', '',
         '작성 2026-11-12. 이 문서 하나에 결과, 파이프라인, 일정 출처, 근거 자료, 부록을 모았다. **점수는 모두 로컬 검증(OOF) RMSLE**이고 리더보드 점수는 계산에도 문서에도 쓰지 않았다.', '',
         '## A. 요약', '',
         f"- 제출 파일: `outputs/ensemble_v6/submission_ensemble_v6.csv` 1개 (157,194행, SHA-256 `{sub['sha256']}`).",
         '- 구성: v2·v4·v5 예측의 로컬 OOF 블렌드 → 문학경기장 전용 모델(야구 원편성 + 콘서트) → 행사 보정(전년 급등 제거 → 발표된 행사 상승 반영).',
         f"- 로컬 검증(기존 보고서와 같은 방식) 전체 OOF: v5 {lv['v5']['all_oof']:.4f} → v6 블렌드만 {lv['v6_base_blend']['all_oof']:.4f} → **v6 최종 {lv['v6_final']['all_oof']:.4f}**. 가중 4-Fold {lv['v5']['weighted_4fold']:.4f} → {lv['v6_final']['weighted_4fold']:.4f}, Fold 3+4 {lv['v5']['fold_3_4']:.4f} → {lv['v6_final']['fold_3_4']:.4f}, Fold 4 {lv['v5']['fold_4']:.4f} → {lv['v6_final']['fold_4']:.4f}.",
         f"- 로컬 검증(엄격한 방식) 전체 OOF: v5 {lo['v5']['all_oof']:.4f} → v6 최종 {lo['v6_final']['all_oof']:.4f}.",
         f"- 개선의 출처(엄격한 방식, 전체 OOF): 블렌드 {lo['v5']['all_oof']:.4f} → {lo['v6_base_blend']['all_oof']:.4f}, 문학 전용 모델 → {lo['v6_blend_plus_munhak']['all_oof']:.4f}, 행사 보정 → {lo['v6_final']['all_oof']:.4f}. Fold 4만 보면 {lo['v6_base_blend']['fold_4']:.4f} → {lo['v6_blend_plus_munhak']['fold_4']:.4f} → {lo['v6_final']['fold_4']:.4f}이다.",
         f"- 테스트에서 전용 모델이 바꾼 행은 {audit['rows_changed_total']:,}행(전체의 {100 * audit['rows_changed_total'] / 157194:.1f}%)이고 나머지는 블렌드 값이다.",
         '- 테스트 구간에 반영된 일정: 문학 야구 홈 28일·홈 아님 40일·미발표 55일, 싸이 흠뻑쇼(7/11, 아시아드), 펜타포트(7/31~8/2, 송도), K리그1 부천 홈 9경기, Sustainable Wave Festival(8/29~30, 문학), PLAVE(9/12~13, 문학).', '']
    notes = out / 'HANDOFF_NOTES.md'
    if notes.exists(): L += [notes.read_text().rstrip(), '']
    for title, name in (('B. 결과 (RESULTS.md 전문)', 'RESULTS.md'), ('C. 파이프라인 (PIPELINE_V6.md 전문)', 'PIPELINE_V6.md'),
                        ('D. 일정 출처 (OFFICIAL_SOURCES_V6.md 전문)', 'OFFICIAL_SOURCES_V6.md'), ('E. 2026-04-11 공연 근거 (EVIDENCE 전문)', 'EVIDENCE_20260411_munhak_concert.md')):
        body = (out / name).read_text().rstrip().split('\n')
        body = [('#' + line if line.startswith('#') else line) for line in body]          # demote headings one level
        L += [f'## {title}', ''] + body + ['']

    # ------------------------------------------------------------ appendix
    L += ['## F. 부록', '', '### F.1 테스트 구간 문학경기장 일정 상태 (123일)', '',
          '| 상태 | 일수 | 날짜 |', '|---|---:|---|']
    for s in ('HOME', 'NO_HOME', 'UNANNOUNCED'):
        g = states[states.state == s]
        dates = ', '.join(d.strftime('%m-%d') for d in g.date) if s == 'HOME' else f"{g.date.min().date()} ~ {g.date.max().date()} 중 해당일"
        L.append(f"| {s} | {len(g)} | {dates} |")
    un = states[states.state == 'UNANNOUNCED']
    L += ['', '미발표 구간의 경기 확률(작년 같은 시기 급등일 비율로 추정): ' + ', '.join(f"{m}월 평일 {g[g.group == 0].event_probability.mean():.2f} / 토 {g[g.group == 1].event_probability.mean():.2f} / 일·공휴일 {g[g.group == 2].event_probability.mean():.2f}" for m, g in un.groupby(un.date.dt.month)),
          '', '콘서트 예정일: ' + (', '.join(audit['munhak_concert_days']) or '없음'), '',
          '### F.2 행사일 시간대별 예측 (블렌드 → 최종, 승차 인원)', '',
          '바뀐 시간대만 적었다. 숫자는 `specialist_rows.csv`에서 읽었다.', '']
    seen = set()
    for r in audit['event_overlay']:
        key = (r['station'], r['date'])
        if key in seen: continue
        seen.add(key)
        acts = [x for x in audit['event_overlay'] if (x['station'], x['date']) == key]
        label = ' + '.join('상승 반영' if x['action'] == 'lift' else '전년 급등 제거' for x in acts)
        name = next((x.get('event') for x in acts if x.get('event')), '')
        line = hour_table(rows, r['station'], r['date'])
        if line: L += [f"- **{r['station']} {STATION.get(r['station'], '')} {day(r['date'])}** — {label}{(' · ' + name) if name else ''}", f"  {line}"]
    for d in audit['munhak_concert_days']:
        line = hour_table(rows, 3127, d)
        if line: L += [f"- **3127 문학경기장 {day(d)}** — 콘서트 예정일", f"  {line}"]
    L += ['', '### F.3 문학경기장 테스트 예측: 상태별 저녁 시간 평균 (승차 인원)', '']
    m = rows[rows.station_id == 3127].merge(states[['date', 'state', 'concert']], on='date')
    L += ['| 상태 | 일수 | 20시 | 21시 | 22시 | 23시 |', '|---|---:|---|---|---|---|']
    for s, g in m[~m.concert].groupby('state'):
        L.append(f"| {s} | {g.date.nunique()} | " + ' | '.join(f"{g[g.hour == h].base_boardings.mean():.0f}→{g[g.hour == h].final_boardings.mean():.0f}" for h in (20, 21, 22, 23)) + ' |')
    L += ['', '### F.4 검증 fold에서 행사 보정이 적용된 날 (전체 목록)', '', '| Fold | 역 | 날짜 | 처리 | 행사 | 비교 표본 | 그날 RMSLE 전 | 후 |', '|---:|---|---|---|---|---:|---:|---:|']
    for a in sorted(val['event_actions'], key=lambda r: (r['fold'], r['station'], r['date'], r['action'])):
        L.append(f"| {a['fold']} | {a['station']} {STATION.get(a['station'], '')} | {day(a['date'])} | {'상승 반영' if a['action'] == 'lift' else '전년 급등 제거'} | {a.get('event') or ''} | {a['comparable_events'] if a.get('comparable_events') is not None else ''} | {a['rmsle_before']:.3f} | {a['rmsle_after']:.3f} |")
    L += ['', '### F.5 설정 파일 (`configs/ensemble_v6.json`)', '', '```json', json.dumps(cfg, ensure_ascii=False, indent=1), '```', '',
          '### F.6 블렌드 가중치 산출 기록 (`blend_weights.json` 요약)', '', '```json', json.dumps(dict(weights=bw['weights'], method=bw['report']['method'], alternatives=bw['report']['ordinary_alternatives_reference_only']), ensure_ascii=False, indent=1), '```', '',
          '### F.7 문학 전용 모델 적합값 (기준일 2026-06-30)', '', '```json', json.dumps(audit['munhak_fit'], ensure_ascii=False, indent=1), '```', '',
          '### F.8 산출물 해시', '', '| 파일 | SHA-256 |', '|---|---|']
    for name in ('submission_ensemble_v6.csv', 'blend_weights.json', 'validation_summary.json', 'schedule_audit.json', 'specialist_rows.csv'):
        L.append(f"| outputs/ensemble_v6/{name} | `{hashlib.sha256((out / name).read_bytes()).hexdigest()}` |")
    for rel in ('configs/ensemble_v6.json', 'configs/domain_schedule_v6.json', 'src/v6/blend.py', 'src/v6/schedule.py', 'src/v6/munhak.py', 'src/v6/events.py', 'src/v6/pipeline.py'):
        L.append(f"| {rel} | `{hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()}` |")
    L.append('')
    (out / 'HANDOFF_V6.md').write_text('\n'.join(L))
    print('written', out / 'HANDOFF_V6.md', len(L), 'lines,', len('\n'.join(L)), 'chars')


def main():
    cfg = json.loads((ROOT / 'configs/ensemble_v6.json').read_text()); out = ROOT / cfg['output']
    doc = json.loads((ROOT / cfg['schedule']).read_text())
    sources(cfg, doc, out); handoff(cfg, doc, out)


if __name__ == '__main__':
    main()
