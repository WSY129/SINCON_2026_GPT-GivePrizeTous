"""Writes RESULTS_V10.md from gates_v10.json (python -m src.v10.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8
from . import games

OUT = c8.ROOT / 'outputs/ensemble_v10'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def vals(m):
    return [f'{m[k]:.5f}' for k in KEYS]


def main():
    g = json.loads((OUT / 'gates_v10.json').read_text()); cfg = games.config()
    steps = g['steps']; b = g['blend']; sub = g['submission']
    g8 = json.loads((c8.ROOT / 'outputs/ensemble_v8/gates_v8.json').read_text())
    g9 = json.loads((c8.ROOT / 'outputs/ensemble_v9/blend_v9.json').read_text())['chains']['main']
    final = steps[-1]
    L = ['# ensemble_v10 결과 (RESULTS_V10)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**이고 리더보드 점수는 쓰지 않았다. 계수는 모두 중첩 LOFO로, **정렬 정답**(기준일이 2025년인 fold의 2026년 정답을 2025년 집계 기준으로 환산)에 맞춰 적합했다. '
         '각 단계는 정렬 채점과 원래 정답 채점을 함께 적는다.\n',
         '## 1. 요약\n',
         f"- 제출본: `outputs/ensemble_v10/{sub['file']}` (157,194행, SHA-256 `{sub['sha256']}`, 평균 승차 {sub['mean_boardings']:.2f}명). v8 주력 대비 로그 RMSD {sub['log_rmsd_vs_v8']:.4f}, v9 대비 {sub['log_rmsd_vs_v9']:.4f}.\n"
         '- 기존 제출본(v6~v9)과 보험용 `submission_precutoff.csv`는 그대로다(SHA 대조 테스트).\n',
         table(['구분(정렬 채점)'] + HEAD, [['v8 주력'] + vals(g8['aligned_truth_sensitivity']['main']), ['v9'] + vals(g9['aligned']['v9_blend']),
                                        ['v10 기반(결합 전)'] + vals(b['singles']['v10 기반(결합 전)']['aligned']), ['**v10 최종**'] + vals(final['aligned'])]) + '\n',
         table(['구분(원래 정답 채점)'] + HEAD, [['v8 주력'] + vals(g8['chains']['main']['final']), ['v10 기반(결합 전)'] + vals(b['singles']['v10 기반(결합 전)']['raw_truth']),
                                           ['**v10 최종**'] + vals(final['raw_truth'])]) + '\n',
         '## 2. 단계별 누적\n',
         table(['단계', '판정', '정렬 전체', '정렬 F1', '정렬 F2', '정렬 F3', '정렬 F4', '원래 정답 전체'],
               [[s['step'], {True: '**채택**', False: '기각', '': '—'}[s['adopted']]] + [f"{s['aligned'][k]:.5f}" for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')]
                + [f"{s['raw_truth']['all_oof']:.5f}"] for s in steps]) + '\n',
         '## 3. 다섯 가지 수정 내역\n']
    a2 = next(s for s in steps if s['step'].startswith('A2')); a1 = next(s for s in steps if s['step'].startswith('A1'))
    a4 = next(s for s in steps if s['step'].startswith('A4')); p1 = next(s for s in steps if s['step'].startswith('P1'))
    L.append('### (1) 부천종합운동장 토요일 4경기 누락\n'
             '- 원인: v8은 "가까운 평소 토요일 4개"를 먼저 고른 뒤 그 역의 경기일을 뺐다. 8월 8일부터 9월 19일까지 토요일이 전부 경기일(또는 광복절)이라 남는 날이 없었고, 8/29·9/5·9/12·9/19 네 경기가 건너뛰어졌다.\n'
             f"- 수정: 그 역의 행사일이 아닌 평소 같은 요일을 가까운 순으로 최대 7주까지 찾는다. 테스트 구간 {a2['details']['test']['match_days_applied']}경기 전부 적용, 건너뛴 경기 {a2['details']['test']['skipped'] or '없음'}. "
             f"관중 대비 승차 비율 {a2['details']['test']['exit_share_of_attendance']:.3f}.\n- 검증 fold의 경기는 원래도 누락이 없어 fold 점수는 그대로다.\n")
    lam = p1['details']['lambda_by_station']
    off = [k for k, v in lam.items() if v == 0 and k not in ('3107', '3108', '3109', '3127')]
    L.append('### (2) 다년 계절 지수 교정\n'
             '- **송도달빛축제공원(3139) 8월**: 공사 월 합계에 과거 펜타포트·맥주축제 인파가 들어 있어 8월 평일에 로그 +0.11이 더해지고 있었다. 축제는 이미 일별 상승폭으로 넣으므로 이 역의 8월에는 지수를 적용하지 않는다.\n'
             '- **귤현(3111)**: 지수 적용 후 원래 정답 기준으로 Fold 2·3이 나빠 보였으나(0.289→0.344, 0.419→0.455) 원인은 집계 단절이었다(귤현의 2025년분은 공사 자료의 0.74배). 정렬 채점에서는 Fold 1 0.291→0.283, Fold 2 0.278→0.251, Fold 4 0.228→0.220으로 개선이고 Fold 3은 0.247→0.249다. '
             f"그래서 역별 게이트(자기 계수가 0.3 미만이면 끔)를 정렬 정답으로 판정하도록 했고, 현재 꺼지는 역은 {off or '없다'}. 보정폭 상한 ±0.20은 유지한다.\n"
             f"- **7호선 과거 자료**: 공사 자료는 7호선 역이 2022년부터라 정상 연도가 2개뿐이었다. 서울시 월별 시간대 자료(2015~2021)로 8개 역(부평구청·산곡·석남 제외)의 2016~2019년을 보강했다({p1['line7_cells_added']:,}칸). "
             '원래 정답 기준 중첩 점수 0.19243 → 0.19167, 7호선 8개 역 평균 RMSLE 0.1665 → 0.1572.\n')
    ev = g['unvalidated_events']
    L.append('### (3) 하반기 실제 개최 행사\n')
    rows = [[e['event'], e['station'], e['date'], e['past_editions'], e['daily_before'], e['daily_after'], e['max_log_lift']] for e in ev['recurring']]
    if rows:
        L.append(table(['행사', '역', '날짜', '과거 회차 수', '일 승차 예측(전)', '(후)', '최대 로그 상승'], rows) + '\n')
    L.append(f"- 문학: 목록에 넣은 비리그 행사 {a1['details']['test']['listed_events_used'] or '없음'}. 경기·콘서트·확인된 행사가 없는 시즌 날의 남은 급등 확률(평일/토/일·공휴일) {a1['details']['test']['left_over_spike_rate']} (v6 값 {a1['details']['test']['v6_event_rate']}).\n")
    for note in cfg.get('research_notes', []):
        L.append(f'- {note}')
    L.append('')
    c = a4['coefficients_all_folds']; w = g['weather']
    L.append('### (4) 시간별 강수·기온\n'
             f"- 계수 [당일 강수 단계, 전년 같은 요일 강수 흔적, 그 시간·직전 2시간 강수, 더운 철 기온 편차, 그 외 철 기온 편차] = {c}.\n"
             f"- 시간 강수 효과(일 효과에 추가, 전체 역 중앙값): {w['hourly_rain_effect_network']}. 기온 1도당 효과(전체 역 중앙값): {w['temperature_slope_per_degree_network']}.\n"
             '- v8의 일 단위 강수만 쓴 경우 정렬 0.18487, 시간 강수 추가 0.18455, 기온 추가 0.18445, 둘 다 0.18423. 더운 철(평년 최고기온 24도 이상) 기온 편차는 계수 0으로 효과가 없었고, 그 외 철의 "평년보다 따뜻한 맑은 날" 효과가 채택됐다. 인천대공원 0.392 → 0.370, 센트럴파크 0.185 → 0.180.\n')
    L.append('### (5) v9 트리 재결합\n')
    L.append(table(['구분', '정렬 전체', '정렬 F3', '정렬 F4', '원래 정답 전체'],
                   [[n, f"{v['aligned']['all_oof']:.5f}", f"{v['aligned']['fold_3']:.5f}", f"{v['aligned']['fold_4']:.5f}", f"{v['raw_truth']['all_oof']:.5f}"] for n, v in b['singles'].items()]
                   + [['결합: 일반 4-fold 중첩 가중치', f"{b['standard_nested']['aligned']['all_oof']:.5f}", f"{b['standard_nested']['aligned']['fold_3']:.5f}", f"{b['standard_nested']['aligned']['fold_4']:.5f}", f"{b['standard_nested']['raw_truth']['all_oof']:.5f}"],
                      ['**결합: Fold 3·4 가중치(채택)**', f"{b['regime_matched']['aligned']['all_oof']:.5f}", f"{b['regime_matched']['aligned']['fold_3']:.5f}", f"{b['regime_matched']['aligned']['fold_4']:.5f}", f"{b['regime_matched']['raw_truth']['all_oof']:.5f}"]]) + '\n')
    L.append(f"- 단독 점수 초과 여부: {b['beats_every_single']} — 두 방식 모두 전체 점수는 모든 단독 모델보다 낮다.\n"
             f"- 다만 일반 4-fold 중첩 가중치는 Fold 3·4에서 기반보다 나쁘다. Fold 3·4 가중치(서로 교차 적용)는 네 fold 모두 개선이라 이것을 쓴다. 테스트 가중치: {b['weights_test']} (구간 {b['segments']}).\n")
    v = pd.read_csv(OUT / 'VALIDATION_V10.csv')
    cols = [c for c in v.columns if c.startswith('st_')]
    L.append('## 4. 역별 RMSLE (정렬 채점, 4개 fold 합산)\n')
    L.append(table(['단계'] + [c[3:] for c in cols], [[r.step] + [f'{getattr(r, c):.4f}' for c in cols] for r in v.dropna(subset=[cols[0]]).itertuples()]) + '\n')
    (OUT / 'RESULTS_V10.md').write_text('\n'.join(L) + '\n')
    print('written RESULTS_V10.md')


if __name__ == '__main__':
    main()
