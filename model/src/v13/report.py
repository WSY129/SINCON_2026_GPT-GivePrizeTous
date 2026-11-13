"""Writes RESULTS_V13.md and VALIDATION_V13.csv (python -m src.v13.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v13'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STEP = {'events': '행사(콘서트·배구, 상승폭 없던 날)', 'games': '홈경기 종료 전후 시간대', 'school': '학교 방학 변화 × 등하교 시간', 'low': '5명 미만 예측 보정'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    g = json.loads((OUT / 'gates_v13.json').read_text()); ge = json.loads((OUT / 'gates_v13e.json').read_text()); g12 = json.loads((c8.ROOT / 'outputs/ensemble_v12/gates_v12.json').read_text())
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v13.json').read_text())
    gen = {k: g12['generations'][k] for k in ('v10', 'v11')}; gen.update(g['generations'])
    vals = lambda m: [f'{m[k]:.5f}' for k in KEYS]; bold = lambda n, c: f'**{c}**' if n == 'v13' else c
    sub = g['submission']; st = g['steps']; sc = st['school']; lo = st['low']; ev = st['events']
    L = ['# ensemble_v13 결과 (RESULTS_V13)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. v13의 보정은 모두 "그 fold를 뺀 fold"로만 적합했고, '
         '그에 더해 **그 fold와 날짜가 겹치는 행을 학습에서 뺐다**(fold끼리 두 달씩 겹쳐 같은 날의 정답이 새는 것을 막기 위해서다). 단계는 전체 점수가 좋아지고 **어느 fold도 나빠지지 않을 때만** 채택했다.\n',
         '## 1. 점수\n',
         table(['정렬 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['aligned'])] for n, v in gen.items()]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['raw_truth'])] for n, v in gen.items()]) + '\n',
         f"- v12 대비 네 fold 모두 좋아졌다(정렬·원래 정답 채점 둘 다).\n",
         f"- 제출본 두 개:\n  - `outputs/ensemble_v13/{sub['file']}` — 검증을 통과한 보정만. SHA-256 `{sub['sha256']}`, v12 대비 로그 RMSD {sub['log_rmsd_vs_v12']:.4f}, 바뀐 행 {sub['rows_changed_vs_v12']:,}개.\n"
         f"  - `outputs/ensemble_v13/{ge['submission']['file']}` — 위 파일 + 행사별 개별 보정 {len(ge['entries'])}건(4절, **fold로 검증 불가**). SHA-256 `{ge['submission']['sha256']}`, v13과 다른 칸 {ge['submission']['rows_changed_vs_v13']}개.\n",
         '## 2. 단계별 결과 (정렬 채점, 중첩 + 날짜 차단)\n',
         table(['단계', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정'],
               [[STEP[n], f"{s['before']['all_oof']:.5f} → {s['after']['all_oof']:.5f}"] + [f"{s['after'][f'fold_{k}'] - s['before'][f'fold_{k}']:+.5f}" for k in (1, 2, 3, 4)] + ['채택' if s['adopted'] else '기각']
                for n, s in st.items()]) + '\n',
         '## 3. 채택한 보정\n',
         '### (1) 학교 방학 변화 × 등하교 시간 — 테스트에 직접 작용\n',
         f"- v12의 수준은 기준일 직전 8주에서 잡는다. 그 8주와 예측일의 **방학 중인 중·고교 비율(NEIS 학사일정)** 이 다르면 등하교 시간대가 체계적으로 어긋났다. "
         f"기준 8주의 방학 비율은 fold별 {', '.join(f'{v:.2f}' for v in sc['anchor_share'].values())}, 테스트는 {sc['anchor_share_test']:.2f}(5~6월, 학기 중)이다.\n",
         f"- 보정 = (예측일 방학 비율 − 기준 8주 방학 비율) × 기울기. 기울기는 06~08시와 15~17시 각각 전체 공통 하나 + 역별 편차(리지 {sc['ridge_station']:.0f}). "
         '역 전체 수준이 틀린 것을 방학 효과로 오인하지 않도록, 같은 역·같은 날의 **다른 시간대 대비**로 추정했다. 자료 기간 안에 개통한 신설 3개 역은 공통 기울기만 쓴다(아침 통근이 아직 늘고 있어 방학 효과처럼 보인다).\n',
         '- fold별로 따로 구한 공통 기울기(06~08시, 15~17시): ' + ' / '.join(f"({a:+.3f}, {b:+.3f})" for a, b in sc['global_slopes_nested'].values()) + ' — 네 번 모두 같은 부호, 비슷한 크기다.\n',
         f"- 테스트 적용: 방학 비율이 절반을 넘는 평일 {sc['test_days_above_half']}일({sc['test_first_last'][0]} ~ {sc['test_first_last'][1]})에 등하교 시간대 평균 {100 * sc['test_mean_log_change_school_hours']:+.1f}%, 하루 합계 {100 * sc['test_mean_log_change_day_total']:+.1f}%. "
         '기울기가 큰 역: ' + ', '.join(f"{k} ({v[0]:+.2f}, {v[1]:+.2f})" for k, v in sc['strongest_stations'].items()) + '.\n',
         '### (2) 5명 미만 예측 보정 — 테스트에 직접 작용\n',
         f"- 예측값이 5명 미만인 칸은 네 fold 모두에서 실제가 예측보다 컸다. 예측 구간({lo['edges']})별 평균 잔차를 더한다(50칸만큼 0 쪽으로 축소). 테스트에서 바뀌는 칸은 {lo['test_cells_changed']}개, 더하는 값(로그) {lo['test_added_log']}.\n",
         '### (3) 행사일(콘서트·배구) — **fold 예측만 바뀌고 테스트 예측은 그대로**\n',
         f"- v12에 상승폭이 아직 없는 행사일에, 같은 역·같은 종류의 다른 행사일 잔차를 시간대별로 평균해 더한다(유사일 {ev['min_analogue_days']}일 이상 필요, {ev['shrink_days']:.0f}일만큼 축소). "
         f"제곱오차 감소: 콘서트 {ev['sse_gain_by_kind_and_fold']['concert']}, 배구 {ev['sse_gain_by_kind_and_fold']['kovo']}. 역별: {ev['sse_gain_by_station']}.\n",
         '- 테스트 기간의 콘서트(흠뻑쇼·PLAVE)는 v12에 이미 상승폭이 들어 있고 삼산체육관 배구는 7~10월에 없어서, **이 단계는 로컬 점수만 내린다.** 로컬 개선 0.0026 가운데 0.0013이 여기서 나온다.\n',
         '- 넣지 않은 종류: 공연장(예술회관·부천아트센터)과 축제는 fold에서 이득이 0이거나 음수였는데도 테스트의 예술회관 공연일 예측을 평균 6%(최대 18%) 올렸다. 공연일로 분류된 날 가운데 원인 모를 급증일(2026-03-01, 27,762명)이 있는 영향으로 보이지만 따로 떼어 확인하지는 않았다. '
         '야구장 비리그 행사는 표본이 적고 팀 목록의 2026-06-20 행사일은 승차가 평일 수준이라(행사 흔적 없음) 일괄 적용하지 않았다.\n',
         '## 4. 행사별 개별 보정 (v13e에만, fold로 검증 불가)\n',
         table(['행사', '역', '날짜', '보정 전(시:명)', '보정 후'], [[e['name'], e['station'], e['date'], e['before'], e['after']] for e in ge['entries']]) + '\n']
    for e in cfg['extras']:
        L.append(f"- **{e['name']}**: {e['basis']}\n")
    L += ['- 싸이 흠뻑쇼(7/11)는 손대지 않았다. v12 예측(아시아드 21~23시 475·1,367·297)이 작년 같은 공연의 실측(612·1,652·269)에 이미 맞춰져 있다.\n',
          '- 문학 우천취소 2일(8/22, 9/25)은 경기 시작 1시간 반~2시간 20분 전에 취소돼 현장 인파 보정을 넣지 않았다. 8/5~6 폭염취소일은 이미 경기 없는 날로 처리돼 있다.\n',
          '## 5. 시험했지만 버린 것 (모두 같은 검증 방식)\n',
          table(['시도', '전체 변화', '버린 이유'],
                [['새 자료 47개 입력으로 잔차를 학습하는 LightGBM', '+0.0090', '네 fold 모두 악화'],
                 ['역×요일유형×시간대 평균 잔차 보정', '+0.0044', '네 fold 모두 악화: 시간대 모양 편향은 기간마다 달라 옮겨 쓸 수 없음'],
                 ['기상청 실측 강수·적설·한파(계수 19개)', '+0.0014', '네 fold 모두 악화'],
                 ['기상청 강수 1mm 이상 × 평일/휴일(계수 3개)', '−0.00002', 'Fold 2·4 악화, 효과 없음'],
                 ['기상특보', '+0.0001', '효과 없음'],
                 ['대학 학기 변화(4개 역)', '−0.00002', '효과 없음'],
                 ['서울 일별 10년 자료의 주 단위 계절 사전값', '−0.0002', 'Fold 1·2 악화(+0.0006, +0.0008), Fold 3·4만 개선'],
                 ['행사 직접 추정값과 v12의 혼합', '−0.0006', 'Fold 4 악화(가중치가 fold마다 0~1로 흔들림)'],
                 ['행사 LightGBM(종류·역·시간·시작 시각·기존 상승폭)', '−0.0007', 'Fold 4 악화'],
                 ['홈경기 종료 전후 시간대 평균 잔차', '−0.00001', 'Fold 4 악화, 효과 없음'],
                 ['역세권 인구 변화', '−0.0001', '효과 없음']]) + '\n',
          '## 6. 0.13까지 남은 거리\n',
          f"- 로컬 검증기 기준 현재 **{gen['v13']['aligned']['all_oof']:.4f}**다. 지금까지 로컬에서 0.15대가 나온 적은 없고(v11 0.1704, v12 0.1700), 0.15대는 Public(사용자 제공) 점수에서 본 값이다. 두 점수는 척도가 다르다.\n",
          '- v12 오차 분해(`outputs/analysis_20261113/WEAKNESS_AND_NEW_DATA.md`): 모든 역-날짜의 하루 총량을 완벽히 알아도 0.149, 셀 수 자체의 잡음만 0.098이다. 그 사이를 메우려면 "그 시간에 그 역에 몇 명이 탔는지"에 준하는 정보가 필요한데, 이번에 시험한 자료 가운데 그런 것은 없었다.\n',
          '- 그래서 0.13을 약속하지는 않는다. 대신 위 표처럼 **검증을 통과하는 것만 쌓는 방식**으로 계속 내릴 수 있고, 이번 판에서 0.1700 → 0.1675가 그렇게 나왔다. 남은 후보는 연휴 일별 배율(서울 일별 자료), 신설 역 성장, 문학 비야구 행사 실제 목록이다.\n',
          '## 7. 검증하지 못한 것\n',
          '- 여름방학(7/22~8/14) 보정의 기울기는 겨울방학과 2025년 여름방학 직후 구간에서 배운 값이다. 여름방학 한가운데를 예측하는 fold는 없다.\n',
          '- 4절의 개별 보정 전부(종료 시각·관중 수 미공개 포함).\n- 추석 일별 배율: v12는 9/24·9/25·9/27을 평소 같은 요일의 0.44·0.29·1.02배로 본다. 2019년(같은 요일 배치)은 0.37·0.35·0.92였지만 2025년 추석 당일은 0.27이어서 방향을 정하지 못해 그대로 뒀다.\n']
    (OUT / 'RESULTS_V13.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(v[s][k], 6) for k in KEYS}) for n, v in gen.items() for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n + (' (adopted)' if s['adopted'] else ' (rejected)'), scoring='aligned', **{k: round(s['after'][k], 6) for k in KEYS}) for n, s in st.items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V13.csv', index=False)
    print('written RESULTS_V13.md, VALIDATION_V13.csv')


if __name__ == '__main__':
    main()
