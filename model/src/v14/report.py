"""Writes RESULTS_V14.md and VALIDATION_V14.csv (python -m src.v14.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8
from .panel import EXTRA

OUT = c8.ROOT / 'outputs/ensemble_v14'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STEP = {'events': '행사(콘서트·배구, 상승폭 없던 날)', 'school': '학교 방학 변화 × 등하교 시간', 'low': '5명 미만 예측 보정', 'holiday': '연휴·공휴일 달력 사전값'}
KIND = {'hol_wd': '평일 공휴일', 'hol_sat': '토요일 공휴일', 'long_sat': '연휴 속 토요일', 'long_sun': '연휴 속 일요일', 'lunar0': '명절 당일', 'lunar_adj': '명절 전후', 'bridge': '징검다리', 'labor': '노동절', 'newyear': '신정'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    g = json.loads((OUT / 'gates_v14.json').read_text()); ge = json.loads((OUT / 'gates_v14e.json').read_text()); g12 = json.loads((c8.ROOT / 'outputs/ensemble_v12/gates_v12.json').read_text())
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v14.json').read_text())
    gen = {'v11': g12['generations']['v11']}; gen.update(g['generations'])
    vals = lambda m: [f'{m[k]:.5f}' for k in KEYS]; bold = lambda n, c: f'**{c}**' if n == 'v14' else c
    sub = g['submission']; st = g['steps']; rm = g['retrained_models']; sm = rm['single_model_aligned']; hd = st['holiday']; sl = ge['summer_leave']
    L = ['# ensemble_v14 결과 (RESULTS_V14)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 모델·가중치는 그 fold를 뺀 fold로만 정했고, 보정 단계는 그에 더해 그 fold와 날짜가 겹치는 행도 뺐다. '
         '보정 단계는 전체 점수가 좋아지고 어느 fold도 나빠지지 않을 때만 채택했다. 그래픽카드는 쓰지 않았다(LightGBM 재학습 포함, 이 컴퓨터 CPU).\n',
         '## 1. 점수\n',
         table(['정렬 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['aligned'])] for n, v in gen.items()]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['raw_truth'])] for n, v in gen.items()]) + '\n',
         f"- v13 대비 네 fold 모두 좋아졌다.\n- 제출본 두 개:\n  - `outputs/ensemble_v14/{sub['file']}` — 검증을 통과한 것만. SHA-256 `{sub['sha256']}`, v13 대비 로그 RMSD {sub['log_rmsd_vs_v13']:.4f}.\n"
         f"  - `outputs/ensemble_v14/{ge['submission']['file']}` — 위 파일 + fold로 검증할 수 없는 보정(5절). SHA-256 `{ge['submission']['sha256']}`, v14와 다른 행 {ge['submission']['rows_changed_vs_v14']:,}개.\n",
         '## 2. LightGBM 재학습 (새 입력 24개)\n',
         '- v11의 50개 입력에 다음을 더해 같은 설정으로 다시 학습했다: ' + ', '.join(f'`{e}`' for e in EXTRA[:24]) + '.\n',
         '  - `vac` 방학 중인 중·고교 비율(2025년 1~2월은 수집 구간 밖이라 1년 뒤 같은 요일 값으로 채움), `leave_week` 8월 1일이 낀 주와 다음 주의 평일, 달력 상세 7개, '
         '행사 종류 6개와 시작 시각 기준 경과 시간, 기상청 인천 관측 6개, 호우·폭염 특보 단계.\n',
         table(['모델(단독, 정렬)', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'],
               [[n] + [f"{sm[k][x]:.5f}" for x in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] for n, k in (('v11 LightGBM 127잎 (lgb_b)', 'lgb_b'), ('**재학습 127잎 (lgb_c)**', 'lgb_c'), ('v11 LightGBM 31잎 (lgb_a)', 'lgb_a'), ('재학습 31잎 (lgb_d)', 'lgb_d'))]) + '\n',
         f"- 스택(v12와 같은 절차)의 테스트 가중치: " + ', '.join(f'{k} {v:.3f}' for k, v in rm['stage1_weights_test'].items() if v > 0.004) + '. 재학습 모델이 v11 모델들을 대체했다. '
         f"스택만으로 {gen['v12']['aligned']['all_oof']:.5f} → {gen['v14 base (v12 pipeline + retrained models)']['aligned']['all_oof']:.5f}.\n",
         '- Fold 1은 그대로다(학습 자료가 8개월뿐이라 겨울방학·행사 사례가 없다).\n',
         '- 더 시험한 것: 다년 달력·주간 계절 사전값 2개를 입력으로 넣고 6,000회까지 학습 → Fold 4 0.1622 → 0.1618, Fold 2 0.1668 → 0.1671로 차이가 없어 쓰지 않았다.\n',
         '## 3. 보정 단계 (재학습 스택 위에서, 중첩 + 날짜 차단)\n',
         table(['단계', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정'],
               [[STEP[n], f"{s['before']['all_oof']:.5f} → {s['after']['all_oof']:.5f}"] + [f"{s['after'][f'fold_{k}'] - s['before'][f'fold_{k}']:+.5f}" for k in (1, 2, 3, 4)] + ['채택' if s['adopted'] else '기각']
                for n, s in st.items()]) + '\n',
         f"- 행사 단계는 v13과 같이 fold 예측만 바꾼다(테스트에서 바뀌는 역-날짜 {st['events'].get('test_station_days_changed', 0)}개).\n",
         f"- 방학 단계: 재학습 모델이 방학 비율을 직접 입력으로 받는데도 스택의 절반이 v10이라 효과가 남는다. 공통 기울기(06~08시, 15~17시) {st['school'].get('test_global_slopes')}, 테스트 방학 평일의 등하교 시간대 평균 {100 * st['school'].get('test_mean_log_change_school_hours', 0):+.1f}%.\n",
         '## 4. 연휴·공휴일 달력 사전값 (새 단계)\n',
         '- 서울 교통카드 **일별** 자료(7호선 부천·인천 9개 역 2016~2019, 환승 상대역 6곳 2016~2019·2022~기준일 전)로 "달력 위치만 보고 그날이 평소 같은 요일의 몇 배인지"를 예측하는 작은 모델을 만들었다. '
         '입력은 요일, 공휴일 여부, 연휴 길이와 연휴 안 위치, 연휴 전날·다음날, 징검다리, 설·추석까지 남은 날 등 13개다. 평가 기간의 승하차는 어느 노선 것도 들어 있지 않다(자료가 2026-06-30에서 끝남).\n',
         f"- 특수일마다 (사전값 − 우리 예측이 함의하는 배율의 역 중앙값)에 계수를 곱해 그날 전체를 같은 폭으로 옮긴다. 그날 배율이 중앙값에서 로그 {hd['typical_band']} 넘게 벗어난 역(성묘·행사 역)은 건드리지 않는다. "
         f"계수는 fold별 {', '.join(f'{v:.2f}' for v in hd['gamma_nested'].values())}, 테스트 {hd.get('gamma_test')}.\n",
         table(['날짜', '종류', '사전값 − 예측', '적용한 로그 이동', '적용 역 수'], [[d['date'], KIND.get(d['kind'], d['kind']), f"{d['prior_minus_forecast']:+.3f}", f"{d['applied_log_shift']:+.3f}", d['typical_stations']] for d in hd.get('test_days', [])]) + '\n',
         '- 추석 구간은 9/24(전날) 낮추고, 9/25(당일) 올리고, 9/27(일) 낮춘다. 앞서 2019년(같은 요일 배치) 실측과 비교했을 때 보였던 방향과 같다.\n',
         '- 종류별로 계수를 따로 두는 방식은 Fold 2가 나빠져(+0.0010) 쓰지 않았다.\n',
         '## 5. fold로 검증할 수 없는 보정 (v14e에만)\n',
         f"### (1) 여름 집단휴가 2주 — 평일 {sl['days'][0]} ~ {sl['days'][-1]}\n",
         f"- {cfg['summer_leave']['basis']}\n",
         table(['시간대', '06~09시', '10~15시', '16~19시', '20~23시'], [['2025년 실제 하락(역 중앙값)'] + [f'{x:+.3f}' for x in sl['median_fall_2025']], ['v14 예측의 하락'] + [f'{x:+.3f}' for x in sl['median_fall_v14']],
                                                                ['v14e에서 더한 이동(역 중앙값)'] + [f'{x:+.3f}' for x in sl['median_shift_by_hour_group']]]) + '\n',
         f"- 대상 {sl['stations']}개 역, 2주 평일 하루 합계 평균 {100 * sl['mean_day_total_change']:+.1f}%. 이동이 큰 역(시간대 순): " + ', '.join(f'{k} {v}' for k, v in list(sl['largest'].items())[:5]) + '.\n',
         '### (2) 행사별\n',
         table(['행사', '역', '날짜', '보정 전(시:명)', '보정 후'], [[e['name'], e['station'], e['date'], e['before'], e['after']] for e in ge['entries']]) + '\n',
         '- 근거는 v13e와 같다(`configs/ensemble_v14.json`). 송도마라톤은 재학습 모델이 이미 30%쯤 올려 놓아 배율을 1.8에서 1.4로 줄였다.\n',
         '- 문학 기존 행사일(PLAVE·불꽃야구·Sustainable Wave), 흠뻑쇼, 펜타포트·맥주축제는 v14에서도 v10의 조사 기반 상승폭을 유지한다. 재학습 모델은 PLAVE 22시를 약 1,100명으로 보지만(v10 695명) 스택 규칙상 쓰이지 않는다.\n',
         '## 6. 시험했지만 버린 것 (이번 판)\n',
         table(['시도', '결과', '버린 이유'],
               [['역별 일관 편향 보정(학습 fold들에서 부호가 같은 역만)', '−0.0001 ~ +0.0002', '조건을 만족하는 역이 0~2개뿐, 효과 없음'],
                ['방학 비율을 구 단위·학교급별로', '−0.0008 ~ −0.0009', '전체 비율 하나를 쓴 것과 차이 없음'],
                ['방학 효과 시간대 확장(저녁 18~20시, 시간별 계수)', '−0.0008 ~ −0.0010', '차이가 잡음 수준'],
                ['연휴 사전값을 종류별 계수로', '+0.0002', 'Fold 2 악화'],
                ['연휴 사전값을 7호선 자료만으로', '−0.0002', 'Fold 2·3 소폭 악화(두 자료원 평균은 네 fold 모두 개선)'],
                ['사전값 2개 입력 + 6,000회 학습', '차이 없음', '2절 참조']]) + '\n',
         '## 7. 한계\n',
         '- 여름방학·휴가철을 예측 대상으로 하는 fold가 없다. 방학 기울기는 겨울방학과 개학 전후에서, 휴가 주간은 2025년 한 해에서 배운 것이다.\n',
         '- 5절 전부. 특히 휴가 주간이 7/27 주인지 8/3 주인지, PLAVE·최광야구의 실제 종료 시각과 관중 수는 확인되지 않았다.\n',
         f"- 로컬 검증 현재 {gen['v14']['aligned']['all_oof']:.4f}. 오차 구성은 그대로다: 칸 단위 잡음 65%(그 절반이 셀 수 잡음), 일 충격 11%, 시간대 모양 12.5%, 수준 11%.\n"]
    (OUT / 'RESULTS_V14.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(v[s][k], 6) for k in KEYS}) for n, v in gen.items() for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n + (' (adopted)' if s['adopted'] else ' (rejected)'), scoring='aligned', **{k: round(s['after'][k], 6) for k in KEYS}) for n, s in st.items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V14.csv', index=False)
    print('written RESULTS_V14.md, VALIDATION_V14.csv')


if __name__ == '__main__':
    main()
