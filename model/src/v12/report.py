"""Writes RESULTS_V12.md and VALIDATION_V12.csv from gates_v12.json (python -m src.v12.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v12'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
NAME = {'v10': 'v10', 'lgb_a': 'LightGBM(31잎)', 'lgb_b': 'LightGBM(127잎)', 'xgb': 'XGBoost', 'cb': 'CatBoost', 'dec_lgb': '분해형 LightGBM', 'dec_xgb': '분해형 XGBoost',
        'dec_cb': '분해형 CatBoost', 'low_tw': '저수요 트위디', 'low_po': '저수요 포아송', 'tabm': 'TabM', 'stg': '시공간 그래프'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def vals(m):
    return [f'{m[k]:.5f}' for k in KEYS]


def wtext(w):
    return ', '.join(f'{NAME.get(k, k)} {v:.3f}' for k, v in sorted(w.items(), key=lambda x: -x[1]) if v >= 0.005)


def main():
    g = json.loads((OUT / 'gates_v12.json').read_text()); cfg = json.loads((c8.ROOT / 'configs/ensemble_v12.json').read_text())
    meta = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    sub = g['submission']; t = g['test']; gen = g['generations']; f4 = lambda d: ' / '.join(f'{d[str(k)]:.2f}' for k in (1, 2, 3, 4))
    bold = lambda n, c: f'**{c}**' if n == 'v12' else c
    worse = g.get('folds_worse_than_v11', {})
    L = ['# ensemble_v12 결과 (RESULTS_V12)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 가중치·혼합 비율·채택 여부는 모두 중첩(그 fold를 뺀 fold로만 결정)이고 정렬 정답으로 적합했다. '
         '정렬 채점 = 기준일이 2025년인 fold의 2026년 정답을 2025년 집계 기준으로 환산한 채점, 원래 정답 채점 = 환산 없이 채점. 테스트(기준일 2026-06-30)에는 집계 기준 변화가 없어 정렬 채점이 테스트 상황에 해당한다.\n',
         '그래픽카드 없이 저장된 v11 모델 11종의 반복 지점 예측과 v10 예측만으로 만들었다. 기존 판본(v1~v11)과 보험용 파일은 바꾸지 않았다(SHA 대조 테스트).\n',
         '## 1. 세대별 비교\n',
         table(['정렬 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['aligned'])] for n, v in gen.items()]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(v['raw_truth'])] for n, v in gen.items()]) + '\n',
         f"- **정렬 채점**: v11 대비 전체 {gen['v11']['aligned']['all_oof']:.5f} → {gen['v12']['aligned']['all_oof']:.5f}, Fold 3+4 {gen['v11']['aligned']['fold_3_4']:.5f} → {gen['v12']['aligned']['fold_3_4']:.5f}. "
         + ('네 fold 모두 나빠지지 않았다.\n' if not worse else '나빠진 fold: ' + ', '.join(f"Fold {k} {gen['v11']['aligned'][f'fold_{k}']:.5f} → {gen['v12']['aligned'][f'fold_{k}']:.5f} ({d:+.5f})" for k, d in worse.items())
            + '. **「전체 fold 비악화」 조건은 이 fold에서 지키지 못했다.** 역별 수축 강도를 0.5·1·2배로 바꿔 본 점검에서 Fold 2와 Fold 3이 서로 반대로 움직였고(1.0배: Fold 2 −0.00000 / Fold 3 +0.00004, 2.0배: −0.00007 / +0.00019) '
            '네 fold를 모두 만족하는 설정이 없어, 사전에 정한 0.25를 그대로 두었다.\n'),
         f"- **원래 정답 채점**: 전체 {gen['v11']['raw_truth']['all_oof']:.5f} → {gen['v12']['raw_truth']['all_oof']:.5f}로 좋아졌지만 Fold 2({gen['v11']['raw_truth']['fold_2']:.5f} → {gen['v12']['raw_truth']['fold_2']:.5f})와 "
         f"Fold 3({gen['v11']['raw_truth']['fold_3']:.5f} → {gen['v12']['raw_truth']['fold_3']:.5f})은 나빠졌다. 이 두 fold는 정답에 2026년 집계 기준 상승이 섞여 있다. 어느 단계가 원인인지는 v11과 단계 구성이 달라 분해하지 못했다.\n",
         f"- 제출본: `outputs/ensemble_v12/{sub['file']}` — 157,194행, SHA-256 `{sub['sha256']}`, 평균 승차 {sub['mean_boardings']:.2f}명, v11 대비 로그 RMSD {sub['log_rmsd_vs_v11']:.4f}, 평균 로그 차이 {sub['mean_log_diff_vs_v11']:+.4f}.\n",
         '## 2. 단계별 변화 (정렬 채점, 중첩)\n',
         table(['단계', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], [[k] + [f"{v[x]:.5f}" for x in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] for k, v in g['ladder'].items()]) + '\n',
         '- "행사일 v10 유지"와 "신설 역 규칙"은 fold 점수를 올리는 단계가 아니다. 테스트에서 적용하는 규칙을 fold 점수에도 똑같이 적용한 것이다(v11과 같은 방식).\n']
    ts, ca = g['two_stage'], g['checkpoint_average']
    L += ['## 3. 네 가지 교정\n', '### (1) 다중 지점 평균과 이단계 구간별 스태킹\n',
          f"- 지점 평균: 다른 fold로 고른 지점과 양옆 지점을 {ca['kernel']} 비중으로 평균했다. 효과는 거의 없다 — 가중치 한 벌 스태킹 {ca['single_checkpoint']['all_oof']:.5f} → {ca['averaged']['all_oof']:.5f}, "
          + ', '.join(f"{NAME[n]} 단독 {v['single']:.5f} → {v['averaged']:.5f}" for n, v in ca['best_single_model'].items()) + '.\n',
          f"- 이단계: 1단계는 전체 칸의 비음수 가중치, 2단계는 구간별 비음수 가중치(1단계 가중치 쪽으로 리지 {ts['ridge_to_stage1']}). 구간 전용 가중치는 **네 fold 모두에서 그 구간 오차가 줄어든 구간만** 채택했다.\n",
          table(['구간', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '전용 가중치'], [[k] + [f'{x:+.1f}' for x in v] + ['채택' if k in ts['segments_with_own_weights'] else '기각(1단계 가중치 사용)']
                                                                         for k, v in ts['sse_change_by_segment_and_fold'].items()]) + '\n',
          '  (값 = 구간 전용 가중치를 썼을 때 그 구간 제곱오차 합의 변화, 음수가 개선)\n',
          f"- 가중치 한 벌 {ts['one_weight_set']['all_oof']:.5f} → 모든 구간 전용 {ts['all_segments']['all_oof']:.5f}(Fold 4 {ts['one_weight_set']['fold_4']:.5f} → {ts['all_segments']['fold_4']:.5f}로 악화) → "
          f"채택 구간만 {ts['gated']['all_oof']:.5f}(Fold 4 {ts['gated']['fold_4']:.5f}). 문학경기장 전용 가중치는 Fold 1~3에서는 좋아지지만 Fold 4에서 나빠져 기각했다.\n",
          f"- 테스트 가중치 — 1단계: {wtext(t['stage1_weights'])}. " + ' '.join(f"{k}: {wtext(v)}." for k, v in t['segment_weights'].items()) + '\n']
    sw = g['station_weights']
    L += ['### (2) 역별 연속 수축 (이진 게이트 대체)\n',
          f"- 역마다 스택과 v10 사이의 가중치 α를 최소제곱으로 구하고 1(스택) 쪽으로 수축했다(수축 강도 = 역별 분모 중앙값의 0.25배, 사전 고정).\n",
          table(['방식'] + HEAD, [['게이트 없음'] + vals(sw['no_gate']), ['이진 게이트(v11 방식)'] + vals(sw['binary_gate']), ['**연속 수축(채택)**'] + vals(sw['empirical_bayes'])]) + '\n',
          '- 테스트 α가 낮은 역: ' + ', '.join(f"{s} {meta[int(s)]} {a:.2f}" for s, a in sorted(t['station_alpha'].items(), key=lambda x: x[1])[:8]) + '. v11에서 0으로 묶였던 5개 역(계양·동수·예술회관·국제업무지구·완정)이 0.16~0.56 사이 값으로 바뀌었다.\n']
    ed = g['event_days']
    L += [f"- 행사 역-날짜: fold에서는 v10 유지 {ed['all_v10']['all_oof']:.5f}, 규칙 없음 {ed['no_rule']['all_oof']:.5f}로 유지 쪽이 불리하지만 이 비교는 v10에 불리하게 치우쳐 있다. fold에서는 과거 회차가 곧 채점 대상이라 v10에 상승폭이 없고, "
          '테스트에서는 그 과거 회차에서 뽑은 상승폭이 v10에 들어 있기 때문이다. 그래서 v11과 같이 v10 값을 유지했다(문학은 중첩 적합 가중치도 0.92~1.00으로 같은 결론).\n']
    b = g['bucheon']; bo, be = b['outside_exit_hours'], b['exit_hours']
    L += ['### (3) 부천종합운동장 퇴장 시간대\n',
          f"- 원인: v11은 이 역 전체를 전용 트리 모델로 바꿨고, 트리는 학습 범위를 넘는 관중(저녁 경기 최대 6,569명)에서 값이 멈춘다. 관중 {b['crowd_band']:,}명 이상 경기의 퇴장 시간대 평균 로그 오차(fold 1~4): "
          f"v10 {f4(be['mean_log_error_high_crowd']['v10'])}, 스택 {f4(be['mean_log_error_high_crowd']['stack'])}, 혼합 {f4(be['mean_log_error_high_crowd']['blended'])}.\n",
          f"- 비퇴장 시간: 전용 모델 가중치를 중첩 적합(fold별 {f4(bo['weight_on_station_model_nested'])}, 테스트 {bo['weight_test']:.2f}). 제곱오차 합 "
          + ', '.join(f"F{k} {bo['sse_before'][str(k)]:.1f} → {bo['sse_after'][str(k)]:.1f}" for k in (1, 2, 3, 4)) + ' (Fold 1·2는 소폭 악화, Fold 3·4와 합계는 개선).\n',
          f"- 퇴장 시간(킥오프 +1~+4시간): 전용 모델을 쓰지 않고, 관중 {b['crowd_band']:,}명 이상 경기만 v10의 관중 비례 회귀와 혼합. v10 가중치 fold별 "
          + ' / '.join(f"{be['weight_on_v10_nested_low_high'][str(k)][1]:.2f}" for k in (1, 2, 3, 4)) + f", 테스트 {be['weight_test_low_high'][1]:.2f} (미만 경기는 0.00 — 스택이 더 정확). 제곱오차 합:\n",
          table(['퇴장 시간대', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '합계'], [[n] + [f"{d[str(k)]:.2f}" for k in (1, 2, 3, 4)] + [f"{sum(d.values()):.2f}"]
                                                                  for n, d in (('v10', be['sse_v10']), ('스택', be['sse_stack']), ('**혼합(채택)**', be['sse_blended']))]) + '\n',
          f"- 역 전체 RMSLE(정렬, 4 fold): v10 {b['station']['v10']:.4f} → v11 {b['station']['v11']:.4f} → v12 {b['station']['after']:.4f}. Fold 3 퇴장 시간대는 혼합이 스택보다 조금 나쁘다(그 fold의 v10은 고관중 경기를 평균 로그 0.34 낮게 봤다).\n",
          '테스트 홈경기 퇴장 시간대 예측(명):\n',
          table(['날짜', '관중', '시간대', 'v10', 'v11', 'v12'], [[m['date'], f"{m['attendance']:,}", f"{m['hours'][0]}~{m['hours'][-1]}시", m['v10'], m['v11'], m['v12']] for m in t['bucheon_matches']]) + '\n',
          '- 한계: 9천 명대 저녁 경기는 학습 구간에 없다. 관중 6천 명대 저녁 경기의 실제 최대 시간대 승차가 1,300~2,100명이었던 점을 보면 v12의 1,200~1,300명도 여전히 낮을 수 있다. 중첩 적합 가중치(0.47)를 넘겨 올리지는 않았다.\n']
    ns = g['new_stations']
    L += ['### (4) 신설 3개 역 기상 잔차 모델\n',
          f"- 연중 일자·개통 후 경과일·전년 항 없이 시간별 강수·기온·일사와 공휴일 항 {len(ns['features'])}개만 쓰는 작은 LightGBM(7잎, 300회, 3시드)이다. 개통 후 {ns['excluded_first_days']}일은 학습에서 빼고, "
          '학습 잔차에서 성장 추세를 뺀 뒤 학습했다. 예측에서는 **v10과의 월 평균 차이를 제거**해, 수준과 계절 흐름은 v10의 선형 성장 기준선을 그대로 두고 일별·시간별 기상·공휴일 효과만 얹었다. '
          '혼합 가중치는 역별로 Fold 3 ↔ Fold 4를 서로 바꿔 적합했고, 두 fold 모두에서 v10보다 좋아진 역만 채택했다.\n',
          table(['역', '이름', '채택', '테스트 가중치', 'v10 (F2/F3/F4)', '혼합 (F2/F3/F4)', '참고: 월 평균을 빼지 않은 혼합'],
                [[s_, meta[int(s_)], '예' if v['adopted'] else '아니오', f"{v['weight_test']:.2f}", ' / '.join(f"{v['v10'][k]:.4f}" for k in ('2', '3', '4')),
                  ' / '.join(f"{v['blended'][k]:.4f}" for k in ('2', '3', '4')), ' / '.join(f"{x:.4f}" for x in ns['blended_without_centring'][s_].values())] for s_, v in ns['per_station'].items()]) + '\n',
          '- 채택은 아라 한 역뿐이고 효과도 작다(Fold 3 0.1512 → 0.1502, Fold 4 0.1277 → 0.1155, Fold 2는 0.1650 → 0.1662로 소폭 악화).\n',
          '- 월 평균을 빼지 않으면 신검단중앙·아라가 Fold 3·4 모두 더 크게 좋아진다(참고 열). 그러나 분해해 보니 그 개선은 기상 효과가 아니라 **기준선 수준 차이**였다. Fold 3·4에서 v10은 두 역을 월 평균 로그 0.02~0.13 높게 봤고(하루 이틀뿐인 첫 달 제외) '
          '새 모델의 기준선이 그만큼 낮았을 뿐이며, 기상 항 자체의 월 평균은 ±0.04 이내였다. 테스트 달에서는 이 수준 차이가 7~8월 +0.04~+0.07, 9~10월 −0.01~−0.07로 부호가 달라 fold에서 확인된 방향이 아니다. 그래서 수준 차이는 싣지 않았다.\n',
          '- 관찰(반영하지 않음): v10의 신설 역 공통 성장 기울기는 Fold 3·4에서 신검단중앙·아라에는 과대(잔차÷성장 항 −0.33~−0.72), 검단호수공원에는 과소(+1.6~+2.3)였다. 역마다 부호가 다르고 Fold 2의 신검단중앙은 반대(+0.34)여서 테스트에 반영하지 않았다.\n']
    L += ['### (5) 시민의 날 대축제 (2026-10-10, 인천대공원)\n']
    for e, o in zip(cfg['one_off_events'], t['one_off']):
        L += [f"- {e['name']}: 예정대로 열렸다(10월 11일자 보도). 예정 일정은 11:00~21:00, 기념식 18:00~18:40, 이후 축하공연과 드론 라이트 쇼(마지막 순서)다. **실제 종료 시각과 관중 수는 어디에도 발표되지 않았다.** "
              '같은 장소였던 2024년 회차의 주최 측 발표는 하루 "3만여 명"이다.\n',
              f"- 반영: 저녁 관객 {e['attendance']:,}명 가정 × 이 역 이용 10% × 축소 0.6 = {e['attendance'] * e['transit_share']:.0f}명을 20:30~22:30에 배분. 17~23시 승차 {o['boardings_17_23_before']} → {o['boardings_17_23_after']}.\n",
              '- 크기 점검(마감 이전 이 역 실측): 평소 10월 토요일 20~21시는 30~50명, 저녁 행사가 있던 날의 한 시간 급등은 283명(2025-10-10 20시)·530명(2025-06-07 19시). 검증할 과거 표본이 없는 가정값이다.\n']
    L.append('## 4. 고오차 역 변화 (정렬 채점, 4개 fold 합산 RMSLE)\n')
    cols = [c for c in next(iter(g['stations'].values()))]
    L.append(table(['역', '이름'] + [c.replace('v8p', 'v8 보험용') for c in cols] + ['v11→v12'],
                   [[s, meta[int(s)]] + [f'{v[c]:.4f}' for c in cols] + [f"{v['v12'] - v['v11']:+.4f}"] for s, v in g['stations'].items()]) + '\n')
    up = [f"{meta[int(s_)]}({v['v12'] - v['v11']:+.4f})" for s_, v in g['stations'].items() if v['v12'] - v['v11'] > 0.0005]
    L += ['- v11보다 나빠진 역: ' + (', '.join(up) or '없음') + '. 역별로 어느 단계 때문인지는 분해하지 않았다.\n',
          '## 5. 검증하지 못한 것\n',
          '- 시민의 날 상승폭(관중·종료 시각 미발표, 과거 표본 없음), 테스트 기간 행사일의 v10 상승폭 유지(위 설명대로 fold가 대표하지 못함), 부천 9천 명대 저녁 경기(학습 범위 밖).\n',
          '- 신설 역 기상 모델의 학습 구간(개통 60일 이후인 2025-08-27부터 기준일까지)에는 7월과 8월 초·중순이 없고, 검증 fold도 겨울·봄이다. 한여름 반응은 검증되지 않았다.\n',
          '- 저수요·공휴일·문학 구간 전용 가중치는 기각했으므로 이 구간들은 v11과 같은 구조(가중치 한 벌)다.\n']
    (OUT / 'RESULTS_V12.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(v[s][k], 6) for k in KEYS}) for n, v in gen.items() for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n, scoring='aligned', **{k: round(v[k], 6) for k in KEYS}) for n, v in g['ladder'].items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V12.csv', index=False)
    print('written RESULTS_V12.md, VALIDATION_V12.csv')


if __name__ == '__main__':
    main()
