"""Writes RESULTS_V15.md and VALIDATION_V15.csv (python -m src.v15.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v15'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STEP = {'mix': '요일 유형별 가중(스택 재조정)', 'events': '행사(콘서트·배구, 상승폭 없던 날)', 'school': '학교 방학 변화 × 등하교 시간', 'low': '5명 미만 예측 보정', 'holiday': '연휴·공휴일 달력 사전값'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    g = json.loads((OUT / 'gates_v15.json').read_text()); ge = json.loads((OUT / 'gates_v15e.json').read_text()); x = json.loads((OUT / 'experiments_v15.json').read_text())
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v15.json').read_text()); gen = g['generations']; order = ['v12', 'v13', 'v14', 'v15 base (stack + event-day rule)', 'v15']
    vals = lambda m: [f'{m[k]:.5f}' for k in KEYS]; bold = lambda n, c: f'**{c}**' if n == 'v15' else c
    sub = g['submission']; st = g['steps']; rm = g['retrained_models']; sm = rm['single_model_aligned']; er = g['event_day_rule']; sl = ge['summer_leave']; a14, a15 = gen['v14']['aligned'], gen['v15']['aligned']
    worse = g['folds_worse_than_v14']; basis = {(e['station'], e['date']): e for e in cfg['extras']}
    corr = pd.read_csv(OUT / 'external/event_corrections.csv'); split = pd.read_csv(OUT / 'external/exit_split_by_end_minute.csv', dtype=str)
    L = ['# ensemble_v15 결과 (RESULTS_V15)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 모델·가중치는 그 fold를 뺀 fold로만 정했고, 보정 단계는 그에 더해 그 fold와 날짜가 겹치는 행도 뺐다. '
         '보정 단계는 전체 점수가 좋아지고 어느 fold도 나빠지지 않을 때만 채택했다. 그래픽카드는 쓰지 않았다(이 컴퓨터 CPU).\n',
         '## 1. 점수\n',
         table(['정렬 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(gen[n]['aligned'])] for n in order]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[bold(n, n)] + [bold(n, c) for c in vals(gen[n]['raw_truth'])] for n in order]) + '\n',
         f"- v14 대비 전체 {a15['all_oof'] - a14['all_oof']:+.5f}, Fold 1 {a15['fold_1'] - a14['fold_1']:+.5f}, Fold 4 {a15['fold_4'] - a14['fold_4']:+.5f}. "
         + (f"Fold {', '.join(worse)}은 v14보다 {', '.join(f'{v:+.6f}' for v in worse.values())} 나쁘다. 이 차이는 LightGBM 시드 하나를 바꿀 때의 변동(단일 모델 Fold 2가 시드에 따라 {x['seed_spread_fold2']})보다 작다. v14는 시드 하나였고 v15는 시드 평균이다.\n" if worse else '어느 fold도 v14보다 나쁘지 않다.\n'),
         f"- 'v15 base'는 보정 단계를 거치기 전(스택 + 행사일 규칙)의 값이다. v14의 같은 단계는 {x['v14_base']}였다.\n",
         f"- 제출본 두 개:\n  - `outputs/ensemble_v15/{sub['file']}` — 검증을 통과한 것만. SHA-256 `{sub['sha256']}`, v14 대비 로그 RMSD {sub['log_rmsd_vs_v14']:.4f}.\n"
         f"  - `outputs/ensemble_v15/{ge['submission']['file']}` — 위 파일 + fold로 검증할 수 없는 보정(5절). SHA-256 `{ge['submission']['sha256']}`, v15와 다른 행 {ge['submission']['rows_changed_vs_v15']:,}개.\n",
         '## 2. 이번 판에서 바꾼 것\n',
         '### (1) 학습 자료 정정 (v15 패널)\n',
         '- 팀원 재조사(v4, 답변서)로 확인된 것 중 **학습 구간 실측으로도 확인되는 7건**만 행사 표시에 반영했다.\n',
         table(['조치', '역', '날짜', '종류', '이유', '학습 자료 근거'], [[{'remove': '표시 제거', 'add': '표시 추가'}[r.action], r.station_id, r.date, {'festival': '축제', 'sports': '경기·행사', 'concert': '콘서트'}[r.kind], r.reason, r.evidence.replace('train: ', '')] for r in corr.itertuples()]) + '\n',
         f"- 집계 이상·운행 중단 3개 역-일(예술회관 2026-03-01 전 시간, 3222 2026-01-14 19~23시, 3112 2025-08-13 11~14시; 27칸)은 학습 목표에서 뺐다. 검증 채점에서는 빼지 않았다.\n",
         f"- **v4의 새 행사 약 40건을 전부 '축제'로 넣는 것은 해로웠다**: 단일 모델 Fold 1이 {x['blanket_labels']['fold1_before']} → {x['blanket_labels']['fold1_after']}. 검암 국화축제 14일(밤 승차 평소 수준)처럼 흔적이 없는 날까지 밤 시간대를 2.5배로 올렸기 때문이다.\n",
         '### (2) 모델\n',
         table(['모델(단독, 정렬)', '시드', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'],
               [[n, s] + [f"{sm[k][z]:.5f}" for z in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] for n, k, s in (('v11 LightGBM 127잎 (lgb_b)', 'lgb_b', 1), ('**v15 LightGBM 127잎 (lgb_h)**', 'lgb_h', rm['seeds']['lgb_h']['folds']),
                                                                                                                         ('v14 LightGBM 31잎 (lgb_d)', 'lgb_d', 1), ('v11 XGBoost (xgb)', 'xgb', 1), ('v15 XGBoost 깊이 8 (xgb_h)', 'xgb_h', rm['seeds']['xgb_h']['folds']), ('**v15 XGBoost 깊이 10 (xgb_i)**', 'xgb_i', rm['seeds'].get('xgb_i', {}).get('folds', 0))) if k in sm]) + '\n',
         f"- v14의 lgb_c(시드 1개)는 0.17753이었다. 스택 테스트 가중치: " + ', '.join(f'{k} {v:.3f}' for k, v in rm['stage1_weights_test'].items() if v > 0.004) + '.\n',
         f"- 시험한 설정 변형(시드 1개, Fold 2/3/4, 3,000회 기준; 기준값 {x['lgb_variants']['baseline']}): " + '; '.join(f"{k} {v}" for k, v in x['lgb_variants']['variants'].items()) + '. 세 fold가 함께 좋아지는 설정이 없어 기존 설정을 유지했다.\n',
         f"- XGBoost 변형(기준 {x['xgb_variants']['baseline']}): " + '; '.join(f"{k} {v}" for k, v in x['xgb_variants']['variants'].items()) + f". {x['xgb_variants']['verdict']}\n",
         '### (3) 행사일 규칙\n',
         f"- v12~v14는 목록에 있는 행사 역-일 전부를 v10 값으로 고정했다. v10이 실제로 조사 기반 상승폭(어느 시간대든 평소보다 로그 {er['lift_threshold']} 이상)을 갖고 있는 날만 고정하고, 나머지는 스택 값을 쓴다. "
         f"전부 고정 {er['always_v10']['all_oof']:.5f} | 고정 안 함 {er['never']['all_oof']:.5f} | 상승폭 있는 날만 {er['only_where_v10_has_a_lift']['all_oof']:.5f}.\n",
         f"- 테스트: 행사 역-일 {er['listed_station_days_test']}개 중 {er['kept_on_v10_test']}개를 v10으로 유지. 푼 날: " + ', '.join(er['released_test'][:14]) + (' …' if len(er['released_test']) > 14 else '') + '\n',
         '## 3. 보정 단계 (중첩 + 날짜 차단)\n',
         table(['단계', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정'],
               [[STEP[n], f"{s['before']['all_oof']:.5f} → {s['after']['all_oof']:.5f}"] + [f"{s['after'][f'fold_{k}'] - s['before'][f'fold_{k}']:+.5f}" for k in (1, 2, 3, 4)] + ['채택' if s['adopted'] else '**기각**']
                for n, s in st.items()]) + '\n',
         '- 요일 유형별 가중은 따로 떼어 시험했을 때는 좋아 보였지만, 파이프라인 위에서는 Fold 3이 나빠진다. 형태를 바꿔 봐도 같았다: ' + '; '.join(f"{k} {v}" for k, v in x['mix_variants'].items()) + '.\n',
         '## 4. 새 자료 평가\n',
         '### (1) 박태동 자료(입주·광역버스·도로) — 효과 없음\n',
         table(['변수', '전체 OOF 변화', 'Fold별 변화', '판정'], [[k, v['all'], v['folds'], v['verdict']] for k, v in x['taedong'].items()]) + '\n',
         f"- {x['taedong_note']}\n",
         '### (2) 행사 자료(`SINCON_Data_Event`) — 쓸모 있음, v15e에 반영\n',
         '- 7~10월 행사 35행의 시작·종료 시각과 "종료가 :40분 전이면 그 시간, 이후면 다음 시간에 퇴장이 몰린다"는 규칙이 들어 있다. 규칙을 우리 학습 자료(문학 홈경기, 기준일 전)로 다시 계산했다.\n',
         table(['종료 분', '경기 수', '종료한 시간대 비중', '다음 시간대 비중'], [[r.end_minute_bucket, r.games, r.share_in_end_hour, r.share_in_next_hour] for r in split.itertuples()]) + '\n',
         f"- 현재 문학 경기 모듈은 이미 종료 시각을 쓰고 있어 고칠 것이 거의 없다(경기일 종료·다음 시간대의 편향을 다 없애도 전체 제곱오차의 {x['game_split_removable']}).\n",
         f"- PLAVE 종료 시각은 근거가 엇갈린다: {x['plave_evidence']}\n",
         '### (3) 카본 제안 검토\n',
         table(['제안', '확인한 것', '결론'], [[k, v['finding'], v['verdict']] for k, v in x['carbon'].items()]) + '\n',
         table(['멤버(단독, 정렬)', '전체 OOF'], [[k, v] for k, v in x['member_scores'].items()]) + '\n',
         '## 5. fold로 검증할 수 없는 보정 (v15e에만)\n',
         f"### (1) 여름 집단휴가 2주 — 평일 {sl['days'][0]} ~ {sl['days'][-1]}\n",
         table(['시간대', '06~09시', '10~15시', '16~19시', '20~23시'], [['2025년 실제 하락(역 중앙값)'] + [f'{v:+.3f}' for v in sl['median_fall_2025']], ['v15 예측의 하락'] + [f'{v:+.3f}' for v in sl['median_fall_v15']],
                                                                ['v15e에서 더한 이동(역 중앙값)'] + [f'{v:+.3f}' for v in sl['median_shift_by_hour_group']]]) + '\n',
         f"- 대상 {sl['stations']}개 역, 2주 평일 하루 합계 평균 {100 * sl['mean_day_total_change']:+.1f}%. 방식은 v14e와 같다.\n",
         '### (2) 행사별\n',
         table(['행사', '역', '날짜', '보정 전(시:명)', '보정 후', '시나리오 확률'], [[e['name'], e['station'], e['date'], e['before'], e['after'], e['probabilities'] or '배율'] for e in ge['entries']]) + '\n',
         '- 값은 시나리오별 승차 수의 **로그 평균**(확률 가중)이다. RMSLE에서는 이것이 기대 손실을 가장 줄인다. 확률은 근거의 강도에 따른 판단값이다.\n',
         '- **최광야구 10/17**: 23,000석 매진(10/19·11/5 보도). 퇴장 승차 총량은 학습 기간 홈경기 관계(종료 후 두 시간 승차 = 0.263 × 관중 + 204, 약 6,300명)로 잡았다. 종료 시각은 보도되지 않았다. '
         '작년 같은 행사(2025-10-24, 18:00 시작)는 20·21·22시에 391·5,279·1,140명이었고, 분배 곡선으로 보면 21:30 무렵 종료(약 3시간 30분)다. 같은 길이면 21:00, 3시간이면 20:30 종료라서 네 구간에 확률을 나눴다.\n',
         '- **PLAVE 9/12·13**: 19:00 시작. 종료 시각 근거가 엇갈려(4절) 다섯 구간으로 나눴다. 퇴장 승차 총량은 과거 주경기장 콘서트 9일(두 시간 합 8,100~11,300명)에서 8,500명으로 잡았고, 첫 스타디움 공연이라 관객 수가 보도되지 않아 기존 예측에 0.15의 가중을 남겼다.\n',
         '- **Sustainable Wave 8/29·30**: 이틀 약 4만 명(한국콘텐츠진흥원), 공식 13:10~21:10. 종료 시간대에 이 역으로 나오는 비율을 8%(종일 축제: 2025 펜타포트 하루 약 5만 명에 4,000~4,100명), 15%, 24%(이 역의 야구·콘서트)로 나누고, 21:40 이후로 늦어지는 경우를 0.2로 두었다. 일별 관객은 보도되지 않아 이틀에 같은 값을 넣었다.\n',
         '- **불꽃야구 10/31**: v10부터 "시작 + 190분"으로 들어 있어 퇴장 인파가 거의 전부 17시 칸에 있다. 과거 직관 다섯 경기 중 네 경기는 이 가정과 맞고 한 경기(2025-07-27)는 4시간 가까이 걸렸다. 그 경우에 대비해 일부를 18시 칸으로 옮겼다.\n',
         '- **이슬라이브 9/12**: 재학습 모델이 22:00에 끝나는 축제(맥주축제·펜타포트)의 모양대로 22시에 정점을 둔다. 공식 종료는 21:30(14:00 시작 + 450분)이라 규모는 그대로 두고 일부를 21시 칸으로 옮겼다.\n',
         '- **가재울 둘째 토요일**: 2026년 3~6월 둘째 토요일마다 12시 승차가 791·1,192·932·1,105명(같은 달 다른 토요일 208명), 11시가 225·406·373·528명(219명)이었고 3월 이전에는 없었다. 원인을 모르고 7~10월에 이어졌는지도 확인되지 않아 "열렸을 확률"(7·9월 0.6, 8월과 연휴인 10/10은 0.5)로 절충했다.\n',
         '- **능허대 문화축제 10/9·10(선학)**: 날짜·시간(12~21시)은 문체부 목록, 주 무대 위치는 출처 한 곳뿐이다. 같은 공원의 3월 축제는 20시에 925·1,017명(평소 219·132명)이었다. 20시 정점 0.30, 21시 정점 0.25, 흔적 없음 0.45.\n',
         '- **송도마라톤 9/20**: 2025년 대회 날 10·11시 713·746명(평소 약 230~250명). 모델이 이미 50%쯤 올려 놓아 1.4배만 더했다(v14e와 같은 방식).\n',
         '- 시나리오의 승차 수와 확률 전체는 `configs/ensemble_v15.json`에 있다.\n',
         '## 6. 시험했지만 버린 것 (이번 판)\n',
         table(['시도', '결과(로컬 검증)', '버린 이유'], [[k, v['result'], v['why']] for k, v in x['rejected'].items()]) + '\n',
         '## 7. 한계\n',
         '- Fold 3은 v14보다 0.0001 나쁘다(1절). 네 검증 지표(전체, 가중 4-Fold, Fold 3+4, Fold 4)는 모두 v14보다 좋다.\n',
         '- 5절 전부. 특히 PLAVE·최광야구·불꽃야구의 실제 종료 시각, Sustainable Wave의 역 이용 비율, 가재울 둘째 토요일의 원인과 지속 여부, 능허대 축제의 주 무대 위치는 확인되지 않았다.\n',
         f"- 모델 쪽 여지는 작다: 시드 3개 평균, XGBoost 추가, 표시 정정을 합쳐 스택 기준 {gen['v14']['aligned']['all_oof']:.5f} → {a15['all_oof']:.5f}. 설정 변형·스택 재조정·새 외부 변수는 채택 기준을 통과하지 못했다.\n",
         f"- 로컬 검증 현재 {a15['all_oof']:.4f}. 오차 구성은 그대로다: 칸 단위 잡음 65%(그 절반이 셀 수 잡음), 일 충격 11%, 시간대 모양 12.5%, 수준 11%.\n",
         '## 8. 남은 자료 중 값이 큰 것\n',
         '- **역 주변 버스 정류장의 시간대별 승하차(2025-07 ~ 2026-10)**: 운영진이 버스 이용량을 허용한다고 명시했다면 지금 남은 것 중 가장 크다. 하루 총량을 완벽히 알 때의 상한이 로컬 0.170 → 0.149였다. 아직 수집되지 않았다(답변서 11번: 버스 담당 팀원 몫).\n',
         '- **PLAVE 9/12·13, 최광야구 10/17, 불꽃야구 10/31의 실제 종료 시각**(공연·경기 후기, 구단 기록)과 **Sustainable Wave의 일별 관객**: 5절의 확률 절충을 실제 값으로 바꿀 수 있다. 종료가 몇 시 몇 분인지만 알면 된다.\n',
         '- **가재울 둘째 토요일의 정체**(3월부터 생긴 월례 행사·시험)와 7~10월 개최 여부.\n']
    (OUT / 'RESULTS_V15.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(gen[n][s][k], 6) for k in KEYS}) for n in order for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n + (' (adopted)' if s['adopted'] else ' (rejected)'), scoring='aligned', **{k: round(s['after'][k], 6) for k in KEYS}) for n, s in st.items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V15.csv', index=False)
    print('written RESULTS_V15.md, VALIDATION_V15.csv')


if __name__ == '__main__':
    main()
