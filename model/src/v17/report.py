"""Writes RESULTS_V17.md and VALIDATION_V17.csv (python -m src.v17.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v17'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STEP = {'concert': '문학 주경기장 콘서트 (v16과 같음)', 'lunar': '부평삼거리 명절 성묘 — 목표값 혼합으로 교체', 'mart': '마트 의무휴업 일요일, 동춘·독정 (v16과 같음)', 'night': '송도달빛축제공원 비행사일 22시 — 절반 강도'}
OFF = {-3: '3일 전', -2: '2일 전', -1: '전날', 0: '당일', 1: '다음 날', 2: '2일 뒤', 3: '3일 뒤'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    g = json.loads((OUT / 'gates_v17.json').read_text()); ge = json.loads((OUT / 'gates_v17e.json').read_text()); x = json.loads((OUT / 'experiments_v17.json').read_text()); cfg = json.loads((c8.ROOT / 'configs/ensemble_v17.json').read_text())
    gen = g['generations']; st = g['steps']; sub = g['submission']; vals = lambda m: [f'{m[k]:.5f}' for k in KEYS]; lt = st['lunar']['test']; order = ['v15', 'v16', 'v17']
    hours = list(range(9, 19))
    L = ['# ensemble_v17 결과 (RESULTS_V17)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 각 단계는 그 fold를 뺀 fold에서 날짜가 겹치는 행까지 빼고 맞췄고, 전체 점수가 좋아지고 어느 fold도 나빠지지 않을 때만 채택했다.\n',
         '## 1. 검증 비교표\n',
         table(['정렬 채점'] + HEAD, [[n if n != 'v17' else '**v17**'] + vals(gen[n]['aligned']) for n in order]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[n] + vals(gen[n]['raw_truth']) for n in order]) + '\n',
         table(['단계 (v15 위에 순서대로)', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정'],
               [[STEP[n], f"{s['before']['all_oof']:.5f} → {s['after']['all_oof']:.5f}"] + [f"{s['fold_changes'][k]:+.6f}" for k in ('1', '2', '3', '4')] + ['채택' if s['adopted'] else '기각'] for n, s in st.items()]) + '\n',
         f"- `outputs/ensemble_v17/{sub['file']}` — 검증 통과분만. SHA-256 `{sub['sha256']}`, v15와 다른 행 {sub['rows_changed_vs_v15']:,}개.\n"
         f"- `outputs/ensemble_v17/{ge['submission']['file']}` — 위 파일 + 검증 불가 보정(4절). SHA-256 `{ge['submission']['sha256']}`, v17과 다른 행 {ge['submission']['rows_changed_vs_v17']:,}개.\n",
         '## 2. 부평삼거리 추석 성묘 보정: 이중 가산 수정\n',
         '- 지적이 맞았다. v16은 fold 잔차의 평균을 그대로 더했는데, fold에서는 그날 예측이 매우 낮았고(추석 당일 2,145명) 최종 모델은 두 명절을 모두 학습해 이미 높다(6,191명). 같은 값을 더해 추석 당일 9~16시가 12,619명이 됐다(실측 8,285·9,968명).\n',
         '- v17: **목표값 = 예측의 평소 일요일 수준 + 다른 명절 같은 날의 초과 승차(명, 대상 기간의 집계 기준으로 환산)**, 예측과 목표를 로그에서 가중 n/(n+1)로 섞는다(n = 참조한 명절 수). 참조 명절은 설 2025, 추석 2025, 설 2026 중 그 fold 기간 밖의 것. 9~18시에만 적용한다.\n',
         '- 배율이 아니라 초과 인원으로 잡은 이유: 성묘객 수는 역의 평소 승차와 무관한 인원이다. 평소 대비 배율(추석 2025 당일 7.5배, 설 2026 당일 8.8배)을 2026년 9월의 더 높은 평소 수준에 곱하면 목표가 부풀려진다.\n',
         table(['날짜', '위치', '참조 명절 수', '가중', '9~18시 합: v15', 'v16', '목표', '**v17**'],
               [[d, OFF[v['offset']], v['analog_days'], v['weight'], sum(v['before']), x['v16_lunar_sums'].get(d, '-'), sum(v['target']), f"**{sum(v['after'])}**"] for d, v in lt.items()]) + '\n',
         table(['시간'] + [f'{h}시' for h in hours], [r for d, v in lt.items() for r in ([f'{d[5:]} v15'] + v['before'], [f'{d[5:]} **v17**'] + v['after'])]) + '\n',
         '- 추석 다음 날(9/26)은 v15가 이미 과거 다음 날 실측(2,563·2,735명)보다 높아 오히려 내려간다.\n',
         '## 3. 모델 보강 요청에 대한 검증\n',
         table(['요청', '확인한 것', '결론'], [[k, v['finding'], v['verdict']] for k, v in x['models'].items()]) + '\n',
         '## 4. v17e: 출처가 확인된 행사 반영\n',
         table(['행사', '확인', 'v16e (시:명)', 'v17e (시:명)'], [[k, v['check'], v['v16e'], v['v17e']] for k, v in x['events'].items()]) + '\n',
         '- 서울세계불꽃축제(9/5)는 넣지 않았다: ' + cfg['not_applied'][0]['reason'].replace('On the day of the 2025 edition (2025-09-27, Saturday) the night hours 20-23 at 계양 3110, 검암 3207 and 부평 3120 were 0.9-1.6 times an ordinary Saturday with no return wave at 22-23h at 계양 (0.93 at 22h); 부평 was raised through the whole afternoon and the median station was 1.04. One day, mixed with other events of that Saturday: not applied.',
                                                                                 '작년 개최일(2025-09-27 토) 계양·검암·부평의 20~23시 승차는 평소 토요일의 0.9~1.6배였지만, 계양 22시는 0.93배로 귀가 인파 모양이 아니다. 부평은 오후 내내 높았는데 그날은 부평풍물대축제였다. 하루뿐이고 다른 행사와 섞여 있다.') + '\n',
         '- PLAVE 토요일은 21:35 종료라 인파가 21시 칸에 몰리는 경계(:40분) 바로 앞이다. 시작이 5분만 늦어도 22시 칸으로 넘어가므로 두 경우에 0.55 / 0.40을 주었다.\n',
         '- 최광야구는 경기 종료(20:23)는 확인됐지만 시상식·불꽃놀이가 끝난 시각은 문서에 없다. 경기 직후 떠난 사람의 비율과 불꽃놀이 종료 시각을 세 경우로 나눴다.\n',
         '- 나머지 항목(Sustainable Wave, 불꽃야구 10/31, 이슬라이브, 가재울, 송도마라톤, 여름휴가)은 v16e와 같다.\n',
         '## 5. 한계\n',
         '- 성묘 보정의 참조는 명절 세 번이다. 2026년 추석은 목~토라 요일 배치가 다르다.\n',
         '- 22시 보정은 강도를 절반으로 낮춰 통과시킨 것이다(Fold 2 변화 −0.000013). 효과는 −0.00004로 작다.\n',
         f"- 로컬 검증 현재 {gen['v17']['aligned']['all_oof']:.4f}.\n"]
    (OUT / 'RESULTS_V17.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(gen[n][s][k], 6) for k in KEYS}) for n in order for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n + (' (adopted)' if s['adopted'] else ' (rejected)'), scoring='aligned', **{k: round(s['after'][k], 6) for k in KEYS}) for n, s in st.items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V17.csv', index=False); print('written RESULTS_V17.md, VALIDATION_V17.csv')


if __name__ == '__main__':
    main()
