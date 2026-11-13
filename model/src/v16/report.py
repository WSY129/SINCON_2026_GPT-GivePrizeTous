"""Writes RESULTS_V16.md and VALIDATION_V16.csv (python -m src.v16.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v16'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STEP = {'concert': '② 문학 주경기장 콘서트 (20~23시를 다른 콘서트일의 평균 로그 모양과 혼합)', 'lunar': '③ 부평삼거리 명절 기간 성묘 (당일·전후일·옆 주말 × 시간대 평균 잔차)',
        'growth': '① 신설 역 시간대별 성장 기울기', 'mart': '④ 대형마트 의무휴업 일요일 (동춘·독정, 11~18시)', 'night': '⑤ 송도달빛축제공원 비행사일 22시'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def main():
    g = json.loads((OUT / 'gates_v16.json').read_text()); ge = json.loads((OUT / 'gates_v16e.json').read_text()); x = json.loads((OUT / 'experiments_v16.json').read_text())
    gen = g['generations']; st = g['steps']; sub = g['submission']; vals = lambda m: [f'{m[k]:.5f}' for k in KEYS]; a15, a16 = gen['v15']['aligned'], gen['v16']['aligned']; c = st['concert']; lu = st['lunar']; ma = st['mart']
    L = ['# ensemble_v16 결과 (RESULTS_V16)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 다섯 가설을 v15 예측의 잔차 위에서 하나씩 시험했고, 각 단계는 그 fold를 뺀 fold에서 그 fold와 날짜가 겹치는 행까지 빼고 맞췄다. 전체 점수가 좋아지고 어느 fold도 나빠지지 않을 때만 채택했다.\n',
         '## 1. 점수\n',
         table(['정렬 채점'] + HEAD, [[n] + vals(gen[n]['aligned']) for n in ('v15', 'v16')] + [['차이'] + [f'{a16[k] - a15[k]:+.5f}' for k in KEYS]]) + '\n',
         table(['원래 정답 채점'] + HEAD, [[n] + vals(gen[n]['raw_truth']) for n in ('v15', 'v16')]) + '\n',
         f"- 네 fold 모두 v15보다 좋다.\n- `outputs/ensemble_v16/{sub['file']}` — 검증 통과분만. SHA-256 `{sub['sha256']}`, v15와 다른 행 {sub['rows_changed_vs_v15']:,}개.\n"
         f"- `outputs/ensemble_v16/{ge['submission']['file']}` — 위 파일 + v15e와 같은 검증 불가 보정. SHA-256 `{ge['submission']['sha256']}`, v16과 다른 행 {ge['submission']['rows_changed_vs_v16']:,}개.\n",
         '## 2. 다섯 가설의 판정 (순서대로 쌓음)\n',
         table(['가설', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정'],
               [[STEP[n], f"{s['before']['all_oof']:.5f} → {s['after']['all_oof']:.5f}"] + [f"{s['after'][f'fold_{k}'] - s['before'][f'fold_{k}']:+.5f}" for k in (1, 2, 3, 4)] + ['**채택**' if s['adopted'] else '기각'] for n, s in st.items()]) + '\n',
         '## 3. 가설별 내용\n',
         '### ② 문학 주경기장 콘서트 — 채택\n',
         f"- 원인: 콘서트일은 행사일 규칙으로 v10 값에 고정돼 있었는데 v10의 상승폭이 작았다(예: Fold 3 2026-04-05 22시 실측 7,598명, 예측 108명). 고칠 대상이 맞았다.\n"
         f"- 방법: 학습 구간의 주경기장 콘서트일 {len(c['concert_days_in_training_period'])}일 가운데 그 fold 기간 밖의 날들로 20~23시 평균 로그 승차를 구해 기존 예측과 섞는다(가중 = 일수 / (일수 + 2)). 종료 시각을 쓰지 않으므로 정점이 21시인 공연과 22시인 공연의 평균이 된다.\n"
         f"- 테스트: PLAVE 9/12 20~23시 {c['test']['2026-09-12']['before']} → {c['test']['2026-09-12']['after']}, 9/13 {c['test']['2026-09-13']['before']} → {c['test']['2026-09-13']['after']}. v16e에서는 이 값 위에 종료 시각 시나리오가 다시 얹힌다.\n",
         '### ③ 명절 전후 — 부평삼거리만 채택\n',
         f"- 부평삼거리(인천가족공원 성묘): 추석 2025와 설 2026 모두 큰 과소예측이었다(명절 당일 9~16시 실측 8,285·9,892명에 예측 2,145·4,001명). 추석 ↔ 설로 서로 맞춰도 좋아진다. 테스트 적용일: {', '.join(lu['test_days'])}. 더한 로그값: " + ', '.join(f'{k.replace("3122 부평삼거리 ", "")} {v:+.2f}' for k, v in list(lu['largest_test'].items())[:6]) + '.\n',
         '- 산단·통근역의 "주말과 겹친 명절 전후일 이중 차감"은 같은 방식(모든 역 × 날 종류 × 시간대)으로는 Fold 1~3이 모두 나빠져 기각했다. 추석과 설의 역별 잔차가 서로 옮겨지지 않는다. 주말과 겹친 명절 전후일은 검증 구간에 하루(2025-10-05)뿐이라 그 규칙만 따로 검증할 수 없다.\n',
         '### ④ 마트 의무휴업 — 동춘·독정만 채택\n',
         f"- 휴업 일요일과 영업 일요일의 차이(11~18시, 로그): 동춘 {ma['dongchun']:+.3f}, 독정 {ma['dokjeong']:+.3f}. 팀원 자료의 −20%, −12%와 맞는다. 네 fold가 같은 폭으로 좋아지지만 크기는 작다(−0.00004).\n"
         '- 모든 역에 적용하면 부천종합운동장·문학경기장처럼 일요일 경기와 섞인 역이 끼어들어 Fold 1이 나빠진다.\n',
         '### ① 검단호수공원 평일 아침 — 기각\n',
         '- 편향은 실제로 있다: 평일 06~09시 잔차가 Fold 1 +0.45, Fold 3 +0.20, Fold 4 +0.53. 그런데 Fold 2는 −0.02라서 다른 fold에서 배운 값을 옮기면 Fold 2가 나빠진다(+0.00136). 기준일 직전 8주의 시간대별 기울기 차이로 예측하는 방식은 Fold 4를 악화시켰다(+0.00214).\n',
         '### ⑤ 송도달빛축제공원 22시 — 기각(기준에 근소하게 미달)\n',
         '- 과대예측은 네 fold·세 요일 유형 모두에서 같은 방향이다(잔차 −0.06 ~ −0.58). 22시만 보정하면 Fold 1·3·4는 좋아지지만 Fold 2가 +0.0000115로 허용 폭(0.00001)을 넘는다. 기준을 그대로 지켜 넣지 않았다. 넣었다면 테스트 22시를 평일 −0.14, 토 −0.19, 일 −0.22(로그) 낮추게 된다.\n',
         '## 4. 시험한 형태 전체 (각각 v15 위에서 단독)\n',
         table(['형태', '결과'], [[k, v] for k, v in x['variants'].items()]) + '\n',
         '## 5. v16e\n',
         '- 추가 보정 항목과 확률은 v15e와 같다(`configs/ensemble_v16.json`). 카본이 알려 준 PLAVE·최광야구·능허대 종료 시각은 출처가 오지 않아 반영하지 않았다.\n',
         table(['행사', '역', '날짜', '보정 전(시:명)', '보정 후'], [[e['name'][:34], e['station'], e['date'], e['before'], e['after']] for e in ge['entries']]) + '\n',
         '## 6. 한계\n',
         '- 콘서트 단계는 종료 시각을 쓰지 않는다. 시작 시각별로 나누기에는 학습 구간 콘서트가 9일뿐이다.\n',
         '- 부평삼거리 보정은 추석 1회와 설 1회에서 배운 값이다. 2026년 추석은 목~토라 요일 배치가 다르다.\n',
         f"- 로컬 검증 현재 {a16['all_oof']:.4f}.\n"]
    (OUT / 'RESULTS_V16.md').write_text('\n'.join(L) + '\n')
    rows = [dict(item=n, scoring=s, **{k: round(gen[n][s][k], 6) for k in KEYS}) for n in ('v15', 'v16') for s in ('aligned', 'raw_truth')]
    rows += [dict(item='step: ' + n + (' (adopted)' if s['adopted'] else ' (rejected)'), scoring='aligned', **{k: round(s['after'][k], 6) for k in KEYS}) for n, s in st.items()]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V16.csv', index=False); print('written RESULTS_V16.md, VALIDATION_V16.csv')


if __name__ == '__main__':
    main()
