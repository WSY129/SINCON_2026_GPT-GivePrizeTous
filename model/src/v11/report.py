"""Writes RESULTS_V11.md from gates_v11.json (python -m src.v11.report). Local-validation numbers only."""
import json
import pandas as pd
from src.v8 import core as c8
from .panel import FEATURES

OUT = c8.ROOT / 'outputs/ensemble_v11'
KEYS = ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')
HEAD = ['전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
NAME = {'lgb_a': 'LightGBM 직접(31잎)', 'lgb_b': 'LightGBM 직접(127잎)', 'xgb': 'XGBoost CUDA 직접', 'cb': 'CatBoost GPU 직접(대칭)', 'dec_lgb': '분해형 LightGBM(일 총량×시간 비중)',
        'dec_xgb': '분해형 XGBoost', 'dec_cb': '분해형 CatBoost', 'low_tw': '저수요 트위디', 'low_po': '저수요 포아송', 'tabm': '다중과제 TabM(승차+하차)', 'stg': '다중과제 시공간 그래프'}


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def vals(m):
    return [f'{m[k]:.5f}' for k in KEYS]


def main():
    g = json.loads((OUT / 'gates_v11.json').read_text()); cfg = json.loads((c8.ROOT / 'configs/ensemble_v11.json').read_text())
    meta = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    sub = g.get('submission', {}); st = g['stack']; t = g.get('test', {})
    L = ['# ensemble_v11 결과 (RESULTS_V11)\n',
         '> 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수는 쓰지 않았다. 가중치·게이트·전용 모델 교체는 모두 중첩(그 fold를 뺀 fold로만 결정)이고 정렬 정답으로 적합했다. '
         '정렬 채점 = 기준일이 2025년인 fold의 2026년 정답을 2025년 집계 기준으로 환산한 채점, 원래 정답 채점 = 환산 없이 채점.\n',
         '## 1. 기존 최고 판본(v10) 대비\n',
         table(['정렬 채점'] + HEAD, [['v10'] + vals(g['v10']['aligned']), ['v11 스태킹'] + vals(st['aligned']), ['**v11 최종(전용 모델 포함)**'] + vals(g['final']['aligned'])]) + '\n',
         table(['원래 정답 채점'] + HEAD, [['v10'] + vals(g['v10']['raw_truth']), ['v11 스태킹'] + vals(st['raw_truth']), ['**v11 최종(전용 모델 포함)**'] + vals(g['final']['raw_truth'])]) + '\n']
    if sub:
        L.append(f"제출본: `outputs/ensemble_v11/{sub['file']}` — 157,194행, SHA-256 `{sub['sha256']}`, 평균 승차 {sub['mean_boardings']:.2f}명, v10 대비 로그 RMSD {sub['log_rmsd_vs_v10']:.4f}, 평균 로그 차이 {sub['mean_log_diff_vs_v10']:+.4f}.\n")
    L.append('## 2. 새 모델군의 단독 성능\n')
    L.append('고유 일별 패널(역·날짜·시간당 한 행, 가상 기준일 복제 없음, 특징 ' + str(len(FEATURES)) + '개)에서 학습했다. 조기 종료 없이 3,000회까지 학습하며 8개 지점(100~3,000회)의 예측을 모두 저장했고, '
             '각 fold에 쓴 지점은 나머지 fold의 검증 곡선 최저점 평균으로 정했다.\n')
    rows = []
    for m, v in sorted(g['singles'].items(), key=lambda x: x[1]['aligned']['all_oof']):
        if m == 'v10':
            rows.append(['v10 (기존 최고)', '—'] + [f"{v['aligned'][k]:.5f}" for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + [f"{v['raw_truth']['all_oof']:.5f}"]); continue
        best = {k: min(c) for k, c in v['valid_curves'].items()}
        rows.append([NAME.get(m, m), str(list(v['checkpoint'].values()))] + [f"{v['aligned'][k]:.5f}" for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + [f"{v['raw_truth']['all_oof']:.5f}"])
    L.append(table(['모델', '사용 지점(fold 1~4)', '정렬 전체', 'F1', 'F2', 'F3', 'F4', '원래 정답 전체'], rows) + '\n')
    L.append('## 3. 스태킹\n')
    L.append(f"- 리지 강도 {st['ridge']}, 가중치 결정 방식 `{st['protocol']}` (standard = 나머지 3개 fold, matched = Fold 3·4 교차). 시도한 조합:\n")
    L.append(table(['조합', '정렬 전체', 'F1', 'F2', 'F3', 'F4', 'v10보다 나쁜 fold 없음'],
                   [[k] + [f"{v['aligned'][x]:.5f}" for x in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + ['예' if v['no_fold_worse_than_v10'] else '아니오'] for k, v in g['stack_trials'].items()]) + '\n')
    if t:
        L.append('테스트에 쓴 가중치: ' + ', '.join(f"{NAME.get(k, k)} {v:.3f}" for k, v in sorted(t['weights'].items(), key=lambda x: -x[1]) if v >= 0.005) + '\n')
        L.append(f"- 역별 게이트로 v10을 유지한 역: {[f'{s} {meta[s]}' for s in t['stations_kept_on_v10']] or '없음'}\n- 행사 역-날짜(축제·콘서트 등, 문학·부천 제외)는 v10 값을 유지한다.\n")
        L.append(f"- 개통 400일 미만 역 {t.get('young_stations_kept_on_v10', [])}은 v10 값을 유지한다. 이 역들은 테스트 달(7~10월)의 과거 관측이 개통 직후뿐이라, 연중 일자 항을 가진 모델이 성장을 여름 계절성으로 읽는다. "
                 '새 LightGBM은 검단호수공원의 2026년 7월 평일을 6월 실제보다 로그 0.16 낮게 봤다. fold에서는 이 겹침이 생기지 않아 게이트로 잡을 수 없으므로 규칙으로 막았다(fold 점수에도 같은 규칙을 적용했고, 적용 전 전체 점수는 정렬 0.16988이었다).\n'
                 '- 문학경기장은 새 모델이 관중·종료 시각·점수 차를 직접 입력으로 받는다. 콘서트·직관일 가운데 v10이 그 기준일에 이미 알고 있던 날만 v10의 명시적 상승폭을 유지한다.\n')
    L.append('## 4. 고오차 역 전후 (정렬 채점, 4개 fold 합산)\n')
    L.append(table(['역', '이름', 'v10', 'v11 스태킹', 'v11 최종', '변화'], [[k, meta[int(k)], f"{v['v10']:.4f}", f"{v['stack']:.4f}", f"{v['final']:.4f}", f"{v['final'] - v['v10']:+.4f}"] for k, v in g['stations'].items()]) + '\n')
    L.append('전용 모델(그 역의 행만으로 학습한 LightGBM 3시드 평균)의 판정:\n')
    L.append(table(['역', '묶음', 'v10', '스태킹', '전용 모델', '최종', 'fold별 사용(1~4)', '테스트 사용'],
                   [[k, v['group'], f"{v['rmsle']['v10']:.4f}", f"{v['rmsle']['stack']:.4f}", f"{v['rmsle']['specialist']:.4f}", f"{v['rmsle']['final']:.4f}",
                     ''.join('O' if x else '-' for x in v['used_nested'].values()), '예' if v['used_for_test'] else '아니오'] for k, v in g['specialists'].items()]) + '\n')
    L.append('## 5. 자료와 정제\n')
    L.append('- 기상: 전 기간 시간별 강수·기온·체감온도·일사량·일조시간·운량·풍속(Open-Meteo). 기온은 30도 초과분, 33도 이상 폭염, 15~26도 맑은 날(쾌적), 평년 대비 편차로 나눠 넣었다.\n'
             '- 야구: KBO 스코어보드에서 2025~2026년 문학 160경기의 이닝별 점수를 받아 최종 점수 차와 7회 종료 점수 차를 넣었다. 우천취소일은 예정 시작 시각 전후 3시간 강수량을 전용 모델 입력으로 넣었다(취소 시각은 자료에 없다).\n'
             '- 정제: 팀 급증일 파일의 역-날짜(원인 미상 236건 포함)와 등록된 행사·경기일 558개 역-날짜를 수준과 전년 커널 계산에서 뺐다. 송도달빛축제공원 8월의 다년 곡선은 7·9월 값으로 대체했다.\n'
             '- 앵커: 기준일 수준 + 다년 일별 곡선(월 지수의 주기 스플라인) + 공휴일 효과 + 신규 역 성장. 전년 커널은 같은 요일 ±2주 삼각 가중이다.\n')
    for note in cfg.get('research_notes', []):
        L.append(f'- {note}')
    if t.get('new_events'):
        L.append('\n새로 반영한 행사(과거 회차 상승폭 × 축소 계수, 검증 불가):\n')
        L.append(table(['행사', '역', '날짜', '과거 날짜', '일 승차 전', '후', '평균 로그 변화'], [[e['event'], e['station'], e['date'], e['past'], e['daily_before'], e['daily_after'], e['mean_log_change']] for e in t['new_events']]) + '\n')
    L.append('\n## 6. 하지 않은 것\n')
    L.append('- 우천취소 시각·사유는 스코어보드에 없어 수집하지 못했다(사유는 팀 자료의 우천/폭염 구분만 사용).\n- 인하대 일정, 대학 중간고사 기간은 공식 일정에서 확인되지 않아 넣지 않았다.\n'
             '- 검단 입주 세대 수는 모델 입력으로 넣지 않았다(월 단위 5~6개 시점뿐이라 학습 표본이 안 된다). 신규 역은 성장 기울기와 전용 모델로 다뤘다.\n'
             '- 시민의 날 대축제(10월 10일 인천대공원)는 이 역의 과거 표본이 없어 반영하지 않았다.\n')
    (OUT / 'RESULTS_V11.md').write_text('\n'.join(L) + '\n')
    print('written RESULTS_V11.md')


if __name__ == '__main__':
    main()
