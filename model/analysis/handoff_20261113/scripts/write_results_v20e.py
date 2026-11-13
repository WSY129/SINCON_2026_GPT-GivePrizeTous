"""Writes outputs/ensemble_v20/RESULTS_V20E.md from gates_v20e.json and the three submission files (no leaderboard value is used)."""
import json, hashlib
import numpy as np, pandas as pd
from pathlib import Path
R = Path('.'); g = json.loads((R / 'outputs/ensemble_v20/gates_v20e.json').read_text())
t = pd.read_csv(R / 'data/raw/test.csv')
S = {n: pd.read_csv(p).boardings.to_numpy() for n, p in [('v19', 'outputs/ensemble_v19/submission_ensemble_v19.csv'), ('v19e', 'outputs/ensemble_v19/submission_ensemble_v19e.csv'), ('v20e', 'outputs/ensemble_v20/submission_ensemble_v20e.csv')]}
sha = {n: hashlib.sha256((R / p).read_bytes()).hexdigest() for n, p in [('v19', 'outputs/ensemble_v19/submission_ensemble_v19.csv'), ('v19e', 'outputs/ensemble_v19/submission_ensemble_v19e.csv'), ('v20e', 'outputs/ensemble_v20/submission_ensemble_v20e.csv')]}
key = {(d, s, h): i for i, (d, s, h) in enumerate(zip(t.date, t.station_id, t.hour))}
L = []; w = L.append
w('# ensemble_v20e 결과 (RESULTS_V20E)\n')
w('> 점수 표기: **로컬 검증(OOF)** = 내부 4-fold 검증기, **Public(사용자 제공)** = 사용자가 알려 준 리더보드 값(어떤 계산·선택에도 쓰지 않음), **추정** = 근거 있는 추정.\n')
w('## 1. 무엇이 다른가\n')
w('v20e = v19(검증 통과분) + 평가 기간 행사·휴가 항목(검증 불가). v19e와의 차이는 **행사 항목 크기를 교통카드 통계 사이트의 버스 승차 실측으로 다시 잡은 것뿐**이다(26칸, 8개 역-날짜).\n')
w('| 파일 | 로컬 검증(OOF) 전체 / 가중 4-Fold / Fold 3+4 / Fold 4 | Public(사용자 제공) | SHA-256 |')
w('|---|---|---|---|')
w(f'| v19 | 0.16541 / 0.16333 / 0.16017 / 0.15891 | 제출 안 함 | `{sha["v19"]}` |')
w(f'| v19e | 기반 v19와 같음 (추가 항목은 검증 불가) | 0.147859 | `{sha["v19e"]}` |')
w(f'| v20e | 기반 v19와 같음 (추가 항목은 검증 불가) | 0.146860 | `{sha["v20e"]}` |\n')
d = np.log1p(S['v20e']) - np.log1p(S['v19e']); m = np.abs(d) > 1e-6
w(f'- v19 대비 v20e 변경 행 {g["submission"]["rows_changed_vs_v19"]:,}개(여름 집단휴가 보정 포함), v19e 대비 {int(m.sum())}개.')
w('- 근거: 학습 기간 8개 경기일에 문학경기장(야구장) 정류장(37144) 버스 승차 초과분 대비 문학경기장역 지하철 승차 = 8.3~12.9배(중앙 11.9배). `external/bus_to_subway_calibration.csv`.\n')
w('## 2. 행사 항목 (예측 승차 인원, v19 → v19e → v20e)\n')
for e in g['entries']:
    hrs = sorted(int(h) for h in e['after'])
    w(f'### {e["name"]}  — 역 {e["station"]}, {e["date"]}')
    w('| 시 | v19 | v19e | v20e | 항목 시나리오(이전→이후) |'); w('|---|---|---|---|---|')
    for h in hrs:
        i = key[(e['date'], e['station'], h)]
        w(f'| {h} | {S["v19"][i]:.0f} | {S["v19e"][i]:.0f} | {S["v20e"][i]:.0f} | {e["before"].get(str(h), "")}→{e["after"][str(h)]} |')
    if e.get('probabilities'):
        w(f'\n시나리오 확률 {e["probabilities"]} (로그 공간 확률가중 평균).')
    w('')
sl = g['summer_leave']
w('## 3. 여름 집단휴가 보정\n')
w(f'- 대상일 {len(sl["days"])}일({sl["days"][0]}~{sl["days"][-1]}), 역 {sl["stations"]}개, 하루 합계 평균 변화 {sl["mean_day_total_change"]:+.4f}(로그).')
w('- 2025년 같은 기간 실제 감소폭(학습 자료)과 v19가 이미 예측한 감소폭의 차이 중 절반만 반영.\n')
w('## 4. 한계\n')
w('- 행사 항목은 평가 기간 값이라 로컬 검증기로 측정할 수 없다. Public 차이(0.147859 → 0.146860, 사용자 제공)는 방향 확인일 뿐, 항목 크기 조정에 쓰지 않았다.')
w('- 버스 자료는 사이트 화면 조회값을 옮긴 것이며 원본은 `outputs/ensemble_v20/external/`에 있다.')
(R / 'outputs/ensemble_v20/RESULTS_V20E.md').write_text('\n'.join(L) + '\n'); print('written', len(L))
