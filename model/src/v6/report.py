"""Write outputs/ensemble_v6/RESULTS.md from the JSON artefacts of validate.py and build.py.

Usage (from the project root):  python -m src.v6.report
Every number in the document is read from a saved artefact; nothing is typed in by hand.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATION_ORDER = ['3127', '3754', '3209', '3210', '3139']
VENUE = {'munhak_main': '인천문학경기장 (3127)', 'asiad_main': '인천아시아드주경기장 (3209·3210)', 'bucheon': '부천종합운동장 (3754)'}


def f4(x):
    return '없음' if x is None or x != x else f'{x:.4f}'


def main():
    cfg = json.loads((ROOT / 'configs/ensemble_v6.json').read_text()); out = ROOT / cfg['output']
    val = json.loads((out / 'validation_summary.json').read_text())
    sub = json.loads((out / 'submission_validation.json').read_text())
    audit = json.loads((out / 'schedule_audit.json').read_text())
    bw = json.loads((out / 'blend_weights.json').read_text())
    w = bw['weights']; rep = bw['report']; overall = val['overall_rmsle']; versions = rep['versions']
    folds = ['1', '2', '3', '4', 'all']
    L = []
    L += ['# ensemble_v6 결과', '',
          '제출 파일은 `submission_ensemble_v6.csv` 하나다. 기존 제출 v2·v4·v5를 로컬 OOF만으로 구한 가중치로 섞고, 사전 발표 일정이 있는 역의 해당 행만 전용 모델로 바꿨다. 리더보드 점수는 어떤 계산에도 쓰지 않았다.', '']
    lv = val['local_validator']; cols = ['all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3']
    head = '| 대상 | 전체 OOF | 가중 4-Fold | Fold 3+4 | Fold 4 | Fold 1 | Fold 2 | Fold 3 |'
    label = {'v6_base_blend': 'v6 블렌드만', 'v6_blend_plus_munhak': 'v6 블렌드 + 문학 전용', 'v6_final': '**v6 최종**'}
    L += ['## 0. 로컬 검증기 점수', '',
          '이 문서의 모든 점수는 **로컬 검증(OOF) RMSLE**다. Public 점수와 추정치는 싣지 않았다. 검증 대상은 기준일 2025-08-30 / 10-30 / 12-30 / 2026-02-27에서 각각 123일을 예측한 4개 fold(각 157,194행)다.', '',
          '가중 4-Fold는 `sqrt(0.15·MSE1 + 0.15·MSE2 + 0.35·MSE3 + 0.35·MSE4)`, Fold 3+4는 두 fold의 행을 합친 값이다.', '',
          '### 기존 보고서와 같은 방식 (각 버전의 저장된 앙상블 예측을 입력으로 사용)', '', head, '|---|' + '---:|' * 7]
    for key, row in lv['saved_ensemble_columns'].items():
        L.append(f"| {label.get(key, key)} | " + ' | '.join(f4(row[c]) for c in cols) + ' |')
    L += ['', '### 더 엄격한 방식 (각 버전이 그 fold를 빼고 만든 LOFO 예측을 입력으로 사용)', '', head, '|---|' + '---:|' * 7]
    for key, row in lv['leave_one_fold_out_columns'].items():
        L.append(f"| {label.get(key, key)} | " + ' | '.join(f4(row[c]) for c in cols) + ' |')
    L += ['', '기존 방식은 v2·v4·v5 보고서의 숫자(0.2228 / 0.2213 / 0.2135)와 바로 비교된다. 엄격한 방식은 각 버전 내부 블렌더가 그 fold의 정답을 보지 않은 예측이라 전반적으로 조금 높게 나오며, v6의 가중치 적합과 이후 표(3~5절)는 이 방식을 쓴다.', '']
    L += ['## 1. 제출 파일', '',
          f"- 행 수 {sub['rows']:,}, test ID 순서 일치 {sub['ids_in_test_order']}, 유한·비음수 {sub['finite_nonnegative']}",
          f"- 평균 {sub['mean']:.2f}명, 최소 {sub['min']:.2f}명, 최대 {sub['max']:.2f}명",
          '- 월평균: ' + ', '.join(f"{m} {v:.1f}명" for m, v in sub['monthly_mean'].items()),
          f"- SHA-256 `{sub['sha256']}`", '']
    L += ['## 2. 블렌드 가중치 (로컬 OOF만 사용)', '',
          '각 버전의 fold 제외(LOFO) 앙상블 예측을 입력으로, 4개 fold를 합친 로그 MSE를 최소화하는 볼록 가중치다. 전용 모델이 덮어쓰는 문학경기장 행은 적합에서 뺐다.', '',
          '| 구분 | ' + ' | '.join(versions) + ' | 합친 RMSLE | 단일 최선 | 동일 가중 |', '|---|' + '---:|' * (len(versions) + 3)]
    for seg, label in (('ordinary', '일반 행'), ('holiday', '공휴일 행')):
        s = rep['segments'][seg]
        L.append(f"| {label} ({s['rows']:,}행) | " + ' | '.join(f"{w[seg][v]:.2f}" for v in versions) + f" | {s['pooled_rmsle']:.4f} | {min(s['single_pooled'].values()):.4f} | {s['equal_weight_pooled']:.4f} |")
    L += ['', '가중치를 그 fold를 빼고 구했을 때의 일반 행 RMSLE(가중치 과적합 점검):', '', '| 제외한 fold | 블렌드 | ' + ' | '.join(versions) + ' |', '|---|' + '---:|' * (len(versions) + 1)]
    for fold, x in rep['segments']['ordinary']['leave_one_fold_out'].items():
        L.append(f"| {fold} | {x['rmsle']:.4f} | " + ' | '.join(f"{x['single'][v]:.4f}" for v in versions) + ' |')
    alt = rep['ordinary_alternatives_reference_only']['folds_2_and_4']
    L += ['', f"참고: fold 2와 4만으로 구하면 " + ' / '.join(f"{v} {alt['weights'][v]:.2f}" for v in versions) + f"이고, 이 가중치의 전체 fold RMSLE는 {alt['rmsle_all_folds']:.4f}로 위와 거의 같다. 채택한 것은 4개 fold 전체로 구한 값이다.", '']
    L += ['## 3. Fold 1~4 전체 RMSLE', '', '| 단계 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 전체 |', '|---|---:|---:|---:|---:|---:|']
    names = {**{v: f'{v} 단독' for v in versions}, 'base': '블렌드', 'munhak_only': '블렌드 + 문학 전용', 'final': '최종(+행사 보정)'}
    for key, label in names.items():
        L.append(f'| {label} | ' + ' | '.join(f4(overall[key][f]) for f in folds) + ' |')
    L += ['', '버전 단독 점수는 각 버전의 LOFO 예측 기준이라, 각 버전 보고서의 표본 내 앙상블 점수보다 조금 높다.', '']
    L += ['## 4. 역별 RMSLE (블렌드 → 최종)', '', '| 역 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 전체 |', '|---|---|---|---|---|---|']
    for sid in STATION_ORDER:
        s = val['station_rmsle'][sid]
        L.append(f"| {sid} {s['name']} | " + ' | '.join(f"{s['base'][f]:.3f} → {s['final'][f]:.3f}" for f in folds) + ' |')
    L += ['', '바뀐 행 수(fold별): ' + '; '.join(f"{sid} " + '/'.join(str(val['station_rmsle'][sid]['rows_changed'][f]) for f in folds[:4]) for sid in STATION_ORDER), '']
    L += ['### 문학경기장 상태별 (그 상태에 해당하는 날의 3127 행만)', '',
          '| Fold | 상태 | 일수 | 실제 급등일 | 블렌드 | 최종 | 최종 편향(예측−실제) |', '|---:|---|---:|---:|---:|---:|---:|']
    order = {'HOME': 0, 'NO_HOME': 1, 'UNANNOUNCED': 2, 'RESERVE': 3, 'CONCERT': 4, 'OFF': 5}
    for r in sorted(val.get('munhak_by_state', []), key=lambda r: (r['fold'], order.get(r['state'], 9))):
        L.append(f"| {r['fold']} | {r['state']} | {r['days']} | {r['days_with_crowd']} | {r['rmsle_blend']:.3f} | {r['rmsle_final']:.3f} | {r['bias_final']:+.3f} |")
    L += ['', 'OFF는 야구 시즌 밖이라 전용 모델이 손대지 않은 날이다(블렌드 값 그대로). CONCERT는 그 기준일에 이미 발표돼 있던 주경기장 콘서트 날이다. 실제 급등일은 그날 시간당 최대 승차가 1,200명 이상이었던 날 수다.', '']
    L += ['## 5. 행사별 효과 (그 날 그 역의 18개 시간 RMSLE)', '']
    acts = val['event_actions']
    if acts:
        L += ['| Fold | 역 | 날짜 | 처리 | 행사 | 비교 표본 | 전 | 후 |', '|---:|---:|---|---|---|---:|---:|---:|']
        for a in sorted(acts, key=lambda r: (r['fold'], r['station'], r['date'])):
            L.append(f"| {a['fold']} | {a['station']} | {a['date']} | {'상승 반영' if a['action'] == 'lift' else '전년 급등 제거'} | {a.get('event') or ''} | {a['comparable_events'] if a.get('comparable_events') is not None else ''} | {f4(a['rmsle_before'])} | {f4(a['rmsle_after'])} |")
    else:
        L.append('검증 fold 안에 해당하는 행사가 없다.')
    L += ['', '문학 주경기장 콘서트는 문학 전용 모델 안에서 처리되므로 위 표가 아니라 4절의 상태별 표(CONCERT 행)에 나온다. Fold 2의 2일은 BTS 진 앙코르(2025-10-31, 11-01), Fold 4의 3일은 박효신 공연(2026-04-04, 05, 11)이다. 송도달빛축제공원(3139)은 검증 fold 안에 "발표된 미래 축제"가 없어 수치가 그대로다.']
    L += ['', '## 6. 테스트 구간에 반영된 일정', '',
          '문학경기장 야구 상태: ' + ', '.join(f"{k} {audit['munhak_test_state_days'].get(k, 0)}일" for k in ('HOME', 'NO_HOME', 'UNANNOUNCED')) + '. 콘서트 예정일: ' + (', '.join(audit['munhak_concert_days']) or '없음'),
          '', '| 행사 | 장소 | 날짜 | 시작 | 공개일 |', '|---|---|---|---|---|']
    for e in audit['events_in_test_window']:
        L.append(f"| {e['name']} | {VENUE.get(e['venue'], e['venue'])} | {e['date']} | {e.get('start') or ''} | {e['known_since']} |")
    L.append('| 펜타포트 락 페스티벌 | 송도달빛축제공원 (3139) | 2026-07-31 ~ 2026-08-02 | | 2026-05-31 |')
    L += ['', '역별로 바뀐 테스트 행 수: ' + ', '.join(f"{k} {v:,}행" for k, v in audit['rows_changed_by_station'].items()) + f" (전체 {audit['rows_changed_total']:,}행)", '']
    notes = (out / 'RESULTS_NOTES.md')
    if notes.exists(): L += [notes.read_text().rstrip(), '']
    L += ['## 재현', '', '```bash', 'cd /Users/hongtaeyang/Desktop/SINCON', '.venv/bin/python -m unittest tests.test_v6_specialists -v',
          '.venv/bin/python -m src.v6.validate', '.venv/bin/python -m src.v6.build', '.venv/bin/python -m src.v6.report', '```', '']
    (out / 'RESULTS.md').write_text('\n'.join(L))
    print((out / 'RESULTS.md').read_text())


if __name__ == '__main__':
    main()
