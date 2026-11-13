"""Write outputs/ensemble_v6/PIPELINE_V6.md: what the v6 pipeline does to data, models, inference and post-processing.

Usage (from the project root):  python -m src.v6.pipeline_doc
Scores and counts are read from the artefacts of validate.py / build.py. Findings from one-off analyses
are quoted with the row subset they were measured on.
"""
import json
from pathlib import Path
from . import munhak, events

ROOT = Path(__file__).resolve().parents[2]
VENUE = {'munhak_main': '인천문학경기장 주경기장', 'asiad_main': '인천아시아드주경기장', 'bucheon': '부천종합운동장', 'songdo_festival_park': '송도달빛축제공원'}
KIND = {'concert': '콘서트', 'football': '축구', 'festival': '축제'}


def main():
    cfg = json.loads((ROOT / 'configs/ensemble_v6.json').read_text()); out = ROOT / cfg['output']
    val = json.loads((out / 'validation_summary.json').read_text()); sub = json.loads((out / 'submission_validation.json').read_text())
    audit = json.loads((out / 'schedule_audit.json').read_text()); bw = json.loads((out / 'blend_weights.json').read_text())
    doc = json.loads((ROOT / cfg['schedule']).read_text())
    w = bw['weights']; versions = bw['report']['versions']; lv = val['local_validator']
    saved, lofo = lv['saved_ensemble_columns'], lv['leave_one_fold_out_columns']
    ev = doc.get('events', []); test_ev = [e for e in ev if e['date'] > cfg['final_cutoff']]
    lift = json.loads((out / 'event_lift_summary.json').read_text())
    fam = lambda sid, f: lift[str(sid)].get(f, dict(events=0, mean_peak_log_lift=float('nan'), ratio=float('nan')))
    k1, k2 = fam(3754, 'k1_2026'), fam(3754, 'k2_2025')
    mp, ep = munhak.PARAMS, events.PARAMS; fit = audit['munhak_fit']
    wt = lambda seg: ' / '.join(f"{v} {w[seg][v]:.2f}" for v in versions)
    row = lambda r: f"{r['all_oof']:.4f} | {r['weighted_4fold']:.4f} | {r['fold_3_4']:.4f} | {r['fold_4']:.4f}"
    st = val['station_rmsle']
    L = []
    L += ['# SINCON ensemble_v6 파이프라인', '',
          f"작성 기준 2026-11-12. 이 문서의 점수는 모두 **로컬 검증(OOF) RMSLE**이며, 리더보드 점수는 계산에도 문서에도 쓰지 않았다. 숫자는 `outputs/ensemble_v6/`의 JSON 산출물에서 읽어 넣었다.", '',
          '## 0. 한눈에 보기', '',
          'v6는 새 전역 모델을 학습하지 않는다. 이미 학습·검수된 v2·v4·v5의 예측을 로컬 OOF로 구한 가중치로 섞고, 사전 발표 일정이 있는 역의 해당 날짜만 전용 모델로 바꾼다.', '',
          '```text',
          '원본 CSV (train / test / metaData)',
          '   ├─ 기존 앙상블 v2 · v4 · v5 의 OOF 예측과 테스트 예측 (읽기 전용)',
          '   │      └─ ① 버전 블렌드: 로컬 OOF로 구한 볼록 가중치, 일반 행 / 공휴일 행',
          '   ├─ 사전 발표 일정 (공개일 ≤ 기준일인 것만)',
          '   │      ├─ ② 문학경기장(3127) 전용 모델: 야구 원편성 3상태 + 주경기장 콘서트',
          '   │      └─ ③ 행사 보정: 전년 급등 제거 → 발표된 행사 상승 반영 (3209·3210·3754·3139)',
          '   └─ ④ expm1 복원, 비음수 확인, test ID 순서로 저장 → submission_ensemble_v6.csv (1개)',
          '```', '',
          '| 로컬 검증(OOF), 기존 보고서와 같은 방식 | 전체 OOF | 가중 4-Fold | Fold 3+4 | Fold 4 |', '|---|---:|---:|---:|---:|']
    for key, label in [(v, f'{v} 단독') for v in versions] + [('v6_base_blend', 'v6 ① 블렌드만'), ('v6_final', '**v6 최종 (①+②+③)**')]:
        L.append(f'| {label} | {row(saved[key])} |')
    L += ['', f"엄격한 방식(각 버전의 fold 제외 예측 사용)으로는 v6 블렌드만 {lofo['v6_base_blend']['all_oof']:.4f}, 문학 전용까지 {lofo['v6_blend_plus_munhak']['all_oof']:.4f}, 최종 {lofo['v6_final']['all_oof']:.4f}이다(전체 OOF). 두 방식의 정의는 6절에 있다.", '']

    # ------------------------------------------------------------------ data
    L += ['## 1. 데이터', '', '### 1.1 대회 제공 데이터', '',
          '- `train.csv` 688,176행: 2025-01-01~2026-06-30, 71개 역, 06~23시 18개 시간대의 승차·하차.',
          '- `test.csv` 157,194행: 2026-07-01~10-31. 기준일 2026-06-30 하나에서 123일 전체를 예측한다.',
          '- 타겟은 `log1p(boardings)`로 다루고 마지막에 `expm1`로 되돌린다. 공식 지표 RMSLE와 같은 공간이다.', '',
          '### 1.2 v6에서 추가한 외부 일정', '',
          '규칙은 하나다. **공개일(known_since)이 기준일보다 늦은 일정은 그 기준일의 예측에 보이지 않는다.** 검증 fold마다 같은 필터가 걸리고, 테스트는 2026-06-30까지 공개된 것만 쓴다.', '',
          '| 종류 | 역 | 내용 | 공개일 | 테스트 구간 해당일 |', '|---|---|---|---|---|']
    kbo = {r['season']: r for r in doc['kbo_munhak'] if r['stage'] == 'original'}
    states = audit['munhak_test_state_days']
    L.append(f"| KBO 원편성 | 3127 | 2025년 문학 홈 {kbo[2025]['n_home']}일, 2026년 {kbo[2026]['n_home']}일(~{kbo[2026]['coverage_end']}) | {kbo[2025]['known_since']}, {kbo[2026]['known_since']} | 홈 {states.get('HOME', 0)}일, 홈 아님 {states.get('NO_HOME', 0)}일, 미발표 {states.get('UNANNOUNCED', 0)}일 |")
    for venue in ('munhak_main', 'asiad_main', 'bucheon'):
        sel = [e for e in ev if e['venue'] == venue]
        if not sel: continue
        tsel = [e for e in sel if e['date'] > cfg['final_cutoff']]
        stations = ', '.join(str(s) for s in sel[0]['stations'])
        L.append(f"| {KIND[sel[0]['kind']]} ({VENUE[venue]}) | {stations} | 과거 {len(sel) - len(tsel)}일, 테스트 구간 {len(tsel)}일 | 행사별(출처 문서) | " + (', '.join(sorted(e['date'] for e in tsel)) if tsel else '없음') + ' |')
    for f in doc['festivals']:
        if f['start'] > cfg['final_cutoff']:
            L.append(f"| 축제 (펜타포트) | 3139 | 2025년 3일 관측, 2026년 3일 예정 | {f['known_since']} | {f['start']}~{f['end']} |")
    L += ['', '출처, 공개일 근거, 별첨 파일 해시는 `OFFICIAL_SOURCES_V6.md`에, 기계 판독본은 `configs/domain_schedule_v6.json`에 있다.', '',
          '### 1.3 일정 데이터를 다루는 원칙', '',
          '- **최초 발표본만 쓴다.** 우천 취소 경기를 일정에서 지우지 않고, 나중에 편성된 순연·잔여 경기와 포스트시즌은 넣지 않는다. 2026-08-25에 발표된 KBO 잔여 일정은 포함하지 않았다.',
          '- **사후 정보는 쓰지 않는다.** 경기 결과, 관중 수, 실제 개최 여부, 시즌 중 변경은 입력이 아니다.',
          '- **과거 행사는 관측으로 확인한다.** 과거 날짜의 행사는 그날의 승차 급등으로 대조했다. 문학은 야구 편성일 중 실제 급등이 있었던 날만 경기일 표본으로 쓴다.',
          f"- **행사일은 평시 수준 계산에서 뺀다.** 문학의 평시 표는 일 최대 승차 {munhak.SPIKE:,}명 미만인 날로만 만들고, 행사 보정의 평시 수준은 행사·공휴일이 아닌 같은 요일의 중앙값이다.", '',
          '### 1.4 데이터에서 확인한 사실', '',
          f"- 문학 야구: 기준일까지 편성 {fit['scheduled']}일 중 {fit['realised_games']}일에 급등이 있었고 {fit['cancelled']}일은 없었다. 일정에 없던 급등은 {fit['unscheduled_events']}일이며 모두 토·일·공휴일이었다. 그중 확인된 것은 주경기장 콘서트와 며칠 전에 발표되는 '불꽃야구' 직관 경기다.",
          '- 문학 주경기장 콘서트: 2026-04-04·05·11에 16~17시 하차가 몰리고 22시 승차가 7,300~8,850명이었다. 평시 토요일 22시는 15명 안팎이다.',
          '- 아시아드주경기장 콘서트는 아시아드경기장역(3209)과 서구청역(3210) 두 역에서 22~23시에 같이 급등한다. 서구청역 쪽이 더 큰 날도 있다.',
          f"- 부천종합운동장 축구: 경기 종료 직후 한 시간(킥오프 + 2시간)에 승차가 몰린다. 그 시간의 상승폭은 2025년 K리그2 {k2['events']}경기 평균 로그 {k2['mean_peak_log_lift']:.2f}(평시의 약 {k2['ratio']:.1f}배), 2026년 K리그1 {k1['events']}경기 평균 로그 {k1['mean_peak_log_lift']:.2f}(약 {k1['ratio']:.1f}배)다. 승격 후 상승폭이 약 {k1['ratio'] / k2['ratio']:.1f}배 커졌다. 일정표의 모든 경기에서 상승이 관측됐다. 3월 말~4월 중순의 낮 시간 급등은 축구가 아니라 매년 반복되는 봄꽃철 패턴이다.",
          f"- 아시아드주경기장 콘서트의 최대 시간 상승폭(3209 / 3210): 싸이 흠뻑쇼 로그 {fam(3209, 'psy_summer_swag')['mean_peak_log_lift']:.2f} / {fam(3210, 'psy_summer_swag')['mean_peak_log_lift']:.2f}, 세븐틴 {fam(3209, 'seventeen')['mean_peak_log_lift']:.2f} / {fam(3210, 'seventeen')['mean_peak_log_lift']:.2f}. 가수에 따라 규모가 달라 같은 계열 표본을 먼저 쓴다.",
          '- 작년 콘서트의 급등은 올해 같은 주말 예측에 묻어난다. 블렌드 기준으로 2026-09-13과 10-18의 아시아드경기장역 22~23시 예측이 이웃 일요일보다 로그 0.5~0.7 높았다. 전년 같은 날 값을 피처로 쓰는 v2에서 주로 생긴다.', '']

    # ----------------------------------------------------------------- model
    L += ['## 2. 모델', '', '### 2.1 기반 앙상블 (재학습 없음)', '',
          '| 버전 | 구성 | 기준선 |', '|---|---|---|',
          '| v2 | CatBoost·TabM 중심 4모델, 반복 수 2,000~3,200회 | 최근 28·56일 평시 평균(rolling) + 전년 계절 피처 |',
          '| v4 | 직접 트리 3종 + TabM + 분해 GBDT + 저수요 + 그래프 Transformer | 전년 같은 시기(±21일) 커널을 표본 수에 따라 최근 기준선과 결합 |',
          '| v5 | v4 구조 + 5개 버킷별 가중 | 전년 커널(±14일, 급등 진폭에 따라 원시 평균 혼합) + 전년 대비 성장 + 공휴일 영향 + 신설역 감쇠 추세 |', '',
          '세 버전은 잘하는 구간이 다르다. 전년 정보가 없는 fold 1·2에서는 v2가, 전년 정보가 풍부한 fold 3·4에서는 v5가 낫다. 테스트는 전년 정보가 있지만 계절이 다른 구간이어서 어느 한쪽을 고르기보다 섞는다.', '',
          '### 2.2 ① 버전 블렌드', '',
          '- 입력: 각 버전의 fold 제외(LOFO) 앙상블 예측. 버전 내부 블렌더가 그 fold의 정답을 보지 않은 값이다.',
          '- 방법: 4개 fold를 합친 로그 MSE를 최소화하는 볼록 가중치(0 이상, 합 1, 절편 없음)를 0.01 격자에서 정확히 찾는다.',
          '- 구분: 공휴일 행과 그 외 행에 따로 가중치를 둔다. 전용 모델이 덮어쓰는 3127 행은 적합에서 뺀다.',
          f"- 결과: 일반 행 {wt('ordinary')}, 공휴일 행 {wt('holiday')}.",
          '- 점검: 한 fold를 빼고 구한 가중치를 그 fold에 적용해도 점수가 비슷하다(RESULTS.md 2절). 블렌드는 어느 fold에서도 최악 버전보다 크게 낫고 최선 버전에 가깝다.', '',
          '### 2.3 ② 문학경기장 전용 모델 (역 3127)', '',
          '경기 있는 날과 없는 날의 수요가 수십 배 다르므로, 한 모델이 평균을 내면 양쪽 모두에서 틀린다. 지표가 로그 오차이므로 최선의 예측은 `E[log1p(승차)]`이고, 이를 일정 상태별 혼합으로 직접 만든다.', '',
          '| 상태 | 예측 | 테스트 일수 |', '|---|---|---:|',
          f"| HOME (홈 편성) | `(1−q)·경기 프로필 + q·평시 프로필`, q = 과거 취소율 | {states.get('HOME', 0)} |",
          f"| NO_HOME (홈 아님) | `(1−r)·평시 + r·비편성 행사 프로필`, r = 과거 비편성 급등률 | {states.get('NO_HOME', 0)} |",
          f"| UNANNOUNCED (미발표) | `p·경기 + (1−p)·평시`, p = 작년 같은 시기(±21일) 같은 요일군의 급등일 비율 | {states.get('UNANNOUNCED', 0)} |",
          f"| 콘서트 예정일 | `(1−c)·max(그날 혼합, 평시 + {mp['concert_shrink']}·콘서트 상승) + c·그날 혼합`, c = {mp['concert_cancel']} | {len(audit['munhak_concert_days'])} |", '',
          '- 프로필은 시간대별 로그 평균이다. 경기 프로필은 요일군(평일/토/일·공휴일) × 계절 구간(봄/6월/여름/가을)으로 나눠 개시 시각 차이를 흡수하고, 표본이 적은 칸은 개시 시각이 같은 구간과 요일군 평균으로 수축한다.',
          f"- 테스트 적합값: 취소율 여름 {fit['cancel']['summer']:.3f} / 그 외 {fit['cancel']['other']:.3f}, 비편성 급등률 평일 {fit['event_rate'][0]:.3f} / 토 {fit['event_rate'][1]:.3f} / 일·공휴일 {fit['event_rate'][2]:.3f}, 관측된 콘서트일 {fit.get('observed_concert_days', 0)}일.",
          '- 콘서트 상승은 과거 콘서트일의 (실제 − 평시) 로그 차이 평균이며, 발표된 시작 시각 차이만큼 시간축을 옮긴다.',
          '- 설정값 8가지를 하나씩 바꿔도 전체 OOF 변화는 0.0002 이내였다(야구 부분, 2026-11-12 측정). 검증 점수로 고른 값은 없다.', '',
          '### 2.4 ③ 행사 보정 (역 3209·3210·3754·3139)', '',
          '평시 수요는 기반 앙상블이 이미 잘 맞히는 역이므로, 행사일의 해당 시간만 고친다.', '',
          f"1. **전년 급등 제거.** 확인된 과거 콘서트·축제일 e에 대해 e+364일의 예측을, 그 행사가 로그 {ep['despike_min_lift']} 넘게 올렸던 시간대에 한해 이웃 주(±1, ±2주 같은 요일, 행사·공휴일 제외)의 예측 수준으로 낮춘다. 리그 축구는 격주로 반복되므로 대상에서 뺀다.",
          f"2. **상승 반영.** 발표된 미래 행사일에는 `평시 수준 + {ep['shrink']} × 상승폭`을 쓴다. 상승폭은 비교 가능한 과거 행사의 (실제 − 같은 요일 평시 중앙값, ±{ep['ordinary_window']}일) 로그 차이 평균이고, {ep['min_lift']} 이하인 시간대는 건드리지 않는다.",
          f"3. **정렬.** 축구는 관중이 빠지는 시각(킥오프 + {ep['football_exit_offset']:.0f}시간)에 맞춰 정렬한다. 콘서트·축제는 시계 시각 기준이고, 시작 시각이 둘 다 알려져 있으면 그 차이만큼 옮긴다.",
          f"4. **비교 표본.** 같은 계열(같은 가수 투어, 같은 리그 시즌, 축제의 같은 일차)을 먼저 쓴다. 축구는 같은 시즌 경기가 {ep['min_family_events']}경기 이상 쌓였을 때만 그 시즌 표본을 쓰고, 아니면 그 역의 과거 축구 전체를 쓴다.", '',
          f"2026년 부천은 K리그1로 올라와 경기일 상승폭이 2025년보다 크다. 테스트 예측은 2026-03~05에 관측된 K리그1 홈경기 {k1['events']}경기의 상승폭을 쓴다. 기준일 2026-02-27 fold에서는 그 표본이 아직 없어 2025년 표본으로 예측하므로, 검증 수치는 테스트보다 불리한 조건이다.", '',
          '행사 예정일에는 그날의 기반 예측을 이웃 주 수준 이하로 먼저 제한한 뒤 상승폭을 더한다. 작년 같은 주말의 경기가 전년 피처로 남긴 흔적 위에 상승폭이 한 번 더 얹히는 것을 막기 위해서다.', '',
          '### 2.5 모델을 더 키우지 않은 이유', '',
          'v5 OOF로 잰 값이다(2026-11-12, 전체 628,776행 기준).', '',
          '- v5의 7개 모델 잔차 상관 평균이 0.863이고, 행마다 가장 가까운 모델을 정답을 보고 골라도 0.152다. 모델 종류를 늘려 얻을 다양성은 거의 남아 있지 않다.',
          '- 역×시간×요일유형×월 수준을 완벽히 맞혀도 0.157이 남는다. 나머지는 그날의 잡음이다.',
          '- 미래 ±2주 실측을 보고 평활한 예측기도 비공휴일 행에서 0.179다(v5는 같은 행에서 0.206).',
          '- 반면 문학 한 역이 v5 제곱오차의 15.5%였고, 이 오차는 "그날 경기가 있는지"라는 정보로만 줄어든다. 그래서 v6는 모델 용량이 아니라 일정 정보에 투자했다.', '']

    # ------------------------------------------------------------- inference
    changed = audit['rows_changed_by_station']
    L += ['## 3. 추론', '',
          '`python -m src.v6.build` 한 번으로 아래 순서가 실행된다. 검증(`src.v6.validate`)도 같은 함수 `apply_specialists`를 fold 기준일마다 호출하므로, 검증한 코드와 제출을 만드는 코드가 같다.', '',
          '1. 로컬 OOF에서 블렌드 가중치를 다시 계산해 `blend_weights.json`에 저장한다(가중치는 설정 파일에 적혀 있지 않다).',
          '2. v2·v4·v5 테스트 예측을 로그 공간에서 가중 평균한다. 공휴일 행은 공휴일 가중치를 쓴다.',
          '3. 2026-06-30까지의 관측과 그때까지 공개된 일정으로 문학 전용 모델을 적합하고, 3127의 시즌 구간 행을 교체한다.',
          '4. 행사 보정을 적용한다(전년 급등 제거 → 상승 반영).',
          '5. `expm1`로 되돌리고 157,194행·ID 순서·유한·비음수를 확인해 저장한다.', '',
          f"테스트에서 바뀐 행은 전체 {audit['rows_changed_total']:,}행이다: " + ', '.join(f"{k} {v:,}행" for k, v in changed.items()) + '. 나머지 행은 블렌드 값 그대로다.', '',
          '테스트 구간에 반영된 행사:', '', '| 행사 | 장소 | 날짜 | 시작 | 공개일 |', '|---|---|---|---|---|']
    for e in sorted(test_ev, key=lambda e: e['date']):
        L.append(f"| {e['name']} | {VENUE.get(e['venue'], e['venue'])} | {e['date']} | {e.get('start') or ''} | {e['known_since']} |")
    for f in doc['festivals']:
        if f['start'] > cfg['final_cutoff']: L.append(f"| 펜타포트 락 페스티벌 | 송도달빛축제공원 | {f['start']}~{f['end']} | | {f['known_since']} |")
    L += ['', f"제출 파일 평균 {sub['mean']:.1f}명, 최소 {sub['min']:.2f}명, 최대 {sub['max']:.1f}명.", '']

    # ------------------------------------------------------- post-processing
    L += ['## 4. 후처리', '', '### 4.1 적용한 것', '',
          '| 처리 | 대상 | 근거 |', '|---|---|---|',
          '| 로그 공간 결합 후 `expm1`, 0 미만 절삭 | 전체 | 지표가 로그 오차다. 인원수를 평균하면 큰 값에 끌린다 |',
          '| 공휴일 행 별도 가중치 | 공휴일 9일 | 공휴일 행에서 v5가 4개 fold 모두 v2·v4보다 나았다 |',
          '| 전년 급등 제거 | 과거 콘서트·축제일 + 364일 | 일회성 행사가 전년 피처를 통해 올해 같은 주말로 넘어온다 |',
          f"| 행사 상승 반영 (축소 {ep['shrink']}) | 발표된 행사일의 해당 시간 | 과거 같은 계열 행사의 관측 상승폭. 표본이 적어 15% 줄여 쓴다 |",
          '| 문학 상태별 혼합 | 3127 시즌 구간 | 2.3절 |', '',
          '### 4.2 검토했지만 적용하지 않은 것', '',
          '모두 v5 OOF에서 2026-11-12에 측정했다. "앞 fold에서 정하고 뒤 fold에 적용"했을 때 나빠지거나 fold마다 방향이 달랐던 것들이다.', '',
          '| 후보 | 결과 (로컬 검증) | 판단 |', '|---|---|---|',
          '| 전역 편향 상수 보정 | 로그 기준 최종 편향이 이미 0 근처. 인원수 평균 편향을 0으로 맞추는 것은 RMSLE와 다른 목표 | 미적용 |',
          '| 저수요 구간 편향 보정 | fold 1·2에서 정해 fold 3·4에 적용하면 0.1956 → 0.2034 | 미적용 |',
          '| 최근 수준(56일 평균) 쪽으로 수축 | 최적 비중이 fold 1·2는 0.2~0.8, fold 4는 1.0~1.2로 엇갈림 | 미적용 |',
          '| 이전 fold의 셀 잔차로 보정 | 역×시간 셀 편향의 fold 간 상관이 −0.15~+0.09 | 미적용 |',
          '| 작년 주별 형상으로 수준 이동 | fold 4에서 v2는 개선(0.1886 → 0.1797), v5는 악화(0.1586 → 0.1624) | 미적용 |',
          '| 일총량과의 계층 정합 | 일총량 분해 모델(0.239)이 직접 예측(0.224)보다 나쁨 | 미적용 |',
          '| 노선 그래프·시간 평활 | 예측 잔차를 평활해도 그날의 충격은 맞힐 수 없음. 검증하지 않음 | 미적용 |',
          '| 리더보드 점수로 가중치 조정 | 대회 규정상 사용하지 않기로 결정 | 제거 |', '']

    # ------------------------------------------------------------ validation
    folds = ['1', '2', '3', '4', 'all']
    L += ['## 5. 검증 결과 (로컬 검증, 엄격한 방식)', '', '| 단계 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 전체 |', '|---|---:|---:|---:|---:|---:|']
    for key, label in [(v, f'{v} 단독') for v in versions] + [('base', '① 블렌드'), ('munhak_only', '①+② 문학 전용'), ('final', '①+②+③ 최종')]:
        L.append(f'| {label} | ' + ' | '.join(f"{val['overall_rmsle'][key][f]:.4f}" for f in folds) + ' |')
    L += ['', '역별 RMSLE (블렌드 → 최종):', '', '| 역 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 전체 |', '|---|---|---|---|---|---|']
    for sid in ('3127', '3754', '3209', '3210', '3139'):
        L.append(f"| {sid} {st[sid]['name']} | " + ' | '.join(f"{st[sid]['base'][f]:.3f} → {st[sid]['final'][f]:.3f}" for f in folds) + ' |')
    L += ['', '행사일 하나하나의 전후 수치는 `RESULTS.md` 5절에 있다.', '',
          '## 6. 점수 읽는 법', '',
          '- **로컬 검증(OOF)**: 기준일 4개(2025-08-30, 10-30, 12-30, 2026-02-27)에서 각각 123일을 예측해 실제값과 비교한 RMSLE. fold당 157,194행.',
          '- **기존 방식**: 각 버전의 저장된 앙상블 예측을 입력으로 쓴다. v2·v4·v5 보고서의 0.2228 / 0.2213 / 0.2135와 같은 기준이다.',
          '- **엄격한 방식**: 각 버전이 그 fold를 빼고 내부 가중치를 구해 만든 예측을 입력으로 쓴다. 버전 내부 블렌더의 낙관이 빠져 조금 높게 나온다. v6의 가중치는 이 예측으로 구했다.',
          '- **가중 4-Fold**: `sqrt(0.15·MSE1 + 0.15·MSE2 + 0.35·MSE3 + 0.35·MSE4)`. **Fold 3+4**: 두 fold의 행을 합친 값.',
          '- 역별·행사일별 수치는 그 역, 그 날짜 행만의 RMSLE다. 전체 점수와 직접 비교하지 않는다.', '',
          '## 7. 한계', '',
          '- 7~8월을 예측 대상으로 하면서 전년 정보까지 갖춘 로컬 fold는 만들 수 없다(데이터가 2025-01부터). 여름 구간의 블렌드 가중치는 다른 계절 fold에서 구한 값이다.',
          '- 콘서트·축제 상승폭은 과거 같은 계열 1~5회의 관측에 기댄다. 올해 관객 규모가 다르면 그만큼 틀린다. 축소 계수 0.85는 이 불확실성을 감안한 고정값이고 검증으로 고른 값이 아니다.',
          '- 펜타포트는 과거 사례가 2025년 1회뿐이라 표본 외 검증이 없다.',
          '- 2026-06-30 이후에 발표된 행사(9~10월 콘서트, KBO 잔여 일정, 포스트시즌)는 알 수 없다. 문학은 확률 혼합으로 대비하지만 다른 역은 대비가 없다.',
          '- 일반 행의 남은 오차는 대부분 그날의 잡음이다. 로컬 검증 0.13 수준은 이 데이터와 정보로는 도달할 수 없다.', '',
          '## 8. 파일과 재현', '',
          '| 역할 | 파일 |', '|---|---|',
          '| 블렌드 가중치 | `src/v6/blend.py` |', '| 일정 로딩·공개일 필터 | `src/v6/schedule.py` |', '| 문학 전용 모델 | `src/v6/munhak.py` |',
          '| 행사 보정 | `src/v6/events.py` |', '| 공통 추론 경로 | `src/v6/pipeline.py` |', '| 검증 / 제출 생성 / 문서 | `src/v6/validate.py`, `build.py`, `report.py`, `pipeline_doc.py`, `handoff.py` |',
          '| 설정 | `configs/ensemble_v6.json`, `configs/domain_schedule_v6.json` |', '| 테스트 | `tests/test_v6_specialists.py` |', '',
          '```bash', 'cd /Users/hongtaeyang/Desktop/SINCON', '.venv/bin/python -m unittest tests.test_v6_specialists -v', '.venv/bin/python -m src.v6.validate',
          '.venv/bin/python -m src.v6.build', '.venv/bin/python -m src.v6.report', '.venv/bin/python -m src.v6.pipeline_doc', '.venv/bin/python -m src.v6.handoff', '```', '',
          '테스트는 (1) v6 코드·설정에 리더보드 점수가 없을 것, (2) 가중치가 OOF에서 재현될 것, (3) 정보 마감 이후 공개된 일정이 없을 것, (4) 공개일 이전 기준일에서는 행사가 보이지 않을 것, (5) 기준일 이후 관측을 전부 바꿔도 예측이 한 값도 달라지지 않을 것, (6) 전용 모델이 지정된 역 밖을 건드리지 않을 것을 확인한다.', '']
    (out / 'PIPELINE_V6.md').write_text('\n'.join(L))
    print('written', out / 'PIPELINE_V6.md', len(L), 'lines')


if __name__ == '__main__':
    main()
