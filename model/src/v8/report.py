"""Writes RESULTS_V8.md and HANDOFF_V8.md from gates_v8.json (python -m src.v8.report). Local-validation numbers only."""
import hashlib
import json
import pandas as pd
from . import core

OUT = core.OUT
NAME = {3127: '문학경기장', 3226: '인천대공원', 3754: '부천종합운동장', 3107: '검단호수공원', 3136: '인천대입구', 3139: '송도달빛축제공원',
        3209: '아시아드경기장', 3121: '동수', 3110: '계양', 3111: '귤현'}
HEAD = ['구분', '전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']


def f5(x):
    return f'{x:.5f}'


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def row(label, m):
    return [label] + [f5(m[k]) for k in ('all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3')]


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    g = json.loads((OUT / 'gates_v8.json').read_text())
    v = pd.read_csv(OUT / 'VALIDATION_V8.csv')
    base = g['baseline_v7']; pre = g['chains']['precutoff']; mn = g['chains']['main']; al = g['aligned_truth_sensitivity']
    L = ['# ensemble_v8 로컬 검증 결과와 비교표 (RESULTS_V8)\n',
         '> **모든 점수는 로컬 검증(OOF) RMSLE다.** 리더보드 점수는 어떤 계산에도 쓰지 않았다. 숫자는 중첩 LOFO(각 fold에 쓴 계수는 나머지 3개 fold에서만 적합)이고, '
         '출발점은 v7의 중첩 LOFO 예측이다. 로컬 검증기 4개 지표 = 전체 OOF, 가중 4-Fold, Fold 3+4, Fold 4.\n',
         '## 1. 세 제출본 비교\n',
         table(HEAD, [row('v7 (기준)', base), row('**보험용 submission_precutoff.csv** — 상반기 마감 기준만', pre['final']),
                      row('**주력 submission_ensemble_v8.csv** — 실측 경기·강수 포함', mn['final'])]) + '\n',
         table(['파일', 'SHA-256', '평균 승차', 'v7 대비 로그 RMSD', 'v7 대비 평균 로그 차이', '바뀐 행'],
               [[c['submission']['file'], f"`{c['submission']['sha256']}`", f"{c['submission']['mean_boardings']:.2f}", f"{c['submission']['log_rmsd_vs_v7']:.4f}",
                 f"{c['submission']['mean_log_diff_vs_v7']:+.4f}", f"{c['submission']['rows_changed_vs_v7']:,}"] for c in (pre, mn)]) + '\n',
         '## 2. 모듈별 누적 (주력 체인; 앞의 세 단계가 보험용 체인과 같다)\n']
    rows = [['v7'] + [f5(base[k]) for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + ['—', '—']]
    for s in mn['steps']:
        if 'after' not in s:
            continue
        a = s['after']
        rows.append([s['step']] + [f5(a[k]) for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + [str(s['coefficients_all_folds']), '**채택**' if s['adopted'] else '기각'])
    L.append(table(['단계', '전체 OOF', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '계수(4개 fold 전체 적합, 테스트에 사용)', '판정'], rows) + '\n')
    L.append('채택 규칙: 중첩 전체 점수가 좋아지고 나빠지는 fold가 없을 것. 기각된 단계의 수치는 그 단계를 넣었을 때의 값이며 다음 단계는 기각 전 상태에서 이어진다.\n')
    L.append('## 3. 역별 RMSLE (주력 체인, 4개 fold 합산)\n')
    m = v[v.chain == 'main']
    cols = [c for c in m.columns if c.startswith('st_')]
    L.append(table(['단계'] + [f"{c[3:]} {NAME.get(int(c[3:]), '')}" for c in cols], [[r.step] + [f'{getattr(r, c):.4f}' for c in cols] for r in m.itertuples()]) + '\n')
    L.append('## 4. 집계 단절을 뺀 민감도\n')
    L.append('train(공공데이터포털 일별 파일)은 2025년분만 공사 월별 자료보다 역×시간별로 0.74~1.00배 적고 2026년분은 정확히 같다. 그래서 기준일이 2025년인 fold의 2026년 정답에는 '
             '어떤 예측도 알 수 없는 수준 차이가 섞여 있다(귤현 로그 −0.31, 검단호수공원 −0.15, 검단오류·운연·부천종합운동장 −0.06). 그 정답을 2025년 기준으로 환산해 다시 채점하면 다음과 같다. '
             '테스트는 기준일과 대상이 모두 2026년이라 이 문제가 없다.\n')
    L.append(table(HEAD, [row('v7', al['v7']), row('보험용', al['precutoff']), row('주력 v8', al['main'])]) + '\n')
    L.append('## 5. 적합된 내용\n')
    for s in mn['steps']:
        if s['step'].startswith('A1'):
            d = s['details']['test']
            L.append(f"- **문학 실제 경기**: 과거 {d['games_fitted']}경기로 적합. 관중 대비 경기 후 승차 비율 {d['exit_share_of_attendance']:.3f}, 요일군 배율(평일/토/일·공휴일) {d['group_scale']}, "
                     f"경기·콘서트가 아닌 날의 급등 확률 {d['residual_event_rate']}. 종료 시각 기준 분 단위 유출 비율: {d['kernel']}. 테스트 구간 경기일 {d['game_days_in_block']}일.")
        if s['step'].startswith('A2'):
            d = s['details']['test']
            L.append(f"- **부천 실제 경기**: 과거 {d['matches_fitted']}경기, 관중 대비 승차 비율 {d['exit_share_of_attendance']:.3f}, 테스트 구간 {d['match_days_in_block']}경기(8월 22일 포항전 포함).")
        if s['step'].startswith('A3'):
            L.append(f"- **이슬라이브 페스티벌(3139, 2026-09-12)**: 비교할 과거 회차가 없어 과거 토요일 축제일({', '.join(s['past_days'])}) 평균 상승폭의 {s['scale']}배를 더했다. 검증 불가. 17~23시 예측 {s['boardings_before'][11:]} → {s['boardings_after'][11:]}.")
        if s['step'].startswith('A4'):
            L.append(f"- **실측 강수**: 계수 [당일 비, 전년 같은 요일 비의 흔적] = {s['coefficients_all_folds']}. 비 효과 추정: {g['weather']['mean_effect']} (level1 = 운행시간 강수 1~10mm, level2 = 10mm 이상), "
                     f"테스트 기준일까지 비 온 날 {g['weather']['wet_days']}일.")
        if s['step'].startswith('P2'):
            L.append(f"- **공휴일**: 계수 순서 = [일반 공휴일 보통역, 특이역, 명절 당일 보통역, 특이역, 명절 전후 보통역, 특이역, 신정 ×2, 징검다리 ×2, 연휴 주말 ×2] = {s['coefficients_all_folds']}.")
        if s['step'].startswith('P1'):
            L.append(f"- **다년 계절 지수**: 계수 {s['coefficients_all_folds']} (fold별 {s['coefficients_nested']}).")
    (OUT / 'RESULTS_V8.md').write_text('\n'.join(L) + '\n')

    d = lambda a, b, k='all_oof': f"{f5(a[k])} → {f5(b[k])}"
    steps = {s['step'][:2]: s for s in mn['steps']}
    H = ['# ensemble_v8 인계 문서 (HANDOFF_V8)\n',
         '작성 2026-11-12. **모든 점수는 로컬 검증(OOF, 중첩 LOFO) RMSLE**이고 리더보드 점수는 계산·선택에 쓰지 않았다. v1~v7 파일은 그대로다(SHA 대조 테스트). 새 작업은 `src/v8/`, `outputs/ensemble_v8/`에만 있다.\n',
         '## 0. 결과\n',
         table(HEAD, [row('v7', base), row('보험용 `submission_precutoff.csv`', pre['final']), row('주력 `submission_ensemble_v8.csv`', mn['final'])]) + '\n',
         f"- 보험용: SHA-256 `{pre['submission']['sha256']}`\n- 주력: SHA-256 `{mn['submission']['sha256']}`\n- 둘 다 157,194행. Kaggle 제출은 하지 않았다.\n",
         '## 1. 규칙과 두 파일의 차이\n',
         '사용자가 운영진에 직접 확인한 내용: 평가 기간(2026-07~10)의 **지하철 승하차 수치는 출처를 불문하고 금지**, 실제 경기 결과·관중·종료 시각·취소, 실측 날씨, 실제 개최 행사는 평가 기간 기록도 사용 가능.\n',
         table(['', '보험용 (precutoff)', '주력 (v8)'], [
             ['쓰는 정보', '2026-06-30까지 관측·공개된 것만', '왼쪽 + 평가 기간의 실제 경기(관중·종료 시각·취소), 실측 강수, 실제 개최 행사'],
             ['모듈', 'v7 → 다년 계절 지수 → 역별 공휴일 효과', '왼쪽 → 실측 강수 → 부천 실제 경기 → 문학 실제 경기 → 이슬라이브'],
             ['평가 기간 승하차', '사용 안 함', '사용 안 함'],
             ['코드 보장', '실측 자료 로더를 막아도 같은 파일이 나오는 테스트', '—']]) + '\n',
         '## 2. 모듈 설명\n',
         '### P1 다년 계절 지수 (보험용·주력 공통)\n'
         '- 자료: 인천교통공사 월별 역×시간대 승하차 2015-10~2024-12(팀 공유 자료, train 기준으로 보정된 값). 파손 표시 월(2015-11 일부)과 2020~2022년은 제외.\n'
         '- 계산: 월 합계를 그 달의 평일·토·일공휴일·명절 일수로 나눠 "평일 하루 수준"으로 환산한다(요일 비율은 기준일 이전 train에서 역×시간별로 계산). '
         '각 정상 연도(2016~2019, 2023, 2024)에 대해 [대상 월 수준 − 기준일 직전 두 달 수준]의 로그 변화를 구하고 가중 중앙값을 낸다. 가중치는 최근 연도일수록 크고(1, 1, 1, 1.5, 2.5, 3), '
         '9·10월은 추석이 올해와 같은 달에 있던 연도에 2배를 준다(2026년 추석은 9월).\n'
         '- 구조 변화: 역×월 단위로 최근 2년과 그 이전의 변화가 로그 0.12 넘게 다르면 최근 2년만 쓴다(국제업무지구 8월, 인천대입구 방학, 부평구청 등이 여기에 걸린다). 시간대별 값은 역 평균 쪽으로 절반 수축, 보정폭은 ±0.20으로 제한.\n'
         '- 적용: 예측이 내포한 변화(대상 월 평일 예측 평균 − 직전 두 달 실제 평균)와 지수의 차이에 계수를 곱해 더한다. 신규 3개 역과 문학경기장은 대상이 아니다.\n'
         f"- 로컬: {d(base, steps['P1']['after'])}. 단순 중앙값만 쓴 경우 0.19314.\n",
         '### P2 역별 공휴일 효과 (공통)\n'
         '- 기준일 이전의 공휴일마다 [그날 − 같은 요일 평소 중앙값]을 역×시간별로 구해 범주별 평균을 낸다. 범주: 평일 공휴일, 토요일 공휴일, 근로자의 날, 명절 당일, 명절 전후, 신정, 징검다리, 연휴에 낀 토·일.\n'
         '- 예측에서는 주변 같은 요일 평소 예측의 중앙값에 그 효과를 더한 값을 만들고, 기존 예측과 섞는 비율을 범주군×(보통역/특이역)별로 적합한다. 특이역 = 자기 효과가 전체 역 중앙값과 로그 0.4 이상 다른 역(부평삼거리 성묘, 산업단지 역 등).\n'
         f"- 로컬: {d(steps['P1']['after'], steps['P2']['after'])}. 공휴일 발표일을 지킨다(2026-07-17 제헌절은 2026-02-03 지정).\n"
         '- 지시서의 "연휴 마지막 날"은 연휴에 낀 주말 범주로 넣었고 계수가 0.22(보통역)로 작다. 테스트 구간에 징검다리 평일은 없다.\n',
         '### 실측 모듈 (주력만)\n'
         f"- **A4 실측 강수**: 기준일 이전 자료에서 역×(평일/휴일)×시간별 비 효과를 구하고(운행시간 강수 1~10mm, 10mm 이상 두 단계), 대상일의 실제 강수 단계에 해당하는 효과에서 평균 효과를 뺀 만큼 보정한다. 전년 같은 요일(−364일)의 비가 예측에 남긴 흔적도 반대 부호로 넣는다. 로컬: {d(steps['P2']['after'], steps['A4']['after'])}. 2026년 7월은 비 온 날이 22일로 많다.\n"
         f"- **A2 부천**: 실제 홈경기 관중 × 유출 비율(킥오프+115분 기준)로 교체. 로컬: {d(steps['A4']['after'], steps['A2']['after'])}.\n"
         f"- **A1 문학**: 확률 혼합 대신 실제 개최 여부·관중·분 단위 종료 시각을 쓴다. 경기일 = 평소 프로필 + 관중 × 유출 비율(종료 전 120분~종료 후 150분, 16개 구간, 로그 공간 적합). 경기 없는 시즌 날 = 평소 프로필에 남은 급등 확률만 반영. 발표된 콘서트 날은 v7 값을 유지. 로컬: {d(steps['A2']['after'], steps['A1']['after'])}, 문학경기장역 0.553 → 0.483.\n"
         '- **A3 이슬라이브**: RESULTS_V8 5절. 검증 불가.\n',
         '## 3. 넣지 않은 것과 이유\n',
         table(['항목', '결과'], [
             ['전년 추석 연휴의 유령 급락 제거', '검증 fold에서 계수 0(개선 없음). v7 테스트 예측에서 실제 급락은 10월 4일(일) −0.09, 10월 6일(화) −0.06 두 날뿐이고 나머지는 0에 가깝다'],
             ['집계 단절 배율을 예측에 직접 더하기', '테스트와 조건이 같은 Fold 4에서 역 분할 교차검증 0.18214 → 0.18211. 계절 지수가 2026년 수준을 기준으로 삼아 이미 흡수한다. 대신 RESULTS_V8 4절의 민감도 채점에 썼고, v9 재학습에서는 2025년 자료를 2026년 기준으로 환산해 쓴다'],
             ['삼산체육관 배구', '테스트 구간에 삼산 홈경기가 없다(첫 경기 2026-11-01). 10월 31일 개막전은 계양체육관이고 대응 역이 자료에 없다'],
             ['송도해변축제', '지시대로 제외(효과 없음)'],
             ['기온', '이번에는 강수만 썼다. 시간별 기온은 받아 두었고 v9 입력으로 넘긴다'],
             ['가을 아시아드 행사', '공유 자료에 7월 11일 싸이 이후 아시아드 행사가 없다. 이미 반영된 것 외에 추가 없음']]) + '\n',
         '## 4. 위험\n',
         '- 계절 지수 계수는 fold에서 0.80~0.90이고 테스트에는 0.86을 쓴다. 테스트에서는 v4·v5가 이미 작년 7~10월을 알고 있어 fold보다 효과가 작을 수 있다(전년 정보가 있던 Fold 4에서도 0.19008 → 0.18305였다).\n'
         '- 문학 회귀는 과거 124경기로 적합했다. 연장전·우천 중단 경기는 종료 시각과 실제 퇴장 시점이 어긋난다.\n'
         '- 주력 파일은 운영진 답변의 해석에 의존한다. 해석이 달라지면 보험용 파일을 쓴다.\n',
         '## 5. 재현\n',
         '```bash\npython -m src.v8.run both      # 두 제출본, gates_v8.json, VALIDATION_V8.csv\npython -m src.v8.report        # RESULTS_V8.md, HANDOFF_V8.md\npython -m src.v8.tests.test_v8 # 7개 테스트\n```\n',
         '## 6. 외부 자료 (`outputs/ensemble_v8/external/`)\n',
         table(['파일', 'SHA-256', '내용'], [[p.name, f'`{sha(p)[:16]}…`', note] for p, note in [
             (core.SHARE / 'ictr_hourly_monthly_2015_2024.csv', '공사 월별 역×시간대 승하차 2015-10~2024-12 (P1)'),
             (core.SHARE / 'calendar_2015_2026.csv', '공휴일 달력(2015~2024 요일 구성 계산에만 사용)'),
             (core.SHARE / 'ictr_to_train_factor_station_hour.csv', '2025년 train÷공사 배율(민감도 채점)'),
             (core.SHARE / 'games_ssg_munhak.csv', 'SSG 홈경기 2016~2026-10: 관중·종료 시각·취소 (A1)'),
             (core.SHARE / 'games_bucheonfc.csv', '부천FC 홈경기·관중 (A2)'),
             (core.SHARE / 'events_venues.csv', '콘서트·축제 목록(A3 대조)'),
             (core.EXT / 'open_meteo_incheon_hourly_20250101_20261031.json', 'Open-Meteo 시간별 강수·기온, 평가 기간 실측 포함 (A4)'),
             (core.EXT / 'weather_incheon_daily_20250101_20261031.csv', '위 자료의 일별 운행시간 강수 요약')]]) + '\n',
         '팀 공유 자료의 수집 경위와 출처 URL은 `external/team_share/README_team_share.md`에 있다. 평가 기간의 승하차 수치는 어느 파일에도 없다(테스트로 확인).\n']
    (OUT / 'HANDOFF_V8.md').write_text('\n'.join(H) + '\n')
    print('written RESULTS_V8.md, HANDOFF_V8.md')


if __name__ == '__main__':
    main()
