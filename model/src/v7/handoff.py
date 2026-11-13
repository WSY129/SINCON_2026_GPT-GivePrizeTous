"""Writes HANDOFF_V7.md and EXTERNAL_DATA_AUDIT_V7.md (python -m src.v7.handoff). Local-validation numbers only."""
import json
from pathlib import Path
from . import core
from .report import f4, f5, table, metric_row, METRIC_HEAD, model_label, sha, FOLD_NAME

OUT = core.OUT
EXT = OUT / 'external'


def audit_md(g):
    files = sorted(p for p in EXT.glob('*') if p.is_file())
    L = ['# 외부 자료 감사 (EXTERNAL_DATA_AUDIT_V7)\n',
         '규칙: 관측 자료는 **관측일 ≤ 기준일**, 일정·계획 자료는 **공개일(known_since) ≤ 기준일**(최종 추론은 2026-06-30)인 것만 쓴다. '
         '미래 날씨, 미래 하차량, 마감 이후의 일정 변경·결과·취소는 쓰지 않는다. 조사일은 2026-11-12다.\n',
         '## 1. 보관 파일과 SHA-256 (`outputs/ensemble_v7/external/`)\n',
         table(['파일', '크기(바이트)', 'SHA-256'], [[p.name, p.stat().st_size, f'`{sha(p)}`'] for p in files]) + '\n',
         '## 2. 항목별 정리\n',
         '### 2.1 기상 (M1.1 → M2 게이트에 사용, 최종 제출본에는 미사용)\n',
         table(['항목', '내용'], [
             ['출처', 'Open-Meteo Historical Weather API (ERA5 계열 재분석, CC BY 4.0) — `https://archive-api.open-meteo.com/v1/archive`'],
             ['요청', '위도 37.4563, 경도 126.7052(인천), 시간대 Asia/Seoul. 일별: precipitation_sum, rain_sum, snowfall_sum, precipitation_hours, 기온 3종. 시간별: precipitation, temperature_2m. 응답 격자점 37.434N 126.742E'],
             ['기간', '2025-01-01 ~ 2026-06-30 (546일, 결측 없음). 기후값용으로 2015-01-01 ~ 2024-12-31 일 강수량도 받음'],
             ['known_since', '각 날짜의 관측값은 그날 이후 알 수 있으므로 관측일 ≤ 기준일인 행만 읽는다(코드에서 기준일 이후 행을 잘라 쓰고, 미래 값을 바꿔도 특징이 같다는 테스트가 있다)'],
             ['가공', '`weather_incheon_daily.csv`: 운행 시간대(06~23시) 강수 합 service_mm, 비 온 날 표시. 비 온 날 기준은 일 강수 5mm 이상(546일 중 58일 안팎)'],
             ['KMA ASOS(112) / Meteostat', '기상자료개방포털은 로그인·인증키가 필요해 자동 수집하지 않았다. Open-Meteo 재분석 격자값이라 관측소 실측과 강수량이 다를 수 있다'],
             ['사용 결과', '강수 효과는 평일 평균 로그 −0.057, 주말 −0.03으로 추정됐다. 이를 이용한 보정(M2)은 계수 적합형·이론 계수형 모두 로컬 게이트에서 기각돼 제출본에는 들어가지 않는다'],
         ]) + '\n',
         '### 2.2 송도맥주축제 (M1.2 → 채택)\n',
         table(['사실', '값', '출처', '공개일', '마감 전'], [
             ['2023년 일정', '2023-08-25(금) ~ 09-02(토)', 'https://www.newsis.com/view/NISX20230804_0002403164', '2023-08-04', '예'],
             ['2024년 일정', '2024-08-23(금) ~ 08-31(토)', 'https://web.archive.org/web/20260307025930/https://www.ifm.kr/cultur/event/event_beer_2025 (주최 측 연혁, 보관본)', '2026-03-07 보관', '예'],
             ['2025년 일정', '2025-08-22(금) ~ 08-30(토)', 'https://www.insight.co.kr/news/516529', '2025-08-20', '예'],
             ['**2026년 일정**', '**2026-08-22(토) ~ 08-30(일), 9일간, 송도달빛축제공원**', 'https://news.ifm.kr/news/articleView.html?idxno=473558 (경인방송, 기사 문구: "오는 8월 22일부터 30일까지 9일간 송도달빛축제공원에서")', '**2026-06-23 13:45**', '**예**'],
             ['2026년 일정 2차 보도', '같은 일정, 전체 라인업', 'https://news.ifm.kr/news/articleView.html?idxno=473791', '2026-06-25', '예'],
             ['2026년 운영 시간·입장 방식', '마감 전 출처에서 확인되지 않음', '—', '—', '미확인'],
         ]) + '\n',
         '- 이전 조사(v6)에서는 마감 전 출처를 찾지 못해 제외했다. 이번 재조사에서 2026-06-23 기사를 직접 열어 날짜·장소·게시 시각을 확인했다.\n'
         '- 반영 방식: 2025년 축제 9일을 역 3139의 과거 행사(요일 유형별 4개 군: 월~목, 금, 토, 일)로 등록하고, 2026년 9일을 `known_since = 2026-06-23`인 예정 행사로 등록했다. '
         'v6의 행사 보정 코드가 그대로 처리한다(예정일은 이웃 주 수준으로 맞춘 뒤 2025년 같은 요일 유형의 시간대별 상승폭 × 0.85를 더함, 2025-08-22의 +364일인 2026-08-21은 전년 급등 제거).\n'
         '- 시간대 프로필은 2025년 실측에서 나온다(운영 시간 정보는 쓰지 않는다). 2025년 축제일 22시 승차는 1,016~4,230명, 평소 같은 요일은 60~70명이었다.\n',
         '### 2.3 인천대학교 학사일정 (역 3136, M1.3 → 미사용)\n',
         table(['사실', '값', '출처', '공개일', '마감 전'], [
             ['2025-1학기 개강 / 기말 / 보강주', '03-04 / 06-10~16 / ~06-23', 'https://www.inu.ac.kr/inu/651/subview.do', '게시일 표기 없음(과거 사실)', '예'],
             ['2025-2학기 개강 / 기말 / 보강주', '09-01 / 12-08~12 / ~12-19', '같은 페이지', '게시일 표기 없음(과거 사실)', '예'],
             ['2026-1학기 개강 / 기말 / 보강주', '03-03 / 06-09~15 / ~06-22', '같은 페이지 + 공지(2026-02-02) + 보관본(2026-06-14)', '2026-02-02', '예'],
             ['2026 여름 계절학기', '06-23 ~ 07-13', 'https://www.inu.ac.kr/bbs/inu/246/423726/artclView.do', '2026-05-06', '예'],
             ['2026-2학기 개강', '09-01(화)', '현재 페이지에만 있음', '게시일 미확인', '**미확인**'],
         ]) + '\n',
         '### 2.4 경인교육대학교 학사일정 (역 3115, M1.3 → 미사용)\n',
         table(['사실', '값', '출처', '공개일', '마감 전'], [
             ['2025-1학기 개강 / 종강', '03-04 / 06-23', 'https://www.ginue.ac.kr/kor/CMS/ScheduleMgr/YearList.do?mCode=MN058&calendar_year=2025', '과거 사실', '예'],
             ['2025-2학기 개강 / 종강', '09-01 / 12-19', '같은 페이지', '과거 사실', '예'],
             ['2026-1학기 개강 / 종강', '03-03 / 06-22', '2026년 목록 보관본(2026-04-21)', '2026-04-21 보관', '예'],
             ['2026-2학기 개강 / 종강', '08-31(월) / 12-18', '같은 보관본', '2026-04-21 보관', '예'],
         ]) + '\n',
         '- 미사용 사유: (1) 인천대 2026년 2학기 일정은 날짜가 찍힌 마감 전 출처로 확인되지 않았다. (2) 경인교대는 2025-09-01(월) 개강과 2026-08-31(월) 개강이 정확히 364일 차이여서 v4·v5의 전년 정렬이 이미 개강일을 맞춘다. '
         '(3) 학사 상태 특징은 두 역에만 값이 있어 모델이 배울 표본이 적다. 패널에 `acad` 특징 자리는 남겨 두었고 설정의 `academic_periods`가 비어 있어 상수다.\n',
         '### 2.5 검단신도시 입주 물량 (M1.4 → 성장 보정의 방향 근거로만 사용, 모델 입력 아님)\n',
         table(['월', '단지', '세대', '출처(공개일)'], [
             ['2025-01', '힐스테이트 검단 웰카운티 / 불로 포레스트', '1,535 / 736', '블로그 목록(2024-12-14, 청약홈 인용) — 날짜 있는 보도자료로는 미확인'],
             ['2025-04', '신검단중앙역 풍경채 어바니티 2차', '1,734', 'https://www.newsis.com/view/NISX20250326_0003113221 (2025-03-26)'],
             ['2025-06', '검단역 금강펜테리움 더 시글로 2차', '483', '블로그 목록(2025-04-26 / 2025-06-04)'],
             ['2025-09', '신검단중앙역 우미린 클래스원', '875', 'https://www.hankyung.com/article/2025082588926 (2025-08-25)'],
             ['2025-11', '신검단중앙역 금강펜테리움 센트럴파크', '1,049', 'https://www.newsis.com/view/NISX20250623_0003222977 (2025-06-23)'],
             ['2026-02', '검단호수공원역 호반써밋', '856', 'https://www.hankyung.com/article/2026012633811 (2026-01-26)'],
             ['2026-04', '신검단중앙역 디에트르 더 에듀 / 롯데캐슬 넥스티엘', '781 / 372', 'https://news.jkn.co.kr/post/856868 (2026-03-23)'],
             ['2026-06', '신검단중앙역 칸타빌 더 스위트', '625', 'https://www.heraldk.com/article/2026052515415166618 (2026-05-25)'],
             ['2026-07', 'e편한세상 검단 웰카운티 / 에코비스타', '1,458 / 732', 'https://news.tf.co.kr/read/economy/2334911.htm (2026-06-22)'],
             ['2026-10(계획)', '신검단중앙역 센트레빌 / 제일풍경채 검단 3차', '1,224 / 610', 'LH 공고(2026-01-06), https://www.fnnews.com/news/202401110831045553 (2024-01-11)'],
         ]) + '\n',
         '- 입주 "월"은 입주 시작 시점이고 실제 입주는 이후 2~3개월에 걸친다. 단지와 역의 대응은 단지명으로만 추정했다. 공공임대는 집계에 없다.\n'
         '- 사용 방식: 역 3개·이력 1년으로는 입주 세대당 탄력성을 추정할 수 없어 모델 입력으로 쓰지 않았다. 다만 2026년 7월 입주(2,190세대)가 2025년 1월 이후 최대라는 점은 '
         '신규 역 성장 추세가 테스트 구간에도 이어진다는 가정(M4 성장 보정)을 뒷받침한다.\n',
         '## 3. 쓰지 않은 것\n',
         '- 마감 이후에 공개된 모든 정보(축제 운영 결과, 2026년 KBO 잔여 일정, K리그 일정 변경 등). 조사 중 본 마감 이후 항목은 이 폴더에 저장하지 않았다.\n'
         '- 정부 입주예정물량 단지별 파일(2026-02-27 공표)은 열어 보지 않았다.\n'
         '- 조사 원문 표는 `external/external_research_v7.md`에 있다(영문, 출처 URL·게시일 포함).\n']
    (OUT / 'EXTERNAL_DATA_AUDIT_V7.md').write_text('\n'.join(L))


def handoff_md(g):
    b = g['baseline']; fin = g['final_strict']; base = b['strict_nested']
    sv = json.loads((OUT / 'submission_validation.json').read_text())
    au = json.loads((OUT / 'MUNHAK_QUOTA_AND_DIFF_AUDIT.json').read_text())
    diff = au['diff_v7_minus_v6_log']; q = au['munhak_quota']
    conv = json.loads((OUT / 'CONVERGENCE_AND_MDN_REPORT.json').read_text())
    single = g['single']; dec = g['decisions']
    eq_adopted = 'eq' in g['adopted']
    d = lambda key, k='all_oof': f"{f5(base[k])} → {f5(single[key]['strict'][k])}"
    L = ['# ensemble_v7 인계 문서 (HANDOFF_V7)\n',
         '작성 기준: 2026-11-12, 로컬 검증만 사용. **이 문서의 모든 점수는 로컬 검증(OOF) RMSLE이며 리더보드 점수는 어떤 계산·선택·추정에도 쓰지 않았다**(단위 테스트로 강제). '
         'v1~v6의 코드·설정·산출물은 손대지 않았고(30개 v6 파일 SHA-256 대조 테스트), 새 작업은 `src/v7/`, `configs/ensemble_v7.json`, `outputs/ensemble_v7/`에만 있다.\n',
         '## 0. 한눈에 보기\n',
         f"- v7 = v6 파이프라인 + 채택 모듈({', '.join(single[m]['module'] for m in g['adopted'])}).\n"
         f"- 로컬 검증기(엄격, 중첩 LOFO): 전체 OOF **{f5(base['all_oof'])} → {f5(fin['all_oof'])}**, 가중 4-Fold {f5(base['weighted_4fold'])} → {f5(fin['weighted_4fold'])}, "
         f"Fold 3+4 {f5(base['fold_3_4'])} → {f5(fin['fold_3_4'])}, Fold 4 {f5(base['fold_4'])} → {f5(fin['fold_4'])}.\n"
         f"- 같은 비교를 기존 방식(저장된 앙상블 열)으로 하면 전체 OOF {f5(b['plain_saved_columns']['all_oof'])} → {f5(g['final_plain']['all_oof'])}.\n"
         f"- 제출본: `outputs/ensemble_v7/submission_ensemble_v7.csv` (157,194행, SHA-256 `{sv['sha256']}`), v6 대비 로그 RMSD {diff['log_rmsd_all']:.5f}, 바뀐 역 {diff['stations_changed']}개.\n"
         '- 로컬 검증으로 잴 수 없는 변경이 하나 있다: 송도맥주축제 2026 일정 반영(검증 fold에 해당 날짜가 없음). 지시서의 명시 지시와 마감 전 발표 확인에 따라 넣었다. 3절 참고.\n',
         table(METRIC_HEAD, [metric_row('v6 — 엄격(중첩 LOFO)', base), metric_row('v7 — 엄격(중첩 LOFO)', fin),
                             metric_row('v6 — v6 보고 방식(가중치를 전체 fold로 적합)', b['v6_convention_weights_on_all_folds']),
                             metric_row('v6 — 기존 방식', b['plain_saved_columns']), metric_row('v7 — 기존 방식', g['final_plain'])]) + '\n',
         '중첩 LOFO는 v6 문서의 "엄격한 방식"(0.2014)보다 더 엄격하다. v6 문서는 결합 가중치를 4개 fold 전체로 적합했고, 여기서는 fold마다 나머지 3개 fold로 적합했다. 그래서 같은 v6가 0.2014가 아니라 '
         f"{f5(base['all_oof'])}로 나온다. v7의 개선폭은 같은 방식끼리 비교한 값이다.\n",
         '## 1. 지시서 항목별 수행과 판정\n',
         table(['항목', '한 일', '로컬 측정(엄격, 전체 / Fold 2 / Fold 4)', '판정'], [
             ['M1.1 기상', 'Open-Meteo 재분석으로 2025-01-01~2026-06-30 일·시간 강수·기온 수집, 2015~2024 기후값 수집', 'M2에서 사용', '수집 완료, 제출본 미사용'],
             ['M1.2 송도맥주축제', '2025-08-22~30을 과거 행사로 등록. 2026년 일정(8/22~30)이 2026-06-23 발표된 것을 확인해 예정 행사로 등록', f"fold에 해당 행 없음({d('beer')})", '**채택**(지시서 명시)'],
             ['M1.3 대학 학사일정', '인천대·경인교대 2025·2026 일정 조사', '—', '미사용(인천대 2학기 게시일 미확인, 경인교대는 전년 정렬과 일치)'],
             ['M1.4 검단 입주 물량', '2025-01~2026-10 월별 단지·세대 조사', '—', '모델 입력 아님, M4의 방향 근거'],
             ['M2 강수 g-계산 / 비 온 날 흔적', '비 효과를 기준일 이전 자료로 추정해 (기후 비 비율 − 기준 구간 비 비율)만큼 보정. 계수 적합형과 이론 계수형 두 가지', f"적합형 {d('rain')} / {d('rain', 'fold_2')} / {d('rain', 'fold_4')}; 이론형 {d('rain0')}", '기각'],
             ['M2 v2 주간 형태 전이', '이전 세션 측정: v2 Fold 2 0.1804→0.1806, Fold 4 0.1845→0.1848', '—', '기각'],
             ['M3 평형 기준선 모델군', '7개 계열(+변형)을 RTX 4090에서 6개 기준일로 교차검증, v2·v4·v5와 볼록 결합', (d('eq') + ' / ' + d('eq', 'fold_2') + ' / ' + d('eq', 'fold_4')) if 'eq' in single else '측정 중', ('**채택**' if eq_adopted else '기각') if 'eq' in single else '—'],
             ['M4 신규 역 성장 추세(3107~3109)', '세 역 공통의 최근 8주 상대 성장 기울기를 예측거리만큼 연장(적합 계수 없음)', f"{d('growth')} / {d('growth', 'fold_2')} / {d('growth', 'fold_4')}", '**채택**'],
             ['M4 기존 역 성장 복원(James-Stein)', '기존 68개 역에 같은 방식(창 56~240일) 적용', '창 56일: Fold 4 악화(0.1695→0.1705, 기존 역 행). 창 120일: 네 fold 모두 0.0006 이하 개선이나 Fold 2는 0.00001', '기각'],
             ['M4 세그먼트별 결합 가중치', '3107~3111 별도 가중치(이전 세션)', '개선 없음', '기각'],
             ['M5.1 문학 미발표일 쿼터 정규화', '10경기로 정규화 / 월요일 분리 두 가지를 Fold_Summer25·Fold 1에서 검증', f"Summer25 미발표일 행: v6 {f4(q['validation_summer25']['v6']['rmsle_unannounced_rows'])}, 10경기 {f4(q['validation_summer25']['quota_scaled_to_10_games']['rmsle_unannounced_rows'])}, 월요일 분리 {f4(q['validation_summer25']['v7_monday_split']['rmsle_unannounced_rows'])}", '둘 다 기각'],
             ['M5.2 부천 유령 날짜', '3754에서 전년 경기일 +364일의 급등 제거를 축구에도 적용', f"{d('bucheon')} (56행)", '기각'],
             ['M5.3 최근 구간 급등 오염', '최근 8주 평균−중앙값 차이로 수준 보정', f"{d('gap')} / {d('gap', 'fold_2')} / {d('gap', 'fold_4')}", '기각'],
             ['M5.4 공원 역 맑은 날 곡선', 'v6 제출본의 인천대공원 가을 상승폭을 2025년 실제와 대조', '예측: 7월 대비 9월 +0.18, 10월 +0.48 / 2025년 실제 +0.27, +0.40', '별도 모듈 불필요(문제는 4월 벚꽃철이고 테스트 구간 밖)'],
             ['M6 보조 fold', 'Fold_Hot26, Fold_Summer25에서 평형 모델·성장 보정·문학 쿼터 측정', 'RESULTS_V7 5절', '완료(v2·v4·v5 예측이 없어 최종 결합본 자체는 평가 불가)'],
         ]) + '\n',
         '## 2. 지시서 내용 중 확인 결과가 달랐던 것\n',
         table(['지시서', '확인 결과'], [
             ['문학 미발표 55일의 기대 15.04경기는 과대이고 9~11경기가 맞다', f"콘서트 2일을 빼면 미발표일은 53일, v6 기대 관중일은 {q['test_window']['v6']['expected_crowd_days']}일이다. 2025년에는 같은 조건(기준일 2025-06-30, 9~10월 61일)에서 v6 기대 {q['validation_summer25']['v6']['expected_crowd_days']}일, 실제 {q['validation_summer25']['v6']['actual_crowd_days']}일로 거의 일치했다. 10경기로 줄이면 그 구간 RMSLE가 나빠진다. 2025년 잔여 일정에는 홈 16경기가 있었고(미편성 3 + 우천 순연), 2026년은 미편성만 6경기다"],
             ['월요일을 화~금과 분리', f"2025년 9~10월 월요일 9일 중 2일에 경기가 있었다(잔여 일정의 예비일). 분리하면 Fold 1은 좋아지지만 Fold_Summer25가 나빠진다({f4(q['validation_summer25']['v6']['rmsle_unannounced_rows'])} → {f4(q['validation_summer25']['v7_monday_split']['rmsle_unannounced_rows'])})"],
             ['v5 성장 항이 Fold 3·4에서 켜져 있다', 'Fold 3(기준일 2025-12-30)은 꺼져 있다. v5 코드는 기준일이 2026-01-11 이전이면 0을 돌려준다'],
             ['송도맥주축제 2026은 마감 전 발표가 없다(v6 판단)', '2026-06-23 경인방송 기사에 8월 22~30일·송도달빛축제공원이 명시돼 있다. 2025년 급등의 +364일(8/21~29) 가운데 축제가 없는 날은 8/21 하루뿐이다'],
             ['조기 종료 경보의 원인은 학습·평가 행의 기준선 불일치', '기준선은 모든 행이 같은 함수·같은 정의다. 원인은 목표 계절이 학습 구간에 없는 것이다(Fold 1~3). 계절이 학습에 있는 Fold_Hot26에서는 그래프 모델의 경보가 사라진다'],
             ['모든 모듈을 Fold 2·Fold 4 동시 개선으로 판정', '맥주축제·문학 월요일처럼 Fold 2·4의 행을 바꾸지 않는 모듈은 그 규칙으로 잴 수 없다. 이 경우 바뀐 fold와 보조 fold로 판정하고 표에 그대로 적었다'],
             ['신규 역은 상태공간 모형으로', '역 3개·이력 1년이라 상태공간 모형의 분산 모수를 추정할 표본이 없다. 같은 취지(세 역을 묶은 공통 추세의 연장)를 적합 계수 없는 식으로 구현했다'],
         ]) + '\n']
    gr = sv.get('growth') or {}
    L += ['## 3. 채택한 모듈\n',
          '### 3.1 신규 역 성장 추세 (M4)\n',
          f"- 발견: v6는 2025-06-28 개통한 3개 역(3107 검단호수공원, 3108 신검단중앙, 3109 아라)을 네 fold 모두에서 낮게 예측했고 예측거리가 멀수록 차이가 커졌다(예측거리 1~41일 / 42~82일 / 83~123일 평균 편향: Fold 1 −0.10/−0.19/−0.23, Fold 2 −0.09/−0.15/−0.21, Fold 3 −0.02/−0.10/−0.21, Fold 4 −0.05/−0.12/−0.13). 이 3개 역이 v6 오차 제곱합의 8.9%였다.\n"
          '- 원인: v2·v4·v5 모두 최근 수준에 고정돼 있어서, 입주가 진행 중인 역의 상승 추세를 따라가지 못한다.\n'
          '- 식: `보정(역, h) = 기울기 × (28 + h)`. 기울기는 세 역 각각의 (역 일 총량 로그 − 기존 역 평균)을 최근 56일 평일에 대해 직선 적합한 값의 **중앙값**(세 역 공통), 28은 8주 평균이 갖는 수준 지연, h는 예측거리(일). 상한 0.25. 적합한 계수는 없다(참고용 최소제곱 계수는 fold별 a≈7~37, b≈0.9~1.4로 이 값과 비슷했다).\n'
          '- 개통 111일 미만이면 보정하지 않는다. Fold 1 기준일에는 개통 63일째였고, 그때 기울기는 부호가 틀려 강제로 적용하면 신규 역 RMSLE가 0.32에서 0.40~0.43으로 나빠졌다.\n'
          f"- 로컬 측정(엄격): 전체 {d('growth')}, Fold 2 {d('growth', 'fold_2')}, Fold 3 {d('growth', 'fold_3')}, Fold 4 {d('growth', 'fold_4')}, Fold 1 미적용. 역 3107: {f4(base['st_3107'])} → {f4(single['growth']['strict']['st_3107'])}.\n"
          + (f"- 테스트 적용: 공통 기울기 {list(gr['slopes'].values())[0]:.5f}/일, {gr['rows']:,}행, 평균 로그 보정 +{gr['mean_log_correction']:.3f}, 최대 +{gr['max_log_correction']:.3f}(10월 31일).\n" if gr else '')
          + '- 기반 모델 없이 한 점검(이동평균형 예측의 신규 역 상대 편향, 공휴일 제외): '
          + '; '.join(f"{FOLD_NAME[k]} {v['relative_bias_before']:+.3f} → {v['relative_bias_after']:+.3f} (RMS {v['relative_rms_before']:.3f} → {v['relative_rms_after']:.3f}, 기울기 {v['common_slope_per_day']:.5f})" for k, v in g['growth_proxy_check'].items() if v.get('applied')) + '.\n'
          '- 중앙값을 쓰는 이유: 2026-05-15 기준일에는 3107(검단호수공원)의 봄철 공원 수요가 그 역 기울기를 0.0044/일로 부풀려, 세 역 평균을 쓰면 이후 46일을 과대 보정했다(상대 편향 RMS 0.050 → 0.058). 중앙값은 그 기준일에서도 개선된다.\n'
          f"- 평형 트리 모델을 기반으로 한 Fold_Hot26에서는 신규 역이 {f4(g['aux_folds']['hot26']['new_stations']['before'])} → {f4(g['aux_folds']['hot26']['new_stations']['after'])}로 조금 나빠진다. 평형 모델은 추세 특징을 갖고 있어 성장 편향이 작기 때문이며, 이 보정은 v2·v4·v5 기반(최근 수준 고정형)에만 쓴다.\n"
          '- 위험: 성장 속도는 계속 줄고 있다(기울기 0.0013 → 0.0011 → 0.0010 → 0.0006 → 테스트 기준일 0.0009). 성장이 완전히 멈추면 평균 +0.08의 과대 예측이 된다. 반대 근거로 2026년 7월 입주 2,190세대(2026-06-22 발표)가 있다.\n',
          '### 3.2 송도맥주축제 2026 (M1.2)\n',
          '- 근거와 방식은 EXTERNAL_DATA_AUDIT_V7 2.2절. 역 3139의 2026-08-22~30 예측이 22시 기준 80~210명에서 610~3,500명으로 바뀐다(2025년 실제 1,016~4,230명).\n'
          '- 로컬 검증으로 잴 수 없다. v6가 펜타포트(2025년 실측으로 2026년 상승폭 추정)를 다룬 것과 같은 방식이며 지시서가 무조건 반영을 지시했다. 축제가 취소·이전됐다면 이 9일의 해당 시간대는 큰 과대 예측이 된다.\n'
          '- 부수 효과: 2025년 축제일이 "평소 날" 표본에서 빠지면서 펜타포트 2026 상승폭 추정이 조금 달라진다(8월 1일 22시 3,399명 → 3,634명).\n']
    if 'eq' in single:
        es = g.get('eq_scores', {})
        best = sorted(es.items(), key=lambda x: x[1]['strict']['all_oof'])[:4]
        L += ['### 3.3 평형 기준선 모델군 (M3)' + (' — 채택\n' if eq_adopted else ' — 기각\n'),
              '- 기준선: `mu_eq(역, 시간, 유효요일)` = 기준일 이전 70일 안의 같은 유효요일(월~금 각각, 토, 일·공휴일) 최근 8일의 중앙값. 모든 행의 학습 목표는 `r_eq = log1p(y) − mu_eq`.\n'
              '- 특징 48개를 기준일마다 한 번의 벡터 계산으로 만든다(기준일 하나에 0.03~0.07초). 수준·추세·최근 편차, 달력(기준일에 알려진 공휴일), 문학 일정 상태·행사 표시, 전년 같은 요일(±1주 중앙값) 기반 계절·성장 특징.\n'
              '- 학습 표본: 기준일에서 7일 간격으로 거슬러 올라간 가상 기준일들(정답이 기준일 이전인 것만), 약 290만 행.\n'
              f"- 개별 성능(엄격 전체 OOF 상위): " + ', '.join(f"{model_label(m)} {f5(r['strict']['all_oof'])}" for m, r in best) + '. v2·v4·v5보다 나쁘다(RESULTS_V7 6절).\n'
              f"- 결합 게이트: 전체 {d('eq')}, Fold 1 {d('eq', 'fold_1')}, Fold 2 {d('eq', 'fold_2')}, Fold 3 {d('eq', 'fold_3')}, Fold 4 {d('eq', 'fold_4')}. 판정: {dec['eq']['reason']}\n"
              '- 다른 결합 방식도 모두 기각이다: 트리 5종 평균 하나만 넣기(0.20340 → 0.20447), 계절 변형 3종 평균(→ 0.20393), 기본 3종 평균(→ 0.20466), xgb 계절 변형 하나(→ 0.20354), 10종 평균(→ 0.20503). 모두 Fold 3·4가 나빠진다.\n'
              '- 최종 모델 10개의 테스트 예측은 모두 v6보다 높다(행사 역 제외 월평균 로그 +0.01~+0.08, RESULTS_V7 9절). 그러나 검증 fold에서는 평형 모델의 전체 수준이 v6보다 더 낮게 치우쳤고(Fold 4 평균 편향: 평형 트리 −0.084, v6 −0.047), 두 예측의 일별 전체 수준 차이를 v6에 일부 반영하는 보정도 중첩 LOFO에서 0.20214 → 0.20201~0.20271(Fold 2 악화)이었다. 이 신호는 제출본에 쓰지 않았다.\n'
              f"- 수렴: 조기 종료 경보 {len(conv['early_stop_alarms'])}건(트리 200회 미만 또는 신경망 5에폭 미만). 원인·조치·남은 경보는 `CONVERGENCE_AND_MDN_REPORT.json`의 alarm_diagnosis. 모델별 학습 시간 합계 {conv['gpu_minutes_logged']}분(병렬 실행), 포드 실행 시간 {conv['wall_clock']['minutes']}분(예산 120분).\n"]
    L += ['## 4. 기각한 모듈의 수치\n',
          table(['모듈', '전체', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '사유'],
                [[single[k]['module']] + [d(k, x) for x in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + [dec[k]['reason']] for k in single if not dec[k]['adopt']]) + '\n',
          '- 강수 적합형의 계수는 기후 항이 −2.7~−3.8로 이론값(+1)과 부호가 반대다. 비 효과가 아니라 월별 계절 차이를 대신 맞춘 것이어서, Fold 4가 좋아져도 채택하지 않았다.\n'
          '- 부천: 3754의 전년 경기일 흔적은 v6 예측에 로그 0.1~0.2 정도만 남아 있고, 제거하면 오히려 조금 나빠졌다.\n',
          '## 5. 제출본과 v6의 차이\n',
          f"- 바뀐 행 {diff['rows_changed']:,}개, 그중 |로그 차이| > 0.05는 {diff['rows_changed_over_0_05']:,}개. 평균 승차 {diff['mean_boardings_v6']:.2f} → {diff['mean_boardings_v7']:.2f}명.\n"
          '- 월별 평균 로그 차이: ' + ', '.join(f"{k}월 {v['mean']:+.4f}" for k, v in diff['by_month'].items()) + '.\n'
          '- 역별·시간대별 표는 `MUNHAK_QUOTA_AND_DIFF_AUDIT.json`의 diff_v7_minus_v6_log, 문학 쿼터 전후와 요일별 확률표는 같은 파일의 munhak_quota.\n',
          '## 6. 재현 방법\n',
          '```bash\n# (포드) 평형 모델군: 교차검증 -> 엄격 예측 정리 -> 최종 2개 시드\npython -m src.v7.train_eq cv --models lgb,lgb_s\npython -m src.v7.train_eq cv --models xgb,xgb_s,cb,cb_s,tabm,mdn,stg,rkhs\npython -m src.v7.train_eq finalize\npython -m src.v7.train_eq final --models <결합 가중치가 0보다 큰 모델>\n'
          '# (로컬) 게이트 -> 제출본 -> 문서 -> 테스트\npython -m src.v7.munhak_audit\npython -m src.v7.gates\npython -m src.v7.build\npython -m src.v7.report && python -m src.v7.handoff\npython -m src.v7.tests.test_v7\n```\n',
          '## 7. 파일\n',
          table(['파일', '내용'], [
              ['submission_ensemble_v7.csv', '단일 통합 제출본(157,194행)'], ['RESULTS_V7.md', '로컬 검증 수치 전체'], ['ABLATION_LOFO.csv', '모듈별 단독·누적: 엄격 LOFO, 기존 방식, Fold 1~4, 역 7개'],
              ['gates.json', '게이트 원자료(적합 계수, 보조 fold, 판정 사유)'], ['EXTERNAL_DATA_AUDIT_V7.md', '외부 자료 출처·공개일·SHA-256'],
              ['CONVERGENCE_AND_MDN_REPORT.json', '모델별 학습 곡선, 검증 RMSLE, 기울기 노름, MDN 성분, 경보 진단'],
              ['MUNHAK_QUOTA_AND_DIFF_AUDIT.json', '문학 쿼터 전후·요일 확률표·검증, v6 대비 월·역·시간 차이'],
              ['submission_validation.json', '제출본 SHA, 결합 가중치, 성장 기울기, 입력 파일 SHA'], ['domain_schedule_v7.json', 'v6 일정 + v7 추가 행사(자동 생성)'],
              ['eq/', '평형 모델군의 fold별 예측(npz)과 학습 기록(json)'], ['external/', '기상 원자료와 조사 원문'],
              ['src/v7/tests/test_v7.py', '리더보드 값 금지, v6 보존, 기준일 이후 자료 변조 불변, 공개일 필터, 제출본 무결성']]) + '\n',
          '## 8. 남은 판단\n',
          '1. **맥주축제 반영을 유지할지.** 뺄 때는 `configs/ensemble_v7.json`의 `adoption_overrides`에서 beer를 지우고 `python -m src.v7.gates && python -m src.v7.build`.\n'
          '2. **성장 보정의 세기.** 지금은 추세가 그대로 이어진다고 본다(b=1). 보수적으로 하려면 `src/v7/growth.py`의 `PERSISTENCE`를 낮춘다. 로컬 검증 3개 fold는 1 안팎을, 2026-05 점검은 0.4를 가리킨다.\n'
          '3. 제출은 하지 않았다.\n']
    (OUT / 'HANDOFF_V7.md').write_text('\n'.join(L))


def main():
    g = json.loads((OUT / 'gates.json').read_text())
    audit_md(g); handoff_md(g)
    print('written: EXTERNAL_DATA_AUDIT_V7.md, HANDOFF_V7.md')


if __name__ == '__main__':
    main()
