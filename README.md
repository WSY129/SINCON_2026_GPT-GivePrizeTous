# SINCON 2026 제출 자료 — 인천 지하철 시간대별 승차 예측

최종 예측 = **팀 앙상블 모델 v20e** (`model/`) → **사후 조정 10단계** (`postprocess/`) → `predictions/submission_final.csv`

| 경로 | 내용 (안내문 항목) |
|---|---|
| `predictions/submission_final.csv` | **최종 예측 파일** (리더보드 제출 원본). 같은 파일의 원래 이름본도 함께 둠. SHA-256 `65cbacb7e9d994706e26240a100449f65ac9b0f4e0d9857d00fa277713a9bc88` |
| `model/` | **전체 실행 코드 (모델)**: 팀 저장소 스냅숏 (commit `8c89a1c`). 특징·라벨 생성, 검증, 학습, 앙상블, 보정 코드와 저장된 모델 예측(npz), 재현 스크립트. 설명서는 `model/README.md`, 재현은 `model/reproduce/REPRODUCE.md` |
| `postprocess/` | **전체 실행 코드 (사후 조정) + 외부 데이터**: `analysis/` 프로그램, `data/external/` 수집 자료 45개, 실행 순서 `analysis/run_postprocess_chain.sh` |
| `records/` | **데이터 출처·생성 기록, 설명서, 실행 로그, 개발 기록** |

---

## 1. 최종 예측 파일과 코드의 관계
1. `model/` → `model/outputs/ensemble_v20/submission_ensemble_v20e.csv` (SHA-256 `1a5dbd7c…`, `model/README.md` 0 · 5 · 6절)
   - 기준선(직전 수준 + 여러 해 계절 곡선 + 휴일 + 신규 역 성장) 위에서 트리 모델이 로그 잔차를 학습
   - 앙상블 실효 가중: LightGBM 0.345 · XGBoost 0.178 · TabM 0.179 · CatBoost 0.125 · LightGBM(v9) 0.086 · 기타 0.09 (`model/README.md` 5.3절)
   - 보정 단계(행사·방학·저수요·휴일) → v19 → v20e(평가 기간 행사 항목, 버스 실측으로 크기 조정)
2. `postprocess/analysis/run_postprocess_chain.sh` → v20e(+ v10)에서 최종 파일까지 10단계 (모두 결정적, 난수 없음)

| # | 프로그램 | 바꾼 것 | 바뀐 칸 |
|---|---|---|---|
| 1 | `make_0926up.py` | 9/26(추석 다음날 토) 06~08·17~21시 = max(v20e, v10) — 버스 실측 근거 | 503 |
| 2 | `adjust_0926_2019.py 0.5` | 9/26 아침·저녁을 2019 실측(같은 요일 배치) 쪽으로 α 0.5 | 700 |
| 3 | `adjust_july_weekday.py 1.02` | 7월 평일 ×1.02 (검단 연장 영향 역·문학 제외) | 25,740 |
| 4 | `adjust_chuseok_days_2019.py 0.5` | 9/23 낮·9/27 아침 (2019 실측) | 767 |
| 5 | `adjust_ssg_0719.py` | SSG 경기 날 문학경기장 추가 승차 (관중 × 과거 관계), 7/19 연휴 끝 | 1,589 |
| 6 | `adjust_bfc_0711.py` | 부천FC 경기 날, 7/11 흠뻑쇼 | 134 |
| 7 | `adjust_oct_festivals.py` | 10월 축제 3개 (부평풍물·국화축제·퀴어축제) | 177 |
| 8 | `adjust_autumn_night.py` | 9~10월 20~23시 비중 | 13,124 |
| 9 | `adjust_exam_days.py 0.5 3208:0.75` | 학교 시험 날 학교 앞 7개 역 12~13시 ↑ 16~17시 ↓ (학교 홈페이지 시험 일정) | 401 |
| 10 | `adjust_kopis.py` | 8/29 문학 20~22시 ×0.6, 10/31 남동구청 20~22시 ×1.4 (KOPIS 공연 예매 수) | 6 |

- 각 단계의 근거·계수 결정 방법: `records/postprocess_adjustments_README.md` (= `postprocess/outputs/v20e_0926v10/README.md`)
- `analysis/` 안의 `adjust_bul1031.py`, `adjust_aug_weekday.py`, `adjust_school_vacation.py`는 **최종 파일에 쓰지 않았음** (해당 없음).

## 2. 실행 환경
| 부분 | 환경 |
|---|---|
| 모델 (`model/`) | macOS, Apple M4 Pro (12스레드) + RunPod GPU 포드(RTX 3090, RTX PRO 6000, A100, RTX 4090 등). Python 3.12, `model/reproduce/requirements.txt` (lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numpy 2.3.5, pandas 2.2.3, torch 2.15.0 …), 전체 목록 `requirements_full.txt` |
| 사후 조정 (`postprocess/`) | macOS (Apple M4 Pro), Python 3.14.6, pandas 3.1.1, numpy 2.5.4 (`postprocess/requirements.txt`). CPU만, 약 1분 |

- 시드: 모델 시드 2026 ~ 2030 (모델별, `model/README.md` 0 · 5절). LightGBM 결과는 스레드 수에 따라 달라지므로 `V11_THREADS=12`. 사후 조정은 난수 없음.

## 3. 실행 방법
```bash
# (1) 모델: 저장된 예측에서 v10 · v15 · v19 · v20e 재생성 + SHA-256 비교 (약 2분)
cd model && python3.12 -m venv .venv && .venv/bin/pip install -r reproduce/requirements.txt
export V11_THREADS=12
.venv/bin/python -m reproduce.verify_from_stored          # 처음부터 전체: reproduce/run_all.sh (GPU 단계 포함, 10시간 안팎)
# 대회 제공 파일(train.csv, test.csv, sample_submission.csv, metaData.csv)은 공개 저장소에 없음 → Kaggle 원본을 model/data/raw/ 에 넣기

# (2) 사후 조정: v20e → 최종 파일 (약 1분)
cd ../postprocess && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && cp ../model/data/raw/{train,test,metaData,sample_submission}.csv data/
PY=.venv/bin/python bash analysis/run_postprocess_chain.sh \
  ../model/outputs/ensemble_v20/submission_ensemble_v20e.csv \
  ../model/outputs/ensemble_v10/submission_ensemble_v10.csv  out
shasum -a 256 out/submission_final.csv     # 65cbacb7… 이면 predictions/submission_final.csv 와 같음
```

## 4. 재현 결과 (실행 로그)
- 사후 조정: 2026-11-13 21시대에 위 (2)를 실행해 **최종 파일과 SHA-256이 같음**을 확인 → `records/postprocess_chain_run_log.txt`
- 모델: 팀 저장소에서 `verify_from_stored`로 v10 · v15 스택 · v19 · v20e가 비트 단위로 같음을 확인 (Apple M4 Pro, 12스레드), MANIFEST 1,034개 파일 해시 일치 → `model/reproduce/verify_result.json`, `model/README.md` 7절
- 재현 변동: LightGBM을 다른 스레드 수(예: 4)로 돌리면 로그 차이 RMS 0.0032. GPU XGBoost(xgb_g11)는 CUDA 필요.

## 5. 데이터 출처 및 생성 기록
- 제공 자료: train.csv · test.csv · metaData.csv · sample_submission.csv (변형 없음)
- 외부 자료 전체 목록·출처 URL·수집 방법·생성 경로: **`records/데이터_공유_README.md`** (1 ~ 62번 파일, 5 · 6절) — 각 파일을 만든 프로그램은 `postprocess/analysis/fetch_*.py`, `build_*.py`
- 모델 쪽 외부 자료: `model/README.md` 2.2절 (`model/outputs/ensemble_vN/external/`에 사본)
- 원본 다운로드(약 450MB: 과거 승하차 원본, 공연 원본 등)는 용량 때문에 뺐고, `fetch_*.py`로 다시 받을 수 있음. 스크립트가 직접 읽는 작은 원본(`raw/neis`, `raw/osm`, `raw/school_web`)은 포함.
- 인증키가 필요한 수집 프로그램은 환경변수(`DATA_GO_KR_KEY`, `NEIS_KEY`)로만 키를 읽고, 키는 제출물에 없음.

## 6. 대회 규정 관련 사실 기재
- **2026-07-01 ~ 10-31 지하철 승하차·통행량 자료는 어디서도 받거나 열거나 쓰지 않았음.** 과거 승하차는 2026-06-30 이전분만 (공공데이터포털 과거판, 인천교통공사 공개자료). 수집 프로그램도 2026-06까지만 받도록 막음.
- 평가 기간의 날씨·경기 결과·관중·행사·버스 이용량·공연 예매 수는 규정상 허용 자료로 사용함.
- **리더보드 사용**: 모델의 가중·보정 계수는 리더보드로 정하지 않았음(`model/README.md`). 사후 조정은 각 단계의 계수를 과거 자료로 먼저 정했고, 일부 단계는 Public 점수를 보고 **채택 여부**를 정했음: 7월 평일 ×1.02 · 시험 날 조정 = 채택 / 10/31 불꽃야구 · 8월 평일 ×1.02 = Public이 나빠져 버림 (최종 파일에 없음). 기록: `records/postprocess_adjustments_README.md`.

## 7. 설명서 · 개발 기록
- 모델 설명서(학습·검증 분할·주요 변수·모델 선택·앙상블 비중·후처리): `model/README.md`, `model/SINCON_PIPELINE.md`, `model/reports/`, 버전별 `model/outputs/ensemble_vN/RESULTS*.md`
- 검증: 시간순 4-fold (원점 2025-08-30, 10-30, 12-30, 2026-02-27, 각 123일). 전체 OOF가 좋아지고 어느 폴드도 나빠지지 않을 때만 채택.
- 외부 자료 × 승차 분석: `postprocess/analysis/분석_보고서.md`, `postprocess/outputs/analysis/`
- 사후 조정 단계별 근거·버전 이력: `records/postprocess_adjustments_README.md`
- v21(팀원 후속 버전)은 최종 파일에 쓰지 않아 용량상 이 묶음에서 뺐음 (팀 저장소에 있음).
