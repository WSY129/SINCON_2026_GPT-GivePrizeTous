# 재현 안내 (submission_ensemble_v20e.csv 기준, 2026-11-13 작성)

> 점수는 적지 않는다. 이 문서는 "같은 파일이 다시 만들어지는가"만 다룬다.

## 1. 한눈에

| 항목 | 상태 |
|---|---|
| 자료 경로 | 전부 프로젝트 폴더 안(`data/raw`, `outputs/ensemble_vN/external`, `configs`). 코드에 절대 경로·`Downloads` 참조 없음 |
| 패키지 | `reproduce/requirements.txt` (핵심 10개 고정), `requirements_full.txt` (전체 61개). Python 3.12.14 |
| 산출물 해시 | `reproduce/MANIFEST.csv` (1,034개 파일: 입력 자료, 판별 제출본·검증 예측·모델 예측·설정) |
| 실행 순서 | `reproduce/run_all.sh` (v2 → v20e, 단계별 명령) |
| 재실행 검증 | `python -m reproduce.verify_from_stored --retrain` → `verify_result.json` |

**앞서 `outputs/ensemble_v19/EXTERNAL_SOURCES.md`에 "팀 공유 자료를 `~/Downloads`에서 읽는 부분이 있다"고 적은 것은 틀렸다.** 다시 확인하니 제출본을 만드는 코드는 모두 프로젝트 안의 사본만 읽는다(`Downloads`를 읽은 것은 일회성 분석 스크립트뿐이고, 쓰인 표는 `outputs/ensemble_vN/external`에 복사돼 있다).

## 2. 직접 확인한 것 (이 컴퓨터, Apple M4 Pro, 12스레드)

| 다시 돌린 단계 | 결과 |
|---|---|
| v10 (`python -m src.v10.run`, 2초) | 제출 파일 SHA-256 동일 |
| v15 스택·보정 (`src.v15.run`) | 제출 파일·검증 예측 동일 |
| v19 보정 (`src.v19.run`, 다시 만든 v15 위에서) | 제출 파일 동일 |
| v20e (`src.v20.extras`, 다시 만든 v19 위에서) | 제출 파일 동일 (`1a5dbd7c…5c69`) |
| LightGBM 재학습 1건 (lgb_h, Fold 4, 시드 2026, 12스레드) | 저장된 예측과 완전히 동일 |
| XGBoost 재학습 1건 (xgb_i, Fold 4, 시드 2026) | 완전히 동일 |
| 같은 두 모델을 **4스레드**로 재학습 | XGBoost는 동일. **LightGBM은 다름**(예측 로그값 차이 RMS 0.0032, 최대 0.08) |

- 즉 시드·패키지·스레드 수가 같으면 이 컴퓨터에서는 비트 단위로 재현된다.
- **LightGBM은 스레드 수에 따라 결과가 조금 달라진다.** 코랩에서도 `V11_THREADS=12`를 그대로 두고 돌려야 한다(코어가 적어도 설정값은 12로). 그래도 CPU 종류(ARM ↔ x86)가 달라 완전히 같다고 보장할 수는 없다. 이 부분은 다른 컴퓨터에서 확인하지 못했다.

## 3. 단계와 의존 관계

| 단계 | 명령 | 장비 | 걸린 시간(기록) | 재실행 확인 |
|---|---|---|---|---|
| v2, v4, v5 기반 모델 | `src.pipeline v2 run`, `src.v4.pipeline run`, `src.v5.pipeline run` (+ `infer`) | 신경망(TabM, 시공간 트랜스포머)은 GPU | v4 약 1.7시간, v5 약 0.7시간(작업자 합), v2 기록 없음 | 안 함 |
| v6~v8 규칙 계층 | `src.v6.build`, `src.v7.build`, `src.v8.run both` | CPU | 각 수 초~수 분 | 안 함 |
| v9 모델 | `src.v9.train …`, `src.v9.blend` | LightGBM은 CPU, 나머지 GPU | 0.6시간 | 안 함 |
| v10 | `src.v10.run` | CPU | 2초 | **동일** |
| v11 모델 11종 | `src.v11.train …`, `src.v11.stack` | GPU 장비 | 5.4시간 | 안 함 |
| v12~v14 | `src.v12.run`, `src.v13.run`, `src.v14.train`, `src.v14.run` | CPU | 학습 0.2시간 | 안 함 |
| v15 학습 | `src.v15.train …` (LightGBM 15회, XGBoost 30회) | CPU 12스레드 | 0.6시간 | 표본 2건 **동일** |
| v15 스택 → v19 → v20e | `src.v15.run`, `src.v19.run`, `src.v20.extras` | CPU | 약 2분 | **동일** |

처음부터 다시 만들 때의 연산량은 합쳐서 10시간 안팎으로, 코랩에서 가능한 규모다.

## 4. 최종 파일이 GPU 학습 결과에 기대는 부분

- **v10 안의 기반 예측**: v6가 v2·v4·v5 제출본을 섞어 만들고, 그 위에 v7·v8·v10이 쌓인다. v2·v4·v5에는 GPU로 학습한 신경망이 들어 있다. 스택에서 v10의 가중은 약 0.48이다.
- **v9**: v10 결합에서 가중을 받는 것은 CPU로 학습한 LightGBM(0.18~0.28)과 CatBoost 0.02(한 구간)뿐이다.
- **v11**: 최종 스택 가중은 v10 0.477, v15 LightGBM 0.345, v15 XGBoost 0.178이다. v11의 GPU 모델들은 전체 가중을 받지 못하고, 부천·공원 역의 보조 모델로만 일부 쓰인다.

GPU 학습은 시드를 고정해도 장비·드라이버가 다르면 값이 조금 달라질 수 있다. 따라서 **v2·v4·v5·v9·v11을 처음부터 다시 학습하면 최종 파일이 비트 단위로 같지는 않을 것**이다. 저장된 예측(`outputs/ensemble_v2|v4|v5`, `v9/models`, `v11/models`)을 입력으로 쓰면 그 뒤는 CPU만으로 같은 파일이 나온다.

## 5. 코랩에서 돌리는 순서

1. 프로젝트 폴더 전체(코드, `data/raw`, `configs`, `outputs`)를 올린다. `outputs`는 약 7.4GB다.
2. `pip install -r reproduce/requirements.txt` (Python 3.12).
3. `export V11_THREADS=12`
4. 저장된 모델 예측에서 시작해 확인: `python -m reproduce.verify_from_stored --retrain` (약 5분). `identical: true`가 나와야 한다.
5. 처음부터 전부: `bash reproduce/run_all.sh` (GPU 런타임 필요, 약 10시간). 이 스크립트는 한 번에 끝까지 돌려 본 적이 없다.

## 6. 아직 안 한 것

- v2~v9, v11~v14 단계의 재실행 확인(특히 GPU 학습 단계).
- 다른 컴퓨터(x86, 코랩)에서의 확인.
- 버스 자료(`outputs/ensemble_v20/external`)는 사이트 화면에서 조회한 값을 옮겨 적은 것이다. 조회 절차는 그 폴더의 README에 있고, 자동 내려받기 스크립트는 없다.
