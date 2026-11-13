# 코랩 단일 노트북 전환: 외부 파일 의존과 해시 대조 (2026-11-13 KST)

> 점수는 다루지 않는다. "같은 파일이 다시 만들어지는가"와 "무엇이 있어야 돌아가는가"만 적는다.

## 1. 요약

| 항목 | 상태 |
|---|---|
| 해시 목록 `reproduce/MANIFEST.csv` (1,034개 파일, 1.17GB) | 2026-11-13 다시 대조: **1,034개 전부 일치**. 누락·불일치 0개 (`manifest_check_20261113.csv`) |
| 끝단(v15 스택 → v19 → v20e) 재실행 | 2026-11-13 11:31 KST 재실행. v15·v19·v20e 제출 파일 SHA-256 **동일** (`1a5dbd7c…5c69`) |
| 끝단이 실제로 여는 입력 파일 | **229개**를 실행 중 감사 훅(`sys.addaudithook`)으로 기록했다. 225개는 MANIFEST와 해시가 일치한다. 나머지 4개는 MANIFEST 밖이며 시스템 파일 2개와 `reproduce/` 코드 2개다. 목록: `traced_inputs_cpu_tail.json`, `traced_inputs_vs_manifest.csv` |
| 프로젝트 밖 파일 | 시스템 파일 2개(macOS 버전 plist, 시간대 UTC 파일)뿐. `~/Downloads` 등 사용자 폴더 참조는 없음 |
| 끝단 입력 용량 | 약 550MB (모델 예측 npz 176개 약 440MB, 대회 자료 22MB, 외부 표 약 15MB, 이전 단계 OOF·제출 파일) |
| GitHub 저장소 | 위 입력과 코드 전부를 비공개 저장소 `taeyang0505/SINCON`에 올렸다(약 640MB). 코랩에서 `git clone`만으로 끝단 검증 가능 |

## 2. 끝단이 읽는 파일 (단계별)

| 묶음 | 파일 수 | 용량 | 성격 |
|---|---|---|---|
| `data/raw/` (train, test, metaData) | 3 | 22.4MB | 대회 자료 |
| `configs/` (domain_schedule_v6, ensemble_v7/v10/v11/v12/v20) | 6 | 0.1MB | 설정·행사 일정 |
| `outputs/ensemble_v7/` (oof_v6_v7.pkl, domain_schedule_v7.json, 제출본) | 3 | 39.2MB | **v2·v4·v5 GPU 학습 결과가 녹아 있는 OOF·예측** |
| `outputs/ensemble_v8/` (oof_v8.npz) + `external/team_share/` 5개 | 6 | 14.9MB | 팀 공유 자료: 인천교통공사 2015~2024 시간대 이력, 보정계수, SSG·부천FC 경기, 달력 |
| `outputs/ensemble_v9/oof_v9_main.npz` | 1 | 2.1MB | v9(A100에서 학습한 트리) 결과 |
| `outputs/ensemble_v10/` (oof, 제출본) + `external/` 2개 | 4 | 6.2MB | 서울 7호선 이력, 급증일 라벨 |
| `outputs/ensemble_v11/` (oof, specialists, 제출본) + `external/` 2개 + `models/` 88개 | 93 | 200MB | **v11 모델 예측(GPU 장비에서 학습)**. 기상(open-meteo), SSG 경기 점수 |
| `outputs/ensemble_v12/oof_v12.npz` | 1 | 2.1MB | |
| `outputs/ensemble_v13/` oof + `external/` 7개 | 8 | 2.9MB | 공연장·경기·축제·방학 표 |
| `outputs/ensemble_v14/` oof·제출본 + `external/` 2개 + `models/` 8개 | 12 | 28.9MB | 달력, 서울 일별 카드 자료(2015~2026 상반기) |
| `outputs/ensemble_v15/` oof·제출본 + `external/event_corrections.csv` + `models/` 80개 | 83 | 233MB | v15 LightGBM·XGBoost 예측(CPU에서 학습) |
| `outputs/ensemble_v16`, `v17` oof | 2 | 4.3MB | |
| `outputs/ensemble_v19/` 제출본 + `external/mart_closure_calendar.csv` | 2 | 3.3MB | |
| `outputs/ensemble_v20/submission_ensemble_v20e.csv` | 1 | 3.3MB | 비교 기준 |

## 3. 단일 노트북으로 묶을 때의 걸림돌

1. **GPU 학습 단계의 저장 예측에 의존한다.**
   - `oof_v6_v7.pkl`에는 v2·v4·v5의 TabM·시공간 신경망 결과가 들어 있다(RTX PRO 4500 Blackwell).
   - `outputs/ensemble_v11/models`도 GPU 장비에서 학습한 결과다.
   - 이 단계들을 처음부터 다시 학습하면 약 10시간이 걸리고, 비트 단위로 같은 결과는 보장되지 않는다.
   - 따라서 노트북은 "저장 예측을 입력으로 받아 끝단만 다시 만드는 모드"를 기본으로 두고, 전체 재학습은 선택 모드로 두는 것이 현실적이다.
2. **LightGBM은 스레드 수에 따라 결과가 바뀐다.**
   - 같은 모델을 4스레드로 재학습하면 로그 차이가 RMS 0.0032, 최대 0.08이었다.
   - 코랩에서도 `V11_THREADS=12`로 고정해야 한다.
   - 그래도 CPU 종류(ARM ↔ x86)가 다르면 같은 결과를 보장할 수 없다. 다른 컴퓨터에서는 확인하지 못했다.
3. **파이썬과 패키지 버전.**
   - Python 3.12, numpy 2.3.5에 고정해야 한다.
   - numpy 1.x에서는 `oof_v6_v7.pkl`을 열지 못한다(`numpy._core` 모듈 없음 오류를 직접 확인).
   - 버전 목록은 `reproduce/requirements.txt`에 있다.
4. **외부 표는 사람이 모은 사본이다.**
   - 대상: `outputs/ensemble_v*/external/` 아래 표들.
   - 원천과 수집일은 `outputs/ensemble_v19/EXTERNAL_SOURCES.md`에 있으나, 라이선스 칸은 초안이다.
   - 버스 표(`outputs/ensemble_v20/external/`)는 교통카드 통계 사이트 화면에서 조회한 값이며, 자동 내려받기 코드가 없다.
5. **명령이 여러 단계로 이어진다.**
   - v2부터 v20e까지 20여 개 모듈을 순서대로 실행한다(`reproduce/run_all.sh`). 이 스크립트를 처음부터 끝까지 한 번에 돌려 본 적은 없다.
   - 확인한 것은 v10과 v15→v19→v20e의 재실행, LightGBM·XGBoost 재학습 표본 2건이다.
6. **경로.**
   - 코드는 저장소 루트 기준 상대 경로(`c8.ROOT`)를 쓴다. 노트북은 저장소 루트로 `cd`한 뒤 `python -m …`으로 실행해야 한다.
7. **용량.**
   - `outputs/` 전체는 7.4GB이고, 대부분 v2~v5의 폴드별 산출물이다.
   - 끝단 검증에는 약 550MB만 있으면 되고, 그 부분만 저장소에 넣었다.
8. **추적 범위.**
   - 감사 훅 추적은 끝단(v15 → v20e)만 다룬다.
   - v10 이전 단계가 읽는 파일은 실행 추적 없이 MANIFEST에만 올라 있다.

## 4. 노트북 최소 구성 (제안)

```bash
git clone https://github.com/taeyang0505/SINCON.git && cd SINCON
pip install -r reproduce/requirements.txt
export V11_THREADS=12
python -m reproduce.verify_from_stored          # 끝단 재실행, identical: true 확인 (약 2분)
python -m reproduce.verify_from_stored --retrain   # + LightGBM·XGBoost 표본 재학습 (약 5분)
```
