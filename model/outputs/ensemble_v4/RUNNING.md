# ensemble_v4 실행 안내

새 구현은 기존 v1/v2/v3를 수정하지 않고 `src/v4/`, `configs/ensemble_v4.json`, `outputs/ensemble_v4/`에만 저장한다.

프로젝트 루트에서:

```bash
python -m unittest tests.test_v4_pipeline -v
python -m src.v4.pipeline run
python outputs/ensemble_v4/build_report.py
```

전체 파이프라인은 Step A(Fold 3·4, 4모델) → 통과 시 Step B(4폴드, 7모델) → 전체 관측 2시드 재학습 → 새 프로세스 재추론 순서다. Step A의 Fold 4 개선 게이트에 실패하면 확장을 중단하고 원인을 기록한다. 실제 실행 상태는 `run_status.json` 및 `run.log`로 판단한다. `RESULTS.md`가 있더라도 미완료 단계까지 성공했다는 뜻은 아니다.

완료 후 별도 재추론:

```bash
python -m src.v4.pipeline infer
```

입력 SHA-256 불일치, 소스/설정 변경 후 기존 캐시 재사용, 제출 ID 불일치, 비유한 예측 또는 재로딩 불일치는 오류로 처리한다. 새 환경에서는 원본 4개 CSV를 `data/raw/`에 배치하고 기존 공통 src 모듈과 v4 소스를 함께 사용한다. 검증 전처리와 모델은 해당 폴드 학습 자료로만 적합한다.

현재 실행은 RunPod에서 7,200초 제한을 걸었다. Fourier 기준선 교정으로 재시작하면서 `SINCON_V4_ORIGINAL_START`에 최초 시작 시각을 전달해 원래 마감을 유지했다. 재시작 전 소스·로그·결과는 `attempt_a_initial/`에 보관한다. 단일 실험의 소요 시간을 축소해 보고하지 않는다.

RunPod의 장치는 RTX PRO 4500 Blackwell 32GB 1개이며, 표시된 CPU 48개와 달리 컨테이너 cgroup 할당은 약 10.2코어다. 따라서 계획서의 실행 시간은 보장이 아니고 실측 기록을 따른다.
