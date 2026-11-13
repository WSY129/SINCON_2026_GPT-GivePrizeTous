# ensemble_v5 로컬 회수 기록

RunPod 크레딧 종료에 대비해 완료된 산출물을 45초 간격으로 증분 복사한다.

- 대상: `/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/`
- 완료된 모델: 같은 디렉터리의 `result.json` 생성 이후 `model.joblib`을 회수한다.
- 포함: 예측 로그 배열, 학습 이력, 진단, OOF, 제출 CSV, 보고서, 입력·소스 해시.
- 제외: 재생성 가능한 `origin_cache/`, `matrix.joblib`, 감사용 중간 pickle.
- 삭제 동기화는 사용하지 않는다. 기존 v1~v4 파일은 변경하지 않는다.
- `receive_status.json`은 마지막 성공 회수 시간과 파일 수를 기록한다.
- GPU 강제 종료 시 완료되지 않은 모델의 메모리 상태는 회수할 수 없다.

## 재현

`src/v5/`와 `configs/ensemble_v5.json`, 원본 `data/raw/`를 유지한다. 최종 모델이 모두 있으면 `python -m src.v5.pipeline infer`로 재추론한다. 재현 환경의 패키지 버전과 장치를 기존 학습 환경에 맞추고, XGBoost 저장 모델의 CUDA 재추론 경로를 유지한다. 결과 검수는 `reload_validation.json`과 `preservation_check.json`에서 확인한다.
