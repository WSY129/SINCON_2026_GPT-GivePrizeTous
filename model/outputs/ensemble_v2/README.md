# SINCON ensemble_v2

구현·학습·추론 재현 자료의 시작점이다. 공식 테스트 정답이나 새 Public 점수는 포함하지 않는다.

- [구현 설명](IMPLEMENTATION.md): 데이터 정제, 부분 horizon CV, 83개 피처, 4종 모델, 전처리, 로그 앙상블, 재학습과 추론.
- [실행 결과](RESULTS.md): 실제 4-fold 점수, LOFO, 잔차 상관, 동일 Fold 4 기존 모델 비교, 반복 수와 분포 검수. 최종 검수 완료 후 생성된다.
- [제출 파일](submission_ensemble_v2.csv): 157,194개 test ID의 승차 예측. Kaggle에 자동 제출하지 않는다.
- [제출 검수 JSON](submission_validation.json), [전체 전달물 해시](delivery_manifest.json), [저장 모델 재추론 검수](verification/reload_agreement.json).
- [설정](../../configs/ensemble_v2.json), [재현 노트북](../../notebooks/SINCON_reproduce.ipynb).

작업 폴더 `/Users/hongtaeyang/Desktop/SINCON`에서:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.pipeline v2 run
.venv/bin/python -m src.pipeline v2 infer
```

같은 소스·설정·입력 해시로 실행하면 완료 모델을 재사용한다. 새 입력이나 설정을 기존 실행에 몰래 섞지 않는다. 보호 대상 기존 제출 파일·모델 번들·설정은 `protected_sha256.json`으로 확인한다.

`source_before/`는 변경 전 소스, `attempt_1_partial_before_elasticity_fix/`는 정제 수정 전 미완료 실험, `xgboost_validation_fix/`는 XGBoost 검증 경로 수정 전 결과 및 수정 감사 기록이다. 이 폴더의 현재 OOF·모델·제출 파일만 최종 산출물로 사용한다.
