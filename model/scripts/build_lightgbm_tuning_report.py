"""Write the Korean LightGBM tuning report and selected-trial learning curves.

Run after ``python -m src.tune_lightgbm`` has completed. This script only reads
experiment/model artifacts and writes the report and figure under ``reports``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / "outputs" / "lightgbm_tuned"
PREVIOUS = ROOT / "outputs" / "three_tree"
ARCHIVE_NAME = "lightgbm_large_220_before_tuning_20261111_194707"
ARCHIVE = ROOT / "archives" / ARCHIVE_NAME
REPORT = ROOT / "reports" / "LIGHTGBM_TUNING.md"
FIGURE = ROOT / "reports" / "lightgbm_tuning_curves.png"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(columns: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def format_parameter(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def plot_curves(curves: list[pd.DataFrame], best: dict) -> None:
    selected = int(best["selected_iteration"])
    common = int(best["common_observed_iterations"])
    xmax = max(int(frame.iteration.max()) for frame in curves)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True, sharey=True, layout="constrained")
    for index, (ax, frame, info) in enumerate(zip(axes, curves, best["folds"]), 1):
        ax.plot(frame.iteration, frame.training_rmsle, color="#2563eb", linewidth=1.7, label="Training RMSLE")
        ax.plot(frame.iteration, frame.validation_rmsle, color="#0f766e", linewidth=1.9, label="Validation RMSLE (known stations)")
        ax.axvline(selected, color="#b45309", linewidth=1.6, linestyle="--", label=f"Selected common round: {selected}")
        own_best = int(info["best_iteration"])
        if 1 <= own_best <= len(frame):
            ax.scatter([own_best], [frame.validation_rmsle.iloc[own_best - 1]], s=38, color="#0f766e", zorder=5)
        if xmax > common:
            ax.axvspan(common + .5, xmax, color="#64748b", alpha=.08, label="Outside common selection prefix")
        ax.set_title(f"Development fold {index}\nEarly-stop best: {own_best}; observed: {len(frame)} rounds")
        ax.set_xlabel("Boosting round")
        ax.set_xlim(1, xmax)
        ax.set_yscale("log")
        ax.grid(alpha=.18, which="both")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("RMSLE (logarithmic axis; lower is better)")
    fig.suptitle(f"LightGBM trial {int(best['trial'])}: max 1,000 rounds, early-stopping patience 100", fontsize=13)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


def main() -> None:
    run = read(OUT / "run_summary.json")
    if run.get("status") != "completed":
        raise RuntimeError("LightGBM tuning has not completed; do not report a partial run as final.")
    selection = read(OUT / "selected_plan.json")
    design = selection["search_design"]
    best = selection["best_trial"]
    config = read(OUT / "config_resolved.json")
    verification = read(OUT / "verification.json")
    submission_check = read(OUT / "submission_validation.json")
    archive = read(ARCHIVE / "manifest.json")
    archive_checks = {name: digest(ARCHIVE / name) == expected for name, expected in archive["files"].items()}
    if not all(archive_checks.values()):
        raise AssertionError(f"Archived original model changed: {archive_checks}")
    if digest(PREVIOUS / "model_bundle.joblib") != archive["files"]["model_bundle.joblib"]:
        raise AssertionError("Original three-tree bundle no longer matches the archived pre-tuning bundle.")
    if not submission_check["valid"] or int(submission_check["rows"]) != 157194:
        raise AssertionError("The final submission did not pass validation.")
    if not verification["reload_bitwise_identical"] or not verification["hard_cap_satisfied"]:
        raise AssertionError("Model verification failed.")
    if selection.get("selection_uses_fold3_labels") or not selection.get("selection_frozen_before_fold3"):
        raise AssertionError("Unexpected selection/evaluation metadata.")

    # The archived artifact was generated locally by this project and is trusted.
    previous_model = joblib.load(ARCHIVE / "lightgbm_large.joblib")
    previous_params = previous_model.estimator.get_params()
    new_params = config["lightgbm_params"]
    selected_trial = int(best["trial"])
    trial_dir = OUT / "search" / f"trial_{selected_trial:02d}"
    curves = [pd.read_csv(trial_dir / f"fold_{fold}_learning_curve.csv") for fold in [1, 2]]
    plot_curves(curves, best)

    old_score = float(archive["reference_fold3_rmsle"])
    old_ensemble = float(read(PREVIOUS / "audit_summary.json")["rmsle"])
    new_score = float(run["audit_rmsle"])
    relative = 100 * (new_score / old_score - 1)
    comparison = f"상대 {abs(relative):.2f}% " + ("감소" if relative < 0 else "증가" if relative > 0 else "변화 없음")
    comparison_note = (
        "이번 설정은 기존 LightGBM보다 이 비교 구간의 RMSLE가 낮았다. 다만 이미 분석한 구간이므로 새로운 독립 검증 성능으로 보지 않는다."
        if new_score < old_score else
        "이번 설정은 개발 구간에서 선정됐지만 기존 LightGBM보다 이 비교 구간의 RMSLE가 높았다. 더 많은 탐색이나 early stopping 적용이 미래 성능 개선을 자동 보장하지 않는다."
        if new_score > old_score else
        "이번 설정과 기존 LightGBM의 비교 구간 RMSLE가 같았다."
    )
    original_scores = pd.read_csv(PREVIOUS / "cv_scores.csv")
    new_scores = pd.read_csv(OUT / "cv_scores.csv")
    all_cv = []
    for fold in [1, 2, 3]:
        old = original_scores.loc[(original_scores.fold == fold) & (original_scores.model == "lightgbm_large"), "rmsle"].iloc[0]
        new = new_scores.loc[new_scores.fold == fold, "rmsle"].iloc[0]
        scope = "전체 역 사후 진단" if fold < 3 else "재사용 마지막 비교 구간"
        all_cv.append([fold, scope, f"{old:.9f}", f"{new:.9f}"])

    selection_total = sum(int(f["selection_rows"]) for f in design["folds"])
    dev_rows = []
    for fold, info in enumerate(design["folds"], 1):
        old_oof = pd.read_csv(PREVIOUS / f"fold_{fold}" / "oof.csv.gz")
        new_oof = pd.read_csv(OUT / f"fold_{fold}" / "oof.csv.gz")
        pd.testing.assert_frame_equal(old_oof[["date", "station_id", "hour", "boardings"]], new_oof[["date", "station_id", "hour", "boardings"]])
        known = new_oof.known_at_origin.to_numpy(dtype=bool)
        old_known = float(np.sqrt(np.mean((np.log1p(old_oof.loc[known, "lightgbm_large"]) - np.log1p(old_oof.loc[known, "boardings"])) ** 2)))
        new_known = float(read(OUT / f"fold_{fold}" / "known_station_metrics.json")["rmsle"])
        dev_rows.append([fold, info["origin"], f"{info['selection_rows']:,}", f"{info['excluded_cold_rows']:,}",
                         f"{int(info['selection_rows']) / selection_total:.3%}", f"{old_known:.9f}", f"{new_known:.9f}"])

    keys = ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples", "reg_lambda", "reg_alpha", "colsample_bytree", "subsample", "subsample_freq", "min_split_gain"]
    parameter_rows = [[name, format_parameter(previous_params[name]), format_parameter(new_params[name])] for name in keys]
    stop_rows = []
    for info, curve in zip(best["folds"], curves):
        stop_rows.append([info["fold"], info["best_iteration"], info["recorded_iterations"],
                          info["retained_trees"], f"{info['best_validation_rmsle']:.9f}",
                          "예" if info["early_stopped_before_cap"] else "아니오: 상한 도달",
                          f"{float(curve.training_rmsle.iloc[int(info['best_iteration']) - 1]):.9f}"])
    search = pd.read_csv(OUT / "search_results.csv")
    if len(search) != int(run["trials"]):
        raise AssertionError("The recorded search trial count is incomplete.")
    trial_rows = []
    total_cap_hits = 0
    for trial in search.itertuples(index=False):
        detail = read(OUT / "search" / f"trial_{int(trial.trial):02d}" / "trial.json")
        total_cap_hits += sum(bool(f["cap_reached"]) for f in detail["folds"])
        path = f"../outputs/lightgbm_tuned/search/trial_{int(trial.trial):02d}"
        trial_rows.append([f"{int(trial.trial)}" + (" **선정**" if int(trial.trial) == selected_trial else ""),
                           f"{trial.learning_rate:g}", int(trial.num_leaves), int(trial.min_child_samples),
                           f"{trial.reg_lambda:g}", int(trial.selected_iteration), int(trial.common_observed_iterations),
                           f"{trial.pooled_development_rmsle:.9f}",
                           f"[개발 1]({path}/fold_1_learning_curve.csv) · [개발 2]({path}/fold_2_learning_curve.csv)"])

    report = f'''# LightGBM 재학습·early stopping·설정 탐색 결과

작성 기준: 2026-11-11. 이 문서는 실제 완료된 `outputs/lightgbm_tuned` 산출물을 읽어 생성했다.

## 결과

- 사용자 요청대로 **최대 1,000 부스팅 라운드**, **early-stopping patience {design['early_stopping_rounds']}회**, **최소 개선폭 {design['min_delta']:g}**를 적용했다.
- 개발 구간에서 **{int(run['trials'])}개 설정**을 비교한 결과, 후보 **{selected_trial}번**, 공통 반복 수 **{int(best['selected_iteration'])}회**가 선정됐다.
- 개발의 기존 역 행을 합산한 RMSLE는 **{float(run['development_rmsle']):.9f}**다. 개발 점수는 설정 선택에 사용한 값이다.
- 2026-02-28~06-30의 123일·157,194행에서 기존 LightGBM **{old_score:.9f} → 재학습 LightGBM {new_score:.9f}**, {comparison}.
- 이전 3모델 앙상블의 동일 구간 RMSLE는 **{old_ensemble:.9f}**다.
- 최종 저장 모델은 **LightGBM 단독 100%**이며 실제 트리는 **{verification['actual_final_trees']}개**다. 계절 모델·XGBoost·CatBoost 가중치를 새로 적합하지 않았다.
- 전체 탐색·재학습·진단·제출 생성에 **{float(run['elapsed_seconds']):.1f}초**가 걸렸다. 패키지 설치와 이 보고서 생성 시간은 제외한다.
- Kaggle에 제출하지 않았다. 이 점수는 리더보드 점수가 아니다.

{comparison_note}

## 1. Optimizer와 전역 최적점의 의미

**LightGBM은 AdamW를 사용하지 않는다.** 로그 타겟의 gradient/Hessian을 이용해 트리 분할과 리프 출력을 순차적으로 적합하는 GBDT다.
프로젝트의 `torch.optim.AdamW`는 이전 MLP 구현에만 사용한다. 이번 학습에는 MLP가 포함되지 않는다.
사용자의 1,000 epoch 요청은 이 회귀 GBDT에서 **1,000 boosting rounds를 상한으로 설정**하는 것으로 적용했다.

이번 탐색은 가능한 모든 트리 구조·하이퍼파라미터 조합의 전역 최적해를 증명하지 않는다.
정확한 결론은 **미리 고정한 {int(run['trials'])}개 후보와 각 후보의 두 개발 fold가 공통으로 관측한 반복 범위에서 최선**이라는 것이다.
Early stopping은 각 후보를 항상 1,000회까지 돌리지 않으므로 모든 후보의 1~1,000회 전체를 비교했다고 말할 수 없다.
이번 실행에서 개발 모델 {int(run['trials']) * 2}개 중 1,000회 상한까지 관측한 경우는 **{total_cap_hits}개**다.

## 2. 기존 가중치 별도 보관

원래 220회 LightGBM은 재학습 전에 아래 경로로 별도 보관했다.

`archives/{ARCHIVE_NAME}`

- [원래 LightGBM 단독 객체](../archives/{ARCHIVE_NAME}/lightgbm_large.joblib)
- [원래 LightGBM 텍스트 모델](../archives/{ARCHIVE_NAME}/lightgbm_large.txt)
- [원래 전체 모델 bundle](../archives/{ARCHIVE_NAME}/model_bundle.joblib)
- [원래 LightGBM 제출 파일](../archives/{ARCHIVE_NAME}/submission_primary.csv)
- [백업 파일 SHA-256 목록](../archives/{ARCHIVE_NAME}/manifest.json)

이 보고서 생성 시 백업 목록 {len(archive_checks)}개 파일의 해시를 모두 대조했고, 기존 `outputs/three_tree/model_bundle.joblib`도 백업과 동일함을 확인했다.
새 모델은 별도 `outputs/lightgbm_tuned`에 저장했다.

## 3. 시간 순서와 검증 대상

입력 특징 60개와 학습 행 생성 방식은 유지했다. 각 예측 기준일 이전 관측으로 특징을 만들며,
학습 타겟의 전체 123일 구간은 해당 기준일까지 종료되도록 제한했다.
개발 탐색에 전달한 관측은 **{design['development_label_end']}까지**이며 그 뒤의 마지막 비교 구간 타겟을 early stopping에 사용하지 않았다.

실제 테스트에는 관측 이력이 전혀 없는 역이 없으므로, 개발 지표도 **해당 기준일까지 관측된 역**에서 계산했다.
개발 1의 신규 역 **6,588행**은 선택 지표에서 제외하되, 전체 역 점수는 별도 진단으로 남겼다.
이 선택은 앞선 분석 이후 정한 검증 설계이므로, 마지막 구간을 새로운 독립 holdout으로 해석하면 안 된다.

{table(['개발 fold', '예측 기준일', '선택 행', '제외 신규 역 행', 'pooled 지표 행 가중치', '기존 220회 RMSLE', '재학습 RMSLE'], dev_rows)}

여기서 행 가중치는 **CV 손실 합산에 쓰는 비중**이다. 여러 모델의 앙상블 가중치를 뜻하지 않는다.
평균 RMSLE를 단순 평균하지 않고 각 fold RMSLE를 제곱한 뒤 행 수로 가중 평균하고 제곱근을 취한다.

전체 역 결과는 다음과 같으며, 개발 1의 전체 역 점수에는 선택 지표에서 제외한 신규 역이 포함돼 위 표와 다르다.

{table(['fold', '평가 범위', '기존 LightGBM RMSLE', '재학습 LightGBM RMSLE'], all_cv)}

## 4. Early stopping과 공통 반복 수 선택

LightGBM 기본 회귀 지표를 `metric='None'`으로 끄고, 다음 평가 함수를 사용했다.

```python
# y_log = log1p(boardings), pred_log = 모델의 로그 공간 출력
rmsle = sqrt(mean((maximum(pred_log, 0) - y_log) ** 2))
```

이는 추론 시 음수 로그 출력을 0으로 자른 뒤 `expm1`으로 복원하는 실제 제출 예측과 일치한다.
Training과 validation 곡선을 모두 기록하고, validation RMSLE가 {design['min_delta']:g} 이상 개선되지 않는 상태가
{design['early_stopping_rounds']}회 연속 이어지면 해당 개발 학습을 종료했다. Training 지표는 종료 기준에 사용하지 않았다.

선정 후보의 개발별 종료 기록:

{table(['개발 fold', '개별 best iteration', '관측한 반복 수', 'early-stop 모델에 남은 트리', '개별 best 검증 RMSLE', '1,000회 이전 종료', '개별 best 학습 RMSLE'], stop_rows)}

서로 다른 fold의 개별 best iteration을 단순 평균하거나 중앙값으로 정하지 않았다.
선정 후보에서는 두 학습 곡선이 공통으로 존재하는 **1~{int(best['common_observed_iterations'])}회**에서
행 수 가중 RMSLE가 최소인 **{int(best['selected_iteration'])}회**를 최종 반복 수로 정했다.
같은 반복 수의 예측을 비교하므로 관측하지 않은 구간을 추정하거나, 중단 이후 곡선을 임의로 연장하지 않는다.

![선정 후보의 학습·검증 곡선](lightgbm_tuning_curves.png)

점선은 최종 공통 반복 수, 녹색 점은 각 fold의 early-stopping best iteration이다.
회색 영역은 두 fold의 공통 비교 범위를 벗어난 반복 구간이다. 세로축은 RMSLE의 로그 눈금이며,
선택 알고리즘에는 원래 RMSLE 수치를 사용했다. 곡선 원자료는 후보별 CSV에 보관했다.

최종 전체 데이터에는 개발에서 선정한 설정과 반복 수를 적용해 다시 학습했다.
**최종 전체 데이터 모델 자체에는 별도 검증을 다시 붙여 early stopping을 하지 않는다.**
종료점이 개발에서 이미 고정됐기 때문이다. 2026년 7~10월 실제 정답은 사용하지 않았다.

## 5. 선정된 학습·규제 설정

{table(['파라미터', '기존 Large', '이번 선정 설정'], parameter_rows)}

공통으로 `objective='regression'`, 타겟 `log1p(boardings)`, 시드 2026, CPU 4스레드,
`linear_tree=False`를 유지한다. `station_id`, `canonical_line`, `hour`, `dow`를 범주형으로 처리한다.
`n_estimators`의 이번 값은 최종 재학습에 쓰는 선정 반복 수이며 탐색 상한 1,000회와 구별한다.
최종 학습 사례는 **{int(run['final_training_rows']):,}행**, 특징은 **{int(run['feature_count'])}개**다.

리프 수·최소 리프 표본·깊이 상한·L1/L2 규제·행/특징 샘플링을 후보 설정에서 달리했다.
규제와 조기 종료를 사용했다는 사실만으로 과적합이 완전히 제거됐다고 단정하지 않는다.
개발 지표에 여러 후보를 비교한 선택 편향과 계절·역별 분포 변화는 남아 있다.

## 6. 전체 후보 기록

아래 표는 모든 후보의 공통 관측 구간 내 최선 반복 수와 개발 점수다.
표의 일부 설정 외 나머지 규제값은 [전체 search_results.csv](../outputs/lightgbm_tuned/search_results.csv)와
각 후보의 `trial.json`에 기록돼 있다.

{table(['후보', '학습률', '리프', '최소 표본', 'L2', '공통 선정 반복', '공통 관측 상한', '개발 RMSLE', '학습 곡선 CSV'], trial_rows)}

## 7. 저장·재현·제출 검수

- 개발에서 선정한 점수: **{float(verification['selected_development_rmsle']):.12f}**.
- 같은 설정·고정 반복으로 재학습한 개발 점수: **{float(verification['refitted_development_rmsle']):.12f}**.
- Early stopping 모델의 저장 트리 수가 공통 선정 반복보다 적을 수 있으므로, 선정 설정을 고정 반복으로 다시 학습해 점수 일치를 확인했다.
- 최종 저장 모델의 재로딩 예측과 원래 예측이 일치했고, 생성 CSV의 SHA-256도 일치했다.
- 실제 최종 트리 **{int(verification['actual_final_trees'])}개 ≤ 1,000회**를 확인했다.
- 제출 파일은 **157,194행**, 전체 test ID와 순서, 결측·무한대·음수 검사를 통과했다.
- 원본 sample의 92일 범위 대신 공식 123일 test 전체를 사용한다.
- 최종 제출 SHA-256: `{verification['submission_sha256']}`.

산출물:

- [재학습 제출 CSV](../outputs/lightgbm_tuned/submission.csv)
- [재학습 모델 bundle](../outputs/lightgbm_tuned/model_bundle.joblib)
- [LightGBM 텍스트 모델](../outputs/lightgbm_tuned/lightgbm_tuned.txt)
- [실제 적용 설정](../outputs/lightgbm_tuned/config_resolved.json)
- [탐색 설계와 동결된 선정 결과](../outputs/lightgbm_tuned/selected_plan.json)
- [검증·재현 결과](../outputs/lightgbm_tuned/verification.json)
- [기존 LightGBM 학습 설정 점검](../analysis/lightgbm/TRAINING_AUDIT.md)

```bash
cd /Users/hongtaeyang/Desktop/SINCON
# 기존 완료 결과를 덮어쓰지 않도록 새 출력 폴더 지정
.venv/bin/python -m src.tune_lightgbm --config configs/three_tree.json --max-rounds 1000 --patience 100 --trials 16 --output outputs/lightgbm_tuned_repeat
# 현재 저장 모델만 다시 추론
.venv/bin/python -m src.pipeline infer --model outputs/lightgbm_tuned/model_bundle.joblib --output outputs/lightgbm_tuned_cli_reload
# 현재 완료 결과에서 이 보고서와 그림 재생성
.venv/bin/python scripts/build_lightgbm_tuning_report.py
```
'''
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "figure": str(FIGURE), "selected_trial": selected_trial,
                      "selected_rounds": int(best["selected_iteration"]), "audit_rmsle": new_score,
                      "archive_files_verified": len(archive_checks), "model_artifacts_modified": False},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
