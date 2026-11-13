"""Posthoc diagnosis of the frozen baseline ensemble, never a deployment job.

Reads preserved OOF artifacts. Alternative weights/calibrations are fitted
only on development folds 1/2 (or a stated subset); audit labels never enter
an optimizer. Outputs are diagnostic tables/plans, not prediction CSVs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, data):
    path.write_text(json.dumps(native(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(y, prediction):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if np.any(y < 0) or np.any(p < 0) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Counts must be finite and nonnegative.")
    error = np.log1p(p) - np.log1p(y)
    mse = float(np.mean(error**2))
    return {
        "rows": len(y), "rmsle": float(np.sqrt(mse)), "log_mse": mse,
        "mean_log_error": float(error.mean()), "squared_log_bias": float(error.mean()**2),
        "log_error_variance": float(error.var()),
        "bias_fraction_of_log_mse": float(error.mean()**2 / mse) if mse else 0.,
        "mae": float(np.mean(np.abs(p-y))), "raw_mbe": float(np.mean(p-y)),
        "mean_prediction": float(p.mean()), "mean_truth": float(y.mean()),
    }


def md_table(frame, cols=None):
    if cols is not None:
        frame = frame[cols]
    def formatted(x):
        return f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x)
    return "| " + " | ".join(map(str, frame.columns)) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(formatted(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/Users/hongtaeyang/Desktop/SINCON"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    project, out = args.project.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project))
    sys.dont_write_bytecode = True
    from src.ensemble import fit_ensemble, predict_ensemble

    baseline = project / "outputs/baseline"
    plan_path = baseline / "ensemble_plan.json"
    frozen = json.loads(plan_path.read_text())
    config = json.loads((project / "configs/default.json").read_text())
    names = config["models"]
    folds = {}
    source_paths = [plan_path, project / "src/ensemble.py", project / "src/metrics.py", project / "configs/default.json"]
    for fold in (1, 2, 3):
        path = baseline / f"fold_{fold}/oof.csv.gz"
        source_paths.append(path)
        frame = pd.read_csv(path, parse_dates=["date"])
        if frame.duplicated(["date", "station_id", "hour"]).any():
            raise ValueError(f"Duplicate OOF keys in fold {fold}.")
        frame["fold"] = fold
        folds[fold] = frame
    if folds[1].date.max() >= folds[2].date.min() or folds[2].date.max() >= folds[3].date.min():
        raise ValueError("Development/audit forecast dates overlap.")
    dev = pd.concat([folds[1], folds[2]], ignore_index=True)
    populations = {"development_1": folds[1], "development_2": folds[2], "development_pooled": dev, "audit_3": folds[3]}
    source_paths.append(project / "data/raw/metaData.csv")
    meta = pd.read_csv(project / "data/raw/metaData.csv", parse_dates=["first_date"])
    cold_ids = meta.loc[meta.first_date > pd.Timestamp(config["fold_origins"][0]), "station_id"].tolist()

    # These frozen candidates do not learn from any audit value.
    single_best_name = min(frozen["individual_scores"], key=frozen["individual_scores"].get)
    selected = [name for name, weight in frozen["weights"].items() if weight > 0]
    trees = ["lightgbm_small", "lightgbm_large", "catboost", "xgboost"]
    no_bias = {**frozen, "bias": 0.}
    swapped = {**frozen, "weights": dict(frozen["weights"])}
    swapped["weights"]["lightgbm_large"] = swapped["weights"].pop("lightgbm_small")
    plans = {
        "original_frozen": frozen,
        "frozen_weights_no_bias": no_bias,
        "replace_small_with_large_same_weights": swapped,
        "replace_small_with_large_no_bias": {**swapped, "bias": 0.},
    }
    fit_records = {
        "original_frozen": "Original deployment plan, fitted development folds 1+2 only.",
        "frozen_weights_no_bias": "Diagnostic removal of the existing scalar calibration; no fit.",
        "replace_small_with_large_same_weights": "Posthoc structural ablation of the excluded near-duplicate, weights and bias unchanged; no audit fit.",
        "replace_small_with_large_no_bias": "Same structural ablation without scalar calibration; no fit.",
    }
    fits = [
        ("refit_dev1_only", folds[1], 0.95),
        ("refit_dev2_only", folds[2], 0.95),
        ("refit_pooled_no_correlation_filter", dev, 1.01),
        ("refit_pooled_existing_stations_only", dev.loc[~dev.station_id.isin(cold_ids)], 0.95),
    ]
    for label, frame, cutoff in fits:
        print(f"Fitting diagnostic {label} on {len(frame):,} development rows only", flush=True)
        p, _ = fit_ensemble(frame.boardings.to_numpy(), {name: frame[name].to_numpy() for name in names}, metric="rmsle", correlation_cutoff=cutoff)
        plans[label] = p
        plans[label + "_no_bias"] = {**p, "bias": 0.}
        fit_records[label] = {"training_folds": sorted(frame.fold.unique().tolist()), "training_rows": len(frame), "correlation_cutoff": cutoff, "audit_used_for_fit": False, "excluded_station_ids": cold_ids if "existing_stations" in label else []}
        fit_records[label + "_no_bias"] = {"same_weights_as": label, "bias": 0., "audit_used_for_fit": False}

    plan_rows = []
    for label, p in plans.items():
        for name, weight in p["weights"].items():
            plan_rows.append({"candidate": label, "model": name, "weight": weight, "log_bias": p["bias"]})
    pd.DataFrame(plan_rows).to_csv(out / "diagnostic_plan_weights.csv", index=False)
    save_json(out / "diagnostic_plans_NOT_DEPLOYED.json", {"status": "posthoc_diagnostic_only_not_deployed", "warning": "Alternatives were inspected after seeing audit results. Do not reinterpret their audit score as a fresh unbiased test or select a deployment from it.", "fit_populations": fit_records, "plans": plans})

    evaluations, decomposition, pairs, cohorts, monthly, source_matches = [], [], [], [], [], {}
    predictions_by_population = {}
    for population, frame in populations.items():
        y = frame.boardings.to_numpy()
        component = {name: frame[name].to_numpy() for name in names}
        candidates = {label: predict_ensemble(component, p) for label, p in plans.items()}
        candidates["frozen_development_best_single"] = component[single_best_name]
        for label, members in [("all_7", names), ("trees_4", trees), ("original_positive_weight_5", selected)]:
            matrix = np.column_stack([component[name] for name in members])
            candidates[f"equal_log_{label}"] = np.expm1(np.log1p(matrix).mean(axis=1))
            candidates[f"equal_raw_{label}"] = matrix.mean(axis=1)
        candidates.update({f"single_{name}": p for name, p in component.items()})
        predictions_by_population[population] = candidates
        for label, pred in candidates.items():
            evaluations.append({"population": population, "candidate": label, "interpretation": "posthoc_audit_diagnosis" if population == "audit_3" and label != "original_frozen" else "development_or_frozen_original", **stats(y, pred)})
        source_matches[population] = float(np.max(np.abs(candidates["original_frozen"] - frame.ensemble.to_numpy())))
        if source_matches[population] > 1e-8:
            raise AssertionError("Recomputed frozen ensemble differs from preserved OOF.")

        target_log = np.log1p(y)
        z = sum(frozen["weights"].get(name, 0.) * np.log1p(component[name]) for name in names)
        error_before = z - target_log
        b = frozen["bias"]
        z_after = np.maximum(0., z + b)
        raw_unclipped_after_mse = np.mean((error_before + b)**2)
        actual_after_mse = np.mean((z_after-target_log)**2)
        weighted_mse = sum(frozen["weights"].get(name, 0.) * np.mean((np.log1p(component[name])-target_log)**2) for name in names)
        ambiguity = sum(frozen["weights"].get(name, 0.) * np.mean((np.log1p(component[name])-z)**2) for name in names)
        before_mse = np.mean(error_before**2)
        identity_gap = weighted_mse - ambiguity - before_mse
        if abs(identity_gap) > 1e-12:
            raise AssertionError("Log-MSE ambiguity identity failed.")
        decomposition.append({
            "population": population, "rows": len(frame), "log_bias_applied": b,
            "mean_log_error_before": error_before.mean(), "mean_log_error_after": (z_after-target_log).mean(),
            "before_log_mse": before_mse, "after_log_mse": actual_after_mse,
            "bias_mse_cross_term_2b_meanerror": 2*b*error_before.mean(), "bias_mse_square_term_b2": b*b,
            "bias_mse_delta_total": actual_after_mse-before_mse,
            "clipping_rows": int(np.sum(z+b<0)), "clipping_mse_delta": actual_after_mse-raw_unclipped_after_mse,
            "log_error_variance_before": error_before.var(), "log_error_variance_after": (z_after-target_log).var(),
            "squared_mean_log_error_before": error_before.mean()**2,
            "squared_mean_log_error_after": (z_after-target_log).mean()**2,
            "weighted_component_log_mse": weighted_mse, "ambiguity_log_mse_gain": ambiguity,
            "ambiguity_fraction_of_weighted_component_mse": ambiguity/weighted_mse,
            "ambiguity_identity_numeric_residual": identity_gap,
        })
        for space in ("raw", "log1p"):
            residual = pd.DataFrame({name: component[name]-y if space == "raw" else np.log1p(component[name])-target_log for name in names})
            corr = residual.corr()
            corr.to_csv(out / f"residual_correlations_{population}_{space}.csv")
            for i, name in enumerate(names):
                for other in names[i+1:]:
                    pairs.append({"population": population, "space": space, "model_1": name, "model_2": other, "correlation": corr.loc[name, other]})
        diagnostic_labels = ["original_frozen", "frozen_weights_no_bias"] + [f"single_{n}" for n in names]
        for cohort, mask in [("new_since_fold1_origin", frame.station_id.isin(cold_ids)), ("existing_at_fold1_origin", ~frame.station_id.isin(cold_ids))]:
            for label in diagnostic_labels:
                s = stats(y[mask], candidates[label][mask])
                contribution = s["log_mse"]*float(mask.mean())
                total_mse = float(np.mean((np.log1p(candidates[label])-target_log)**2))
                cohorts.append({"population": population, "cohort": cohort, "candidate": label, "row_fraction": float(mask.mean()), **s, "contribution_to_population_log_mse": contribution, "share_of_population_squared_log_error": contribution / total_mse})
        for month, mask in frame.groupby(frame.date.dt.to_period("M")).groups.items():
            ix = frame.index.get_indexer(mask)
            for label in ["original_frozen", "frozen_weights_no_bias", "frozen_development_best_single", "single_lightgbm_large"]:
                monthly.append({"population": population, "month": str(month), "candidate": label, **stats(y[ix], candidates[label][ix])})

    scores = pd.DataFrame(evaluations)
    decomp = pd.DataFrame(decomposition)
    pair_frame = pd.DataFrame(pairs)
    cohort_frame = pd.DataFrame(cohorts)
    scores.to_csv(out / "candidate_scores.csv", index=False)
    decomp.to_csv(out / "log_mse_bias_ambiguity_decomposition.csv", index=False)
    pair_frame.to_csv(out / "residual_correlation_pairs.csv", index=False)
    cohort_frame.to_csv(out / "station_cohort_contributions.csv", index=False)
    pd.DataFrame(monthly).to_csv(out / "monthly_transport.csv", index=False)

    def row(population, candidate):
        return scores.loc[(scores.population == population) & (scores.candidate == candidate)].iloc[0]
    original = row("audit_3", "original_frozen")
    uncalibrated = row("audit_3", "frozen_weights_no_bias")
    best_single = row("audit_3", "single_lightgbm_large")
    dev_row = decomp.loc[decomp.population == "development_pooled"].iloc[0]
    audit_row = decomp.loc[decomp.population == "audit_3"].iloc[0]
    summary = {
        "status": "posthoc_diagnostic_only_original_files_preserved",
        "original_plan_recomputed_max_abs_difference_by_population": source_matches,
        "official_metric": "rmsle", "single_best_selected_on_original_development": single_best_name,
        "audit_original": original.to_dict(), "audit_same_weights_without_bias": uncalibrated.to_dict(),
        "audit_best_observed_single_hindsight_comparison_only": best_single.to_dict(),
        "original_bias_log1p": frozen["bias"], "one_plus_count_multiplier": float(np.exp(frozen["bias"])),
        "bias_audit_rmsle_increase": original.rmsle - uncalibrated.rmsle,
        "bias_audit_log_mse_increase": original.log_mse - uncalibrated.log_mse,
        "bias_share_of_original_audit_log_mse": (original.log_mse - uncalibrated.log_mse)/original.log_mse,
        "bias_share_of_gap_to_hindsight_best_single_log_mse": (original.log_mse - uncalibrated.log_mse)/(original.log_mse-best_single.log_mse),
        "bias_inflation_relative_to_uncalibrated_audit_log_mse": original.log_mse/uncalibrated.log_mse-1,
        "original_audit_squared_log_bias_fraction": original.squared_log_bias/original.log_mse,
        "development_bias_effect_log_mse": dev_row.bias_mse_delta_total,
        "audit_ambiguity_gain_before_scalar_bias": audit_row.ambiguity_log_mse_gain,
        "cold_station_ids_for_diagnostic_cohort": cold_ids,
        "caution": "Audit alternatives were examined after observing baseline failure. No alternative is newly validated or deployed; obtain another untouched temporal block before selecting changes.",
    }
    save_json(out / "summary.json", summary)
    save_json(out / "input_sha256.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})

    highlighted = ["original_frozen", "frozen_weights_no_bias", "frozen_development_best_single", "single_lightgbm_large", "equal_log_trees_4", "equal_raw_trees_4", "equal_log_original_positive_weight_5", "equal_log_all_7", "refit_dev2_only", "refit_dev2_only_no_bias", "refit_pooled_no_correlation_filter", "refit_pooled_no_correlation_filter_no_bias", "replace_small_with_large_no_bias", "refit_pooled_existing_stations_only", "refit_pooled_existing_stations_only_no_bias"]
    comparison = scores.loc[scores.candidate.isin(highlighted)].pivot(index="candidate", columns="population", values="rmsle").reindex(highlighted).reset_index()
    lgb_corr = pair_frame.loc[(pair_frame.model_1 == "lightgbm_small") & (pair_frame.model_2 == "lightgbm_large")]
    cold_contrib = cohort_frame.loc[(cohort_frame.population == "development_1") & (cohort_frame.cohort == "new_since_fold1_origin")]
    draft = f'''# 앙상블 실패 원인 정량 진단

이 문서는 원본 OOF를 다시 계산한 사후 진단이다. 대안 가중치는 development fold 1/2 또는 명시된 부분집합에만 적합했다. audit 정답은 대안 적합에 넣지 않았지만, audit 결과를 본 뒤 대안을 고안·비교했으므로 대안 점수를 새로운 독립 검증으로 해석하지 않는다. 원본 모델·설정·제출 CSV는 변경하지 않았다.

## 주된 원인: 시간에 따라 부호가 바뀐 편향에 고정 보정을 적용

- 원래 가중치를 그대로 두고 로그 보정만 제거하면 audit RMSLE가 **{original.rmsle:.9f} → {uncalibrated.rmsle:.9f}**로 개선된다.
- 기존 로그 보정값은 **{frozen['bias']:.9f}**다. `(예측+1)`에 **{np.exp(frozen['bias']):.6f}**를 곱하는 것과 같다.
- audit의 보정 전 평균 로그 오차는 **{audit_row.mean_log_error_before:.9f}**로 이미 과소예측이다. 같은 음수 보정을 추가하여 **{audit_row.mean_log_error_after:.9f}**가 됐다.
- 로그 보정이 추가한 audit log-MSE는 **{summary['bias_audit_log_mse_increase']:.9f}**, 원래 audit log-MSE의 **{100*summary['bias_share_of_original_audit_log_mse']:.2f}%**다. 보정 없는 상태보다 log-MSE가 **{100*summary['bias_inflation_relative_to_uncalibrated_audit_log_mse']:.2f}%** 증가했다.
- 감사 구간에서 사후 관찰한 최선의 단일 모델 LightGBM large와 원래 앙상블의 log-MSE 차이 중 **{100*summary['bias_share_of_gap_to_hindsight_best_single_log_mse']:.2f}%**를 이 보정 하나가 설명한다. 이는 최선 모델을 감사 구간에서 선택해도 된다는 의미가 아니다.

{md_table(decomp, ['population','mean_log_error_before','mean_log_error_after','before_log_mse','after_log_mse','bias_mse_delta_total','clipping_rows'])}

fold 1은 양의 로그 편향이 컸고 fold 2는 거의 0이어서, 합쳐 적합한 음수 상수는 fold 1을 개선하고 fold 2부터 악화시킨다. audit에서는 평균 로그 편향이 더욱 음수가 되어 악화가 커졌다. 평균값 하나를 안정적인 모집단 상수로 옮긴 것이 핵심 문제다.

## 로그 MSE 분해와 앙상블 자체의 이점

`MSE(log error) = Var(log error) + Mean(log error)^2`를 모집단 분산(ddof=0)으로 계산했다. audit의 기존 MSE는 분산 **{original.log_error_variance:.9f}**와 제곱 편향 **{original.squared_log_bias:.9f}**의 합이다. 제곱 편향 비중은 **{100*summary['original_audit_squared_log_bias_fraction']:.2f}%**다.

모든 구간에서 보정 후 0 절삭 행은 0개다. 따라서 보정의 MSE 영향은 정확히 `2*b*Mean(error_before) + b^2`이며 분산은 바뀌지 않는다. 음수 절삭이나 CSV 오염이 이번 악화 원인이 아니다.

양의 가중치 로그 앙상블에는 `weighted component log-MSE - ambiguity = uncalibrated blend log-MSE`가 성립한다. 여기서 ambiguity는 각 모델 로그 예측이 앙상블 로그 예측과 다른 정도의 가중 제곱 평균이다. 이 식은 RMSLE 자체에 직접 적용하지 않는다.

{md_table(decomp, ['population','weighted_component_log_mse','ambiguity_log_mse_gain','before_log_mse','bias_mse_delta_total'])}

즉, 예측을 섞는 과정 자체는 audit에서도 분산/오차 감소 효과가 있고, 고정 보정이 그 이점 일부를 되돌린다. 정확한 동일성 검사의 최대 수치 오차는 `log_mse_bias_ambiguity_decomposition.csv`에 있다.

## 개발 구간만 적합한 대안의 사후 비교

{md_table(comparison)}

`frozen_development_best_single`은 원래 개발 OOF에서 고른 CatBoost다. `single_lightgbm_large`는 감사 결과의 참고 비교이며 실제 선택에 사용하지 않았다. `refit_dev2_only`는 최근 개발 fold 2만으로 가중치와 편향을 다시 적합했다. `no_bias`는 같은 가중치에서 상수만 제거했다. 단순 평균은 가중치를 학습하지 않았고 `raw`/`log`의 평균 공간을 이름에 표시했다.

## 잔차 상관 0.95 필터의 영향

원래 pooled 개발 점수가 더 나쁜 LightGBM large가 LightGBM small과의 로그 잔차 상관이 0.95를 넘어 제외됐다. 상관은 중복성이고, 미래 성능 우열을 보장하지 않는다.

{md_table(lgb_corr, ['population','space','correlation'])}

상관 필터를 끈 pooled 개발 재적합 및 small 가중치를 large로 그대로 옮긴 ablation 결과를 위 표에 따로 제시했다. 필터 제거가 pooled 개발 목적함수에서 large를 실제로 선택했는지는 `diagnostic_plan_weights.csv`로 확인한다. 필터가 무엇을 바꿨는지와 부호가 바뀐 편향 보정 문제를 구별해야 한다.

## 첫 개발 fold의 신규역과 구간 이동

첫 origin 이후 관측되는 역은 {cold_ids}다. 아래는 fold 1 신규역의 행 비중과 전체 log-MSE에 대한 기여다. 이 구간은 테스트의 모든 역에 과거 이력이 존재하는 조건과 다르다.

{md_table(cold_contrib, ['candidate','rows','row_fraction','rmsle','contribution_to_population_log_mse','share_of_population_squared_log_error'])}

후보의 성능 순위와 편향은 개발 fold 사이에서도 크게 달라진다. 이는 단순한 개발 표본 내 과적합만으로 단정할 수 없고, 신규역·계절·학습 이력 길이가 함께 바뀌는 검증 분포 이동의 증거다. `refit_pooled_existing_stations_only`는 신규역을 제외한 개발 자료에서만 재적합한 진단이다. 신규역 처리 정책과 새 검증 구간을 먼저 정한 뒤 수정해야 한다.

## 재현 파일

- 실행: `python analysis/ensemble/analyze_ensemble.py --project /Users/hongtaeyang/Desktop/SINCON`
- `candidate_scores.csv`: 모든 고정·진단 후보의 개발/감사 RMSLE, 로그 평균오차·분산·원단위 편향.
- `log_mse_bias_ambiguity_decomposition.csv`: 보정 영향과 ambiguity 항등식.
- `residual_correlation_pairs.csv`, `residual_correlations_*.csv`: 원단위/로그 잔차 상관.
- `station_cohort_contributions.csv`, `monthly_transport.csv`: 신규역과 월별 이동.
- `diagnostic_plans_NOT_DEPLOYED.json`, `diagnostic_plan_weights.csv`: 적합 모집단과 대안 가중치.
- `input_sha256.json`: 원본 코드·OOF·설정·메타데이터 지문.

새 모델이나 제출 파일을 생성하지 않았다. 이 진단에서 좋아 보인 대안은 다른 손대지 않은 시간 구간에서 확인하기 전까지 새 최종 모델로 부르지 않는다.
'''
    (out / "ENSEMBLE_DIAGNOSIS_DRAFT.md").write_text(draft, encoding="utf-8")
    print(json.dumps(native({"audit_rmsle_original": original.rmsle, "audit_rmsle_no_bias": uncalibrated.rmsle, "bias_log_mse_delta": summary["bias_audit_log_mse_increase"], "output": str(out)}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
