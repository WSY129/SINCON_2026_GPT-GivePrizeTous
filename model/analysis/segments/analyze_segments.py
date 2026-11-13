"""Attribute baseline OOF error without editing/retraining the baseline.

Only observed training-period OOF labels are read. No test truth or network.
Example:
python analysis/segments/analyze_segments.py --project-dir ~/Desktop/SINCON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["seasonal7", "weekday_mean", "lightgbm_small", "lightgbm_large",
          "catboost", "xgboost", "mlp", "ensemble", "blend_without_bias"]


def mdtable(frame, limit=20):
    f = frame.head(limit)
    lines = ["| " + " | ".join(map(str, f.columns)) + " |",
             "| " + " | ".join(["---"] * len(f.columns)) + " |"]
    for row in f.itertuples(index=False, name=None):
        parts=[]
        for v in row:
            if isinstance(v, (float, np.floating)):
                parts.append(f"{v:.6f}" if abs(v) < 10 else f"{v:,.2f}")
            else:
                parts.append(str(v))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def analyze(project: Path, out: Path):
    baseline=project/"outputs"/"baseline"
    sys.path.insert(0,str(project))
    from src.holiday_calendar import calendar_features
    out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((baseline/"ensemble_plan.json").read_text())
    fold_info=json.loads((baseline/"cv_folds.json").read_text())
    meta=pd.read_csv(project/"data"/"raw"/"metaData.csv")
    frames=[]
    sources=[baseline/"ensemble_plan.json",baseline/"cv_folds.json",project/"data"/"raw"/"metaData.csv",project/"src"/"holiday_calendar.py"]
    for fold in fold_info:
        file=baseline/f"fold_{fold['fold']}"/"oof.csv.gz"
        sources.append(file)
        d=pd.read_csv(file,parse_dates=["date"])
        assert d.date.max() <= pd.Timestamp("2026-06-30"), "Only training-period OOF labels may be read"
        assert not d.duplicated(["date","station_id","hour"]).any()
        d["fold"]=fold["fold"]
        d["origin"]=fold["origin"]
        d["cold_status"]=np.where((fold["fold"]==1)&d.station_id.isin([3107,3108,3109]),"cold_3_new_stations","warm")
        d["station_cohort"]=np.where(d.station_id.isin([3107,3108,3109]),"three_extension_stations","original_68_stations")
        holidays=calendar_features(d.date,as_of=fold["origin"])
        d["holiday_known_at_origin"]=holidays.is_holiday.to_numpy()
        d["holiday_type_known"]=holidays.holiday_type.to_numpy()
        # Diagnostic cross-check only; this retrospective field is never a
        # model input and exposes which later declarations were unknowable.
        final_holidays=calendar_features(d.date,as_of="2026-06-30")
        d["holiday_final_calendar"]=final_holidays.is_holiday.to_numpy()
        d["newly_declared_holiday_after_origin"]=((d.holiday_final_calendar==1)&(d.holiday_known_at_origin==0)).astype("int8")
        d["is_weekend"]=(d.date.dt.dayofweek>=5).astype("int8")
        frames.append(d)
    data=pd.concat(frames,ignore_index=True).merge(meta[["station_id","station_name","canonical_line"]],on="station_id",validate="many_to_one")
    data["month"]=data.date.dt.strftime("%Y-%m")
    data["horizon_bin"]=pd.cut(data.horizon,[0,7,28,56,84,123],labels=["001-007","008-028","029-056","057-084","085-123"]).astype(str)
    data["day_of_week"]=data.date.dt.dayofweek
    data["special_3127_night"] = np.where((data.station_id==3127)&(data.hour>=20),"3127_20to23","other")
    data["spike_label_for_diagnosis_only"] = np.where((data.station_id==3127)&(data.hour>=20)&(data.boardings>=1000),"3127_night_actual_ge1000","other")
    data["all_rows"]="all"
    ylog=np.log1p(data.boardings.to_numpy())
    blend_log=sum(weight*np.log1p(data[model].to_numpy()) for model,weight in plan["weights"].items())
    data["blend_without_bias"]=np.expm1(np.maximum(blend_log,0.0))
    reconstructed=np.expm1(np.maximum(blend_log+plan["bias"],0.0))
    reconstruction_max_error=float(np.max(np.abs(reconstructed-data.ensemble.to_numpy())))
    assert reconstruction_max_error < 1e-8, "Saved ensemble disagrees with saved weights/bias"
    for model in MODELS:
        loge=np.log1p(data[model].to_numpy())-ylog
        data[f"{model}_l2"]=loge**2
        data[f"{model}_log_bias"]=loge
        data[f"{model}_raw_bias"]=data[model]-data.boardings
        data[f"{model}_raw_abs"]=np.abs(data[model]-data.boardings)
        data[f"{model}_delta_l2"]=data[f"{model}_l2"]-data.seasonal7_l2
    scopes={"fold_1":data[data.fold==1],"fold_2":data[data.fold==2],"audit_fold_3":data[data.fold==3],
            "development_folds_1_2":data[data.fold<3],"all_oof":data}
    dimensions=["all_rows","cold_status","station_cohort","station_id","canonical_line","hour","month","horizon_bin",
                "holiday_known_at_origin","holiday_type_known","holiday_final_calendar","newly_declared_holiday_after_origin",
                "is_weekend","day_of_week","special_3127_night","spike_label_for_diagnosis_only"]
    result=[]
    for scope,frame in scopes.items():
        n_total=len(frame)
        for dim in dimensions:
            group=frame.groupby(dim,sort=True,observed=True)
            count=group.size()
            truths=group.boardings.mean()
            for model in MODELS:
                sums=group[[f"{model}_l2",f"{model}_delta_l2"]].sum()
                means=group[[f"{model}_log_bias",f"{model}_raw_bias",f"{model}_raw_abs"]].mean()
                for value in count.index:
                    n=int(count.loc[value]); l2=float(sums.loc[value,f"{model}_l2"])
                    delta=float(sums.loc[value,f"{model}_delta_l2"])
                    result.append({"scope":scope,"dimension":dim,"segment":str(value),"model":model,"n":n,
                                   "row_share":n/n_total,"truth_mean":float(truths.loc[value]),
                                   "rmsle":float(np.sqrt(l2/n)),"msle":l2/n,"sum_log_squared_error":l2,
                                   "msle_contribution_to_scope":l2/n_total,
                                   "log_mbe":float(means.loc[value,f"{model}_log_bias"]),
                                   "raw_mbe":float(means.loc[value,f"{model}_raw_bias"]),
                                   "mae":float(means.loc[value,f"{model}_raw_abs"]),
                                   "delta_sum_lse_vs_seasonal7":delta,"delta_msle_contribution_vs_seasonal7":delta/n_total})
    segments=pd.DataFrame(result)
    segments.to_csv(out/"segment_metrics.csv",index=False)
    overall=segments[segments.dimension=="all_rows"].copy()
    overall.to_csv(out/"model_fold_summary.csv",index=False)
    audit=scopes["audit_fold_3"]
    audit_station=segments[(segments.scope=="audit_fold_3")&(segments.dimension=="station_id")&(segments.model=="ensemble")].copy()
    audit_station["station_id"]=audit_station.segment.astype(int)
    audit_station=audit_station.merge(meta[["station_id","station_name","canonical_line"]],on="station_id")
    audit_station=audit_station.sort_values("delta_sum_lse_vs_seasonal7",ascending=False)
    audit_station.to_csv(out/"audit_station_attribution.csv",index=False)
    # All listed partitions must exactly reconstruct the row-weighted score.
    for scope,frame in scopes.items():
        for dim in dimensions:
            for model in ["ensemble","seasonal7"]:
                part=segments[(segments.scope==scope)&(segments.dimension==dim)&(segments.model==model)]
                assert part.n.sum()==len(frame)
                np.testing.assert_allclose(part.msle_contribution_to_scope.sum(),frame[f"{model}_l2"].mean(),rtol=1e-12,atol=1e-12)
    daily=audit.groupby("date").agg(n=("boardings","size"),ensemble_lse=("ensemble_l2","sum"),seasonal_lse=("seasonal7_l2","sum"),unbiased_lse=("blend_without_bias_l2","sum"))
    daily["delta_lse"]=daily.ensemble_lse-daily.seasonal_lse
    daily.to_csv(out/"audit_daily_errors.csv")
    rng=np.random.default_rng(2026)
    indices=rng.integers(0,len(daily),size=(5000,len(daily)))
    sampled_n=daily.n.to_numpy()[indices].sum(axis=1)
    boot_delta=np.sqrt(daily.ensemble_lse.to_numpy()[indices].sum(axis=1)/sampled_n)-np.sqrt(daily.seasonal_lse.to_numpy()[indices].sum(axis=1)/sampled_n)
    cold=scopes["fold_1"].cold_status=="cold_3_new_stations"
    cold_shares={model:float(scopes["fold_1"].loc[cold,f"{model}_l2"].sum()/scopes["fold_1"][f"{model}_l2"].sum()) for model in MODELS}
    mean_before=float(audit.blend_without_bias_log_bias.mean()); b=float(plan["bias"])
    exact_calibration_delta=float((audit.ensemble_l2-audit.blend_without_bias_l2).mean())
    net_delta=float(audit.ensemble_l2.mean()-audit.seasonal7_l2.mean())
    calibrated_correction=2*b*mean_before+b*b
    audit_no3226=audit[audit.station_id!=3226]
    known_holiday_rows=audit[audit.holiday_known_at_origin==1]
    ordinary_rows=audit[audit.holiday_known_at_origin==0]
    report={
        "input_source":str(baseline),"row_counts":{key:len(frame) for key,frame in scopes.items()},
        "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "numpy_version":np.__version__,"pandas_version":pd.__version__,
        "weights":plan["weights"],"log_bias_correction":b,"ensemble_reconstruction_max_abs_difference":reconstruction_max_error,
        "audit":{"ensemble_rmsle":float(np.sqrt(audit.ensemble_l2.mean())),"seasonal7_rmsle":float(np.sqrt(audit.seasonal7_l2.mean())),
                  "delta_rmsle":float(np.sqrt(audit.ensemble_l2.mean())-np.sqrt(audit.seasonal7_l2.mean())),
                  "delta_msle":net_delta,"delta_sum_lse":float((audit.ensemble_l2-audit.seasonal7_l2).sum()),
                  "blend_without_bias_rmsle_diagnostic_only":float(np.sqrt(audit.blend_without_bias_l2.mean())),
                  "calibration_added_msle":exact_calibration_delta,"calibration_formula_2b_mean_e_plus_b2":calibrated_correction,
                  "calibration_added_msle_relative_to_net_gap":exact_calibration_delta/net_delta,
                  "paired_date_bootstrap_delta_rmsle_ci95":np.quantile(boot_delta,[.025,.975]).tolist(),
                  "paired_date_bootstrap_p_delta_le_zero":float((boot_delta<=0).mean()),
                  "excluding_station_3226_ensemble_rmsle":float(np.sqrt(audit_no3226.ensemble_l2.mean())),
                  "excluding_station_3226_seasonal7_rmsle":float(np.sqrt(audit_no3226.seasonal7_l2.mean())),
                  "known_holiday_ensemble_lse_share":float(known_holiday_rows.ensemble_l2.sum()/audit.ensemble_l2.sum()),
                  "known_holiday_row_share":len(known_holiday_rows)/len(audit),
                  "nonholiday_delta_sum_lse":float((ordinary_rows.ensemble_l2-ordinary_rows.seasonal7_l2).sum()),
                  "holiday_delta_sum_lse":float((known_holiday_rows.ensemble_l2-known_holiday_rows.seasonal7_l2).sum())},
        "fold1_cold":{"rows":int(cold.sum()),"row_share":float(cold.mean()),"share_of_model_log_squared_error":cold_shares},
        "limitations":["RMSLE is not additive: partition contributions are log-squared-error sums or MSLE contributions.",
                       "Each dimension separately partitions the same rows; do not sum contributions across dimensions.",
                       "Removing calibration is a diagnostic counterfactual, not a newly validated model selection.",
                       "Audit labels have now informed diagnosis; subsequent tuning needs a new nested/rolling validation protocol.",
                       "Bootstrap resamples dates jointly across models, not rows; conditional interval omits model-selection and future-regime uncertainty.",
                       "The spike label uses OOF truth only for explanation, never for production routing."]}
    (out/"segment_analysis.json").write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False))
    def sel(scope,dim,models=("seasonal7","lightgbm_small","lightgbm_large","catboost","ensemble","blend_without_bias")):
        return segments[(segments.scope==scope)&(segments.dimension==dim)&segments.model.isin(models)].copy()
    audit_model=sel("audit_fold_3","all_rows")[["model","rmsle","log_mbe","raw_mbe"]]
    fold_model=sel("fold_1","cold_status")[["segment","model","n","rmsle","log_mbe","raw_mbe"]]
    warm_development=sel("development_folds_1_2","cold_status")
    warm_development=warm_development[warm_development.segment=="warm"][["model","rmsle","log_mbe","raw_mbe"]]
    month=sel("audit_fold_3","month",("ensemble",))[["segment","n","rmsle","log_mbe","delta_sum_lse_vs_seasonal7","delta_msle_contribution_vs_seasonal7"]]
    horizons=sel("audit_fold_3","horizon_bin",("seasonal7","lightgbm_large","catboost","ensemble"))[["segment","model","rmsle","log_mbe","raw_mbe"]]
    holiday=sel("audit_fold_3","holiday_known_at_origin")[["segment","model","n","rmsle","log_mbe","delta_sum_lse_vs_seasonal7"]]
    line=sel("audit_fold_3","canonical_line",("ensemble",))[["segment","n","rmsle","log_mbe","delta_sum_lse_vs_seasonal7"]]
    hours=sel("audit_fold_3","hour",("ensemble",))[["segment","rmsle","log_mbe","delta_sum_lse_vs_seasonal7"]].sort_values("delta_sum_lse_vs_seasonal7",ascending=False)
    night=sel("audit_fold_3","special_3127_night")
    night=night[night.segment=="3127_20to23"][["model","n","rmsle","log_mbe","raw_mbe","msle_contribution_to_scope","delta_sum_lse_vs_seasonal7"]]
    station=audit_station[["station_id","station_name","canonical_line","rmsle","log_mbe","raw_mbe","delta_sum_lse_vs_seasonal7"]]
    text=f'''# Baseline OOF 구간별 오차 기여 분석

분석일 2026-11-11. 입력은 바탕화면 SINCON의 **고정된 baseline OOF**와 저장된 앙상블 계획이다. 원본·가중치·예측은 변경하지 않았다. 2026-07월 이후 정답이나 웹 자료를 사용하지 않았다.

## 핵심 진단

Audit의 ensemble RMSLE는 {report['audit']['ensemble_rmsle']:.6f}, seasonal7은 {report['audit']['seasonal7_rmsle']:.6f}로 차이는 **{report['audit']['delta_rmsle']:+.6f}**다. MSLE 차이는 {net_delta:+.8f}, log squared error 합 차이는 {report['audit']['delta_sum_lse']:+.3f}다. RMSLE 자체를 구간별로 합하지 않는다.

1. **음의 전역 로그 보정이 audit의 과소예측을 더 키웠다.** 저장된 보정 b={b:.6f}를 동일 가중치에서 제거하는 진단 결과는 RMSLE {report['audit']['blend_without_bias_rmsle_diagnostic_only']:.6f}다. 보정은 audit MSLE를 {exact_calibration_delta:+.8f} 증가시켰다. 이는 net ensemble-seasonal 격차의 {exact_calibration_delta/net_delta:.2f}배이며, 보정 전 앙상블의 이점과 상쇄되어 작은 최종 격차가 된다. 이 결과만 보고 production 보정을 새로 선택하면 audit를 튜닝에 사용한 것이므로 별도 검증이 필요하다.
2. **인천대공원(3226)의 봄 수요 과소예측**이 net 격차보다 큰 악화 기여를 낸다. 이 역을 진단 목적으로 제외하면 ensemble {report['audit']['excluding_station_3226_ensemble_rmsle']:.6f}, seasonal7 {report['audit']['excluding_station_3226_seasonal7_rmsle']:.6f}다. 제외 제출을 제안하는 것이 아니다.
3. **Fold1의 cold 3역**은 {int(cold.sum()):,}행({cold.mean()*100:.2f}%)이지만 모델 선택 지표에 불균형하게 큰 영향을 준다. 테스트에는 이런 완전 신규 역이 없으므로 warm 성능을 함께 보아야 한다. 다만 cold만 제거해도 개발 구간 로그 편향은 양수이므로 전역 bias 문제를 cold만으로 설명할 수 없다.
4. **문학경기장(3127) 야간 급등**은 절대 오차의 큰 원인이지만 seasonal7 대비 ensemble 악화의 주원인은 아니다. 해당 역 전체의 ensemble은 seasonal7보다 개선한다.

날짜를 두 모델에 동일하게 재표집한 5,000회 paired bootstrap에서 RMSLE 차이의 95% 구간은 **[{report['audit']['paired_date_bootstrap_delta_rmsle_ci95'][0]:+.6f}, {report['audit']['paired_date_bootstrap_delta_rmsle_ci95'][1]:+.6f}]**다. 이는 현재 audit 날짜에 조건부인 기술적 구간이며 미래 분포나 선택 불확실성을 포함하지 않는다.

## 1. 모델별 audit 편향

`log_MBE = mean(log1p(pred)-log1p(y))`, `raw_MBE = mean(pred-y)`다. 부호가 다를 수 있으므로 공식 로그 손실과 원단위 손실을 구별한다.

{mdtable(audit_model)}

보정에 의한 MSLE 변화는 clipping이 없을 때 `2*b*mean(e_before)+b²`다. 관측된 변화 {exact_calibration_delta:.9f}와 식 계산 {calibrated_correction:.9f}가 일치하는지 JSON에 저장했다. 각 fold의 원래·무보정 로그 편향은 `model_fold_summary.csv`에서 확인할 수 있다.

## 2. Fold1 신규 역과 warm 역

Fold1 origin은 2025-06-26이다. 3107·3108·3109의 최초 관측일은 6/28이어서 validation에서 122일×3역×18시간이 cold다. 6/27의 신규역 54행은 실제 자료에 없으며 0명으로 추가하지 않는다.

{mdtable(fold_model,30)}

개발 1·2폴드 warm만 합친 성능:

{mdtable(warm_development)}

Cold의 log squared error 점유율은 `segment_analysis.json.fold1_cold`에 있다. 개체 ID를 쓰는 트리가 관측 이력 없는 개체에서 무너지고, 평균 기반 fallback도 크게 과대/과소예측할 수 있다. 테스트의 3역은 모두 과거 이력이 있으므로 최종 문제는 cold 일반화와 다르다. Fold1을 없애거나 가중치를 다시 고르는 결정은 추가 시간순 CV로 검증해야 한다.

## 3. 역별 net 오차 차이

양의 Δsum LSE는 ensemble 악화, 음수는 개선이다. 큰 양수·음수가 상쇄될 수 있어 특정 역의 기여가 전체 차이의 100%를 넘을 수 있다.

{mdtable(station,12)}

개선 기여가 큰 역:

{mdtable(station.sort_values('delta_sum_lse_vs_seasonal7'),8)}

## 4. 노선·시간·월·horizon

노선별:

{mdtable(line)}

시간별 악화 순:

{mdtable(hours,18)}

월별:

{mdtable(month)}

예측 거리별:

{mdtable(horizons,30)}

월과 horizon은 강하게 겹치는 분할이다. 4월 효과와 긴 horizon 효과를 독립적인 원인으로 더하지 않는다. 계절 변화, origin의 기준 수요 수준, 모델의 추세 학습 부재가 얽힌 진단이다.

## 5. 공휴일과 일반 날짜

다음 표는 **각 fold origin에 이미 발표된** 공휴일을 사용한다. Audit origin 2026-02-27에는 5/1의 신규 관공서 공휴일 지정이 아직 발표되지 않았으므로 5/1은 이 표에서 일반 날짜다. 최종 달력과의 차이는 별도 차원 `newly_declared_holiday_after_origin`에 남기며 모델에 새로 입력하지 않았다.

{mdtable(holiday,20)}

Audit known holiday 비중은 행 기준 {report['audit']['known_holiday_row_share']*100:.2f}%, ensemble log squared error 기준 {report['audit']['known_holiday_ensemble_lse_share']*100:.2f}%다. 공휴일 Δsum LSE={report['audit']['holiday_delta_sum_lse']:+.3f}, 일반 날짜 Δsum LSE={report['audit']['nonholiday_delta_sum_lse']:+.3f}다.

## 6. 문학경기장 야간

3127번 역 20~23시:

{mdtable(night)}

`spike_label_for_diagnosis_only`의 `boardings>=1000`은 이미 알려진 OOF 정답으로 구간을 설명하는 사후 분류다. 미래에 이런 label을 사용해 모델을 선택할 수 없다. 원인 사건이나 개최 여부를 추정하기 위해 웹을 조회하지 않았다. 경기·행사 특징을 도입한다면 별도 cutoff 준수 조사와 fold별 발표시점 검증이 필요하다.

## 7. 다음 실험의 검증 조건

- 전역 log bias를 개발폴드 전체 평균에 고정하는 대신 무보정·최근 구간 보정·축소 보정을 **개발 구간에서** 비교한다. 이번 audit 점수로 correction을 직접 선택하지 않는다.
- cold 일반화 성능과 실제 테스트 구조인 warm 장기 예측을 분리 보고한다. 완전 신규역 stress test 때문에 이력이 있는 역에 강한 모델을 제외하는지 점검한다.
- 3226의 봄 변화는 train 안의 전년도 월/계절 및 충분한 horizon으로 검증할 후보다. 이 단일 audit 역의 정답 비율을 곱하는 후처리는 검증 근거가 되지 않는다.
- 3127 야간 급등은 전체적인 log/원단위 양쪽 오차를 함께 보고, 이벤트 없는 fallback의 한계를 명시한다.
- 새로운 모델·가중치 실험은 추가 rolling/nested split을 사용한다. 이 보고서 작성 이후 fold3은 진단에 노출된 구간이다.

재현:

```bash
python analysis/segments/analyze_segments.py --project-dir /Users/hongtaeyang/Desktop/SINCON
```

`segment_metrics.csv`는 모든 구간×모델의 행수·RMSLE·MSLE·로그 및 원단위 MBE·LSE 기여를 담는다. 각각의 차원은 전체 행을 완전히 분할하며 합이 전체 MSLE와 일치함을 코드에서 검사한다. 서로 다른 차원의 기여를 합하지 않는다. 입력 SHA-256과 라이브러리 버전은 JSON에 기록했다.
'''
    (out/"ANALYSIS_DRAFT.md").write_text(text,encoding="utf-8")
    print(json.dumps(clean({"audit":report["audit"],"fold1_cold":report["fold1_cold"],"output_dir":str(out)}),ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=Path,default=Path("/Users/hongtaeyang/Desktop/SINCON"))
    parser.add_argument("--out-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    analyze(args.project_dir.resolve(),args.out_dir.resolve())
