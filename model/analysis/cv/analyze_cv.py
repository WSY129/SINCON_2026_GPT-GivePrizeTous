"""Read-only SINCON CV audit; write derived evidence beside this script.

No model is retrained and no Desktop file is written. Reconstructs the exact
anchor sampler from features.py using the original row order and seed.
All outcome data comes from the provided train.csv / existing OOF predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    se = (np.log1p(p) - np.log1p(y)) ** 2
    return {"rows": len(y), "rmsle": float(np.sqrt(se.mean())),
            "log_sse": float(se.sum()), "mae": float(np.abs(p-y).mean()),
            "target_mean": float(y.mean()), "target_median": float(np.median(y))}


def finite_summary(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return {"coverage": float(len(finite)/len(values)),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None}


def sample_records(train, cutoff, cfg):
    horizon = cfg["horizon_days"]
    start = train.date.min() + pd.Timedelta(days=27)
    end = cutoff - pd.Timedelta(days=horizon)
    origins = pd.date_range(start, end, freq=f'{cfg["origin_stride_days"]}D')
    if len(origins) == 0:
        raise ValueError("No full-horizon anchor")
    if origins[-1] != end:
        origins = origins.append(pd.DatetimeIndex([end]))
    rng = np.random.default_rng(cfg["seed"])
    per_origin = max(1, cfg["max_training_rows"] // len(origins))
    records, audit = [], []
    for origin in origins:
        eligible = train.loc[(train.date > origin) &
                             (train.date <= origin + pd.Timedelta(days=horizon))]
        sample = eligible
        if len(sample) > per_origin:
            sample = sample.iloc[np.sort(rng.choice(len(sample), per_origin, replace=False))]
        rows = sample[["date", "station_id", "hour", "boardings"]].copy()
        rows["target_row"] = sample.index
        rows["origin"] = origin
        rows["horizon"] = (rows.date-origin).dt.days
        records.append(rows)
        audit.append({"origin": str(origin.date()), "candidate_rows": len(eligible),
                      "sample_rows": len(rows),
                      "sample_probability": len(rows)/len(eligible),
                      "sample_label_start": str(rows.date.min().date()),
                      "sample_label_end": str(rows.date.max().date()),
                      "theoretical_label_end": str((origin+pd.Timedelta(days=horizon)).date())})
    return pd.concat(records, ignore_index=True), audit


def feature_scope(rows, origins, first_seen, observed_keys):
    # Reproduces history_days/cold_start/year364 without recomputing 60 features.
    first = rows.station_id.map(first_seen)
    observed = first <= origins
    history_days = ((origins-first).dt.days + 1).where(observed)
    lag_date = rows.date - pd.Timedelta(days=364)
    lag_keys = pd.MultiIndex.from_arrays([lag_date, rows.station_id, rows.hour])
    year_available = (lag_date <= origins) & lag_keys.isin(observed_keys)
    return {"history_days": history_days, "cold_start": ~observed,
            "year364_available": year_available,
            "date_index": (rows.date-pd.Timestamp("2025-01-01")).dt.days,
            "horizon": (rows.date-origins).dt.days}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("/Users/hongtaeyang/Desktop/SINCON"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.baseline_root.resolve(), args.output.resolve()
    if root == out or root in out.parents:
        raise ValueError("Output must be outside baseline root")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/"outputs/baseline/config_resolved.json").read_text())
    train = pd.read_csv(root/"data/raw/train.csv", parse_dates=["date"])
    test = pd.read_csv(root/"data/raw/test.csv", parse_dates=["date"])
    first_seen = train.groupby("station_id").date.min()
    observed_keys = pd.MultiIndex.from_frame(train[["date", "station_id", "hour"]])
    source_files = [root/"src/features.py", root/"src/pipeline.py", root/"src/ensemble.py",
                    root/"configs/default.json", root/"data/raw/train.csv",
                    root/"data/raw/test.csv", root/"outputs/baseline/ensemble_plan.json"]
    source_files += [root/f"outputs/baseline/fold_{i}/oof.csv.gz" for i in (1,2,3)]
    before = {str(p.relative_to(root)): digest(p) for p in source_files}
    stage_rows, anchor_rows, frequency_rows, month_rows = [], [], [], []
    multiplicity_rows, horizon_rows, feature_rows, oof_rows, station_history_rows = [], [], [], [], []
    summary, fold_oof = {}, {}
    stages = [(f"fold_{i+1}", pd.Timestamp(o), cfg["fold_roles"][i])
              for i,o in enumerate(cfg["fold_origins"])]
    stages.append(("final", train.date.max(), "deployment"))
    for stage, cutoff, role in stages:
        sampled, anchors = sample_records(train, cutoff, cfg)
        recorded_path = (root/"outputs/baseline/final_training_origins.json" if stage=="final"
                         else root/f"outputs/baseline/{stage}/training_origins.json")
        recorded = json.loads(recorded_path.read_text())
        assert len(recorded)==len(anchors)
        for actual,expected in zip(anchors,recorded):
            assert actual["origin"]==expected["origin"]
            assert actual["sample_rows"]==expected["rows"]
            assert actual["theoretical_label_end"]==expected["label_end"]
        for row in anchors:
            anchor_rows.append({"stage": stage, **row})
        query = test if stage == "final" else train.loc[(train.date>cutoff)&
                  (train.date<=cutoff+pd.Timedelta(days=cfg["horizon_days"]))].copy()
        query = query.reset_index(drop=True)
        train_scope = feature_scope(sampled, sampled.origin, first_seen, observed_keys)
        query_scope = feature_scope(query, pd.Series(cutoff,index=query.index), first_seen, observed_keys)
        counts = sampled.target_row.value_counts()
        eligible_history = train.loc[(train.date > pd.Timestamp(anchors[0]["origin"])) &
                                     (train.date <= cutoff)]
        record = {"stage": stage, "role": role, "cutoff": str(cutoff.date()),
                  "query_start": str(query.date.min().date()), "query_end": str(query.date.max().date()),
                  "anchors": len(anchors), "training_rows": len(sampled),
                  "unique_target_rows": len(counts),
                  "repeated_target_excess_rows": int(len(sampled)-len(counts)),
                  "repeated_target_excess_fraction": float(1-len(counts)/len(sampled)),
                  "max_target_repetitions": int(counts.max()),
                  "eligible_unique_target_rows": len(eligible_history),
                  "unique_target_coverage": float(len(counts)/len(eligible_history)),
                  "training_cold_start_rows": int(train_scope["cold_start"].sum()),
                  "training_cold_start_fraction": float(train_scope["cold_start"].mean()),
                  "query_rows": len(query), "query_cold_start_rows": int(query_scope["cold_start"].sum()),
                  "query_cold_start_fraction": float(query_scope["cold_start"].mean()),
                  "query_known_station_count": int(query.loc[~query_scope["cold_start"],"station_id"].nunique()),
                  "query_cold_station_count": int(query.loc[query_scope["cold_start"],"station_id"].nunique()),
                  "training_year364_coverage": float(train_scope["year364_available"].mean()),
                  "query_year364_coverage": float(query_scope["year364_available"].mean()),
                  "latest_anchor": anchors[-1]["origin"],
                  "latest_sampled_target_date": str(sampled.date.max().date()),
                  "last_28_days_sample_rows": int((sampled.date>cutoff-pd.Timedelta(days=28)).sum()),
                  "last_28_days_sample_fraction": float((sampled.date>cutoff-pd.Timedelta(days=28)).mean()),
                  "last_28_days_unique_coverage": float(len(sampled.loc[sampled.date>cutoff-pd.Timedelta(days=28),"target_row"].unique()) /
                      len(train.loc[(train.date>cutoff-pd.Timedelta(days=28))&(train.date<=cutoff)]))}
        stage_rows.append(record)
        summary[stage] = {**record, "train_target": finite_summary(sampled.boardings),
                          "query_target": finite_summary(query.boardings) if "boardings" in query else None}
        for feature in train_scope:
            left, right = train_scope[feature], query_scope[feature]
            for partition, vals in [("sampled_training",left),("forecast_query",right)]:
                feature_rows.append({"stage":stage,"partition":partition,"feature":feature,**finite_summary(vals)})
        history_table = sampled[["station_id"]].copy()
        history_table["history_days"] = train_scope["history_days"].to_numpy()
        station_train = history_table.groupby("station_id").history_days.agg(["min","max","count","size"])
        query_history = query[["station_id"]].copy()
        query_history["history_days"] = query_scope["history_days"].to_numpy()
        for station,frame in query_history.groupby("station_id"):
            train_station = station_train.loc[station] if station in station_train.index else None
            query_days = frame.history_days.iloc[0]
            train_max = float(train_station["max"]) if train_station is not None else float("nan")
            station_history_rows.append({"stage":stage,"station_id":int(station),
                "observed_first_date":str(first_seen.loc[station].date()),
                "training_rows":int(train_station["size"]) if train_station is not None else 0,
                "training_known_history_rows":int(train_station["count"]) if train_station is not None else 0,
                "training_history_days_min":float(train_station["min"]) if train_station is not None else None,
                "training_history_days_max":train_max,"query_history_days":query_days,
                "query_rows":len(frame),"query_cold":bool(pd.isna(query_days)),
                "query_known_history_outside_training_support":bool(pd.notna(query_days) and
                    (pd.isna(train_max) or query_days>train_max))})
        for n, number in counts.value_counts().sort_index().items():
            multiplicity_rows.append({"stage":stage,"target_repetitions":int(n),"distinct_targets":int(number)})
        by_date = sampled.groupby("date").agg(sample_count=("target_row","size"),unique_targets=("target_row","nunique"))
        available = eligible_history.groupby("date").size().rename("observed_rows")
        by_date = available.to_frame().join(by_date).fillna(0).reset_index()
        by_date["stage"]=stage
        by_date["effective_samples_per_observation"]=by_date.sample_count/by_date.observed_rows
        frequency_rows.extend(by_date.to_dict("records"))
        sampled_month = sampled.assign(month=sampled.date.dt.strftime("%Y-%m")).groupby("month").agg(
            sample_count=("target_row","size"), unique_targets=("target_row","nunique"), target_mean=("boardings","mean"))
        query_month = query.assign(month=query.date.dt.strftime("%Y-%m")).groupby("month").size()
        for month,row in sampled_month.iterrows():
            month_rows.append({"stage":stage,"partition":"sampled_training","month":month,
                               **row.to_dict(),"row_fraction":row.sample_count/len(sampled)})
        for month,number in query_month.items():
            month_rows.append({"stage":stage,"partition":"forecast_query","month":month,
                               "sample_count":int(number),"row_fraction":int(number)/len(query)})
        for partition, scope in [("sampled_training",train_scope),("forecast_query",query_scope)]:
            for h, number in scope["horizon"].value_counts().sort_index().items():
                horizon_rows.append({"stage":stage,"partition":partition,"horizon":int(h),"rows":int(number)})
        if stage != "final":
            oof = pd.read_csv(root/f"outputs/baseline/{stage}/oof.csv.gz", parse_dates=["date"])
            assert oof[["date","station_id","hour"]].equals(query[["date","station_id","hour"]])
            np.testing.assert_array_equal(oof.boardings.to_numpy(),query.boardings.to_numpy())
            oof["cold_start"] = query_scope["cold_start"].to_numpy()
            oof["fold"] = int(stage[-1])
            fold_oof[stage] = oof
            for model in cfg["models"]+["ensemble"]:
                all_score = metrics(oof.boardings,oof[model])
                for segment,mask in [("all",np.ones(len(oof),bool)),("known",~oof.cold_start),
                                     ("cold",oof.cold_start)]:
                    if not np.any(mask): continue
                    row = metrics(oof.loc[mask,"boardings"],oof.loc[mask,model])
                    oof_rows.append({"stage":stage,"model":model,"segment":segment,**row,
                                     "fraction_of_stage_log_sse":row["log_sse"]/all_score["log_sse"]})
    dev = pd.concat([fold_oof["fold_1"],fold_oof["fold_2"]],ignore_index=True)
    for model in cfg["models"]+["ensemble"]:
        all_score=metrics(dev.boardings,dev[model])
        for segment,mask in [("all",np.ones(len(dev),bool)),("known",~dev.cold_start),
                             ("cold",dev.cold_start), ("fold_1",dev.fold.eq(1)),("fold_2",dev.fold.eq(2))]:
            row=metrics(dev.loc[mask,"boardings"],dev.loc[mask,model])
            oof_rows.append({"stage":"pooled_development","model":model,"segment":segment,**row,
                             "fraction_of_stage_log_sse":row["log_sse"]/all_score["log_sse"]})
    frames = {"stage_summary":stage_rows,"anchor_sampling":anchor_rows,"target_date_weights":frequency_rows,
              "training_months":month_rows,"target_multiplicity":multiplicity_rows,
              "horizon_coverage":horizon_rows,"feature_distribution":feature_rows,
              "oof_known_vs_cold":oof_rows,"station_history_distribution":station_history_rows}
    for name,rows in frames.items():
        pd.DataFrame(rows).to_csv(out/f"{name}.csv",index=False)
    after={str(p.relative_to(root)):digest(p) for p in source_files}
    assert before==after,"Baseline source/input changed during audit"
    document={"baseline_root":str(root),"analysis_date":"2026-11-11 Asia/Seoul",
              "source_sha256":before,"baseline_files_unchanged":True,"config":cfg,
              "summary":summary,"read_only":True,
              "caveat":"Existing audit outcomes were already reviewed. All new comparisons are exploratory; no fresh holdout claim."}
    (out/"cv_audit.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n")
    print(pd.DataFrame(stage_rows).to_string(index=False))
    oof_table=pd.DataFrame(oof_rows)
    print("\nOOF known/cold: RMSLE and squared-log error share")
    print(oof_table.loc[oof_table.stage.isin(["fold_1","pooled_development"]),
          ["stage","model","segment","rows","rmsle","fraction_of_stage_log_sse"]].to_string(index=False))


if __name__ == "__main__":
    main()
