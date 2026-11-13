"""Reproducible, read-only SINCON CSV exploratory analysis.

Usage: python src/eda.py --data-dir /path/to/sincon-2026 --out-dir reports
Requires numpy and pandas; the compact SVG chart uses only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if pd.isna(value):
        return None
    return value


def table(frame, limit=20):
    f = frame.head(limit).copy()
    rows = ['| ' + ' | '.join(map(str, f.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(f.columns)) + ' |']
    for record in f.itertuples(index=False, name=None):
        vals = []
        for v in record:
            if isinstance(v, (float, np.floating)):
                vals.append(f'{v:,.4f}' if abs(v) < 10 else f'{v:,.2f}')
            else:
                vals.append(str(native(v)).replace('|', '\\|'))
        rows.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(rows)


def draw_chart(path, series, title, ylabel):
    """Small dependency-free SVG with explicit axes and all source points."""
    width, height = 1100, 340
    left, top, plot_w, plot_h = 82, 55, 980, 220
    values = np.asarray(series.values, dtype=float)
    ymax = max(float(np.nanmax(values)) * 1.08, 1.0)
    xs = np.linspace(left, left + plot_w, len(values))
    ys = top + plot_h - values / ymax * plot_h
    points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#fff"/>',
              '<g font-family="Arial, sans-serif" fill="#172b4d">',
              f'<text x="{left}" y="28" font-size="20" font-weight="bold">{html.escape(title)}</text>',
              f'<text x="{left}" y="46" font-size="12">{html.escape(ylabel)}</text>']
    for i in range(5):
        v = ymax * i / 4
        y = top + plot_h - plot_h * i / 4
        pieces += [f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#e3e8ef"/>',
                   f'<text x="{left-10}" y="{y+4}" font-size="12" text-anchor="end">{v:,.0f}</text>']
    pieces.append(f'<polyline points="{points}" fill="none" stroke="#007e87" stroke-width="1.7"/>')
    for i in sorted(set(np.linspace(0, len(values)-1, min(7, len(values))).astype(int))):
        label = series.index[i]
        if isinstance(label, pd.Timestamp):
            label = label.strftime('%Y-%m-%d')
        pieces.append(f'<text x="{xs[i]}" y="{top+plot_h+25}" font-size="12" text-anchor="middle">{html.escape(str(label))}</text>')
    pieces += ['<text x="82" y="325" font-size="11" fill="#56657a">Source: supplied train.csv; no target values from test.csv are available.</text>', '</g></svg>']
    path.write_text('\n'.join(pieces), encoding='utf-8')


def analyze(data_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = {name: data_dir / f'{name}.csv' for name in ['train', 'test', 'metaData', 'sample_submission']}
    train = pd.read_csv(files['train'], parse_dates=['date'], dtype={'station_id': 'int32', 'hour': 'int8', 'boardings': 'int32', 'alightings': 'int32'})
    test = pd.read_csv(files['test'], parse_dates=['date'], dtype={'station_id': 'int32', 'hour': 'int8'})
    meta = pd.read_csv(files['metaData'], parse_dates=['first_date', 'last_date'], dtype={'station_id': 'int32'})
    sample = pd.read_csv(files['sample_submission'])
    frames = {'train': train, 'test': test, 'metaData': meta, 'sample_submission': sample}
    report = {'source_directory': str(data_dir.resolve()), 'files': {}}
    for name, frame in frames.items():
        report['files'][name] = {
            'rows': len(frame), 'columns': list(frame.columns), 'column_count': frame.shape[1],
            'sha256': hashlib.sha256(files[name].read_bytes()).hexdigest(),
            'bytes': files[name].stat().st_size,
            'missing_cells': frame.isna().sum().to_dict(),
            'memory_bytes': int(frame.memory_usage(deep=True).sum()),
            'dtypes': frame.dtypes.astype(str).to_dict(),
        }
    key = ['date', 'station_id', 'hour']
    train = train.sort_values(['station_id', 'date', 'hour']).reset_index(drop=True)
    hours = sorted(train.hour.unique().tolist())
    station_coverage = train.groupby('station_id').agg(
        first_date=('date', 'min'), last_date=('date', 'max'),
        observed_dates=('date', 'nunique'), rows=('date', 'size'))
    station_coverage['calendar_dates'] = (station_coverage.last_date - station_coverage.first_date).dt.days + 1
    station_coverage['expected_rows_within_observed_lifetime'] = station_coverage.calendar_dates * len(hours)
    station_coverage['missing_rows_within_observed_lifetime'] = station_coverage.expected_rows_within_observed_lifetime - station_coverage.rows
    station_coverage = station_coverage.reset_index().merge(meta[['station_id', 'station_name', 'canonical_line']], on='station_id', validate='one_to_one')
    station_coverage.to_csv(out / 'station_coverage.csv', index=False)
    actual_metadata = station_coverage.merge(meta[['station_id', 'first_date', 'last_date']], on='station_id', suffixes=('_observed', '_metadata'))
    metadata_matches = bool(((actual_metadata.first_date_observed == actual_metadata.first_date_metadata) & (actual_metadata.last_date_observed == actual_metadata.last_date_metadata)).all())
    train_timestamp = train.date + pd.to_timedelta(train.hour.astype(int), unit='h')
    delta = train_timestamp.groupby(train.station_id).diff().dt.total_seconds().dropna() / 3600
    expected_grid = train.date.nunique() * train.station_id.nunique() * len(hours)
    report['coverage'] = {
        'train_start': train.date.min(), 'train_end': train.date.max(), 'train_days': train.date.nunique(),
        'test_start': test.date.min(), 'test_end': test.date.max(), 'test_days': test.date.nunique(),
        'station_count': train.station_id.nunique(), 'hours': hours,
        'train_duplicate_keys': int(train.duplicated(key).sum()), 'test_duplicate_keys': int(test.duplicated(key).sum()),
        'train_test_key_overlap': len(train[key].merge(test[key], on=key)),
        'test_id_duplicates': int(test.example_id.duplicated().sum()),
        'sample_id_duplicates': int(sample.example_id.duplicated().sum()),
        'new_test_stations': sorted(set(test.station_id) - set(train.station_id)),
        'metadata_unique_station_ids': bool(meta.station_id.is_unique),
        'metadata_dates_match_observed': metadata_matches,
        'global_full_grid_rows': expected_grid,
        'global_grid_absent_rows': expected_grid - len(train),
        'within_observed_lifetime_missing_rows': station_coverage.missing_rows_within_observed_lifetime.sum(),
        'station_start_date_counts': station_coverage.first_date.value_counts().sort_index().to_dict(),
        'observed_timestamp_gaps_hours': delta.value_counts().sort_index().to_dict(),
        'test_complete_grid': bool(len(test) == test.date.nunique() * test.station_id.nunique() * test.hour.nunique() and not test.duplicated(key).any()),
        'canonical_line_station_counts': meta.canonical_line.value_counts().sort_index().to_dict(),
    }
    sample_keys = sample[['example_id']].merge(test, on='example_id', how='left', validate='one_to_one', indicator=True)
    uncovered = test.loc[~test.example_id.isin(sample.example_id)]
    report['submission_mismatch'] = {
        'test_rows': len(test), 'sample_rows': len(sample), 'difference': len(test)-len(sample),
        'sample_ids_absent_from_test': int((sample_keys['_merge'] != 'both').sum()),
        'sample_period_start': sample_keys.date.min(), 'sample_period_end': sample_keys.date.max(),
        'sample_days': sample_keys.date.nunique(),
        'uncovered_start': uncovered.date.min(), 'uncovered_end': uncovered.date.max(),
        'sample_matches_test_prefix_order': bool(sample.example_id.tolist() == test.example_id.head(len(sample)).tolist()),
        'sample_values_all_zero': bool((sample.boardings == 0).all()),
        'resolution': 'User-supplied official rules confirm 157194 rows for July through October. Use full test IDs in their original order; the local 117576-row sample is stale/inconsistent and is not the final row-count authority.'
    }
    for target in ['boardings', 'alightings']:
        y = train[target]
        report[target] = {
            'min': y.min(), 'max': y.max(), 'mean': y.mean(), 'std': y.std(), 'median': y.median(),
            'skewness': y.skew(), 'negative_rows': int((y < 0).sum()), 'zero_rows': int((y == 0).sum()),
            'zero_fraction': (y == 0).mean(), 'quantiles': y.quantile([.01, .05, .25, .5, .75, .9, .95, .99, .999]).to_dict(),
        }
    report['same_row_boardings_alightings_correlation'] = train.boardings.corr(train.alightings)
    stats = train.groupby('station_id').boardings.agg(['size', 'mean', 'median', 'std', 'min', 'max']).reset_index().merge(meta[['station_id','station_name','canonical_line']], on='station_id')
    stats['zero_fraction'] = train.groupby('station_id').boardings.apply(lambda y: (y == 0).mean()).reindex(stats.station_id).to_numpy()
    stats.to_csv(out / 'station_target_stats.csv', index=False)
    report['highest_mean_stations'] = stats.nlargest(10, 'mean').to_dict('records')
    report['lowest_mean_stations'] = stats.nsmallest(10, 'mean').to_dict('records')
    train['weekday'] = train.date.dt.dayofweek.astype('int8')
    train['month'] = train.date.dt.strftime('%Y-%m')
    weekday = train.groupby('weekday').boardings.agg(['mean','median','size']).reset_index()
    hourly = train.groupby('hour').boardings.mean()
    monthly = train.groupby('month').agg(rows=('boardings','size'), mean=('boardings','mean'), total=('boardings','sum')).reset_index()
    daily = train.groupby('date').boardings.sum()
    daily_mean = train.groupby('date').boardings.mean()
    report['weekday_profiles'] = weekday.to_dict('records')
    report['hour_mean_boardings'] = hourly.to_dict()
    report['monthly_profiles'] = monthly.to_dict('records')
    report['aggregate_daily_autocorrelation'] = {lag: daily.corr(daily.shift(lag)) for lag in [1, 7, 14, 28]}
    weekday.to_csv(out / 'weekday_profile.csv', index=False)
    monthly.to_csv(out / 'monthly_profile.csv', index=False)
    daily.rename('boardings_total').to_csv(out / 'daily_boardings.csv')
    maxima = train.nlargest(12,'boardings')[key+['boardings','alightings']].merge(meta[['station_id','station_name','canonical_line']],on='station_id')
    report['largest_single_slot_counts'] = maxima.to_dict('records')
    # Past-only weekly reference used strictly for an EDA spike flag, never as
    # a whole-data training feature. A 3x / 1000 threshold is heuristic.
    group = train.groupby(['station_id','hour','weekday'], sort=False).boardings
    past = group.transform(lambda y: y.shift(1).rolling(8, min_periods=4).median())
    ratios = train.boardings / past.replace(0,np.nan)
    spikes = train.loc[(ratios > 3) & (train.boardings >= 1000), key+['boardings']].copy()
    spikes['past_8_same_weekday_median'] = past.loc[spikes.index]
    spikes['ratio'] = ratios.loc[spikes.index]
    spikes = spikes.merge(meta[['station_id','station_name']],on='station_id').sort_values('ratio',ascending=False)
    spikes.to_csv(out / 'spike_candidates.csv',index=False)
    report['spike_heuristic'] = {'definition':'boardings >= 1000 and > 3x median of up to eight prior same-weekday observations at this station/hour; at least four prior observations', 'rows':len(spikes),'top_rows':spikes.head(15).to_dict('records')}
    # Synchrony is descriptive. A shared calendar/trend can explain correlations;
    # no coordinates or physical adjacency are present in the input.
    panel = train.pivot_table(index='date',columns='station_id',values='boardings',aggfunc='sum')
    corr = panel.corr(min_periods=100)
    residual = panel - panel.groupby(panel.index.dayofweek).transform('mean')
    adjusted = residual.corr(min_periods=100)
    pairs=[]
    names=meta.set_index('station_id').station_name.to_dict()
    lines=meta.set_index('station_id').canonical_line.to_dict()
    for a_i,a in enumerate(corr.columns):
        for b in corr.columns[a_i+1:]:
            pairs.append({'station_a':int(a),'station_b':int(b),'name_a':names[a],'name_b':names[b], 'same_canonical_line':bool(lines[a]==lines[b]), 'raw_daily_correlation':corr.loc[a,b], 'weekday_demeaned_correlation':adjusted.loc[a,b]})
    pair_df=pd.DataFrame(pairs).sort_values('raw_daily_correlation',ascending=False)
    pair_df.to_csv(out / 'station_correlations.csv',index=False)
    lag_rows=[]
    for row in pair_df.head(10).itertuples(index=False):
        a,b=row.station_a,row.station_b
        lag_corr={str(lag):residual[a].corr(residual[b].shift(lag)) for lag in range(-3,4)}
        lag_rows.append({'station_a':a,'station_b':b,'name_a':names[a],'name_b':names[b],'daily_lag_correlations':lag_corr, 'highest_correlation_lag_days':int(max(lag_corr,key=lambda k: lag_corr[k]))})
    report['spatial_and_synchrony']={
        'coordinates_available':False,'adjacency_available':False,'canonical_lines': sorted(meta.canonical_line.unique().tolist()),
        'pair_count':len(pair_df), 'raw_pair_correlation_quantiles':pair_df.raw_daily_correlation.quantile([0,.25,.5,.75,.9,1]).to_dict(),
        'weekday_demeaned_pair_correlation_quantiles':pair_df.weekday_demeaned_correlation.quantile([0,.25,.5,.75,.9,1]).to_dict(),
        'top_pairs':pair_df.head(10).to_dict('records'), 'lag_diagnostics_top_10_pairs':lag_rows,
        'interpretation':'Daily synchrony and coarse day-lag association only. Not a physical travel-time/propagation estimate. Station ID numeric proximity does not establish adjacency.'}
    report['feature_availability']={
        'future_known_from_files':['date','station_id','hour','canonical_line'],
        'train_only_observations':['boardings','alightings'],
        'absent_from_files':['weather','holiday indicator','discount/event indicator','station coordinates','physical adjacency','service disruption labels'],
        'calendar_derivable':['weekday','month','day of year','cyclic hour/weekday/day of year'],
        'leakage_notes':['No contemporaneous test alightings; use only values observed by forecast origin.', 'Metadata last_date is a dataset coverage boundary, not a verified station closing date.', 'No holdout target data may enter rolling means, scale normalization, correlations used as features or fitted calibration.']}
    report['official_task'] = {
        'evidence': 'User-supplied official Overview/Data/Rules; reports/sources/rules_user_provided.txt',
        'metric': 'RMSLE', 'horizon_days': 123, 'required_submission_rows': 157194,
        'training_loss': 'MSE on log1p(boardings); invert with expm1 then clip at zero',
        'external_data_cutoff': '2026-06-30',
        'holidays_allowed': True, 'external_sources_and_license_record_required': True,
    }
    (out/'eda.json').write_text(json.dumps(native(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    draw_chart(out/'daily_boardings.svg',daily,'Daily boardings across observed stations','Boardings / day (68 stations initially, 71 from 2025-06-28)')
    draw_chart(out/'daily_mean_boardings.svg',daily_mean,'Daily mean boardings per station-hour','Mean boardings / observed station-hour')
    draw_chart(out/'hourly_boardings.svg',hourly,'Average boardings by hour','Mean boardings / station-hour; all training dates')
    r=report
    introduction='''# SINCON 2026 데이터 EDA

이 보고서는 제공된 CSV의 실제 값에서 재현한 관찰 결과다. 사용자가 제공한 [공식 Overview·Data·Rules 사본](sources/rules_user_provided.txt)에 따라 과제는 **인천 지하철 71개 역의 시간대별 승차 인원(boardings)을 2026-06-30 기준으로 7~10월 123일간 한 번에 예측하는 패널 시계열 회귀**다. 공식 지표는 **RMSLE**, 제출 행 수는 **157,194행**이다. `alightings`는 학습 데이터에만 있는 하차 인원 보조 변수다.

## 1. 파일과 예측 범위

'''
    summary=pd.DataFrame([{'파일':f'{name}.csv','행':len(df),'열':df.shape[1],'셀 결측':int(df.isna().sum().sum())} for name,df in frames.items()])
    md=introduction+table(summary)+f'''

- 학습: **2025-01-01~2026-06-30, {train.date.nunique()}일**.
- 테스트: **2026-07-01~2026-10-31, {test.date.nunique()}일**, 71역 × 18시간 × 123일.
- 제출 예시: 테스트 앞 **92일(2026-07-01~09-30)**의 ID 117,576개와 순서까지 일치한다.
- 테스트에는 **10월 39,618행**이 추가로 있다. sample ID의 test 외 항목은 없고, 모든 ID와 (date, station_id, hour) 키는 고유하다.
- **공식 규정은 157,194행을 요구하므로 전체 test의 123일을 예측한다.** 로컬 sample의 117,576행은 구버전/불일치로 기록하고, 최종 제출의 행 수는 공식 규정과 test ID를 기준으로 검수한다. 원본 sample은 수정하지 않는다.
- `sample_submission.boardings`의 0은 양식 placeholder이며 실제 테스트 타겟이 아니다.

## 2. 규칙성·결측·역별 관측 기간

- 71개 역, 매일 **06~23시 18개 슬롯**이 제공된다. 각 역의 최초~최종 관측일 내부에는 누락 행이 **{int(station_coverage.missing_rows_within_observed_lifetime.sum())}개**, 중복 키도 0개다.
- timestamp 차이는 **1시간**(같은 날) 또는 **7시간**(23시→다음 날 06시)뿐이다. 00~05시가 파일에 없다는 뜻이며 실제 운휴를 확인한 것은 아니다. 그 시간에 0을 채워 24시간 자료로 바꾸지 않는다.
- 전체 71역 × 546일 × 18시간 격자에서 빈 {expected_grid-len(train):,}행은 3개 역의 첫 관측일 이전 구간이다. 이 값들은 결측 승객 수 또는 0명으로 취급하지 않는다.
- 68개 역은 2025-01-01, 3개 역(검단호수공원·신검단중앙·아라)은 **2025-06-28**부터 관측된다. 테스트에는 완전히 새로운 station_id가 없다.
- 메타데이터 `first_date`, `last_date`는 실제 관측 범위와 일치한다. 특히 `last_date=2026-06-30`은 데이터 종료일이지 폐역일이 아니다.
- 노선별 역 수: {json.dumps(native(r['coverage']['canonical_line_station_counts']),ensure_ascii=False)}. 같은 역명이라도 다른 station_id를 자동 합치지 않는다.

## 3. 타겟 분포와 급등

'''
    ys=r['boardings']
    md+=table(pd.DataFrame([{'통계':'평균','값':ys['mean']},{'통계':'중앙값','값':ys['median']},{'통계':'표준편차','값':ys['std']},{'통계':'최솟값','값':ys['min']},{'통계':'최댓값','값':ys['max']},{'통계':'왜도','값':ys['skewness']},{'통계':'0인 행 수','값':ys['zero_rows']},{'통계':'0 비율','값':ys['zero_fraction']}]))
    md+=f'''

우측 꼬리가 길다. 99% 분위수는 {ys['quantiles'][.99]:,.0f}, 99.9% 분위수는 {ys['quantiles'][.999]:,.2f}다. 음수는 없다. 공식 RMSLE에 맞춰 **log1p(boardings)의 MSE**를 기본 학습 손실로 사용한다. 예측은 expm1로 복원한 뒤 음수를 0으로 자른다. 원단위 MAE·RMSE·편향도 진단용으로 병행한다. 양수 카운트라는 사실만으로 Poisson 분포를 가정하지 않는다.

최대 관측값:

'''+table(maxima,12)+f'''

동일 역·시간·요일의 과거 4~8주 중앙값 대비 3배 초과이면서 승차 1,000명 이상인 휴리스틱 급등 후보는 **{len(spikes):,}행**이다. 이벤트·오류·수송 변화 여부는 원자료 설명 없이는 확정할 수 없다. 무조건 삭제하거나 상한을 자르지 않는다. 상세 목록은 `spike_candidates.csv`에 있다.

'''+table(spikes.head(10),10)+'''

## 4. 달력 패턴과 시점 변화

요일 0=월요일, 6=일요일이다. 다음 수치는 역·시간당 평균이며 총량과 구별한다.

'''+table(weekday)+'''

월별 표는 그 달에 관측된 행의 평균이다. 3개 역 추가에 따른 구성 변화와 달력 효과를 함께 고려해야 한다.

'''+table(monthly,30)+f'''

전체 일별 승차 총량의 자기상관: {json.dumps(native(r['aggregate_daily_autocorrelation']),ensure_ascii=False)}. 공통 요일과 추세가 포함된 EDA 상관이며 모델 잔차 ACF가 아니다. CV 후에는 역·시간별 잔차를 날짜순으로 다시 평가한다.

![일별 총승차](daily_boardings.svg)

![역·시간당 일별 평균](daily_mean_boardings.svg)

![시간대별 평균 승차](hourly_boardings.svg)

## 5. 역 규모와 다개체 연결성

평균 승차가 큰 역:

'''+table(stats.nlargest(10,'mean')[['station_id','station_name','canonical_line','mean','median','max']])+'''

제공 메타데이터에는 노선은 있으나 좌표·역 간 거리·연결 목록이 없다. ID의 숫자 차이로 거리나 인접 관계를 만들지 않는다. 같은 노선 구분과 역별 이력만 즉시 사용할 수 있다.

일별 총량의 역 간 상관 상위쌍(최소 동시 관측 100일):

'''+table(pair_df.head(10),10)+f'''

- 전체 {len(pair_df):,}개 역 쌍의 원자료 상관 중앙값: {pair_df.raw_daily_correlation.median():.4f}.
- 각 역의 요일 평균을 제거한 상관 중앙값: {pair_df.weekday_demeaned_correlation.median():.4f}.
- 상위 10쌍의 ±3일 지연 상관은 `eda.json`에 기록했다. 이는 날짜 단위의 동조 점검이다. 1시간보다 짧은 열차 이동/승객 전파를 검증하지 못하며 인과적 연결 또는 물리적 전파 지연의 근거가 아니다.
- 전체 학습 구간으로 만든 상관은 EDA 전용이다. 모델의 그룹/이웃 특징으로 쓰려면 각 CV origin 이전 이력에서만 다시 계산한다.

## 6. 입력 가용성과 누수 방지

| 구분 | 변수/처리 |
| --- | --- |
| 미래에도 주어짐 | date, station_id, hour, canonical_line |
| 파일에서 직접 파생 가능 | 요일·주말·월·연중일·시간 및 달력 주기 sin/cos |
| 학습에만 있음 | boardings, alightings |
| 없음 | 날씨, 공휴일 라벨, 할인·행사, 운행 장애, 좌표, 물리적 인접 정보 |
| 규정상 허용된 외부 입력 | 2026-06-30 이전 공개되고 비상업 이용이 허용된 공휴일·일정 등. 출처·조회일·라이선스 기록 필요 |

공휴일은 `src/holiday_calendar.py`의 검증된 정적 달력에서 생성한다. 공식 출처와 공개시점은 [holiday_sources.md](holiday_sources.md)에 기록했다. 원본 CSV에는 공휴일이 없으며 외부 특징으로 추가한 것이다. 임시공휴일과 신규 지정일은 각 CV 예측 기준일에 알려진 정보만 활성화한다. 2026-05-01 노동절과 2026-07-17 제헌절을 포함하고, 2026-09-28은 공휴일로 표시하지 않는다.

같은 행의 승차·하차 상관은 {r['same_row_boardings_alightings_correlation']:.4f}다. **미래 alightings는 test에 없으므로 동시점 하차 값을 입력하면 누수/추론 불일치다.** origin 이전의 요약·지연 하차만 쓸 수 있다. 123일 전체를 한 번에 예측할 때 day 8의 실제 lag7처럼 origin 이후 관측을 참조해서는 안 된다. 직접 다중 horizon 특징이나 origin에 고정한 계절/이동 통계가 필요하다.

## 7. 재현 및 보존

```bash
python src/eda.py --data-dir data/raw --out-dir reports
```

이 스크립트는 원본을 수정하지 않는다. `eda.json`에 입력 SHA-256, 자료형, 정확한 수치와 정의가 있고, 보조 CSV에 역별 범위·분포·상관·급등 후보가 있다. 그림은 집계값을 그대로 사용한다. 파일 크기는 약 24 MB이므로 이 단계의 원본만으로 29 GB RAM이 부족하지 않다. 다중 origin × horizon × 파생변수를 확장하면 메모리가 증가하므로 float32·정수 다운캐스팅, origin 배치 생성, 메모리 사용량 계측이 필요하다.
'''
    (out/'EDA.md').write_text(md,encoding='utf-8')
    print(json.dumps(native({'train_rows':len(train),'test_rows':len(test),'sample_rows':len(sample),'missing_rows':station_coverage.missing_rows_within_observed_lifetime.sum(),'spikes':len(spikes),'out_dir':str(out.resolve())}),ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1]/'data'/'raw')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1]/'reports')
    args=parser.parse_args()
    analyze(args.data_dir,args.out_dir)
