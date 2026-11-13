"""Writes the v7 documents from the measured JSON files (every number is a local-validation number).

    python -m src.v7.report
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import core
from .train_eq import FOLDS, MAIN, TREE, NN, NNREG, LAMBDAS

OUT = core.OUT
EQ = OUT / 'eq'
FOLD_NAME = dict(f1='Fold 1', f2='Fold 2', f3='Fold 3', f4='Fold 4', hot26='Fold_Hot26', summer25='Fold_Summer25')
MODEL_NAME = dict(lgb='lgb_eq (LightGBM, CPU)', xgb='xgb_eq (XGBoost, CUDA)', cb='cb_eq (CatBoost, GPU)',
                  tabm='tabm_eq (TabM K=16, hidden 384)', mdn='mdn_eq (3성분 로그 MDN)', stg='st_graph_eq (역 그래프+시간 합성곱)',
                  rkhs='kridge_rkhs_eq (Nystrom 4096, Laplace+RBF)')


def f5(x):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.5f}'


def f4(x):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_label(name):
    kind, _, variant = name.partition('_')
    extra = {'': '', 's': ' + 절대 계절 특징', 'r': ' − 역 식별 특징', 'rs': ' + 계절 − 역 식별', 'noyoy': ' − 전년 특징'}[variant]
    return MODEL_NAME[kind] + extra


# ------------------------------------------------------------------------------ convergence report
def convergence():
    infos = {p.stem: json.loads(p.read_text()) for p in EQ.glob('*__*.json') if not p.stem.startswith('final')}
    finals = {p.stem: json.loads(p.read_text()) for p in EQ.glob('final__*.json')}
    first = {p.stem: json.loads(p.read_text()) for p in (OUT / 'eq_first_nn').glob('*.json')} if (OUT / 'eq_first_nn').exists() else {}
    summary = json.loads((EQ / 'cv_summary.json').read_text()) if (EQ / 'cv_summary.json').exists() else {}
    models = {}
    for tag, info in sorted(infos.items()):
        fold, name = tag.split('__')
        keep = {k: info[k] for k in ('seed', 'n_train', 'features', 'seconds', 'best', 'trained', 'valid_rmse_best', 'best_epoch', 'epochs',
                                     'valid_rmsle_best', 'valid_rmsle_swa', 'patience_stop_epoch', 'alarm_early_stop', 'train_curve',
                                     'valid_curve', 'train_loss', 'valid_rmsle', 'grad_norm', 'mdn', 'regularisation', 'landmarks',
                                     'gamma_laplace', 'gamma_rbf', 'swa_epochs', 'note') if k in info}
        keep['strict'] = summary.get(name, {}).get(fold)
        models.setdefault(name, {})[fold] = keep
    alarms = [dict(model=n, fold=f, best=v.get('best', v.get('best_epoch')), unit='rounds' if 'best' in v else 'epochs')
              for n, folds in models.items() for f, v in folds.items() if v.get('alarm_early_stop')]
    seconds = sum(i.get('seconds', 0) for i in infos.values()) + sum(i.get('seconds', 0) for i in finals.values()) \
        + sum(i.get('seconds', 0) for i in first.values())
    report = dict(
        information='All validation numbers are local (fold evaluation periods). RMSLE is computed in log1p space after adding the baseline back.',
        protocol=dict(trees=dict(learning_rate=TREE['lr'], max_rounds=TREE['rounds'], patience=TREE['patience'],
                                 alarm='best round < 200', strict_prediction='rounds = mean best round of the other main folds',
                                 final='1.1 x mean best round of Fold 3 and Fold 4, seeds 2026 and 2027'),
                      neural=dict(schedule={k: v for k, v in NN.items()}, warmup_epochs=3, cosine=True, swa_last_epochs=5,
                                  alarm='best epoch < 5', strict_prediction='fixed schedule, SWA weights (never selected on the fold)',
                                  regularisation_after_alarm=NNREG, final='full schedule with SWA, seeds 2026 and 2027'),
                      mdn=dict(components=3, sigma_floor=0.05, entropy_weight=0.01, point_forecast='E[max(0,Z)|X] by 15-point Gauss-Hermite quadrature'),
                      rkhs=dict(landmarks=4096, kernel='0.5*Laplace + 0.5*RBF, median-heuristic bandwidths', solver='Cholesky (float64)', lambdas=LAMBDAS),
                      gradient_norm='neural nets: mean L2 norm of the parameter gradient per epoch (before clipping at 5). trees: for an L2 objective the per-row gradient norm equals sqrt(train mse), which is what train_curve holds.'),
        folds={k: dict(origin=v[0], days=v[1]) for k, v in FOLDS.items()},
        gpu_seconds_logged=round(seconds, 1), gpu_minutes_logged=round(seconds / 60, 1),
        wall_clock=dict(start='2026-11-12 16:19:58 KST', end='2026-11-12 17:45:23 KST', minutes=85.4, budget_minutes=120,
                        hardware='RunPod RTX 4090 24GB (pod CPU limit 15.3 cores)', note='CPU and GPU streams ran in parallel, so the per-model seconds add up to more than the wall clock'),
        early_stop_alarms=alarms,
        alarm_diagnosis=dict(
            first_run='TabM/MDN/ST-Graph with station embedding, dropout 0.1, weight decay 1e-5, lr 2e-3: best epoch 1-5 on Fold 4, then validation RMSLE rose while the training loss kept falling.',
            cause='Not a baseline mismatch between train and evaluation rows (both come from the same panel function and the same mu_eq definition). The residual r_eq contains the seasonal drift between origin and target; a flexible model fits that drift for the training months per station, and the evaluation months of Folds 1-3 (and partly 4) were never seen one year earlier. On Fold_Hot26, whose season is present in training, the alarm disappears for ST-Graph (best epoch 13-14) and tree rounds are in the hundreds.',
            fix='Station embedding removed, dropout 0.3, weight decay 3e-2, peak lr 6e-4, Huber loss (delta 0.4); all three nets retrained on every fold. The first-run logs are kept under first_run_before_fix.',
            result='The alarm persists on the folds whose target season is unseen (see early_stop_alarms). The strict predictions therefore use the fixed schedule with SWA, and the blend weight of each model is decided by its strict out-of-fold error.'),
        first_run_before_fix={k: {a: v[a] for a in ('best_epoch', 'valid_rmsle_best', 'valid_rmsle_swa', 'train_loss', 'valid_rmsle', 'grad_norm') if a in v} for k, v in first.items()},
        models=models, final_models=finals, cv_summary=summary)
    (OUT / 'CONVERGENCE_AND_MDN_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    return report


# ------------------------------------------------------------------------------ markdown helpers
def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def metric_row(label, m):
    return [label, f5(m['all_oof']), f5(m['weighted_4fold']), f5(m['fold_3_4']), f5(m['fold_4']), f5(m['fold_1']), f5(m['fold_2']), f5(m['fold_3'])]


METRIC_HEAD = ['구분', '전체 OOF', '가중 4-Fold', 'Fold 3+4', 'Fold 4', 'Fold 1', 'Fold 2', 'Fold 3']
STATION_NAME = {3127: '문학경기장', 3226: '인천대공원', 3754: '부천종합운동장', 3107: '검단호수공원', 3136: '인천대입구',
                3139: '송도달빛축제공원', 3209: '아시아드경기장'}


def results_md(g, conv):
    b = g['baseline']; L = []
    L.append('# ensemble_v7 로컬 검증 결과 (RESULTS_V7)\n')
    L.append('> **점수 표기 규칙.** 이 문서의 모든 점수는 **로컬 검증(OOF) RMSLE**다. 리더보드 점수와 그로부터 계산한 값은 없다.\n'
             '> - **엄격(중첩 LOFO)**: 각 fold에 적용한 결합 가중치·모듈 계수를 나머지 3개 fold로만 구했고, 입력도 각 모델이 그 fold를 빼고 만든 예측이다.\n'
             '> - **v6 보고 방식**: v6 문서의 "엄격한 방식"과 같다(입력은 LOFO 예측, 결합 가중치는 4개 fold 전체로 적합).\n'
             '> - **기존 방식(plain OOF)**: v2·v4·v5가 저장해 둔 fold 내 앙상블 열을 입력으로 쓰고 가중치·계수를 4개 fold 전체로 적합.\n'
             '> - 로컬 검증기 4개 지표 = 전체 OOF, 가중 4-Fold(0.15·0.15·0.35·0.35), Fold 3+4, Fold 4. 역별 수치는 그 역 행만의 RMSLE다.\n')
    L.append('## 1. v6 대 v7 (로컬 검증기 4개 지표)\n')
    fin = g['final_strict']
    rows = [metric_row('v6 — 엄격(중첩 LOFO)', b['strict_nested']), metric_row('**v7 — 엄격(중첩 LOFO)**', fin),
            metric_row('v6 — v6 보고 방식', b['v6_convention_weights_on_all_folds']),
            metric_row('v6 — 기존 방식', b['plain_saved_columns']), metric_row('**v7 — 기존 방식**', g['final_plain'])]
    L.append(table(METRIC_HEAD, rows) + '\n')
    L.append(f"채택된 모듈: {', '.join(g['single'][m]['module'] for m in g['adopted']) or '없음'}\n")
    L.append('참고로 개별 버전의 같은 지표(엄격 = 각 버전의 LOFO 열, 기존 = 저장된 앙상블 열):\n')
    L.append(table(METRIC_HEAD, [metric_row(f'{v} — 엄격', r['strict']) for v, r in g['base_versions'].items()]
                   + [metric_row(f'{v} — 기존', r['saved']) for v, r in g['base_versions'].items()]) + '\n')
    L.append('## 2. 모듈별 단독 게이트 (v6 엄격 기준선 위에 그 모듈 하나만 추가)\n')
    L.append('채택 규칙: 엄격 전체 점수 개선 + 악화 fold 없음 + Fold 2·Fold 4 동시 개선. 그 모듈이 행을 바꾸지 않는 fold는 "미적용"으로 표기한다.\n')
    head = ['모듈', '변경 행', '전체', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '판정']
    base = b['strict_nested']
    rows = [['v6 기준선', 0] + [f5(base[k]) for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')] + ['—']]
    for key, r in g['single'].items():
        s = r['strict']; d = g['decisions'][key]
        rows.append([r['module'], r['rows_changed']] + [f"{f5(s[k])} ({s[k] - base[k]:+.5f})" for k in ('all_oof', 'fold_1', 'fold_2', 'fold_3', 'fold_4')]
                    + [('**채택**' if d['adopt'] else '기각')])
    L.append(table(head, rows) + '\n')
    L.append('판정 사유:\n')
    for key, d in g['decisions'].items():
        L.append(f"- **{g['single'][key]['module']}** — {d['reason']}")
    L.append('\n## 3. 누적 적용 (채택 모듈을 순서대로 더함)\n')
    rows = [metric_row('v6 기준선', base)] + [metric_row('+ ' + r['module'], r['strict']) for r in g['cumulative']]
    L.append(table(METRIC_HEAD, rows) + '\n')
    L.append('## 4. 역별 RMSLE (엄격, 4개 fold 합산)\n')
    rows = []
    for sid, name in STATION_NAME.items():
        k = f'st_{sid}'
        rows.append([sid, name, f4(base[k]), f4(fin[k]), f'{fin[k] - base[k]:+.4f}'] + [f4(r['strict'][k]) for r in g['single'].values()])
    L.append(table(['역', '이름', 'v6', 'v7', '차이'] + [r['module'].split(' ')[0] + ' ' + key for key, r in g['single'].items()], rows) + '\n')
    L.append('## 5. 보조 fold (Fold_Hot26, Fold_Summer25)\n')
    L.append('v2·v4·v5는 이 두 기준일의 예측이 없어서 최종 결합본을 이 fold에서 평가할 수 없다. 대신 평형 트리 모델들의 엄격 예측 평균을 기반으로 같은 전용 모델·성장 보정을 적용해 측정했다.\n')
    for name, r in g.get('aux_folds', {}).items():
        L.append(f"**{FOLD_NAME[name]}** (기준일 {r['origin']}, {r['days']}일, {r['rows']:,}행) — 평형 트리 평균 {f5(r['eq_tree_mean'])} → + v6 전용 모델 {f5(r['plus_v6_specialists'])} → + 성장 보정 {f5(r['plus_growth'])}. "
                 f"신규 역 {r['new_stations']['rows']:,}행: {f4(r['new_stations']['before'])} → {f4(r['new_stations']['after'])} (평균 편향 {r['new_stations']['mean_bias_before']:+.3f} → {r['new_stations']['mean_bias_after']:+.3f}).\n")
        L.append(table(['모델', 'RMSLE(엄격)'], [[model_label(m), f5(v)] for m, v in sorted(r['models'].items(), key=lambda x: x[1])]) + '\n')
    L.append('## 6. 평형 기준선 모델군 개별 성능 (Fold 1~4)\n')
    L.append('엄격 = 그 fold에서 고르지 않은 반복 수/고정 일정 SWA. "fold에서 조기 종료"는 그 fold 평가 구간에서 최적 반복을 고른 낙관적 수치다.\n')
    rows = []
    for m, r in sorted(g.get('eq_scores', {}).items(), key=lambda x: x[1]['strict']['all_oof']):
        s, e = r['strict'], r['early_stopped_on_fold']
        rows.append([model_label(m), f5(s['all_oof']), f5(s['fold_1']), f5(s['fold_2']), f5(s['fold_3']), f5(s['fold_4']), f5(e['all_oof'])])
    L.append(table(['모델', '엄격 전체', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'fold에서 조기 종료 전체'], rows) + '\n')
    if 'eq' in g['single']:
        w = g['single']['eq']['fitted'].get('blend', {})
        L.append('결합 가중치(중첩, fold별로 나머지 3개 fold에서 적합):\n')
        L.append('```json\n' + json.dumps(w, ensure_ascii=False, indent=1) + '\n```\n')
    q = json.loads((OUT / 'munhak_quota.json').read_text())
    L.append('## 7. 문학경기장(3127) 미발표일 쿼터 검증\n')
    L.append('관중일 = 3127역 시간당 최대 승차 1,200명 이상인 날. 미발표일 = 그 기준일에 KBO 일정이 아직 발표되지 않은 시즌 날짜(콘서트일 제외).\n')
    for key, title in (('validation_summer25', 'Fold_Summer25 (기준일 2025-06-30, 2025-09~10월이 미발표)'), ('validation_fold1', 'Fold 1 (기준일 2025-08-30, 2025-10월이 미발표)')):
        rows = [[n, v['unannounced_days'], v['expected_crowd_days'], v['actual_crowd_days'], f4(v['rmsle_unannounced_rows']),
                 ' / '.join(f"{k} {x['expected_crowd_days']}→실제 {x['actual_crowd_days']}" for k, x in v['by_weekday'].items())] for n, v in q[key].items()]
        L.append(f'**{title}**\n\n' + table(['방식', '미발표일', '기대 관중일', '실제 관중일', '미발표일 행 RMSLE', '요일별 기대→실제'], rows) + '\n')
    rows = [[n, v['unannounced_days'], v['expected_crowd_days'], ' / '.join(f"{k} {x['days']}일 p={x['mean_probability']}" for k, x in v['by_weekday'].items()),
             ' / '.join(f'{k}월 {x}' for k, x in v['by_month'].items())] for n, v in q['test_window'].items()]
    L.append('**테스트 구간(2026-07-01~10-31)**\n\n' + table(['방식', '미발표일', '기대 관중일(쿼터)', '요일별 평균 확률', '월별 기대 관중일'], rows) + '\n')
    sv = json.loads((OUT / 'submission_validation.json').read_text()); au = json.loads((OUT / 'MUNHAK_QUOTA_AND_DIFF_AUDIT.json').read_text())['diff_v7_minus_v6_log']
    L.append('## 8. 제출본\n')
    L.append(f"- 파일: `outputs/ensemble_v7/{sv['file']}` — {sv['rows']:,}행, 평균 승차 {sv['mean_boardings']:.2f}명 (v6 {au['mean_boardings_v6']:.2f}명)\n- SHA-256: `{sv['sha256']}`\n"
             f"- v6 대비 로그 RMSD {au['log_rmsd_all']:.5f}, 평균 로그 차이 {au['mean_log_diff']:+.5f}, 바뀐 행 {au['rows_changed']:,}개(그중 |차이|>0.05는 {au['rows_changed_over_0_05']:,}개), 바뀐 역 {au['stations_changed']}개\n"
             f"- 월별 평균 로그 차이: " + ', '.join(f"{k}월 {v['mean']:+.4f}" for k, v in au['by_month'].items()) + '\n')
    L.append(table(['역', '평균 로그 차이', '로그 RMSD', '최대 |차이|'], [[k, f"{v['mean']:+.4f}", f4(v['rmsd']), f4(v['max_abs'])] for k, v in au['by_station_top15'].items() if v['rmsd'] > 1e-6]) + '\n')
    # equilibrium finals against the v6 submission (diagnostic only; these forecasts are not in the submission)
    finals = sorted({p.stem.split('__')[1] for p in EQ.glob('final__*.npz')})
    if finals:
        test = pd.read_csv(core.ROOT / 'data/raw/test.csv', parse_dates=['date'])
        v6 = np.log1p(pd.read_csv(core.ROOT / 'outputs/ensemble_v6/submission_ensemble_v6.csv').boardings.to_numpy())
        stations = np.sort(test.station_id.unique())
        di = (test.date.to_numpy() - np.datetime64('2026-06-30')).astype('timedelta64[D]').astype(int) - 1
        si = np.searchsorted(stations, test.station_id.to_numpy()); hi = test.hour.to_numpy() - 6
        keep = ~test.station_id.isin([3127, 3139, 3209, 3210, 3754]).to_numpy(); month = test.date.dt.month.to_numpy()
        rows = []
        for m in finals:
            files = sorted(EQ.glob(f'final__{m}__*.npz'))
            z = np.mean([np.load(f)['pred'] for f in files], 0)[di, si, hi]
            d = (z - v6)
            rows.append([model_label(m), len(files), f4(float(np.sqrt(np.mean(d[keep] ** 2))))] + [f"{d[keep & (month == k)].mean():+.4f}" for k in (7, 8, 9, 10)])
        L.append('## 9. 평형 모델군 최종 예측과 v6 제출본의 차이 (참고용, 제출본에는 쓰지 않음)\n')
        L.append('최종 모델(기준일 2026-06-30, 시드 2026·2027 평균)의 테스트 예측에서 v6 제출본을 뺀 로그 차이. 행사 역 5개는 제외. 정답이 없으므로 점수가 아니라 두 예측의 거리다.\n')
        L.append(table(['모델', '시드 수', '로그 RMSD', '7월 평균차', '8월', '9월', '10월'], rows) + '\n')
    (OUT / 'RESULTS_V7.md').write_text('\n'.join(L))


def main():
    g = json.loads((OUT / 'gates.json').read_text())
    conv = convergence()
    results_md(g, conv)
    print('written: CONVERGENCE_AND_MDN_REPORT.json, RESULTS_V7.md; alarms', len(conv['early_stop_alarms']), 'gpu minutes logged', conv['gpu_minutes_logged'])


if __name__ == '__main__':
    main()
