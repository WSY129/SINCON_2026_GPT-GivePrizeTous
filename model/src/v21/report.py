"""Writes outputs/ensemble_v21/RESULTS_V21.md, VALIDATION_V21.csv, CHANGED_CELLS_V21.csv, MANIFEST_V21.csv and v21_global_report.zip.
Local-validator numbers only; Public numbers appear only as user-provided facts and are never used.  .venv/bin/python -m src.v21.report"""
import hashlib, json, zipfile
import numpy as np, pandas as pd
from src.v8 import core as c8
from . import compare as C

R = c8.ROOT; V = R / 'outputs/ensemble_v21'; F = V / 'final'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    g = json.loads((F / 'gates_v21.json').read_text()); gb = json.loads((F / 'gates_busholiday.json').read_text()); ge = json.loads((F / 'gates_v21e.json').read_text())
    chain = g['chain']; gc = json.loads((R / chain / 'v19/gates_v19.json').read_text()); g15 = json.loads((R / chain / 'v15/gates_v15.json').read_text())
    v20e = g['v20e_aligned']; keys = ['all_oof', 'weighted_4fold', 'fold_3_4', 'fold_4', 'fold_1', 'fold_2', 'fold_3']
    rows = [dict(version='v20e (= v19 base)', **{k: v20e[k] for k in keys}),
            dict(version='v21 chain alone', **{k: round(gc['generations']['v19']['aligned'][k], 5) for k in keys}),
            dict(version='v21 = chain x w + v19 x (1-w)', **{k: round(g['aligned'][k], 5) for k in keys}),
            dict(version='v21 + bus holiday step (submission_ensemble_v21_bus)', **{k: round(gb['after'][k], 5) for k in keys})]
    val = pd.DataFrame(rows); val.to_csv(V / 'VALIDATION_V21.csv', index=False)
    singles = []
    for m, b in [('lgb_hk', 'lgb_h'), ('xgb_ik', 'xgb_i'), ('xgb_g11k', 'xgb_i'), ('lgb_h', 'lgb_h'), ('xgb_i', 'xgb_i'), ('xgb_g11', 'xgb_i'), ('xgb_h', 'xgb_h'), ('lgb_i', 'lgb_d'), ('lgb_x', 'lgb_h'), ('cb_g8', None)]:
        t = C.table(C.DIRS21, m)
        if 'partial' in t:
            print(f'WARNING: {m} skipped (stored predictions incomplete in this checkout: {t["partial"]})'); continue
        r = dict(model=m, seeds=t['f1']['seeds'], v21_weighted=t['weighted_strict'], **{f'v21_F{i}': t[f'f{i}']['strict'] for i in (1, 2, 3, 4)})
        if b:
            u = C.table(C.DIRS15, b); r.update(v20e_pair=b, v20e_weighted=u['weighted_strict'], **{f'v20e_F{i}': u[f'f{i}']['strict'] for i in (1, 2, 3, 4)})
        singles.append(r)
    S1 = pd.DataFrame(singles); S1.to_csv(V / 'SINGLE_MODELS_V21.csv', index=False)
    te = pd.read_csv(R / 'data/raw/test.csv'); names = pd.read_csv(R / 'data/raw/metaData.csv').set_index('station_id').station_name
    a = pd.read_csv(R / 'outputs/ensemble_v20/submission_ensemble_v20e.csv').boardings.to_numpy(); b = pd.read_csv(F / 'submission_ensemble_v21e.csv').boardings.to_numpy()
    c = pd.read_csv(F / 'submission_ensemble_v21_bus.csv').boardings.to_numpy()
    te = te.assign(v20e=a, v21=c, v21e=b, dlog=np.log1p(b) - np.log1p(a))
    ch = te.groupby(['date', 'station_id']).agg(v20e=('v20e', 'sum'), v21e=('v21e', 'sum'), mean_abs_dlog=('dlog', lambda s: s.abs().mean()), mean_dlog=('dlog', 'mean')).reset_index()
    ch['station'] = ch.station_id.map(names); ch = ch.sort_values('mean_abs_dlog', ascending=False); ch.to_csv(V / 'CHANGED_CELLS_V21.csv', index=False, encoding='utf-8-sig')
    files = [F / 'submission_ensemble_v21.csv', F / 'submission_ensemble_v21_bus.csv', F / 'submission_ensemble_v21e.csv', F / 'oof_v21.npz', F / 'oof_v21_bus.npz'] + sorted((V / 'panel').glob('*')) + sorted(V.glob('models_*/*.npz'))
    man = pd.DataFrame([dict(path=str(p.relative_to(R)), bytes=p.stat().st_size, sha256=sha(p)) for p in files if p.exists()]); man.to_csv(V / 'MANIFEST_V21.csv', index=False)
    L = []; w = L.append
    w('# ensemble_v21 결과 (RESULTS_V21)\n')
    w('> 점수는 모두 **로컬 검증(OOF)**(4-fold, aligned 채점)이다. 리더보드 점수는 어떤 계산·선택에도 쓰지 않았다. 평가 기간(2026-07~10) 지하철 승하차는 어디서도 쓰지 않았다(버스는 허용).\n')
    w('## 1. 최종 비교 (v20e 대비)\n'); w(val.to_markdown(index=False) if hasattr(val, 'to_markdown') and False else '\n'.join(['| ' + ' | '.join(val.columns) + ' |', '|' + '---|' * len(val.columns)] + ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in val.itertuples(index=False)]) + '\n')
    w(f"- 결합 가중(체인 쪽): 폴드별 {[g['weights']['nested'][k]['w_A'] for k in sorted(g['weights']['nested'])]} (각 폴드는 나머지 폴드에서 선택), 평가셋 {g['weights']['w_test']}.")
    w(f"- 명절 버스 단계: 계수 추석 {gb['beta_chuseok']:.3f} / 설 {gb['beta_seollal']:.3f} / 전체 {gb['beta_all']:.3f}, 버스 이상치와 지하철 잔차의 역 간 상관 추석 {gb['corr_xdev_rdev']['chuseok25']} / 설 {gb['corr_xdev_rdev']['seollal26']}. 날 전체 수준 상관은 {gb.get('day_level_corr_xmean_rmean')}로 명절마다 달라 쓰지 않았다.\n")
    w('## 2. 무엇이 바뀌었나\n')
    w('1. **주말 명절 라벨 수정** (`src/v21/kinds.py`): 추석·설에 붙은 토·일이 `lunar_adj`로 분류돼, 평일에서 배운 명절 감소가 주말 수준에 한 번 더 빠지던 오류. 2025-10-05(일)에서 트리가 −0.57 로그 과소였다. `long_sat`/`long_sun`으로 바꿔 패널을 다시 만들고 모든 트리를 재학습. 평가셋 해당일: 2026-09-26(토).')
    w('2. **드리프트 특징은 제외**: 직전 28일 잔차 특징은 기준일이 연말인 Fold 3에서 해가 돼(단일 모델 F3 +0.0013) 최종 모델에서 뺐다(제거 실험으로 확인).')
    w('3. **트리 재학습**: RunPod 2대(EPYC 7C13 31코어 + RTX 3090, EPYC 9555 13.6코어 + RTX PRO 6000 MIG) + Apple M4 Pro. LightGBM 6스레드·결정적 모드, XGBoost CPU/GPU, CatBoost GPU. 동결 패널 SHA-256은 MANIFEST_V21.csv.')
    tol = chain.split('_tol')[1].split('_')[0] if '_tol' in chain else '?'; adopted = [k for k, v in g15['steps'].items() if v.get('adopted')]
    w(f"4. **스택**: v15 스택 코드 그대로(멤버만 교체). 단계 채택 허용치 {tol}: 채택 단계 {adopted} (v20e의 v15와 같은 구성). 1단계 평가셋 가중: { {k: v for k, v in g15['retrained_models']['stage1_weights_test'].items() if v > 0} }")
    w('5. **2단 결합**: v21 체인과 v19(v20e의 기반)를 로그 공간에서 섞음. 폴드 k의 가중은 나머지 폴드에서, 폴드 k와 날짜가 겹치는 날을 빼고 고름(날짜 차단).')
    w(f"6. **명절 버스 단계**(검증 통과): 대표 정류장의 명절 버스 이상치(같은 창의 평소 요일 대비)를 역 간 편차로 바꿔 지하철 예측을 이동. 추석↔설 교차 검증. 평가셋 이동: {gb.get('test_shifts')}. 두 버스 파일의 겹친 54개 정류장-일은 중복 제거(첫 구현에서 2배 집계 버그를 교차 검증으로 발견해 수정). 30개 역은 검증 예측의 특수일 오차 순으로 골랐으므로 개선폭은 다소 낙관적일 수 있다.")
    w(f"7. **v21e**(검증 불가 항목): v20e의 여름 집단휴가·항목 14건(시나리오 13 + 마라톤 배율 1)을 v21 위에 다시 적용 + 송도 맥주축제 하향만(버스 증가분이 2025의 0.05~0.46, 2026 유료 입장 전환) + 부평삼거리 추석 당일 {ge.get('chuseok_3122')}. v21_bus 대비 변경 {ge.get('rows_changed_vs_base')}칸.\n")
    w('## 3. 단일 모델 (v21 대 v20e의 같은 모델, 가중 4-Fold, 시드 평균)\n'); w('\n'.join(['| ' + ' | '.join(S1.columns) + ' |', '|' + '---|' * len(S1.columns)] + ['| ' + ' | '.join('' if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r) + ' |' for r in S1.itertuples(index=False)]) + '\n')
    w('## 4. 검증하지 않은 것과 한계\n- 행사·휴가·맥주축제 항목은 평가 기간 값이라 로컬로 측정할 수 없다.\n- 명절 버스 단계는 학습 기간 명절 2회(추석 2025, 설 2026)로만 검증했다.\n- RunPod(x86)에서 학습한 LightGBM은 Apple M4에서 다시 돌리면 미세하게 다를 수 있다(스레드 수 6으로 고정, 결정적 모드).\n')
    (V / 'RESULTS_V21.md').write_text('\n'.join(L) + '\n')
    with zipfile.ZipFile(V / 'v21_global_report.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [V / 'RESULTS_V21.md', V / 'VALIDATION_V21.csv', V / 'SINGLE_MODELS_V21.csv', V / 'CHANGED_CELLS_V21.csv', V / 'MANIFEST_V21.csv', F / 'gates_v21.json', F / 'gates_busholiday.json', F / 'gates_v21e.json',
                  F / 'bus_holiday_test_shifts.csv', F / 'v21e_beer_festival.csv', V / 'panel/panel_manifest.json']:
            if p.exists(): z.write(p, p.relative_to(V))
    print(val.to_string(index=False)); print('written', V / 'v21_global_report.zip')


if __name__ == '__main__':
    main()
