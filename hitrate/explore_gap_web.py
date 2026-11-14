#!/usr/bin/env python3
"""web 情報は「AI が持っていない情報」をどれだけ埋めるか（記述統計）。

`hitrate/EXPLORE_solo_market_gap.md` §3 の続き。市場なしモデルと市場のずれのうち、DB のどの変数でも
説明できない部分（約66%）を、いま過去分がある web 情報（kayochinkeiba の能力指数）がどれだけ説明するか。

    python3 hitrate/explore_gap_web.py     # 指数つきの data/horse_features.csv が要る（ランナーで実行）

- 2024-08-31 までで学習（検証 2024-09-01〜11-24）・2025-01-01 以降で測る。
- F0＝指数なしの市場なしモデル／F1＝指数ありの市場なしモデル。
- 🔴 記述統計。採否の判定ではない。モデル・表示は何も変えない。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TRAIN_END, VAL_START, VAL_END, OOS_START = '2024-08-31', '2024-09-01', '2024-11-24', '2025-01-01'


def main():
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from hitrate.gap_common import history_extras, load_frame, z_in_race
    from src.features.web_signals import fill_missing, is_web_col
    from src.tools.train_solo import PARAMS

    df = load_frame(ROOT)
    with open(os.path.join(ROOT, 'data', 'xgb_feature_cols.json'), encoding='utf-8') as f:
        cols = list(json.load(f)['feature_cols'])
    web = [c for c in cols if is_web_col(c)]
    base = [c for c in cols if not is_web_col(c)]
    if not web or any(c not in df.columns for c in web):
        raise SystemExit('指数の列が学習データに無い')
    hx, extra = history_extras(ROOT)
    df = df.merge(hx, on=['race_id', 'horse_num'], how='left')
    for c in list(extra):
        df['z' + c] = z_in_race(df, c)
    extra = extra + ['z' + c for c in extra]

    tr = df[df['date'] <= TRAIN_END]
    va = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)]
    oos = df[df['date'] >= OOS_START].copy()
    pos = tr['is_fukusho'].mean()
    P = dict(PARAMS, seed=42, scale_pos_weight=round((1 - pos) / pos, 2))

    def fit(c):
        dt = xgb.DMatrix(fill_missing(tr[c]), label=tr['is_fukusho'])
        dv = xgb.DMatrix(fill_missing(va[c]), label=va['is_fukusho'])
        b = xgb.train(P, dt, 500, evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
        return b.predict(xgb.DMatrix(fill_missing(oos[c])), output_margin=True,
                         iteration_range=(0, b.best_iteration + 1))

    oos['s0'], oos['s1'] = fit(base), fit(cols)
    out = {'coverage_all_oos': round(float(oos[web[0]].notna().mean()), 4), 'web_cols': web}
    # 指数が全馬に付いたレースだけで比べる
    full = oos.groupby('race_id')[web[0]].transform(lambda s: s.notna().all())
    o = oos[full].copy()
    o['z_m'] = z_in_race(o, 'bm'); o['z0'] = z_in_race(o, 's0'); o['z1'] = z_in_race(o, 's1')
    o = o.dropna(subset=['z_m', 'z0', 'z1']).reset_index(drop=True)
    o['gap0'] = o['z_m'] - o['z0']
    y = o['is_fukusho'].values
    out['n'] = dict(horses=int(len(o)), races=int(o['race_id'].nunique()), frm=o['date'].min(), to=o['date'].max())

    auc = lambda c: round(float(roc_auc_score(y, o[c])), 4)
    a0, a1, am = auc('z0'), auc('z1'), auc('z_m')
    out['auc'] = dict(AI指数なし=a0, AI指数あり=a1, 市場=am, 指数だけ=round(float(roc_auc_score(y, -o[web[0]])), 4),
                      市場との差のうち指数が埋めた割合=round((a1 - a0) / (am - a0), 4) if am > a0 else None)

    RP = dict(max_depth=6, eta=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
              objective='reg:squarederror', seed=42, nthread=8)

    def oof(c):
        X, t, pred = o[c], o['gap0'].values, np.zeros(len(o))
        for a, b in GroupKFold(5).split(X, t, o['race_id']):
            m = xgb.train(RP, xgb.DMatrix(X.iloc[a], label=t[a]), 300)
            pred[b] = m.predict(xgb.DMatrix(X.iloc[b]))
        return round(float(1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4), pred

    rA, _ = oof(base); rB, pB = oof(base + extra); rW, pW = oof(base + extra + web); rWonly, _ = oof(web)
    out['r2_gap'] = dict(既存特徴量=rA, 既存と追加変数=rB, さらに指数=rW, 指数の上乗せ=round(rW - rB, 4),
                         指数だけ=rWonly, 説明できない残り=round(1 - rW, 4))
    o['resid_B'] = o['gap0'] - pB
    v = o[[web[1], 'resid_B', 'gap0', 'z_m']].dropna()
    out['corr'] = dict(指数zと市場z=round(float(np.corrcoef(v[web[1]], v['z_m'])[0, 1]), 3),
                       指数zとずれ=round(float(np.corrcoef(v[web[1]], v['gap0'])[0, 1]), 3),
                       指数zと説明できないずれ=round(float(np.corrcoef(v[web[1]], v['resid_B'])[0, 1]), 3))
    # 指数を足したあとも説明できないずれは、まだ結果を当てるか（AI を固定）
    o['resid_W'] = o['gap0'] - pW
    lr = LogisticRegression(C=1e6).fit(o[['z1', 'resid_W']], y)
    out['残りのずれの係数（AI指数ありを固定）'] = round(float(lr.coef_[0][1]), 3)

    # 朝の web 見解の台帳（予想家の印・追い切り評価など）はまだ日が浅い。件数だけ数える
    try:
        with open(os.path.join(ROOT, 'data', 'web_findings_log.json'), encoding='utf-8') as f:
            ent = json.load(f)['entries']
        ent = ent if isinstance(ent, list) else list(ent.values())
        led = {}
        for e in ent:
            k = led.setdefault(e.get('kind'), dict(件数=0, 結果つき=0, 開催日=set()))
            k['件数'] += 1; k['結果つき'] += int(bool(e.get('result'))); k['開催日'].add(e.get('date'))
        out['ledger'] = {str(k): dict(件数=v['件数'], 結果つき=v['結果つき'], 開催日数=len(v['開催日'])) for k, v in led.items()}
    except Exception as e:
        out['ledger'] = f'読めない: {e}'

    with open(os.path.join(ROOT, 'hitrate', 'gap_web_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
