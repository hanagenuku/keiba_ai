#!/usr/bin/env python3
"""「AI単」列の単独モデルを作り、事前登録カード hitrate/CRITERIA_SOLO.md の測定を1回行う。

    python3 scripts/build_solo_model.py            # data/horse_features.csv が要る（本番と同じ列）

出力: data/xgb_solo_model.pkl・data/xgb_solo_cols.json・hitrate/solo_result.json
🔴 本番の予想モデル（xgb_fukusho_model*.pkl）・較正器には触らない。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TEST_START = '2026-09-01'


def main():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from hitrate.gap_common import history_extras, load_frame
    from src.features.web_signals import fill_missing
    from src.tools.train_solo import train_solo

    df = load_frame(ROOT)
    solo, meta = train_solo(ROOT, df=df)
    cols = meta['feature_cols']
    print('単独モデル:', {k: v for k, v in meta.items() if k != 'feature_cols'})

    t = df[df['date'] >= TEST_START].copy()
    prod = xgb.Booster()
    prod.load_model(os.path.join(ROOT, 'data', 'xgb_fukusho_model.pkl'))
    X = fill_missing(t[cols])
    d0 = xgb.DMatrix(X, feature_names=cols)
    d0.set_base_margin(t['bm'].values)
    t['old'] = prod.predict(d0, output_margin=True, iteration_range=(0, prod.num_boosted_rounds())) - t['bm'].values
    t['new'] = solo.predict(xgb.DMatrix(X, feature_names=cols), output_margin=True)
    hx, _ = history_extras(ROOT)
    t = t.merge(hx[['race_id', 'horse_num', 'x_jk_pop', 'x_last_win']], on=['race_id', 'horse_num'], how='left')
    t = t.sort_values(['race_id', 'horse_num']).reset_index(drop=True)
    y = t['is_fukusho'].values
    jk_hi = t['x_jk_pop'] <= t['x_jk_pop'].quantile(0.10)
    out = {'meta': {k: v for k, v in meta.items() if k != 'feature_cols'}, 'n_cols': len(cols),
           'window': [t['date'].min(), t['date'].max()], 'n_horses': int(len(t)), 'n_races': int(t['race_id'].nunique()),
           'auc_market': round(float(roc_auc_score(y, t['bm'])), 4)}
    for k in ('old', 'new'):
        t['rk_' + k] = t.groupby('race_id')[k].rank(ascending=False, method='first')
        rg = t['rk_' + k] - t['f_popularity']
        top = t[t['rk_' + k] == 1]
        out[k] = dict(
            auc=round(float(roc_auc_score(y, t[k])), 4),
            top1_fukusho=round(float(top['is_fukusho'].mean()), 4),
            top1_is_fav=round(float((top['f_popularity'] == 1).mean()), 4),
            big_gap_share=round(float((rg.abs() >= 4).mean()), 4),
            gap_popular_jockey=round(float(rg[jk_hi].mean()), 3),
            gap_last_win=round(float(rg[t['x_last_win'] == 1].mean()), 3),
            fav_with_solo_ge6=int(((t['f_popularity'] == 1) & (t['rk_' + k] >= 6)).sum()),
        )
    o, n = out['old'], out['new']
    out['criteria'] = {
        '1_auc_diff>=0.020': dict(value=round(n['auc'] - o['auc'], 4), ok=bool(n['auc'] - o['auc'] >= 0.020)),
        '2_top1_fukusho_new>=old': dict(value=round(n['top1_fukusho'] - o['top1_fukusho'], 4),
                                        ok=bool(n['top1_fukusho'] >= o['top1_fukusho'])),
        '3_gap_halved': dict(
            jockey=[o['gap_popular_jockey'], n['gap_popular_jockey']], last_win=[o['gap_last_win'], n['gap_last_win']],
            ok=bool(abs(n['gap_popular_jockey']) <= abs(o['gap_popular_jockey']) / 2
                    and abs(n['gap_last_win']) <= abs(o['gap_last_win']) / 2)),
    }
    out['verdict'] = 'PASS' if all(c['ok'] for c in out['criteria'].values()) else 'FAIL'
    with open(os.path.join(ROOT, 'hitrate', 'solo_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
