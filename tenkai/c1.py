"""T'-C: コースの物理情報を入れてペースモデルを測り直す（基準は CRITERIA_COURSE.md）。

M0 いまの4列 → M1 +競馬場 → M2 +コースの静的特徴 → M3 +スタート→1角
→ M4 +馬特徴（前回悪化した群）

使い方: python3 tenkai/c1.py <db_path> <base_dir>
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from src.features.race_shape import (
    course_pace_features, race_shape_features, pace_model_inputs,
    COURSE_PACE_COLS, PACE_INPUT_COLS, RACE_FEATURE_COLS,
)
from src.tools.train_race_shape import _load, _horse_features, LEAD_COLS

WINDOWS = [('窓A', '2025-12-31', '2026-06-30'), ('窓B', '2026-06-30', '9999')]
COURSE_STATIC = ['straight_m', 'turn_num', 'loop_num', 'course_type_num',
                 'uphill', 'uphill_sev', 'hill_m', 'elev_range_m',
                 'corner_tight_num', 'dirt_turf_start']
HORSE_COLS = [c for c in RACE_FEATURE_COLS if c != 'n_horses']


def build(db, base_dir):
    H, R = _load(db)
    H = _horse_features(H)

    pin = pd.DataFrame([pace_model_inputs(d, s, c, n) for d, s, c, n
                        in zip(R.dist, R.sf, R.race_class, R.n_horses)])
    for c in PACE_INPUT_COLS:
        R[c] = pin[c].values

    # コース特徴は (会場, 馬場, 距離) の組み合わせ単位で1回だけ作る
    keys = R[['rc', 'sf', 'dist']].drop_duplicates()
    cache = {(r.rc, r.sf, int(r.dist)): course_pace_features(r.rc, r.sf, int(r.dist), base_dir)
             for r in keys.itertuples()}
    cf = pd.DataFrame([cache[(rc, sf, int(d))] for rc, sf, d in zip(R.rc, R.sf, R.dist)])
    for c in COURSE_PACE_COLS:
        R[c] = cf[c].values
    return H, R


def horse_aggregate(H, train_end):
    """P(先頭) を学習期だけで学習し、レース単位の隊列特徴量を作る。"""
    from sklearn.linear_model import LogisticRegression
    tr = H[H.date <= train_end]
    med = tr[LEAD_COLS].median()
    X = tr[LEAD_COLS].fillna(med)
    mu, sd = X.mean(), X.std().replace(0, 1)
    lm = LogisticRegression(max_iter=3000).fit((X - mu) / sd, tr.led)
    H = H.copy()
    H['P'] = lm.predict_proba((H[LEAD_COLS].fillna(med) - mu) / sd)[:, 1]
    rows = []
    for rid, s in H.groupby('race_id'):
        summ = [{'p_led': r.p_led, 'p_mean': r.p_mean, 'p_min': r.p_min,
                 'p_last': r.p_last, 'p_std': r.p_std, 'p_ag': r.p_ag, 'p_n': r.p_n}
                for r in s.itertuples()]
        f = race_shape_features(summ, list(s.P), list(s['pop']))
        f['race_id'] = rid
        rows.append(f)
    return pd.DataFrame(rows)


def rmse(m, d, cols, y):
    return float(np.sqrt(((m.predict(d[cols].astype(float)) - y) ** 2).mean()))


def run(db, base_dir, seed=42):
    H, R = build(db, base_dir)
    print(f'レース {len(R):,}  頭 {len(H):,}\n')
    arms = {
        'M0 いまの4列': PACE_INPUT_COLS,
        'M1 +競馬場': PACE_INPUT_COLS + ['venue_idx'],
        'M2 +コース静的': PACE_INPUT_COLS + ['venue_idx'] + COURSE_STATIC,
        'M3 +ｽﾀｰﾄ→1角': PACE_INPUT_COLS + ['venue_idx'] + COURSE_STATIC + ['start_to_corner_m'],
        'M4 +馬特徴': PACE_INPUT_COLS + ['venue_idx'] + COURSE_STATIC + ['start_to_corner_m'] + HORSE_COLS,
    }
    res = {}
    for wname, tend, vend in WINDOWS:
        F = horse_aggregate(H, tend)
        d = R.merge(F, on='race_id', how='inner', suffixes=('', '_h'))
        trd = d[d.date <= tend]
        ted = d[(d.date > tend) & (d.date <= vend)]
        base = float(np.sqrt(((trd.early.mean() - ted.early) ** 2).mean()))
        print(f'── {wname}  学習 {len(trd):,}R / 検定 {len(ted):,}R  '
              f'（学習平均を答える {base:.4f}s）')
        prev = None
        for tag, cols in arms.items():
            m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, verbosity=0)
            m.fit(trd[cols].astype(float), trd.early)
            r = rmse(m, ted, cols, ted.early.values)
            delta = f'{r - prev:+.4f}' if prev is not None else '  ----'
            print(f'   {tag:16s} {len(cols):3d}列  RMSE {r:.4f}s   前段比 {delta}')
            res.setdefault(tag, {})[wname] = r
            prev = r
        print()
    print('── まとめ（前段からの変化・秒。マイナスが改善）')
    tags = list(arms)
    for i, t in enumerate(tags):
        if i == 0:
            print(f'   {t:16s} 窓A {res[t]["窓A"]:.4f}   窓B {res[t]["窓B"]:.4f}')
        else:
            p = tags[i - 1]
            da = res[t]['窓A'] - res[p]['窓A']
            db_ = res[t]['窓B'] - res[p]['窓B']
            mark = '✅両窓で改善' if (da < 0 and db_ < 0) else ('❌両窓で悪化' if (da > 0 and db_ > 0) else '△不一致')
            print(f'   {t:16s} 窓A {res[t]["窓A"]:.4f} ({da:+.4f})  '
                  f'窓B {res[t]["窓B"]:.4f} ({db_:+.4f})   {mark}')
    return res


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '.')
