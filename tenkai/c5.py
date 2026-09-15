"""⚠ 不備が2つあり、結果は tenkai/c6.py で測り直した（そちらを見ること）。

   ① 分散ゼロの列で相関が nan になり、並べ替え検定が誤って P=0.000 を出していた
   ② 窓A と 窓B の学習期が重なっていたので ρ=+0.966 は独立な再現ではなかった

T'-C2b: 会場の効果の中身を、手持ちの物理量で説明できるかまで見る（診断）。

c4.py で分かったこと:
  会場ごとの単純な定数ずらしでは改善の 7〜18% しか回収できず、
  会場×馬場×距離帯まで細かくして 54〜66%。
  ＝ 会場の「クセ」は1会場1つの数字ではなく、距離構成ごとに違う。

ここでは (会場, 馬場, 距離帯) ごとのズレが、手持ちの物理量
（直線長・坂・コーナー・スタート→1角・芝スタート・内外回り）で
説明できるかを見る。説明できるなら物理特徴で置き換えて一般化できる。

使い方: python3 tenkai/c5.py <db_path>
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from tenkai.c1 import build
from src.features.race_shape import (PACE_INPUT_COLS, VENUES, dist_zone,
                                     course_pace_features, COURSE_PACE_COLS)

SEEDS = (42, 7, 2026)
BASE4 = ['dist', 'surface_num', 'cls', 'n_horses']
WIN = [('窓A', '2025-12-31', '2026-06-30'), ('窓B', '2026-06-30', '9999')]


def _fit(trd, ted, cols, seed):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0)
    m.fit(trd[cols].astype(float), trd.early)
    return m.predict(trd[cols].astype(float)), m.predict(ted[cols].astype(float))


def _rmse(p, y):
    return float(np.sqrt(((np.asarray(p) - np.asarray(y)) ** 2).mean()))


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 8:
        return None, None, int(ok.sum())
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    r = float(np.corrcoef(ra, rb)[0, 1])
    n = int(ok.sum())
    # 並べ替え検定（2,000回）。n が小さいので正規近似は使わない
    rng = np.random.default_rng(0)
    cnt = sum(abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) >= abs(r) for _ in range(2000))
    return r, (cnt + 1) / 2001, n


def run(db):
    H, R = build(db, '.')
    for v in VENUES:
        R[f'venue_{v}'] = (R.rc == v).astype(float)
    R['dz'] = R.dist.map(dist_zone)
    OH = [f'venue_{v}' for v in VENUES]

    print('■ ① 会場ごとの改善（窓ごと・全会場。開催の無い窓は "-"）\n')
    per = defaultdict(dict)
    offs = {}
    for wname, tend, vend in WIN:
        trd = R[R.date <= tend].copy()
        ted = R[(R.date > tend) & (R.date <= vend)].copy()
        p0tr = np.mean([_fit(trd, ted, BASE4, s)[0] for s in SEEDS], axis=0)
        p0 = np.mean([_fit(trd, ted, BASE4, s)[1] for s in SEEDS], axis=0)
        p1 = np.mean([_fit(trd, ted, BASE4 + OH, s)[1] for s in SEEDS], axis=0)
        ted['p0'], ted['p1'] = p0, p1
        trd['r0'] = trd.early.values - p0tr
        for v, g in ted.groupby('rc'):
            per[v][wname] = (len(g), _rmse(g.p1, g.early) - _rmse(g.p0, g.early))
        offs[wname] = trd.groupby(['rc', 'sf', 'dz'])['r0'].agg(['mean', 'size'])

    print(f'{"会場":6s} {"窓A n":>6s} {"窓A 差":>9s}  {"窓B n":>6s} {"窓B 差":>9s}')
    for v in VENUES:
        a = per[v].get('窓A'); b = per[v].get('窓B')
        fa = f'{a[0]:6d} {a[1]:+9.4f}' if a else f'{"-":>6s} {"-":>9s}'
        fb = f'{b[0]:6d} {b[1]:+9.4f}' if b else f'{"-":>6s} {"-":>9s}'
        print(f'{v:6s} {fa}  {fb}')

    print('\n■ ② (会場,馬場,距離帯) ごとの水準ずれ（学習期の残差平均・秒）')
    print('   + は M0 の予測より実際は遅い。両窓で安定しているか（n>=40）\n')
    A, B = offs['窓A'], offs['窓B']
    common = [k for k in A.index if k in B.index
              and A.loc[k, 'size'] >= 40 and B.loc[k, 'size'] >= 40]
    rows = [(k, A.loc[k, 'mean'], B.loc[k, 'mean'], int(B.loc[k, 'size'])) for k in common]
    rows.sort(key=lambda x: -abs(x[2]))
    print(f'{"会場 馬場 距離帯":26s} {"窓A":>8s} {"窓B":>8s} {"n":>6s}')
    for k, a, b, n in rows[:18]:
        print(f'{" ".join(k):26s} {a:+8.3f} {b:+8.3f} {n:6d}')
    ra = [r[1] for r in rows]; rb = [r[2] for r in rows]
    r, p, n = _spearman(ra, rb)
    print(f'\n   窓Aのズレ と 窓Bのズレ の順位相関: ρ={r:+.3f} (P={p:.3f}, n={n})')
    print('   → 高ければ「会場×距離のクセ」は年をまたいで持ち越す実体がある')

    print('\n■ ③ そのズレは手持ちの物理量で説明できるか（窓Bのズレ vs 物理量）\n')
    feat = {}
    for k in common:
        rc, sf, dz = k
        sub = R[(R.rc == rc) & (R.sf == sf) & (R.dz == dz)]
        dist = int(sub.dist.median()) if len(sub) else 1600
        feat[k] = course_pace_features(rc, sf, dist, '.')
    print(f'{"物理量":22s} {"ρ":>8s} {"P":>7s} {"n":>5s}')
    for c in COURSE_PACE_COLS:
        if c == 'venue_idx':
            continue
        vals = [feat[k].get(c) for k in common]
        vals = [np.nan if v is None else v for v in vals]
        r, p, n = _spearman(vals, rb)
        if r is None:
            print(f'{c:22s} {"—":>8s} {"—":>7s} {n:5d}   （データ不足）')
        else:
            mark = ' ←' if p < 0.05 else ''
            print(f'{c:22s} {r:+8.3f} {p:7.3f} {n:5d}{mark}')
    print(f'\n   ⚠ 物理量を {len(COURSE_PACE_COLS)-1} 個見ているので、'
          f'P<0.05 が1〜2個出るのは偶然でもありうる（多重比較）')


if __name__ == '__main__':
    run(sys.argv[1])
