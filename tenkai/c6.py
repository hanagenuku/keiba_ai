"""T'-C2c: 会場×距離の「クセ」は本当に持ち越すのか／物理量で説明できるのか。

c5.py の2つの不備を直す:
  🔴 分散ゼロの列で相関が nan になり、並べ替え検定が誤って P=0.000 を出していた
  🔴 窓A と 窓B の学習期が重なっていた（窓Bの学習 ⊃ 窓Aの学習）ので
     ρ=0.966 は「独立な再現」ではない。**期間を分離して**測り直す

使い方: python3 tenkai/c6.py <db_path>
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from tenkai.c1 import build
from src.features.race_shape import (VENUES, dist_zone, course_pace_features,
                                     COURSE_PACE_COLS, start_to_first_corner_m)

BASE4 = ['dist', 'surface_num', 'cls', 'n_horses']
SEEDS = (42, 7, 2026)
SPLIT = '2026-01-01'          # 前期: 2025年 / 後期: 2026年  ← **重ならない**
                              # ⚠ lap_times は2025年以降しか無い（2023-24は0%）ので
                              #    ここより前では分けられない


def spearman(a, b, n_perm=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 8:
        return None, None, n, '標本不足'
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return None, None, n, '分散ゼロ（全部同じ値）'
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    cnt = sum(abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) >= abs(r) for _ in range(n_perm))
    return r, (cnt + 1) / (n_perm + 1), n, ''


def offsets(R, period):
    """その期間だけで M0 を学習し、(会場,馬場,距離帯) ごとの残差平均を返す。"""
    d = R[period].copy()
    pred = np.mean([xgb.XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, random_state=s, verbosity=0
    ).fit(d[BASE4].astype(float), d.early).predict(d[BASE4].astype(float))
        for s in SEEDS], axis=0)
    d['r0'] = d.early.values - pred
    return d.groupby(['rc', 'sf', 'dz'])['r0'].agg(['mean', 'size'])


def run(db):
    H, R = build(db, '.')
    R['dz'] = R.dist.map(dist_zone)
    early_p = R.date < SPLIT
    late_p = R.date >= SPLIT
    print(f'前期 {early_p.sum():,}R（2025年） / 後期 {late_p.sum():,}R（2026年）  ※重複なし\n')

    A, B = offsets(R, early_p), offsets(R, late_p)
    common = [k for k in A.index if k in B.index
              and A.loc[k, 'size'] >= 40 and B.loc[k, 'size'] >= 40]
    ea = [A.loc[k, 'mean'] for k in common]
    la = [B.loc[k, 'mean'] for k in common]
    r, p, n, note = spearman(ea, la)
    print('■ ① 「会場×馬場×距離帯のクセ」は期間をまたいで持ち越すか')
    print(f'   前期(2025)のズレ vs 後期(2026)のズレ  ρ={r:+.3f}  P={p:.4f}  n={n}')
    print(f'   ズレの大きさ: 前期 std {np.std(ea):.3f}s / 後期 std {np.std(la):.3f}s')
    print('   （c5 の ρ=+0.966 は学習期が重なっていたので取り消し。これが正しい値）\n')

    rows = sorted(((k, A.loc[k, 'mean'], B.loc[k, 'mean'], int(B.loc[k, 'size']))
                   for k in common), key=lambda x: -abs(x[2]))
    print(f'{"会場 馬場 距離帯":26s} {"前期":>8s} {"後期":>8s} {"n":>6s}')
    for k, a, b, nn in rows[:14]:
        print(f'{" ".join(k):26s} {a:+8.3f} {b:+8.3f} {nn:6d}')

    print('\n■ ② そのズレは手持ちの物理量で説明できるか（後期のズレ vs 物理量）\n')
    feat = {}
    for k in common:
        rc, sf, dz = k
        sub = R[(R.rc == rc) & (R.sf == sf) & (R.dz == dz)]
        feat[k] = course_pace_features(rc, sf, int(sub.dist.median()), '.')
    print(f'{"物理量":22s} {"ρ":>8s} {"P":>8s} {"n":>5s}  備考')
    for c in COURSE_PACE_COLS:
        if c == 'venue_idx':
            continue
        vals = [np.nan if feat[k].get(c) is None else feat[k][c] for k in common]
        r2, p2, n2, note2 = spearman(vals, la)
        if r2 is None:
            print(f'{c:22s} {"—":>8s} {"—":>8s} {n2:5d}  {note2}')
        else:
            print(f'{c:22s} {r2:+8.3f} {p2:8.4f} {n2:5d}  {"← P<0.05" if p2 < 0.05 else ""}')
    print(f'\n   ⚠ {len(COURSE_PACE_COLS)-1} 個見ているので P<0.05 が1〜2個出るのは偶然でもありうる')

    print('\n■ ③ 物理データのカバー率（何が足りないか）\n')
    combos = R.groupby(['rc', 'sf', 'dist']).size()
    combos = combos[combos >= 20]
    have = sum(1 for (rc, sf, d) in combos.index
               if start_to_first_corner_m(rc, sf, d, '.') is not None)
    print(f'   実データに20R以上ある (会場,馬場,距離) の組み合わせ: {len(combos)}')
    print(f'   そのうち「スタート→1角」が出典つきで入っている: {have}'
          f'  （{100*have/len(combos):.1f}%）')
    print(f'   → ズレは (会場,馬場,距離) 単位で決まっているのに、'
          f'その単位の物理データが {100*have/len(combos):.0f}% しか無い')


if __name__ == '__main__':
    run(sys.argv[1])
