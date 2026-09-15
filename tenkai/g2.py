"""T'-G2: 物理特徴に「未見コースへの一般化能力」はあるか（第2版DBで測り直し）。

leave-one-course-out: (会場,馬場,実距離) を丸ごと学習から外して、その組を予測する。
  A  base4 + 会場one-hot   ← 会場は知っているが、その距離の癖は知らない
  B  A + 物理特徴          ← 物理構造から推測できるか

🔑 §18（ユーザー指示・前回の測定失敗を受けて）:
   本データを測る前に **感度検査** を通す。
   「既知の人工的な差を入れたら、この計測器は本当に差を検出できるのか」。
   通らなければ本データの結果は読まない。

   前回（g1.py）は数値が 17/46 組しか無く、31列を足す操作そのものが
   RMSE を平均 -0.021s 動かしていた。本物の -0.045s はその分布に埋もれ P≈0.20。
   第2版DBで 79/85 組が埋まったので測り直す。

使い方: python3 tenkai/g2.py <db_path>
"""
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from tenkai.c1 import build
from src.features.race_shape import VENUES

MIN_R = 30
SEED = 42
N_SHUFFLE = 24
# 感度検査で注入する傾き（秒/m）。オフセットの実測 std 0.24s と同じ桁になるよう選ぶ
INJECT_BETA = 0.0015


def phys_cols(starts):
    """(rc,sf,dist) -> 物理特徴。数値は status=='ok' の行にしか入っていない。"""
    locs = sorted({v['start_location'] for v in starts.values()
                   if v.get('start_location')})
    out = {}
    for v in starts.values():
        key = (v['venue'], v['surface'], int(v['distance_m']))
        row = {
            'ph_s2c': (v['start_to_first_corner_m']
                       if v['start_to_first_corner_status'] == 'ok' else None),
            'ph_fc': {'1C': 1.0, '2C': 2.0, '3C': 3.0, '4C': 4.0,
                      'NONE': 0.0}.get(v.get('first_corner')),
            'ph_slope': {'上り': 1.0, '下り': -1.0, '平坦': 0.0}.get(v.get('slope_start')),
            'ph_variant': {'内回り': 1.0, '外回り': 2.0, '直線': 3.0}.get(
                v.get('course_variant')),
            'ph_spiral': 1.0 if v.get('turn_shape') == 'spiral' else 0.0,
            'ph_turfstart': 1.0 if v.get('turf_start') else 0.0,
            'ph_straight': v.get('straight_distance_m'),
            'ph_elev': v.get('course_elevation_m'),
            'ph_lap': v.get('one_lap_distance_m'),
            'ph_turn_dir': {'右': 1.0, '左': -1.0}.get(v.get('turn_direction')),
        }
        for loc in locs:
            row[f'ph_loc_{loc}'] = 1.0 if v.get('start_location') == loc else 0.0
        out[key] = row
    return out


def fit_rmse(tr, te, cols, target='early', seed=SEED):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0)
    m.fit(tr[cols].astype(float), tr[target])
    p = m.predict(te[cols].astype(float))
    return float(np.sqrt(((p - te[target].values) ** 2).mean()))


def loco(R, targets, A, extra, w, target='early'):
    r = [fit_rmse(R[R.key != k], R[R.key == k], A + extra, target) for k in targets]
    return float(np.sqrt((w * np.array(r) ** 2).sum()))


def shuffle_map(P, seed):
    rng = np.random.default_rng(seed)
    ks = list(P.keys()); perm = list(ks); rng.shuffle(perm)
    return {k: P[v] for k, v in zip(ks, perm)}


def run(db):
    _, R = build(db, '.')
    for v in VENUES:
        R[f'venue_{v}'] = (R.rc == v).astype(float)
    A = ['dist', 'surface_num', 'cls', 'n_horses'] + [f'venue_{v}' for v in VENUES]

    doc = json.load(open('data/course_starts.json'))['starts']
    P = phys_cols(doc)
    PH = sorted(next(iter(P.values())).keys())

    R['key'] = list(zip(R.rc, R.sf, R.dist.astype(int)))
    R = R[R.key.isin(P)].copy()
    for c in PH:
        R[c] = [P[k][c] if P[k][c] is not None else np.nan for k in R.key]

    sz = R.groupby('key').size()
    targets = [k for k in sz.index if sz[k] >= MIN_R]
    D0 = pd.DataFrame({'key': targets, 'n': [sz[k] for k in targets]})
    w = (D0.n / D0.n.sum()).values
    n_num = sum(1 for k in targets if P[k]['ph_s2c'] is not None)
    print(f'対象 {len(targets)} 組 / {sz[targets].sum():,}レース   物理特徴 {len(PH)}列')
    print(f'スタート→初角の数値が入っている組: {n_num}/{len(targets)}\n')

    def shuffled(seed):
        m = shuffle_map(P, seed)
        for c in PH:
            R['sh_' + c] = [m[k][c] if m[k][c] is not None else np.nan for k in R.key]
        return ['sh_' + c for c in PH]

    def pval_vs_shuffle(base, real_delta, target='early'):
        ds = []
        for i in range(N_SHUFFLE):
            cols = shuffled(200 + i)
            ds.append(loco(R, targets, A, cols, w, target) - base)
        ds = np.array(ds)
        better = int((ds <= real_delta).sum())
        return ds, (better + 1) / (N_SHUFFLE + 1)

    # ══════════════════════════════════════════════════════════════
    # §18 感度検査: 既知の人工的な差を入れて、検出できるかを先に確かめる
    # ══════════════════════════════════════════════════════════════
    print('=' * 70)
    print('§18 感度検査（本データを測る前に必ず通す）')
    print('=' * 70)
    s2c = np.array([P[k]['ph_s2c'] if P[k]['ph_s2c'] is not None else np.nan
                    for k in R.key], dtype=float)
    gm = R.groupby('key')['early'].transform('mean').values
    # 組ごとの差を「s2c の一次関数」だけに差し替える。組内のばらつきは実データのまま。
    R['synth'] = INJECT_BETA * np.nan_to_num(s2c, nan=float(np.nanmean(s2c))) \
        + (R.early.values - gm)
    lo, hi = np.nanmin(s2c), np.nanmax(s2c)
    print(f'注入: early := {INJECT_BETA} * s2c + 組内のばらつき')
    print(f'      s2c {lo:.0f}〜{hi:.0f}m → 組間の差は {INJECT_BETA*(hi-lo):.3f}s まで開く')
    a_s = loco(R, targets, A, [], w, 'synth')
    b_s = loco(R, targets, A, PH, w, 'synth')
    d_s = b_s - a_s
    print(f'\n  A  会場one-hotのみ      {a_s:.4f}s')
    print(f'  B  + 物理特徴           {b_s:.4f}s   A比 {d_s:+.4f}s')
    ds_s, p_s = pval_vs_shuffle(a_s, d_s, 'synth')
    print(f'  対照 シャッフル{N_SHUFFLE}回      平均 {ds_s.mean():+.4f}s (std {ds_s.std():.4f})')
    print(f'\n  → P={p_s:.3f}   ', end='')
    if p_s < 0.05:
        print('✅ 計測器は本物の差を検出できる。本データへ進む\n')
    else:
        print('❌ 検出できない。**この計測器では本データを読めない**。ここで中止\n')
        return

    # ══════════════════════════════════════════════════════════════
    # 本データ
    # ══════════════════════════════════════════════════════════════
    print('=' * 70)
    print('本データ（前半600m）')
    print('=' * 70)
    a = loco(R, targets, A, [], w)
    b = loco(R, targets, A, PH, w)
    R['oracle'] = R.groupby('key')['early'].transform('mean')
    o = loco(R, targets, A, ['oracle'], w)
    d = b - a
    print(f'\n  A  会場one-hotのみ      {a:.4f}s')
    print(f'  B  + 物理特徴           {b:.4f}s   A比 {d:+.4f}s')
    print(f'  注入 真のオフセット       {o:.4f}s   A比 {o-a:+.4f}s  （到達不能な上限）')
    ds, p = pval_vs_shuffle(a, d)
    print(f'  対照 シャッフル{N_SHUFFLE}回      平均 {ds.mean():+.4f}s (std {ds.std():.4f})')
    print(f'\n■ 基準1 B < A                  {"✅" if b < a else "❌"}')
    print(f'■ 基準2 改善が 0.01s 以上       {a-b:+.4f}s  {"✅" if a-b >= 0.01 else "❌"}')
    print(f'■ 基準3 シャッフルと区別できる    P={p:.3f}  '
          f'{"✅" if p < 0.05 else "❌ 列を足すこと自体で動く範囲"}')
    print(f'■ 基準4 感度検査を通っている      ✅（P={p_s:.3f}）')


if __name__ == '__main__':
    run(sys.argv[1])
