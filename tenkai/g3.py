"""T'-G3: g2 の陽性を確認する。

g2 で事前登録4基準が全部通った（本プロジェクトで久しぶりの陽性）。
過去、陽性はことごとく消えている（2026-07-30 回収率261% / 2026-08-08 AUC+0.037 /
2026-08-27 SHAP+0.1783）。採用の前に3つ確かめる。

  ① シャッフルを24→100回に増やして P を細かく出す
     （24回だと P の下限が 1/25=0.040 で、それ以上分解できない）
  ② どの列が効いているのか。今回埋めた「スタート→初角」が本体か、
     それとも他の列（直線長・高低差・1周＝会場×馬場の定数）か
     🔑 直線長などは venue one-hot が既に持っている情報なので、
        そこが効いているなら「物理量が効いた」とは言えない
  ③ 改善が一部の組に集中していないか

使い方: python3 tenkai/g3.py <db_path>
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from tenkai.c1 import build
from tenkai.g2 import MIN_R, fit_rmse, loco, phys_cols, shuffle_map
from src.features.race_shape import VENUES

N_SHUFFLE = 100

# 会場×馬場で一定＝venue one-hot が既に持っている列
VENUE_CONST = {'ph_straight', 'ph_elev', 'ph_lap', 'ph_turn_dir'}


def run(db):
    _, R = build(db, '.')
    for v in VENUES:
        R[f'venue_{v}'] = (R.rc == v).astype(float)
    A = ['dist', 'surface_num', 'cls', 'n_horses'] + [f'venue_{v}' for v in VENUES]

    P = phys_cols(json.load(open('data/course_starts.json'))['starts'])
    PH = sorted(next(iter(P.values())).keys())
    R['key'] = list(zip(R.rc, R.sf, R.dist.astype(int)))
    R = R[R.key.isin(P)].copy()
    for c in PH:
        R[c] = [P[k][c] if P[k][c] is not None else np.nan for k in R.key]
    sz = R.groupby('key').size()
    targets = [k for k in sz.index if sz[k] >= MIN_R]
    n = np.array([sz[k] for k in targets], dtype=float)
    w = n / n.sum()

    a = loco(R, targets, A, [], w)
    b = loco(R, targets, A, PH, w)
    real = b - a
    print(f'対象 {len(targets)}組 / {int(n.sum()):,}レース')
    print(f'A {a:.4f}s   B {b:.4f}s   A比 {real:+.4f}s\n')

    # ── ② どの列が効いているのか ────────────────────────────
    print('■ 列の切り分け（A比）')
    groups = {
        'スタート→初角だけ': ['ph_s2c'],
        '会場×馬場の定数だけ（直線長・高低差・1周・回り）': sorted(VENUE_CONST),
        '初角を抜いた全部': [c for c in PH if c != 'ph_s2c'],
        '会場定数を抜いた全部': [c for c in PH if c not in VENUE_CONST],
        '全部（=B）': PH,
    }
    for label, cols in groups.items():
        d = loco(R, targets, A, cols, w) - a
        print(f'   {label:44s} {d:+.4f}s  ({len(cols)}列)')

    # ── ① シャッフル100回 ──────────────────────────────────
    print(f'\n■ 対照群（物理特徴を組み合わせ間でシャッフル）×{N_SHUFFLE}')
    ds = []
    for i in range(N_SHUFFLE):
        m = shuffle_map(P, 1000 + i)
        for c in PH:
            R['sh_' + c] = [m[k][c] if m[k][c] is not None else np.nan for k in R.key]
        ds.append(loco(R, targets, A, ['sh_' + c for c in PH], w) - a)
    ds = np.array(ds)
    better = int((ds <= real).sum())
    p = (better + 1) / (N_SHUFFLE + 1)
    print(f'   平均 {ds.mean():+.4f}s  std {ds.std():.4f}  '
          f'範囲 {ds.min():+.4f}〜{ds.max():+.4f}')
    print(f'   本物 {real:+.4f}s 以上に良かったシャッフル {better}/{N_SHUFFLE}')
    print(f'   → P = {p:.3f}   {"✅" if p < 0.05 else "❌"}')

    # ── ③ 改善の集中度 ────────────────────────────────────
    rows = [(k, int(sz[k]), fit_rmse(R[R.key != k], R[R.key == k], A),
             fit_rmse(R[R.key != k], R[R.key == k], A + PH)) for k in targets]
    D = pd.DataFrame(rows, columns=['key', 'n', 'A', 'B'])
    D['gain'] = D.A - D.B
    win = int((D.gain > 0).sum())
    print(f'\n■ 組ごとの改善   良くなった {win}/{len(D)} 組')
    contrib = (D.n / D.n.sum()) * (D.A ** 2 - D.B ** 2)
    top3 = contrib.nlargest(3).sum() / contrib.sum()
    print(f'   上位3組が改善の {top3:.0%} を占める  '
          f'{"⚠ 一部に集中" if top3 > 0.6 else "✅ 分散している"}')
    D = D.sort_values('gain', ascending=False)
    print(f'\n   {"会場 馬場 距離":22s} {"n":>5s} {"A":>8s} {"B":>8s} {"改善":>8s}')
    for r in list(D.head(5).itertuples()) + list(D.tail(5).itertuples()):
        print(f'   {" ".join(map(str, r.key)):22s} {r.n:5d} {r.A:8.4f} {r.B:8.4f} {r.gain:+8.4f}')


if __name__ == '__main__':
    run(sys.argv[1])
