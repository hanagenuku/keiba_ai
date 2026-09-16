"""T'-G4: g3 の陽性が「どこから来ているか」を確かめる。

g3 は P 値と上位3組の占有率までしか出さない。採用の前に、
ユーザーの指摘した4点に答える。

  ① 改善は何組に分散しているか
  ② 上位1〜2組を除くと消えるのか
  ③ もっと強い外挿（会場ごと丸ごと未見）でも再現するのか
  ④ 観測値はシャッフル分布のどの位置か（z）

🔑 ③ が本命。g2/g3 の leave-one-course-out は「その距離は未見だが、
   同じ会場の別距離は学習に入っている」という弱い外挿。
   会場ごと落とす（leave-one-venue-out）と、物理特徴だけが頼りになる。
   ⚠ ただし A（会場one-hot）も同時に無力化されるので、A 自身が弱くなる。
      比較は同じ条件どうしで行う。

使い方: python3 tenkai/g4.py <db_path>
"""
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from tenkai.c1 import build
from tenkai.g2 import MIN_R, fit_rmse, phys_cols
from src.features.race_shape import VENUES

G3_LOG = ('/tmp/claude-0/-home-user-keiba-ai/'
          '7c917634-7760-54ae-973c-8978edd110a6/scratchpad/g3.log')


def prep(db):
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
    return R, A, PH


def weighted(D, col):
    w = D.n / D.n.sum()
    return float(np.sqrt((w * D[col] ** 2).sum()))


def run(db):
    R, A, PH = prep(db)
    sz = R.groupby('key').size()
    targets = [k for k in sz.index if sz[k] >= MIN_R]

    rows = [(k, int(sz[k]),
             fit_rmse(R[R.key != k], R[R.key == k], A),
             fit_rmse(R[R.key != k], R[R.key == k], A + PH)) for k in targets]
    D = pd.DataFrame(rows, columns=['key', 'n', 'A', 'B'])
    D['gain'] = D.A - D.B
    # 重み付き二乗和への寄与（全体の RMSE 差を分解したもの）
    D['contrib'] = (D.n / D.n.sum()) * (D.A ** 2 - D.B ** 2)

    a, b = weighted(D, 'A'), weighted(D, 'B')
    print(f'■ 全体   A {a:.4f}s   B {b:.4f}s   A比 {b - a:+.4f}s'
          f'   ({len(D)}組 / {int(D.n.sum()):,}レース)\n')

    # ── ① 分散しているか ───────────────────────────────
    win = int((D.gain > 0).sum())
    print(f'■ ① 改善の分散   良くなった {win}/{len(D)}組 ({win / len(D):.0%})')
    print(f'   組ごとの改善 中央値 {D.gain.median():+.4f}s  '
          f'四分位 {D.gain.quantile(.25):+.4f} 〜 {D.gain.quantile(.75):+.4f}')
    pos = D.contrib[D.contrib > 0].sum()
    print(f'   寄与が正の組の合計 {pos:+.4f} / 負の組 {D.contrib[D.contrib < 0].sum():+.4f}')

    # ── ② 上位を除くと消えるか ─────────────────────────
    print('\n■ ② 上位の組を除いて測り直す')
    print(f'   {"除いた組":26s} {"残り":>4s} {"A":>8s} {"B":>8s} {"A比":>9s}')
    order = D.sort_values('contrib', ascending=False)
    for k in range(4):
        drop = list(order.key[:k])
        E = D[~D.key.isin(drop)]
        ae, be = weighted(E, 'A'), weighted(E, 'B')
        label = 'なし（全69組）' if k == 0 else ' / '.join(
            ''.join(map(str, d)) for d in drop[-1:]) + (f' 他{k-1}組' if k > 1 else '')
        print(f'   {label:26s} {len(E):4d} {ae:8.4f} {be:8.4f} {be - ae:+9.4f}')

    # ── ③ もっと強い外挿（会場ごと落とす）────────────────
    print('\n■ ③ 会場ごと丸ごと未見にする（leave-one-venue-out）')
    vs = sorted(R.rc.unique())
    rows2 = []
    for v in vs:
        te, tr = R[R.rc == v], R[R.rc != v]
        if len(te) < MIN_R or len(tr) < 1000:
            continue
        rows2.append((v, len(te),
                      fit_rmse(tr, te, A), fit_rmse(tr, te, A + PH)))
    V = pd.DataFrame(rows2, columns=['venue', 'n', 'A', 'B'])
    V['gain'] = V.A - V.B
    av, bv = weighted(V, 'A'), weighted(V, 'B')
    print(f'   {"会場":8s} {"n":>6s} {"A":>8s} {"B":>8s} {"改善":>9s}')
    for r in V.sort_values('gain', ascending=False).itertuples():
        print(f'   {r.venue:8s} {r.n:6d} {r.A:8.4f} {r.B:8.4f} {r.gain:+9.4f}')
    print(f'   全体 A {av:.4f}s   B {bv:.4f}s   A比 {bv - av:+.4f}s   '
          f'良くなった {int((V.gain > 0).sum())}/{len(V)}会場')

    # ── ④ シャッフル分布のどこか ───────────────────────
    print('\n■ ④ シャッフル分布での位置（g3 のログから）')
    try:
        log = open(G3_LOG).read()
        m = re.search(r'平均 ([+-][\d.]+)s\s+std ([\d.]+)', log)
        p = re.search(r'P = ([\d.]+)', log)
        real = re.search(r'A比 ([+-][\d.]+)s', log)
        if m and real:
            mu, sd, rv = float(m.group(1)), float(m.group(2)), float(real.group(1))
            print(f'   本物 {rv:+.4f}s / シャッフル 平均 {mu:+.4f}s std {sd:.4f}')
            print(f'   z = {(rv - mu) / sd:+.2f}   '
                  f'P = {p.group(1) if p else "(未出力)"}')
        else:
            print('   g3 がまだシャッフルを出力していない')
    except OSError:
        print('   g3 のログが無い')


if __name__ == '__main__':
    run(sys.argv[1])
