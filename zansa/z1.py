"""Z: 市場残差に (会場, 馬場, 実距離) の固定オフセットはあるか（基準は CRITERIA.md）。

使い方: python3 zansa/z1.py <history.db>
"""
import sys

import numpy as np
import pandas as pd

MIN_N = 300
SPLIT = '2025-01-01'


def field_band(n):
    return '~9' if n <= 9 else ('10-13' if n <= 13 else '14~')


def spearman(a, b, n_perm=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 8:
        return None, None, len(a)
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return None, None, len(a)
    r = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = sum(abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) >= abs(r) for _ in range(n_perm))
    return r, (cnt + 1) / (n_perm + 1), len(a)


def load(db):
    import sqlite3
    con = sqlite3.connect(db)
    H = pd.read_sql("""
        SELECT h.race_id, h.date, h.horse_num, h.place, h.popularity pop,
               r.racecourse rc, r.surface sf, r.distance dist
        FROM horse_history h JOIN race_history r USING(race_id)
        WHERE h.place > 0 AND h.place < 99 AND h.popularity BETWEEN 1 AND 18
    """, con)
    con.close()
    H['date'] = H.date.astype(str).str[:10]
    H['n_h'] = H.groupby('race_id')['horse_num'].transform('count')
    H = H[H.n_h.between(5, 18)].copy()
    H['fb'] = H.n_h.map(field_band)
    H['top3'] = (H.place <= 3).astype(float)
    return H


def run(db):
    H = load(db)
    early, late = H[H.date < SPLIT], H[H.date >= SPLIT]
    print(f'前期 {len(early):,}頭（〜2024） / 後期 {len(late):,}頭（2025〜）  ※重複なし')
    print(f'全体の3着内率  前期 {early.top3.mean()*100:.2f}%  後期 {late.top3.mean()*100:.2f}%\n')

    # 期待値テーブル（人気 × 頭数帯）は前期だけで作る
    tbl = early.groupby(['pop', 'fb'])['top3'].agg(['mean', 'size'])
    tbl = tbl[tbl['size'] >= 100]['mean']
    print(f'期待値テーブル: {len(tbl)} セル（人気×頭数帯・前期 n>=100）')

    def resid(df):
        e = pd.Series(list(zip(df['pop'], df.fb))).map(tbl).values
        d = df.copy()
        d['exp'] = e
        d = d[~pd.isna(d['exp'])]
        d['res'] = (d.top3 - d['exp']) * 100        # pt
        return d

    E, L = resid(early), resid(late)
    print(f'期待値を引けた頭数: 前期 {len(E):,} / 後期 {len(L):,}')
    print(f'残差の平均（0近辺なら計測器は正常）: 前期 {E.res.mean():+.3f}pt / 後期 {L.res.mean():+.3f}pt\n')

    key = ['rc', 'sf', 'dist']
    ge = E.groupby(key)['res'].agg(['mean', 'size'])
    gl = L.groupby(key)['res'].agg(['mean', 'size'])
    common = [k for k in ge.index if k in gl.index
              and ge.loc[k, 'size'] >= MIN_N and gl.loc[k, 'size'] >= MIN_N]
    ea = [ge.loc[k, 'mean'] for k in common]
    la = [gl.loc[k, 'mean'] for k in common]
    print(f'■ 基準5: 前期・後期とも n>={MIN_N}頭 の (会場,馬場,実距離) セル: {len(common)}\n')

    r, p, n = spearman(ea, la)
    print('■ 基準1: 持ち越すか（前期のオフセット vs 後期のオフセット）')
    print(f'   ρ={r:+.3f}  P={p:.4f}  n={n}      {"✅ P<0.01" if p and p < 0.01 else "❌ 基準未達"}')
    print('\n■ 基準2: 大きさ')
    print(f'   後期オフセットの std = {np.std(la):.3f}pt   '
          f'{"✅ >=1.0pt" if np.std(la) >= 1.0 else "❌ <1.0pt（あっても使えない）"}')
    print(f'   （参考）前期 std {np.std(ea):.3f}pt / 後期の範囲 '
          f'{min(la):+.2f} 〜 {max(la):+.2f}pt')

    print('\n■ 基準3: 前期だけで選んだ上位10セルは、後期でも同符号か')
    order = sorted(range(len(common)), key=lambda i: -abs(ea[i]))[:10]
    same = sum(1 for i in order if np.sign(ea[i]) == np.sign(la[i]))
    print(f'{"会場 馬場 距離":22s} {"前期":>8s} {"後期":>8s} {"後期n":>7s}')
    for i in order:
        k = common[i]
        print(f'{k[0]+" "+k[1]+" "+str(int(k[2])):22s} {ea[i]:+8.2f} {la[i]:+8.2f} '
              f'{int(gl.loc[k, "size"]):7d}')
    print(f'   同符号 {same}/10   {"✅ 7以上" if same >= 7 else "❌ 基準未達"}')

    print('\n■ 基準4: 対照群（人気帯の中でセルのラベルをシャッフル）')
    rng = np.random.default_rng(7)
    rs = []
    for _ in range(5):
        Ls = L.copy()
        Ls['fake'] = Ls.groupby('pop')['rc'].transform(lambda s: rng.permutation(s.values))
        gf = Ls.groupby(['fake', 'sf', 'dist'])['res'].agg(['mean', 'size'])
        cf = [k for k in ge.index if (k[0], k[1], k[2]) in gf.index
              and gf.loc[(k[0], k[1], k[2]), 'size'] >= MIN_N and ge.loc[k, 'size'] >= MIN_N]
        if len(cf) >= 8:
            rr, _, _ = spearman([ge.loc[k, 'mean'] for k in cf],
                                [gf.loc[(k[0], k[1], k[2]), 'mean'] for k in cf], n_perm=200)
            if rr is not None:
                rs.append(rr)
    print(f'   偽セルでの ρ: {[f"{x:+.3f}" for x in rs]}  平均 {np.mean(rs):+.3f}'
          f'   {"✅ ほぼ0" if abs(np.mean(rs)) < 0.2 else "❌ 計測器が壊れている"}')

    print('\n■ 参考: 会場だけ / 会場×馬場 の粒度ではどうか')
    for name, kk in [('会場', ['rc']), ('会場×馬場', ['rc', 'sf']),
                     ('会場×馬場×距離帯', ['rc', 'sf'])]:
        if name == '会場×馬場×距離帯':
            E2 = E.copy(); L2 = L.copy()
            for d in (E2, L2):
                d['dz'] = pd.cut(d.dist, [0, 1400, 1800, 2200, 9999],
                                 labels=['~1400', '1401-1800', '1801-2200', '2201~'])
            kk = ['rc', 'sf', 'dz']
            a2 = E2.groupby(kk, observed=True)['res'].agg(['mean', 'size'])
            b2 = L2.groupby(kk, observed=True)['res'].agg(['mean', 'size'])
        else:
            a2 = E.groupby(kk)['res'].agg(['mean', 'size'])
            b2 = L.groupby(kk)['res'].agg(['mean', 'size'])
        c2 = [k for k in a2.index if k in b2.index
              and a2.loc[k, 'size'] >= MIN_N and b2.loc[k, 'size'] >= MIN_N]
        if len(c2) < 8:
            print(f'   {name:16s} セル{len(c2):3d}  標本不足')
            continue
        v = [b2.loc[k, 'mean'] for k in c2]
        rr, pp, _ = spearman([a2.loc[k, 'mean'] for k in c2], v)
        print(f'   {name:16s} セル{len(c2):3d}  ρ={rr:+.3f} P={pp:.4f}  std={np.std(v):.3f}pt')


if __name__ == '__main__':
    run(sys.argv[1])
