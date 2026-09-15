"""Z2: 発走地点ごとに、市場の「配り方」が先行馬に対して系統的にずれるか。

z1.py の設計は構造的にゼロしか返せなかった（レース内の残差の和が固定）。
レース内で変わる属性＝**先行度**に対する傾きで測り直す。基準は CRITERIA.md。

使い方: python3 zansa/z2.py <history.db>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.tools.train_race_shape import _load, _horse_features

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


def slope(g, xcol='lead', ycol='res'):
    """セル内で res を lead に回帰した傾き（pt / 先行度1.0）。"""
    x, y = g[xcol].values, g[ycol].values
    if len(x) < 30 or np.std(x) < 1e-6:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def prepare(db):
    import sqlite3
    H, _ = _load(db)
    H = _horse_features(H)                     # p_mean = それ以前の1角通過位置の平均
    # ⚠ _load の H には会場・馬場が無い（R 側にある）。R は lap_times で絞られて
    #    いるので、ここは race_history から直接引く（2023年以降すべて使う）
    con = sqlite3.connect(db)
    meta = pd.read_sql('SELECT race_id, racecourse rc, surface sf FROM race_history', con)
    con.close()
    H = H.merge(meta, on='race_id', how='inner')
    H = H[H.rc.notna() & H.sf.isin(['芝', 'ダート'])].copy()
    H = H[H.p_mean.notna()].copy()
    H['date'] = H.date.astype(str).str[:10]
    H['lead'] = 1.0 - H.p_mean                 # 大きいほど前に行く馬（0=最後方寄り）
    H['n_h'] = H.groupby('race_id')['horse_num'].transform('count')
    H = H[H.n_h.between(5, 18)].copy()
    H['fb'] = H.n_h.map(field_band)
    H['top3'] = (H.place <= 3).astype(float)
    H = H[H['pop'].between(1, 18)].copy()
    return H


def run(db):
    H = prepare(db)
    E0, L0 = H[H.date < SPLIT], H[H.date >= SPLIT]
    print(f'前期 {len(E0):,}頭（〜2024） / 後期 {len(L0):,}頭（2025〜）')
    tbl = E0.groupby(['pop', 'fb'])['top3'].agg(['mean', 'size'])
    tbl = tbl[tbl['size'] >= 100]['mean']

    def resid(df):
        d = df.copy()
        d['exp'] = pd.Series(list(zip(d['pop'], d.fb))).map(tbl).values
        d = d[~pd.isna(d['exp'])].copy()
        d['res'] = (d.top3 - d['exp']) * 100
        return d

    E, L = resid(E0), resid(L0)
    print(f'期待値を引けた頭数: 前期 {len(E):,} / 後期 {len(L):,}')
    print(f'先行度の分布: 5%点 {L.lead.quantile(.05):.2f} / 中央 {L.lead.median():.2f}'
          f' / 95%点 {L.lead.quantile(.95):.2f}\n')

    # 全体の傾き（セルに分ける前に、そもそも先行度が市場残差と関係あるか）
    print(f'■ 全体の傾き（セル分割前）: 前期 {slope(E):+.2f}pt / 後期 {slope(L):+.2f}pt')
    print('   ＝「前に行く馬ほど市場評価より走る」が全体としてあるか\n')

    key = ['rc', 'sf', 'dist']

    def cell_slopes(df):
        sz = df.groupby(key).size()
        keep = set(sz[sz >= MIN_N].index)
        out = {}
        for k, g in df.groupby(key):
            if k in keep:
                v = slope(g)
                if not np.isnan(v):
                    out[k] = v
        return out

    ge, gl = cell_slopes(E), cell_slopes(L)
    sizes = L.groupby(key).size()
    common = sorted(set(ge) & set(gl))
    ea, la = [ge[k] for k in common], [gl[k] for k in common]
    print(f'■ 基準6: 前期・後期とも n>={MIN_N}頭 のセル: {len(common)}\n')

    r, p, n = spearman(ea, la)
    print('■ 基準1: 傾きは持ち越すか')
    print(f'   ρ={r:+.3f}  P={p:.4f}  n={n}      '
          f'{"✅ P<0.01" if p is not None and p < 0.01 else "❌ 基準未達"}')
    print('\n■ 基準2: 大きさ')
    print(f'   後期の傾きの std = {np.std(la):.2f}pt   '
          f'{"✅ >=3.0pt" if np.std(la) >= 3.0 else "❌ <3.0pt（あっても使えない）"}')
    print(f'   （参考）前期 std {np.std(ea):.2f}pt / 後期の範囲 {min(la):+.1f} 〜 {max(la):+.1f}pt')

    print('\n■ 基準3: 前期だけで選んだ上位10セルは後期でも同符号か')
    order = sorted(range(len(common)), key=lambda i: -abs(ea[i]))[:10]
    same = sum(1 for i in order if np.sign(ea[i]) == np.sign(la[i]))
    print(f'{"会場 馬場 距離":22s} {"前期":>8s} {"後期":>8s} {"後期n":>7s}')
    for i in order:
        k = common[i]
        print(f'{k[0]+" "+k[1]+" "+str(int(k[2])):22s} {ea[i]:+8.2f} {la[i]:+8.2f} {sizes[k]:7d}')
    print(f'   同符号 {same}/10   {"✅ 7以上" if same >= 7 else "❌ 基準未達"}')

    print('\n■ 基準4 対照群A: 先行度をレース内でシャッフル')
    rng = np.random.default_rng(7)
    sh = []
    for _ in range(5):
        Ls = L.copy()
        Ls['lead'] = Ls.groupby('race_id')['lead'].transform(lambda s: rng.permutation(s.values))
        gs = cell_slopes(Ls)
        ck = sorted(set(ge) & set(gs))
        rr, _, _ = spearman([ge[k] for k in ck], [gs[k] for k in ck], n_perm=200)
        sh.append(rr)
    print(f'   偽の先行度での ρ: {[f"{x:+.3f}" for x in sh]}  平均 {np.mean(sh):+.3f}'
          f'   {"✅ ほぼ0" if abs(np.mean(sh)) < 0.2 else "❌ 計測器が壊れている"}')

    print('\n■ 基準5 対照群B: 本物の効果を注入したら検出できるか（今回の反省から新設）')
    target = common[0]
    Li = L.copy()
    m = ((Li.rc == target[0]) & (Li.sf == target[1]) & (Li.dist == target[2])
         & (Li.lead > Li.lead.median()))
    Li.loc[m, 'res'] = Li.loc[m, 'res'] + 5.0     # 前寄りの馬だけ +5pt
    gi = cell_slopes(Li)
    print(f'   注入先 {target}  傾き {gl[target]:+.2f}pt → {gi[target]:+.2f}pt   '
          f'{"✅ 検出できる" if gi[target] - gl[target] > 2.0 else "❌ 検出できない＝計測器が壊れている"}')


if __name__ == '__main__':
    run(sys.argv[1])
