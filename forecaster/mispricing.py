"""市場のミスプライシング地図: 131軸すべてで「実測3着内率 − 市場示唆3着内率」を測る。

F-3 は「前走着順」1軸だけを見て 人気1-3 で 11.3pt の差を見つけた。
残る130軸で同じことをやっていないので、一望する。

🔴 人気で統制する。しないと131軸すべてが favorite-longshot bias
   （1番人気 84.8% vs 10+人気 61.2%）を再発見するだけになる。

🔴 ROI は見ない。着順と市場人気だけ。
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
pd.set_option('display.width', 240)

import importlib.util
src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'; exec(compile(src, 'analyze.py', 'exec'), az.__dict__)

D = pickle.load(open(f'{BASE}/cut_data.pkl','rb'))
P = pickle.load(open(f'{BASE}/cut_pred.pkl','rb'))
FEAT = D['feat']
df = pd.read_csv(f'{BASE}/data/horse_features.csv')

TUNE, CONF = ['T1','T2','T3'], ['W1','W2','W3']
POPB = [(1,1,'1番人気'), (2,3,'2-3'), (4,5,'4-5'), (6,9,'6-9'), (10,99,'10+')]


def build():
    rows = []
    for w in TUNE + CONF:
        win = D['windows'][w]; vi = win['va_idx']
        pop = win['popv'].astype(float); fs = D['meta'].iloc[vi]['field'].to_numpy(float)
        base, gp = az.market_table(win['popi'].astype(float),
                                   D['meta'].iloc[win['in_idx']]['field'].to_numpy(float),
                                   win['y3i'])
        mkt = az.apply_market(base, gp, pop, fs)
        ok = np.isfinite(pop) & (pop > 0) & (pop < 99)
        sub = df.iloc[vi][FEAT].copy()
        sub['__y3'] = win['y3v'].astype(int); sub['__mkt'] = mkt
        sub['__pop'] = pop; sub['__per'] = '探索' if w in TUNE else '確認'
        rows.append(sub[ok])
    return pd.concat(rows, ignore_index=True)


def main():
    f = build()
    print(f'行 {len(f):,}（探索 {(f.__per=="探索").sum():,} / 確認 {(f.__per=="確認").sum():,}）')
    # 検算: 市場は平均では較正されているはず
    for per in ['探索','確認']:
        a = f[f.__per==per]
        print(f'  検算 {per}: 実測3着内 {a.__y3.mean()*100:.2f}%  市場示唆平均 {a.__mkt.mean()*100:.2f}%  '
              f'ズレ {(a.__y3.mean()-a.__mkt.mean())*100:+.2f}pt')
    f['__res'] = f['__y3'] - f['__mkt']          # 人気で説明できない分
    f['__pb'] = 0
    for i,(lo,hi,_) in enumerate(POPB):
        f.loc[f.__pop.between(lo,hi), '__pb'] = i

    NB = 10
    out = []
    for c in FEAT:
        x = f[c]
        if x.nunique() < 4: continue
        # 🔴 人気帯の中で分位を切る（favorite-longshot bias を再発見しないため）
        try:
            q = f.groupby('__pb')[c].transform(
                lambda s: pd.qcut(s.rank(method='first'), NB, labels=False, duplicates='drop'))
        except Exception:
            continue
        d = {}
        for per in ['探索','確認']:
            a = f[(f.__per==per)]; qa = q[f.__per==per]
            m = a.groupby(qa)['__res'].mean() * 100
            n = a.groupby(qa)['__res'].size()
            m = m[n >= 200]
            if len(m) < 6: d = None; break
            d[per] = m
        if not d: continue
        e, cf = d['探索'], d['確認']
        common = e.index.intersection(cf.index)
        if len(common) < 6: continue
        rho = spearmanr(e[common], cf[common]).statistic
        out.append(dict(feat=c,
                        spread_ex=e.max()-e.min(), spread_cf=cf.max()-cf.min(),
                        hi_cf=cf.max(), lo_cf=cf.min(), rho=rho, nbin=len(common)))
    r = pd.DataFrame(out).sort_values('spread_cf', ascending=False)
    r.to_csv(f'{BASE}/mispricing_map.csv', index=False)

    print(f'\n{"="*112}')
    print('■ 市場のミスプライシング地図（人気帯の中で統制済み・確認期のスプレッド降順）')
    print(f'{"="*112}')
    print(f'  {"特徴量":<28}{"確認期スプレッド":>14}{"探索期":>10}{"最大帯":>9}{"最小帯":>9}{"探索→確認 ρ":>13}')
    for _, x in r.head(25).iterrows():
        print(f'  {x.feat:<28}{x.spread_cf:>13.1f}pt{x.spread_ex:>9.1f}pt'
              f'{x.hi_cf:>+8.1f}pt{x.lo_cf:>+8.1f}pt{x.rho:>+12.2f}')
    print(f'\n  --- 参考: F-3 が見つけた「前走着順」の軸 ---')
    for c in ['f_last1_rank','f_recent','f_last2_rank','f_career_runs','f_days_since_last']:
        s = r[r.feat==c]
        if len(s): 
            x=s.iloc[0]
            print(f'  {c:<28}{x.spread_cf:>13.1f}pt{x.spread_ex:>9.1f}pt{x.hi_cf:>+8.1f}pt{x.lo_cf:>+8.1f}pt{x.rho:>+12.2f}')

    print(f'\n{"="*112}')
    print('■ 🔴 再現性: 探索期と確認期でスプレッドの順位は持ち越すか')
    print(f'{"="*112}')
    rr = spearmanr(r.spread_ex, r.spread_cf)
    print(f'  spread の順位相関  ρ = {rr.statistic:+.3f}  (P={rr.pvalue:.2e})   N={len(r)}軸')
    print(f'  各軸の中の形（10分位の並び）が持ち越す軸: ρ>=+0.5 が {(r.rho>=0.5).sum()}/{len(r)}軸'
          f'  ρ<=-0.5 が {(r.rho<=-0.5).sum()}軸')
    print(f'  確認期スプレッドの分布: 中央値 {r.spread_cf.median():.1f}pt / '
          f'90%点 {r.spread_cf.quantile(.9):.1f}pt / 最大 {r.spread_cf.max():.1f}pt')

    print(f'\n  --- 🔴 帰無仮説: 無関係な軸でもスプレッドは出る（乱数列で同じ計算）---')
    rng = np.random.default_rng(0); nulls=[]
    for k in range(40):
        f['__rnd'] = rng.normal(size=len(f))
        q = f.groupby('__pb')['__rnd'].transform(
            lambda s: pd.qcut(s.rank(method='first'), NB, labels=False, duplicates='drop'))
        a = f[f.__per=='確認']; qa = q[f.__per=='確認']
        m = a.groupby(qa)['__res'].mean()*100
        nulls.append(m.max()-m.min())
    nulls=np.array(nulls)
    print(f'    乱数40本のスプレッド: 中央値 {np.median(nulls):.1f}pt / 95%点 {np.percentile(nulls,95):.1f}pt / 最大 {nulls.max():.1f}pt')
    print(f'    実データで帰無の95%点を超える軸: {(r.spread_cf > np.percentile(nulls,95)).sum()}/{len(r)}軸')


if __name__ == '__main__':
    main()
