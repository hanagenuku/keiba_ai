"""⑤⑥ を高速に測り直す（pd.concat の2,000回ループをやめて numpy のインデックスにする）。"""
import os, sys, pickle
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.models.predict import softmax_probs
from src.features.engine import calc_harville_probs
Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb'))
META = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))['meta']
B = Q['OOS']; p_full = 1/(1+np.exp(-B['raw']))
md = META.iloc[B['idx']]
d = pd.DataFrame({'rid': B['rid'], 'hn': md['horse_num'].to_numpy(), 'p': p_full,
                  'y3': B['y3'], 'pop': B['pop'], 'cls': md['race_class'].fillna('').to_numpy()})
rows = []
for rid, g in d.groupby('rid', sort=False):
    wp = softmax_probs([v*10 for v in g['p']], temperature=3.5)
    t3 = [b for _, b in calc_harville_probs(wp)]
    i = int(np.argmax(g['p'].to_numpy()))
    pp = g['pop'].to_numpy(float)
    mi = int(np.nanargmin(np.where(np.isfinite(pp) & (pp > 0), pp, 99)))
    rows.append(dict(rid=rid, cls=g['cls'].iloc[0], ax_p=t3[i],
                     ax_y3=int(g['y3'].iloc[i]), ax_hn=int(g['hn'].iloc[i]),
                     mk_y3=int(g['y3'].iloc[mi]), mk_hn=int(g['hn'].iloc[mi])))
R = pd.DataFrame(rows); R['day'] = [r[:8] for r in R['rid']]
order = ['新馬', '未勝利', '1勝', '2勝', '3勝', 'OP']
oth = R[R['cls'].isin(['1勝', '2勝', '3勝', 'OP'])]
rng = np.random.default_rng(42)
print('■ 5 「新馬でAIが市場に最も負ける」は頑健か（日ブロック bootstrap 2,000回）')
print(f'  {"クラス":<10}{"R数":>6}{"AI軸":>8}{"市場1人気":>10}{"差":>9}{"95%CI(差)":>19}{"前半":>9}{"後半":>9}')
for c in order + ['（1勝以上）']:
    g = oth if c == '（1勝以上）' else R[R['cls'] == c]
    if len(g) < 30: continue
    days, codes = np.unique(g['day'].to_numpy(), return_inverse=True)
    by = [np.where(codes == k)[0] for k in range(len(days))]
    a3, m3 = g['ax_y3'].to_numpy(), g['mk_y3'].to_numpy()
    bs = np.empty(2000)
    for t in range(2000):
        ix = np.concatenate([by[k] for k in rng.integers(0, len(days), len(days))])
        bs[t] = (a3[ix].mean() - m3[ix].mean()) * 100
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    mid = np.sort(days)[len(days)//2]
    h1, h2 = g['day'].to_numpy() < mid, g['day'].to_numpy() >= mid
    f = lambda m: (a3[m].mean()-m3[m].mean())*100 if m.sum() else np.nan
    print(f'  {c:<10}{len(g):>6,}{a3.mean()*100:>7.1f}%{m3.mean()*100:>9.1f}%'
          f'{(a3.mean()-m3.mean())*100:>+8.1f}pt{f"[{lo_:+.1f}, {hi_:+.1f}]":>19}'
          f'{f(h1):>+8.1f}pt{f(h2):>+8.1f}pt')
print('\n■ 6 なぜ新馬で負けるか — 軸が市場1番人気と一致した割合')
print(f'  {"クラス":<10}{"R数":>6}{"軸=市場1人気":>13}{"一致時の軸":>12}{"不一致時の軸":>13}{"不一致時の市場":>14}')
for c in order + ['（1勝以上）']:
    g = oth if c == '（1勝以上）' else R[R['cls'] == c]
    if len(g) < 30: continue
    s = (g['ax_hn'] == g['mk_hn']).to_numpy()
    print(f'  {c:<10}{len(g):>6,}{s.mean()*100:>12.1f}%{g["ax_y3"].to_numpy()[s].mean()*100:>11.1f}%'
          f'{g["ax_y3"].to_numpy()[~s].mean()*100:>12.1f}%{g["mk_y3"].to_numpy()[~s].mean()*100:>13.1f}%')
