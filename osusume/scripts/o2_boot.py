"""O' Step 2: 上の差が分解能の中かを日ブロックbootstrapで測る。"""
import numpy as np, pandas as pd, sqlite3
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
exec(open(f'{B}/osusume/o1_mark.py').read().split("for lo,lab in")[0])

sub=d.reset_index(drop=True); idx=np.ones(len(d),bool)
M={nm:marks(r[idx],pn[idx],sub) for nm,(r,pn) in arms.items()}
days=sub.date.to_numpy()
udays=np.unique(days)
print(f'開催日 {len(udays)}日 / 推マーク ON {M["ON"].sum()} OFF {M["OFF"].sum()} '
      f'NEG {M["NEG"].sum()} SHUF {M["SHUF"].sum()}')

def roi_of(mask, day_sel, kind):
    s=sub[mask & np.isin(days, day_sel)]
    if not len(s): return np.nan
    if kind=='tan':
        return 100*s.tansho_payout.fillna(0).sum()/(100*len(s))
    bad=(s.actual_place<=3)&(s.fukusho_payout.fillna(0)<=0)
    ok=s[~bad]
    return 100*ok.fukusho_payout.fillna(0).sum()/(100*max(len(ok),1)) if len(ok) else np.nan

rng=np.random.default_rng(7)
for kind,lab in (('tan','単勝'),('fuku','複勝')):
    print(f'\n── {lab}回収率 ─────────────────────────')
    base={nm:roi_of(M[nm],udays,kind) for nm in M}
    print('   実測  ' + '  '.join(f'{nm} {base[nm]:.1f}%' for nm in M))
    for other in ('OFF','NEG','SHUF'):
        diffs=[]
        for _ in range(2000):
            ds=rng.choice(udays,len(udays),replace=True)
            a=roi_of(M['ON'],ds,kind); b=roi_of(M[other],ds,kind)
            if not (np.isnan(a) or np.isnan(b)): diffs.append(a-b)
        diffs=np.array(diffs)
        lo,hi=np.percentile(diffs,[2.5,97.5])
        print(f'   ON − {other:4s} = {base["ON"]-base[other]:+7.1f}pt  '
              f'95%CI [{lo:+7.1f}, {hi:+7.1f}]  ONが上の回 {(diffs>0).mean():5.1%}')
