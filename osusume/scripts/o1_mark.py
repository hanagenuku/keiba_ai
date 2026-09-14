"""O' Step 1: 「推」マークの回収率を ON/OFF で比べる。osusume/CRITERIA.md 事前登録。"""
import numpy as np, pandas as pd, sqlite3
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
d=pd.read_pickle(f'{B}/osusume/d.pkl')

# 実配当
con=sqlite3.connect(f'{B}/cutrace/data/history.db')
hh=pd.read_sql('SELECT race_id,horse_num,tansho_payout,fukusho_payout FROM horse_history',con)
con.close()
d=d.merge(hh,on=['race_id','horse_num'],how='left')
d=d[d.actual_place.notna()&(d.actual_place>0)].copy()
d=d[(d.tansho_odds.notna())&(d.tansho_odds>=1.0)].copy()
print(f'評価対象 {len(d)}頭 / {d.race_id.nunique()}レース  {d.date.min()}〜{d.date.max()}')

def softmax_race(t, rid):
    s=pd.Series(t).groupby(rid)
    m=s.transform('max')
    e=np.exp((t-m)/3.5)
    return e/pd.Series(e).groupby(rid).transform('sum')

rng=np.random.default_rng(42)
d['pb_neg']=-d.pb
d['pb_shuf']=d.groupby('race_id')['pb'].transform(lambda s: rng.permutation(s.values))

arms={}
# ON は本番そのもの
arms['ON']=(d.rl_rank.to_numpy(), d.win_prob.to_numpy())
for nm,col in (('OFF',None),('NEG','pb_neg'),('SHUF','pb_shuf')):
    if nm=='OFF': t=d.tON-d.pb
    else:         t=d.tON-d.pb+d[col]          # pb を外して別の pb を足す
    pn=softmax_race(t.to_numpy(), d.race_id.to_numpy())
    r=pd.Series(t.to_numpy()).groupby(d.race_id.to_numpy()).rank(ascending=False,method='first')
    arms[nm]=(r.to_numpy().astype(int), np.asarray(pn))

def marks(rank, pn, sub):
    """_assign_marks と同じ規則で 推 を決める（高マークの除外も再現）"""
    ev = pn*sub.tansho_odds.to_numpy()
    pop= sub.popularity.to_numpy()
    taka = (pop<=3)&(rank<=3)&(ev>=1.0)
    cand = (rank<=pop-3)&(rank<=8)&(ev>=1.2)&(~taka)
    out=np.zeros(len(sub),bool)
    tmp=pd.DataFrame({'rid':sub.race_id.to_numpy(),'ev':ev,'c':cand,'i':np.arange(len(sub))})
    for _,x in tmp[tmp.c].groupby('rid'):
        for i in x.nlargest(2,'ev')['i']: out[i]=True
    return out

def roi(sub, sel, drop_top=0):
    s=sub[sel]
    if not len(s): return None
    tp=s.tansho_payout.fillna(0).to_numpy().copy()
    fp=s.fukusho_payout.fillna(0).to_numpy().copy()
    # 3着内なのに複勝配当が無い行は評価不能（North Star: 0円として混ぜない）
    bad=(s.actual_place<=3)&(s.fukusho_payout.fillna(0)<=0)
    for k in range(drop_top):
        if tp.max()>0: tp[tp.argmax()]=0
        if fp.max()>0: fp[fp.argmax()]=0
    n=len(s); nf=int((~bad).sum())
    return dict(n=n, win=int((s.actual_place==1).sum()), fuku=int((s.actual_place<=3).sum()),
                tan_roi=100*tp.sum()/(100*n),
                fuku_roi=100*fp[~bad.to_numpy()].sum()/(100*max(nf,1)), nf=nf)

for lo,lab in (('2026-06-27','全期間 (error_tags ON期を含む)'),
               ('2026-09-01','09-01以降 (error_tags OFF・器が最も清い)')):
    sub=d[d.date>=lo].reset_index(drop=True)
    idx=d.index.isin(d[d.date>=lo].index)
    print(f'\n{"="*78}\n■ {lab}   {sub.race_id.nunique()}レース / {len(sub)}頭\n{"="*78}')
    print(f'{"腕":6s} {"推":>5s} {"勝":>4s} {"複":>4s} {"単ROI":>8s} {"複ROI":>8s} '
          f'{"単3本抜":>8s} {"複3本抜":>8s}')
    for nm,(r,pn) in arms.items():
        m=marks(r[idx], pn[idx], sub)
        a=roi(sub,m); b=roi(sub,m,drop_top=3)
        if a is None: print(f'{nm:6s}  該当なし'); continue
        print(f'{nm:6s} {a["n"]:5d} {a["win"]:4d} {a["fuku"]:4d} '
              f'{a["tan_roi"]:7.1f}% {a["fuku_roi"]:7.1f}% '
              f'{b["tan_roi"]:7.1f}% {b["fuku_roi"]:7.1f}%')
