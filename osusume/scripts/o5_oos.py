"""O' Step 5: 「10番人気以下 × AIが人気より3以上高い × AI8位以内」を
   完全OOS 2,388レースで、市場最上位対照つき・四半期分割で測る。

  推の構造条件だけを使う（sim_ev はオッズが要るので外す。
  Step 3 で sim_ev<1.2 の落選組が単ROI 114.2% だったので、外す方が保守的でもない）。
"""
import pickle, sqlite3, numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
P=pickle.load(open(f'{B}/cutrace/q/predq.pkl','rb')); O=P['OOS']
df=pd.read_csv(f'{B}/cutrace/data/horse_features.csv',
               usecols=['race_id','horse_num','date','f_pace','f_pace_prob_slow','f_pace_prob_fast'])
df['date_obj']=pd.to_datetime(df['date'].astype(str).str.replace('-','',regex=False).str[:8],
                              format='%Y%m%d',errors='coerce')
df=df.dropna(subset=['date_obj']).reset_index(drop=True)
s=df.iloc[O['idx']].reset_index(drop=True)
assert (s['race_id'].to_numpy()==O['rid']).all()

d=pd.DataFrame({'rid':O['rid'],'num':s.horse_num.to_numpy(),'date':O['date'],
                'raw':O['raw'].astype(float),'pop':O['pop'],'field':O['field'],
                'y1':O['y1'],'y3':O['y3'],
                'f_pace':s.f_pace.to_numpy(float),
                'ps':s.f_pace_prob_slow.to_numpy(float),'pf':s.f_pace_prob_fast.to_numpy(float)})
g=d.groupby('rid')
d['P0']=1/(1+np.exp(-d.raw))*10
d['pb']=(d.f_pace-g['f_pace'].transform('mean'))*(d.ps-d.pf)*0.5
mn,mx=g['P0'].transform('min'),g['P0'].transform('max')
rel=(d.P0-mn)/np.maximum(mx-mn,0.1)*10
d['P1']=0.9*d.P0+0.1*rel+d.pb
for c in ('P0','P1'):
    d[f'ai_{c}']=d.groupby('rid')[c].rank(ascending=False,method='first').astype(int)

con=sqlite3.connect(f'{B}/cutrace/data/history.db')
hh=pd.read_sql('SELECT race_id,horse_num,fukusho_payout,tansho_payout FROM horse_history',con);con.close()
d=d.merge(hh,left_on=['rid','num'],right_on=['race_id','horse_num'],how='left')
d=d[d['pop'].between(1,99)].copy()
LS=(d['pop']>=10)
print(f'OOS {len(d)}頭 / {d.rid.nunique()}レース  10番人気以下 {int(LS.sum())}頭 / '
      f'{d[LS].rid.nunique()}レース  {d.date.min()}〜{d.date.max()}')

sel={}
for arm in ('P1','P0'):
    a=d[f'ai_{arm}']
    sel[f'★構造条件 ({"pace ON" if arm=="P1" else "pace OFF"})'] = LS&(a<=d['pop']-3)&(a<=8)
i1=d[LS].groupby('rid')['pop'].idxmin(); sel['C1 同帯の市場最上位']=d.index.isin(i1)
i2=d[LS].loc[d[LS].groupby('rid')['ai_P1'].idxmin().values].index
sel['C2 同帯のAI最上位']=d.index.isin(i2)
rng=np.random.default_rng(11)
i3=d[LS].groupby('rid').apply(lambda x: rng.choice(x.index),include_groups=False)
sel['C3 同帯から無作為']=d.index.isin(i3.values)
sel['C4 同帯すべて']=LS

def stat(m,sub=None,drop=0):
    x=d[m] if sub is None else d[m&sub]
    if not len(x): return None
    bad=(x.y3==1)&(x.fukusho_payout.fillna(0)<=0)
    ok=x[~bad]; fp=ok.fukusho_payout.fillna(0).to_numpy().copy()
    for _ in range(drop):
        if len(fp) and fp.max()>0: fp[fp.argmax()]=0
    return dict(n=len(x),f=int((x.y3==1).sum()),
                fr=100*fp.sum()/(100*max(len(ok),1)),
                tr=100*x.tansho_payout.fillna(0).sum()/(100*len(x)))

q=pd.to_datetime(d.date).dt.quarter
print(f'\n{"":26s} {"N":>6s} {"複勝率":>6s} {"複ROI":>7s} {"3本抜":>7s} {"Q1":>7s} {"Q2":>7s} {"Q3":>7s}')
for k,m in sel.items():
    a=stat(m); b=stat(m,drop=3)
    qs=[stat(m,(q==i)) for i in (1,2,3)]
    print(f'{k:26s} {a["n"]:6d} {100*a["f"]/a["n"]:5.1f}% {a["fr"]:6.1f}% {b["fr"]:6.1f}% '
          + ' '.join(f'{(x["fr"] if x else float("nan")):6.1f}%' for x in qs))

print('\n── 日ブロックbootstrap: 構造条件(pace ON) − 各対照（複勝回収）──')
days=np.sort(d.date.unique()); rng2=np.random.default_rng(3)
key='★構造条件 (pace ON)'
for k in ['C1 同帯の市場最上位','C2 同帯のAI最上位','C3 同帯から無作為','C4 同帯すべて','★構造条件 (pace OFF)']:
    diffs=[]
    for _ in range(1500):
        ds=set(rng2.choice(days,len(days),replace=True))
        mk=d.date.isin(ds)
        a=stat(sel[key],mk); b=stat(sel[k],mk)
        if a and b: diffs.append(a['fr']-b['fr'])
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
    base=stat(sel[key])['fr']-stat(sel[k])['fr']
    print(f'   vs {k:26s} {base:+7.1f}pt  CI[{lo:+7.1f},{hi:+7.1f}]  上回る {(diffs>0).mean():5.1%}')
