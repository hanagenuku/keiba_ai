"""O' Step 6: 二つの穴を塞ぐ。
  (a) 本番データで「構造条件だけ」を測り、sim_ev フィルタの寄与を分離する
  (b) OOS の 10+人気セルで pace_bonus に符号反転・シャッフル対照を当てる
"""
import pickle, sqlite3, numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'

# ---------- (a) 本番データ ----------
exec(open(f'{B}/osusume/o1_mark.py').read().split("for lo,lab in")[0])
sub=d.reset_index(drop=True); r_on,pn_on=arms['ON']
sub['osu']=marks(r_on,pn_on,sub); sub['ai']=r_on
LS=sub.popularity>=10
struct=LS&(sub.ai<=sub.popularity-3)&(sub.ai<=8)
def st(m):
    x=sub[m]
    if not len(x): return None
    bad=(x.actual_place<=3)&(x.fukusho_payout.fillna(0)<=0); ok=x[~bad]
    return (len(x),100*(x.actual_place<=3).mean(),
            100*ok.fukusho_payout.fillna(0).sum()/(100*max(len(ok),1)))
print('■ (a) 本番 729レース・10番人気以下  sim_ev フィルタの寄与')
for lab,m in (('構造条件のみ',struct),('  うち 推（sim_ev>=1.2・上位2頭）',struct&sub.osu),
              ('  うち sim_ev で落選',struct&~sub.osu)):
    n,f,r=st(m); print(f'   {lab:34s} N={n:4d} 複勝率{f:5.1f}% 複ROI {r:6.1f}%')

# ---------- (b) OOS で対照 ----------
P=pickle.load(open(f'{B}/cutrace/q/predq.pkl','rb')); O=P['OOS']
df=pd.read_csv(f'{B}/cutrace/data/horse_features.csv',
               usecols=['race_id','horse_num','date','f_pace','f_pace_prob_slow','f_pace_prob_fast'])
df['do']=pd.to_datetime(df['date'].astype(str).str.replace('-','',regex=False).str[:8],
                        format='%Y%m%d',errors='coerce')
df=df.dropna(subset=['do']).reset_index(drop=True)
s=df.iloc[O['idx']].reset_index(drop=True)
e=pd.DataFrame({'rid':O['rid'],'num':s.horse_num.to_numpy(),'date':O['date'],
                'raw':O['raw'].astype(float),'pop':O['pop'],'y3':O['y3'],
                'f_pace':s.f_pace.to_numpy(float),'ps':s.f_pace_prob_slow.to_numpy(float),
                'pf':s.f_pace_prob_fast.to_numpy(float)})
g=e.groupby('rid'); e['P0']=1/(1+np.exp(-e.raw))*10
e['pb']=(e.f_pace-g['f_pace'].transform('mean'))*(e.ps-e.pf)*0.5
mn,mx=g['P0'].transform('min'),g['P0'].transform('max')
rel=(e.P0-mn)/np.maximum(mx-mn,0.1)*10
rng=np.random.default_rng(42)
e['pbS']=e.groupby('rid')['pb'].transform(lambda x: rng.permutation(x.values))
for nm,pb in (('ON',e.pb),('NEG',-e.pb),('SHUF',e.pbS),('OFF',0)):
    e[nm]=0.9*e.P0+0.1*rel+pb
    e[f'a_{nm}']=e.groupby('rid')[nm].rank(ascending=False,method='first').astype(int)
con=sqlite3.connect(f'{B}/cutrace/data/history.db')
hh=pd.read_sql('SELECT race_id,horse_num,fukusho_payout FROM horse_history',con);con.close()
e=e.merge(hh,left_on=['rid','num'],right_on=['race_id','horse_num'],how='left')
L=(e['pop']>=10)
print('\n■ (b) OOS 2,057レース・10番人気以下  pace_bonus の対照')
def st2(m,drop=0):
    x=e[m]
    bad=(x.y3==1)&(x.fukusho_payout.fillna(0)<=0); ok=x[~bad]
    fp=ok.fukusho_payout.fillna(0).to_numpy().copy()
    for _ in range(drop):
        if len(fp) and fp.max()>0: fp[fp.argmax()]=0
    return (len(x),100*(x.y3==1).mean(),100*fp.sum()/(100*max(len(ok),1)))
for nm in ('ON','SHUF','NEG','OFF'):
    a=e[f'a_{nm}']; m=L&(a<=e['pop']-3)&(a<=8)
    n,f,r=st2(m); _,_,r3=st2(m,drop=3)
    print(f'   {nm:5s} N={n:4d} 複勝率{f:5.1f}% 複ROI {r:6.1f}%  3本抜 {r3:6.1f}%')
