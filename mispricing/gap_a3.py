"""上位5軸の除外を「実際に重ねて」測る。gap_a2 の 89.6% は重複を無視した上界なので検算する。"""
import os, sys, pickle
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(src, 'analyze.py', 'exec'), az.__dict__)
D = pickle.load(open(f'{BASE}/cut_data.pkl','rb')); FEAT = D['feat']
df = pd.read_csv(f'{BASE}/data/horse_features.csv')
CONF = ['W1','W2','W3']; POPB=[(1,1),(2,3),(4,5),(6,9),(10,99)]; NB=10
POOL_ROI, POOL_RES, RATE = 69.9, -0.37, 4.75

rows=[]
for w in CONF:
    win=D['windows'][w]; vi=win['va_idx']
    pop=win['popv'].astype(float); fs=D['meta'].iloc[vi]['field'].to_numpy(float)
    base,gp=az.market_table(win['popi'].astype(float),
        D['meta'].iloc[win['in_idx']]['field'].to_numpy(float), win['y3i'])
    mkt=az.apply_market(base,gp,pop,fs)
    ok=np.isfinite(pop)&(pop>0)&(pop<99)
    s=df.iloc[vi][FEAT].copy(); s['__y3']=win['y3v'].astype(int); s['__mkt']=mkt; s['__pop']=pop
    rows.append(s[ok])
f=pd.concat(rows,ignore_index=True)
f['__pb']=0
for i,(lo,hi) in enumerate(POPB): f.loc[f.__pop.between(lo,hi),'__pb']=i

CUTS=[('f_agari_at_similar','hi'),('rl_f_speed_fig_max_vs_field','lo'),
      ('rl_f_pl_rating','lo'),('rl_f_speed_fig_max_rank','hi'),('cl_f_jockey_vs_field','lo')]
def mask(c,side):
    q=f.groupby('__pb')[c].transform(lambda s: pd.qcut(s.rank(method='first'),NB,labels=False,duplicates='drop'))
    return (q==q.max()) if side=='hi' else (q==0)
def roi(m):
    s=f[m]; res=(s.__y3.mean()-s.__mkt.mean())*100
    return len(s), res, POOL_ROI+RATE*(res-POOL_RES)

u=np.zeros(len(f),bool)
print(f'{"除外条件":<36}{"単独N":>8}{"累積N":>8}{"累積除外率":>10}{"切った側":>9}{"残りROI":>9}{"改善":>8}')
for c,side in CUTS:
    m=mask(c,side).to_numpy(); n1,_,_=roi(m)
    u=u|m
    nb,resb,roib=roi(u); w=nb/len(f)
    ng,resg,roig=roi(~u)
    print(f'{c+"/"+side:<36}{n1:>8,}{nb:>8,}{w*100:>9.1f}%{roib:>8.1f}%{roig:>8.1f}%{roig-POOL_ROI:>+7.2f}pt')
print(f'\n🔴 重複を無視した上界は 89.6% だったが、実際に重ねると上表の最終行が答え。')
print(f'   全馬 {len(f):,}行 / 除外後 {(~u).sum():,}行')
