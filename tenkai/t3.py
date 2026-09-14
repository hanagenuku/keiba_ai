"""T-3: 展開予想を作り直す。全特徴量を過去走のみから作り、隊列争いを連続量で表す。

  ユーザーの発想「逃げ馬が何頭いるか、人気の先行馬がやり合えば差しが決まる」を
  4分類ラベルではなく **P(先頭に立つ) の分布** で表現する。
"""
import sqlite3, numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
R=pd.read_pickle(f'{B}/tenkai/R.pkl')

con=sqlite3.connect(f'{B}/hist3.db')
H=pd.read_sql("""SELECT h.race_id,h.date,h.horse_name,h.horse_num,h.corner_all,h.running_style,
                        h.popularity pop,h.jockey,h.distance dist,h.agari3f,h.place,
                        r.num_finishers fs
                 FROM horse_history h LEFT JOIN race_history r USING(race_id)
                 WHERE h.place>0 AND h.place<99""",con); con.close()
H['fs']=H.fs.fillna(H.groupby('race_id')['horse_num'].transform('max'))
H=H[H.fs.between(5,18)].copy()
ca=H.corner_all.fillna('')
H['c1']=pd.to_numeric(ca.str.split('-').str[0],errors='coerce')
H=H[H.c1.notna()].copy()
H['led']=(H.c1==1).astype(int)
H['pos_n']=(H.c1-1)/np.maximum(H.fs-1,1)
H=H.sort_values(['horse_name','date']).reset_index(drop=True)
g=H.groupby('horse_name')
# ── 過去走のみ（当該レースを除く）
H['p_led']   =g['led'].transform(lambda s:s.shift().expanding().mean())
H['p_mean']  =g['pos_n'].transform(lambda s:s.shift().expanding().mean())
H['p_min']   =g['pos_n'].transform(lambda s:s.shift().expanding().min())
H['p_last']  =g['pos_n'].transform(lambda s:s.shift())
H['p_std']   =g['pos_n'].transform(lambda s:s.shift().expanding().std())
H['p_ag']    =g['agari3f'].transform(lambda s:s.shift().expanding().mean())   # 過去の上がり
H['p_n']     =g['led'].transform(lambda s:s.shift().expanding().count())
H['d_dist']  =H.dist-g['dist'].transform(lambda s:s.shift())
H=H.sort_values(['jockey','date'])
H['j_led']=H.groupby('jockey')['led'].transform(lambda s:s.shift().expanding().mean())
H=H.sort_values('date').reset_index(drop=True)
H['draw']=H.horse_num/H.fs

# ── 各馬の P(先頭に立つ) を作る（学習は2024以前で、以降に当てる）
FL=['p_led','p_mean','p_min','p_last','p_std','p_n','d_dist','draw','fs','j_led','dist']
tr=H[H.date<'2025-01-01']
X=tr[FL].fillna(tr[FL].median()); mu,sd=X.mean(),X.std().replace(0,1)
lm=LogisticRegression(max_iter=3000).fit((X-mu)/sd,tr.led)
H['P']=lm.predict_proba((H[FL].fillna(tr[FL].median())-mu)/sd)[:,1]
H['Pn']=H.P/H.groupby('race_id')['P'].transform('sum')      # レース内で正規化

# ── レース単位の「隊列争い」特徴量（すべて過去走由来）
def top(s,k): 
    v=np.sort(s.values)[::-1]; return v[k] if len(v)>k else np.nan
G=H.groupby('race_id')
F=pd.DataFrame({
 'n':G['horse_num'].size(),
 'P_max':G['Pn'].max(),
 'P_2nd':G['Pn'].apply(lambda s: top(s,1)),
 'P_3rd':G['Pn'].apply(lambda s: top(s,2)),
 'P_ent':G['Pn'].apply(lambda s: -(s*np.log(s+1e-12)).sum()),   # 逃げ争いの散らばり
 'P_hhi':G['Pn'].apply(lambda s: (s**2).sum()),                 # 集中度
 'n_can_lead':G['p_min'].apply(lambda s:(s<0.10).sum()),        # ハナを切った経験がある頭数
 'n_front':G['p_mean'].apply(lambda s:(s<0.30).sum()),          # 先行型の頭数
 'pos_mean':G['p_mean'].mean(), 'pos_min':G['p_mean'].min(), 'pos_std':G['p_mean'].std(),
 'ag_mean':G['p_ag'].mean(), 'ag_min':G['p_ag'].min(),          # 過去の上がり（最終スピード）
 'pop_of_leader':G.apply(lambda x: x.loc[x.Pn.idxmax(),'pop'] if len(x) else np.nan,
                         include_groups=False),
}).reset_index()
F['P_gap']=F.P_max-F.P_2nd          # 単騎逃げか、やり合いか
F['P_top2']=F.P_max+F.P_2nd
F.to_pickle(f'{B}/tenkai/F.pkl')

d=R.merge(F,on='race_id',how='inner')
d['surface_num']=(d.sf=='芝').astype(int)
d['cond']=d.tc.map({'良':0,'稍重':1,'重':2,'不良':3}).fillna(0)
lab={'high':0,'mid':1,'slow':2}
NEW=['P_max','P_2nd','P_3rd','P_ent','P_hhi','P_gap','P_top2','n_can_lead','n_front',
     'pos_mean','pos_min','pos_std','ag_mean','ag_min','pop_of_leader','n',
     'dist','surface_num','cond']
CUT='2026-01-01'
tr,te=d[d.date<CUT],d[d.date>=CUT]
print(f'学習 {len(tr):,} / 検定 {len(te):,}レース（完全OOS）')
m=xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,
                    colsample_bytree=0.8,random_state=42,objective='multi:softprob',
                    num_class=3,verbosity=0)
m.fit(tr[NEW].fillna(-1),tr.pace.map(lab))
acc=accuracy_score(te.pace.map(lab),m.predict(te[NEW].fillna(-1)))
print(f'\n■ 3分類の精度')
print(f'   無作為                     33.3%')
print(f'   ③ 本番の実力（既存・学習/推論ずれ）  50.45%')
print(f'   ④ 既存を揃えただけ               52.88%')
print(f'   ⑤ 作り直し（隊列争いを連続量に）    {100*acc:.2f}%')
imp=pd.Series(m.feature_importances_,index=NEW).sort_values(ascending=False)
print(f'\n   重要度Top8: '+' / '.join(f'{k} {100*v:.1f}%' for k,v in imp.head(8).items()))
