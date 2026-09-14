"""T-5: どの特徴量群が効いているか。実装を軽くできるか（P(lead)モデルは要るか）。"""
import numpy as np, pandas as pd, xgboost as xgb
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
R=pd.read_pickle(f'{B}/tenkai/R.pkl'); F=pd.read_pickle(f'{B}/tenkai/F.pkl')
d=R.merge(F,on='race_id',how='inner')
d['surface_num']=(d.sf=='芝').astype(int)
d['cond']=d.tc.map({'良':0,'稍重':1,'重':2,'不良':3}).fillna(0)
d['cls']=d.race_class.astype('category').cat.codes
CUT='2026-01-01'; tr,te=d[d.date<CUT],d[d.date>=CUT]
y_tr,y_te=tr.first_3f.values,te.first_3f.values
COND=['dist','surface_num','cond','n','cls']
def ev(cols,nm,base=None):
    m=xgb.XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.04,subsample=0.8,
                       colsample_bytree=0.8,random_state=42,verbosity=0)
    m.fit(tr[cols].fillna(-1),y_tr); p=m.predict(te[cols].fillna(-1))
    r=float(np.sqrt(((p-y_te)**2).mean()))
    print(f'   {nm:40s} RMSE {r:.4f}s' + ('' if base is None else f'  Δ{r-base:+.4f}s'))
    return r
b=ev(COND,'条件だけ')
P=['P_max','P_2nd','P_3rd','P_ent','P_hhi','P_gap','P_top2']   # ロジスティック由来
POS=['n_can_lead','n_front','pos_mean','pos_min','pos_std']     # 過去の通過位置だけ
AG=['ag_mean','ag_min']                                        # 過去の上がり
POP=['pop_of_leader']
print()
ev(COND+POS,'+ 過去の通過位置だけ（モデル不要）',b)
ev(COND+P,'+ P(先頭)モデル由来だけ',b)
ev(COND+AG,'+ 過去の上がりだけ',b)
ev(COND+POS+AG,'+ 通過位置 + 上がり',b)
ev(COND+POS+AG+POP,'+ 通過位置 + 上がり + 逃げ馬の人気',b)
ev(COND+POS+P+AG+POP,'+ 全部',b)
