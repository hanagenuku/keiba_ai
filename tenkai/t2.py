"""T-2: 学習時入力 vs 推論時入力で、ペース予測の精度がどれだけ違うか。"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
R=pd.read_pickle(f'{B}/tenkai/R.pkl'); T=pd.read_pickle(f'{B}/tenkai/T.pkl'); S=pd.read_pickle(f'{B}/tenkai/S.pkl')
FE=['escape_count','front_count','front_density','avg_agari3f','std_agari3f',
    'n','escape_avg_pos','escape_avg_pop']
def prep(F):
    d=R.merge(F,on='race_id',how='inner').copy()
    d['surface_num']=(d.sf=='芝').astype(int)
    d['cond']= d.tc.map({'良':0,'稍重':1,'重':2,'不良':3}).fillna(0)
    return d.dropna(subset=['first_3f'])
dT,dS=prep(T),prep(S)
COLS=FE+['dist','surface_num','cond']
CUT='2026-01-01'
lab={'high':0,'mid':1,'slow':2}
def run(tr_d,te_d,nm):
    tr=tr_d[tr_d.date<CUT]; te=te_d[te_d.date>=CUT]
    common=set(te.race_id)&set(te_d.race_id)
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,random_state=42,
                        objective='multi:softprob',num_class=3,verbosity=0)
    m.fit(tr[COLS].fillna(-1),tr.pace.map(lab))
    p=m.predict(te[COLS].fillna(-1))
    return accuracy_score(te.pace.map(lab),p),len(tr),len(te)

print(f'学習 〜{CUT} / 検定 {CUT}〜（完全OOS）  ラベルは3分類・無作為 = 33.3%')
print(f'多数派クラス（slow）を常に答える = {100*(R[R.date>=CUT].pace=="slow").mean():.1f}%\n')

a,n1,n2=run(dT,dT,'TT'); print(f'① 学習=実際 / 検定=実際     精度 {100*a:.2f}%  (学習{n1:,}/検定{n2:,})')
print('     ← これが「Val Accuracy 0.5436」として報告されてきた数字')
# 同じレースで揃えて比較する
key=set(dT.race_id)&set(dS.race_id)
dT2=dT[dT.race_id.isin(key)]; dS2=dS[dS.race_id.isin(key)]
trT=dT2[dT2.date<CUT]; teS=dS2[dS2.date>=CUT].sort_values('race_id')
teT=dT2[dT2.date>=CUT].sort_values('race_id')
m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                    colsample_bytree=0.8,random_state=42,objective='multi:softprob',
                    num_class=3,verbosity=0)
m.fit(trT[COLS].fillna(-1),trT.pace.map(lab))
aTT=accuracy_score(teT.pace.map(lab),m.predict(teT[COLS].fillna(-1)))
aTS=accuracy_score(teS.pace.map(lab),m.predict(teS[COLS].fillna(-1)))
print(f'② 学習=実際 / 検定=実際（同一レース）  精度 {100*aTT:.2f}%  N={len(teT):,}')
print(f'③ 学習=実際 / 検定=推論時入力          精度 {100*aTS:.2f}%  ← 🔴 **これが本番**')
print(f'     ①②と③の差 = {100*(aTT-aTS):+.2f}pt が学習/推論の食い違いで失われている分')
trS=dS2[dS2.date<CUT]
m2=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                     colsample_bytree=0.8,random_state=42,objective='multi:softprob',
                     num_class=3,verbosity=0)
m2.fit(trS[COLS].fillna(-1),trS.pace.map(lab))
aSS=accuracy_score(teS.pace.map(lab),m2.predict(teS[COLS].fillna(-1)))
print(f'④ 学習=推論時入力 / 検定=推論時入力    精度 {100*aSS:.2f}%  ← 揃えるだけで {100*(aSS-aTS):+.2f}pt')
