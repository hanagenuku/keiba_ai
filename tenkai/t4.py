"""T-4: 展開予想の精度を実数(前半3F)で測る。
   「条件から機械的に決まる分」を先に引いて、**隊列情報がどれだけ足せるか**を見る。
"""
import numpy as np, pandas as pd, xgboost as xgb
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
R=pd.read_pickle(f'{B}/tenkai/R.pkl'); F=pd.read_pickle(f'{B}/tenkai/F.pkl')
S=pd.read_pickle(f'{B}/tenkai/S.pkl')     # 既存19特徴量の推論時版
d=R.merge(F,on='race_id',how='inner').merge(
    S[['race_id','escape_count','front_count','front_density','escape_avg_pos','escape_avg_pop']],
    on='race_id',how='left')
d['surface_num']=(d.sf=='芝').astype(int)
d['cond']=d.tc.map({'良':0,'稍重':1,'重':2,'不良':3}).fillna(0)
d['cls']=d.race_class.astype('category').cat.codes
CUT='2026-01-01'; tr,te=d[d.date<CUT],d[d.date>=CUT]
y_tr,y_te=tr.first_3f.values,te.first_3f.values
print(f'学習 {len(tr):,} / 検定 {len(te):,}レース  前半3F 平均{y_te.mean():.2f}s std {y_te.std():.2f}s')

def ev(cols,nm,base=None):
    m=xgb.XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.04,subsample=0.8,
                       colsample_bytree=0.8,random_state=42,verbosity=0)
    m.fit(tr[cols].fillna(-1),y_tr)
    p=m.predict(te[cols].fillna(-1))
    rmse=float(np.sqrt(((p-y_te)**2).mean()))
    r2=1-((p-y_te)**2).sum()/((y_te-y_tr.mean())**2).sum()
    ex='' if base is None else f'   ΔRMSE {rmse-base:+.4f}s'
    print(f'   {nm:38s} RMSE {rmse:.4f}s  R² {r2:.4f}{ex}')
    return rmse,m,cols

COND=['dist','surface_num','cond','n','cls']
print('\n■ ① 条件だけ（展開情報ゼロ）')
b,_,_=ev(COND,'距離・表面・馬場・頭数・クラス')
print('\n■ ② そこに展開情報を足す')
OLD=['escape_count','front_count','front_density','escape_avg_pos','escape_avg_pop']
NEW=['P_max','P_2nd','P_3rd','P_ent','P_hhi','P_gap','P_top2','n_can_lead','n_front',
     'pos_mean','pos_min','pos_std','ag_mean','ag_min','pop_of_leader']
ev(COND+OLD,'+ 既存の脚質カウント（推論時版）',b)
ev(COND+NEW,'+ 作り直した隊列争い（連続量）',b)
ev(COND+OLD+NEW,'+ 両方',b)
print('\n■ ③ 反則: そのレースの実際の脚質を知っていたら（天井）')
T=pd.read_pickle(f'{B}/tenkai/T.pkl')
d2=d.drop(columns=OLD).merge(T[['race_id']+OLD],on='race_id',how='left')
tr2,te2=d2[d2.date<CUT],d2[d2.date>=CUT]
m=xgb.XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.04,subsample=0.8,
                   colsample_bytree=0.8,random_state=42,verbosity=0)
m.fit(tr2[COND+OLD].fillna(-1),tr2.first_3f); p=m.predict(te2[COND+OLD].fillna(-1))
print(f'   {"+ 実際の脚質（事後・到達不能）":38s} RMSE '
      f'{np.sqrt(((p-te2.first_3f.values)**2).mean()):.4f}s'
      f'   ΔRMSE {np.sqrt(((p-te2.first_3f.values)**2).mean())-b:+.4f}s')
