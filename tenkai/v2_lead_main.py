"""段1の P(先頭に立つ)（OOS AUC 0.7805）を本体モデル（3着内）に入れるとどうなるか。

これは「ペースが誰に効くか」に一番近い未測定項目。
現行モデルが持つ脚質は過去走からの4分類推定（実際との一致率39.8%）で、
P(先頭) はそれより明確に良い量。展開系29列とは別物として測る。

⚠ 事前の見立て: 2026-08-27 のオラクル測定で「完璧なペース予測でも +0.0007」
   「展開系29列を全部抜いても -0.0001」なので、動かないと予想する。
"""
import json, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'/home/user/keiba_ai')
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)
from src.tools.train_race_shape import _load, _horse_features, LEAD_COLS
from src.features.race_shape import race_shape_features, RACE_FEATURE_COLS

SP=sys.argv[1]; DB=sys.argv[2]; WS=f'{SP}/retrain/ws'; REPO='/home/user/keiba_ai'
TE,VS='2026-06-30','2026-07-01'

H,_ = _load(DB); H=_horse_features(H)
tr_h=H[H.date<=TE]; med=tr_h[LEAD_COLS].median(); X=tr_h[LEAD_COLS].fillna(med)
mu,sd=X.mean(),X.std().replace(0,1)
lm=LogisticRegression(max_iter=3000).fit((X-mu)/sd, tr_h.led)
H['P']=lm.predict_proba((H[LEAD_COLS].fillna(med)-mu)/sd)[:,1]
rows=[]
for rid,s in H.groupby('race_id'):
    f=race_shape_features([{'p_led':r.p_led,'p_mean':r.p_mean,'p_min':r.p_min,'p_last':r.p_last,
                            'p_std':r.p_std,'p_ag':r.p_ag,'p_n':r.p_n} for r in s.itertuples()],
                          list(s.P), list(s['pop']))
    f['race_id']=rid; rows.append(f)
F=pd.DataFrame(rows)
HP=H[['race_id','horse_num','P','p_led','p_mean','p_min']].rename(columns={'P':'f_p_lead'})

d=pd.read_csv(f'{WS}/data/horse_features.csv'); d=d[d.place.between(1,98)].copy()
d['date']=d['date'].astype(str)
d=d.merge(HP,on=['race_id','horse_num'],how='left').merge(
    F.rename(columns={c:f'rs_{c}' for c in RACE_FEATURE_COLS}),on='race_id',how='left')
NEW_H=['f_p_lead','p_led','p_mean','p_min']
NEW_R=[f'rs_{c}' for c in RACE_FEATURE_COLS if c!='n_horses']
print(f'P(先頭) 結合率 {d.f_p_lead.notna().mean()*100:.1f}%  隊列特徴 {d["rs_P_gap"].notna().mean()*100:.1f}%')

cols=json.load(open(f'{REPO}/data/xgb_feature_cols.json'))['feature_cols']
tr,te=d[d.date<=TE].copy(), d[d.date>=VS].copy()
def bm(df,seed):
    n=df.groupby('race_id')['horse_num'].transform('count'); p=df['f_popularity'].fillna(n/2)
    dr=load_popularity_drift(WS); p=pd.Series(_apply_popularity_drift(p,df['race_id'].values,dr,seed=seed))
    return np.asarray(_popularity_to_base_margin(p,n))
bt,bv=bm(tr,11),bm(te,12); ytr,yte=tr.is_fukusho.values,te.is_fukusho.values
spw=round((1-ytr.mean())/max(ytr.mean(),.01),2)
print(f'学習 {len(tr):,}頭 / 検定 {len(te):,}頭\n')
def run(c,seed):
    a=xgb.DMatrix(tr[c].astype(float),label=ytr,feature_names=list(c)); a.set_base_margin(bt)
    b=xgb.DMatrix(te[c].astype(float),label=yte,feature_names=list(c)); b.set_base_margin(bv)
    m=xgb.train({'objective':'binary:logistic','eval_metric':'logloss','max_depth':6,'eta':0.05,
                 'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':10,'alpha':0.1,
                 'lambda':1.0,'scale_pos_weight':spw,'seed':seed,'verbosity':0},
                a,num_boost_round=500,evals=[(b,'v')],early_stopping_rounds=50,verbose_eval=False)
    p=1/(1+np.exp(-m.predict(b,output_margin=True,iteration_range=(0,m.best_iteration+1))))
    return roc_auc_score(yte,p),brier_score_loss(yte,p),log_loss(yte,np.clip(p,1e-7,1-1e-7))
base=None
for tag,c in [('A 現行134列',cols),('B +P(先頭)など馬4列',cols+NEW_H),
              ('C +隊列15列',cols+NEW_R),('D +両方',cols+NEW_H+NEW_R)]:
    rs=[run(c,s) for s in (42,7,2026)]
    a,br,l=(np.mean([r[i] for r in rs]) for i in range(3))
    delta='' if base is None else f'   A比 AUC {a-base[0]:+.4f} Brier {br-base[1]:+.4f} LogLoss {l-base[2]:+.4f}'
    print(f'{c.__len__():3d}列 {tag:22s} AUC {a:.4f} (幅 {max(r[0] for r in rs)-min(r[0] for r in rs):.4f}){delta}')
    if base is None: base=(a,br,l)
