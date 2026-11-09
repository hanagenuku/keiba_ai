"""合成データで「残差学習の early stopping が何で止まるか」だけを確かめる（実データ・実結果は使わない）。
本番 train_xgb(residual=True) と同じ形: base_margin=Zipf(人気) / scale_pos_weight=(1-r)/r /
eval_metric=logloss(重みなし) / early_stopping_rounds=50 / max_depth=6 lr=0.05 mcw=10
"""
import numpy as np, xgboost as xgb, sys
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
def make(n_races, hidden):   # hidden = 市場が見ていない能力成分の大きさ（特徴量には載っている）
    rows=[]
    for r in range(n_races):
        n=rng.integers(8,19)
        pub=rng.normal(0,1,n); priv=rng.normal(0,hidden,n)
        ability=pub+priv
        market=pub+rng.normal(0,0.3,n)
        pop=(-market).argsort().argsort()+1
        finish=(-(ability+rng.gumbel(0,1.0,n))).argsort().argsort()+1
        X=np.column_stack([priv+rng.normal(0,0.5*max(hidden,1e-9),n)]+[rng.normal(0,1,n) for _ in range(19)])
        for i in range(n): rows.append((r,n,pop[i],finish[i]<=3,*X[i]))
    a=np.array(rows,dtype=float); return a
def bm(pop,n): 
    p=np.clip((1/pop)/(np.log(n)+0.5772),1e-4,1-1e-4); return np.log(p/(1-p))
def top1_hit(race,score,y):
    o=np.lexsort((-score,race)); first=np.r_[True,race[o][1:]!=race[o][:-1]]; return y[o][first].mean()
def run(hidden,spw_mode):
    tr=make(9000,hidden); va=make(700,hidden); te=make(2500,hidden)
    def D(a): 
        d=xgb.DMatrix(a[:,4:],label=a[:,3]); d.set_base_margin(bm(a[:,2],a[:,1])); return d
    r=tr[:,3].mean(); spw=(1-r)/r if spw_mode=='prod' else 1.0
    params=dict(objective='binary:logistic',max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                min_child_weight=10,reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=spw,eval_metric='logloss',seed=42)
    b=xgb.train(params,D(tr),500,evals=[(D(va),'val')],early_stopping_rounds=50,verbose_eval=False)
    s=b.predict(D(te),output_margin=True)            # 本番と同じく全木で予測
    mk=bm(te[:,2],te[:,1])
    return b.best_iteration,b.num_boosted_rounds(),roc_auc_score(te[:,3],mk),roc_auc_score(te[:,3],s),top1_hit(te[:,0],mk,te[:,3]),top1_hit(te[:,0],s,te[:,3])
print('hidden  mode  best_it trees  AUC(市場) AUC(モデル)  軸3着内(市場) 軸3着内(モデル)')
for hidden in (0.0,0.3,0.6):
    for mode in ('prod','spw1'):
        bi,nt,am,as_,hm,hs=run(hidden,mode)
        print(f'{hidden:5.1f}  {mode:5s} {bi:6d} {nt:5d}   {am:.4f}   {as_:.4f}      {hm:.3f}        {hs:.3f}')
