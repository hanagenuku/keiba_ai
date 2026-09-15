"""本体モデル（3着内）にも競馬場が入っていない。足すとどうなるか。

⚠ ペースモデルと条件が違う: 本体は**残差学習**で base_margin（市場人気）が土台。
   市場は当然「中山ダート1800だ」を知っているので、会場情報の多くは
   既にアンカーに入っている可能性がある。期待は抑えて測る。
"""
import json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
sys.path.insert(0, '/home/user/keiba_ai')
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)
from src.features.race_shape import VENUES

SP = sys.argv[1]; WS = f'{SP}/ws'; REPO = '/home/user/keiba_ai'
PC = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
      '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
d = pd.read_csv(f'{WS}/data/horse_features.csv')
d = d[d.place.between(1,98)].copy(); d['date']=d['date'].astype(str)
d['rc'] = d.race_id.str.split('_').str[1].map(PC)
print('会場を race_id から復元:', d.rc.notna().mean()*100, '%')
for v in VENUES: d[f'venue_{v}'] = (d.rc==v).astype(float)
OH = [f'venue_{v}' for v in VENUES]
cols = json.load(open(f'{REPO}/data/xgb_feature_cols.json'))['feature_cols']

TE, VS = '2026-06-30', '2026-07-01'
tr, te = d[d.date<=TE].copy(), d[d.date>=VS].copy()
def bm_of(df, seed):
    n = df.groupby('race_id')['horse_num'].transform('count')
    p = df['f_popularity'].fillna(n/2)
    dr = load_popularity_drift(WS); assert dr is not None
    p = pd.Series(_apply_popularity_drift(p, df['race_id'].values, dr, seed=seed))
    return np.asarray(_popularity_to_base_margin(p, n))
bm_tr, bm_te = bm_of(tr,11), bm_of(te,12)
y_tr, y_te = tr.is_fukusho.values, te.is_fukusho.values
spw = round((1-y_tr.mean())/max(y_tr.mean(),0.01),2)
print(f'学習 {len(tr):,}頭 / 検定 {len(te):,}頭 ({te.race_id.nunique():,}R)\n')

def run(c, seed):
    dtr = xgb.DMatrix(tr[c].astype(float), label=y_tr, feature_names=list(c)); dtr.set_base_margin(bm_tr)
    dte = xgb.DMatrix(te[c].astype(float), label=y_te, feature_names=list(c)); dte.set_base_margin(bm_te)
    m = xgb.train({'objective':'binary:logistic','eval_metric':'logloss','max_depth':6,
                   'eta':0.05,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':10,
                   'alpha':0.1,'lambda':1.0,'scale_pos_weight':spw,'seed':seed,'verbosity':0},
                  dtr, num_boost_round=500, evals=[(dte,'v')],
                  early_stopping_rounds=50, verbose_eval=False)
    p = 1/(1+np.exp(-m.predict(dte, output_margin=True, iteration_range=(0,m.best_iteration+1))))
    return roc_auc_score(y_te,p), brier_score_loss(y_te,p), log_loss(y_te,np.clip(p,1e-7,1-1e-7))

for tag, c in [('A 現行134列', cols), ('B +競馬場one-hot(144列)', cols+OH)]:
    rs = [run(c,s) for s in (42,7,2026)]
    a = np.mean([r[0] for r in rs]); b=np.mean([r[1] for r in rs]); l=np.mean([r[2] for r in rs])
    print(f'{tag:26s} AUC {a:.4f} (幅 {max(r[0] for r in rs)-min(r[0] for r in rs):.4f})'
          f'  Brier {b:.4f}  LogLoss {l:.4f}')
    if tag.startswith('A'): base=(a,b,l)
    else: print(f'\n  B − A : AUC {a-base[0]:+.4f}  Brier {b-base[1]:+.4f}  LogLoss {l-base[2]:+.4f}')
