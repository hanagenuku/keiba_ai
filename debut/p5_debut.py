"""F-3: 新馬で現行 Layer 1 が市場に −0.0851 負ける原因の分解。
⚠ 新馬専用モデルを作るのが目的ではない。原因の特定のみ。
学習は全レース（本番と同じ）、評価は新馬行のみ。窓・手続きは mf1/M-1 と同一。"""
import os, sys, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
BASE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,BASE)

CID82=dict(max_depth=5,learning_rate=0.02,min_child_weight=300,subsample=0.8,
           colsample_bytree=0.8,reg_lambda=5.0,reg_alpha=0.1)
N_EST,ES=4000,50
D=pickle.load(open(f'{BASE}/p4_data.pkl','rb')); FEAT=D['feat']; META=D['meta']
IX={c:k for k,c in enumerate(FEAT)}
WINS=['T2','T3','W1','W2']          # 新馬行がある窓のみ（T1/W3 は0行）
CONF={'W1','W2'}
SEEDS=[42,7,2026]

JK=['f_jockey','f_jockey_rate','cl_f_jockey_rank','cl_f_jockey_vs_field']
TR=['f_trainer','cl_f_trainer_rank','cl_f_trainer_vs_field']
AVAIL=JK+TR+['f_post','f_sex','f_age','f_weight_load','cl_f_weight_load_rank',
             'cl_f_weight_load_vs_field','f_blood','cl_f_blood_rank','cl_f_blood_vs_field',
             'f_track_cond','f_dirt_turf_start','f_course_hill_diff','f_course_corner_tight',
             'f_course_stamina_demand','f_course_speed_demand']

# レース条件の生値（131列に距離・頭数そのものが無いため補う。予測時点で既知）
meta = META.copy()
meta['field'] = meta.groupby('race_id')['horse_num'].transform('count')
EXTRA = np.column_stack([
    meta['distance'].fillna(0).to_numpy(np.float32),
    meta['field'].to_numpy(np.float32),
    (meta['surface'].fillna('')=='ダート').to_numpy(np.float32),
    pd.factorize(meta['racecourse'].fillna(''))[0].astype(np.float32),
])
EXTRA_NAMES=['x_distance','x_field','x_is_dirt','x_venue']


def debut_mask(w):
    idx=D['windows'][w]['va_idx']
    return META.iloc[idx].race_class.fillna('').str.contains('新馬').values


def noise_rank_cols():
    """新馬行で馬番順のコピーになっている列（元の値が定数の相対順位列）。"""
    Xs,hn=[],[]
    for w in WINS:
        d=debut_mask(w); idx=D['windows'][w]['va_idx']
        Xs.append(D['windows'][w]['Xv'][d]); hn.append(META.iloc[idx].loc[d,'horse_num'].values)
    X=np.vstack(Xs); hn=np.concatenate(hn)
    out=[]
    for c,k in IX.items():
        if X[:,k].std()<1e-9: continue
        r=abs(np.corrcoef(X[:,k],hn)[0,1])
        if r>0.9: out.append(c)
    return out

NOISE=noise_rank_cols()

ARMS={
 'A 騎手のみ': JK,
 'B 調教師のみ': TR,
 'C 騎手+調教師': JK+TR,
 'E 新馬で使える情報': AVAIL+EXTRA_NAMES,
 'D 現行131列': list(FEAT),
 "D' 131列−馬番ノイズ": [c for c in FEAT if c not in NOISE],
}


def cols_idx(cols):
    base=[IX[c] for c in cols if c in IX]
    ext=[EXTRA_NAMES.index(c) for c in cols if c in EXTRA_NAMES]
    return base, ext


def mat(w, key, cols):
    base,ext=cols_idx(cols)
    Xw=D['windows'][w][key]
    if key=='Xf': rows=None
    parts=[Xw[:,base]] if base else []
    if ext:
        # 行インデックスを復元して EXTRA を貼る
        idx = {'Xf':None,'Xi':None,'Xv':D['windows'][w]['va_idx']}[key]
        if idx is None:
            raise RuntimeError('EXTRA は評価行以外では使わない設計')
        parts.append(EXTRA[np.ix_(idx, ext)])
    return np.hstack(parts) if len(parts)>1 else parts[0]


def build(w, cols):
    """学習/内側HO/評価 の行列を作る。EXTRA は全行に貼れるよう別途индексを持つ。"""
    base,ext=cols_idx(cols)
    win=D['windows'][w]
    # 全行インデックスを再構成（prep と同じ順序で mask を作り直す）
    # → EXTRA は META と同じ行順なので、va_idx で引ける。学習/内側HOは元インデックスが必要。
    return base, ext, win


def job(a):
    name, cols, w, seed = a
    win=D['windows'][w]
    base,ext=cols_idx(cols)
    # 学習・内側HO の元インデックスを復元
    n=len(META); mask_v=np.zeros(n,bool); mask_v[win['va_idx']]=True
    def take(key, n_rows):
        X=win[key][:,base] if base else np.zeros((n_rows,0),np.float32)
        return X
    Xf=take('Xf',len(win['y3f'])); Xi=take('Xi',len(win['y3i'])); Xv=take('Xv',len(win['y3v']))
    if ext:
        # 学習/内側HOの元行を特定するため date ベースで再構成
        fi=win['fit_idx']; ii=win['in_idx']
        Xf=np.hstack([Xf, EXTRA[np.ix_(fi,ext)]])
        Xi=np.hstack([Xi, EXTRA[np.ix_(ii,ext)]])
        Xv=np.hstack([Xv, EXTRA[np.ix_(win['va_idx'],ext)]])
    m=XGBClassifier(n_estimators=N_EST,**CID82,scale_pos_weight=1.0,eval_metric='logloss',
                    early_stopping_rounds=ES,random_state=seed,n_jobs=1,verbosity=0,tree_method='hist')
    m.fit(Xf,win['y3f'],eval_set=[(Xi,win['y3i'])],verbose=False)
    pv=m.predict_proba(Xv)[:,1]; pi=m.predict_proba(Xi)[:,1]
    return name,w,seed,pv,pi,int(m.best_iteration)

if __name__=='__main__':
    print(f'馬番順のコピーになっている列: {len(NOISE)} 個')
    print('  ' + ', '.join(NOISE))

def ece(y,p,bins=10):
    q=np.quantile(p,np.linspace(0,1,bins+1)); q[0],q[-1]=-np.inf,np.inf
    idx=np.digitize(p,q[1:-1]); y=np.asarray(y); p=np.asarray(p)
    return sum((idx==b).sum()/len(p)*abs(p[idx==b].mean()-y[idx==b].mean())
               for b in range(bins) if (idx==b).sum())

def run_all():
    tasks=[(n,c,w,s) for n,c in ARMS.items() for w in WINS for s in SEEDS]
    with Pool(4) as pool:
        res=pool.map(job,tasks)
    rows=[]
    for w in WINS:
        d=debut_mask(w); win=D['windows'][w]
        idxv=win['va_idx']; m=META.iloc[idxv]
        y3=win['y3v'][d]; y1=win['y1v'][d]; pop=win['popv'][d]
        ok=np.isfinite(pop)&(pop>0)                      # 市場と母集団を揃える
        di=debut_mask_inner(w)
        for n in ARMS:
            rs=[r for r in res if r[0]==n and r[1]==w]
            pv=np.mean([r[3][d] for r in rs],axis=0)
            pi=np.mean([r[4] for r in rs],axis=0)
            iso=IsotonicRegression(out_of_bounds='clip').fit(pi[di],win['y3i'][di]) if di.sum()>=150 \
                else IsotonicRegression(out_of_bounds='clip').fit(pi,win['y3i'])
            cv=np.clip(iso.predict(pv),1e-6,1-1e-6)
            rows.append(dict(arm=n,win=w,n=int(ok.sum()),
                             auc3=roc_auc_score(y3[ok],pv[ok]),auc1=roc_auc_score(y1[ok],pv[ok]),
                             ll=log_loss(y3[ok],cv[ok]),ece=ece(y3[ok],cv[ok]),
                             corr_hn=abs(np.corrcoef(pv[ok],m.loc[d,'horse_num'].values[ok])[0,1]),
                             trees=np.mean([r[5] for r in rs])))
        # 市場
        iso_m=IsotonicRegression(out_of_bounds='clip').fit(-win['popi'][np.isfinite(win['popi'])&(win['popi']>0)],
                                                          win['y3i'][np.isfinite(win['popi'])&(win['popi']>0)])
        cm=np.clip(iso_m.predict(-pop[ok]),1e-6,1-1e-6)
        rows.append(dict(arm='M 市場(確定人気)',win=w,n=int(ok.sum()),
                         auc3=roc_auc_score(y3[ok],-pop[ok]),auc1=roc_auc_score(y1[ok],-pop[ok]),
                         ll=log_loss(y3[ok],cm),ece=ece(y3[ok],cm),corr_hn=np.nan,trees=0))
    return pd.DataFrame(rows)

def debut_mask_inner(w):
    ii=D['windows'][w]['in_idx']
    return META.iloc[ii].race_class.fillna('').str.contains('新馬').values
