"""Phase 4: cid82 で6窓(探索期2025×3 / 確認期2026×3)の out-of-sample 予測を作る。
学習コードは mf1/fit.py と同一の手続き（内側HOで early stopping と Isotonic 較正）。"""
import os, sys, pickle
from multiprocessing import Pool
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)

CID82 = dict(max_depth=5, learning_rate=0.02, min_child_weight=300, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1)
N_EST, ES = 4000, 50
WINS = ['T1', 'T2', 'T3', 'W1', 'W2', 'W3']
SEEDS = [42, 7, 2026]
D = pickle.load(open(f'{BASE}/p4_data.pkl', 'rb'))


def job(a):
    w, seed = a
    win = D['windows'][w]
    m = XGBClassifier(n_estimators=N_EST, **CID82, scale_pos_weight=1.0,
                      eval_metric='logloss', early_stopping_rounds=ES,
                      random_state=seed, n_jobs=1, verbosity=0, tree_method='hist')
    m.fit(win['Xf'], win['y3f'], eval_set=[(win['Xi'], win['y3i'])], verbose=False)
    pi = m.predict_proba(win['Xi'])[:, 1]
    pv = m.predict_proba(win['Xv'])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(pi, win['y3i'])
    cv = np.clip(iso.predict(pv), 1e-6, 1 - 1e-6)
    return w, seed, pv, cv, int(m.best_iteration)


if __name__ == '__main__':
    with Pool(4) as p:
        res = p.map(job, [(w, s) for w in WINS for s in SEEDS])
    out = {}
    for w in WINS:
        rs = [r for r in res if r[0] == w]
        out[w] = dict(prob=np.mean([r[2] for r in rs], axis=0),
                      cal=np.mean([r[3] for r in rs], axis=0),
                      trees=[r[4] for r in rs],
                      per_seed=np.stack([r[2] for r in rs]))
        print(f'{w}: 木 {out[w]["trees"]}  n={len(out[w]["prob"]):,}', flush=True)
    with open(f'{BASE}/p4_pred.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('-> p4_pred.pkl')
