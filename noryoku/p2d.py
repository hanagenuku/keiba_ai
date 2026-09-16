# -*- coding: utf-8 -*-
"""Phase 2-C 補強: 「独立した情報の塊」は**目的変数に対して**何個必要か。

並行分析は n=164,915 では帰無の95%点が 1.05 まで下がり、実質 Kaiser 基準に
degenerate している（＝弱い因子を止める力が無い）。そこで分散ではなく
**3着内を当てる力**で軸の本数を測り直す。

因子は 2026-03-01 より前（全検定窓より前）だけで推定し、全体に射影する。

usage: python3 noryoku/p2d.py <scratch_base_dir>
"""
import io, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = '/home/user/keiba_ai'
sys.path.insert(0, REPO)
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)

WS = sys.argv[1]
KK = 18                       # 事前登録した「累積70%」の本数
COLS = json.load(io.open(f'{REPO}/data/xgb_feature_cols.json', encoding='utf-8'))['feature_cols']
DROP = ['f_early_speed', 'f_same_course_rate', 'f_course_stamina_demand', 'f_is_debut']
LIVE = [c for c in COLS if c not in DROP]

d = pd.read_csv(f'{WS}/data/horse_features.csv', low_memory=False)
d = d[d.place.between(1, 98)].copy()
d['date'] = d['date'].astype(str)
d = d.reset_index(drop=True)
X = d[LIVE].astype(float)
X = X.fillna(X.median())
rid = d['race_id'].values
Xc = X - X.groupby(rid).transform('mean')
keep = [c for c in LIVE if Xc[c].std() > 1e-9]
A = Xc[keep].values
fit = (d.date < '2026-03-01').values
mu, sd = A[fit].mean(0), A[fit].std(0)
sd[sd < 1e-12] = 1.0
Z = (A - mu) / sd


def varimax(Phi, q=60, tol=1e-6):
    p, k = Phi.shape
    R, dd = np.eye(k), 0
    for _ in range(q):
        old = dd
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - (1.0 / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh
        dd = np.sum(s)
        if old != 0 and dd / old < 1 + tol:
            break
    return Phi @ R


C = np.cov(Z[fit], rowvar=False)
w, v = np.linalg.eigh(C)
idx = np.argsort(w)[::-1][:KK]
L = varimax(v[:, idx] * np.sqrt(np.maximum(w[idx], 0)))
F = Z @ L @ np.linalg.pinv(L.T @ L)          # 因子スコア
y = d.is_fukusho.values

# 因子を「単独の当てる力」で並べる（学習期だけで決める）
order = sorted(range(KK), key=lambda j: -abs(roc_auc_score(y[fit], F[fit, j]) - 0.5))
uni = {('F%d' % (j + 1)): round(float(abs(roc_auc_score(y[fit], F[fit, j]) - 0.5) + 0.5), 4)
       for j in order}
print('因子の単独AUC（学習期・降順）:', flush=True)
for j in order:
    top = [keep[i] for i in np.argsort(-np.abs(L[:, j]))[:3]]
    print('  F%-3d AUC %.4f  %s' % (j + 1, uni['F%d' % (j + 1)], ' / '.join(top)), flush=True)

DRIFT = load_popularity_drift(WS)
assert DRIFT is not None
WINDOWS = [('W1', '2026-07-04', '2026-08-16'),
           ('W2', '2026-05-02', '2026-06-28'),
           ('W3', '2026-03-01', '2026-04-26')]
MS = [1, 2, 3, 4, 5, 6, 8, 10, 14, 18]
RES = {}
for wn, s0, s1 in WINDOWS:
    tr = np.where((d.date < s0).values)[0]
    te = np.where(((d.date >= s0) & (d.date <= s1)).values)[0]
    y_tr, y_te = y[tr], y[te]
    spw = round((1 - y_tr.mean()) / max(y_tr.mean(), 0.01), 2)

    def bm(idx, seed):
        sub = d.iloc[idx]
        n = sub.groupby('race_id')['horse_num'].transform('count')
        p = sub['f_popularity'].fillna(n / 2)
        p = pd.Series(_apply_popularity_drift(p, sub['race_id'].values, DRIFT, seed=seed))
        return np.asarray(_popularity_to_base_margin(p, n))

    bm_tr, bm_te = bm(tr, 11), bm(te, 12)
    print('\n== %s ==' % wn, flush=True)
    for m in MS:
        cols = order[:m]
        aucs = []
        for seed in (42, 7):
            dtr = xgb.DMatrix(F[np.ix_(tr, cols)], label=y_tr)
            dtr.set_base_margin(bm_tr)
            dte = xgb.DMatrix(F[np.ix_(te, cols)], label=y_te)
            dte.set_base_margin(bm_te)
            mo = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'logloss',
                            'max_depth': 6, 'eta': 0.05, 'subsample': 0.8,
                            'colsample_bytree': 0.8, 'min_child_weight': 10, 'alpha': 0.1,
                            'lambda': 1.0, 'scale_pos_weight': spw, 'seed': seed,
                            'verbosity': 0}, dtr, num_boost_round=500,
                           evals=[(dte, 'v')], early_stopping_rounds=50, verbose_eval=False)
            p = 1 / (1 + np.exp(-mo.predict(dte, output_margin=True,
                                            iteration_range=(0, mo.best_iteration + 1))))
            aucs.append(roc_auc_score(y_te, p))
        RES.setdefault(wn, {})[m] = round(float(np.mean(aucs)), 4)
        print('  因子 %2d本  AUC %.4f' % (m, np.mean(aucs)), flush=True)

print('\n本数別の平均AUC:', flush=True)
for m in MS:
    print('  %2d本  %.4f' % (m, np.mean([RES[w][m] for w in RES])), flush=True)
json.dump({'uni_auc': uni, 'order': [int(j) + 1 for j in order], 'curve': RES},
          io.open(f'{REPO}/noryoku/p2d_report.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('wrote noryoku/p2d_report.json', flush=True)
