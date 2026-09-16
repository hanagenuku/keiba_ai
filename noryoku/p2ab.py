# -*- coding: utf-8 -*-
"""Phase 2 の A/B（P2-A-4 圧縮 / P2-B-2 絶対値 vs 相対値）。

本番同型: residual・base_margin にドリフト注入ON・3窓 × 3シード。
全アームで**同じ行・同じ行順**を使う（行順が変わると base_margin の乱数列が変わる）。

usage: python3 noryoku/p2ab.py <scratch_base_dir>
"""
import io, json, os, re, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

REPO = '/home/user/keiba_ai'
sys.path.insert(0, REPO)
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)

WS = sys.argv[1]
COLS = json.load(io.open(f'{REPO}/data/xgb_feature_cols.json', encoding='utf-8'))['feature_cols']

G8 = ['f_last1_rank', 'f_pl_rating', 'f_recent',
      'rl_f_pl_rating', 'rl_f_pl_rating_rank', 'rl_f_pl_rating_z',
      'rl_f_recent_rank', 'rl_f_recent_vs_field']
PL4 = ['f_pl_rating', 'rl_f_pl_rating', 'rl_f_pl_rating_z', 'rl_f_pl_rating_rank']
REC4 = ['f_recent', 'rl_f_recent_vs_field', 'rl_f_recent_rank', 'f_last1_rank']

PAIRS = []
for c in COLS:
    m = re.match(r'^(?:cl|rl)_(.+)_vs_field$', c)
    if m and m.group(1) in COLS:
        PAIRS.append((m.group(1), c))
ABSC = [a for a, _ in PAIRS]
RELC = [b for _, b in PAIRS]

WINDOWS = [('W1', '2026-07-04', '2026-08-16'),
           ('W2', '2026-05-02', '2026-06-28'),
           ('W3', '2026-03-01', '2026-04-26')]
SEEDS = (42, 7, 2026)

d = pd.read_csv(f'{WS}/data/horse_features.csv', low_memory=False)
d = d[d.place.between(1, 98)].copy()
d['date'] = d['date'].astype(str)
DRIFT = load_popularity_drift(WS)
assert DRIFT is not None, 'drift 分布が作れない → 中止（本番と条件が違う比較になる）'
print('rows %d  drift keys %d' % (len(d), len(DRIFT)), flush=True)


def pc1(train_block, all_block):
    """train で主成分を決め、全体に適用（標準化込み）。"""
    mu, sd = train_block.mean(0), train_block.std(0)
    sd[sd < 1e-12] = 1.0
    Zt = (train_block - mu) / sd
    C = np.cov(Zt, rowvar=False)
    w, v = np.linalg.eigh(C)
    u = v[:, int(np.argmax(w))]
    if u.sum() < 0:
        u = -u
    return ((all_block - mu) / sd) @ u


def build(tr_idx, te_idx):
    """アームごとの (列名, DataFrame) を作る。"""
    full = d
    arms = {}
    base = full[COLS].astype(float)
    arms['P0 現行134列'] = base

    def comp(cols, name):
        blk = full[cols].astype(float).fillna(full[cols].astype(float).median()).values
        s = pc1(blk[tr_idx], blk)
        return pd.Series(s, index=full.index, name=name)

    rest8 = [c for c in COLS if c not in G8]
    arms['P1 8列→合成1列'] = pd.concat([base[rest8], comp(G8, 'z_g8')], axis=1)
    arms['P2 8列→絶対量3列'] = base[[c for c in COLS if c not in G8] +
                                 ['f_last1_rank', 'f_recent', 'f_pl_rating']]
    arms['P3 8列→PL/着順の2列'] = pd.concat(
        [base[rest8], comp(PL4, 'z_pl'), comp(REC4, 'z_rec')], axis=1)
    arms['B-abs 相対17列を落とす'] = base[[c for c in COLS if c not in RELC]]
    arms['B-rel 絶対17列を落とす'] = base[[c for c in COLS if c not in ABSC]]
    return arms


def run(X, y_tr, y_te, bm_tr, bm_te, tr_idx, te_idx, seed, spw):
    dtr = xgb.DMatrix(X.iloc[tr_idx], label=y_tr, feature_names=list(X.columns))
    dtr.set_base_margin(bm_tr)
    dte = xgb.DMatrix(X.iloc[te_idx], label=y_te, feature_names=list(X.columns))
    dte.set_base_margin(bm_te)
    m = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6,
                   'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
                   'min_child_weight': 10, 'alpha': 0.1, 'lambda': 1.0,
                   'scale_pos_weight': spw, 'seed': seed, 'verbosity': 0},
                  dtr, num_boost_round=500, evals=[(dte, 'v')],
                  early_stopping_rounds=50, verbose_eval=False)
    p = 1 / (1 + np.exp(-m.predict(dte, output_margin=True,
                                   iteration_range=(0, m.best_iteration + 1))))
    return (roc_auc_score(y_te, p), brier_score_loss(y_te, p),
            log_loss(y_te, np.clip(p, 1e-7, 1 - 1e-7)))


RES = {}
for wn, s0, s1 in WINDOWS:
    tr_mask = (d.date < s0).values
    te_mask = ((d.date >= s0) & (d.date <= s1)).values
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    y_tr, y_te = d.is_fukusho.values[tr_idx], d.is_fukusho.values[te_idx]
    spw = round((1 - y_tr.mean()) / max(y_tr.mean(), 0.01), 2)

    def bm_of(idx, seed):
        sub = d.iloc[idx]
        n = sub.groupby('race_id')['horse_num'].transform('count')
        p = sub['f_popularity'].fillna(n / 2)
        p = pd.Series(_apply_popularity_drift(p, sub['race_id'].values, DRIFT, seed=seed))
        return np.asarray(_popularity_to_base_margin(p, n))

    bm_tr, bm_te = bm_of(tr_idx, 11), bm_of(te_idx, 12)
    arms = build(tr_idx, te_idx)
    print('\n== %s  train %d / val %d (%dR) ==' %
          (wn, len(tr_idx), len(te_idx), d.iloc[te_idx].race_id.nunique()), flush=True)
    for name, X in arms.items():
        rs = [run(X, y_tr, y_te, bm_tr, bm_te, tr_idx, te_idx, s, spw) for s in SEEDS]
        a = float(np.mean([r[0] for r in rs]))
        sd = float(np.std([r[0] for r in rs]))
        b = float(np.mean([r[1] for r in rs]))
        l = float(np.mean([r[2] for r in rs]))
        RES.setdefault(name, {})[wn] = {'auc': a, 'auc_sd': sd, 'brier': b, 'logloss': l,
                                        'n_cols': int(X.shape[1])}
        print('  %-24s cols=%3d  AUC %.4f (sd %.4f)  Brier %.4f  LL %.4f'
              % (name, X.shape[1], a, sd, b, l), flush=True)

print('\n===== summary (P0 との差) =====', flush=True)
p0 = RES['P0 現行134列']
SUM = {}
for name, w in RES.items():
    da = [w[k]['auc'] - p0[k]['auc'] for k, _, _ in WINDOWS]
    db = [w[k]['brier'] - p0[k]['brier'] for k, _, _ in WINDOWS]
    dl = [w[k]['logloss'] - p0[k]['logloss'] for k, _, _ in WINDOWS]
    SUM[name] = {'dAUC': [round(x, 4) for x in da], 'mean_dAUC': round(float(np.mean(da)), 4),
                 'worse_windows': int(sum(1 for x in da if x < 0)),
                 'mean_dBrier': round(float(np.mean(db)), 4),
                 'mean_dLogLoss': round(float(np.mean(dl)), 4),
                 'seed_sd': round(float(np.mean([w[k]['auc_sd'] for k, _, _ in WINDOWS])), 4)}
    print('%-24s ΔAUC %s  平均 %+.4f  悪化窓 %d/3  シードsd %.4f'
          % (name, ' '.join('%+.4f' % x for x in da), np.mean(da),
             sum(1 for x in da if x < 0), SUM[name]['seed_sd']), flush=True)

json.dump({'raw': RES, 'summary': SUM},
          io.open(f'{REPO}/noryoku/p2ab_report.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\nwrote noryoku/p2ab_report.json', flush=True)
