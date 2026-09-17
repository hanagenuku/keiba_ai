# -*- coding: utf-8 -*-
"""f_last1_rank に PL+近走の上の固有情報があるか（非線形表現を含めた完全OOS比較）。

CRITERIA_PHASE2.md §7 に事前登録した設計をそのまま実行する。1回だけ。

usage: python3 noryoku/p2e.py <scratch_base_dir>
"""
import io, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = '/home/user/keiba_ai'
sys.path.insert(0, REPO)
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)

WS = sys.argv[1]
COLS = json.load(io.open(f'{REPO}/data/xgb_feature_cols.json', encoding='utf-8'))['feature_cols']
WINDOWS = [('W1', '2026-07-04', '2026-08-16'),
           ('W2', '2026-05-02', '2026-06-28'),
           ('W3', '2026-03-01', '2026-04-26')]
SEEDS = (42, 7, 2026)

d = pd.read_csv(f'{WS}/data/horse_features.csv', low_memory=False)
d = d[d.place.between(1, 98)].copy()
d['date'] = d['date'].astype(str)
d = d.reset_index(drop=True)
print('rows %d' % len(d), flush=True)

# ── last1 の非線形表現 ───────────────────────────────────────
# f_last1_rank は履歴が無い馬に既定値 8.0 を入れる（＝「8着」と「初出走」が同値）。
lr = d['f_last1_rank'].astype(float)
nohist = (d['f_career_runs'].fillna(0) <= 0).astype(float)
BIN = pd.DataFrame(index=d.index)
BIN['l1_no_hist'] = nohist
for lo, hi, nm in [(1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 4, '4'),
                   (5, 5, '5'), (6, 9, '6_9'), (10, 99, '10p')]:
    BIN['l1_bin_%s' % nm] = (((lr >= lo) & (lr <= hi)) & (nohist < 0.5)).astype(float)
BINC = list(BIN.columns)
d = pd.concat([d, BIN], axis=1)
print('bins:', BINC, flush=True)
print('履歴なし %.2f%% / 着順分布:' % (100 * nohist.mean()),
      {c: int(d[c].sum()) for c in BINC}, flush=True)

y = d.is_fukusho.values
OUT = {}


def run(cols, wn, s0, s1, seed, use_anchor, drift=None):
    tr = np.where((d.date < s0).values)[0]
    te = np.where(((d.date >= s0) & (d.date <= s1)).values)[0]
    y_tr, y_te = y[tr], y[te]
    spw = round((1 - y_tr.mean()) / max(y_tr.mean(), 0.01), 2)
    X = d[cols].astype(float)
    dtr = xgb.DMatrix(X.iloc[tr], label=y_tr, feature_names=list(cols))
    dte = xgb.DMatrix(X.iloc[te], label=y_te, feature_names=list(cols))
    if use_anchor:
        def bm(idx, sd):
            sub = d.iloc[idx]
            n = sub.groupby('race_id')['horse_num'].transform('count')
            p = sub['f_popularity'].fillna(n / 2)
            p = pd.Series(_apply_popularity_drift(p, sub['race_id'].values, drift, seed=sd))
            return np.asarray(_popularity_to_base_margin(p, n))
        dtr.set_base_margin(bm(tr, 11))
        dte.set_base_margin(bm(te, 12))
    m = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6,
                   'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
                   'min_child_weight': 10, 'alpha': 0.1, 'lambda': 1.0,
                   'scale_pos_weight': spw, 'seed': seed, 'verbosity': 0},
                  dtr, num_boost_round=500, evals=[(dte, 'v')],
                  early_stopping_rounds=50, verbose_eval=False)
    p = 1 / (1 + np.exp(-m.predict(dte, output_margin=True,
                                   iteration_range=(0, m.best_iteration + 1))))
    return roc_auc_score(y_te, p)


def panel(name, arms, use_anchor, drift=None):
    print('\n===== %s =====' % name, flush=True)
    res = {}
    for tag, cols in arms:
        per = []
        for wn, s0, s1 in WINDOWS:
            a = float(np.mean([run(cols, wn, s0, s1, s, use_anchor, drift) for s in SEEDS]))
            per.append(a)
        res[tag] = per
        print('  %-34s cols=%3d  %s  平均 %.4f'
              % (tag, len(cols), ' '.join('%.4f' % v for v in per), np.mean(per)), flush=True)
    return res


PL, RE, L1 = 'f_pl_rating', 'f_recent', 'f_last1_rank'
A = panel('パネルA 能力だけ（市場アンカーなし）', [
    ('A0 last1 単独（感度検査）', [L1]),
    ('A1 PL + 近走', [PL, RE]),
    ('A2 + last1（生の列）', [PL, RE, L1]),
    ('A3 + last1（非線形表現）', [PL, RE] + BINC),
], use_anchor=False)
OUT['panelA'] = A

a0 = float(np.mean(A['A0 last1 単独（感度検査）']))
print('\n■ 感度検査: last1 単独 AUC %.4f （基準 > 0.60）' % a0, flush=True)
if not (a0 > 0.60):
    print('!! 感度検査に通らない。計測器が last1 の生の信号を拾えていないので中止する。')
    json.dump(OUT, io.open(f'{REPO}/noryoku/p2e_report.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    sys.exit(1)

DRIFT = load_popularity_drift(WS)
assert DRIFT is not None, 'drift が作れない → 本番同型にならないので中止'
B = panel('パネルB 本番同型（市場アンカーあり）', [
    ('B1 現行134列', COLS),
    ('B2 last1 を抜く', [c for c in COLS if c != L1]),
    ('B3 last1 をビン表現に差し替え', [c for c in COLS if c != L1] + BINC),
], use_anchor=True, drift=DRIFT)
OUT['panelB'] = B

print('\n===== 判定 =====', flush=True)


def diff(res, hi, lo, label, thr=0.002):
    dv = [res[hi][i] - res[lo][i] for i in range(3)]
    m = float(np.mean(dv))
    print('  %-28s %s  平均 %+.4f   基準 %s → %s'
          % (label, ' '.join('%+.4f' % v for v in dv), m,
             ('|Δ|<%.3f' % thr) if 'B2' in hi else ('<+%.3f' % thr),
             ('✅' if (abs(m) < thr if 'B2' in hi else m < thr) else '❌')), flush=True)
    return m


OUT['A2_minus_A1'] = diff(A, 'A2 + last1（生の列）', 'A1 PL + 近走', 'A2 − A1')
OUT['A3_minus_A2'] = diff(A, 'A3 + last1（非線形表現）', 'A2 + last1（生の列）', 'A3 − A2')
OUT['A3_minus_A1'] = diff(A, 'A3 + last1（非線形表現）', 'A1 PL + 近走', 'A3 − A1（参考）')
OUT['B2_minus_B1'] = diff(B, 'B2 last1 を抜く', 'B1 現行134列', 'B2 − B1')
OUT['B3_minus_B1'] = diff(B, 'B3 last1 をビン表現に差し替え', 'B1 現行134列', 'B3 − B1')

json.dump(OUT, io.open(f'{REPO}/noryoku/p2e_report.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\nwrote noryoku/p2e_report.json', flush=True)
