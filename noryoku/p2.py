# -*- coding: utf-8 -*-
"""Phase 2: 能力軸の分解（P2-A / P2-B / P2-C）。

CRITERIA_PHASE2.md に書いた設計をそのまま実行する。
本番のモデル・DB・買い目には一切触れない（scratch の CSV を読むだけ）。

usage: python3 noryoku/p2.py <scratch_base_dir>
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = sys.argv[1]
OUT = {}

COLS = json.load(io.open(os.path.join(REPO, 'data', 'xgb_feature_cols.json'), encoding='utf-8'))
FEATS = COLS['feature_cols'] if isinstance(COLS, dict) else COLS

DEAD = ['f_early_speed']                       # 定数36.0
DUP = ['f_same_course_rate', 'f_course_stamina_demand', 'f_is_debut']  # 別列と数学的に同一
LIVE = [c for c in FEATS if c not in DEAD + DUP]

G8 = ['f_last1_rank', 'f_pl_rating', 'f_recent',
      'rl_f_pl_rating', 'rl_f_pl_rating_rank', 'rl_f_pl_rating_z',
      'rl_f_recent_rank', 'rl_f_recent_vs_field']

print('read csv...', flush=True)
df = pd.read_csv(os.path.join(BASE, 'data', 'horse_features.csv'), low_memory=False)
df = df[df['place'].notna() & (df['place'] > 0) & (df['place'] < 99)].reset_index(drop=True)
print('rows', len(df), 'races', df['race_id'].nunique(), flush=True)
OUT['n_rows'] = int(len(df))
OUT['n_races'] = int(df['race_id'].nunique())

y = df['is_fukusho'].astype(float).values
rid = df['race_id'].values


def center(s):
    """レース内中心化。"""
    return s - s.groupby(df['race_id']).transform('mean')


X = df[LIVE].astype(float)
med = X.median()
Xf = X.fillna(med)
print('nan rate mean %.4f' % float(X.isna().mean().mean()), flush=True)
OUT['nan_rate_mean'] = float(X.isna().mean().mean())

print('centering...', flush=True)
grp = Xf.groupby(rid)
Xc = Xf - grp.transform('mean')

# ────────────────────────────────────────────────────────────
# P2-A-2  レース内中心化後の相関
# ────────────────────────────────────────────────────────────
print('\n=== A-2 within-race corr (8 cols) ===', flush=True)
c8 = Xc[G8].corr()
OUT['A2_corr_centered'] = {a: {b: round(float(c8.loc[a, b]), 4) for b in G8} for a in G8}
c8raw = Xf[G8].corr()
OUT['A2_corr_raw'] = {a: {b: round(float(c8raw.loc[a, b]), 4) for b in G8} for a in G8}
print('      ' + ' '.join('%8s' % c[-8:] for c in G8))
for a in G8:
    print('%-22s' % a + ' '.join('%8.3f' % c8.loc[a, b] for b in G8))

# ────────────────────────────────────────────────────────────
# P2-A-3  固有情報（他7列で回帰した残差）
# ────────────────────────────────────────────────────────────
print('\n=== A-3 unique information ===', flush=True)
A3 = {}
for col in G8:
    others = [c for c in G8 if c != col]
    A = np.column_stack([Xc[others].values, np.ones(len(Xc))])
    b = Xc[col].values
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    sst = float(np.var(b))
    r2 = 1.0 - float(np.var(resid)) / sst if sst > 0 else 0.0
    # 残差もレース内中心化してから AUC（水準差を持ち込まない）
    rs = pd.Series(resid, index=df.index)
    rc = rs - rs.groupby(rid).transform('mean')
    try:
        auc = float(roc_auc_score(y, rc.values))
    except Exception:
        auc = float('nan')
    auc = max(auc, 1 - auc)
    A3[col] = {'r2_by_others': round(r2, 4), 'resid_share': round(1 - r2, 4),
               'resid_auc': round(auc, 4)}
    print('%-24s R2=%.4f  resid=%.4f  resid_auc=%.4f' % (col, r2, 1 - r2, auc))
OUT['A3'] = A3

# 3つの絶対量だけでも同じことをする
ABS3 = ['f_last1_rank', 'f_recent', 'f_pl_rating']
A3b = {}
for col in ABS3:
    others = [c for c in ABS3 if c != col]
    A = np.column_stack([Xc[others].values, np.ones(len(Xc))])
    b = Xc[col].values
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    r2 = 1.0 - float(np.var(resid)) / float(np.var(b))
    rs = pd.Series(resid, index=df.index)
    rc = rs - rs.groupby(rid).transform('mean')
    auc = float(roc_auc_score(y, rc.values)); auc = max(auc, 1 - auc)
    A3b[col] = {'r2_by_others': round(r2, 4), 'resid_share': round(1 - r2, 4),
                'resid_auc': round(auc, 4)}
    print('[abs3] %-18s R2=%.4f  resid=%.4f  resid_auc=%.4f' % (col, r2, 1 - r2, auc))
OUT['A3_abs3'] = A3b

# 単独 AUC（レース内中心化した値）
solo = {}
for col in G8:
    v = Xc[col].values
    a = float(roc_auc_score(y, v)); solo[col] = round(max(a, 1 - a), 4)
OUT['A3_solo_auc'] = solo
print('solo auc:', solo)

# ────────────────────────────────────────────────────────────
# P2-A-5  層別に3つの絶対量の順位が食い違うか
# ────────────────────────────────────────────────────────────
print('\n=== A-5 concept mixing by stratum ===', flush=True)
from scipy.stats import spearmanr


def rho(mask, a, b):
    if mask.sum() < 500:
        return None, int(mask.sum())
    r = spearmanr(Xc.loc[mask, a].values, Xc.loc[mask, b].values).statistic
    return round(float(r), 4), int(mask.sum())


strata = {'ALL': pd.Series(True, index=df.index)}
cr = df['f_career_runs'].fillna(0)
strata['career_1-2'] = (cr >= 1) & (cr <= 2)
strata['career_3-5'] = (cr >= 3) & (cr <= 5)
strata['career_6-9'] = (cr >= 6) & (cr <= 9)
strata['career_10+'] = cr >= 10
cl = df['f_class_level'].fillna(-1)
for v in sorted(cl.unique()):
    if (cl == v).sum() >= 2000:
        strata['class_%s' % v] = cl == v
ds = df['f_days_since_last'].fillna(-1)
strata['rest_<=14'] = (ds >= 0) & (ds <= 14)
strata['rest_15-56'] = (ds > 14) & (ds <= 56)
strata['rest_57+'] = ds > 56

A5 = {}
for name, m in strata.items():
    row = {}
    for pa, pb in [('f_pl_rating', 'f_recent'), ('f_pl_rating', 'f_last1_rank'),
                   ('f_recent', 'f_last1_rank')]:
        r, n = rho(m, pa, pb)
        row['%s|%s' % (pa, pb)] = r
        row['n'] = n
    A5[name] = row
    print('%-14s n=%7d  pl~recent %s  pl~last1 %s  recent~last1 %s'
          % (name, row['n'], row['f_pl_rating|f_recent'],
             row['f_pl_rating|f_last1_rank'], row['f_recent|f_last1_rank']))
OUT['A5'] = A5

# ────────────────────────────────────────────────────────────
# P2-B-1  ICC（レース平均の分散比）
# ────────────────────────────────────────────────────────────
print('\n=== B-1 ICC of race mean ===', flush=True)
import re
pairs = []
for c in FEATS:
    m = re.match(r'^(?:cl|rl)_(.+)_vs_field$', c)
    if m and m.group(1) in FEATS:
        pairs.append((m.group(1), c))
pairs.sort()
B1 = {}
for base_c, rel_c in pairs:
    v = Xf[base_c]
    mr = v.groupby(rid).transform('mean')
    tot = float(np.var(v.values))
    icc = float(np.var(mr.values)) / tot if tot > 0 else 0.0
    rr = float(np.corrcoef(Xf[base_c].values, Xf[rel_c].values)[0, 1])
    B1[base_c] = {'rel': rel_c, 'icc': round(icc, 4), 'r_raw': round(rr, 4)}
    print('%-24s icc=%.4f  r_raw=%+.4f' % (base_c, icc, rr))
OUT['B1'] = B1

json.dump(OUT, io.open(os.path.join(REPO, 'noryoku', 'p2_report.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\nwrote noryoku/p2_report.json', flush=True)
