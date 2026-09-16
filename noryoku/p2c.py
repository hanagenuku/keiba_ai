# -*- coding: utf-8 -*-
"""Phase 2-C: 独立した情報の塊はいくつあるか（並行分析 + 回転 + 年跨ぎ）。

対照群B（既知の2因子を注入して検出できるか）を**本データより先に**通す。
通らなければ本データの結果を印字せず中止する。

usage: python3 noryoku/p2c.py <scratch_base_dir>
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = sys.argv[1]
RNG = np.random.default_rng(20260916)
OUT = {}

COLS = json.load(io.open(os.path.join(REPO, 'data', 'xgb_feature_cols.json'), encoding='utf-8'))
FEATS = COLS['feature_cols'] if isinstance(COLS, dict) else COLS
DROP = ['f_early_speed', 'f_same_course_rate', 'f_course_stamina_demand', 'f_is_debut']
LIVE = [c for c in FEATS if c not in DROP]


def eigs(Z):
    """標準化済み行列の相関行列の固有値（降順）。"""
    n = Z.shape[0]
    C = (Z.T @ Z) / (n - 1)
    w = np.linalg.eigvalsh(C)
    return w[::-1]


def standardize(A):
    A = A - A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return A / sd


def parallel_k(Z, reps=100, tag=''):
    """並行分析。列ごとに独立にシャッフルした偽データの95%点を超えた固有値の数。"""
    real = eigs(Z)
    n, p = Z.shape
    null = np.empty((reps, p))
    S = Z.copy()
    for r in range(reps):
        for j in range(p):
            S[:, j] = Z[RNG.permutation(n), j]
        null[r] = eigs(S)
        if tag and (r + 1) % 25 == 0:
            print('   %s shuffle %d/%d' % (tag, r + 1, reps), flush=True)
    p95 = np.percentile(null, 95, axis=0)
    k = int(np.sum(real > p95))
    return k, real, p95


# ────────────────────────────────────────────────────────────
# 対照群B（感度検査）— 本データより先に
# ────────────────────────────────────────────────────────────
print('=== control B: inject a known 2-factor structure ===', flush=True)
n_syn, p_syn = 20000, 130
F1 = RNG.standard_normal(n_syn)
F2 = RNG.standard_normal(n_syn)
load = 0.6
S = np.empty((n_syn, p_syn))
for j in range(p_syn):
    f = F1 if j < p_syn // 2 else F2
    S[:, j] = load * f + np.sqrt(1 - load ** 2) * RNG.standard_normal(n_syn)
kB, realB, p95B = parallel_k(standardize(S), reps=30, tag='ctrlB')
print('control B  k = %d  (expect 2)   top eig %s' % (kB, np.round(realB[:5], 3)), flush=True)
OUT['controlB_k'] = kB
OUT['controlB_top_eig'] = [round(float(v), 4) for v in realB[:6]]
if kB != 2:
    print('!! control B failed (k=%d). 本データの結果は印字せず中止する。' % kB)
    json.dump(OUT, io.open(os.path.join(REPO, 'noryoku', 'p2c_report.json'), 'w',
                           encoding='utf-8'), ensure_ascii=False, indent=1)
    sys.exit(1)
print('control B OK\n', flush=True)

# ────────────────────────────────────────────────────────────
# 本データ
# ────────────────────────────────────────────────────────────
print('read csv...', flush=True)
df = pd.read_csv(os.path.join(BASE, 'data', 'horse_features.csv'), low_memory=False)
df = df[df['place'].notna() & (df['place'] > 0) & (df['place'] < 99)].reset_index(drop=True)
X = df[LIVE].astype(float)
X = X.fillna(X.median())
rid = df['race_id'].values
Xc = X - X.groupby(rid).transform('mean')          # レース内中心化
keep = [c for c in LIVE if Xc[c].std() > 1e-9]
print('live %d -> within-race varying %d' % (len(LIVE), len(keep)), flush=True)
OUT['n_live'] = len(LIVE)
OUT['n_varying'] = len(keep)
OUT['zero_within_race_var'] = [c for c in LIVE if c not in keep]

Z = standardize(Xc[keep].values)
print('parallel analysis on real data (%d x %d)...' % Z.shape, flush=True)
k, real, p95 = parallel_k(Z, reps=100, tag='real')
print('k = %d' % k, flush=True)
print('real eig top20 :', np.round(real[:20], 3), flush=True)
print('null p95 top20 :', np.round(p95[:20], 3), flush=True)
OUT['k'] = k
OUT['real_eig'] = [round(float(v), 4) for v in real[:30]]
OUT['null_p95'] = [round(float(v), 4) for v in p95[:30]]
cum = np.cumsum(real) / real.sum()
OUT['cum_var'] = [round(float(v), 4) for v in cum[:30]]
print('cum var top10  :', np.round(cum[:10], 3), flush=True)


# ────────────────────────────────────────────────────────────
# 回転して中身を見る
# ────────────────────────────────────────────────────────────
def varimax(Phi, gamma=1.0, q=60, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


def loadings(Zm, kk):
    n = Zm.shape[0]
    C = (Zm.T @ Zm) / (n - 1)
    w, v = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:kk]
    L = v[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    return varimax(L)[0] if kk > 1 else L


KK = min(max(k, 1), 8)
L = loadings(Z, KK)
fac = {}
for f in range(KK):
    order = np.argsort(-np.abs(L[:, f]))
    top = [(keep[i], round(float(L[i, f]), 3)) for i in order[:12] if abs(L[i, f]) >= 0.40]
    ss = float(np.sum(L[:, f] ** 2))
    fac['F%d' % (f + 1)] = {'ss_loading': round(ss, 3),
                            'var_share': round(ss / len(keep), 4), 'top': top}
    print('\nF%d  SS=%.2f (%.1f%% of columns)' % (f + 1, ss, 100 * ss / len(keep)), flush=True)
    for c, v in top:
        print('    %+.3f  %s' % (v, c))
OUT['factors'] = fac

# 因子スコア間の相関（varimax は直交なので参考値）
Fs = Z @ L @ np.linalg.pinv(L.T @ L)
Fc = np.corrcoef(Fs.T)
OUT['factor_corr'] = [[round(float(Fc[i, j]), 3) for j in range(KK)] for i in range(KK)]
print('\nfactor score corr:\n', np.round(Fc, 3), flush=True)

# ────────────────────────────────────────────────────────────
# C-5 年跨ぎ
# ────────────────────────────────────────────────────────────
print('\n=== C-5 year crossing ===', flush=True)
d = pd.to_datetime(df['date'], errors='coerce')
m_early = (d < '2025-01-01').values
m_late = (d >= '2025-01-01').values
print('early %d / late %d' % (m_early.sum(), m_late.sum()), flush=True)
Le = loadings(standardize(Xc[keep].values[m_early]), KK)
Ll = loadings(standardize(Xc[keep].values[m_late]), KK)
# 因子の対応づけ（絶対相関が最大のものを貪欲に対応させる）
M = np.abs(np.corrcoef(Le.T, Ll.T)[:KK, KK:])
used, pairs = set(), []
for i in range(KK):
    j = int(np.argmax([M[i, j] if j not in used else -1 for j in range(KK)]))
    used.add(j)
    pairs.append((i, j, round(float(M[i, j]), 4)))
OUT['C5_loading_corr'] = pairs
for i, j, r in pairs:
    print('  early F%d <-> late F%d   |rho(loading)| = %.4f' % (i + 1, j + 1, r), flush=True)
ke, _, _ = parallel_k(standardize(Xc[keep].values[m_early]), reps=30)
kl, _, _ = parallel_k(standardize(Xc[keep].values[m_late]), reps=30)
OUT['C5_k_early'] = ke
OUT['C5_k_late'] = kl
print('  k early=%d  late=%d  (all=%d)' % (ke, kl, k), flush=True)

json.dump(OUT, io.open(os.path.join(REPO, 'noryoku', 'p2c_report.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\nwrote noryoku/p2c_report.json', flush=True)
