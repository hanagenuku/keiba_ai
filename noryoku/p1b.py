# -*- coding: utf-8 -*-
"""Phase 1b — 現在の特徴量の棚卸し（本番モデルは一切変更しない）

出力
  1. 死んだ列      … 定数 / 全欠損 / 分割に一度も出ない
  2. 重複列        … |r| >= 0.90 の対
  3. 同義の群      … 相関でまとめた連結成分
  4. 能力軸候補    … 摂動分類（inv1.py）と突き合わせ
  5. 市場との関係  … 今回の人気との相関（本番で市場が入るのは base_margin だけ）
"""
import io, json, os, sys, math
import numpy as np
import pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else '.'
REPO = os.path.dirname(os.path.abspath(__file__)) + '/..'
CSV  = os.path.join(BASE, 'data', 'horse_features.csv')
OUT  = os.path.join(REPO, 'noryoku', 'p1b_report.json')

COLS = json.load(io.open(os.path.join(REPO, 'data', 'xgb_feature_cols.json'), encoding='utf-8'))
FEATS = COLS['feature_cols'] if isinstance(COLS, dict) else COLS
print(f'本番の特徴量 {len(FEATS)} 列')

df = pd.read_csv(CSV, low_memory=False)
print(f'CSV {df.shape[0]:,} 行 × {df.shape[1]} 列')
missing = [c for c in FEATS if c not in df.columns]
if missing:
    print('⚠ CSVに無い本番列:', missing)
F = [c for c in FEATS if c in df.columns]
X = df[F].apply(pd.to_numeric, errors='coerce')

rep = {'n_rows': int(len(df)), 'n_feats': len(F)}

# ---------- 1. 死んだ列 ----------
print('\n=== 1. 死んだ列 ===')
dead = []
for c in F:
    v = X[c]
    nn = v.notna().sum()
    nuniq = v.nunique(dropna=True)
    if nn == 0:
        dead.append((c, 'all-nan', None, 0))
    elif nuniq <= 1:
        dead.append((c, 'constant', float(v.dropna().iloc[0]) if nn else None, int(nn)))
    elif v.std(skipna=True) == 0:
        dead.append((c, 'zero-var', float(v.dropna().iloc[0]), int(nn)))
for c, kind, val, nn in dead:
    print(f'  {c:34s} {kind:9s} value={val} nonnull={nn:,}')
rep['dead_constant'] = [{'col': c, 'kind': k, 'value': v, 'nonnull': n} for c, k, v, n in dead]

# gain（本番モデルの分割に出ない列）
gains = {}
try:
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(os.path.join(REPO, 'data', 'xgb_fukusho_model.pkl'))
    raw = b.get_score(importance_type='gain')
    tot = sum(raw.values()) or 1.0
    gains = {c: 100.0 * raw.get(c, 0.0) / tot for c in F}
    zero = sorted([c for c in F if gains[c] == 0.0])
    print(f'\n  分割に一度も出ない列 {len(zero)}')
    for c in zero:
        print(f'    {c}')
    rep['zero_gain'] = zero
except Exception as e:
    print('  gain 取得失敗', e)
rep['gain'] = {c: round(gains.get(c, 0.0), 4) for c in F}

# ---------- 2. 重複列 ----------
print('\n=== 2. 重複列（|r| >= 0.90） ===')
live = [c for c in F if X[c].nunique(dropna=True) > 1]
C = X[live].corr(method='pearson', min_periods=2000)
pairs = []
for i, a in enumerate(live):
    for b2 in live[i+1:]:
        r = C.loc[a, b2]
        if pd.notna(r) and abs(r) >= 0.90:
            pairs.append((a, b2, float(r)))
pairs.sort(key=lambda t: -abs(t[2]))
for a, b2, r in pairs:
    print(f'  {r:+.3f}  {a:32s} × {b2}   gain {gains.get(a,0):.2f}% / {gains.get(b2,0):.2f}%')
print(f'  計 {len(pairs)} 対')
rep['dup_pairs'] = [{'a': a, 'b': b2, 'r': round(r, 4),
                     'gain_a': round(gains.get(a, 0), 3), 'gain_b': round(gains.get(b2, 0), 3)}
                    for a, b2, r in pairs]

# ---------- 3. 同義の群（|r| >= 0.80 の連結成分） ----------
print('\n=== 3. 同じ意味の特徴量群（|r| >= 0.80 の連結成分） ===')
par = {c: c for c in live}
def find(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
def uni(a, b2):
    ra, rb = find(a), find(b2)
    if ra != rb: par[rb] = ra
for i, a in enumerate(live):
    for b2 in live[i+1:]:
        r = C.loc[a, b2]
        if pd.notna(r) and abs(r) >= 0.80:
            uni(a, b2)
grp = {}
for c in live:
    grp.setdefault(find(c), []).append(c)
groups = [sorted(v) for v in grp.values() if len(v) > 1]
groups.sort(key=lambda g: -sum(gains.get(c, 0) for c in g))
for g in groups:
    print(f'  [{len(g)}列 gain {sum(gains.get(c,0) for c in g):.2f}%] ' + ' / '.join(g))
rep['groups'] = [{'cols': g, 'gain': round(sum(gains.get(c, 0) for c in g), 3)} for g in groups]

# ---------- 4. 能力軸候補（摂動分類と突き合わせ） ----------
print('\n=== 4. 能力軸候補 ===')
inv = json.load(io.open(os.path.join(REPO, 'noryoku', 'inventory.json'), encoding='utf-8'))
buckets = {}
for c in F:
    d = inv.get(c)
    if not d:
        k = 'unknown'
    elif not d['varies_horse']:
        k = 'レース共通'
    elif d['varies_course']:
        k = '馬×今回の条件'
    else:
        k = '馬のみ'
    buckets.setdefault(k, []).append(c)
for k in ['馬のみ', '馬×今回の条件', 'レース共通', 'unknown']:
    v = buckets.get(k, [])
    if not v: continue
    print(f'  {k}: {len(v)}列 gain {sum(gains.get(c,0) for c in v):.1f}%')
rep['buckets'] = {k: v for k, v in buckets.items()}

# ---------- 5. 市場情報との関係 ----------
print('\n=== 5. 市場との関係（今回の人気との相関） ===')
mk = None
for cand in ['popularity', 'f_popularity', 'pop']:
    if cand in df.columns:
        mk = pd.to_numeric(df[cand], errors='coerce'); print(f'  市場列: {cand}'); break
if mk is not None and mk.notna().sum() > 1000:
    rows = []
    for c in live:
        r = X[c].corr(mk)
        if pd.notna(r):
            rows.append((c, float(r)))
    rows.sort(key=lambda t: -abs(t[1]))
    print('  |r| 上位20（今回の人気と近い＝市場と情報が重なる）')
    for c, r in rows[:20]:
        print(f'    {r:+.3f}  {c:34s} gain {gains.get(c,0):.2f}%')
    rep['market_corr'] = [{'col': c, 'r': round(r, 4)} for c, r in rows]
    strong = [c for c, r in rows if abs(r) >= 0.5]
    print(f'  |r| >= 0.5 の列 {len(strong)} / {len(rows)}')
else:
    print('  市場列がCSVに無い（本番でも市場は base_margin だけ）')

json.dump(rep, io.open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n→ {OUT}')
