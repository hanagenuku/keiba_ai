#!/usr/bin/env python3
"""探索（記述統計・判定ではない）: 4角の位置取りは事前に予測できるか／オッズが割れたレースで効くか。

ユーザーの問い（2026-11-13）:
  ① 4コーナーの位置取りにフォーカスして予測できないか
  ② オッズが割れているレースなら、高配当の予想の的中率が上がるのでは

先に決めたこと（数字を見る前に固定）:
  - 4角の相対位置 rel4 = (corner_4 - 1) / (field_size - 1)   0=先頭 1=最後方
  - 特徴量は**その馬の過去走だけ**（当該レースを除く）＋レース条件。S=市場なし / SM=＋人気
  - 学習 〜2025-12-31 / 検定 2026-01-01〜（完全OOS）
  - 割れたレース = 予想時点の1番人気の単勝オッズが 3.5倍以上（keiba.db race_predictions）
  - 高配当の候補 = 予想時点で5番人気以下
  - 「前に行けそう」= S モデルの予測 rel4 がレース内で前から 1/3
使い方: cd <隔離ワークスペース> && python3 tenkai/explore_corner4.py
"""
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

h = sqlite3.connect('data/history.db')
df = pd.read_sql_query(
    """SELECT race_id, date, racecourse, distance, surface, place, horse_num, horse_name,
              popularity, fukusho_payout, corner_4, corner_all, field_size
       FROM horse_history WHERE place IS NOT NULL AND place < 99""", h)
df['date'] = df['date'].astype(str).str[:10]
n = df.groupby('race_id')['horse_num'].transform('count')
df['n'] = np.where(df['field_size'].fillna(0) > 1, df['field_size'], n)
df = df[(df['n'] >= 5)].copy()
# 🔴 corner_4 列は 2026-07 以降ずっと NULL（corner_all だけ入っている）。corner_all の最後の値を正とする
last = pd.to_numeric(df['corner_all'].astype(str).str.split('-').str[-1], errors='coerce')
both = df['corner_4'].fillna(0).gt(0) & last.notna()
print(f'corner_4 と corner_all 末尾の一致率（両方ある {int(both.sum()):,}頭）: {(df.loc[both, "corner_4"] == last[both]).mean():.1%}')
c4 = last.where(last.notna(), df['corner_4'].where(df['corner_4'].fillna(0) > 0))
df['popularity'] = df['popularity'].where(df['popularity'].between(1, 18))   # 99 は壊れた値
df['rel4'] = (c4 - 1) / (df['n'] - 1)
df['rel4'] = df['rel4'].clip(0, 1)
first = pd.to_numeric(df['corner_all'].astype(str).str.split('-').str[0], errors='coerce')
df['rel1'] = ((first - 1) / (df['n'] - 1)).clip(0, 1)
df = df.sort_values(['horse_name', 'date', 'race_id']).reset_index(drop=True)
g = df.groupby('horse_name')
for col in ('rel4', 'rel1'):
    s = g[col]
    df[f'p_{col}_mean'] = s.transform(lambda x: x.shift().expanding().mean())
    df[f'p_{col}_last3'] = s.transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df[f'p_{col}_min'] = s.transform(lambda x: x.shift().expanding().min())
df['p_runs'] = g.cumcount()
df['p_lead_rate'] = g['rel4'].transform(lambda x: (x.shift() <= 0.0001).astype(float).expanding().mean())
df['surf'] = (df['surface'].astype(str).str.contains('ダ')).astype(int)
df['num_rel'] = (df['horse_num'] - 1) / (df['n'] - 1)
df['venue'] = df['racecourse'].astype('category').cat.codes
S = ['p_rel4_mean', 'p_rel4_last3', 'p_rel4_min', 'p_rel1_mean', 'p_rel1_last3', 'p_rel1_min',
     'p_runs', 'p_lead_rate', 'surf', 'distance', 'n', 'num_rel', 'venue']
# レース内の相対化（相手関係）: 過去の位置取り平均のレース内順位
df['p_rel4_rank'] = df.groupby('race_id')['p_rel4_mean'].rank(pct=True)
df['p_rel1_rank'] = df.groupby('race_id')['p_rel1_mean'].rank(pct=True)
df['n_front'] = df.groupby('race_id')['p_rel4_mean'].transform(lambda x: (x < 0.25).sum())
S += ['p_rel4_rank', 'p_rel1_rank', 'n_front']
df['pop_rel'] = (df['popularity'] - 1) / (df['n'] - 1)
SM = S + ['popularity', 'pop_rel']

ok = df['rel4'].notna()
tr, te = df[ok & (df['date'] <= '2025-12-31')], df[ok & (df['date'] >= '2026-01-01')]
print(f'学習 {len(tr):,}頭 / 検定 {len(te):,}頭・{te.race_id.nunique():,}レース')

def fit(cols):
    m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=50, random_state=42)
    m.fit(tr[cols], tr['rel4'])
    return m

def race_spearman(d, col):
    out = []
    for _, x in d.groupby('race_id'):
        if x[col].notna().sum() >= 5 and x[col].nunique() > 1:
            out.append(spearmanr(x[col], x['rel4'], nan_policy='omit')[0])
    return float(np.nanmean(out))

mS, mSM = fit(S), fit(SM)
df['pred_S'] = mS.predict(df[S]); df['pred_SM'] = mSM.predict(df[SM])
te = df[ok & (df['date'] >= '2026-01-01')].copy()
print('\n■ ① 4角の相対位置はどれだけ当たるか（検定 2026年・レース内の順位相関の平均）')
for lab, col in (('過去の4角位置の平均だけ', 'p_rel4_mean'), ('人気だけ', 'popularity'),
                 ('S: 過去の位置取り＋条件＋相手関係', 'pred_S'), ('SM: S＋人気', 'pred_SM')):
    d = te.dropna(subset=[col])
    r2 = 1 - ((d['rel4'] - d[col]) ** 2).sum() / ((d['rel4'] - tr['rel4'].mean()) ** 2).sum() if col.startswith('pred') else float('nan')
    print(f'  {lab:28s} ρ = {race_spearman(d, col):+.3f}   R² = {r2:.3f}')
te['pr_rank'] = te.groupby('race_id')['pred_S'].rank(method='first')
te['ac_rank'] = te.groupby('race_id')['rel4'].rank(method='first')
f3 = te[te['pr_rank'] <= 3]
print(f'  S が「4角で前から3番手以内」と予測した馬が実際に3番手以内: {(f3["ac_rank"] <= 3).mean():.1%}'
      f'（でたらめなら {(3 / te.groupby("race_id")["n"].first()).mean():.1%}）')
lead = te[te['pr_rank'] == 1]
print(f'  S が先頭と予測した馬が実際に4角先頭: {(lead["ac_rank"] == 1).mean():.1%}')

print('\n■ 4角位置と3着内の関係を「予測できた分」と「予測からのずれ」に分ける（検定 2026年）')
te['top3'] = (te['place'] <= 3).astype(int)
te['surprise'] = te['rel4'] - te['pred_SM']          # 負＝予測より前にいた
for lab, col in (('予測した4角位置（SM）', 'pred_SM'), ('予測からのずれ（実際−予測）', 'surprise'), ('実際の4角位置', 'rel4')):
    q = pd.qcut(te[col], 5, labels=False, duplicates='drop')
    print(f'  {lab}: 五分位ごとの3着内率（前→後ろ）', [f'{v:.1%}' for v in te.groupby(q)['top3'].mean()])
pm = te.groupby(pd.qcut(te['pred_SM'], 5, labels=False))['popularity'].mean()
print('  参考: 予測4角位置の五分位ごとの平均人気', [f'{v:.1f}' for v in pm])

# ── ② オッズが割れたレース ─────────────────────────────────────
k = sqlite3.connect('data/keiba.db')
rp = pd.read_sql_query("SELECT race_id, horse_num, popularity AS pop_am, tansho_odds FROM race_predictions WHERE tansho_odds > 0", k)
m = rp.merge(df[['race_id', 'horse_num', 'place', 'fukusho_payout', 'pred_S', 'rel4', 'n']], on=['race_id', 'horse_num'])
m = m.dropna(subset=['pred_S'])
m['fav_odds'] = m.groupby('race_id')['tansho_odds'].transform('min')
m['front'] = m.groupby('race_id')['pred_S'].rank(pct=True) <= 1 / 3
m['top3'] = (m['place'] <= 3).astype(int)
m['pay'] = np.where(m['top3'] == 1, m['fukusho_payout'].fillna(0), 0)
m['evalable'] = ~((m['top3'] == 1) & (m['pay'] <= 0))
print(f'\n■ ② オッズが割れたレース（予想時点のオッズがある {m.race_id.nunique():,}レース・2026-06-27〜11-08）')

def cell(d):
    e = d[d['evalable']]
    return f'N={len(d):5d}  3着内率 {d["top3"].mean():6.1%}  複勝回収 {e["pay"].sum() / max(len(e), 1):6.1f}%'

rng = np.random.default_rng(0)
m['band'] = pd.cut(m['pop_am'], [4, 7, 10, 18], labels=['5-7番人気', '8-10番人気', '11番人気以下'])
THR = 3.5
for name, sel in (('割れた（1番人気3.5倍以上）', m['fav_odds'] >= THR), ('割れていない', m['fav_odds'] < THR)):
    d = m[sel & (m['pop_am'] >= 5)].copy()
    print(f'  ── {name}レース {d.race_id.nunique()}本・5番人気以下 ──')
    a, b = d[d['front']], d[~d['front']]
    print(f'      全体  前に行けそう {cell(a)} 平均人気 {a.pop_am.mean():.1f}')
    print(f'            それ以外     {cell(b)} 平均人気 {b.pop_am.mean():.1f}')
    diffs, ws = [], []
    for bn, x in d.groupby('band', observed=True):
        xa, xb = x[x['front']], x[~x['front']]
        print(f'      {bn:9s} 前 {cell(xa)} | 他 {cell(xb)}')
        diffs.append(xa['top3'].mean() - xb['top3'].mean()); ws.append(len(xa))
    adj = float(np.average(diffs, weights=ws))
    # 対照: 同じレース・同じ人気帯の中で「前に行けそう」の札を並べ替える
    keys = (d['race_id'].astype(str) + '|' + d['band'].astype(str)).to_numpy()
    idx = pd.Series(np.arange(len(d))).groupby(keys).apply(lambda q: q.to_numpy()).tolist()
    fr, t3, bd = d['front'].to_numpy(), d['top3'].to_numpy(), d['band'].astype(str).to_numpy()
    def stat(f):
        ds, w = [], []
        for bn in np.unique(bd):
            mk = bd == bn
            if f[mk].sum() and (~f[mk]).sum():
                ds.append(t3[mk & f].mean() - t3[mk & ~f].mean()); w.append((mk & f).sum())
        return np.average(ds, weights=w)
    null = []
    for _ in range(500):
        pf = fr.copy()
        for ix in idx:
            pf[ix] = rng.permutation(fr[ix])
        null.append(stat(pf))
    null = np.array(null)
    print(f'      人気帯をそろえた3着内率の差 {adj:+.2%}  並べ替え 平均 {null.mean():+.2%}・sd {null.std():.2%}・P={(np.sum(null >= adj) + 1) / 501:.3f}')
ee = m[(m['pop_am'] >= 5) & m['evalable']]
print('  参考: 5番人気以下の複勝回収率 全体', f'{ee["pay"].sum() / len(ee):.1f}%')

# ── ③ 広い期間での確かめ（確定人気で代用。予想時点のオッズは 2026-06-27 以降しか無いため）──
print('\n■ ③ 5番人気以下（確定人気）を「前に行けそう」で分けた複勝（学習 〜2024-12-31 のモデルで 2025年・2026年を別々に）')
tr2 = df[ok & (df['date'] <= '2024-12-31')]
m2 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=50, random_state=42).fit(tr2[S], tr2['rel4'])
z = df[df['date'] >= '2025-01-01'].copy()
z['pred'] = m2.predict(z[S])
z = z.dropna(subset=['popularity'])
z['front'] = z.groupby('race_id')['pred'].rank(pct=True) <= 1 / 3
z['top3'] = (z['place'] <= 3).astype(int)
z['pay'] = np.where(z['top3'] == 1, z['fukusho_payout'].fillna(0), 0)
z['evalable'] = ~((z['top3'] == 1) & (z['pay'] <= 0))
z['band'] = pd.cut(z['popularity'], [4, 7, 10, 18], labels=['5-7番人気', '8-10番人気', '11番人気以下'])
z['year'] = z['date'].str[:4]
for yr, d in z[z['popularity'] >= 5].groupby('year'):
    a, b = d[d['front']], d[~d['front']]
    print(f'  {yr}年 {d.race_id.nunique():,}レース  前に行けそう {cell(a)} | それ以外 {cell(b)}')
    for bn, x in d.groupby('band', observed=True):
        print(f'      {bn:9s} 前 {cell(x[x["front"]])} | 他 {cell(x[~x["front"]])}')
    # 半年ごと
    for hf, x in d.groupby(d['date'].str[5:7].astype(int).le(6).map({True: '前半', False: '後半'})):
        e1, e2 = x[x['front'] & x['evalable']], x[~x['front'] & x['evalable']]
        print(f'      {hf} 複勝回収 前 {e1["pay"].sum() / len(e1):.1f}% | 他 {e2["pay"].sum() / len(e2):.1f}%')
allp = z[z['evalable']]
print('  参考: 同じ期間の1〜4番人気の複勝回収', f'{allp[allp.popularity <= 4]["pay"].sum() / (allp.popularity <= 4).sum():.1f}%')
