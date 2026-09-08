"""Phase 4: レース条件ごとに AI(市場フリー cid82) と 市場 の性能差を測る。

⚠ 買い条件は作らない。ROI も測らない（指示#1・#17）。
⚠ 市場と比べるときは必ず「人気が取れている行だけ」に揃える（M-1 の教訓）。
⚠ 探索期(2025 T1-T3) と 確認期(2026 W1-W3) を分けて、再現するかを見る（指示#14/#16）。
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)

D = pickle.load(open(f'{BASE}/p4_data.pkl', 'rb'))
P = pickle.load(open(f'{BASE}/p4_pred.pkl', 'rb'))
META = D['meta']
TUNE, CONF = ['T1', 'T2', 'T3'], ['W1', 'W2', 'W3']
CPROF = json.load(open(f'{BASE}/data/course_profiles.json'))


def build_frame():
    """全窓の評価行を1枚に。AI確率・市場スコア・条件を横に並べる。"""
    fr = []
    for w in TUNE + CONF:
        win = D['windows'][w]
        idx = win['va_idx']
        m = META.iloc[idx].reset_index(drop=True)
        # 市場側も内側HOで Isotonic 較正して LogLoss を公平に比べる
        oki = np.isfinite(win['popi']) & (win['popi'] > 0)
        iso_m = IsotonicRegression(out_of_bounds='clip').fit(
            -win['popi'][oki], win['y3i'][oki])
        m['win'] = w
        m['period'] = 'TUNE' if w in TUNE else 'CONF'
        m['y3'] = win['y3v']; m['y1'] = win['y1v']
        m['pop'] = win['popv']
        m['ai'] = P[w]['prob']; m['ai_cal'] = P[w]['cal']
        m['mkt'] = -win['popv']
        okv = np.isfinite(win['popv']) & (win['popv'] > 0)
        mc = np.full(len(okv), np.nan)
        mc[okv] = np.clip(iso_m.predict(-win['popv'][okv]), 1e-6, 1 - 1e-6)
        m['mkt_cal'] = mc
        fr.append(m)
    f = pd.concat(fr, ignore_index=True)
    f = f[np.isfinite(f['pop']) & (f['pop'] > 0)].reset_index(drop=True)  # 母集団を揃える
    # レース内の AI 確率分布からレース構造の指標を作る
    g = f.groupby(['win', 'race_id'])['ai']
    f['ai_rank'] = g.rank(ascending=False, method='first')
    f['mkt_rank'] = f.groupby(['win', 'race_id'])['pop'].rank(method='first')
    return f


def race_struct(f):
    """レース単位の構造指標（AIの集中度・市場との一致度）。"""
    rows = []
    for (w, rid), g in f.groupby(['win', 'race_id'], sort=False):
        p = np.sort(g['ai'].values)[::-1]
        s = p.sum()
        q = p / s if s > 0 else np.full(len(p), 1 / len(p))
        top1_ai = g.loc[g.ai_rank == 1, 'mkt_rank']
        rows.append(dict(win=w, race_id=rid,
                         gap12=p[0] - p[1] if len(p) > 1 else np.nan,
                         top3=q[:3].sum(),
                         entropy=-(q * np.log(q + 1e-12)).sum() / np.log(len(q)),
                         ai1_pop=float(top1_ai.iloc[0]) if len(top1_ai) else np.nan,
                         agree3=len(set(g.nsmallest(3, 'ai_rank').horse_num) &
                                    set(g.nsmallest(3, 'mkt_rank').horse_num))))
    return pd.DataFrame(rows)


def auc_pair(sub, ycol='y3'):
    y = sub[ycol].values
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    return roc_auc_score(y, sub['ai'].values), roc_auc_score(y, sub['mkt'].values)


def boot_diff(sub, ycol='y3', n=300, seed=0):
    """レース単位ブートストラップで AI−市場 の差のCIを出す。"""
    rng = np.random.default_rng(seed)
    races = sub['race_id'].unique()
    by = {r: g for r, g in sub.groupby('race_id', sort=False)}
    out = []
    for _ in range(n):
        pick = rng.choice(races, len(races), replace=True)
        s = pd.concat([by[r] for r in pick], ignore_index=True)
        y = s[ycol].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        out.append(roc_auc_score(y, s['ai'].values) - roc_auc_score(y, s['mkt'].values))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def summarize(f, cond_col, label, min_race=60, boot=True):
    rows = []
    for val, sub_all in f.groupby(cond_col, sort=False):
        rec = dict(axis=label, cond=str(val))
        ok = True
        for per in ['TUNE', 'CONF']:
            s = sub_all[sub_all.period == per]
            nr = s.race_id.nunique()
            rec[f'{per}_race'] = nr; rec[f'{per}_horse'] = len(s)
            if nr < min_race:
                ok = False
                rec[f'{per}_d3'] = np.nan; rec[f'{per}_d1'] = np.nan
                continue
            a3, m3 = auc_pair(s, 'y3'); a1, m1 = auc_pair(s, 'y1')
            rec[f'{per}_ai3'] = a3; rec[f'{per}_mk3'] = m3; rec[f'{per}_d3'] = a3 - m3
            rec[f'{per}_ai1'] = a1; rec[f'{per}_mk1'] = m1; rec[f'{per}_d1'] = a1 - m1
            rec[f'{per}_field'] = s.field.mean()
            rec[f'{per}_dll'] = (log_loss(s.y3, s.ai_cal) - log_loss(s.y3, s.mkt_cal))
        if ok and boot:
            lo, hi = boot_diff(sub_all[sub_all.period == 'CONF'])
            rec['CONF_lo'], rec['CONF_hi'] = lo, hi
        rows.append(rec)
    return pd.DataFrame(rows)
