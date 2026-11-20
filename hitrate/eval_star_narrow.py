#!/usr/bin/env python3
"""事前登録カード hitrate/CRITERIA_STAR_NARROW.md の測定（1回・GitHub ランナー）。

    python3 hitrate/eval_star_narrow.py     # リポジトリ直下で。data/horse_features.csv（web 列つき）と history.db が要る
"""
import importlib.util
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
_s = importlib.util.spec_from_file_location('eval_c', os.path.join(ROOT, 'hitrate', 'eval_c.py'))
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)
from src.tools.train_xgb import _popularity_to_base_margin  # noqa: E402

WINDOWS = [('W0', '2024-02-29', '2024-03-01', '2024-11-24'),
           ('W1', '2024-12-31', '2025-01-01', '2025-04-30'),
           ('W2', '2025-04-30', '2025-05-01', '2025-08-31'),
           ('W3', '2025-08-31', '2025-09-01', '2025-12-31'),
           ('W4', '2025-12-31', '2026-01-01', '2026-11-15')]
SEED = 42
if os.environ.get('STAR_NARROW_SMOKE'):
    # 配管の確認だけ（判定の窓は使わない）。2024年初の1窓・並べ替えを少なく
    WINDOWS = [('S', '2023-12-31', '2024-01-01', '2024-02-29')]
Q, QS = 85, 95
POP_LO, POP_HI = 4, 9
PAID_P = 0.66
N_BOOT, N_PERM = (200, 5) if os.environ.get('STAR_NARROW_SMOKE') else (2000, 200)


def fsb(n):
    return 0 if n <= 12 else (1 if n <= 15 else 2)


def main():
    df = pd.read_csv('data/horse_features.csv')
    df['date_obj'] = pd.to_datetime(df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
                                    format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj']).sort_values(['race_id', 'horse_num']).reset_index(drop=True)
    cols = C.feat_cols(df)
    web = [c for c in cols if c.startswith('f_ws_')]
    if not web:
        raise SystemExit('web 列が無い（指数DBの復元に失敗している）')
    X = df[cols].copy()
    for c in cols:
        if c not in web:
            X[c] = X[c].fillna(5.0)
    X = X.to_numpy(dtype=np.float32)
    d = df['date_obj'].to_numpy()
    rid = df['race_id'].to_numpy()
    hn = df['horse_num'].to_numpy()
    con = sqlite3.connect('data/history.db')
    hh = pd.read_sql('SELECT race_id, horse_num, place, popularity, fukusho_payout FROM horse_history', con)
    key = {(r, n): (p, q, f) for r, n, p, q, f in hh.itertuples(index=False)}
    info = np.array([key.get((r, n), (np.nan, np.nan, 0)) for r, n in zip(rid, hn)], dtype=float)
    place, pop, fuku = info[:, 0], info[:, 1], np.nan_to_num(info[:, 2])
    pop = np.where((pop >= 1) & (pop <= 18), pop, np.nan)
    nfield = df.groupby('race_id')['horse_num'].transform('count').to_numpy()
    y3 = (place <= 3).astype(float)
    bm = _popularity_to_base_margin(pd.Series(np.nan_to_num(pop, nan=nfield / 2)), pd.Series(nfield))
    print(f'行 {len(df):,} / 列 {len(cols)}（web {web}）', flush=True)

    idx_of = {(r, int(n)): i for i, (r, n) in enumerate(zip(rid, hn))}
    mid = (pop >= POP_LO) & (pop <= POP_HI)
    fav_place = {}
    for r, p, q in zip(rid, place, pop):
        if q == 1:
            fav_place[r] = p

    out = {'windows': {}}
    stars = []
    pooled_by = defaultdict(list)
    for name, te, v0, v1 in WINDOWS:
        m_fit, m_in, _ = C.inner_split(d, te)
        m_va = (d >= np.datetime64(pd.Timestamp(v0))) & (d <= np.datetime64(pd.Timestamp(v1)))
        ok = np.isfinite(place) & (place > 0)
        # 期待値（学習期間から）
        tr = (m_fit | m_in) & ok & np.isfinite(pop)
        e3, eb = defaultdict(list), defaultdict(list)
        for i in np.flatnonzero(tr):
            k = (int(min(pop[i], 12)), fsb(nfield[i]))
            e3[k].append(y3[i])
            fp = fav_place.get(rid[i])
            if pop[i] != 1 and fp is not None and np.isfinite(fp):
                eb[k].append(float(place[i] < fp))
        E3 = {k: float(np.mean(v)) for k, v in e3.items()}
        EB = {k: float(np.mean(v)) for k, v in eb.items()}
        # S1: 実際の1番人気との対戦
        pf = C.build_pairs(np.flatnonzero(m_fit & ok), X, place, rid, hn, np.nan_to_num(pop))
        pi = C.build_pairs(np.flatnonzero(m_in & ok), X, place, rid, hn, np.nan_to_num(pop))
        pv = C.build_pairs(np.flatnonzero(m_va & ok), X, place, rid, hn, np.nan_to_num(pop))
        fp_ = C.fit_pair(pf[0], pf[1], pi[0], pi[1], pv[0], cols, SEED)
        s1_in, s1_in_m = defaultdict(float), {}
        for r, n_, p in zip(pi[2], pi[3], fp_['p_in']):
            s1_in[r] = max(s1_in[r], p)
            if mid[idx_of[(r, int(n_))]]:
                s1_in_m[r] = max(s1_in_m.get(r, 0.0), p)
        tau1 = float(np.percentile(list(s1_in.values()), Q))
        tau1m = {q_: float(np.percentile(list(s1_in_m.values()), q_)) for q_ in (Q, QS)}
        S1 = {(r, int(n)): float(p) for r, n, p in zip(pv[2], pv[3], fp_['p_va'])}
        # S2: 残差モデル − 市場（どちらも較正）
        iF, iI, iV = np.flatnonzero(m_fit & ok), np.flatnonzero(m_in & ok), np.flatnonzero(m_va & ok)
        fr = C.fit_residual(X[iF], y3[iF], bm[iF], X[iI], y3[iI], bm[iI],
                            np.vstack([X[iI], X[iV]]), np.r_[bm[iI], bm[iV]], cols, SEED)
        sig = lambda z: 1 / (1 + np.exp(-z))
        m_in_, m_va_ = fr['m_all'][:len(iI)], fr['m_all'][len(iI):]
        cm = IsotonicRegression(out_of_bounds='clip').fit(sig(m_in_), y3[iI])
        ck = IsotonicRegression(out_of_bounds='clip').fit(sig(bm[iI]), y3[iI])
        s2_in = cm.predict(sig(m_in_)) - ck.predict(sig(bm[iI]))
        s2_va = cm.predict(sig(m_va_)) - ck.predict(sig(bm[iV]))
        mx, mxm = defaultdict(lambda: -9.0), {}
        for i, v in zip(iI, s2_in):
            if pop[i] != 1:
                mx[rid[i]] = max(mx[rid[i]], v)
            if mid[i]:
                mxm[rid[i]] = max(mxm.get(rid[i], -9.0), v)
        tau2 = float(np.percentile(list(mx.values()), Q))
        tau2m = {q_: float(np.percentile(list(mxm.values()), q_)) for q_ in (Q, QS)}
        # 本命側（無料に回す案の姿）: 軸＝較正済み3着内確率が最大の馬・その確率 >= 0.66
        pv_ = cm.predict(sig(m_va_))
        axis = {}
        for i, v in zip(iV, pv_):
            if rid[i] not in axis or v > axis[rid[i]][1]:
                axis[rid[i]] = (i, float(v))
        # 評価窓のレースごとの候補
        races = defaultdict(list)
        for j, i in enumerate(iV):
            if pop[i] == 1 or not np.isfinite(pop[i]):
                continue
            races[rid[i]].append((i, S1.get((rid[i], int(hn[i])), np.nan), s2_va[j]))
        win = dict(tau1=round(tau1, 4), tau2=round(tau2, 5),
                   tau1m85=round(tau1m[Q], 4), tau2m85=round(tau2m[Q], 5), tau1m95=round(tau1m[QS], 4), tau2m95=round(tau2m[QS], 5),
                   races=len({rid[i] for i in iV}), days=len({str(d[i])[:10] for i in iV}))

        def make_pick(mode):
            # 'v2'（2番人気以下すべて・85）/ 'M85'（4〜9番人気・85）/ 'M95'（4〜9番人気・95）
            t1, t2 = (tau1, tau2) if mode == 'v2' else (tau1m[Q], tau2m[Q]) if mode == 'M85' else (tau1m[QS], tau2m[QS])

            def pick(rs, s1sh=None, s2sh=None):
                out_ = []
                for r, hs in rs.items():
                    if r not in fav_place:
                        continue
                    a1 = [h[1] for h in hs] if s1sh is None else s1sh[r]
                    a2 = [h[2] for h in hs] if s2sh is None else s2sh[r]
                    ok_ = [True] * len(hs) if mode == 'v2' else [bool(mid[h[0]]) for h in hs]
                    a1 = [v if o else np.nan for v, o in zip(a1, ok_)]
                    a2 = [v if o else -9.0 for v, o in zip(a2, ok_)]
                    if not any(ok_):
                        continue
                    j1 = int(np.nanargmax(a1)) if np.isfinite(np.nanmax(a1)) else None
                    if j1 is not None and a1[j1] >= t1:
                        out_.append((r, hs[j1][0], 'S1'))
                        continue
                    j2 = int(np.argmax(a2))
                    if a2[j2] >= t2:
                        out_.append((r, hs[j2][0], 'S2'))
                return out_
            return pick, t2

        def stats(sel):
            if not sel:
                return dict(n=0)
            b, b_e, t, t_e, f, f1 = [], [], [], [], [], []
            for r, i, _k in sel:
                k = (int(min(pop[i], 12)), fsb(nfield[i]))
                b.append(float(place[i] < fav_place[r])); b_e.append(EB.get(k, np.nan))
                t.append(y3[i]); t_e.append(E3.get(k, np.nan)); f.append(fuku[i])
                f1.append(float(fav_place[r] <= 3))
            return dict(n=len(sel), beat=np.mean(b), beat_exp=np.nanmean(b_e), ex_beat=np.nanmean(np.array(b) - np.array(b_e)),
                        top3=np.mean(t), top3_exp=np.nanmean(t_e), ex_t3=np.nanmean(np.array(t) - np.array(t_e)),
                        fuku_roi=np.sum(f) / len(f), fav_top3=np.mean(f1),
                        by_kind={kk: sum(1 for x in sel if x[2] == kk) for kk in ('S1', 'S2')},
                        _rows=[(r, float(place[i] < fav_place[r]) - EB.get((int(min(pop[i], 12)), fsb(nfield[i])), np.nan),
                                y3[i] - E3.get((int(min(pop[i], 12)), fsb(nfield[i])), np.nan)) for r, i, _k in sel])
        clean = lambda st: {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                            for k, v in st.items() if k != '_rows'}
        picks = {m: make_pick(m) for m in ('v2', 'M85', 'M95')}
        for m, (pk, _t2) in picks.items():
            sel = pk(races)
            st = stats(sel)
            win[m] = clean(st)
            win[m]['freq'] = round(len(sel) / max(win['races'], 1), 4)
            win[m]['per_day'] = round(len(sel) / max(win['days'], 1), 2)
            win[m]['win_rate'] = round(float(np.mean([place[i] == 1 for _r, i, _k in sel])), 4) if sel else None
            hits = [fuku[i] for _r, i, _k in sel if place[i] <= 3 and fuku[i] > 0]
            win[m]['fuku_pay_median_hit'] = float(np.median(hits)) if hits else None
            pooled_by[m] += st.get('_rows', [])
        ax = [(r, i) for r, (i, v) in axis.items() if v >= PAID_P and r in fav_place]
        hits = [fuku[i] for _r, i in ax if place[i] <= 3 and fuku[i] > 0]
        win['axis066'] = dict(races=len(ax), per_day=round(len(ax) / max(win['days'], 1), 2),
                              top3=round(float(np.mean([place[i] <= 3 for _r, i in ax])), 4) if ax else None,
                              fuku_roi=round(float(np.sum([fuku[i] for _r, i in ax]) / max(len(ax), 1)), 2),
                              fuku_pay_median_hit=float(np.median(hits)) if hits else None)
        out['windows'][name] = win
        stars.append((name, races, picks, stats, iV))
        print(name, {m: {k: win[m].get(k) for k in ('n', 'freq', 'per_day', 'ex_beat', 'ex_t3')} for m in picks},
              win['axis066'], flush=True)

    rng = np.random.default_rng(0)

    def pool(rows_):
        rows = np.array([(x[1], x[2]) for x in rows_], dtype=float)
        idx_by_r = defaultdict(list)
        for i, x in enumerate(rows_):
            idx_by_r[x[0]].append(i)
        keys = list(idx_by_r)
        bb, bt = [], []
        for _ in range(N_BOOT):
            pick_ = rng.choice(len(keys), len(keys))
            ii = [i for k in pick_ for i in idx_by_r[keys[k]]]
            bb.append(np.nanmean(rows[ii, 0])); bt.append(np.nanmean(rows[ii, 1]))
        q = lambda v: [round(float(np.nanpercentile(v, 2.5)) * 100, 2), round(float(np.nanpercentile(v, 97.5)) * 100, 2)]
        return dict(n=len(rows), ex_beat=round(float(np.nanmean(rows[:, 0])) * 100, 2), ex_beat_ci=q(bb),
                    ex_t3=round(float(np.nanmean(rows[:, 1])) * 100, 2), ex_t3_ci=q(bt))

    out['pooled'] = {m: pool(v) for m, v in pooled_by.items() if v}
    ws = out['windows']
    P = 'M85'
    pos = sum(1 for w in ws.values() if w[P].get('ex_beat', -1) > 0)
    w0 = ws.get('W0', {}).get(P, {}).get('ex_beat', -1) > 0
    passed = bool(out['pooled'][P]['ex_beat_ci'][0] > 0 and pos >= 4 and w0)

    # 対照群A（M85）
    perm_b, perm_t = [], []
    for _ in range(N_PERM):
        allrows = []
        for name, races, picks, stats, iV in stars:
            s1sh = {r: list(rng.permutation([h[1] for h in hs])) for r, hs in races.items()}
            s2sh = {r: list(rng.permutation([h[2] for h in hs])) for r, hs in races.items()}
            allrows += stats(picks[P][0](races, s1sh, s2sh)).get('_rows', [])
        a = np.array([(x[1], x[2]) for x in allrows], dtype=float)
        perm_b.append(np.nanmean(a[:, 0])); perm_t.append(np.nanmean(a[:, 1]))
    cen = lambda v: bool(abs(np.mean(v)) < 2 * np.std(v) / np.sqrt(N_PERM))
    ex_b, ex_t = out['pooled'][P]['ex_beat'] / 100, out['pooled'][P]['ex_t3'] / 100
    out['control_A'] = dict(beat_mean=round(float(np.mean(perm_b)) * 100, 3), beat_sd=round(float(np.std(perm_b)) * 100, 3),
                            beat_p95=round(float(np.percentile(perm_b, 95)) * 100, 2), beat_centered=cen(perm_b),
                            t3_mean=round(float(np.mean(perm_t)) * 100, 3), t3_sd=round(float(np.std(perm_t)) * 100, 3),
                            t3_p95=round(float(np.percentile(perm_t, 95)) * 100, 2), t3_centered=cen(perm_t),
                            beat_exceeds_p95=bool(ex_b > np.percentile(perm_b, 95)),
                            t3_exceeds_p95=bool(ex_t > np.percentile(perm_t, 95)),
                            beat_net=round((ex_b - float(np.mean(perm_b))) * 100, 2),
                            t3_net=round((ex_t - float(np.mean(perm_t))) * 100, 2))
    # 対照群B（M85・3着内だった4〜9番人気の馬の半分の S2 に τ2 を足す）
    allrows = []
    for name, races, picks, stats, iV in stars:
        pk, t2 = picks[P]
        s2inj = {r: [h[2] + (t2 if (y3[h[0]] == 1 and rng.random() < 0.5) else 0.0) for h in hs] for r, hs in races.items()}
        s1none = {r: [np.nan] * len(hs) for r, hs in races.items()}
        allrows += stats(pk(races, s1none, s2inj)).get('_rows', [])
    a = np.array([(x[1], x[2]) for x in allrows], dtype=float)
    out['control_B'] = dict(injected_ex_t3=round(float(np.nanmean(a[:, 1])) * 100, 2), original_ex_t3=round(ex_t * 100, 2),
                            detects=bool(np.nanmean(a[:, 1]) - ex_t >= 0.05))
    out['criteria'] = dict(rule=P, pos_windows=pos, w0_pos=bool(w0), passed=passed)
    out['verdict'] = '合格' if passed else '不合格'
    if os.environ.get('STAR_NARROW_SMOKE'):
        print('SMOKE OK', out['verdict'], list(out['pooled']))
        return
    with open('hitrate/star_narrow_result.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
