#!/usr/bin/env python3
"""候補C の測定ハーネス（事前登録カード `hitrate/CRITERIA_C.md` のとおり）。

問い: 1番人気以外の馬が1番人気を上回る「根拠」で◎を差し替えると、
      的中率と回収率はどうなるか。

腕は2つ:
  C-1  市場アンカーの部分的な弱化      score_λ = λ·base_margin + 木の出力
  C-2  「1番人気を超えるか」の直接判定  y = 1{着順(X) < 着順(P1)} を較正して τ で絞る

🔴 **測定は1回。** 閾値・腕・期間・母集団・指標はカードのとおりで、結果を見てから変えない。
🔴 中止条件は的中率・回収率を読む**手前**で印字して判定する。
🔴 隔離ワークスペースで実行する（本番 data/ に1バイトも書かない）。

使い方:
    cd <隔離ワークスペース>      # data/horse_features.csv と data/history.db がある場所
    python3 hitrate/eval_c.py
"""
import importlib.util
import json
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

from src.tools.train_xgb import (  # noqa: E402
    _EXCLUDE_COLS, _MARKET_FEAT_COLS, _apply_popularity_drift,
    _popularity_to_base_margin, load_popularity_drift,
)

# ── logic2026 のハーネスを再利用する（自前で作り直さない・2026-08-08 の教訓）──
_spec = importlib.util.spec_from_file_location(
    'logic2026_eval', os.path.join(os.getcwd(), 'logic2026', 'eval.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
roi, pick_axis = L.roi, L.pick_axis

# ── カード §4 の窓（2025年の walk-forward 3分割）──
WINDOWS = [('W1', '2024-12-31', '2025-01-01', '2025-04-30'),
           ('W2', '2025-04-30', '2025-05-01', '2025-08-31'),
           ('W3', '2025-08-31', '2025-09-01', '2025-12-31')]
SEEDS = (42, 7, 2026)
INNER_DAYS = 42
N_ROUNDS, ES_ROUNDS = 4000, 50
DRIFT_SEED_TRAIN, DRIFT_SEED_EVAL = 11, 12

LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.0)        # カード §4 C-1
TAUS = (0.55, 0.60, 0.65, 0.70)                 # カード §4 C-2
N_PERM = 200                                    # カード §10 対照群A
INJECT_SIGMA = 1.8                              # カード §10 対照群B（理論 AUC 0.6528）
INJECT_SEED = 20261110

# 本番と同じハイパーパラメータ（retrain/ab.py の N 腕）
PROD_HP = dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0)

# 停止条件7の運用定義（カード §9-7 を実行可能な形にしたもの。結果を見てから変えない）
#   2026-07-30 の λ=0 は「大穴ばかり選ぶ」だった。再現の線は
#     λ=0 の ◎ の平均市場人気 >= 4.0  かつ  複勝率が λ=1 比で −10.0pt 以下
STOP7_MEAN_POP = 4.0
STOP7_RATE_DROP = -10.0


def feat_cols(df):
    ex = set(_EXCLUDE_COLS) | {'date_obj', '_n_horses'} | set(_MARKET_FEAT_COLS)
    return [c for c in df.columns if c not in ex and df[c].dtype.kind in 'fi']


def drifted_pop(df, drift, seed):
    """本番と同じ「朝オッズ相当」の人気順位（確定人気にドリフトを注入したもの）。"""
    n = df.groupby('race_id')['horse_num'].transform('count')
    pop = df['f_popularity'].fillna(n / 2)
    if drift is not None:
        pop = pd.Series(_apply_popularity_drift(
            pop, df['race_id'].values, drift, seed=seed))
    return np.asarray(pop, dtype=float), np.asarray(n, dtype=float)


def inner_split(d, train_end):
    """内側ホールドアウト42日。

    🔴 起点は「学習期間に**実在する最後の日**」から42日前に取る。
    `history.db` には 2024-11-25〜2024-12-31 のレースが1件も無いので、
    train_end から単純に日数を引くと穴に当たる（候補Aで実際に踏んだ）。
    """
    te = pd.Timestamp(train_end)
    last_tr = pd.Timestamp(d[d <= np.datetime64(te)].max())
    inner_from = last_tr - pd.Timedelta(days=INNER_DAYS)
    m_fit = d <= np.datetime64(inner_from)
    m_in = (d > np.datetime64(inner_from)) & (d <= np.datetime64(te))
    return m_fit, m_in, inner_from


def fit_residual(Xf, yf, bmf, Xi, yi, bmi, Xv, bmv, cols, seed):
    """本番と同じ残差学習モデルを1本学習して、評価窓の margin を返す。"""
    spw = round((1 - yf.mean()) / max(yf.mean(), 0.01), 2)
    params = dict(PROD_HP, objective='binary:logistic', eval_metric='logloss',
                  scale_pos_weight=spw, seed=seed, nthread=-1)
    dtr = xgb.DMatrix(Xf, label=yf, feature_names=cols); dtr.set_base_margin(bmf)
    din = xgb.DMatrix(Xi, label=yi, feature_names=cols); din.set_base_margin(bmi)
    dva = xgb.DMatrix(Xv, feature_names=cols); dva.set_base_margin(bmv)
    bst = xgb.train(params, dtr, num_boost_round=N_ROUNDS, evals=[(din, 'inner_ho')],
                    early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    ntree = bst.num_boosted_rounds()
    # 🔴 木の本数は版の既定に任せない（カード §6・本番 engine.py と同じ全木）
    m_all = bst.predict(dva, output_margin=True, iteration_range=(0, ntree))
    m_best = bst.predict(dva, output_margin=True,
                         iteration_range=(0, int(bst.best_iteration) + 1))
    return dict(it=int(bst.best_iteration), ntree=ntree, spw=float(spw),
                m_all=m_all, m_best=m_best)


# ── C-2 ペアワイズ ────────────────────────────────────────────────────────
def build_pairs(idx, X, place, race_ids, horse_nums, pop):
    """P1（人気1位）を基準にした差分データを作る。

    返り値: dX, y, race, num, p1num, n_bad_race
    🔴 特徴量は**129特徴量の差分だけ**。オッズ・人気は1列も入れない（North Star #1）。
       市場は「誰が P1 か」という土台としてだけ入る（base_margin と同じ役回り）。
    """
    dXs, ys, rs, ns, p1s = [], [], [], [], []
    bad = 0
    order = np.argsort(race_ids[idx], kind='stable')
    ridx = idx[order]
    rid = race_ids[ridx]
    bounds = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = ridx[a:b]
        p1 = rows[pop[rows] == 1.0]
        if len(p1) != 1:                      # 人気1位が特定できない（停止条件5）
            bad += 1
            continue
        p1 = p1[0]
        if not np.isfinite(place[p1]) or place[p1] <= 0:
            bad += 1
            continue
        others = rows[rows != p1]
        ok = np.isfinite(place[others]) & (place[others] > 0)
        others = others[ok]
        if len(others) == 0:
            bad += 1
            continue
        dXs.append(X[others] - X[p1])
        ys.append((place[others] < place[p1]).astype(np.int8))
        rs.append(race_ids[others])
        ns.append(horse_nums[others])
        p1s.append(np.full(len(others), horse_nums[p1]))
    if not dXs:
        return (np.zeros((0, X.shape[1]), np.float32), np.zeros(0, np.int8),
                np.zeros(0, object), np.zeros(0), np.zeros(0), bad)
    return (np.vstack(dXs).astype(np.float32), np.concatenate(ys),
            np.concatenate(rs), np.concatenate(ns), np.concatenate(p1s), bad)


def fit_pair(dXf, yf, dXi, yi, dXv, cols, seed):
    params = dict(PROD_HP, objective='binary:logistic', eval_metric='logloss',
                  scale_pos_weight=1.0, seed=seed, nthread=-1)
    dtr = xgb.DMatrix(dXf, label=yf, feature_names=cols)
    din = xgb.DMatrix(dXi, label=yi, feature_names=cols)
    dva = xgb.DMatrix(dXv, feature_names=cols)
    bst = xgb.train(params, dtr, num_boost_round=N_ROUNDS, evals=[(din, 'inner_ho')],
                    early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    nt = bst.num_boosted_rounds()
    raw_i = bst.predict(din, iteration_range=(0, nt))
    raw_v = bst.predict(dva, iteration_range=(0, nt))
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_i, yi)
    return dict(it=int(bst.best_iteration), ntree=nt,
                p_in=iso.predict(raw_i), p_va=iso.predict(raw_v),
                raw_in=raw_i, raw_va=raw_v)


def main():
    base = os.getcwd()
    out = {'windows': [], 'c1': {}, 'c2': {}, 'stop': [], 'control': {}}

    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = feat_cols(df)
    drift = load_popularity_drift(base)

    X = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    y = df['is_fukusho'].to_numpy(dtype=np.int8)
    d = df['date_obj'].to_numpy()
    race_ids = df['race_id'].to_numpy()
    horse_nums = df['horse_num'].to_numpy()

    pop_tr, nh = drifted_pop(df, drift, DRIFT_SEED_TRAIN)
    pop_ev, _ = drifted_pop(df, drift, DRIFT_SEED_EVAL)
    bm_tr = np.asarray(_popularity_to_base_margin(pd.Series(pop_tr),
                                                  pd.Series(nh)), dtype=float)
    bm_ev = np.asarray(_popularity_to_base_margin(pd.Series(pop_ev),
                                                  pd.Series(nh)), dtype=float)

    # 着順と払戻（評価は 2025 年だけ）
    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= '2023-01-01' AND r.date <= '2026-12-31'""", con)
    con.close()
    pmap = {(r, n): p for r, n, p in
            zip(pay['race_id'], pay['horse_num'], pay['place'])}
    place = np.array([pmap.get((r, n), np.nan) for r, n in zip(race_ids, horse_nums)],
                     dtype=float)
    pay_eval = pay[['race_id', 'horse_num', 'place', 'fukusho_payout', 'tansho_payout']]

    # ── 窓を切る ───────────────────────────────────────────────────────
    masks = {}
    for name, te, v0, v1 in WINDOWS:
        m_fit, m_in, inner_from = inner_split(d, te)
        m_va = (d >= np.datetime64(pd.Timestamp(v0))) & (d <= np.datetime64(pd.Timestamp(v1)))
        masks[name] = (m_fit, m_in, m_va)
        out['windows'].append(dict(
            window=name, train_end=te,
            inner_from=str((inner_from + pd.Timedelta(days=1)).date()),
            eval_from=v0, eval_to=v1, n_fit=int(m_fit.sum()),
            n_inner=int(m_in.sum()), n_eval=int(m_va.sum()),
            n_eval_races=int(pd.unique(race_ids[m_va]).size)))

    # ── 学習（印字はまだしない）────────────────────────────────────────
    rng = np.random.default_rng(INJECT_SEED)
    fits, pfits, pairs = {}, {}, {}
    for name, te, v0, v1 in WINDOWS:
        m_fit, m_in, m_va = masks[name]
        for sd in SEEDS:
            fits[(name, sd)] = fit_residual(
                X[m_fit], y[m_fit], bm_tr[m_fit], X[m_in], y[m_in], bm_tr[m_in],
                X[m_va], bm_ev[m_va], cols, sd)
            print(f'  …C-1 fit {name} seed={sd} it={fits[(name, sd)]["it"]} '
                  f'ntree={fits[(name, sd)]["ntree"]}', flush=True)

        # ペアワイズ（学習・内側HOは train ドリフト、評価は eval ドリフト）
        pf = build_pairs(np.flatnonzero(m_fit), X, place, race_ids, horse_nums, pop_tr)
        pi = build_pairs(np.flatnonzero(m_in), X, place, race_ids, horse_nums, pop_tr)
        pv = build_pairs(np.flatnonzero(m_va), X, place, race_ids, horse_nums, pop_ev)
        pairs[name] = (pf, pi, pv)
        # 対照群B: 既知の信号 z を 1列足す（y と相関・σ=1.8）
        zf = pf[1] + rng.normal(0, INJECT_SIGMA, len(pf[1]))
        zi = pi[1] + rng.normal(0, INJECT_SIGMA, len(pi[1]))
        zv = pv[1] + rng.normal(0, INJECT_SIGMA, len(pv[1]))
        colsz = cols + ['z_inject']
        for sd in SEEDS:
            pfits[(name, sd, 'plain')] = fit_pair(pf[0], pf[1], pi[0], pi[1],
                                                  pv[0], cols, sd)
            pfits[(name, sd, 'inject')] = fit_pair(
                np.column_stack([pf[0], zf]).astype(np.float32), pf[1],
                np.column_stack([pi[0], zi]).astype(np.float32), pi[1],
                np.column_stack([pv[0], zv]).astype(np.float32), colsz, sd)
            print(f'  …C-2 fit {name} seed={sd} it={pfits[(name, sd, "plain")]["it"]} '
                  f'pairs fit/in/va = {len(pf[1])},{len(pi[1])},{len(pv[1])}', flush=True)

    # ── C-1 の集計（λごと・窓ごと・シード平均）───────────────────────────
    def c1_metrics(name, lam):
        m_fit, m_in, m_va = masks[name]
        va = df.loc[m_va, ['race_id', 'horse_num']]
        res = []
        for sd in SEEDS:
            tree = fits[(name, sd)]['m_all'] - bm_ev[m_va]
            score = lam * bm_ev[m_va] + tree
            ax = pick_axis(va, score)
            r_f = roi(ax, pay_eval, 'fukusho')
            r_t = roi(ax, pay_eval, 'tansho')
            # ◎ の市場人気（評価ドリフト後）と、1番人気と違う率
            key = {(r, n): i for i, (r, n) in
                   enumerate(zip(race_ids[m_va], horse_nums[m_va]))}
            ii = np.array([key[(r, n)] for r, n in zip(ax['race_id'], ax['horse_num'])])
            axpop = pop_ev[m_va][ii]
            res.append(dict(rate=r_f['rate'], roi_f=r_f['roi'], roi_t=r_t['roi'],
                            n=r_f['n'], auc=roc_auc_score(y[m_va], score),
                            mean_pop=float(np.nanmean(axpop)),
                            not_pop1=float(np.mean(axpop != 1.0)) * 100))
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != 'n'}
        agg['n'] = res[0]['n']
        return agg

    c1 = {lam: {w: c1_metrics(w, lam) for w, _, _, _ in WINDOWS} for lam in LAMBDAS}
    for lam in LAMBDAS:
        ws = [c1[lam][w] for w, _, _, _ in WINDOWS]
        c1[lam]['POOL'] = {k: float(np.mean([x[k] for x in ws]))
                           for k in ws[0] if k != 'n'}
        c1[lam]['POOL']['n'] = int(sum(x['n'] for x in ws))
    out['c1'] = {str(k): {w: v for w, v in d_.items()} for k, d_ in c1.items()}

    # ── 中止条件（的中率・回収率を読む手前）─────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §9）— C-1/C-2 の成績を読む手前で判定する')
    print('=' * 78)
    stop = []

    n_races = [w['n_eval_races'] for w in out['windows']]
    pair_per_race = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        nr = max(pd.unique(pv[2]).size, 1)
        pair_per_race.append(len(pv[1]) / nr)
    ok1 = sum(n_races) >= 3000 and min(pair_per_race) >= 10.0
    print(f'  1. 評価レース数 = {n_races} 計 {sum(n_races):,}（>= 3,000 か）／'
          f'1レースあたりペア = {[round(x, 1) for x in pair_per_race]}（>= 10 か）'
          f'  → {ok1}')
    if not ok1:
        stop.append(f'母集団が小さい: races={sum(n_races)} pair/race={pair_per_race}')

    swap_counts = {}
    for tau in TAUS:
        c = 0
        for name, _, _, _ in WINDOWS:
            pv = pairs[name][2]
            for sd in SEEDS:
                p = pfits[(name, sd, 'plain')]['p_va']
                dd = pd.DataFrame({'r': pv[2], 'p': p})
                mx = dd.groupby('r')['p'].max()
                c += int((mx >= tau).sum())
        swap_counts[tau] = c / len(SEEDS)
    ok2 = any(v >= 100 for v in swap_counts.values())
    print(f'  2. 差し替えレース数（シード平均）= '
          f'{ {k: round(v, 1) for k, v in swap_counts.items()} }  → どれか >= 100 か: {ok2}')
    if not ok2:
        stop.append(f'どの τ でも差し替えレースが100未満: {swap_counts}')

    ok3 = True
    for w in out['windows']:
        thin = w['n_inner'] < 1000
        ok3 = ok3 and not thin
        print(f'  3. {w["window"]} 学習 {w["n_fit"]:,} / 内側HO {w["n_inner"]:,}'
              f'（起点 {w["inner_from"]}）/ 評価 {w["n_eval"]:,}'
              f'  1000行未満: {thin}')
    if not ok3:
        stop.append('内側HOが1000行未満（2024-12 の穴に当たっている）')
    for name, _, _, _ in WINDOWS:
        m_fit, m_in, m_va = masks[name]
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
    print('  3b. 学習・内側HO と評価窓の重なり: なし（assert 済み）')

    # 4. 較正（基準1）— これが落ちたら以降は読まない
    cal = {}
    for tau in TAUS:
        rows = []
        for name, _, _, _ in WINDOWS:
            pi, pv = pairs[name][1], pairs[name][2]
            for sd in SEEDS:
                f = pfits[(name, sd, 'plain')]
                for lab, p, yy in (('inner', f['p_in'], pi[1]), ('eval', f['p_va'], pv[1])):
                    sel = p >= tau
                    rows.append((lab, int(sel.sum()),
                                 float(yy[sel].mean()) if sel.any() else np.nan))
        for lab in ('inner', 'eval'):
            sub = [(n_, r) for l_, n_, r in rows if l_ == lab and n_ > 0]
            tot = sum(n_ for n_, _ in sub)
            obs = (sum(n_ * r for n_, r in sub) / tot) if tot else np.nan
            cal[(tau, lab)] = (tot, obs)
    ok4 = False
    for tau in TAUS:
        a, b = cal[(tau, 'inner')], cal[(tau, 'eval')]
        good = (a[0] > 0 and b[0] > 0 and a[1] >= tau - 0.05 and b[1] >= tau - 0.05)
        ok4 = ok4 or good
        print(f'  4. τ={tau}: 内側HO n={a[0]:,} 実測勝ち越し率={a[1]:.4f} / '
              f'評価 n={b[0]:,} 実測={b[1]:.4f}  → 両方 >= {tau - 0.05:.2f} か: {good}')
    if not ok4:
        stop.append('どの τ でも較正が基準1を満たさない（P̂>=τ の群が τ−0.05 に届かない）')

    # 5. P1 が特定できないレース
    bad_rate = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        nr = pd.unique(race_ids[masks[name][2]]).size
        bad_rate.append(pv[5] / max(nr, 1) * 100)
    ok5 = max(bad_rate) <= 5.0
    print(f'  5. P1 を特定できないレースの割合 = {[round(x, 2) for x in bad_rate]}%'
          f'  → 5% 以下か: {ok5}')
    if not ok5:
        stop.append(f'P1 を特定できないレースが5%超: {bad_rate}')

    # 6. 対照群B（注入の復元）
    inj = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        for sd in SEEDS:
            a = roc_auc_score(pv[1], pfits[(name, sd, 'plain')]['raw_va'])
            b = roc_auc_score(pv[1], pfits[(name, sd, 'inject')]['raw_va'])
            inj.append((a, b))
    z_alone = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        zz = pv[1] + np.random.default_rng(INJECT_SEED).normal(0, INJECT_SIGMA, len(pv[1]))
        z_alone.append(roc_auc_score(pv[1], zz))
    d_obs = float(np.mean([b - a for a, b in inj]))
    d_exp = float(np.mean(z_alone)) - float(np.mean([a for a, _ in inj]))
    ok6 = abs(d_obs - d_exp) <= 0.01
    print(f'  6. 対照群B: 注入後 − 注入前 = {d_obs:+.4f}  /  注入量 = {d_exp:+.4f}'
          f'  （z 単独 AUC = {np.mean(z_alone):.4f}・理論 0.6528）'
          f'  → ±0.01 で一致か: {ok6}')
    out['control']['B'] = dict(d_obs=d_obs, d_exp=d_exp,
                               z_alone=float(np.mean(z_alone)), ok=bool(ok6))
    if not ok6:
        stop.append(f'対照群Bが復元しない: 実測 {d_obs:+.4f} vs 注入量 {d_exp:+.4f}')

    # 7. λ=0 が 2026-07-30 を再現するか
    p0, p1_ = c1[0.0]['POOL'], c1[1.0]['POOL']
    ok7 = (p0['mean_pop'] >= STOP7_MEAN_POP
           and (p0['rate'] - p1_['rate']) <= STOP7_RATE_DROP)
    print(f'  7. λ=0 の ◎ 平均人気 = {p0["mean_pop"]:.2f}（>= {STOP7_MEAN_POP} か）／'
          f'複勝率 {p1_["rate"]:.2f}% → {p0["rate"]:.2f}%'
          f'（{p0["rate"] - p1_["rate"]:+.2f}pt・<= {STOP7_RATE_DROP} か）  → {ok7}')
    if not ok7:
        stop.append(f'λ=0 が 2026-07-30 を再現しない: mean_pop={p0["mean_pop"]:.2f} '
                    f'Δrate={p0["rate"] - p1_["rate"]:+.2f}pt')

    out['stop'] = stop
    print('-' * 78)
    print('中止条件: ' + ('発火なし（続行）' if not stop else '発火 → ' + ' / '.join(stop)))
    print('-' * 78)

    # ── C-1 の結果 ─────────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('■ C-1 市場アンカーの部分的な弱化（λスイープ・プール）')
    print('=' * 78)
    print(f'{"λ":>5} {"◎≠人気1位":>10} {"◎平均人気":>9} {"複勝率":>8} {"複勝回収":>9} '
          f'{"単勝回収":>9} {"軸AUC":>8}')
    for lam in LAMBDAS:
        p = c1[lam]['POOL']
        print(f'{lam:5.1f} {p["not_pop1"]:9.2f}% {p["mean_pop"]:9.2f} '
              f'{p["rate"]:7.2f}% {p["roi_f"]:8.2f}% {p["roi_t"]:8.2f}% {p["auc"]:8.4f}')
    print()
    print('窓ごと（複勝率 / 複勝回収率 / ◎≠人気1位）:')
    for lam in LAMBDAS:
        s = '  λ=%.1f  ' % lam
        for w, _, _, _ in WINDOWS:
            q = c1[lam][w]
            s += f'{w} {q["rate"]:5.1f}%/{q["roi_f"]:6.1f}%/{q["not_pop1"]:4.1f}%  '
        print(s)

    base_ = c1[1.0]['POOL']
    print()
    print('C-1 合格基準（カード §7）:')
    c1_pass = []
    for lam in LAMBDAS:
        if lam == 1.0:
            continue
        p = c1[lam]['POOL']
        k1 = p['not_pop1'] >= 10.0
        k2 = (p['rate'] - base_['rate']) >= -1.0
        k3 = (p['roi_f'] - base_['roi_f']) >= -2.0
        k5 = (p['roi_t'] - base_['roi_t']) >= -3.0
        k4 = all(
            (c1[lam][w]['not_pop1'] >= 10.0)
            and (c1[lam][w]['rate'] - c1[1.0][w]['rate'] >= -1.0)
            and (c1[lam][w]['roi_f'] - c1[1.0][w]['roi_f'] >= -2.0)
            for w, _, _, _ in WINDOWS)
        allok = k1 and k2 and k3 and k4 and k5
        c1_pass.append((lam, allok))
        print(f'  λ={lam:.1f}: ①≠率>=10%: {k1} ②複勝率−1.0pt以内: {k2} '
              f'③複勝回収−2.0pt以内: {k3} ④3窓同符号: {k4} ⑤単勝回収−3.0pt以内: {k5} '
              f'→ {"合格" if allok else "不合格"}')
    print('  C-1 判定: ' + ('非劣化域あり → ' + str([l for l, o in c1_pass if o])
                           if any(o for _, o in c1_pass) else '**不採用**（非劣化域なし）'))

    # ── C-2 の結果 ─────────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('■ C-2 「1番人気を超えるか」の直接判定（較正確率 >= τ で差し替え）')
    print('=' * 78)
    c2 = {}
    for tau in TAUS:
        acc = {'n_sw': [], 'rate_sw': [], 'rate_p1': [], 'roi_sw': [], 'roi_p1': [],
               'roif_sw': [], 'roif_p1': [], 'perm': []}
        per_w = {}
        for name, _, _, _ in WINDOWS:
            pv = pairs[name][2]
            wres = []
            for sd in SEEDS:
                p = pfits[(name, sd, 'plain')]['p_va']
                dd = pd.DataFrame({'r': pv[2], 'n': pv[3], 'p1': pv[4], 'p': p})
                dd = dd.sort_values(['r', 'p', 'n'], ascending=[True, False, True])
                top = dd.groupby('r', as_index=False).first()
                sw = top[top['p'] >= tau]
                if len(sw) == 0:
                    wres.append(None)
                    continue
                ax_sw = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                ax_p1 = sw[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
                r_sw = roi(ax_sw, pay_eval, 'fukusho')
                r_p1 = roi(ax_p1, pay_eval, 'fukusho')
                t_sw = roi(ax_sw, pay_eval, 'tansho')
                t_p1 = roi(ax_p1, pay_eval, 'tansho')
                wres.append(dict(n=len(sw), rate_sw=r_sw['rate'], rate_p1=r_p1['rate'],
                                 roif_sw=r_sw['roi'], roif_p1=r_p1['roi'],
                                 roi_sw=t_sw['roi'], roi_p1=t_p1['roi']))
                # 対照群A: ペアの符号反転で0中心にする
                m_sw = ax_sw.merge(pay_eval, on=['race_id', 'horse_num'], how='left')
                m_p1 = ax_p1.merge(pay_eval, on=['race_id', 'horse_num'], how='left')
                delta = (m_sw['place'].between(1, 3).astype(float).values
                         - m_p1['place'].between(1, 3).astype(float).values)
                g = np.random.default_rng(INJECT_SEED + sd)
                perm = [float(np.mean(delta * g.choice([-1.0, 1.0], len(delta))))
                        for _ in range(N_PERM)]
                acc['perm'].append((float(np.mean(delta)), float(np.mean(perm)),
                                    float(np.std(perm)), len(delta)))
            ok_w = [w for w in wres if w]
            per_w[name] = ({k: float(np.mean([w[k] for w in ok_w])) for k in ok_w[0]}
                           if ok_w else None)
        poolk = [k for k in ('n', 'rate_sw', 'rate_p1', 'roif_sw', 'roif_p1',
                             'roi_sw', 'roi_p1')]
        okw = [per_w[w] for w, _, _, _ in WINDOWS if per_w[w]]
        c2[tau] = dict(pool=({k: float(np.mean([w[k] for w in okw])) for k in poolk}
                             if okw else None),
                       n_total=(sum(w['n'] for w in okw) if okw else 0),
                       per_w=per_w, perm=acc['perm'])

    print(f'{"τ":>5} {"差し替え":>8} {"複勝率(差替)":>12} {"複勝率(P1)":>11} '
          f'{"複勝回収(差替)":>13} {"複勝回収(P1)":>12}')
    for tau in TAUS:
        c = c2[tau]
        if not c['pool']:
            print(f'{tau:5.2f}  該当なし')
            continue
        p = c['pool']
        print(f'{tau:5.2f} {c["n_total"]:8.0f} {p["rate_sw"]:11.2f}% {p["rate_p1"]:10.2f}% '
              f'{p["roif_sw"]:12.2f}% {p["roif_p1"]:11.2f}%')

    print()
    print('C-2 合格基準（カード §8）:')
    any_pass = False
    for tau in TAUS:
        c = c2[tau]
        if not c['pool']:
            print(f'  τ={tau:.2f}: 該当レースなし → 不合格')
            continue
        p = c['pool']
        a, b = cal[(tau, 'inner')], cal[(tau, 'eval')]
        k1 = (a[0] > 0 and b[0] > 0 and a[1] >= tau - 0.05 and b[1] >= tau - 0.05)
        tot_races = sum(w['n_eval_races'] for w in out['windows'])
        k2 = c['n_total'] >= 100 and (c['n_total'] / tot_races * 100) >= 3.0
        k3 = (p['rate_sw'] - p['rate_p1']) >= 2.0
        k4 = (p['roif_sw'] - p['roif_p1']) >= 0.0
        k5 = all((c['per_w'][w] and c['per_w'][w]['rate_sw'] - c['per_w'][w]['rate_p1'] > 0
                  and c['per_w'][w]['roif_sw'] - c['per_w'][w]['roif_p1'] >= 0)
                 for w, _, _, _ in WINDOWS)
        pm = c['perm']
        se = float(np.mean([s / np.sqrt(N_PERM) for _, _, s, _ in pm])) if pm else np.nan
        nullmean = float(np.mean([m for _, m, _, _ in pm])) if pm else np.nan
        k6a = abs(nullmean) < 2 * se if pm else False
        k6 = k6a and ok6
        print(f'  τ={tau:.2f}: ①較正: {k1} ②発生率({c["n_total"]/tot_races*100:.2f}%・'
              f'{c["n_total"]:.0f}レース): {k2} ③的中+2.0pt以上'
              f'({p["rate_sw"] - p["rate_p1"]:+.2f}pt): {k3} ④回収率マイナスにしない'
              f'({p["roif_sw"] - p["roif_p1"]:+.2f}pt): {k4} ⑤3窓同符号: {k5} '
              f'⑥対照群: {k6} → {"合格" if all([k1, k2, k3, k4, k5, k6]) else "不合格"}')
        any_pass = any_pass or all([k1, k2, k3, k4, k5, k6])
        print(f'       対照群A（符号反転・{N_PERM}回）: 帰無平均 {nullmean:+.5f} / '
              f'2*SE {2 * se:.5f} → 0中心か: {k6a}  '
              f'実測 mean(Δ) = {np.mean([o for o, _, _, _ in pm]):+.5f}')

    print()
    print('=' * 78)
    print('■ 判定')
    print('=' * 78)
    print('  C-1: ' + ('非劣化域あり' if any(o for _, o in c1_pass) else '不採用'))
    print('  C-2: ' + ('合格（採用提案をユーザー判断に回す）' if any_pass else '不採用'))
    if stop:
        print('  ⚠ 中止条件が発火している: ' + ' / '.join(stop))

    out['c2'] = {str(k): dict(pool=v['pool'], n_total=v['n_total'],
                              per_w=v['per_w']) for k, v in c2.items()}
    out['cal'] = {f'{t}_{l}': cal[(t, l)] for t, l in cal}
    with open(os.path.join(base, 'hitrate', 'c_result.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n→ hitrate/c_result.json に保存した')


if __name__ == '__main__':
    main()
