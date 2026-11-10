#!/usr/bin/env python3
"""STAR の測定ハーネス（事前登録カード `hitrate/CRITERIA_STAR.md` のとおり）。

問い: ☆（1番人気を着順で上回る可能性が比較的高い馬の表示）を本番に出してよいか。
判定窓は **2024年（この統計量では未読）**。2026年は矛盾しないかの確認だけに使う。

🔴 ◎については合否を出さない（カード §6 の報告だけ）。
🔴 集計は1つに固定する（カード §3）: シードごとに3窓をプール → 3シードを単純平均。
   0件のシードも分母に数える。
🔴 中止条件は結果を読む手前で印字する。測定は1回。隔離ワークスペースで実行する。

使い方:
    cd <隔離ワークスペース>
    python3 hitrate/eval_star.py
"""
import importlib.util
import json
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

_s = importlib.util.spec_from_file_location(
    'eval_c', os.path.join(os.getcwd(), 'hitrate', 'eval_c.py'))
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

_s2 = importlib.util.spec_from_file_location(
    'logic2026_eval', os.path.join(os.getcwd(), 'logic2026', 'eval.py'))
L = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(L)
roi = L.roi

from src.tools.train_xgb import load_popularity_drift  # noqa: E402

# カード §2 の判定窓（2024年・未読）
JUDGE = [('U1', '2023-12-31', '2024-01-01', '2024-04-30'),
         ('U2', '2024-04-30', '2024-05-01', '2024-08-31'),
         ('U3', '2024-08-31', '2024-09-01', '2024-12-31')]
# 既読の期間（基準4の頻度・基準7の矛盾確認・中止条件6の再現確認にだけ使う）
Y2025 = C.WINDOWS
Y2026 = [('V1', '2025-12-31', '2026-01-01', '2026-04-30'),
         ('V2', '2026-04-30', '2026-05-01', '2026-08-31'),
         ('V3', '2026-08-31', '2026-09-01', '2026-11-08')]

SEEDS = C.SEEDS
TAU = 0.60                 # 腕 S-1
TOP_FRAC = 0.040           # 腕 S-2（2025年の τ=0.60 の頻度 3.61% に合わせた値）
N_PERM, PERM_SEED = 200, 20261110
BONUS, BIG = 5.0, 1000.0


def load():
    base = os.getcwd()
    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = C.feat_cols(df)
    drift = load_popularity_drift(base)
    X = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    d = df['date_obj'].to_numpy()
    rid = df['race_id'].to_numpy()
    hn = df['horse_num'].to_numpy()
    pop_tr, _ = C.drifted_pop(df, drift, C.DRIFT_SEED_TRAIN)
    pop_ev, _ = C.drifted_pop(df, drift, C.DRIFT_SEED_EVAL)
    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= '2022-01-01' AND r.date <= '2026-12-31'""", con)
    con.close()
    pm = {(r, n): p for r, n, p in zip(pay['race_id'], pay['horse_num'], pay['place'])}
    place = np.array([pm.get((r, n), np.nan) for r, n in zip(rid, hn)], dtype=float)
    return (df, cols, X, d, rid, hn, pop_tr, pop_ev, place,
            pay[['race_id', 'horse_num', 'place', 'fukusho_payout', 'tansho_payout']])


def fit_period(wins, cols, X, d, rid, hn, pop_tr, pop_ev, place, tag):
    masks, pairs, fits = {}, {}, {}
    for name, te, v0, v1 in wins:
        m_fit, m_in, inf_ = C.inner_split(d, te)
        m_va = ((d >= np.datetime64(pd.Timestamp(v0)))
                & (d <= np.datetime64(pd.Timestamp(v1))))
        masks[name] = (m_fit, m_in, m_va, inf_)
        pf = C.build_pairs(np.flatnonzero(m_fit), X, place, rid, hn, pop_tr)
        pi = C.build_pairs(np.flatnonzero(m_in), X, place, rid, hn, pop_tr)
        pv = C.build_pairs(np.flatnonzero(m_va), X, place, rid, hn, pop_ev)
        pairs[name] = (pf, pi, pv)
        for sd in SEEDS:
            fits[(name, sd)] = C.fit_pair(pf[0], pf[1], pi[0], pi[1], pv[0], cols, sd)
            print(f'  …{tag} fit {name} seed={sd} it={fits[(name, sd)]["it"]} '
                  f'pairs {len(pf[1])},{len(pi[1])},{len(pv[1])}', flush=True)
    return masks, pairs, fits


def picks(pairs, fits, name, sd):
    """レースごとに P̂ 最大の1頭（＝☆の候補）。"""
    pv = pairs[name][2]
    dd = pd.DataFrame({'r': pv[2], 'n': pv[3], 'p1': pv[4], 'y': pv[1],
                       'p': fits[(name, sd)]['p_va']})
    return (dd.sort_values(['r', 'p', 'n'], ascending=[True, False, True])
              .groupby('r', as_index=False).first())


def select(tp, arm):
    """腕ごとの ☆ 集合。S-1 は τ 固定、S-2 は P̂ 上位 TOP_FRAC。"""
    if arm == 'S1':
        return tp[tp['p'] >= TAU]
    k = max(int(round(len(tp) * TOP_FRAC)), 1)
    return tp.nlargest(k, 'p')


def agg(wins, pairs, fits, pay, arm, n_races_total):
    """カード §3 の集計: シードごとに3窓をプール → 3シードを単純平均。"""
    per_seed, per_w = [], {w: [] for w, _, _, _ in wins}
    for sd in SEEDS:
        sel = []
        for name, _, _, _ in wins:
            tp = picks(pairs, fits, name, sd)
            s = select(tp, arm)
            s = s.assign(_w=name)
            sel.append(s)
            per_w[name].append(s)
        allsel = pd.concat(sel) if sel else pd.DataFrame()
        per_seed.append(allsel)

    def stats(frames):
        rows = []
        for f in frames:
            if len(f) == 0:
                rows.append(dict(n=0, decl=np.nan, obs=np.nan, rate_sw=np.nan,
                                 rate_p1=np.nan, roif_sw=np.nan, roif_p1=np.nan,
                                 cut3_sw=np.nan, cut3_p1=np.nan))
                continue
            ax = f[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
            ap = f[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
            rs, rp = roi(ax, pay, 'fukusho'), roi(ap, pay, 'fukusho')
            rows.append(dict(n=len(f), decl=float(f['p'].median()),
                             obs=float(f['y'].mean()),
                             rate_sw=rs['rate'], rate_p1=rp['rate'],
                             roif_sw=rs['roi'], roif_p1=rp['roi'],
                             cut3_sw=rs['cut3'], cut3_p1=rp['cut3']))
        keys = rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}

    pool = stats(per_seed)
    pool['freq'] = pool['n'] / max(n_races_total, 1) * 100
    return pool, {w: stats(per_w[w]) for w, _, _, _ in wins}


def main():
    base = os.getcwd()
    (df, cols, X, d, rid, hn, pop_tr, pop_ev, place, pay) = load()
    out = {'judge': {}, 'y2025': {}, 'y2026': {}, 'control': {}, 'stop': [], 'omote': {}}

    print('■ 判定窓（2024・未読）を学習する')
    jm, jp, jf = fit_period(JUDGE, cols, X, d, rid, hn, pop_tr, pop_ev, place, '2024')
    print('■ 2025（既読・再現確認と頻度用）を学習する')
    m25, p25, f25 = fit_period(Y2025, cols, X, d, rid, hn, pop_tr, pop_ev, place, '2025')
    print('■ 2026（既読・矛盾確認と頻度用）を学習する')
    m26, p26, f26 = fit_period(Y2026, cols, X, d, rid, hn, pop_tr, pop_ev, place, '2026')

    nr = {'2024': sum(int(pd.unique(rid[jm[w][2]]).size) for w, _, _, _ in JUDGE),
          '2025': sum(int(pd.unique(rid[m25[w][2]]).size) for w, _, _, _ in Y2025),
          '2026': sum(int(pd.unique(rid[m26[w][2]]).size) for w, _, _, _ in Y2026)}

    res = {}
    for arm in ('S1', 'S2'):
        res[arm] = {
            '2024': agg(JUDGE, jp, jf, pay, arm, nr['2024']),
            '2025': agg(Y2025, p25, f25, pay, arm, nr['2025']),
            '2026': agg(Y2026, p26, f26, pay, arm, nr['2026']),
        }

    # ── 中止条件 ───────────────────────────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §5）— 結果を読む手前で判定する')
    print('=' * 78)
    stop = []
    print(f'  1. 判定窓(2024)のレース数 = {nr["2024"]:,}（>= 2,500 か）→ {nr["2024"] >= 2500}')
    if nr['2024'] < 2500:
        stop.append(f'判定窓 {nr["2024"]} < 2500')

    n1, n2 = res['S1']['2024'][0]['n'], res['S2']['2024'][0]['n']
    ok2 = (n1 >= 100) or (n2 >= 100)
    print(f'  2. 2024 の ☆ レース数: S-1 {n1:.1f} / S-2 {n2:.1f}  → どちらか >= 100 か: {ok2}')
    if not ok2:
        stop.append(f'☆レースが100未満: S-1 {n1:.1f} / S-2 {n2:.1f}')

    ok3 = True
    for name, _, _, _ in JUDGE:
        m_fit, m_in, m_va, inf_ = jm[name]
        thin = int(m_in.sum()) < 1000
        ok3 = ok3 and not thin
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
        print(f'  3. {name} 学習 {int(m_fit.sum()):,} / 内側HO {int(m_in.sum()):,}'
              f'（起点 {(inf_ + pd.Timedelta(days=1)).date()}）/ 評価 {int(m_va.sum()):,}'
              f'  1000行未満: {thin}')
    if not ok3:
        stop.append('内側HOが1000行未満')

    bad = [jp[w][2][5] / max(pd.unique(rid[jm[w][2]]).size, 1) * 100
           for w, _, _, _ in JUDGE]
    ok4 = max(bad) <= 5.0
    print(f'  4. P1 を特定できないレース = {[round(x, 2) for x in bad]}%  → 5%以下か: {ok4}')
    if not ok4:
        stop.append(f'P1 特定不能が5%超: {bad}')

    b1 = []
    for name, _, _, _ in JUDGE:
        for sd in SEEDS:
            s = select(picks(jp, jf, name, sd), 'S1')
            if len(s) == 0:
                continue
            ax = s[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
            b1.append(roi(ax, pay, 'fukusho', bonus=BONUS)['roi']
                      - roi(ax, pay, 'fukusho', bonus=0.0)['roi'])
    b1o = float(np.mean(b1)) if b1 else float('nan')
    okB1 = abs(b1o - BONUS) <= 0.01
    print(f'  5. B-1 複勝ROI: 注入後 − 注入前 = {b1o:+.4f}pt / 注入量 = {BONUS:+.2f}pt'
          f'（本物が無いとき = +0.00pt）→ ±0.01 か: {okB1}')
    b2 = []
    for name, _, _, _ in JUDGE:
        pv = jp[name][2]
        for sd in SEEDS:
            s_ = jf[(name, sd)]['raw_va']
            b2.append((roc_auc_score(pv[1], s_), roc_auc_score(pv[1], s_ + BIG * pv[1])))
    a0 = float(np.mean([a for a, _ in b2])); a1 = float(np.mean([b for _, b in b2]))
    okB2 = abs(a1 - 1.0) <= 0.0001
    print(f'     B-2 AUC: 注入後 = {a1:.6f} / 注入量 = 1 − {a0:.6f} = {1 - a0:.6f}'
          f'（本物が無いとき = 0.000000）→ ±0.0001 か: {okB2}')
    out['control']['B'] = dict(b1=b1o, b1_ok=bool(okB1), auc=a0, auc_inj=a1,
                               b2_ok=bool(okB2))
    if not (okB1 and okB2):
        stop.append(f'対照群B が復元しない（B-1 {okB1} / B-2 {okB2}）')

    # 中止条件6: 2025 を**旧集計式**で出して c_rerun_result.json と突き合わせる
    prev_p = os.path.join(base, 'hitrate', 'c_rerun_result.json')
    ok6 = True
    if os.path.exists(prev_p):
        prev = json.load(open(prev_p))
        pr = prev['perf'].get('0.6') or prev['perf'].get('0.60')
        if pr and pr['pool']:
            # 旧式: 窓ごとにシード平均（0件シードを落とす）→ 窓間で平均
            old = []
            for name, _, _, _ in Y2025:
                wr = []
                for sd in SEEDS:
                    s = select(picks(p25, f25, name, sd), 'S1')
                    if len(s) == 0:
                        continue
                    ax = s[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                    wr.append(roi(ax, pay, 'fukusho')['rate'])
                if wr:
                    old.append(float(np.mean(wr)))
            oldv = float(np.mean(old)) if old else float('nan')
            dv = abs(oldv - pr['pool']['rate_sw'])
            ok6 = dv <= 0.01
            print(f'  6. 2025 の複勝率(差替) 旧集計式で再計算 = {oldv:.4f}% / '
                  f'c_rerun_result.json = {pr["pool"]["rate_sw"]:.4f}%  差 {dv:.4f}pt'
                  f'  → ±0.01 で一致か: {ok6}')
            print(f'     （参考・カード §3 の新集計式では {res["S1"]["2025"][0]["rate_sw"]:.4f}%）')
            if not ok6:
                stop.append('2025 が前回と一致しない（再現性）')
    print('-' * 78)
    print('中止条件: ' + ('発火なし（続行）' if not stop else '発火 → ' + ' / '.join(stop)))
    print('-' * 78)

    # ── 結果 ───────────────────────────────────────────────────────────
    for arm, label in (('S1', f'S-1 τ固定（τ={TAU}）'),
                       ('S2', f'S-2 頻度固定（上位{TOP_FRAC * 100:.1f}%）')):
        print()
        print('=' * 78)
        print(f'■ {label}')
        print('=' * 78)
        for per in ('2024', '2025', '2026'):
            p, _w = res[arm][per]
            tagd = '【判定】' if per == '2024' else '（既読・参考）'
            print(f'  {per}{tagd} ☆ {p["n"]:6.1f}レース / {nr[per]:,}  = {p["freq"]:5.2f}%   '
                  f'宣言値(中位P̂) {p["decl"]:.4f}  実測「P1を上回った率」 {p["obs"]:.4f}')
            print(f'          複勝率 差替 {p["rate_sw"]:6.2f}% / P1 {p["rate_p1"]:6.2f}% '
                  f'(Δ{p["rate_sw"] - p["rate_p1"]:+6.2f}pt)   '
                  f'複勝回収 差替 {p["roif_sw"]:6.2f}% / P1 {p["roif_p1"]:6.2f}% '
                  f'(Δ{p["roif_sw"] - p["roif_p1"]:+6.2f}pt)   '
                  f'3本抜き 差替 {p["cut3_sw"]:6.2f}% / P1 {p["cut3_p1"]:6.2f}% '
                  f'(Δ{p["cut3_sw"] - p["cut3_p1"]:+6.2f}pt)')
        print('  2024 の窓ごと:')
        for name, _, _, _ in JUDGE:
            q = res[arm]['2024'][1][name]
            print(f'    {name} n={q["n"]:5.1f}  実測 {q["obs"]:.4f}  '
                  f'複勝率Δ{q["rate_sw"] - q["rate_p1"]:+6.2f}pt  '
                  f'複勝回収Δ{q["roif_sw"] - q["roif_p1"]:+6.2f}pt')

    # 対照群A（判定窓・腕ごと）
    print()
    print('■ 対照群A（符号反転200回・判定窓2024）')
    permres = {}
    for arm in ('S1', 'S2'):
        ds = []
        for name, _, _, _ in JUDGE:
            for sd in SEEDS:
                s = select(picks(jp, jf, name, sd), arm)
                if len(s) == 0:
                    continue
                ax = s[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                ap = s[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
                ms = ax.merge(pay, on=['race_id', 'horse_num'], how='left')
                mp = ap.merge(pay, on=['race_id', 'horse_num'], how='left')
                ds.append(ms['place'].between(1, 3).astype(float).values
                          - mp['place'].between(1, 3).astype(float).values)
        dall = np.concatenate(ds) if ds else np.zeros(0)
        g = np.random.default_rng(PERM_SEED)
        sh = [float(np.mean(dall * g.choice([-1.0, 1.0], len(dall))))
              for _ in range(N_PERM)] if len(dall) else [0.0]
        nm, se = float(np.mean(sh)), float(np.std(sh)) / np.sqrt(N_PERM)
        okA = abs(nm) < 2 * se
        permres[arm] = dict(null=nm, se=se, obs=float(dall.mean()) if len(dall) else np.nan,
                            ok=bool(okA))
        print(f'  {arm}: 帰無平均 {nm:+.5f} / 2*SE {2 * se:.5f} → 0中心か: {okA}  '
              f'実測 mean(Δ) = {permres[arm]["obs"]:+.5f}')

    # ── 合格基準 ───────────────────────────────────────────────────────
    print()
    print('■ 合格基準（カード §4）')
    verdict = {}
    for arm in ('S1', 'S2'):
        p, w = res[arm]['2024']
        decl = TAU if arm == 'S1' else p['decl']
        k1 = (p['obs'] >= decl - 0.05) and all(
            w[n]['obs'] >= (decl if arm == 'S1' else w[n]['decl']) - 0.10
            for n, _, _, _ in JUDGE)
        k2 = (all(w[n]['rate_sw'] - w[n]['rate_p1'] > 0 for n, _, _, _ in JUDGE)
              and (p['rate_sw'] - p['rate_p1']) >= 10.0)
        k3 = ((p['roif_sw'] - p['roif_p1']) >= -2.0
              and (p['cut3_sw'] - p['cut3_p1']) >= -2.0)
        fq = [res[arm][y][0]['freq'] for y in ('2024', '2025', '2026')]
        k4 = (max(fq) / max(min(fq), 1e-9)) <= 3.0
        k5 = p['n'] >= 100
        k6 = permres[arm]['ok'] and okB1 and okB2
        p26 = res[arm]['2026'][0]
        k7 = (p26['rate_sw'] - p26['rate_p1']) > 0
        allok = all([k1, k2, k3, k4, k5, k6, k7])
        verdict[arm] = allok
        print(f'  {arm}: ①較正(実測{p["obs"]:.4f} vs 宣言{decl:.2f}−0.05): {k1} '
              f'②的中率(Δ{p["rate_sw"] - p["rate_p1"]:+.2f}pt・3窓すべて正): {k2} '
              f'③回収率ガード(Δ{p["roif_sw"] - p["roif_p1"]:+.2f}pt・3本抜きΔ'
              f'{p["cut3_sw"] - p["cut3_p1"]:+.2f}pt): {k3}')
        print(f'      ④頻度の安定({[round(x, 2) for x in fq]}・max/min='
              f'{max(fq) / max(min(fq), 1e-9):.2f}): {k4} '
              f'⑤N>=100({p["n"]:.1f}): {k5} ⑥対照群: {k6} '
              f'⑦2026で矛盾しない(Δ{p26["rate_sw"] - p26["rate_p1"]:+.2f}pt): {k7} '
              f'→ {"合格" if allok else "不合格"}')

    print()
    print('=' * 78)
    print('■ 判定（カード §4 の順序）')
    print('=' * 78)
    if stop:
        print('  ⚠ 中止条件が発火している: ' + ' / '.join(stop))
    if verdict['S1'] and not stop:
        p = res['S1']['2024'][0]
        print(f'  → **τ={TAU} 固定で ☆ を出してよい。** '
              f'記事に書く確率 = 実測 {p["obs"] * 100:.1f}% を5%刻みで切り下げて '
              f'**{int(p["obs"] * 100 // 5 * 5)}%以上**')
    elif verdict['S2'] and not stop:
        p = res['S2']['2024'][0]
        print(f'  → **頻度固定（上位{TOP_FRAC * 100:.1f}%）で ☆ を出してよい。** '
              f'記事に書く確率 = 実測 {p["obs"] * 100:.1f}% を5%刻みで切り下げて '
              f'**{int(p["obs"] * 100 // 5 * 5)}%以上**')
    else:
        print('  → **☆ は出さない。**')
    print('  🔴 ◎は動かさない（このカードでは合否を出さない）。')

    print()
    print('=' * 78)
    print('■ カード §6 報告のみ（◎の差し替えを 2024年で読んだ数字・合否は出さない）')
    print('=' * 78)
    for arm in ('S1', 'S2'):
        p = res[arm]['2024'][0]
        print(f'  {arm}: 複勝率 差替 {p["rate_sw"]:.2f}% / P1 {p["rate_p1"]:.2f}% '
              f'(Δ{p["rate_sw"] - p["rate_p1"]:+.2f}pt)   '
              f'複勝回収 差替 {p["roif_sw"]:.2f}% / P1 {p["roif_p1"]:.2f}% '
              f'(Δ{p["roif_sw"] - p["roif_p1"]:+.2f}pt)   '
              f'3本抜き Δ{p["cut3_sw"] - p["cut3_p1"]:+.2f}pt')
    print('  🔴 ◎を動かすかはユーザー判断。動かすなら v2 の線を写した別カードを書いてから。')

    out['judge'] = {a: dict(pool=res[a]['2024'][0], per_w=res[a]['2024'][1])
                    for a in ('S1', 'S2')}
    out['y2025'] = {a: res[a]['2025'][0] for a in ('S1', 'S2')}
    out['y2026'] = {a: res[a]['2026'][0] for a in ('S1', 'S2')}
    out['control']['A'] = permres
    out['verdict'] = verdict
    out['stop'] = stop
    out['n_races'] = nr
    with open(os.path.join(base, 'hitrate', 'star_result.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n→ hitrate/star_result.json に保存した')


if __name__ == '__main__':
    main()
