#!/usr/bin/env python3
"""C-再測の測定ハーネス（事前登録カード `hitrate/CRITERIA_C_RERUN.md` のとおり）。

問い: 「◎を差し替えると的中率が +13.7〜29.4pt 上がる」は本物か（＝☆を出してよいか）。

🔴 C-2 の**不採用判定は再検討しない**。基準④（複勝回収率）と⑤（3窓同符号）は落ちたまま。
🔴 対照群Bは「**閉じた式で注入量が分かる統計量**」にしか置かない（前回の失敗の対策）。
🔴 中止条件は結果を読む**手前**で印字する。測定は1回。隔離ワークスペースで実行する。

使い方:
    cd <隔離ワークスペース>
    python3 hitrate/eval_c_rerun.py
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

# eval_c.py の部品をそのまま使う（腕の設定を1つも変えないため）
_s = importlib.util.spec_from_file_location(
    'eval_c', os.path.join(os.getcwd(), 'hitrate', 'eval_c.py'))
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

_s2 = importlib.util.spec_from_file_location(
    'logic2026_eval', os.path.join(os.getcwd(), 'logic2026', 'eval.py'))
L = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(L)
roi = L.roi

from src.tools.train_xgb import _popularity_to_base_margin, load_popularity_drift  # noqa: E402

WINDOWS, SEEDS, TAUS = C.WINDOWS, C.SEEDS, C.TAUS
N_PERM = C.N_PERM
BONUS = 5.0                 # カード §3 B-1
BIG = 1000.0                # カード §3 B-2
PERM_SEED = 20261110


def main():
    base = os.getcwd()
    out = {'repro': {}, 'control': {}, 'star': {}}

    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = C.feat_cols(df)
    drift = load_popularity_drift(base)

    X = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    d = df['date_obj'].to_numpy()
    race_ids = df['race_id'].to_numpy()
    horse_nums = df['horse_num'].to_numpy()
    pop_tr, nh = C.drifted_pop(df, drift, C.DRIFT_SEED_TRAIN)
    pop_ev, _ = C.drifted_pop(df, drift, C.DRIFT_SEED_EVAL)
    _ = _popularity_to_base_margin(pd.Series(pop_ev), pd.Series(nh))  # 使わないが経路を揃える

    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= '2023-01-01' AND r.date <= '2026-12-31'""", con)
    con.close()
    pmap = {(r, n): p for r, n, p in zip(pay['race_id'], pay['horse_num'], pay['place'])}
    place = np.array([pmap.get((r, n), np.nan) for r, n in zip(race_ids, horse_nums)],
                     dtype=float)
    pay_eval = pay[['race_id', 'horse_num', 'place', 'fukusho_payout', 'tansho_payout']]

    # ── 学習（C-2 と完全に同じ設定・注入列は足さない）────────────────────
    masks, pairs, pfits = {}, {}, {}
    for name, te, v0, v1 in WINDOWS:
        m_fit, m_in, inner_from = C.inner_split(d, te)
        m_va = ((d >= np.datetime64(pd.Timestamp(v0)))
                & (d <= np.datetime64(pd.Timestamp(v1))))
        masks[name] = (m_fit, m_in, m_va, inner_from)
        pf = C.build_pairs(np.flatnonzero(m_fit), X, place, race_ids, horse_nums, pop_tr)
        pi = C.build_pairs(np.flatnonzero(m_in), X, place, race_ids, horse_nums, pop_tr)
        pv = C.build_pairs(np.flatnonzero(m_va), X, place, race_ids, horse_nums, pop_ev)
        pairs[name] = (pf, pi, pv)
        for sd in SEEDS:
            pfits[(name, sd)] = C.fit_pair(pf[0], pf[1], pi[0], pi[1], pv[0], cols, sd)
            print(f'  …fit {name} seed={sd} it={pfits[(name, sd)]["it"]} '
                  f'pairs {len(pf[1])},{len(pi[1])},{len(pv[1])}', flush=True)

    # ── 共通の集計関数 ─────────────────────────────────────────────────
    def top_pick(name, sd, oracle=False):
        """レースごとに差し替え候補1頭を選ぶ。oracle=True ならカード §3 B-3 の選び方。"""
        pv = pairs[name][2]
        dd = pd.DataFrame({'r': pv[2], 'n': pv[3], 'p1': pv[4],
                           'y': pv[1], 'p': pfits[(name, sd)]['p_va']})
        if oracle:
            dd = dd.sort_values(['r', 'y', 'p', 'n'], ascending=[True, False, False, True])
        else:
            dd = dd.sort_values(['r', 'p', 'n'], ascending=[True, False, True])
        return dd.groupby('r', as_index=False).first()

    # ── 中止条件 ───────────────────────────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §5）— 結果を読む手前で判定する')
    print('=' * 78)
    stop = []

    n_races = [int(pd.unique(race_ids[masks[w][2]]).size) for w, _, _, _ in WINDOWS]
    ok1 = sum(n_races) >= 3000
    print(f'  1. 評価レース数 = {n_races} 計 {sum(n_races):,}（>= 3,000 か）→ {ok1}')
    if not ok1:
        stop.append(f'評価レース数 {sum(n_races)} < 3000')

    swap_n = {}
    for tau in TAUS:
        c = 0
        for name, _, _, _ in WINDOWS:
            for sd in SEEDS:
                c += int((top_pick(name, sd)['p'] >= tau).sum())
        swap_n[tau] = c / len(SEEDS)
    ok2 = any(v >= 100 for v in swap_n.values())
    print(f'  2. 差し替えレース数（シード平均）= {({k: round(v, 1) for k, v in swap_n.items()})}'
          f'  → どれか >= 100 か: {ok2}')
    if not ok2:
        stop.append(f'差し替えレースが100未満: {swap_n}')

    ok3 = True
    for name, _, _, _ in WINDOWS:
        m_fit, m_in, m_va, inf_ = masks[name]
        thin = int(m_in.sum()) < 1000
        ok3 = ok3 and not thin
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
        print(f'  3. {name} 学習 {int(m_fit.sum()):,} / 内側HO {int(m_in.sum()):,}'
              f'（起点 {(inf_ + pd.Timedelta(days=1)).date()}）/ 評価 {int(m_va.sum()):,}'
              f'  1000行未満: {thin}')
    if not ok3:
        stop.append('内側HOが1000行未満')

    bad = [pairs[w][2][5] / max(pd.unique(race_ids[masks[w][2]]).size, 1) * 100
           for w, _, _, _ in WINDOWS]
    ok4 = max(bad) <= 5.0
    print(f'  4. P1 を特定できないレース = {[round(x, 2) for x in bad]}%  → 5%以下か: {ok4}')
    if not ok4:
        stop.append(f'P1 特定不能が5%超: {bad}')

    # ── 対照群B（閉じた式で復元・カード §3）──────────────────────────────
    print()
    print('  5. 対照群B（閉じた式で注入量が分かる統計量にしか置かない）')

    # B-1 ROI の復元
    b1 = []
    for name, _, _, _ in WINDOWS:
        for sd in SEEDS:
            tp = top_pick(name, sd)
            sw = tp[tp['p'] >= 0.55]
            if len(sw) == 0:
                continue
            ax = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
            r0 = roi(ax, pay_eval, 'fukusho', bonus=0.0)['roi']
            r1 = roi(ax, pay_eval, 'fukusho', bonus=BONUS)['roi']
            b1.append(r1 - r0)
    b1_obs, b1_null = float(np.mean(b1)), 0.0
    okB1 = abs(b1_obs - BONUS) <= 0.01
    print(f'     B-1 ROI: 注入後 − 注入前 = {b1_obs:+.4f}pt  /  注入量 = {BONUS:+.2f}pt'
          f'  （本物が無いときこの式が返す値 = {b1_null:+.2f}pt）  → ±0.01 で一致か: {okB1}')

    # B-2 AUC とラベル配線の復元
    b2 = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        for sd in SEEDS:
            s_ = pfits[(name, sd)]['raw_va']
            a0 = roc_auc_score(pv[1], s_)
            a1 = roc_auc_score(pv[1], s_ + BIG * pv[1])
            b2.append((a0, a1))
    a0m = float(np.mean([a for a, _ in b2])); a1m = float(np.mean([b for _, b in b2]))
    okB2 = abs(a1m - 1.0) <= 0.0001 and abs((a1m - a0m) - (1.0 - a0m)) <= 0.0001
    print(f'     B-2 AUC: 注入後 = {a1m:.6f}（注入量 = 1 − {a0m:.6f} = {1 - a0m:.6f}）'
          f'  注入後 − 注入前 = {a1m - a0m:.6f}'
          f'  （本物が無いときこの式が返す値 = 0.000000）  → ±0.0001 で一致か: {okB2}')

    # B-3 差し替え集合の復元
    b3 = []
    for name, _, _, _ in WINDOWS:
        pv = pairs[name][2]
        share = float(pd.DataFrame({'r': pv[2], 'y': pv[1]})
                      .groupby('r')['y'].max().mean())      # 閉じた式（P̂に依らない）
        for sd in SEEDS:
            orc = top_pick(name, sd, oracle=True)
            b3.append((float(orc['y'].mean()), share))
    b3_obs = float(np.mean([a for a, _ in b3])); b3_exp = float(np.mean([b for _, b in b3]))
    okB3 = abs(b3_obs - b3_exp) <= 0.005
    print(f'     B-3 オラクル: 勝ち越し率 = {b3_obs:.5f}  /  '
          f'「P1を上回った馬が1頭以上いるレースの割合」= {b3_exp:.5f}'
          f'  （本物が無いときこの差が返す値 = 0.000）  → ±0.005 で一致か: {okB3}')

    out['control']['B'] = dict(b1=b1_obs, b1_ok=bool(okB1), auc_plain=a0m, auc_inj=a1m,
                               b2_ok=bool(okB2), b3_obs=b3_obs, b3_exp=b3_exp,
                               b3_ok=bool(okB3))
    if not (okB1 and okB2 and okB3):
        stop.append('対照群B が復元しない（B-1 %s / B-2 %s / B-3 %s）'
                    % (okB1, okB2, okB3))

    # ── 較正・的中率（カード §4 の基準3・4・5 の材料）───────────────────
    cal, perf = {}, {}
    for tau in TAUS:
        rows = []
        for name, _, _, _ in WINDOWS:
            pi, pv = pairs[name][1], pairs[name][2]
            for sd in SEEDS:
                f = pfits[(name, sd)]
                for lab, p, yy in (('inner', f['p_in'], pi[1]), ('eval', f['p_va'], pv[1])):
                    sel = p >= tau
                    rows.append((lab, int(sel.sum()),
                                 float(yy[sel].mean()) if sel.any() else np.nan))
        for lab in ('inner', 'eval'):
            sub = [(n_, r) for l_, n_, r in rows if l_ == lab and n_ > 0]
            tot = sum(n_ for n_, _ in sub)
            cal[(tau, lab)] = (tot, (sum(n_ * r for n_, r in sub) / tot) if tot else np.nan)

        per_w, perm = {}, []
        for name, _, _, _ in WINDOWS:
            wr = []
            for sd in SEEDS:
                tp = top_pick(name, sd)
                sw = tp[tp['p'] >= tau]
                if len(sw) == 0:
                    continue
                ax = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                ap = sw[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
                rs, rp = roi(ax, pay_eval, 'fukusho'), roi(ap, pay_eval, 'fukusho')
                wr.append(dict(n=len(sw), rate_sw=rs['rate'], rate_p1=rp['rate'],
                               roif_sw=rs['roi'], roif_p1=rp['roi']))
                ms = ax.merge(pay_eval, on=['race_id', 'horse_num'], how='left')
                mp = ap.merge(pay_eval, on=['race_id', 'horse_num'], how='left')
                delta = (ms['place'].between(1, 3).astype(float).values
                         - mp['place'].between(1, 3).astype(float).values)
                g = np.random.default_rng(PERM_SEED + sd)
                sh = [float(np.mean(delta * g.choice([-1.0, 1.0], len(delta))))
                      for _ in range(N_PERM)]
                perm.append((float(np.mean(delta)), float(np.mean(sh)), float(np.std(sh))))
            per_w[name] = ({k: float(np.mean([w[k] for w in wr])) for k in wr[0]}
                           if wr else None)
        okw = [per_w[w] for w, _, _, _ in WINDOWS if per_w[w]]
        perf[tau] = dict(per_w=per_w, perm=perm, n_total=sum(w['n'] for w in okw) if okw else 0,
                         pool=({k: float(np.mean([w[k] for w in okw])) for k in okw[0]}
                               if okw else None))

    # 中止条件6: 前回の結果と一致するか
    prev_p = os.path.join(base, 'hitrate', 'c_result.json')
    ok6 = True
    if os.path.exists(prev_p):
        prev = json.load(open(prev_p))
        for tau in TAUS:
            pr = prev['c2'].get(str(tau)) or prev['c2'].get(str(float(tau)))
            if not pr or not pr['pool'] or not perf[tau]['pool']:
                continue
            for k in ('rate_sw', 'rate_p1', 'roif_sw', 'roif_p1'):
                dv = abs(pr['pool'][k] - perf[tau]['pool'][k])
                if dv > 0.01:
                    ok6 = False
                    print(f'  6. 再現しない: τ={tau} {k} 前回 {pr["pool"][k]:.2f} '
                          f'→ 今回 {perf[tau]["pool"][k]:.2f}')
        print(f'  6. 前回 c_result.json との一致（±0.01pt）: {ok6}')
        if not ok6:
            stop.append('前回の結果と一致しない（再現性）')
    else:
        print('  6. 前回の c_result.json が無いので一致確認を省略')

    print('-' * 78)
    print('中止条件: ' + ('発火なし（続行）' if not stop else '発火 → ' + ' / '.join(stop)))
    print('-' * 78)

    # ── 結果 ───────────────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('■ 較正と的中率（☆の基準の材料）')
    print('=' * 78)
    print(f'{"τ":>5} {"内側HO n":>9} {"内側HO 実測":>11} {"評価 n":>8} {"評価 実測":>10} '
          f'{"差替":>6} {"複勝率(差替)":>12} {"複勝率(P1)":>11} {"Δ":>9}')
    for tau in TAUS:
        a, b = cal[(tau, 'inner')], cal[(tau, 'eval')]
        p = perf[tau]['pool']
        if not p:
            print(f'{tau:5.2f} 該当なし'); continue
        print(f'{tau:5.2f} {a[0]:9,} {a[1]:11.4f} {b[0]:8,} {b[1]:10.4f} '
              f'{perf[tau]["n_total"]:6.0f} {p["rate_sw"]:11.2f}% {p["rate_p1"]:10.2f}% '
              f'{p["rate_sw"] - p["rate_p1"]:+8.2f}pt')

    print()
    print('窓ごとの複勝率差（基準4）:')
    for tau in TAUS:
        s = f'  τ={tau:.2f}  '
        for w, _, _, _ in WINDOWS:
            q = perf[tau]['per_w'][w]
            s += (f'{w} n={q["n"]:5.1f} Δ{q["rate_sw"] - q["rate_p1"]:+6.2f}pt  '
                  if q else f'{w} 該当なし  ')
        print(s)

    print()
    print('■ ☆を出してよいか（カード §4 の5基準）')
    okB = okB1 and okB2 and okB3
    star_ok, star_tau = False, None
    for tau in TAUS:
        p = perf[tau]['pool']
        if not p:
            print(f'  τ={tau:.2f}: 該当なし → 不合格'); continue
        pm = perf[tau]['perm']
        se = float(np.mean([s / np.sqrt(N_PERM) for _, _, s in pm]))
        nm = float(np.mean([m for _, m, _ in pm]))
        k2 = abs(nm) < 2 * se
        k3 = (cal[(tau, 'inner')][1] >= tau - 0.05 and cal[(tau, 'eval')][1] >= tau - 0.05)
        k4 = all(perf[tau]['per_w'][w] and
                 (perf[tau]['per_w'][w]['rate_sw'] - perf[tau]['per_w'][w]['rate_p1']) > 0
                 for w, _, _, _ in WINDOWS)
        k5 = perf[tau]['n_total'] >= 100
        allok = okB and k2 and k3 and k4 and k5
        star_ok = star_ok or allok
        if allok and star_tau is None:
            star_tau = tau
        print(f'  τ={tau:.2f}: ①対照群B: {okB} ②対照群A(帰無 {nm:+.5f} / 2SE {2 * se:.5f}): {k2} '
              f'③較正: {k3} ④3窓すべて正: {k4} ⑤差替>=100({perf[tau]["n_total"]:.0f}): {k5} '
              f'→ {"合格" if allok else "不合格"}')

    print()
    print('=' * 78)
    print('■ 判定')
    print('=' * 78)
    if stop:
        print('  ⚠ 中止条件が発火している: ' + ' / '.join(stop))
    print('  ☆を出す: ' + (f'**よい**（τ={star_tau:.2f}）' if (star_ok and not stop)
                           else '**出さない**'))
    print('  🔴 C-2 の不採用（複勝回収率が改善しない）は変わらない。◎は動かさない。')

    out['star'] = dict(ok=bool(star_ok and not stop),
                       tau=(float(star_tau) if star_tau is not None else None))
    out['repro'] = dict(ok=bool(ok6))
    out['perf'] = {str(t): dict(pool=perf[t]['pool'], n_total=perf[t]['n_total'],
                                per_w=perf[t]['per_w']) for t in TAUS}
    out['cal'] = {f'{t}_{l}': cal[(t, l)] for t, l in cal}
    out['stop'] = stop
    with open(os.path.join(base, 'hitrate', 'c_rerun_result.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n→ hitrate/c_rerun_result.json に保存した')


if __name__ == '__main__':
    main()
