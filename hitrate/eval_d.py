#!/usr/bin/env python3
"""候補D の測定ハーネス（事前登録カード `hitrate/CRITERIA_D.md` のとおり）。

問い: `RESULTS_C.md` §2-3 の単勝回収率 101.46%（τ=0.60・N=134・2025年）は
      別の期間でも残るのか。事前の見込みは「**消える**」。

🔴 探索期（2025）は**判定に使わない**。判定は確認期（2026）だけ。
🔴 信頼区間の幅が 60.0pt を超えたら**点推定を読まずに「N不足で判定不能」**とする。
🔴 中止条件は結果を読む手前で印字する。測定は1回。隔離ワークスペースで実行する。

使い方:
    cd <隔離ワークスペース>
    python3 hitrate/eval_d.py
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

# カード §1 の窓（確認期＝2026年の walk-forward 3分割）
CONFIRM = [('V1', '2025-12-31', '2026-01-01', '2026-04-30'),
           ('V2', '2026-04-30', '2026-05-01', '2026-08-31'),
           ('V3', '2026-08-31', '2026-09-01', '2026-11-08')]
EXPLORE = C.WINDOWS                 # 探索期（2025・再現確認にだけ使う）
TAUS = (0.55, 0.60)                 # カード §2
SEEDS = C.SEEDS
N_BOOT, BOOT_SEED = 10000, 20261110
N_PERM, PERM_SEED = 200, 20261110
BONUS, BIG = 5.0, 1000.0
CI_WIDTH_MAX = 60.0                 # カード §4-5 / §6-7


def payout_vec(axis, pay, kind='tansho'):
    """roi() と**同じ絞り方**で、1点買いの払戻ベクトルを返す（ブートストラップ用）。"""
    m = axis.merge(pay, on=['race_id', 'horse_num'], how='left')
    col = f'{kind}_payout'
    hit = m['place'].between(1, 3) if kind == 'fukusho' else (m['place'] == 1)
    has = m['place'].notna()
    bad = has & hit & (m[col].fillna(0) <= 0)
    ok = has & ~bad
    return np.where(hit[ok], m.loc[ok, col].fillna(0).values, 0.0)


def main():
    base = os.getcwd()
    out = {'confirm': {}, 'explore': {}, 'control': {}, 'stop': []}

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
    pop_tr, _ = C.drifted_pop(df, drift, C.DRIFT_SEED_TRAIN)
    pop_ev, _ = C.drifted_pop(df, drift, C.DRIFT_SEED_EVAL)

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

    def run_windows(wins):
        masks, pairs, pfits = {}, {}, {}
        for name, te, v0, v1 in wins:
            m_fit, m_in, inf_ = C.inner_split(d, te)
            m_va = ((d >= np.datetime64(pd.Timestamp(v0)))
                    & (d <= np.datetime64(pd.Timestamp(v1))))
            masks[name] = (m_fit, m_in, m_va, inf_)
            pf = C.build_pairs(np.flatnonzero(m_fit), X, place, race_ids, horse_nums, pop_tr)
            pi = C.build_pairs(np.flatnonzero(m_in), X, place, race_ids, horse_nums, pop_tr)
            pv = C.build_pairs(np.flatnonzero(m_va), X, place, race_ids, horse_nums, pop_ev)
            pairs[name] = (pf, pi, pv)
            for sd in SEEDS:
                pfits[(name, sd)] = C.fit_pair(pf[0], pf[1], pi[0], pi[1], pv[0], cols, sd)
                print(f'  …fit {name} seed={sd} it={pfits[(name, sd)]["it"]} '
                      f'pairs {len(pf[1])},{len(pi[1])},{len(pv[1])}', flush=True)
        return masks, pairs, pfits

    def top_pick(pairs, pfits, name, sd):
        pv = pairs[name][2]
        dd = pd.DataFrame({'r': pv[2], 'n': pv[3], 'p1': pv[4], 'y': pv[1],
                           'p': pfits[(name, sd)]['p_va']})
        return (dd.sort_values(['r', 'p', 'n'], ascending=[True, False, True])
                  .groupby('r', as_index=False).first())

    print('■ 確認期（2026）を学習・評価する')
    cm, cp, cf = run_windows(CONFIRM)
    print('■ 探索期（2025）を再学習する（再現確認にだけ使う）')
    em, ep, ef = run_windows(EXPLORE)

    # ── 中止条件 ───────────────────────────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §6）— 結果を読む手前で判定する')
    print('=' * 78)
    stop = []

    n_races = [int(pd.unique(race_ids[cm[w][2]]).size) for w, _, _, _ in CONFIRM]
    ok1 = sum(n_races) >= 2500
    print(f'  1. 確認期のレース数 = {n_races} 計 {sum(n_races):,}（>= 2,500 か）→ {ok1}')
    if not ok1:
        stop.append(f'確認期 {sum(n_races)} < 2500')

    swap = {}
    for tau in TAUS:
        c = 0
        for name, _, _, _ in CONFIRM:
            for sd in SEEDS:
                c += int((top_pick(cp, cf, name, sd)['p'] >= tau).sum())
        swap[tau] = c / len(SEEDS)
    ok2 = any(v >= 100 for v in swap.values())
    print(f'  2. 確認期の差し替えレース数（シード平均）= '
          f'{({k: round(v, 1) for k, v in swap.items()})}  → どれか >= 100 か: {ok2}')
    if not ok2:
        stop.append(f'確認期の差し替えレースが100未満: {swap}（N不足で判定不能）')

    ok3 = True
    for name, _, _, _ in CONFIRM:
        m_fit, m_in, m_va, inf_ = cm[name]
        thin = int(m_in.sum()) < 1000
        ok3 = ok3 and not thin
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
        print(f'  3. {name} 学習 {int(m_fit.sum()):,} / 内側HO {int(m_in.sum()):,}'
              f'（起点 {(inf_ + pd.Timedelta(days=1)).date()}）/ 評価 {int(m_va.sum()):,}'
              f'  1000行未満: {thin}')
    if not ok3:
        stop.append('内側HOが1000行未満')

    bad = [cp[w][2][5] / max(pd.unique(race_ids[cm[w][2]]).size, 1) * 100
           for w, _, _, _ in CONFIRM]
    ok4 = max(bad) <= 5.0
    print(f'  4. P1 を特定できないレース = {[round(x, 2) for x in bad]}%  → 5%以下か: {ok4}')
    if not ok4:
        stop.append(f'P1 特定不能が5%超: {bad}')

    # 対照群B（閉じた式で復元）
    b1 = []
    for name, _, _, _ in CONFIRM:
        for sd in SEEDS:
            tp = top_pick(cp, cf, name, sd)
            sw = tp[tp['p'] >= TAUS[0]]
            if len(sw) == 0:
                continue
            ax = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
            b1.append(roi(ax, pay_eval, 'tansho', bonus=BONUS)['roi']
                      - roi(ax, pay_eval, 'tansho', bonus=0.0)['roi'])
    b1o = float(np.mean(b1)) if b1 else float('nan')
    okB1 = abs(b1o - BONUS) <= 0.01
    print(f'  5. B-1 単勝ROI: 注入後 − 注入前 = {b1o:+.4f}pt / 注入量 = {BONUS:+.2f}pt'
          f'（本物が無いとき = +0.00pt）→ ±0.01 か: {okB1}')
    b2 = []
    for name, _, _, _ in CONFIRM:
        pv = cp[name][2]
        for sd in SEEDS:
            s_ = cf[(name, sd)]['raw_va']
            b2.append((roc_auc_score(pv[1], s_), roc_auc_score(pv[1], s_ + BIG * pv[1])))
    a0 = float(np.mean([a for a, _ in b2])); a1 = float(np.mean([b for _, b in b2]))
    okB2 = abs(a1 - 1.0) <= 0.0001
    print(f'     B-2 AUC: 注入後 = {a1:.6f} / 注入量 = 1 − {a0:.6f} = {1 - a0:.6f}'
          f'（本物が無いとき = 0.000000）→ ±0.0001 か: {okB2}')
    out['control']['B'] = dict(b1=b1o, b1_ok=bool(okB1), auc=a0, auc_inj=a1, b2_ok=bool(okB2))
    if not (okB1 and okB2):
        stop.append(f'対照群B が復元しない（B-1 {okB1} / B-2 {okB2}）')

    # 6. 探索期の再現
    prev_p = os.path.join(base, 'hitrate', 'c_result.json')
    ok6 = True
    ex = {}
    for tau in TAUS:
        acc = []
        for name, _, _, _ in EXPLORE:
            for sd in SEEDS:
                tp = top_pick(ep, ef, name, sd)
                sw = tp[tp['p'] >= tau]
                if len(sw) == 0:
                    continue
                ax = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                ap = sw[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
                acc.append((roi(ax, pay_eval, 'fukusho')['rate'],
                            roi(ax, pay_eval, 'tansho')['roi'],
                            roi(ap, pay_eval, 'tansho')['roi'], len(sw)))
        ex[tau] = dict(rate=float(np.mean([a for a, _, _, _ in acc])),
                       roi_sw=float(np.mean([b for _, b, _, _ in acc])),
                       roi_p1=float(np.mean([c for _, _, c, _ in acc])),
                       n=float(np.sum([n for _, _, _, n in acc])) / len(SEEDS))
    if os.path.exists(prev_p):
        prev = json.load(open(prev_p))
        for tau in TAUS:
            pr = prev['c2'].get(str(tau)) or prev['c2'].get(str(float(tau)))
            if pr and pr['pool']:
                dv = abs(pr['pool']['roi_sw'] - ex[tau]['roi_sw'])
                print(f'  6. 探索期 τ={tau} 単勝回収率: 前回 {pr["pool"]["roi_sw"]:.2f}% '
                      f'→ 今回 {ex[tau]["roi_sw"]:.2f}%  差 {dv:.3f}pt')
                if dv > 0.01:
                    ok6 = False
        if not ok6:
            stop.append('探索期が前回と一致しない（再現性）')
    print(f'  6. 探索期の再現（±0.01pt）: {ok6}')

    print('-' * 78)
    print('中止条件: ' + ('発火なし（続行）' if not stop else '発火 → ' + ' / '.join(stop)))
    print('-' * 78)

    # ── 結果（確認期だけで判定する）─────────────────────────────────────
    print()
    print('=' * 78)
    print('■ 探索期（2025・判定に使わない。参考）')
    print('=' * 78)
    for tau in TAUS:
        e = ex[tau]
        print(f'  τ={tau:.2f}  差替 {e["n"]:.0f}レース  複勝率 {e["rate"]:.2f}%  '
              f'単勝回収 差替 {e["roi_sw"]:.2f}% / P1 {e["roi_p1"]:.2f}%')

    print()
    print('=' * 78)
    print('■ 確認期（2026-01-01〜2026-11-08）— ここだけで判定する')
    print('=' * 78)
    rng = np.random.default_rng(BOOT_SEED)
    res = {}
    for tau in TAUS:
        per_w, pv_all, p1_all, deltas = {}, [], [], []
        for name, _, _, _ in CONFIRM:
            wr = []
            for sd in SEEDS:
                tp = top_pick(cp, cf, name, sd)
                sw = tp[tp['p'] >= tau]
                if len(sw) == 0:
                    continue
                ax = sw[['r', 'n']].rename(columns={'r': 'race_id', 'n': 'horse_num'})
                ap = sw[['r', 'p1']].rename(columns={'r': 'race_id', 'p1': 'horse_num'})
                rs, rp = roi(ax, pay_eval, 'tansho'), roi(ap, pay_eval, 'tansho')
                fs = roi(ax, pay_eval, 'fukusho')
                wr.append(dict(n=len(sw), roi_sw=rs['roi'], roi_p1=rp['roi'],
                               win=rs['rate'], rate_f=fs['rate'], roif_sw=fs['roi']))
                if sd == SEEDS[0]:
                    vs, vp = payout_vec(ax, pay_eval), payout_vec(ap, pay_eval)
                    pv_all.append(vs); p1_all.append(vp)
                    m = min(len(vs), len(vp))
                    deltas.append(vs[:m] - vp[:m])
            per_w[name] = ({k: float(np.mean([w[k] for w in wr])) for k in wr[0]}
                           if wr else None)
        okw = [per_w[w] for w, _, _, _ in CONFIRM if per_w[w]]
        if not okw:
            print(f'  τ={tau:.2f}: 該当なし')
            res[tau] = None
            continue
        pool = {k: float(np.mean([w[k] for w in okw])) for k in okw[0]}
        n_tot = sum(w['n'] for w in okw)
        v = np.concatenate(pv_all) if pv_all else np.zeros(0)
        vp = np.concatenate(p1_all) if p1_all else np.zeros(0)
        # ブートストラップ（レース単位・10,000回）
        if len(v) > 0:
            idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
            boots = v[idx].mean(axis=1) / 100 * 100
            lo, hi = np.percentile(boots, [2.5, 97.5])
            se = float(np.std(boots))
        else:
            lo = hi = se = float('nan')
        point = float(v.mean()) if len(v) else float('nan')
        res[tau] = dict(pool=pool, n_total=n_tot, per_w=per_w, point=point,
                        lo=float(lo), hi=float(hi), se=se,
                        p1_point=float(vp.mean()) if len(vp) else float('nan'))
        print(f'  τ={tau:.2f}  差替 {n_tot:.0f}レース（{n_tot / sum(n_races) * 100:.2f}%）')
        print(f'        勝率 {pool["win"]:.2f}%  複勝率 {pool["rate_f"]:.2f}%  '
              f'複勝回収 {pool["roif_sw"]:.2f}%')
        print(f'        単勝回収  差替 {pool["roi_sw"]:.2f}%  /  P1 {pool["roi_p1"]:.2f}%  '
              f'（Δ{pool["roi_sw"] - pool["roi_p1"]:+.2f}pt）')
        print(f'        ブートストラップ95%CI = [{lo:.2f}%, {hi:.2f}%]  '
              f'幅 {hi - lo:.2f}pt（上限 {CI_WIDTH_MAX}pt）  SE {se:.2f}pt')
        s = '        窓ごと: '
        for w, _, _, _ in CONFIRM:
            q = per_w[w]
            s += (f'{w} n={q["n"]:5.1f} 単勝回収 {q["roi_sw"]:6.2f}% vs P1 {q["roi_p1"]:6.2f}% '
                  f'(Δ{q["roi_sw"] - q["roi_p1"]:+6.2f})  ' if q else f'{w} 該当なし  ')
        print(s)
        # 対照群A（符号反転）
        if deltas:
            dall = np.concatenate(deltas)
            g = np.random.default_rng(PERM_SEED)
            sh = [float(np.mean(dall * g.choice([-1.0, 1.0], len(dall))))
                  for _ in range(N_PERM)]
            nm, sd_ = float(np.mean(sh)), float(np.std(sh))
            print(f'        対照群A（符号反転{N_PERM}回・単勝払戻差）: 帰無平均 {nm:+.3f} / '
                  f'2*SE {2 * sd_ / np.sqrt(N_PERM):.3f} → 0中心か: '
                  f'{abs(nm) < 2 * sd_ / np.sqrt(N_PERM)}  実測 {float(dall.mean()):+.3f}円')
            res[tau]['permA'] = dict(null=nm, se=sd_ / np.sqrt(N_PERM),
                                     obs=float(dall.mean()))

    print()
    print('■ 合格基準（カード §4）')
    passed = False
    for tau in TAUS:
        r = res.get(tau)
        if not r:
            print(f'  τ={tau:.2f}: 該当なし → 不採用'); continue
        k1 = r['n_total'] >= 100
        k5 = (r['hi'] - r['lo']) <= CI_WIDTH_MAX
        if not k5:
            print(f'  τ={tau:.2f}: ⑤CI幅 {r["hi"] - r["lo"]:.2f}pt > {CI_WIDTH_MAX}pt '
                  f'→ **点推定を読まずに「N不足で判定不能」**')
            continue
        k2 = r['lo'] > 100.0
        k3 = (r['pool']['roi_sw'] - r['pool']['roi_p1']) >= 10.0
        k4 = all(r['per_w'][w] and
                 (r['per_w'][w]['roi_sw'] - r['per_w'][w]['roi_p1']) > 0
                 for w, _, _, _ in CONFIRM)
        allok = k1 and k2 and k3 and k4 and k5
        passed = passed or allok
        print(f'  τ={tau:.2f}: ①>=100レース: {k1} ②CI下限>100%({r["lo"]:.2f}%): {k2} '
              f'③P1比+10pt以上({r["pool"]["roi_sw"] - r["pool"]["roi_p1"]:+.2f}pt): {k3} '
              f'④3窓同符号: {k4} ⑤CI幅<=60pt({r["hi"] - r["lo"]:.2f}): {k5} '
              f'→ {"合格" if allok else "不採用"}')

    print()
    print('=' * 78)
    print('■ 判定')
    print('=' * 78)
    if stop:
        print('  ⚠ 中止条件が発火している: ' + ' / '.join(stop))
    print('  候補D: ' + ('**残った**（さらに別期間で確認する）' if (passed and not stop)
                        else '**不採用／判定不能**'))
    print('  🔴 回収率100%超は主張しない。本番の買い目・◎・印は変えない。')

    out['confirm'] = {str(t): (None if not res.get(t) else
                               {k: v for k, v in res[t].items()}) for t in TAUS}
    out['explore'] = {str(t): ex[t] for t in TAUS}
    out['stop'] = stop
    with open(os.path.join(base, 'hitrate', 'd_result.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n→ hitrate/d_result.json に保存した')


if __name__ == '__main__':
    main()
