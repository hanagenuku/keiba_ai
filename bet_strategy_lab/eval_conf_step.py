#!/usr/bin/env python3
"""`conf = 55` に段差があるのか（なめらかな坂なのか）を測る。

事前登録カード: bet_strategy_lab/CRITERIA_BET_02_conf_step.md（コミット済み）

🔴 **測定は1回**。ビン・窓・指標・合格線はカードのとおりで、結果を見てから変えない。
🔴 中止条件を**回収率より手前で**印字して判定する（logic2026/eval.py と同じ作り）。
🔴 本番の設定は1つも変えない。読むだけ。

使い方:
    python3 bet_strategy_lab/eval_conf_step.py <horse_features.csv>
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

# 🔑 学習・base_margin・軸の選び方・回収率は `logic2026/eval.py` の実装を
#    そのまま import する（同じ量を2か所で計算しない・2026-11-10 の教訓）。
from logic2026.eval import (  # noqa: E402
    _base_margin, _feat_cols, _xgb_params, pick_axis, roi,
    N_ROUNDS, EARLY_STOP,
)
from src.tools.train_xgb import load_popularity_drift  # noqa: E402

#: カード §3。(名前, 学習の終わり, 評価の始まり, 評価の終わり)
WINDOWS = (
    ('W1', '2024-06-30', '2024-07-01', '2024-11-24'),
    ('W2', '2025-03-31', '2025-04-01', '2025-09-30'),
    ('W3', '2025-12-31', '2026-01-01', '2026-06-30'),
)
SEEDS = (42, 43, 44)

#: カード §4。conf の5刻み。下限だけ持つ（上は次のビンの下限まで）
BIN_LO = (0, 40, 45, 50, 55, 60, 65, 70)
BIN_LABEL = ('<=39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '>=70')
#: カード §5。隣接差を取る境界
EDGES = (45, 50, 55, 60, 65, 70)

N_SHUFFLE = 200          # 対照群A
INJECT_PT = 10.00        # 対照群B


def load_features(path):
    df = pd.read_csv(path)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    return df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)


def load_payouts(db, lo, hi):
    conn = sqlite3.connect(db)
    q = """SELECT h.race_id, h.horse_num, h.place,
                  h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= ? AND r.date <= ?"""
    pay = pd.read_sql_query(q, conn, params=(lo, hi))
    conn.close()
    return pay


def run_window(df, cols, drift, train_end, val_start, val_end, seed):
    """1窓・1シード。軸の行（race_id, horse_num, conf）と軸AUCを返す。"""
    tr = df[df['date_obj'] <= train_end]
    va = df[(df['date_obj'] >= val_start) & (df['date_obj'] <= val_end)]
    if tr.empty or va.empty:
        return None, None
    Xtr, ytr = tr[cols].fillna(5.0), tr['is_fukusho'].values
    Xva, yva = va[cols].fillna(5.0), va['is_fukusho'].values
    spw = round((1 - ytr.mean()) / max(ytr.mean(), 0.01), 2)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=cols)
    dtr.set_base_margin(_base_margin(tr, drift, 11))
    dva.set_base_margin(_base_margin(va, drift, 12))
    bst = xgb.train(_xgb_params(spw, seed), dtr, num_boost_round=N_ROUNDS,
                    evals=[(dva, 'val')], early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
    margin = bst.predict(dva, output_margin=True)
    prob = 1 / (1 + np.exp(-margin))
    axis = pick_axis(va, prob)                       # race_id, horse_num, score
    axis['conf'] = (axis['score'] * 100).round().clip(0, 99).astype(int)
    return axis[['race_id', 'horse_num', 'conf']], roc_auc_score(yva, prob)


def bin_of(conf):
    i = 0
    for k, lo in enumerate(BIN_LO):
        if conf >= lo:
            i = k
    return i


def bin_table(axis, pay, kind='fukusho', bonus_from=None, bonus=0.0):
    """ビンごとの {n, hit, rate, roi, cut3, se}。

    bonus_from を渡すと、その conf 以上の行にだけ払戻へ bonus を足す（対照群B）。
    """
    out = []
    for i, lab in enumerate(BIN_LABEL):
        sel = axis[axis['conf'].map(bin_of) == i]
        if sel.empty:
            out.append({'label': lab, 'n': 0, 'roi': None, 'se': None,
                        'rate': None, 'cut3': None})
            continue
        b = bonus if (bonus_from is not None and BIN_LO[i] >= bonus_from) else 0.0
        r = roi(sel[['race_id', 'horse_num']], pay, kind=kind, bonus=b)
        # 回収率の SE（レース単位の払戻のばらつき）
        m = sel[['race_id', 'horse_num']].merge(pay, on=['race_id', 'horse_num'], how='left')
        col = f'{kind}_payout'
        hit = m['place'].between(1, 3) if kind == 'fukusho' else (m['place'] == 1)
        vals = np.where(hit, m[col].fillna(0).values, 0.0) + b
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
        out.append({'label': lab, 'n': r['n'], 'roi': r['roi'], 'rate': r['rate'],
                    'cut3': r['cut3'], 'se': round(se, 2) if se else None})
    return out


def deltas(tbl):
    """カード §5 の Δ(b)。境界 b の直上ビン − 直下ビン。"""
    by_lo = {BIN_LO[i]: tbl[i] for i in range(len(BIN_LO))}
    out = {}
    for b in EDGES:
        i = BIN_LO.index(b)
        up, dn = tbl[i], tbl[i - 1]
        out[b] = (round(up['roi'] - dn['roi'], 2)
                  if (up['roi'] is not None and dn['roi'] is not None) else None)
    return out


def main(argv):
    hf = argv[0] if argv else 'data/horse_features.csv'
    df = load_features(hf)
    cols = _feat_cols(df)
    drift = load_popularity_drift('.')
    print(f'特徴量 {len(df):,}行 / {df["race_id"].nunique():,}レース / {len(cols)}列')
    print(f'期間 {df["date_obj"].min().date()} 〜 {df["date_obj"].max().date()}')

    per_window = {}
    print('\n' + '=' * 74)
    print('■ 中止条件（回収率より手前で判定する）')
    print('=' * 74)
    stop = []
    for name, te, vs, ve in WINDOWS:
        pay = load_payouts('data/history.db', vs, ve)
        axes, aucs = [], []
        for sd in SEEDS:
            a, auc = run_window(df, cols, drift, te, vs, ve, sd)
            if a is None:
                continue
            axes.append(a)
            aucs.append(auc)
        if not axes:
            stop.append(f'{name}: 評価データが無い')
            continue
        nrace = axes[0]['race_id'].nunique()
        auc = float(np.mean(aucs))
        print(f'{name}: 評価 {nrace:,}レース / 軸AUC {auc:.4f}')
        if nrace < 800:
            stop.append(f'{name}: 評価レース {nrace} < 800（中止条件1）')
        if not (0.75 <= auc <= 0.82):
            stop.append(f'{name}: 軸AUC {auc:.4f} が 0.75〜0.82 の外（中止条件2）')
        per_window[name] = (axes, pay)

    # 中止条件4: 隣接ビンの N
    for name, (axes, pay) in per_window.items():
        t = bin_table(axes[0], pay)
        for lab in ('50-54', '55-59'):
            n = next(x['n'] for x in t if x['label'] == lab)
            if n < 300:
                stop.append(f'{name}: ビン {lab} の N={n} < 300（中止条件4）')

    # 対照群B（注入の復元）— 回収率を読む前に
    print('\n■ 対照群B（conf>=55 にだけ +10.00pt 注入して復元するか）')
    okB = True
    for name, (axes, pay) in per_window.items():
        base = deltas(bin_table(axes[0], pay))[55]
        inj = deltas(bin_table(axes[0], pay, bonus_from=55, bonus=INJECT_PT))[55]
        got = None if (base is None or inj is None) else round(inj - base, 4)
        ok = got is not None and abs(got - INJECT_PT) <= 0.05
        okB &= ok
        print(f'  {name}: Δ55 注入前 {base} → 注入後 {inj} / 増分 {got} '
              f'（注入量 {INJECT_PT:.2f}）{"✅" if ok else "❌"}')
    if not okB:
        stop.append('対照群B が +10.00pt を復元しない（中止条件3）')

    print('\n中止条件: ' + ('発火なし ✅' if not stop else '🔴 発火'))
    for s in stop:
        print('  🔴', s)
    if stop:
        print('\n回収率は読まない（カード §7）。')
        return 1

    # ── ここから本測定 ────────────────────────────────────────────────
    print('\n' + '=' * 74)
    print('■ ビンごとの複勝回収率（主指標）')
    print('=' * 74)
    all_d = {}
    for name, (axes, pay) in per_window.items():
        print(f'\n### {name}')
        print(f"{'ビン':<8}{'レース':>7}{'的中率':>8}{'複勝回収':>9}{'3本抜き':>9}{'SE':>7}")
        tbls = [bin_table(a, pay) for a in axes]
        avg = []
        for i, lab in enumerate(BIN_LABEL):
            rs = [t[i] for t in tbls if t[i]['roi'] is not None]
            if not rs:
                avg.append({'label': lab, 'n': 0, 'roi': None, 'se': None})
                continue
            d = {'label': lab,
                 'n': int(np.mean([x['n'] for x in rs])),
                 'rate': float(np.mean([x['rate'] for x in rs])),
                 'roi': float(np.mean([x['roi'] for x in rs])),
                 'cut3': float(np.mean([x['cut3'] for x in rs])),
                 'se': float(np.mean([x['se'] for x in rs]))}
            avg.append(d)
            print(f"{lab:<8}{d['n']:>7}{d['rate']:>7.1f}%{d['roi']:>8.1f}%"
                  f"{d['cut3']:>8.1f}%{d['se']:>7.2f}")
        all_d[name] = deltas(avg)

    print('\n' + '=' * 74)
    print('■ 隣接ビンの差 Δ(b)（カード §5 の合格線で判定）')
    print('=' * 74)
    print(f"{'窓':<5}" + ''.join(f'Δ({b}){"":>4}' for b in EDGES))
    for name in all_d:
        print(f'{name:<5}' + ''.join(
            f'{(all_d[name][b] if all_d[name][b] is not None else float("nan")):>8.2f}  '
            for b in EDGES))

    d55 = [all_d[n][55] for n in all_d if all_d[n][55] is not None]
    others = [abs(all_d[n][b]) for n in all_d for b in EDGES
              if b != 55 and all_d[n][b] is not None]
    c1 = bool(d55) and all(abs(x) > max(others) for x in d55)
    c2 = bool(d55) and all(x > 0 for x in d55)
    print(f'\n条件1 Δ(55) が他の |Δ| の最大({max(others):.2f}) を超える: '
          f'{"✅" if c1 else "❌"}  (Δ55 = {d55})')
    print(f'条件2 3窓すべて Δ(55) > 0: {"✅" if c2 else "❌"}')
    print(f'\n判定: {"段差あり" if (c1 and c2) else "🔴 段差なし（なめらかな坂）"}')

    # 対照群A（並べ替え）
    print('\n■ 対照群A（レース単位で conf をシャッフル × 200）')
    for name, (axes, pay) in per_window.items():
        rng = np.random.default_rng(0)
        a = axes[0]
        vals = []
        for _ in range(N_SHUFFLE):
            sh = a.copy()
            sh['conf'] = rng.permutation(sh['conf'].values)
            d = deltas(bin_table(sh, pay))[55]
            if d is not None:
                vals.append(d)
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        line = 2 * s / np.sqrt(len(vals))
        print(f'  {name}: 帰無 mean {m:+.3f} / 線 {line:.3f} '
              f'{"✅ 0中心" if abs(m) < line else "❌ 0中心でない"}'
              f'  (std {s:.2f}・実測 Δ55 {all_d[name][55]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
