#!/usr/bin/env python3
"""候補A の測定ハーネス（事前登録カード `hitrate/CRITERIA_A.md` のとおり）。

問い: ◎（`AI+市` 1位）が市場4番人気以下になるレースで軸を市場1番人気に替えると、
      複勝的中率は上がり、回収率は下がらないか。

🔴 **測定は1回。** 閾値・ルール・期間・母集団・指標はカードのとおりで、結果を見てから変えない。
🔴 中止条件は的中率・回収率を読む**手前**で印字して判定する。
🔴 隔離ワークスペースで実行する（本番の data/ には1バイトも書かない）。

使い方:
    cd <隔離ワークスペース>      # data/horse_features.csv と history.db がある場所
    python3 hitrate/eval_a.py
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

from src.tools.train_xgb import (  # noqa: E402
    _EXCLUDE_COLS, _MARKET_FEAT_COLS, _apply_popularity_drift,
    _popularity_to_base_margin, load_popularity_drift,
)

_spec = importlib.util.spec_from_file_location(
    'logic2026_eval', os.path.join(os.getcwd(), 'logic2026', 'eval.py'))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
roi = L.roi

# ── カード §5 の窓 ────────────────────────────────────────────────────
#  (名前, train_end, eval_start, eval_end, 役割)
WINDOWS = [
    ('M', '2024-12-31', '2025-01-01', '2025-12-31', '主判定'),
    ('C', '2023-12-31', '2024-01-01', '2024-12-31', '確認(報告のみ・学習1年)'),
    ('R', '2025-12-31', '2026-01-01', '2026-10-18', '参考(既読)'),
]
SEEDS = (42, 7, 2026)
INNER_DAYS = 42
N_ROUNDS, ES_ROUNDS = 4000, 50
DRIFT_SEED_TRAIN, DRIFT_SEED_EVAL = 11, 12
SWITCH_POP = 4          # カード §4: ◎が「4番人気以下」のとき替える
N_PERM = 200
HP = dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0)


def feat_cols(df):
    ex = set(_EXCLUDE_COLS) | {'date_obj'} | set(_MARKET_FEAT_COLS)
    return [c for c in df.columns if c not in ex and df[c].dtype.kind in 'fi']


def pick_by(race_id, horse_num, key, ascending):
    """レースごとに key が最小(ascending)／最大の馬を1頭選ぶ。同値は馬番の小さい方。"""
    d = pd.DataFrame({'race_id': race_id, 'horse_num': horse_num, 'k': key})
    d = d.sort_values(['race_id', 'k', 'horse_num'],
                      ascending=[True, ascending, True])
    return d.groupby('race_id', as_index=False).first()[['race_id', 'horse_num']]


def hit_and_roi(axis, pay, only_races=None, bonus=0.0):
    a = axis if only_races is None else axis[axis['race_id'].isin(only_races)]
    return roi(a[['race_id', 'horse_num']], pay, bonus=bonus)


def evaluable(axis, pay):
    """roi() が分母に入れるレース（結果があり、的中なら払戻もある）の集合。"""
    m = axis.merge(pay, on=['race_id', 'horse_num'], how='left')
    hit = m['place'].between(1, 3)
    has = m['place'].notna()
    bad = has & hit & (m['fukusho_payout'].fillna(0) <= 0)
    return set(m.loc[has & ~bad, 'race_id'])


def main():
    base = os.getcwd()
    out = {'windows': {}}

    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = feat_cols(df)
    drift = load_popularity_drift(base)
    prod_cols = list(json.load(open(os.path.join(
        base, 'data', 'xgb_feature_cols_residual.json')))['feature_cols'])

    X = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    y = df['is_fukusho'].to_numpy(dtype=np.int8)
    d = df['date_obj'].to_numpy()
    rid = df['race_id'].to_numpy()
    hnum = df['horse_num'].to_numpy()
    n_h = df.groupby('race_id')['horse_num'].transform('count')
    pop_fin = df['f_popularity'].fillna(n_h / 2)            # 確定人気
    pop_tr = pd.Series(_apply_popularity_drift(
        pop_fin, rid, drift, seed=DRIFT_SEED_TRAIN)) if drift is not None else pop_fin
    pop_ev = pd.Series(_apply_popularity_drift(
        pop_fin, rid, drift, seed=DRIFT_SEED_EVAL)) if drift is not None else pop_fin
    bm_tr = np.asarray(_popularity_to_base_margin(pop_tr, n_h), dtype=float)
    bm_ev = np.asarray(_popularity_to_base_margin(pop_ev, n_h), dtype=float)
    pop_ev_v = pop_ev.values.astype(float)

    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= '2024-01-01'""", con)
    con.close()

    # ── 学習（印字はまだしない）──────────────────────────────────────
    preds, masks, lastd, stop = {}, {}, {}, []
    for name, te, v0, v1, _role in WINDOWS:
        te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
        # 🔴 内側HOの起点は「学習期間に**実在する最後の日**」から42日前に取る。
        # カードは「学習期間の末尾42日」と登録したが、`train_end` から素朴に42日引くと
        # データの穴に当たる。実測: `history.db` には **2024-11-25〜2024-12-31 が1レースも無い**
        # （2024-12 は全期間で唯一まるごと欠けている月）。そのため M 窓の内側HOが
        # 661行になり、登録した中止条件3（1,000行以上）が1回目の実行で発火した。
        # **合格線は下げず、起点の取り方だけを穴に強い形に直した**（2026-10-21 と同じ扱い）。
        # 穴が無い窓では登録どおりの切り方とほぼ同じになる。結果は1行も読んでいない。
        last_tr = pd.Timestamp(d[d <= np.datetime64(te)].max())
        inner_from = last_tr - pd.Timedelta(days=INNER_DAYS)
        m_fit = d <= np.datetime64(inner_from)
        m_in = (d > np.datetime64(inner_from)) & (d <= np.datetime64(te))
        m_va = (d >= np.datetime64(v0)) & (d <= np.datetime64(v1))
        masks[name] = (m_fit, m_in, m_va)
        lastd[name] = (last_tr, inner_from)
        spw = round((1 - y[m_fit].mean()) / max(y[m_fit].mean(), 0.01), 2)
        params = dict(HP, objective='binary:logistic', eval_metric='logloss',
                      scale_pos_weight=spw, nthread=-1)
        dfit = xgb.DMatrix(X[m_fit], label=y[m_fit], feature_names=cols)
        dfit.set_base_margin(bm_tr[m_fit])
        din = xgb.DMatrix(X[m_in], label=y[m_in], feature_names=cols)
        din.set_base_margin(bm_tr[m_in])
        dva = xgb.DMatrix(X[m_va], feature_names=cols)
        dva.set_base_margin(bm_ev[m_va])
        for sd in SEEDS:
            p = dict(params, seed=sd)
            bst = xgb.train(p, dfit, num_boost_round=N_ROUNDS,
                            evals=[(din, 'inner_ho')],
                            early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
            # 予測は全木（本番の挙動・hitrate/RESULTS_B.md §3）
            m = bst.predict(dva, output_margin=True,
                            iteration_range=(0, bst.num_boosted_rounds()))
            preds[(name, sd)] = 1 / (1 + np.exp(-m))
            print(f'  …fit {name} seed={sd} best_iter={bst.best_iteration} '
                  f'trees={bst.num_boosted_rounds()}', flush=True)

    # ── 中止条件 ─────────────────────────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §9）— 的中率・回収率を読む手前で判定する')
    print('=' * 78)
    print('  1. 本番 data/ の md5 は呼び出し側で測定前後に比較する。'
          'このハーネスは data/ に1バイトも書かない。')
    for name, te, v0, v1, role in WINDOWS:
        m_fit, m_in, m_va = masks[name]
        last_tr, inner_from = lastd[name]
        leak = bool((d[m_fit | m_in] > np.datetime64(pd.Timestamp(te))).any())
        ov = bool((m_fit & m_va).any() or (m_in & m_va).any())
        print(f'       （{name} 学習期間の実在最終日 {last_tr.date()} → 内側HO は'
              f' {(inner_from + pd.Timedelta(days=1)).date()}〜{last_tr.date()}）')
        print(f'  2-3. {name}({role}) 学習 {m_fit.sum():,} / 内側HO {m_in.sum():,} /'
              f' 評価 {m_va.sum():,}  学習に評価窓の行: {leak}  重なり: {ov}'
              f'  内側HO<1000: {m_in.sum() < 1000}')
        if leak or ov or m_in.sum() < 1000:
            stop.append(f'{name}: 窓の切り方が不正')
    yM = y[masks['M'][2]]
    aucM = [roc_auc_score(yM, preds[('M', s)]) for s in SEEDS]
    auc_bm = roc_auc_score(yM, bm_ev[masks['M'][2]])
    print(f'  4. 主判定窓 M の R0 の AUC = {[round(a, 4) for a in aucM]}'
          f'  → 0.74〜0.82 の内か: {all(0.74 <= a <= 0.82 for a in aucM)}')
    if not all(0.74 <= a <= 0.82 for a in aucM):
        stop.append(f'M の AUC が想定帯の外: {aucM}')
    print(f'  5. M の土台（ドリフト入り base_margin）AUC = {auc_bm:.4f}'
          f'  → R0 がこれを下回らないか: {all(a >= auc_bm for a in aucM)}')
    if not all(a >= auc_bm for a in aucM):
        stop.append('R0 が自分の土台を下回った')
    print(f'  6. 特徴量列 {len(cols)}（本番 {len(prod_cols)}）'
          f'  集合一致: {set(cols) == set(prod_cols)}')
    if set(cols) != set(prod_cols):
        stop.append('特徴量列が本番と一致しない')
    dev = float(np.abs(pop_ev_v - pop_fin.values).mean())
    unchanged = float((pop_ev_v == pop_fin.values).mean())
    print(f'  7. 帯の判定に使う人気はドリフト注入後。確定人気との平均絶対ズレ'
          f' {dev:.3f}位 / 順位が変わらない馬 {unchanged * 100:.1f}%'
          f'  → 注入されているか: {dev > 0.5}')
    if not dev > 0.5:
        stop.append('ドリフトが注入されていない（確定人気で判定してしまう）')
    if stop:
        print('\n🔴 中止条件に該当。結果を読まずに破棄する:')
        for s in stop:
            print('   -', s)
        json.dump({'stopped': stop}, open('hitrate_a_result.json', 'w'),
                  ensure_ascii=False, indent=2)
        return
    print('\n  ✅ 中止条件はすべて通過。測定に進む。')

    # ── 各窓の測定 ────────────────────────────────────────────────────
    for name, te, v0, v1, role in WINDOWS:
        m_va = masks[name][2]
        sub_r, sub_h, sub_p = rid[m_va], hnum[m_va], pop_ev_v[m_va]
        dates = pd.Series(d[m_va])
        print('\n' + '=' * 78)
        print(f'■ 窓 {name}（{role}）  評価 {v0}〜{v1}  {m_va.sum():,}行 '
              f'{len(set(sub_r)):,}レース')
        print('=' * 78)
        W = {}
        for sd in SEEDS:
            p = preds[(name, sd)]
            ax0 = pick_by(sub_r, sub_h, -p, True)                 # R0: 予測最大
            fav = pick_by(sub_r, sub_h, sub_p, True)              # R2: 人気最上位
            ax0p = ax0.merge(pd.DataFrame({'race_id': sub_r, 'horse_num': sub_h,
                                           'pop': sub_p}),
                             on=['race_id', 'horse_num'], how='left')
            sw = set(ax0p.loc[ax0p['pop'] >= SWITCH_POP, 'race_id'])        # 替える
            mid = set(ax0p.loc[ax0p['pop'].between(2, 3), 'race_id'])       # 2-3番人気
            same = int((ax0p['pop'] == 1).sum())
            ax1 = pd.concat([ax0[~ax0['race_id'].isin(sw)],
                             fav[fav['race_id'].isin(sw)]], ignore_index=True)
            ax_mid = pd.concat([ax0[~ax0['race_id'].isin(mid)],
                                fav[fav['race_id'].isin(mid)]], ignore_index=True)
            e0, e1, e2 = (evaluable(ax0, pay), evaluable(ax1, pay),
                          evaluable(fav, pay))
            com01 = e0 & e1
            com012 = e0 & e1 & e2
            r0 = hit_and_roi(ax0, pay, com01)
            r1 = hit_and_roi(ax1, pay, com01)
            g0 = hit_and_roi(ax0, pay, com012)
            g1 = hit_and_roi(ax1, pay, com012)
            g2 = hit_and_roi(fav, pay, com012)
            rm = hit_and_roi(ax_mid, pay, e0 & evaluable(ax_mid, pay))
            W[sd] = dict(n_races=len(set(sub_r)), n_switch=len(sw), same_fav=same,
                         n_mid=len(mid), common01=len(com01), common012=len(com012),
                         R0=r0, R1=r1, G0=g0, G1=g1, G2=g2, MID=rm,
                         auc=round(roc_auc_score(y[m_va], p), 4),
                         ax0=ax0, ax1=ax1, fav=fav, sw=sw, mid=mid, p=p,
                         sub_r=sub_r, sub_h=sub_h, dates=dates)
            print(f'  seed={sd:<5} AUC {W[sd]["auc"]}  替えるレース {len(sw)}'
                  f'（AI1位が既に1番人気 {same} / 2-3番人気 {len(mid)}）'
                  f'  共通集合 {len(com01)}レース')
            print(f'          R0 的中 {r0["hit"]}/{r0["n"]} ({r0["rate"]}%) '
                  f'回収 {r0["roi"]}% 3本抜 {r0["cut3"]}%')
            print(f'          R1 的中 {r1["hit"]}/{r1["n"]} ({r1["rate"]}%) '
                  f'回収 {r1["roi"]}% 3本抜 {r1["cut3"]}%'
                  f'   Δ的中 {r1["rate"] - r0["rate"]:+.2f}pt '
                  f'Δ回収 {r1["roi"] - r0["roi"]:+.2f}pt '
                  f'Δ3本抜 {r1["cut3"] - r0["cut3"]:+.2f}pt')
        out['windows'][name] = {str(s): {k: v for k, v in W[s].items()
                                         if k not in ('ax0', 'ax1', 'fav', 'sw',
                                                      'mid', 'p', 'sub_r', 'sub_h',
                                                      'dates')}
                                for s in SEEDS}

        def avg(key, field):
            return float(np.mean([W[s][key][field] for s in SEEDS]))

        d_rate = avg('R1', 'rate') - avg('R0', 'rate')
        d_roi = avg('R1', 'roi') - avg('R0', 'roi')
        d_c3 = avg('R1', 'cut3') - avg('R0', 'cut3')
        print(f'\n  ── 3シード平均（共通集合 R0∩R1）──')
        print(f'     R0 的中 {avg("R0", "rate"):.2f}%  回収 {avg("R0", "roi"):.2f}%'
              f'  3本抜 {avg("R0", "cut3"):.2f}%')
        print(f'     R1 的中 {avg("R1", "rate"):.2f}%  回収 {avg("R1", "roi"):.2f}%'
              f'  3本抜 {avg("R1", "cut3"):.2f}%')
        print(f'     Δ 的中 {d_rate:+.2f}pt  回収 {d_roi:+.2f}pt  3本抜 {d_c3:+.2f}pt')
        print(f'  ── 市場ゲート（共通集合 R0∩R1∩R2）──')
        print(f'     R0 的中 {avg("G0", "rate"):.2f}%  回収 {avg("G0", "roi"):.2f}%'
              f'  3本抜 {avg("G0", "cut3"):.2f}%')
        print(f'     R1 的中 {avg("G1", "rate"):.2f}%  回収 {avg("G1", "roi"):.2f}%'
              f'  3本抜 {avg("G1", "cut3"):.2f}%')
        print(f'     R2 的中 {avg("G2", "rate"):.2f}%  回収 {avg("G2", "roi"):.2f}%'
              f'  3本抜 {avg("G2", "cut3"):.2f}%  ← 常に市場1番人気')
        print(f'  ── 参考: 2〜3番人気のセルで同じ置き換え ──')
        print(f'     的中 {avg("MID", "rate"):.2f}%  回収 {avg("MID", "roi"):.2f}%'
              f'  3本抜 {avg("MID", "cut3"):.2f}%')
        out['windows'][name]['avg'] = dict(
            d_rate=d_rate, d_roi=d_roi, d_cut3=d_c3,
            R0_rate=avg('R0', 'rate'), R1_rate=avg('R1', 'rate'),
            R0_roi=avg('R0', 'roi'), R1_roi=avg('R1', 'roi'),
            R0_cut3=avg('R0', 'cut3'), R1_cut3=avg('R1', 'cut3'),
            G0_rate=avg('G0', 'rate'), G1_rate=avg('G1', 'rate'),
            G2_rate=avg('G2', 'rate'), G0_roi=avg('G0', 'roi'),
            G1_roi=avg('G1', 'roi'), G2_roi=avg('G2', 'roi'),
            G2_cut3=avg('G2', 'cut3'),
            MID_rate=avg('MID', 'rate'), MID_roi=avg('MID', 'roi'),
            n_switch=float(np.mean([W[s]['n_switch'] for s in SEEDS])))

        if name != 'M':
            continue

        # ── 以下は主判定窓だけ ────────────────────────────────────────
        Ws = W[SEEDS[0]]
        print('\n  ── 基準3: 前半(1〜6月)・後半(7〜12月)（seed 42）──')
        rr = pd.DataFrame({'race_id': Ws['sub_r'], 'dt': Ws['dates'].values})
        first = set(rr.loc[rr['dt'] < np.datetime64('2025-07-01'), 'race_id'])
        half = {}
        for lab, sel in (('前半', first),
                         ('後半', set(rr['race_id']) - first)):
            c = (evaluable(Ws['ax0'], pay) & evaluable(Ws['ax1'], pay)) & sel
            a = hit_and_roi(Ws['ax0'], pay, c)
            b = hit_and_roi(Ws['ax1'], pay, c)
            half[lab] = dict(n=a['n'], d_rate=b['rate'] - a['rate'],
                             d_roi=b['roi'] - a['roi'],
                             d_cut3=b['cut3'] - a['cut3'])
            print(f'     {lab} N={a["n"]:4d}  R0 {a["rate"]}%/{a["roi"]}%  '
                  f'R1 {b["rate"]}%/{b["roi"]}%  Δ的中 {b["rate"] - a["rate"]:+.2f}pt'
                  f'  Δ回収 {b["roi"] - a["roi"]:+.2f}pt'
                  f'  Δ3本抜 {b["cut3"] - a["cut3"]:+.2f}pt')
        out['half'] = half

        # ── 対照群B-1 ─────────────────────────────────────────────────
        print('\n  ── 対照群B-1（注入の復元・算数の検算）──')
        com = evaluable(Ws['ax0'], pay) & evaluable(Ws['ax1'], pay)
        b0 = hit_and_roi(Ws['ax0'], pay, com)
        bi = hit_and_roi(Ws['ax0'], pay, com, bonus=20.0)
        dlt = round(bi['roi'] - b0['roi'], 4)
        print(f'     注入前 {b0["roi"]}% → 注入後 {bi["roi"]}%  差 {dlt}pt'
              f'（20.0 と一致すべき）→ {"✅" if abs(dlt - 20.0) < 1e-3 else "🔴"}')
        okB1 = abs(dlt - 20.0) < 1e-3

        # ── 対照群B-2（オラクル）──────────────────────────────────────
        print('  ── 対照群B-2（替えるレースで軸を実際の1着馬に置換＝オラクル）──')
        win = pay[pay['place'] == 1][['race_id', 'horse_num']]
        ax_or = pd.concat([Ws['ax0'][~Ws['ax0']['race_id'].isin(Ws['sw'])],
                           win[win['race_id'].isin(Ws['sw'])]], ignore_index=True)
        c_or = com & evaluable(ax_or, pay)
        o0 = hit_and_roi(Ws['ax0'], pay, c_or)
        o1 = hit_and_roi(ax_or, pay, c_or)
        sw_only = hit_and_roi(Ws['ax0'], pay, Ws['sw'] & c_or)
        k, n = sw_only['n'], o0['n']
        expect = k / n * (100.0 - sw_only['rate']) if n else 0.0
        print(f'     替えるレース内の R0 的中率 {sw_only["rate"]}% (N={k})'
              f'  → オラクルは 100%')
        print(f'     全レースの的中率 {o0["rate"]}% → {o1["rate"]}%'
              f'（Δ {o1["rate"] - o0["rate"]:+.2f}pt / 期待 {expect:+.2f}pt）'
              f'  → {"✅" if abs((o1["rate"] - o0["rate"]) - expect) < 0.05 else "🔴"}')
        okB2 = abs((o1['rate'] - o0['rate']) - expect) < 0.05

        # ── 対照群A ───────────────────────────────────────────────────
        print(f'\n  ── 対照群A（無作為に同数 K={len(Ws["sw"])} レースを1番人気に替える×'
              f'{N_PERM}回）──')
        rng = np.random.default_rng(20261110)
        all_races = np.array(sorted(set(Ws['sub_r'])))
        base_rate = b0['rate']
        ds = []
        for _ in range(N_PERM):
            pick = set(rng.choice(all_races, size=len(Ws['sw']), replace=False))
            axp = pd.concat([Ws['ax0'][~Ws['ax0']['race_id'].isin(pick)],
                             Ws['fav'][Ws['fav']['race_id'].isin(pick)]],
                            ignore_index=True)
            cp = evaluable(Ws['ax0'], pay) & evaluable(axp, pay)
            ds.append(hit_and_roi(axp, pay, cp)['rate']
                      - hit_and_roi(Ws['ax0'], pay, cp)['rate'])
        ds = np.array(ds)
        obs = half and (W[SEEDS[0]]['R1']['rate'] - W[SEEDS[0]]['R0']['rate'])
        pval = float((ds >= obs).mean())
        print(f'     帰無（無作為置換）の Δ的中率: 平均 {ds.mean():+.3f}pt  '
              f'std {ds.std():.3f}  5〜95%点 [{np.percentile(ds, 5):+.2f},'
              f' {np.percentile(ds, 95):+.2f}]')
        print(f'     実測（このセルを選んで替える・seed42）= {obs:+.3f}pt'
              f'  並べ替え P = {pval:.4f}')
        print(f'     🔑 これは「替えること」ではなく「**このセルを選んで**替えること」に'
              f'意味があるかを測る帰無。')
        out['controlA'] = dict(mean=float(ds.mean()), std=float(ds.std()),
                               p5=float(np.percentile(ds, 5)),
                               p95=float(np.percentile(ds, 95)),
                               observed=float(obs), p=pval)
        out['controlB'] = dict(B1_delta=dlt, B2_ok=bool(okB2),
                               B2_expect=float(expect),
                               B2_actual=float(o1['rate'] - o0['rate']))

        # ── 参考: 替えるレースのうち推奨レース（conf>=66）は何本か ─────
        print('\n  ── 参考: 替えるレースのうち推奨レース（conf>=66）は何本か ──')
        from src.features.engine import calc_harville_probs
        from src.models.predict import softmax_probs
        confs = {}
        order = np.argsort(Ws['sub_r'], kind='stable')
        p42 = Ws['p']
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and Ws['sub_r'][order[j]] == Ws['sub_r'][order[i]]:
                j += 1
            idx = order[i:j]
            tot = [float(p42[t] * 10) for t in idx]
            wp = softmax_probs(tot, temperature=3.5)
            hv = calc_harville_probs(wp)
            best = int(np.argmax([p42[t] for t in idx]))
            confs[Ws['sub_r'][idx[0]]] = int(round(hv[best][1] * 100))
            i = j
        sw_conf = [confs[r] for r in Ws['sw'] if r in confs]
        print(f'     ⚠ 本番の conf の**近似**（相対ブレンド10%・pace_bonus を含まない）。'
              f'seed 42 のみ。')
        print(f'     替えるレース {len(Ws["sw"])} のうち conf>=66 は '
              f'{sum(1 for c in sw_conf if c >= 66)} 本'
              f' / conf>=55 は {sum(1 for c in sw_conf if c >= 55)} 本')
        allc = list(confs.values())
        print(f'     （全レース {len(allc)} のうち conf>=66 は '
              f'{sum(1 for c in allc if c >= 66)} 本）')
        out['conf'] = dict(switch_total=len(Ws['sw']),
                           switch_conf66=sum(1 for c in sw_conf if c >= 66),
                           switch_conf55=sum(1 for c in sw_conf if c >= 55),
                           all_races=len(allc),
                           all_conf66=sum(1 for c in allc if c >= 66))

        # ── 採用基準 ─────────────────────────────────────────────────
        print('\n' + '=' * 78)
        print('■ 採用基準（カード §7・6つすべて必要・主判定窓 M だけで判定）')
        print('=' * 78)
        c1 = d_rate >= 2.0
        c2 = (d_roi >= 0) and (d_c3 >= 0)
        # カード §7 基準3 は「基準1・2 と**同符号**」。各半期に +2.0pt を要求しない
        c3 = all((h['d_rate'] > 0) and (h['d_roi'] >= 0) and (h['d_cut3'] >= 0)
                 for h in half.values())
        c4 = np.mean([W[s]['n_switch'] for s in SEEDS]) >= 300
        c5 = okB1 and okB2 and ds.std() > 0
        c6 = not (avg('G2', 'rate') >= avg('G1', 'rate')
                  and avg('G2', 'roi') >= avg('G1', 'roi'))
        for lbl, ok in [(f'1 複勝的中率 R1−R0 >= +2.0pt（実測 {d_rate:+.2f}pt）', c1),
                        (f'2 複勝回収率 R1−R0 >= 0（実測 {d_roi:+.2f}pt / '
                         f'3本抜 {d_c3:+.2f}pt）', c2),
                        ('3 前半・後半とも 1・2 と同符号', c3),
                        (f'4 替えるレース N >= 300（実測 '
                         f'{np.mean([W[s]["n_switch"] for s in SEEDS]):.0f}）', c4),
                        ('5 対照群A・B が通る', c5),
                        (f'6 市場ゲート: R2 が R1 以上でない'
                         f'（R2 的中 {avg("G2", "rate"):.2f}% / 回収 '
                         f'{avg("G2", "roi"):.2f}% vs R1 {avg("G1", "rate"):.2f}% / '
                         f'{avg("G1", "roi"):.2f}%）', c6)]:
            print(f'   {"✅" if ok else "❌"} 基準{lbl}')
        allok = all([c1, c2, c3, c4, c5, c6])
        print(f'\n  判定: {"採用候補（6基準すべて通過）" if allok else "不採用（基準未達）"}')
        if not c6:
            print('  🔴 基準6 が落ちた＝「◎を AI で選ぶ意味が的中率の面では無い」。'
                  'カード §7 のとおり**採否を決めずにユーザーへ報告する**。')
        out['verdict'] = dict(c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4),
                              c5=bool(c5), c6=bool(c6), adopt=bool(allok))

    json.dump(out, open('hitrate_a_result.json', 'w'), ensure_ascii=False,
              indent=2, default=str)
    print('\n✅ hitrate_a_result.json に保存')


if __name__ == '__main__':
    main()
