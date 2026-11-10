#!/usr/bin/env python3
"""候補B の測定ハーネス（事前登録カード `hitrate/CRITERIA_B.md` のとおり）。

問い: 残差学習の early stopping が切片で止まっているのを直すと、完全 OOS の
      順位付けは良くなるか。

🔴 **測定は1回。** 閾値・腕・期間・母集団・指標はカードのとおりで、結果を見てから変えない。
🔴 中止条件は AUC・回収率を読む**手前**で印字して判定する（2026-10-19② の反省）。
🔴 隔離ワークスペースで実行する。`train_xgb` の保存処理が本番モデルを書き換えた事故が
   2026-09-15 に実際に起きている（このハーネスは1バイトも書かないが、場所は隔離する）。

使い方:
    cd <隔離ワークスペース>      # data/horse_features.csv と history.db がある場所
    python3 hitrate/eval_b.py
"""
import importlib.util
import itertools
import json
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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

# ── カード §4 の窓（retrain/ab.py と同一）──
WINDOWS = [('W1', '2026-07-31', '2026-08-01', '2026-08-31'),
           ('W2', '2026-08-31', '2026-09-01', '2026-09-30'),
           ('W3', '2026-09-30', '2026-10-01', '2026-10-18')]
SEEDS = (42, 7, 2026)
INNER_DAYS = 42
N_ROUNDS, ES_ROUNDS = 4000, 50
DRIFT_SEED_TRAIN, DRIFT_SEED_EVAL = 11, 12
INJECT_SIGMA = 1.8          # カード §7 対照群B-2（理論 AUC 0.653）
INJECT_SEED = 20261109

# ── カード §3 の腕（3つだけ）──
ARMS = {
    'P0': dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, spw='prod'),
    'P1': dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, spw=1.0),
    'P2': dict(max_depth=5, learning_rate=0.02, min_child_weight=300, subsample=0.8,
               colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1, spw=1.0),
}


def feat_cols(df):
    ex = set(_EXCLUDE_COLS) | {'date_obj', '_n_horses'} | set(_MARKET_FEAT_COLS)
    return [c for c in df.columns if c not in ex and df[c].dtype.kind in 'fi']


def base_margins(df, drift, seed):
    """全行ぶんの base_margin を一度だけ作る（腕・窓をまたいで同一配列を使う）。"""
    n = df.groupby('race_id')['horse_num'].transform('count')
    pop = df['f_popularity'].fillna(n / 2)
    if drift is not None:
        pop = pd.Series(_apply_popularity_drift(
            pop, df['race_id'].values, drift, seed=seed))
    return np.asarray(_popularity_to_base_margin(pop, n), dtype=float)


def fit_arm(arm, Xf, yf, bmf, Xi, yi, bmi, Xv, bmv, cols, seed):
    hp = dict(ARMS[arm])
    spw = hp.pop('spw')
    if spw == 'prod':
        spw = round((1 - yf.mean()) / max(yf.mean(), 0.01), 2)
    params = dict(hp, objective='binary:logistic', eval_metric='logloss',
                  scale_pos_weight=spw, seed=seed, nthread=-1)
    df_ = xgb.DMatrix(Xf, label=yf, feature_names=cols); df_.set_base_margin(bmf)
    di = xgb.DMatrix(Xi, label=yi, feature_names=cols); di.set_base_margin(bmi)
    dv = xgb.DMatrix(Xv, feature_names=cols); dv.set_base_margin(bmv)
    bst = xgb.train(params, df_, num_boost_round=N_ROUNDS, evals=[(di, 'inner_ho')],
                    early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    it, ntree = int(bst.best_iteration), bst.num_boosted_rounds()
    # 🔴 木の本数は版に依存させず明示する（カード §5）
    m_all = bst.predict(dv, output_margin=True, iteration_range=(0, ntree))
    m_best = bst.predict(dv, output_margin=True, iteration_range=(0, it + 1))
    sig = lambda m: 1 / (1 + np.exp(-m))          # noqa: E731
    return dict(it=it, ntree=ntree, spw=float(spw),
                p_all=sig(m_all), p_best=sig(m_best))


def main():
    base = os.getcwd()
    out = {'windows': [], 'arms': {}, 'inject': {}, 'control': {}}

    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = feat_cols(df)
    drift = load_popularity_drift(base)
    meta = json.load(open(os.path.join(base, 'data', 'xgb_feature_cols_residual.json')))
    prod_cols = list(meta['feature_cols'])

    X = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    y = df['is_fukusho'].to_numpy(dtype=np.int8)
    d = df['date_obj'].to_numpy()
    bm_tr_all = base_margins(df, drift, DRIFT_SEED_TRAIN)
    bm_ev_all = base_margins(df, drift, DRIFT_SEED_EVAL)

    # 注入列 z（対照群B-2）
    rng = np.random.default_rng(INJECT_SEED)
    z = y.astype(float) + rng.normal(0, INJECT_SIGMA, len(y))

    # ── 窓を切る ───────────────────────────────────────────────────────
    masks = {}
    for name, te, v0, v1 in WINDOWS:
        te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
        inner_from = te - pd.Timedelta(days=INNER_DAYS)
        m_fit = d <= np.datetime64(inner_from)
        m_in = (d > np.datetime64(inner_from)) & (d <= np.datetime64(te))
        m_va = (d >= np.datetime64(v0)) & (d <= np.datetime64(v1))
        masks[name] = (m_fit, m_in, m_va)
        out['windows'].append(dict(
            window=name, train_end=str(te.date()),
            inner_from=str((inner_from + pd.Timedelta(days=1)).date()),
            eval_from=str(v0.date()), eval_to=str(v1.date()),
            n_fit=int(m_fit.sum()), n_inner=int(m_in.sum()), n_eval=int(m_va.sum())))

    pay = None
    hp = os.path.join(base, 'data', 'history.db')
    con = sqlite3.connect(hp)
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= ? AND r.date <= ?""",
        con, params=(WINDOWS[0][2], WINDOWS[-1][3]))
    con.close()

    # ── 学習（全腕・全窓・全シード。印字はまだしない）────────────────────
    fits = {}
    for name, _, _, _ in WINDOWS:
        m_fit, m_in, m_va = masks[name]
        Xf, Xi, Xv = X[m_fit], X[m_in], X[m_va]
        yf, yi = y[m_fit], y[m_in]
        bmf, bmi, bmv = bm_tr_all[m_fit], bm_tr_all[m_in], bm_ev_all[m_va]
        Xfz = np.column_stack([Xf, z[m_fit]]).astype(np.float32)
        Xiz = np.column_stack([Xi, z[m_in]]).astype(np.float32)
        Xvz = np.column_stack([Xv, z[m_va]]).astype(np.float32)
        colsz = cols + ['z_inject']
        for arm in ARMS:
            for sd in SEEDS:
                fits[(arm, name, sd, 'plain')] = fit_arm(
                    arm, Xf, yf, bmf, Xi, yi, bmi, Xv, bmv, cols, sd)
                fits[(arm, name, sd, 'inject')] = fit_arm(
                    arm, Xfz, yf, bmf, Xiz, yi, bmi, Xvz, bmv, colsz, sd)
                print(f'  …fit {arm} {name} seed={sd} '
                      f'it={fits[(arm, name, sd, "plain")]["it"]}', flush=True)

    # ── 中止条件（AUC・回収率より手前）───────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §8）— AUC・回収率を読む手前で判定する')
    print('=' * 78)
    stop = []

    print('  1. 本番 data/ の md5 は呼び出し側（シェル）で測定前後に比較する。'
          ' このハーネスは data/ に1バイトも書かない。')

    p0_its = [fits[('P0', w, sd, 'plain')]['it'] for w, _, _, _ in WINDOWS for sd in SEEDS]
    ok2 = all(7 <= i <= 15 for i in p0_its)
    print(f'  2. P0 の best_iteration（9件）= {p0_its}  → 7〜15 の内か: {ok2}')
    if not ok2:
        stop.append(f'P0 の best_iteration が想定帯の外: {p0_its}')

    pooled = {}
    for arm in ARMS:
        for mode in ('plain', 'inject'):
            for tree in ('p_all', 'p_best'):
                for sd in SEEDS:
                    ys, ps = [], []
                    for w, _, _, _ in WINDOWS:
                        ys.append(y[masks[w][2]])
                        ps.append(fits[(arm, w, sd, mode)][tree])
                    pooled[(arm, mode, tree, sd)] = (np.concatenate(ys),
                                                     np.concatenate(ps))
    p0_auc = [roc_auc_score(*pooled[('P0', 'plain', 'p_all', sd)]) for sd in SEEDS]
    ok3 = all(abs(a - 0.7874) <= 0.01 for a in p0_auc)
    print(f'  3. P0 のプール AUC（全木）= {[round(a, 4) for a in p0_auc]}'
          f'  → retrain の N 0.7867〜0.7880 の ±0.01 以内か: {ok3}')
    if not ok3:
        stop.append(f'P0 のプール AUC が記録から外れた: {p0_auc}')

    ok4 = True
    for w, _, _, _ in WINDOWS:
        m_fit, m_in, m_va = masks[w]
        overlap = bool((m_fit & m_va).any() or (m_in & m_va).any())
        thin = int(m_in.sum()) < 1000
        ok4 = ok4 and not overlap and not thin
        print(f'  4. {w} 学習 {m_fit.sum():,} / 内側HO {m_in.sum():,} / 評価 {m_va.sum():,}'
              f'  評価窓との重なり: {overlap}  内側HOが1000行未満: {thin}')
    if not ok4:
        stop.append('内側HOが評価窓と重なる、または1000行未満')

    obj = 'binary:logistic'
    print(f'  5. objective = {obj}（binary:logistic 以外なら中止）')

    import hashlib
    ok6 = True
    for w, _, _, _ in WINDOWS:
        m_fit, m_in, m_va = masks[w]
        h = hashlib.md5(bm_ev_all[m_va].tobytes()).hexdigest()[:12]
        print(f'  6. {w} 評価窓の base_margin md5 {h}'
              f'（3腕とも同一配列を渡す実装なので構造的に一致）')
    print(f'  7. 特徴量列 {len(cols)}（本番 {len(prod_cols)}）'
          f'  集合一致: {set(cols) == set(prod_cols)}')
    if set(cols) != set(prod_cols):
        ok6 = False
        stop.append('特徴量列が本番の134列と一致しない')
        print('     差分 本-新:', sorted(set(prod_cols) - set(cols))[:10])
        print('     差分 新-本:', sorted(set(cols) - set(prod_cols))[:10])

    auc_z = roc_auc_score(np.concatenate([y[masks[w][2]] for w, _, _, _ in WINDOWS]),
                          np.concatenate([z[masks[w][2]] for w, _, _, _ in WINDOWS]))
    print(f'  （参考）注入列 z 単独の評価窓 AUC = {auc_z:.4f}'
          f'  → カードの 0.63〜0.68 の内か: {0.63 <= auc_z <= 0.68}')
    if not (0.63 <= auc_z <= 0.68):
        stop.append(f'注入列 z の AUC {auc_z:.4f} がカードの帯の外')

    if stop:
        print('\n🔴 中止条件に該当。結果を読まずに破棄する:')
        for s in stop:
            print('   -', s)
        json.dump({'stopped': stop}, open('hitrate_b_result.json', 'w'),
                  ensure_ascii=False, indent=2)
        return
    print('\n  ✅ 中止条件はすべて通過。測定に進む。')

    # ── 対照群B-1（算数の検算）──────────────────────────────────────────
    print('\n' + '=' * 78)
    print('■ 対照群B-1（注入の復元・算数の検算）')
    print('=' * 78)

    def axis_of(arm, sd, mode='plain', tree='p_all'):
        parts = []
        for w, _, _, _ in WINDOWS:
            m_va = masks[w][2]
            sub = df.loc[m_va, ['race_id', 'horse_num']]
            parts.append(pick_axis(sub, fits[(arm, w, sd, mode)][tree]))
        return pd.concat(parts, ignore_index=True)

    ax0 = axis_of('P0', SEEDS[0])
    r0, rinj = roi(ax0, pay), roi(ax0, pay, bonus=20.0)
    dlt = round(rinj['roi'] - r0['roi'], 4)
    okB1 = abs(dlt - 20.0) < 1e-3
    print(f'  注入前 {r0["roi"]}%  注入後 {rinj["roi"]}%  差 {dlt}pt'
          f'（注入量 20.0pt と一致すべき）→ {"✅" if okB1 else "🔴"}')
    truth = pay[pay['place'] == 1][['race_id', 'horse_num']]
    orc = roi(truth, pay, kind='tansho')
    exp = pay[(pay['place'] == 1) & (pay['tansho_payout'] > 0)]['tansho_payout'].mean()
    print(f'  検算(オラクル単勝): R={orc["roi"]}%  単勝払戻の平均/100={exp / 100 * 100:.2f}%'
          f'  → {"✅" if abs(orc["roi"] - exp) < 0.05 else "🔴"}')
    out['control']['B1_delta'] = dlt

    # ── 主測定 ─────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('■ 主測定: 3窓をプールした walk-forward（主＝全木 / 副＝best_iteration まで）')
    print('=' * 78)
    res = {}
    for tree in ('p_all', 'p_best'):
        print(f'\n── 予測に使う木: {"全木（本番の挙動・主）" if tree == "p_all" else "best_iteration まで（副・判定に使わない）"}')
        for arm in ARMS:
            rows = []
            for sd in SEEDS:
                yy, pp = pooled[(arm, 'plain', tree, sd)]
                ax = axis_of(arm, sd, 'plain', tree)
                r = roi(ax, pay)
                rows.append(dict(seed=sd, n=len(yy),
                                 auc=round(roc_auc_score(yy, pp), 4),
                                 brier=round(brier_score_loss(yy, pp), 4),
                                 logloss=round(log_loss(yy, pp), 4),
                                 trees=[fits[(arm, w, sd, 'plain')]['ntree']
                                        for w, _, _, _ in WINDOWS],
                                 its=[fits[(arm, w, sd, 'plain')]['it']
                                      for w, _, _, _ in WINDOWS],
                                 fuku=r))
                print(f'  {arm} seed={sd:<5} AUC {rows[-1]["auc"]}  '
                      f'Brier {rows[-1]["brier"]}  LogLoss {rows[-1]["logloss"]}  '
                      f'◎複勝 {r["hit"]}/{r["n"]} ({r["rate"]}%) 回収 {r["roi"]}% '
                      f'3本抜 {r["cut3"]}%  木 {rows[-1]["trees"]} (best {rows[-1]["its"]})')
            res[(arm, tree)] = rows
        print()
        for arm in ARMS:
            a = [r['auc'] for r in res[(arm, tree)]]
            rt = [r['fuku']['rate'] for r in res[(arm, tree)]]
            ro = [r['fuku']['roi'] for r in res[(arm, tree)]]
            c3 = [r['fuku']['cut3'] for r in res[(arm, tree)]]
            print(f'  {arm} 平均  AUC {np.mean(a):.4f} (std {np.std(a, ddof=1):.4f})  '
                  f'◎的中率 {np.mean(rt):.2f}%  回収 {np.mean(ro):.2f}%  '
                  f'3本抜 {np.mean(c3):.2f}%')

    # ── 窓ごと（基準2）────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('■ 窓ごとの ΔAUC（全木・3シード平均）— 基準2「3窓すべて同符号」')
    print('=' * 78)
    win_delta = {}
    for arm in ('P1', 'P2'):
        ds = []
        for w, _, _, _ in WINDOWS:
            m_va = masks[w][2]
            yv = y[m_va]
            a0 = np.mean([roc_auc_score(yv, fits[('P0', w, sd, 'plain')]['p_all'])
                          for sd in SEEDS])
            aa = np.mean([roc_auc_score(yv, fits[(arm, w, sd, 'plain')]['p_all'])
                          for sd in SEEDS])
            ds.append(aa - a0)
            print(f'  {arm} {w}: P0 {a0:.4f} → {arm} {aa:.4f}  Δ {aa - a0:+.4f}')
        win_delta[arm] = ds

    # ── 対照群A-1（アームの割り当ての並べ替え・符号反転の厳密分布）───────
    print('\n' + '=' * 78)
    print('■ 対照群A-1（ΔAUC・対応のある9差の符号反転 2^9=512 通りの厳密分布）')
    print('=' * 78)
    pair = {}
    for arm in ('P1', 'P2'):
        dd = []
        for w, _, _, _ in WINDOWS:
            yv = y[masks[w][2]]
            for sd in SEEDS:
                dd.append(roc_auc_score(yv, fits[(arm, w, sd, 'plain')]['p_all'])
                          - roc_auc_score(yv, fits[('P0', w, sd, 'plain')]['p_all']))
        dd = np.array(dd)
        signs = np.array(list(itertools.product([1, -1], repeat=len(dd))))
        null = (signs * dd).mean(axis=1)
        pval = float((np.abs(null) >= abs(dd.mean())).mean())
        pair[arm] = dict(diffs=[round(float(v), 5) for v in dd],
                         mean=float(dd.mean()), p=pval)
        print(f'  {arm}: 9差の平均 {dd.mean():+.5f}  '
              f'帰無の平均 {null.mean():+.6f}（0 で当然）  std {null.std():.5f}  '
              f'両側 P={pval:.4f}')
    out['control']['A1'] = pair

    # ── 対照群A-2（レース内シャッフル200回・回収率）──────────────────────
    print('\n' + '=' * 78)
    print('■ 対照群A-2（レース内で予測をシャッフル 200回・回収率）')
    print('=' * 78)
    rng2 = np.random.default_rng(20261109)
    base_roi = roi(ax0, pay)['roi']
    deltas = []
    sub_all = pd.concat([df.loc[masks[w][2], ['race_id', 'horse_num']]
                         for w, _, _, _ in WINDOWS], ignore_index=True)
    for _ in range(200):
        deltas.append(roi(pick_axis(sub_all, rng2.random(len(sub_all))), pay)['roi']
                      - base_roi)
    deltas = np.array(deltas)
    lim = 2 * deltas.std() / np.sqrt(len(deltas))
    print(f'  帰無の平均 {deltas.mean():+.3f}pt  std {deltas.std():.3f}'
          f'  合格線 |mean| < {lim:.3f} → {"✅" if abs(deltas.mean()) < lim else "🔴"}')
    print(f'  🔑 これは「無作為に軸を選んだ時の回収率」との差。帰無が0中心になるのは'
          f'**無作為と同じ**という意味ではない。')
    print(f'  実測 P0 − 無作為 = {-deltas.mean():+.3f}pt  並べ替え P値 = '
          f'{float((deltas >= 0).mean()):.4f}')
    out['control']['A2'] = dict(mean=float(deltas.mean()), std=float(deltas.std()),
                                limit=float(lim))

    # ── 対照群B-2（本物の信号を1列注入して回収できるか）────────────────────
    print('\n' + '=' * 78)
    print('■ 対照群B-2（市場が見ていない信号 z を1列注入・ΔAUC）')
    print('=' * 78)
    print(f'  z = is_fukusho + N(0, {INJECT_SIGMA})  単独 AUC {auc_z:.4f}')
    for arm in ARMS:
        a_no = np.mean([roc_auc_score(*pooled[(arm, 'plain', 'p_all', sd)]) for sd in SEEDS])
        a_z = np.mean([roc_auc_score(*pooled[(arm, 'inject', 'p_all', sd)]) for sd in SEEDS])
        its = [fits[(arm, w, sd, 'inject')]['it'] for w, _, _, _ in WINDOWS for sd in SEEDS]
        out['inject'][arm] = dict(auc_no=float(a_no), auc_z=float(a_z),
                                  delta=float(a_z - a_no), its=its)
        print(f'  {arm}: z なし {a_no:.4f} → z あり {a_z:.4f}  Δ {a_z - a_no:+.4f}'
              f'  （注入時の best_iteration {its}）')
    okB2 = all(out['inject'][a]['delta'] > 0 for a in ARMS)
    print(f'  合格（3腕すべて Δ>0）: {"✅" if okB2 else "🔴"}')

    # ── 採用基準の判定 ─────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('■ 採用基準（カード §6・5つすべて必要）')
    print('=' * 78)
    a0 = [r['auc'] for r in res[('P0', 'p_all')]]
    r0_rate = np.mean([r['fuku']['rate'] for r in res[('P0', 'p_all')]])
    r0_roi = np.mean([r['fuku']['roi'] for r in res[('P0', 'p_all')]])
    r0_c3 = np.mean([r['fuku']['cut3'] for r in res[('P0', 'p_all')]])
    verdict = {}
    for arm in ('P1', 'P2'):
        aa = [r['auc'] for r in res[(arm, 'p_all')]]
        dmean = float(np.mean(aa) - np.mean(a0))
        sd_seed = float(np.std(aa, ddof=1))
        rate = np.mean([r['fuku']['rate'] for r in res[(arm, 'p_all')]])
        r_roi = np.mean([r['fuku']['roi'] for r in res[(arm, 'p_all')]])
        r_c3 = np.mean([r['fuku']['cut3'] for r in res[(arm, 'p_all')]])
        c1 = dmean >= 0.002 and abs(dmean) >= 2 * sd_seed
        c2 = all(v > 0 for v in win_delta[arm])
        c3 = rate >= r0_rate
        c4 = (r_roi >= r0_roi - 0.5) and (r_c3 >= r0_c3 - 0.5)
        c5 = okB1 and okB2 and abs(deltas.mean()) < lim
        verdict[arm] = dict(delta_auc=dmean, seed_std=sd_seed, rate=float(rate),
                            roi=float(r_roi), cut3=float(r_c3),
                            c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4), c5=bool(c5))
        print(f'\n  ── {arm} ──  ΔAUC {dmean:+.4f}（シード std {sd_seed:.4f}）'
              f'  ◎的中率 {rate:.2f}%（P0 {r0_rate:.2f}%）'
              f'  回収 {r_roi:.2f}%（P0 {r0_roi:.2f}%）'
              f'  3本抜 {r_c3:.2f}%（P0 {r0_c3:.2f}%）')
        for lbl, okx in [('1 ΔAUC >= +0.002 かつ シードstdの2倍以上', c1),
                         ('2 3窓すべて同符号（正）', c2),
                         ('3 ◎の複勝的中率が P0 を下回らない', c3),
                         ('4 ◎の複勝回収率が P0 − 0.5pt 以上（素・3本抜きとも）', c4),
                         ('5 対照群A・B が通る', c5)]:
            print(f'     {"✅" if okx else "❌"} 基準{lbl}')
        allok = all([c1, c2, c3, c4, c5])
        print(f'     判定: {"採用候補（5基準すべて通過）" if allok else "不採用（基準未達）"}')

    out['arms'] = {f'{a}|{t}': res[(a, t)] for a in ARMS for t in ('p_all', 'p_best')}
    out['window_delta'] = {k: [float(v) for v in vv] for k, vv in win_delta.items()}
    out['verdict'] = verdict
    out['p0_auc'] = [float(v) for v in p0_auc]
    out['auc_z'] = float(auc_z)
    json.dump(out, open('hitrate_b_result.json', 'w'), ensure_ascii=False,
              indent=2, default=str)
    print('\n✅ hitrate_b_result.json に保存')


if __name__ == '__main__':
    main()
