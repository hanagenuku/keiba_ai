#!/usr/bin/env python3
"""☆モデル（「市場1番人気を着順で上回る確率」）を学習して本番の成果物を作る。

出力（`data/` に3つ。どれか欠けると `src/features/star.py` は☆を出さない）:
    data/star_model.ubj            XGBoost の Booster
    data/star_calibrator.pkl       Isotonic 較正器
    data/star_feature_cols.json    特徴量の列（順序つき）と学習の記録

🔴 **行列の作り方は `src/features/star.py` の関数だけを使う**（学習/推論パリティ）。
   列順・欠損の埋め方をこのファイルに書かない。
🔴 目的変数は `y = 1{着順(X) < 着順(P1)}`、特徴量は **P1 との差分だけ**。
   オッズ・人気を特徴量に入れない（North Star #1）。市場は「誰が P1 か」という
   土台としてだけ入る（`base_margin` と同じ役回り）。
🟢 v2（2026-11-20・hitrate/CRITERIA_STAR_V2.md）: 実際の1番人気（確定人気）と比べる・web 列を入れる・
   τ1（S1）と τ2（S2＝3着内の上積み）を内側ホールドアウトで決めて star_feature_cols.json に残す。

根拠: `hitrate/CRITERIA_C_RERUN.md` / `hitrate/CRITERIA_STAR.md` と
      `hitrate/RESULTS_C_RERUN.md` / `hitrate/RESULTS_STAR.md`

使い方（`data/horse_features.csv` と `data/history.db` がある場所で）:
    python3 src/tools/train_star.py
"""
import json
import os
import pickle
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

from src.features.star import (  # noqa: E402
    STAR_FILL, build_delta, matrix_from_frame, p1_index,
)
from src.tools.train_xgb import (  # noqa: E402
    _EXCLUDE_COLS, _MARKET_FEAT_COLS,
)

# 測定時と同じ設定（`hitrate/eval_c.py` の C-2）。結果を見てから変えない。
HP = dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0)
N_ROUNDS, ES_ROUNDS = 4000, 50
SEED = 42
INNER_DAYS = 42
POP_LO, POP_HI = 4, 9   # ☆の候補（hitrate/CRITERIA_STAR_NARROW.md・2026-11-20 博士「4〜9人気でよい」）
Q_TAU = 85          # τ1・τ2 は内側HOのレース内最大値の85パーセンタイル（hitrate/CRITERIA_STAR_V2.md）


def feat_cols(df):
    """市場列と除外列を抜いた数値列。本番 residual モデルの `feature_cols` から web 由来の列を除いたものと一致する。

    🔴 ☆は web 由来の列（`f_ws_*`・2026-11-13 採用）を使わない（`tests/test_star.py` が守る）。
       2026-11-18 の作り直しで、学習データに web 列が入ったまま ☆ を学習してしまったので明示的に除く。
    """
    ex = set(_EXCLUDE_COLS) | {'date_obj', '_n_horses'} | set(_MARKET_FEAT_COLS)
    return [c for c in df.columns if c not in ex and df[c].dtype.kind in 'fi']


def build_pairs(df, X, place, cols):
    """P1 基準の差分データ。`star.py::p1_index` / `build_delta` を使う。"""
    dXs, ys, meta = [], [], []
    bad = 0
    for rid, idx in df.groupby('race_id', sort=False).indices.items():
        idx = np.asarray(idx)
        i1 = p1_index(list(df['_pop'].values[idx]))
        if i1 is None or not np.isfinite(place[idx][i1]) or place[idx][i1] <= 0:
            bad += 1
            continue
        d = build_delta(X[idx], i1)
        pl = place[idx]
        ok = np.isfinite(pl) & (pl > 0)
        ok[i1] = False
        if not ok.any():
            bad += 1
            continue
        dXs.append(d[ok])
        ys.append((pl[ok] < pl[i1]).astype(np.int8))
        meta.append(len(ok.nonzero()[0]))
    if not dXs:
        raise SystemExit('ペアが1件も作れなかった')
    return np.vstack(dXs), np.concatenate(ys), bad, len(meta)


def main():
    base = os.getcwd()
    t0 = time.time()
    csv = os.path.join(base, 'data', 'horse_features.csv')
    if not os.path.exists(csv):
        raise SystemExit(f'{csv} が無い。build_training_data() を先に走らせること')

    df = pd.read_csv(csv)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj']).reset_index(drop=True)
    cols = feat_cols(df)
    print(f'  学習データ {len(df):,}行 / {df.race_id.nunique():,}レース / 列 {len(cols)}')

    # 本番 residual モデルの列と一致しているか（ずれたら止める）
    meta_p = os.path.join(base, 'data', 'xgb_feature_cols_residual.json')
    if os.path.exists(meta_p):
        with open(meta_p, encoding='utf-8') as f:
            prod = list(json.load(f)['feature_cols'])
        if set(prod) != set(cols):
            raise SystemExit(
                f'本番 residual の列と一致しない（本番 {len(prod)} / ここ {len(cols)}）。'
                f'差: {sorted(set(prod) ^ set(cols))[:10]}')
        cols = prod       # 🔴 列順も本番に合わせる
        print(f'  ✓ 本番 residual モデルの列順に合わせた（{len(cols)}列）')

    # 🔴 v2: **実際の1番人気（確定人気）**と比べる。v1 はドリフトを注入した作り物の1番人気と
    #    比べていて、実際の1番人気と一致したのは ☆ が出たレースの 27〜30% だけだった
    #    （biz/RESULTS_HONMEI_STAR.md §2）。
    n = df.groupby('race_id')['horse_num'].transform('count')
    df['_pop'] = np.asarray(df['f_popularity'].fillna(n / 2), dtype=float)

    # 着順
    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        'SELECT race_id, horse_num, place FROM horse_history WHERE place > 0', con)
    con.close()
    pm = {(r, h): p for r, h, p in zip(pay['race_id'], pay['horse_num'], pay['place'])}
    place = np.array([pm.get((r, h), np.nan)
                      for r, h in zip(df['race_id'], df['horse_num'])], dtype=float)

    X = matrix_from_frame(df, cols)          # 🔴 star.py の関数だけで作る
    print(f'  行列 {X.shape}（欠損は {STAR_FILL} で埋める。web 列は NaN のまま）')

    # 内側ホールドアウト42日。起点は「実在する最後の日」から（2024-12 の穴対策）
    d = df['date_obj'].to_numpy()
    last = pd.Timestamp(d.max())
    inner_from = last - pd.Timedelta(days=INNER_DAYS)
    m_fit = d <= np.datetime64(inner_from)
    m_in = d > np.datetime64(inner_from)
    print(f'  学習 {int(m_fit.sum()):,}行 / 内側HO {int(m_in.sum()):,}行'
          f'（{(inner_from + pd.Timedelta(days=1)).date()} 〜 {last.date()}）')
    if int(m_in.sum()) < 1000:
        raise SystemExit('内側ホールドアウトが1000行未満。期間を確かめること')

    dXf, yf, bad_f, nr_f = build_pairs(df[m_fit], X[m_fit], place[m_fit], cols)
    dXi, yi, bad_i, nr_i = build_pairs(df[m_in], X[m_in], place[m_in], cols)
    print(f'  ペア 学習 {len(yf):,}（{nr_f:,}レース・P1不定 {bad_f}）/ '
          f'内側HO {len(yi):,}（{nr_i:,}レース・P1不定 {bad_i}）')
    print(f'  正例率 学習 {yf.mean():.4f} / 内側HO {yi.mean():.4f}')

    params = dict(HP, objective='binary:logistic', eval_metric='logloss',
                  scale_pos_weight=1.0, seed=SEED, nthread=-1)
    dtr = xgb.DMatrix(dXf, label=yf, feature_names=list(cols))
    din = xgb.DMatrix(dXi, label=yi, feature_names=list(cols))
    bst = xgb.train(params, dtr, num_boost_round=N_ROUNDS, evals=[(din, 'inner_ho')],
                    early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    ntree = bst.num_boosted_rounds()
    print(f'  学習完了 best_iteration={bst.best_iteration} / 木 {ntree}本')

    raw_in = bst.predict(din, iteration_range=(0, ntree))
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw_in, yi)
    auc = roc_auc_score(yi, raw_in)

    # τ1: 内側HOで「4〜9番人気の S1 のレース内最大値」の85パーセンタイル（hitrate/CRITERIA_STAR_NARROW.md）
    s1_max = {}
    sub = df[m_in].reset_index(drop=True)
    mid = lambda pp: POP_LO <= pp <= POP_HI
    Xin = X[m_in]
    for rid, idx in sub.groupby('race_id', sort=False).indices.items():
        idx = np.asarray(idx)
        i1 = p1_index(list(sub['_pop'].values[idx]))
        if i1 is None:
            continue
        dd = build_delta(Xin[idx], i1)
        others = [k for k in range(len(idx)) if k != i1 and mid(sub['_pop'].values[idx][k])]
        if not others:
            continue
        p = iso.predict(bst.predict(xgb.DMatrix(dd[others], feature_names=list(cols)),
                                    iteration_range=(0, ntree)))
        s1_max[rid] = float(np.max(p))
    tau1 = float(np.percentile(list(s1_max.values()), Q_TAU))

    # S2: 本番の残差モデル（市場が土台）の3着内確率 − 市場だけの3着内確率。内側HOで較正
    from src.features.web_signals import fill_missing
    from src.tools.train_xgb import _popularity_to_base_margin
    with open(os.path.join(base, 'data', 'xgb_feature_cols.json'), encoding='utf-8') as f:
        rcols = list(json.load(f)['feature_cols'])
    rb = xgb.Booster()
    rb.load_model(os.path.join(base, 'data', 'xgb_fukusho_model.pkl'))
    nin = sub.groupby('race_id')['horse_num'].transform('count')
    bm = _popularity_to_base_margin(pd.Series(sub['_pop'].values), pd.Series(nin.values))
    dr = xgb.DMatrix(fill_missing(sub.reindex(columns=rcols)), feature_names=rcols)
    dr.set_base_margin(np.asarray(bm))
    raw_r = rb.predict(dr, output_margin=True, iteration_range=(0, rb.num_boosted_rounds()))
    y3 = (place[m_in] <= 3).astype(float)
    okr = np.isfinite(place[m_in]) & (place[m_in] > 0)
    sig = lambda z: 1 / (1 + np.exp(-np.asarray(z)))
    cm = IsotonicRegression(out_of_bounds='clip').fit(sig(raw_r)[okr], y3[okr])
    ck = IsotonicRegression(out_of_bounds='clip').fit(sig(np.asarray(bm))[okr], y3[okr])
    s2 = cm.predict(sig(raw_r)) - ck.predict(sig(np.asarray(bm)))
    s2_max = {}
    for i, (rid, pp) in enumerate(zip(sub['race_id'].values, sub['_pop'].values)):
        if not mid(pp) or not okr[i]:
            continue
        s2_max[rid] = max(s2_max.get(rid, -9.0), float(s2[i]))
    tau2 = float(np.percentile(list(s2_max.values()), Q_TAU))
    print(f'  内側HO AUC {auc:.4f} / τ1 {tau1:.4f}（S1）/ τ2 {tau2:.5f}（S2）')

    out_m = os.path.join(base, 'data', 'star_model.ubj')
    out_c = os.path.join(base, 'data', 'star_calibrator.pkl')
    out_f = os.path.join(base, 'data', 'star_feature_cols.json')
    bst.save_model(out_m)
    with open(out_c, 'wb') as f:
        pickle.dump({'version': 2, 'iso1': iso, 'cm': cm, 'ck': ck}, f)
    with open(out_f, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 2,
            'feature_cols': list(cols),
            'tau1': round(tau1, 6), 'tau2': round(tau2, 6), 'q_tau': Q_TAU,
            'pop_lo': POP_LO, 'pop_hi': POP_HI,
            'star_fill': STAR_FILL,
            'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'rows': int(len(df)), 'races': int(df.race_id.nunique()),
            'pairs_fit': int(len(yf)), 'pairs_inner': int(len(yi)),
            'date_from': str(pd.Timestamp(d.min()).date()),
            'date_to': str(last.date()),
            'inner_from': str((inner_from + pd.Timedelta(days=1)).date()),
            'best_iteration': int(bst.best_iteration), 'n_trees': int(ntree),
            'inner_auc': round(float(auc), 6),
            'hp': HP, 'seed': SEED, 'p1': '確定人気1位（ドリフトなし）',
            'criteria': ['hitrate/CRITERIA_STAR_V2.md', 'hitrate/CRITERIA_STAR_NARROW.md'],
            'results': ['hitrate/RESULTS_STAR_V2.md', 'hitrate/RESULTS_STAR_NARROW.md'],
        }, f, ensure_ascii=False, indent=1)
    for p in (out_m, out_c, out_f):
        print(f'  → {p} ({os.path.getsize(p):,} bytes)')
    print(f'  所要 {time.time() - t0:.1f}秒')


if __name__ == '__main__':
    main()
