"""市場フリーモデル(F) vs 本番同型(R) vs 市場のみ(M) の比較ハーネス。

⚠ base_margin は本番コード(src.tools.train_xgb)の関数をそのまま呼ぶ。
   別実装で作り直すと 2026-08-08 のリーク事故と同型の食い違いが起きる。
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from src.tools.train_xgb import (
    load_popularity_drift, _apply_popularity_drift, _popularity_to_base_margin,
    _EXCLUDE_COLS,
)

# 市場情報を加工した特徴量（方針書の禁止リスト）
MARKET_COLS = {'f_popularity', 'f_pop_last', 'f_pop_avg', 'f_beat_market_rate'}

HP = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1,
          reg_lambda=1.0, early_stopping_rounds=50)

_DF = None
_DRIFT = None


def load_df():
    global _DF, _DRIFT
    if _DF is None:
        df = pd.read_csv(os.path.join(BASE, 'data', 'horse_features.csv'))
        df['date_obj'] = pd.to_datetime(df['date'])
        df['_n_horses'] = df.groupby('race_id')['horse_num'].transform('count')
        _DF = df
        _DRIFT = load_popularity_drift(BASE)
        print(f'CSV: {len(df)} 行 / {df.race_id.nunique()} レース / '
              f'{df.date_obj.min().date()}〜{df.date_obj.max().date()} / 列 {df.shape[1]}')
        print(f'drift 標本: {"あり" if _DRIFT is not None else "なし"}')
    return _DF


def feat_cols_for(df, exclude_market):
    ex = set(_EXCLUDE_COLS) | {'date_obj', '_n_horses'}
    ex |= (MARKET_COLS if exclude_market else {'f_popularity'})
    return [c for c in df.columns
            if c not in ex and df[c].dtype.kind in 'fi']


def base_margin_for(sub, seed):
    """本番と同じ経路で base_margin を作る（ドリフト注入込み）。"""
    pop = sub['f_popularity'].fillna(sub['_n_horses'] / 2)
    if _DRIFT is not None:
        pop = pd.Series(_apply_popularity_drift(
            pop, sub['race_id'].values, _DRIFT, seed=seed))
    return _popularity_to_base_margin(pop, sub['_n_horses'])


def ece(y, p, n_bins=10):
    """等頻度ビンの Expected Calibration Error。"""
    q = pd.qcut(pd.Series(p).rank(method='first'), n_bins, labels=False)
    tot = 0.0
    for b in range(n_bins):
        m = q == b
        if m.sum() == 0:
            continue
        tot += m.sum() * abs(np.mean(np.asarray(p)[m.values]) - np.mean(np.asarray(y)[m.values]))
    return tot / len(y)


def run_arm(arm, train_end, val_start, val_end, seed, es_mode='inner',
            drop_groups=(), shuffle_label=False):
    """arm: 'R'(残差=本番同型) / 'F'(市場フリー) / 'Fplus'(過去人気は残す) / 'M'(市場のみ)"""
    df = load_df()
    tr = df[df.date_obj <= train_end].copy()
    va = df[(df.date_obj >= val_start) & (df.date_obj <= val_end)].copy()
    assert len(va) > 0, 'val が空'
    # 関門2: 窓の独立
    assert va.date_obj.min() > pd.Timestamp(train_end), \
        f'val開始({va.date_obj.min()}) が train_end({train_end}) より後でない'

    y_tr, y_va = tr['is_fukusho'].values, va['is_fukusho'].values

    if arm == 'M':
        bm = base_margin_for(va, seed=12)
        p = 1 / (1 + np.exp(-np.asarray(bm)))
        return dict(arm=arm, n_val=len(va), n_races=va.race_id.nunique(),
                    auc=roc_auc_score(y_va, p), logloss=log_loss(y_va, p),
                    brier=brier_score_loss(y_va, p), ece=ece(y_va, p),
                    best_iter=0, n_feat=0, prob=p, va_idx=va.index.values)

    residual = (arm == 'R')
    cols = feat_cols_for(df, exclude_market=(arm in ('F',)))
    for g in drop_groups:
        cols = [c for c in cols if c not in GROUPS[g]]
    # 関門3: F の列に市場の痕跡が無いこと
    if arm == 'F':
        bad = [c for c in cols if ('pop' in c) or ('market' in c) or ('odds' in c)]
        assert not bad, f'F に市場列が残っている: {bad}'

    if shuffle_label:  # プラセボ: レース内でラベルを入れ替える
        rng = np.random.default_rng(seed)
        y_tr = (tr.groupby('race_id')['is_fukusho']
                  .transform(lambda s: rng.permutation(s.values)).values)

    X_tr, X_va = tr[cols].fillna(5.0), va[cols].fillna(5.0)

    # early stopping 用の内側分割（val を覗かない）
    if es_mode == 'inner':
        cut = tr.date_obj.quantile(0.85)
        m_fit = (tr.date_obj <= cut).values
        X_fit, y_fit = X_tr[m_fit], y_tr[m_fit]
        X_es,  y_es  = X_tr[~m_fit], y_tr[~m_fit]
        es_sub = tr[~m_fit]
        fit_sub = tr[m_fit]
    else:  # 'val' = 本番と同じ（val で早期終了）
        X_fit, y_fit, fit_sub = X_tr, y_tr, tr
        X_es, y_es, es_sub = X_va, y_va, va

    pos = float(np.mean(y_fit))
    spw = round((1 - pos) / max(pos, 0.01), 2)
    model = XGBClassifier(**HP, scale_pos_weight=spw, eval_metric='logloss',
                          random_state=seed, n_jobs=-1)

    if residual:
        bm_fit = base_margin_for(fit_sub, seed=11)
        bm_es  = base_margin_for(es_sub, seed=12)
        bm_va  = base_margin_for(va, seed=12)
        d_fit = xgb.DMatrix(X_fit, label=y_fit, feature_names=cols)
        d_fit.set_base_margin(bm_fit)
        d_es = xgb.DMatrix(X_es, label=y_es, feature_names=cols)
        d_es.set_base_margin(bm_es)
        params = model.get_xgb_params(); params['eval_metric'] = 'logloss'
        bst = xgb.train(params, d_fit, num_boost_round=HP['n_estimators'],
                        evals=[(d_es, 'es')],
                        early_stopping_rounds=HP['early_stopping_rounds'],
                        verbose_eval=False)
        d_va = xgb.DMatrix(X_va, feature_names=cols); d_va.set_base_margin(bm_va)
        p = 1 / (1 + np.exp(-bst.predict(d_va, output_margin=True)))
        best_iter, booster = bst.best_iteration, bst
    else:
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        best_iter, booster = model.best_iteration, model.get_booster()

    return dict(arm=arm, n_val=len(va), n_races=va.race_id.nunique(),
                auc=roc_auc_score(y_va, p), logloss=log_loss(y_va, p),
                brier=brier_score_loss(y_va, p), ece=ece(y_va, p),
                best_iter=best_iter, n_feat=len(cols), prob=p,
                va_idx=va.index.values, booster=booster, cols=cols)


GROUPS = {}  # Phase 2 で埋める

WINDOWS = [
    ('W1', '2026-06-30', '2026-07-04', '2026-08-16'),
    ('W2', '2026-04-30', '2026-05-02', '2026-06-28'),
    ('W3', '2026-02-28', '2026-03-01', '2026-04-26'),
]
SEEDS = [42, 7, 2026]
