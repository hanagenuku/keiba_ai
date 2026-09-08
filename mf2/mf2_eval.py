"""Phase 2: 市場フリーモデルの特徴量グループ・アブレーション。

M-1（別セッション 2026-09-08）の設定と手続きに揃える:
  - ハイパーパラメータ = cid 82
  - early stopping は学習期間の末尾42日（内側ホールドアウト）。外側の窓は評価専用
  - Isotonic 較正も内側ホールドアウトだけで当てる
  - 確認期3窓 W1/W2/W3
⚠ 旧既定(max_depth=6 / mcw=10 / scale_pos_weight=事前確率ベース)では
   市場との差を 0.0128 過大に見るので使わない。
"""
import os, sys, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from src.tools.train_xgb import _EXCLUDE_COLS

MARKET_COLS = {'f_popularity', 'f_pop_last', 'f_pop_avg', 'f_beat_market_rate'}

HP82 = dict(max_depth=5, learning_rate=0.02, min_child_weight=300,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1)
N_ROUNDS, ES_ROUNDS, INNER_DAYS = 4000, 50, 42

# 確認期（M-1 と同一）
# (名前, train_end, val_start, val_end)  ← M-1 prep.py と同一
WINDOWS = [('W1', '2026-06-30', '2026-07-04', '2026-09-06'),
           ('W2', '2026-04-30', '2026-05-02', '2026-06-28'),
           ('W3', '2026-02-28', '2026-03-01', '2026-04-26')]
HEALTHY = {'W1', 'W3'}          # W2 は f_popularity 充足 72.8% の壊れた窓
SEEDS = [42, 7, 2026]
_DF = None


def load_df():
    global _DF
    if _DF is None:
        df = pd.read_csv(os.path.join(BASE, 'data', 'horse_features.csv'))
        df['date_obj'] = pd.to_datetime(df['date'])
        df['_n_horses'] = df.groupby('race_id')['horse_num'].transform('count')
        df['is_win'] = (df['place'] == 1).astype(int)
        _DF = df
        print(f'CSV {len(df):,}行 / {df.race_id.nunique():,}レース / '
              f'{df.date_obj.min().date()}〜{df.date_obj.max().date()}')
    return _DF


def market_free_cols(df, drop=()):
    ex = set(_EXCLUDE_COLS) | {'date_obj', '_n_horses', 'is_win'} | MARKET_COLS
    cols = [c for c in df.columns if c not in ex and df[c].dtype.kind in 'fi']
    for g in drop:
        cols = [c for c in cols if c not in g]
    bad = [c for c in cols if any(k in c for k in ('pop', 'market', 'odds'))]
    assert not bad, f'市場列が残っている: {bad}'
    return cols


def within_race_auc(sub, p, ycol):
    d = pd.DataFrame({'r': sub.race_id.values, 'y': sub[ycol].values, 'p': p})
    num = den = 0.0
    for _, g in d.groupby('r', sort=False):
        pos, neg = g.p[g.y == 1].values, g.p[g.y == 0].values
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum(); den += diff.size
    return num / den


def ece(y, p, n_bins=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), n_bins, labels=False).values
    y, p = np.asarray(y), np.asarray(p)
    return sum(abs(p[q == b].mean() - y[q == b].mean()) * (q == b).sum()
               for b in range(n_bins) if (q == b).sum()) / len(y)


def run(val_start, val_end, seed, drop=(), train_end=None):
    df = load_df()
    train_end = pd.Timestamp(train_end) if train_end else pd.Timestamp(val_start) - pd.Timedelta(days=1)
    tr = df[df.date_obj <= train_end]
    va = df[(df.date_obj >= val_start) & (df.date_obj <= val_end)]
    assert len(va) and va.date_obj.min() > train_end

    inner_start = train_end - pd.Timedelta(days=INNER_DAYS)
    fit = tr[tr.date_obj <= inner_start]
    ho = tr[tr.date_obj > inner_start]          # 内側ホールドアウト
    assert len(ho) > 1000, f'内側HOが薄い: {len(ho)}'

    cols = market_free_cols(df, drop)
    d_fit = xgb.DMatrix(fit[cols].fillna(5.0), label=fit.is_fukusho.values, feature_names=cols)
    d_ho  = xgb.DMatrix(ho[cols].fillna(5.0),  label=ho.is_fukusho.values,  feature_names=cols)
    d_va  = xgb.DMatrix(va[cols].fillna(5.0), feature_names=cols)

    params = dict(HP82, objective='binary:logistic', eval_metric='logloss',
                  seed=seed, nthread=-1)
    bst = xgb.train(params, d_fit, num_boost_round=N_ROUNDS, evals=[(d_ho, 'ho')],
                    early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    it = bst.best_iteration
    p_ho = bst.predict(d_ho, iteration_range=(0, it + 1))
    p_va = bst.predict(d_va, iteration_range=(0, it + 1))
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_ho, ho.is_fukusho.values)
    p_cal = np.clip(iso.predict(p_va), 1e-6, 1 - 1e-6)

    y3, y1 = va.is_fukusho.values, va.is_win.values
    return dict(n=len(va), trees=it + 1, n_feat=len(cols),
                auc3=roc_auc_score(y3, p_va), auc1=roc_auc_score(y1, p_va),
                wr3=within_race_auc(va, p_va, 'is_fukusho'),
                wr1=within_race_auc(va, p_va, 'is_win'),
                logloss=log_loss(y3, p_cal), brier=brier_score_loss(y3, p_cal),
                ece=ece(y3, p_cal))
