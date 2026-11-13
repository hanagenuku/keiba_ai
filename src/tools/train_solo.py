"""「AI単」列のための単独モデル（市場なし）を学習する。hitrate/CRITERIA_SOLO.md。

本番の残差モデルと **同じ特徴量・同じ学習窓・同じ木の設定**で、`base_margin` を付けずに学習する。
🔴 表示専用。予想（rl_rank）・買い目・確率には使わない。
🔴 列は本番の `data/xgb_feature_cols.json` をそのまま使う（engine は列が違うと単独モデルを使わない）。
"""
import json
import os

MODEL_NAME = 'xgb_solo_model.pkl'
COLS_NAME = 'xgb_solo_cols.json'
PARAMS = dict(max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
              alpha=0.1, reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss', nthread=8)


def train_solo(base_dir, train_end='2026-06-30', val_start='2026-07-01', val_end='2026-08-31',
               seed=42, df=None, save=True):
    """学習して (booster, meta) を返す。save=True なら data/ に保存する。"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from src.features.web_signals import fill_missing

    data = os.path.join(base_dir, 'data')
    with open(os.path.join(data, 'xgb_feature_cols.json'), encoding='utf-8') as f:
        cols = list(json.load(f)['feature_cols'])
    if df is None:
        df = pd.read_csv(os.path.join(data, 'horse_features.csv'))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'学習データに本番の列が無い: {missing[:5]}')
    d = df['date'].astype(str).str[:10]
    tr, va = df[d <= train_end], df[(d >= val_start) & (d <= val_end)]
    if len(va) == 0:
        raise ValueError('検証データが空')
    pos = tr['is_fukusho'].mean()
    params = dict(PARAMS, seed=seed, scale_pos_weight=round((1 - pos) / max(pos, 0.01), 2))
    dt = xgb.DMatrix(fill_missing(tr[cols]), label=tr['is_fukusho'], feature_names=cols)
    dv = xgb.DMatrix(fill_missing(va[cols]), label=va['is_fukusho'], feature_names=cols)
    b = xgb.train(params, dt, 500, evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
    # 早期終了で選ばれた本数に切って保存する（推論は「保存した全木」で固定できる）
    b = b[: b.best_iteration + 1]
    auc = float(roc_auc_score(va['is_fukusho'], b.predict(dv, output_margin=True)))
    meta = {'feature_cols': cols, 'trained_at': str(pd.Timestamp.now()), 'val_auc': round(auc, 4),
            'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_trees': int(b.num_boosted_rounds()),
            'train_end': train_end, 'val_start': val_start, 'val_end': val_end, 'market': 'none'}
    if save:
        b.save_model(os.path.join(data, MODEL_NAME))
        with open(os.path.join(data, COLS_NAME), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return b, meta
