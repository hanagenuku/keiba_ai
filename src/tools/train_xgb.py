"""
依頼5: XGB再学習スクリプト

build_training_data.py で生成した horse_features.csv を使って
XGBoostClassifier を再学習し、新モデルを data/xgb_fukusho_model.pkl に保存する。

使い方（Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.train_xgb import train_xgb
    result = train_xgb(BASE_DIR)

    # 残差学習モード（市場ベースライン＋AI残差）:
    result = train_xgb(BASE_DIR, residual=True)
"""

import os
import json
import math
import pickle
import shutil


# 除外する列（ラベル・識別子・リーク情報）
_EXCLUDE_COLS = {'race_id', 'date', 'horse_name', 'horse_num', 'place', 'is_fukusho'}

# 残差学習モードで除外する市場特徴量（base_margin に吸収）
_MARKET_FEAT_COLS = {'f_popularity'}

_CLIP_PROB = 0.001


def _popularity_to_base_margin(pop_series, n_horses_series):
    """人気順位からレース内正規化確率→logitのbase_marginを算出する。

    pop: 1-indexed popularity (1=1番人気)
    n_horses: そのレースの出走頭数
    """
    import numpy as np
    pop = pop_series.values.astype(float)
    n = n_horses_series.values.astype(float)
    pop = np.clip(pop, 1, np.maximum(n, 1))
    n = np.maximum(n, 2)
    # Zipf-like配分: 人気 k の相対確率 ∝ 1/k
    # p_market = (1/pop) / Σ(1/i for i=1..n) ≈ (1/pop) / (ln(n)+0.5772)
    harmonic = np.log(n) + 0.5772
    p_market = (1.0 / pop) / harmonic
    p_market = np.clip(p_market, _CLIP_PROB, 1 - _CLIP_PROB)
    return np.log(p_market / (1 - p_market))


def train_xgb(base_dir,
              train_end='2026-03-31',
              val_start='2026-04-01',
              val_end='2026-05-31',
              n_estimators=500,
              max_depth=6,
              learning_rate=0.05,
              subsample=0.8,
              colsample_bytree=0.8,
              min_child_weight=10,
              reg_alpha=0.1,
              reg_lambda=1.0,
              early_stopping_rounds=50,
              use_optuna=False,
              residual=False):
    """
    Parameters
    ----------
    base_dir   : プロジェクトルート
    train_end  : 学習データの最終日（以前）
    val_start  : 検証データの開始日（以降）
    val_end    : 検証データの終了日（以前）
    residual   : True なら残差学習モード。f_popularity を特徴量から除外し、
                 人気順位から算出した logit(p_market) を base_margin として
                 XGBoost に渡す。モデルは「市場からのズレ」だけを学習する。

    Returns
    -------
    dict: AUC, Brier score, Log loss 等の評価結果
    """
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    csv_path = os.path.join(base_dir, 'data', 'horse_features.csv')

    if residual:
        suffix = '_residual'
        print('━━ 残差学習モード ━━')
        print('  市場確率を base_margin に固定し、AIは「市場からのズレ」だけを学習')
    else:
        suffix = ''

    new_model_path = os.path.join(base_dir, 'data', f'xgb_fukusho_model{suffix}_new.pkl')
    new_cols_path  = os.path.join(base_dir, 'data', f'xgb_feature_cols{suffix}_new.json')
    old_model_path = os.path.join(base_dir, 'data', f'xgb_fukusho_model{suffix}.pkl')
    bak_model_path = os.path.join(base_dir, 'data', f'xgb_fukusho_model{suffix}_old.pkl')
    old_cols_path  = os.path.join(base_dir, 'data', f'xgb_feature_cols{suffix}.json')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'horse_features.csv が見つかりません: {csv_path}\n'
                                'build_training_data.py を先に実行してください。')

    # ── データ読み込みと日付フィルタ ────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f'読み込み: {len(df)} 行 × {len(df.columns)} 列')

    # date を正規化
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce'
    )
    df = df.dropna(subset=['date_obj'])

    train_df = df[df['date_obj'] <= pd.Timestamp(train_end)].copy()
    val_df   = df[(df['date_obj'] >= pd.Timestamp(val_start)) &
                  (df['date_obj'] <= pd.Timestamp(val_end))].copy()

    print(f'Train: {len(train_df)} 行 ({train_df["date_obj"].min().date()} 〜 {train_df["date_obj"].max().date()})')
    print(f'Val  : {len(val_df)} 行 ({val_df["date_obj"].min().date()} 〜 {val_df["date_obj"].max().date()})')

    if len(val_df) == 0:
        raise ValueError(f'検証データが空です。val_start/val_end を確認してください。')

    # ── 特徴量列を決定 ───────────────────────────────────────────────────
    exclude = _EXCLUDE_COLS | ({'date_obj'})
    if residual:
        exclude = exclude | _MARKET_FEAT_COLS
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    if residual:
        removed = [c for c in _MARKET_FEAT_COLS if c in df.columns]
        print(f'残差学習: 除外した市場特徴量 = {removed}')
    print(f'特徴量数: {len(feat_cols)}')

    X_train = train_df[feat_cols].fillna(5.0)
    y_train = train_df['is_fukusho']
    X_val   = val_df[feat_cols].fillna(5.0)
    y_val   = val_df['is_fukusho']

    # ── base_margin 計算（残差学習モード） ───────────────────────────────
    bm_train = None
    bm_val   = None
    if residual:
        # レースごとの頭数を算出
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df['_n_horses'] = train_df.groupby('race_id')['horse_num'].transform('count')
        val_df['_n_horses']   = val_df.groupby('race_id')['horse_num'].transform('count')

        pop_col = 'f_popularity'
        if pop_col not in train_df.columns:
            raise ValueError(f'{pop_col} が CSV に無い。build_training_data を先に実行してください')

        # popularity 欠損行はフィールド中央値で埋める
        train_pop = train_df[pop_col].fillna(train_df['_n_horses'] / 2)
        val_pop   = val_df[pop_col].fillna(val_df['_n_horses'] / 2)

        bm_train = _popularity_to_base_margin(train_pop, train_df['_n_horses'])
        bm_val   = _popularity_to_base_margin(val_pop, val_df['_n_horses'])
        print(f'  base_margin: train mean={bm_train.mean():.3f}, val mean={bm_val.mean():.3f}')

    # ── scale_pos_weight: 複勝率の逆数 ──────────────────────────────────
    pos_rate = y_train.mean()
    spw = round((1 - pos_rate) / max(pos_rate, 0.01), 2)
    print(f'Train 複勝率: {pos_rate*100:.1f}%  →  scale_pos_weight: {spw}')

    # ── XGB学習 ──────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=spw,
        eval_metric='logloss',
        early_stopping_rounds=early_stopping_rounds,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    fit_params = dict(
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    if residual:
        import xgboost as _xgb_fit
        dtrain = _xgb_fit.DMatrix(X_train, label=y_train, feature_names=feat_cols)
        dtrain.set_base_margin(bm_train)
        dval = _xgb_fit.DMatrix(X_val, label=y_val, feature_names=feat_cols)
        dval.set_base_margin(bm_val)
        xgb_params = model.get_xgb_params()
        xgb_params['eval_metric'] = 'logloss'
        booster = _xgb_fit.train(
            xgb_params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        model = booster
    else:
        model.fit(X_train, y_train, **fit_params)

    # ── 評価 ─────────────────────────────────────────────────────────────
    if residual:
        import xgboost as _xgb_eval
        dval_eval = _xgb_eval.DMatrix(X_val, feature_names=feat_cols)
        dval_eval.set_base_margin(bm_val)
        raw_margin = model.predict(dval_eval)
        val_prob = 1 / (1 + np.exp(-raw_margin))
    else:
        val_prob = model.predict_proba(X_val)[:, 1]
    auc    = roc_auc_score(y_val, val_prob)
    brier  = brier_score_loss(y_val, val_prob)
    ll     = log_loss(y_val, val_prob)

    print(f'\n── Val 評価 ──')
    print(f'  AUC   : {auc:.4f}')
    print(f'  Brier : {brier:.4f}')
    print(f'  LogLoss: {ll:.4f}')

    # 旧モデルの評価（比較用）
    old_result = {}
    if os.path.exists(old_model_path):
        try:
            with open(old_model_path, 'rb') as f:
                old_model = pickle.load(f)
            old_cols_path_check = old_cols_path if os.path.exists(old_cols_path) else None
            if old_cols_path_check:
                with open(old_cols_path_check) as f:
                    info = json.load(f)
                old_feats = info.get('feature_cols', feat_cols)
                old_is_residual = info.get('residual', False)
            else:
                old_feats = feat_cols
                old_is_residual = False
            old_X = X_val.reindex(columns=old_feats, fill_value=5.0)
            if old_is_residual and bm_val is not None:
                import xgboost as _xgb_old
                d_old = _xgb_old.DMatrix(old_X, feature_names=list(old_feats))
                d_old.set_base_margin(bm_val)
                old_margin = old_model.predict(d_old)
                old_prob = 1 / (1 + np.exp(-old_margin))
            else:
                old_prob = old_model.predict_proba(old_X)[:, 1]
            old_result = {
                'auc':   round(roc_auc_score(y_val, old_prob), 4),
                'brier': round(brier_score_loss(y_val, old_prob), 4),
                'logloss': round(log_loss(y_val, old_prob), 4),
            }
            print(f'\n── 旧モデル比較 ──')
            print(f'  AUC   : {old_result["auc"]}  ({"↑改善" if auc > old_result["auc"] else "↓悪化"})')
            print(f'  Brier : {old_result["brier"]}  ({"↑改善" if brier < old_result["brier"] else "↓悪化"})')
        except Exception as e:
            print(f'旧モデル評価スキップ: {e}')

    # ── 新モデルを保存 ────────────────────────────────────────────────────
    if residual:
        model.save_model(new_model_path)
    else:
        with open(new_model_path, 'wb') as f:
            pickle.dump(model, f)
    cols_meta = {
        'feature_cols': feat_cols,
        'trained_at':   str(pd.Timestamp.now()),
        'val_auc':      round(auc, 4),
        'val_brier':    round(brier, 4),
        'val_logloss':  round(ll, 4),
        'n_train':      len(train_df),
        'n_val':        len(val_df),
        'residual':     residual,
    }
    with open(new_cols_path, 'w', encoding='utf-8') as f:
        json.dump(cols_meta, f, ensure_ascii=False, indent=2)
    print(f'\n新モデル保存: {new_model_path}')
    print(f'特徴量リスト: {new_cols_path}')

    # ── AUC改善確認後に正式採用 ─────────────────────────────────────────
    if not old_result or auc >= old_result.get('auc', 0):
        if os.path.exists(old_model_path):
            shutil.copy2(old_model_path, bak_model_path)
            print(f'旧モデルを退避: {bak_model_path}')
        shutil.copy2(new_model_path, old_model_path)
        shutil.copy2(new_cols_path, old_cols_path)
        print(f'新モデルを正式採用: {old_model_path}')
    else:
        print(f'\n⚠ 旧モデルより精度低下のため正式採用スキップ。')
        print(f'   手動で確認後、new_model を old_model にコピーしてください。')

    # ── 特徴量重要度トップ20 ───────────────────────────────────────────
    if residual:
        import xgboost as _xgb_imp
        score_dict = model.get_score(importance_type='gain')
        total_gain = sum(score_dict.values()) or 1.0
        importances = sorted(
            [(k, v / total_gain) for k, v in score_dict.items()],
            key=lambda x: x[1], reverse=True,
        )[:20]
    else:
        importances = sorted(zip(feat_cols, model.feature_importances_),
                             key=lambda x: x[1], reverse=True)[:20]
    print('\n── 特徴量重要度 Top 20 ──')
    for name, imp in importances:
        print(f'  {name:<35} {imp*100:.2f}%')

    return {
        'auc':      round(auc, 4),
        'brier':    round(brier, 4),
        'logloss':  round(ll, 4),
        'old_model': old_result,
        'n_features': len(feat_cols),
        'n_train':  len(train_df),
        'n_val':    len(val_df),
        'residual': residual,
    }


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    train_xgb(base)
