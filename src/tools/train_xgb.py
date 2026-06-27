"""
依頼5: XGB再学習スクリプト

build_training_data.py で生成した horse_features.csv を使って
XGBoostClassifier を再学習し、新モデルを data/xgb_fukusho_model.pkl に保存する。

使い方（Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.train_xgb import train_xgb
    result = train_xgb(BASE_DIR)
"""

import os
import json
import pickle
import shutil


# 除外する列（ラベル・識別子・リーク情報）
_EXCLUDE_COLS = {'race_id', 'date', 'horse_name', 'horse_num', 'place', 'is_fukusho'}


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
              use_optuna=False):
    """
    Parameters
    ----------
    base_dir   : プロジェクトルート
    train_end  : 学習データの最終日（以前）
    val_start  : 検証データの開始日（以降）
    val_end    : 検証データの終了日（以前）

    Returns
    -------
    dict: AUC, Brier score, Log loss 等の評価結果
    """
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    csv_path       = os.path.join(base_dir, 'data', 'horse_features.csv')
    new_model_path = os.path.join(base_dir, 'data', 'xgb_fukusho_model_new.pkl')
    new_cols_path  = os.path.join(base_dir, 'data', 'xgb_feature_cols_new.json')
    old_model_path = os.path.join(base_dir, 'data', 'xgb_fukusho_model.pkl')
    bak_model_path = os.path.join(base_dir, 'data', 'xgb_fukusho_model_old.pkl')
    old_cols_path  = os.path.join(base_dir, 'data', 'xgb_feature_cols.json')

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
    feat_cols = [c for c in df.columns
                 if c not in _EXCLUDE_COLS and c != 'date_obj'
                 and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    print(f'特徴量数: {len(feat_cols)}')

    X_train = train_df[feat_cols].fillna(5.0)
    y_train = train_df['is_fukusho']
    X_val   = val_df[feat_cols].fillna(5.0)
    y_val   = val_df['is_fukusho']

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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── 評価 ─────────────────────────────────────────────────────────────
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
            else:
                old_feats = feat_cols
            old_X = X_val.reindex(columns=old_feats, fill_value=5.0)
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
    with open(new_model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(new_cols_path, 'w', encoding='utf-8') as f:
        json.dump({
            'feature_cols': feat_cols,
            'trained_at':   str(pd.Timestamp.now()),
            'val_auc':      round(auc, 4),
            'val_brier':    round(brier, 4),
            'val_logloss':  round(ll, 4),
            'n_train':      len(train_df),
            'n_val':        len(val_df),
        }, f, ensure_ascii=False, indent=2)
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
    }


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    train_xgb(base)
