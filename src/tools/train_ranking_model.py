"""
ランキングモデル学習（rank:ndcg）

horse_features.csv の同じ特徴量を使い、教師を着順(全体)に変更。
複勝モデル(train_xgb.py)と比較するための対抗モデルを学習する。

使い方（Colab）:
    from src.tools.train_ranking_model import train_ranking_model
    train_ranking_model(BASE_DIR, objective='rank:ndcg', model_suffix='ndcg')
"""

import os
import json
import pickle

_EXCLUDE_COLS = {'race_id', 'date', 'horse_name', 'horse_num', 'place',
                 'is_fukusho', 'date_obj'}


def _load_and_filter(df, start, end):
    """date_obj フィルタ済みの DataFrame を返す（start/end は None 可）。"""
    import pandas as pd
    d = df.copy()
    if start:
        d = d[d['date_obj'] >= pd.Timestamp(start)]
    if end:
        d = d[d['date_obj'] <= pd.Timestamp(end)]
    return d


def build_ranking_training_data(df, feat_cols):
    """
    horse_features.csv の DataFrame からランキング学習用データを構築する。

    ラベル設計（正規化逆順位）:
        score = (field_size - place + 1) / field_size
        → 1着が常に約1.0, 最下位が約 1/field_size
        → 頭数に依存しない 0〜1 スコア

    重要: groups の合計 == X の行数 でなければならない。
    place >= 99 の馬はスキップし、groups もその分減らす。

    Returns
    -------
    X      : np.ndarray (N, F)
    y      : np.ndarray (N,)  0〜1 の正規化スコア
    groups : list[int]        各レースの有効馬数（sum == N）
    """
    import numpy as np
    import pandas as pd

    X_list, y_list, groups = [], [], []

    for race_id, grp in df.groupby('race_id', sort=False):
        # place < 99 のみ（DNF/除外を除く）
        valid = grp[grp['place'] < 99].copy()
        n = len(valid)
        if n < 5:
            continue

        # 着順で並べ替えて1〜nの連番スコアを振る（DNF除外で着順にギャップがあっても安全）
        valid = valid.sort_values('place')
        scores = np.arange(n, 0, -1)  # [n, n-1, ..., 2, 1] 1着=n（最高）

        X_list.append(valid[feat_cols].fillna(5.0).values)
        y_list.append(scores)
        groups.append(n)

    if not X_list:
        raise ValueError('有効なレースデータがありません。')

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    assert X.shape[0] == sum(groups), (
        f'groups合計({sum(groups)}) != X行数({X.shape[0]}): バグ'
    )

    return X, y, groups


def train_ranking_model(base_dir,
                        objective='rank:pairwise',
                        model_suffix='pairwise',
                        train_end='2026-03-31',
                        val_start='2026-04-01',
                        val_end='2026-05-31',
                        n_boost_round=500,
                        early_stopping_rounds=50):
    """
    XGBoost ランキングモデルを学習する。

    Parameters
    ----------
    base_dir        : プロジェクトルート
    objective       : 'rank:pairwise' または 'rank:ndcg'
    model_suffix    : 保存ファイル名の接尾辞
    train_end       : 学習データの最終日（以前）
    val_start/end   : 検証データの期間
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    csv_path = os.path.join(base_dir, 'data', 'horse_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'horse_features.csv が見つかりません: {csv_path}\n'
            'build_training_data.py を先に実行してください。'
        )

    df = pd.read_csv(csv_path)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce'
    )
    df = df.dropna(subset=['date_obj'])

    feat_cols = [c for c in df.columns
                 if c not in _EXCLUDE_COLS
                 and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    print(f'特徴量数: {len(feat_cols)}')

    # レースIDでソート（同一レースが連続するよう保証 → group境界を正確に）
    df = df.sort_values(['date_obj', 'race_id', 'horse_num']).reset_index(drop=True)

    train_df = _load_and_filter(df, None, train_end)
    val_df   = _load_and_filter(df, val_start, val_end)

    print(f'Train: {len(train_df)} 行 '
          f'({train_df["date_obj"].min().date()} 〜 {train_df["date_obj"].max().date()})')
    print(f'Val  : {len(val_df)} 行 '
          f'({val_df["date_obj"].min().date()} 〜 {val_df["date_obj"].max().date()})')

    print('学習データ構築中...')
    X_tr, y_tr, g_tr = build_ranking_training_data(train_df, feat_cols)
    X_va, y_va, g_va = build_ranking_training_data(val_df, feat_cols)
    print(f'  Train: {X_tr.shape[0]}頭 / {len(g_tr)}レース')
    print(f'  Val  : {X_va.shape[0]}頭 / {len(g_va)}レース')

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dtrain.set_group(g_tr)

    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
    dval.set_group(g_va)

    params = {
        'objective':       objective,
        'eval_metric':     ['ndcg@3', 'ndcg@5'],
        'eta':             0.05,
        'max_depth':       6,
        'subsample':       0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'seed':            42,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=n_boost_round,
        evals=[(dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    model_path = os.path.join(base_dir, 'data', f'xgb_ranking_{model_suffix}.pkl')
    cols_path  = os.path.join(base_dir, 'data', 'xgb_ranking_feature_cols.json')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    if not os.path.exists(cols_path):
        with open(cols_path, 'w', encoding='utf-8') as f:
            json.dump({'feature_cols': feat_cols,
                       'trained_at': str(pd.Timestamp.now())}, f,
                      ensure_ascii=False, indent=2)

    print(f'\n✅ ランキングモデル保存: {model_path}')
    print(f'   best_iteration: {model.best_iteration}')

    # 特徴量重要度 Top 10
    scores = model.get_score(importance_type='gain')
    top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print('\n── 特徴量重要度 Top 10 (gain) ──')
    for name, imp in top10:
        print(f'  {name:<35} {imp:.2f}')

    return model, feat_cols


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    train_ranking_model(base, objective='rank:ndcg', model_suffix='ndcg')
