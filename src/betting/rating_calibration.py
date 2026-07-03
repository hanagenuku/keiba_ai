"""
Temperature Scaling によるスコア校正

各モデルの出力スケールを統一する。これをやらないと
スケール差を実力差と誤解し、モデル比較が無意味になる。

Gumbel シミュレーションでは:
  scores = rating / T + gumbel_noise
  T が小さい → 確定的（堅い予想）
  T が大きい → ランダム的（荒れ予想）

各モデルで「Val期間の予測1着確率と実際の1着率のズレ（ECE）」が最小になる T を探す。

使い方（Colab）:
    from src.betting.rating_calibration import calibrate_all_models
    calibrate_all_models(BASE_DIR, val_start='2026-04-01', val_end='2026-05-31')
    # → rating_temperature.json に保存
"""

import os
import json
import numpy as np


def calc_ece(probs, hits, n_bins=10):
    """
    Expected Calibration Error.
    予測確率をビンに分け、各ビンで「予測確率の平均」と「実際の的中率」の差を集計。

    「30%と予測した馬が実際に30%来るか」のズレを0〜1 で表す。
    低いほど校正が良い。
    """
    probs = np.asarray(probs, dtype=float)
    hits  = np.asarray(hits,  dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        avg_pred   = probs[mask].mean()
        avg_actual = hits[mask].mean()
        weight     = mask.sum() / len(probs)
        ece       += weight * abs(avg_pred - avg_actual)
    return float(ece)


def find_optimal_temperature(model_ratings_per_race, actual_winners,
                             T_candidates=None, n_sims=5000):
    """
    Val期間のデータで最適温度 T を探す。

    Parameters
    ----------
    model_ratings_per_race : list of (ratings_array, horse_nums_list)
        各レースの能力値配列と馬番リスト
    actual_winners : list of int
        各レースの実際の1着馬番
    T_candidates : list of float | None
        探索する温度の候補（None なら既定値）
    n_sims : int
        シミュレーション回数（速度と精度のトレードオフ）

    Returns
    -------
    best_T : float
    best_ece : float
    """
    from src.betting.race_simulator import simulate_race

    if T_candidates is None:
        T_candidates = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    best_T   = 1.0
    best_ece = float('inf')

    for T in T_candidates:
        all_probs = []
        all_hits  = []

        for (ratings, nums), winner in zip(model_ratings_per_race, actual_winners):
            ratings_arr = np.asarray(ratings, dtype=float)
            nums_arr    = np.asarray(nums,    dtype=int)

            if len(ratings_arr) < 3:
                continue

            scaled = ratings_arr / T
            orders = simulate_race(scaled, n_sims=n_sims)
            first_nums = nums_arr[orders[:, 0]]   # 1着馬番

            for num in nums_arr:
                p = float((first_nums == num).mean())
                all_probs.append(p)
                all_hits.append(1 if int(num) == int(winner) else 0)

        ece = calc_ece(np.array(all_probs), np.array(all_hits))
        print(f'  T={T:.1f}: ECE={ece:.4f}')

        if ece < best_ece:
            best_ece = ece
            best_T   = T

    print(f'✅ 最適温度: T={best_T} (ECE={best_ece:.4f})')
    return best_T, best_ece


def _predict_ratings(model, is_logistic, X, feat_cols):
    """
    モデル種別に応じた能力値を返す。

    Model A (XGBClassifier / is_logistic=True):
        predict_proba → 複勝確率 → logit 変換 → 能力値
    Model B (xgb.Booster / is_logistic=False):
        predict → 直接能力値
    """
    import xgboost as xgb
    import numpy as np

    if is_logistic:
        import pandas as pd
        X_df = pd.DataFrame(X, columns=feat_cols)
        prob = model.predict_proba(X_df.fillna(5.0))[:, 1]
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        return np.log(prob / (1 - prob))   # logit
    else:
        dmat = xgb.DMatrix(X, feature_names=feat_cols)
        return model.predict(dmat)


def calibrate_all_models(base_dir,
                         val_start='2026-04-01',
                         val_end='2026-05-31',
                         T_candidates=None,
                         n_sims=5000):
    """
    3モデル（fukusho / pairwise / ndcg）の温度を校正して
    rating_temperature.json に保存する。

    Parameters
    ----------
    base_dir  : プロジェクトルート
    val_start : Val 期間開始
    val_end   : Val 期間終了
    """
    import pickle
    import pandas as pd

    data_dir = os.path.join(base_dir, 'data')

    # ── horse_features.csv 読み込み ──────────────────────────────────────
    csv_path = os.path.join(data_dir, 'horse_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'horse_features.csv が見つかりません: {csv_path}')

    df = pd.read_csv(csv_path)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce'
    )
    df = df.dropna(subset=['date_obj'])
    val_df = df[(df['date_obj'] >= pd.Timestamp(val_start)) &
                (df['date_obj'] <= pd.Timestamp(val_end))].copy()

    if len(val_df) == 0:
        raise ValueError(f'Val期間にデータがありません: {val_start}〜{val_end}')

    print(f'Val期間: {val_df["date_obj"].min().date()} 〜 {val_df["date_obj"].max().date()}'
          f'  ({len(val_df)} 頭 / {val_df["race_id"].nunique()} レース)')

    # ── モデル定義 ───────────────────────────────────────────────────────
    # (path, is_logistic, feature_cols_json or None)
    model_defs = {
        'fukusho': (
            os.path.join(data_dir, 'xgb_fukusho_model.pkl'),
            True,
            os.path.join(data_dir, 'xgb_feature_cols.json'),
        ),
        'ranking_pairwise': (
            os.path.join(data_dir, 'xgb_ranking_pairwise.pkl'),
            False,
            os.path.join(data_dir, 'xgb_ranking_feature_cols.json'),
        ),
        'ranking_ndcg': (
            os.path.join(data_dir, 'xgb_ranking_ndcg.pkl'),
            False,
            os.path.join(data_dir, 'xgb_ranking_feature_cols.json'),
        ),
    }

    calibration = {}
    _EXCLUDE = {'race_id', 'date', 'horse_name', 'horse_num', 'place',
                'is_fukusho', 'date_obj'}

    for model_name, (model_path, is_logistic, cols_path) in model_defs.items():
        if not os.path.exists(model_path):
            print(f'⚠ {model_name}: モデルファイルなし ({model_path})  → スキップ')
            continue

        print(f'\n── {model_name} 校正中 ──')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 特徴量列を決定
        if cols_path and os.path.exists(cols_path):
            with open(cols_path) as f:
                info = json.load(f)
            feat_cols = info.get('feature_cols', [])
        else:
            feat_cols = [c for c in df.columns
                         if c not in _EXCLUDE
                         and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

        val_df_sorted = val_df.sort_values(['race_id', 'horse_num'])

        ratings_per_race = []
        actual_winners   = []

        for race_id, grp in val_df_sorted.groupby('race_id', sort=False):
            valid = grp[grp['place'].fillna(99) < 99]
            if len(valid) < 3:
                continue
            winner_rows = valid[valid['place'] == 1]
            if winner_rows.empty:
                continue

            nums = valid['horse_num'].values

            # CSVに無い特徴量は 5.0 で補完し、モデルの全特徴量リストを渡す（feature_names mismatch 防止）
            valid_aligned = pd.DataFrame(index=valid.index)
            for c in feat_cols:
                valid_aligned[c] = valid[c] if c in valid.columns else 5.0
            ratings = _predict_ratings(
                model, is_logistic,
                valid_aligned.fillna(5.0).values,
                list(feat_cols)
            )

            winner_num = int(winner_rows.iloc[0]['horse_num'])
            ratings_per_race.append((ratings, nums))
            actual_winners.append(winner_num)

        print(f'  対象: {len(ratings_per_race)} レース')
        best_T, best_ece = find_optimal_temperature(
            ratings_per_race, actual_winners,
            T_candidates=T_candidates, n_sims=n_sims
        )
        calibration[model_name] = {'T': best_T, 'ece': round(best_ece, 4)}

    # ── 保存 ─────────────────────────────────────────────────────────────
    out_path = os.path.join(data_dir, 'rating_temperature.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'calibration': calibration,
            'val_period': {'start': val_start, 'end': val_end},
        }, f, ensure_ascii=False, indent=2)

    print(f'\n✅ rating_temperature.json 保存: {out_path}')
    for name, v in calibration.items():
        print(f'  {name:<20}: T={v["T"]}  ECE={v["ece"]}')

    return calibration
