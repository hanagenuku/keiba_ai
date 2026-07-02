"""
デュアルモデル確率推定（券種別モデル使い分け）

3モデル比較（653レース）に基づく暫定的な使い分け:
  単勝         → B2_ndcg  (xgb_ranking_ndcg.pkl, T=0.7)  的中率 45.5% vs A 43.6%
  複勝・馬連・三連複 → A_fukusho (T=0.7)  複勝 80.6%, 馬連 23.3%, 三連複 21.6%
  B1_pairwise  → 不使用（T=5.0 で確率が均一すぎるため）

⚠ 暫定。単勝の差 (45.5% vs 43.6%) は小さく誤差の可能性あり。
  1,000 レース超のデータ蓄積後に再検証すること。
  ROI は推定配当ベースの理論値であり実際の収益とは異なる。

使い方（Colab / スクリプト）:
    from src.betting.dual_model import build_dual_probs
    probs, meta = build_dual_probs(feat_df, horse_nums, BASE_DIR, n_sims=20000)
    # probs: win は B2_ndcg、place/quinella/trio は A_fukusho のシミュレーション結果
"""

import os
import json
import pickle
import numpy as np

_CACHE = {}  # base_dir → loaded model info (lazy load, module-level cache)


def load_dual_models(base_dir):
    """
    A_fukusho と B2_ndcg のモデル・特徴量・温度をロードして返す（キャッシュ付き）。

    Returns
    -------
    dict with keys:
        model_A, feat_A, T_A   — 複勝モデル (XGBClassifier)
        model_B2, feat_B2, T_B2 — ランキングモデル ndcg (xgb.Booster)
    """
    if base_dir in _CACHE:
        return _CACHE[base_dir]

    data_dir = os.path.join(base_dir, 'data')

    temp_path = os.path.join(data_dir, 'rating_temperature.json')
    temperatures = {}
    if os.path.exists(temp_path):
        with open(temp_path) as f:
            temperatures = json.load(f).get('calibration', {})

    def _load_model(path):
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_feat_cols(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f).get('feature_cols', [])

    result = {
        'model_A':  _load_model(os.path.join(data_dir, 'xgb_fukusho_model.pkl')),
        'feat_A':   _load_feat_cols(os.path.join(data_dir, 'xgb_feature_cols.json')),
        'T_A':      float(temperatures.get('fukusho', {}).get('T', 0.7)),
        'model_B2': _load_model(os.path.join(data_dir, 'xgb_ranking_ndcg.pkl')),
        'feat_B2':  _load_feat_cols(os.path.join(data_dir, 'xgb_ranking_feature_cols.json')),
        'T_B2':     float(temperatures.get('ranking_ndcg', {}).get('T', 0.7)),
    }
    _CACHE[base_dir] = result
    return result


def _predict_a_ratings(model, feat_cols, feat_df, T):
    """A_fukusho (XGBClassifier): logit(複勝確率) / T を返す。"""
    available = [c for c in feat_cols if c in feat_df.columns]
    X = feat_df[available].fillna(5.0)
    prob = model.predict_proba(X)[:, 1]
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob)) / T


def _predict_b2_ratings(model, feat_cols, feat_df, T):
    """B2_ndcg (xgb.Booster): predict() / T を返す。"""
    import xgboost as xgb
    available = [c for c in feat_cols if c in feat_df.columns]
    X = feat_df[available].fillna(5.0)
    dmat = xgb.DMatrix(X.values, feature_names=available)
    return model.predict(dmat) / T


def merge_probs(probs_a, probs_b2):
    """
    2系統のシミュレーション確率をマージする。

      単勝           → B2_ndcg  (probs_b2['win'])
      複勝・馬連・馬単・三連複・三連単 → A_fukusho (probs_a の残り)
    """
    merged = dict(probs_a)
    merged['win'] = dict(probs_b2['win'])
    return merged


def build_dual_probs(feat_df, horse_nums, base_dir, n_sims=20000):
    """
    A_fukusho と B2_ndcg をそれぞれシミュレートして確率をマージする。

    Parameters
    ----------
    feat_df    : 1レース分の特徴量 DataFrame（horse_features.csv と同じ列構成）
    horse_nums : 馬番リスト（feat_df の行順と対応）
    base_dir   : プロジェクトルート
    n_sims     : Gumbelシミュレーション回数

    Returns
    -------
    probs : マージ済み確率 dict
        'win'       → B2_ndcg のシミュレーション結果
        その他      → A_fukusho のシミュレーション結果
    meta  : {'T_A', 'T_B2', 'b2_available': bool}
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities

    info = load_dual_models(base_dir)
    meta = {'T_A': info['T_A'], 'T_B2': info['T_B2'], 'b2_available': False}

    if info['model_A'] is None or not info['feat_A']:
        raise RuntimeError('A_fukusho モデルが見つかりません。xgb_fukusho_model.pkl を確認してください。')

    ratings_A = _predict_a_ratings(info['model_A'], info['feat_A'], feat_df, info['T_A'])
    orders_A  = simulate_race(ratings_A, n_sims=n_sims)
    probs_A   = calc_ticket_probabilities(orders_A, horse_nums)

    if info['model_B2'] is None or not info['feat_B2']:
        print('⚠ B2_ndcg モデルなし → 単勝も A_fukusho を使用')
        return probs_A, meta

    ratings_B2 = _predict_b2_ratings(info['model_B2'], info['feat_B2'], feat_df, info['T_B2'])
    orders_B2  = simulate_race(ratings_B2, n_sims=n_sims)
    probs_B2   = calc_ticket_probabilities(orders_B2, horse_nums)

    meta['b2_available'] = True
    return merge_probs(probs_A, probs_B2), meta
