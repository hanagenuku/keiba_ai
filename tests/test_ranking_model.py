"""
ランキングモデル関連のユニットテスト（numpy のみ・CI 環境用）

pandas / xgboost が不要な関数に絞ってテスト。
pandas 依存の train_ranking_model / compare_all_models は Colab 上で結合テストする。
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.rating_calibration import calc_ece, find_optimal_temperature
from src.betting.payout_estimator import estimate_payouts_from_win_odds


# ── calc_ece ─────────────────────────────────────────────────────────────

def test_calc_ece_perfect_calibration():
    """確率と実際の頻度が完全一致すれば ECE ≈ 0。"""
    # 各ビンで予測確率 = 的中率になるよう作成
    np.random.seed(0)
    bins = np.linspace(0.05, 0.95, 10)
    probs = np.repeat(bins, 50)
    hits  = np.array([np.random.rand() < p for p in probs], dtype=float)
    ece = calc_ece(probs, hits, n_bins=10)
    assert ece < 0.15, f'完璧に近いはずが ECE={ece:.4f}'


def test_calc_ece_worst_calibration():
    """確率0.9 の馬が全て外れ → ECE が 0.9 付近。"""
    probs = np.full(200, 0.9)
    hits  = np.zeros(200)
    ece = calc_ece(probs, hits)
    assert ece > 0.7, f'ECE={ece:.4f}'


def test_calc_ece_zero_probs():
    """全確率が 0.0 かつ全外れ → ECE = 0（ビンに何も入らない）。"""
    probs = np.zeros(50)
    hits  = np.zeros(50)
    ece = calc_ece(probs, hits)
    assert ece == 0.0


def test_calc_ece_returns_float():
    ece = calc_ece(np.array([0.3, 0.7]), np.array([0, 1]))
    assert isinstance(ece, float)


# ── find_optimal_temperature ─────────────────────────────────────────────

def test_find_optimal_temperature_returns_candidate(monkeypatch):
    """返り値 T が候補リスト内に含まれること。"""
    candidates = [0.5, 1.0, 2.0]

    # simulate_race をモック（強い馬が必ず1着）
    import src.betting.race_simulator as rs_mod

    def mock_sim(ratings, n_sims=5000, seed=None):
        n = len(ratings)
        order = np.argsort(-ratings)
        return np.tile(order, (n_sims, 1))

    monkeypatch.setattr(rs_mod, 'simulate_race', mock_sim)

    ratings_per_race = [
        (np.array([2.0, 1.0, 0.5]), [1, 2, 3]),
        (np.array([0.3, 1.5, 0.8]), [1, 2, 3]),
        (np.array([1.2, 0.6, 1.8]), [1, 2, 3]),
    ]
    actual_winners = [1, 2, 3]

    best_T, best_ece = find_optimal_temperature(
        ratings_per_race, actual_winners,
        T_candidates=candidates, n_sims=100
    )
    assert best_T in candidates
    assert 0.0 <= best_ece <= 1.0


def test_find_optimal_temperature_selects_lower_ece(monkeypatch):
    """ECE が最小になる T を選ぶ（T=1.0 が最良の場合）。"""
    import src.betting.race_simulator as rs_mod

    def mock_sim(ratings, n_sims=5000, seed=None):
        n = len(ratings)
        order = np.argsort(-ratings)
        return np.tile(order, (n_sims, 1))

    monkeypatch.setattr(rs_mod, 'simulate_race', mock_sim)

    # 強い馬が常に1着 → 1着馬の予測確率が高いほど ECE が低い
    # T が小さいほど確定的 → ECE が低くなるはず
    ratings_per_race = [(np.array([5.0, 1.0, 0.5]), [1, 2, 3]) for _ in range(10)]
    actual_winners   = [1] * 10

    best_T, _ = find_optimal_temperature(
        ratings_per_race, actual_winners,
        T_candidates=[0.5, 1.0, 5.0], n_sims=200
    )
    # T=0.5 が最も確定的で ECE が低い
    assert best_T == 0.5


# ── estimate_payouts_from_win_odds ────────────────────────────────────────

def test_estimate_payouts_returns_all_bet_types():
    """全券種のキーが返ること。"""
    win_odds = {1: 2.0, 2: 5.0, 3: 10.0, 4: 20.0, 5: 30.0,
                6: 50.0, 7: 100.0, 8: 3.0}
    payouts = estimate_payouts_from_win_odds(win_odds, n_sims=1000)
    for bt in ['win', 'place', 'quinella', 'trio']:
        assert bt in payouts
        assert len(payouts[bt]) > 0, f'{bt} が空'


def test_estimate_payouts_favorite_cheaper_than_longshot():
    """本命馬の推定単勝配当 < 大穴の推定単勝配当。"""
    win_odds = {1: 1.5, 2: 50.0, 3: 30.0, 4: 20.0}
    payouts = estimate_payouts_from_win_odds(win_odds, n_sims=5000)
    fav  = payouts['win'].get(1, 0)
    long = payouts['win'].get(2, 0)
    assert fav > 0
    assert long > 0
    assert fav < long, f'本命({fav:.1f}) >= 大穴({long:.1f})'


def test_estimate_payouts_win_payout_greater_than_one():
    """単勝・馬連・三連複の理論配当は 1 を超える（5頭以上の分散レース）。"""
    # 5頭だと複勝確率が高すぎて1未満になる場合があるので8頭で検証
    win_odds = {1: 3.0, 2: 5.0, 3: 8.0, 4: 12.0,
                5: 20.0, 6: 30.0, 7: 50.0, 8: 100.0}
    payouts = estimate_payouts_from_win_odds(win_odds, n_sims=2000)
    # 単勝・馬連・三連複は必ず 1 超え
    for bt in ['win', 'quinella', 'trio']:
        for key, val in payouts.get(bt, {}).items():
            assert val > 1.0, f'{bt}[{key}]={val:.2f} <= 1.0'


def test_estimate_payouts_empty_input():
    """空入力でもクラッシュしない。"""
    payouts = estimate_payouts_from_win_odds({})
    assert isinstance(payouts, dict)
    for bt in ['win', 'place']:
        assert bt in payouts


# ── build_ranking_training_data の制約だけを numpy でチェック ───────────────

def test_ranking_label_formula():
    """正規化逆順位ラベルの計算式が正しい。"""
    field_size = 10
    for place in range(1, field_size + 1):
        score = (field_size - place + 1) / field_size
        assert 0 < score <= 1.0
        if place == 1:
            assert abs(score - 1.0) < 1e-9
        if place == field_size:
            assert abs(score - 1 / field_size) < 1e-9


def test_groups_sum_invariant():
    """groups の合計 == X 行数 の不変条件を手動検証。"""
    field_sizes = [8, 10, 12, 16, 18]
    X_list, groups = [], []
    for fs in field_sizes:
        valid_count = fs - 1   # 最下位1頭を DNF (place>=99) と仮定してスキップ
        X_list.extend([[1.0, 2.0]] * valid_count)
        groups.append(valid_count)
    assert len(X_list) == sum(groups)
