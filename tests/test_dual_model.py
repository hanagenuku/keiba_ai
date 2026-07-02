"""
デュアルモデル（dual_model.py）のユニットテスト

外部モデルファイル不要。merge_probs の正確さと
build_bets_from_simulation の ratings_win パスをモックで検証する。
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.dual_model import merge_probs


# ── merge_probs ────────────────────────────────────────────────────────────────

def _make_probs(win_vals, place_vals, horse_nums):
    """テスト用の簡易確率 dict を作成する。"""
    return {
        'win':       {n: w for n, w in zip(horse_nums, win_vals)},
        'place':     {n: p for n, p in zip(horse_nums, place_vals)},
        'quinella':  {(1, 2): 0.1, (1, 3): 0.08},
        'exacta':    {(1, 2): 0.05},
        'trio':      {(1, 2, 3): 0.04},
        'trifecta':  {(1, 2, 3): 0.02},
    }


def test_merge_probs_win_from_b2():
    """merge_probs は 'win' を B2 の値で上書きする。"""
    probs_a  = _make_probs([0.30, 0.20, 0.15], [0.70, 0.60, 0.55], [1, 2, 3])
    probs_b2 = _make_probs([0.45, 0.25, 0.10], [0.80, 0.65, 0.50], [1, 2, 3])

    merged = merge_probs(probs_a, probs_b2)

    # 単勝は B2 の値
    assert merged['win'][1] == pytest.approx(0.45)
    assert merged['win'][2] == pytest.approx(0.25)

    # 複勝は A の値
    assert merged['place'][1] == pytest.approx(0.70)
    assert merged['place'][2] == pytest.approx(0.60)


def test_merge_probs_place_quinella_trio_from_a():
    """merge_probs は place / quinella / trio を A から引き継ぐ。"""
    probs_a  = _make_probs([0.30, 0.20, 0.15], [0.70, 0.60, 0.55], [1, 2, 3])
    probs_b2 = _make_probs([0.45, 0.25, 0.10], [0.99, 0.99, 0.99], [1, 2, 3])

    merged = merge_probs(probs_a, probs_b2)

    assert merged['quinella'] == probs_a['quinella']
    assert merged['trio']     == probs_a['trio']
    assert merged['trifecta'] == probs_a['trifecta']


def test_merge_probs_does_not_mutate_inputs():
    """merge_probs は probs_a / probs_b2 を破壊しない。"""
    probs_a  = _make_probs([0.30, 0.20], [0.70, 0.60], [1, 2])
    probs_b2 = _make_probs([0.45, 0.25], [0.80, 0.65], [1, 2])

    orig_a_win  = dict(probs_a['win'])
    orig_b2_win = dict(probs_b2['win'])

    merge_probs(probs_a, probs_b2)

    assert probs_a['win']  == orig_a_win
    assert probs_b2['win'] == orig_b2_win


def test_merge_probs_returns_new_dict():
    """merge_probs の戻り値は probs_a と別オブジェクト。"""
    probs_a  = _make_probs([0.3], [0.7], [1])
    probs_b2 = _make_probs([0.4], [0.8], [1])
    merged = merge_probs(probs_a, probs_b2)
    assert merged is not probs_a


# ── build_bets_from_simulation: ratings_win パス ───────────────────────────────

def test_build_bets_win_from_ratings_win(monkeypatch):
    """ratings_win を渡すと単勝確率が別モデルの値で上書きされる。"""
    import src.betting.race_simulator as rs_mod
    import src.betting.ev_calculator as ev_mod

    call_log = {'calls': []}

    def mock_simulate(ratings, n_sims=20000, seed=None):
        call_log['calls'].append(list(ratings))
        n = len(ratings)
        return np.tile(np.arange(n), (n_sims, 1))

    def mock_calc_ticket(orders, horse_nums):
        # 1番馬が常に1着のシミュレーション
        n = len(horse_nums)
        return {
            'win':      {h: (1.0 if i == 0 else 0.0) for i, h in enumerate(horse_nums)},
            'place':    {h: (1.0 if i < 3 else 0.0) for i, h in enumerate(horse_nums)},
            'quinella': {(horse_nums[0], horse_nums[1]): 1.0},
            'exacta':   {},
            'trio':     {(horse_nums[0], horse_nums[1], horse_nums[2]): 1.0},
            'trifecta': {},
        }

    def mock_calc_ev(probs, odds_map):
        return {}

    def mock_select_bets(ev_results, min_ev=1.25, min_prob=0.01):
        return {'win': [], 'place': [], 'quinella': [], 'trio': []}

    monkeypatch.setattr(rs_mod, 'simulate_race', mock_simulate)
    monkeypatch.setattr(rs_mod, 'calc_ticket_probabilities', mock_calc_ticket)
    monkeypatch.setattr(ev_mod, 'calc_ev_all_tickets', mock_calc_ev)
    monkeypatch.setattr(ev_mod, 'select_value_bets', mock_select_bets)

    from src.betting.make_bets import build_bets_from_simulation

    horses = [
        {'horse_num': 1, 'rating': 2.0},
        {'horse_num': 2, 'rating': 1.0},
        {'horse_num': 3, 'rating': 0.5},
        {'horse_num': 4, 'rating': 0.3},
    ]
    ratings_win = np.array([1.5, 2.5, 0.3, 0.2])  # B2 ratings

    bets, probs, ev_results = build_bets_from_simulation(
        horses, odds_map={}, n_sims=100, ratings_win=ratings_win
    )

    # simulate_race が 2 回呼ばれること（A 用 + B2 用）
    assert len(call_log['calls']) == 2

    # 1回目: A ratings, 2回目: B2 ratings_win
    assert call_log['calls'][0] == pytest.approx([2.0, 1.0, 0.5, 0.3])
    assert call_log['calls'][1] == pytest.approx([1.5, 2.5, 0.3, 0.2])


def test_build_bets_no_ratings_win_single_sim(monkeypatch):
    """ratings_win が None のとき simulate_race は 1 回だけ呼ばれる。"""
    import src.betting.race_simulator as rs_mod
    import src.betting.ev_calculator as ev_mod

    call_count = {'n': 0}

    def mock_simulate(ratings, n_sims=20000, seed=None):
        call_count['n'] += 1
        n = len(ratings)
        return np.tile(np.arange(n), (n_sims, 1))

    def mock_calc_ticket(orders, horse_nums):
        n = len(horse_nums)
        return {
            'win': {h: 1 / n for h in horse_nums},
            'place': {h: 0.5 for h in horse_nums},
            'quinella': {}, 'exacta': {}, 'trio': {}, 'trifecta': {},
        }

    monkeypatch.setattr(rs_mod, 'simulate_race', mock_simulate)
    monkeypatch.setattr(rs_mod, 'calc_ticket_probabilities', mock_calc_ticket)

    import src.betting.ev_calculator as ev_mod
    monkeypatch.setattr(ev_mod, 'calc_ev_all_tickets', lambda p, o: {})
    monkeypatch.setattr(ev_mod, 'select_value_bets', lambda r, **kw: {'win': [], 'place': [], 'quinella': [], 'trio': []})

    from src.betting.make_bets import build_bets_from_simulation

    horses = [{'horse_num': i, 'rating': float(i)} for i in range(1, 5)]
    build_bets_from_simulation(horses, odds_map={}, n_sims=50)

    assert call_count['n'] == 1
