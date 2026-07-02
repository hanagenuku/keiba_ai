"""
Phase 1 テスト: race_simulator + ev_calculator
"""
import pytest
import numpy as np
from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
from src.betting.ev_calculator import calc_ev_all_tickets, select_value_bets


# ── simulate_race ──────────────────────────────────────────────────────────

def test_simulate_race_shape():
    ratings = [2.0, 1.0, 0.5, 0.0, -0.5]
    orders = simulate_race(ratings, n_sims=5000, seed=42)
    assert orders.shape == (5000, 5)


def test_simulate_race_deterministic():
    ratings = [1.0, 0.0, -1.0]
    o1 = simulate_race(ratings, n_sims=1000, seed=7)
    o2 = simulate_race(ratings, n_sims=1000, seed=7)
    np.testing.assert_array_equal(o1, o2)


def test_simulate_race_valid_permutations():
    """各シミュレーションが 0..n-1 の置換であること"""
    orders = simulate_race([1.0, 0.5, 0.0], n_sims=100, seed=1)
    for row in orders:
        assert sorted(row) == [0, 1, 2]


# ── calc_ticket_probabilities ──────────────────────────────────────────────

def _get_probs(n_sims=20000, seed=0):
    ratings   = [2.0, 1.0, 0.5, 0.0, -0.5]
    horse_nums = [1, 2, 3, 4, 5]
    orders = simulate_race(ratings, n_sims=n_sims, seed=seed)
    return calc_ticket_probabilities(orders, horse_nums)


def test_win_prob_strongest_horse():
    probs = _get_probs()
    # 馬番1（最強）の単勝確率が最も高い
    assert probs['win'][1] == max(probs['win'].values())


def test_win_prob_sum_to_one():
    probs = _get_probs()
    total = sum(probs['win'].values())
    assert abs(total - 1.0) < 0.02, f"単勝確率の合計が1でない: {total}"


def test_place_prob_sum():
    probs = _get_probs()
    total = sum(probs['place'].values())
    assert abs(total - 3.0) < 0.05, f"複勝確率の合計が3でない: {total}"


def test_quinella_keys_sorted():
    probs = _get_probs()
    for key in probs['quinella']:
        assert key[0] < key[1], f"馬連キーが昇順でない: {key}"


def test_trio_keys_sorted():
    probs = _get_probs()
    for key in probs['trio']:
        assert key[0] < key[1] < key[2], f"三連複キーが昇順でない: {key}"


def test_trifecta_sum():
    """三連単の合計確率が 1 に近い"""
    probs = _get_probs()
    total = sum(probs['trifecta'].values())
    assert abs(total - 1.0) < 0.02, f"三連単確率の合計が1でない: {total}"


# ── calc_ev_all_tickets ────────────────────────────────────────────────────

def test_ev_calculation():
    probs = _get_probs()
    odds_map = {'win': {1: 2.5, 2: 4.0, 3: 8.0, 4: 15.0, 5: 30.0}}
    ev = calc_ev_all_tickets(probs, odds_map)
    assert len(ev['win']) > 0
    for item in ev['win']:
        assert abs(item['ev'] - item['prob'] * item['odds']) < 0.001


def test_ev_missing_odds_skipped():
    """オッズがない馬券はスキップされる"""
    probs = _get_probs()
    odds_map = {'win': {1: 3.0}}  # 馬番1 だけ
    ev = calc_ev_all_tickets(probs, odds_map)
    assert len(ev['win']) == 1
    assert ev['win'][0]['key'] == 1


def test_ev_sorted_descending():
    probs = _get_probs()
    odds_map = {'win': {1: 2.0, 2: 4.0, 3: 8.0, 4: 15.0, 5: 30.0}}
    ev = calc_ev_all_tickets(probs, odds_map)
    evs = [e['ev'] for e in ev['win']]
    assert evs == sorted(evs, reverse=True)


# ── select_value_bets ──────────────────────────────────────────────────────

def test_select_value_bets_filters_low_ev():
    probs = _get_probs()
    # 全馬に EV=0 になるオッズ（払戻 < 1.0）を設定
    odds_map = {'win': {i: 0.5 for i in [1, 2, 3, 4, 5]}}
    ev = calc_ev_all_tickets(probs, odds_map)
    value = select_value_bets(ev, min_ev=1.25)
    assert 'win' not in value


def test_select_value_bets_high_odds():
    probs = _get_probs()
    # 最強馬に高オッズを設定 → EV > 1.25 になるはず
    odds_map = {'win': {1: 10.0}}
    ev = calc_ev_all_tickets(probs, odds_map)
    value = select_value_bets(ev, min_ev=1.25, min_prob=0.01)
    assert 'win' in value
    assert value['win'][0]['key'] == 1
