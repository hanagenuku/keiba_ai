import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (
    calc_performance_index, f_recent, calc_chaos_score,
    auto_comment, dist_zone_label, dz,
)


def test_calc_performance_index_basic():
    pi = calc_performance_index(34.0, distance=1600, surface='芝', condition='良')
    assert isinstance(pi, float)
    assert 0.0 <= pi <= 100.0


def test_calc_performance_index_pace_correction():
    # ハイペース（前半速い）の場合、上りが遅くても高い指数になる
    pi_high = calc_performance_index(35.5, first_3f=33.0, corner_pos=2, distance=1600, surface='芝')
    pi_slow = calc_performance_index(35.5, first_3f=36.5, corner_pos=10, distance=1600, surface='芝')
    assert pi_high > pi_slow


def test_f_recent_no_history():
    h = {'win_odds': 5.0, 'history': []}
    race = {'distance': 1600, 'surface': '芝', 'num_horses': 16}
    score = f_recent(h, race)
    assert 0.0 <= score <= 10.0


def test_f_recent_with_history():
    h = {
        'win_odds': 5.0,
        'history': [
            {'place': 1, 'finishers': 16, 'margin': 0.0, 'last_3f': 33.5,
             'first_3f': 35.0, 'corner_3': 2, 'surface': '芝', 'distance': 1600},
            {'place': 2, 'finishers': 16, 'margin': 0.2, 'last_3f': 34.0,
             'first_3f': 35.5, 'corner_3': 5, 'surface': '芝', 'distance': 1600},
        ],
    }
    race = {'distance': 1600, 'surface': '芝', 'num_horses': 16}
    score = f_recent(h, race)
    assert 0.0 <= score <= 10.0
    assert score > 5.0  # 1着・2着の実績があるので5以上


def test_calc_chaos_score_clear():
    scored = [
        {'total': 9.0}, {'total': 5.0}, {'total': 4.0},
    ]
    chaos = calc_chaos_score({}, scored)
    assert chaos < 0.5  # 差が大きいので混戦度低い


def test_calc_chaos_score_tight():
    scored = [
        {'total': 6.0}, {'total': 5.9}, {'total': 5.8},
    ]
    chaos = calc_chaos_score({}, scored)
    assert chaos > 0.5  # 差が小さいので混戦度高い


def test_dist_zone_label():
    assert dist_zone_label(1200) == '短距離'
    assert dist_zone_label(1600) == 'マイル'
    assert dist_zone_label(2000) == '中距離'
    assert dist_zone_label(3000) == '長距離'


def test_dz():
    assert dz(1200) == 'sp'
    assert dz(1600) == 'mi'
    assert dz(2000) == 'md'
    assert dz(3200) == 'lo'


if __name__ == '__main__':
    test_calc_performance_index_basic()
    print('✅ test_calc_performance_index_basic passed')
    test_calc_performance_index_pace_correction()
    print('✅ test_calc_performance_index_pace_correction passed')
    test_f_recent_no_history()
    print('✅ test_f_recent_no_history passed')
    test_f_recent_with_history()
    print('✅ test_f_recent_with_history passed')
    test_calc_chaos_score_clear()
    print('✅ test_calc_chaos_score_clear passed')
    test_calc_chaos_score_tight()
    print('✅ test_calc_chaos_score_tight passed')
    test_dist_zone_label()
    print('✅ test_dist_zone_label passed')
    test_dz()
    print('✅ test_dz passed')
