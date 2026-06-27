import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.features.market_correction as mc
from src.features.market_correction import (
    apply_market_correction,
    classify_rl_band,
    classify_pop_band,
    CORRECTION_FACTORS,
)


def _make_horses(specs):
    """(cal_prob, rl_rank, popularity) のリストから馬辞書リストを作る。"""
    return [
        {'cal_prob': cp, 'rl_rank': rl, 'popularity': pop, 'total': cp * 10}
        for cp, rl, pop in specs
    ]


def test_rl1_unpopular_suppressed():
    """テスト1: RL上位×不人気が抑制される。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = True
    try:
        horses = _make_horses([
            (0.40, 1, 12),  # RL1だが12番人気 → ×0.30 で抑制
            (0.30, 2, 1),   # RL2で1番人気 → ×1.0 そのまま
        ])
        result = apply_market_correction(horses)
        # 抑制された側が低くなっているはず
        h_rl1 = next(h for h in result if h['rl_rank_raw'] == 1)
        h_rl2 = next(h for h in result if h['rl_rank_raw'] == 2)
        assert h_rl1['cal_prob'] < h_rl2['cal_prob'], \
            f"RL1不人気({h_rl1['cal_prob']:.4f}) should be < RL2人気({h_rl2['cal_prob']:.4f})"
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


def test_cal_prob_raw_preserved():
    """テスト2: 補正前の値が cal_prob_raw に保存される。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = True
    try:
        horses = _make_horses([(0.40, 1, 12), (0.30, 2, 1)])
        result = apply_market_correction(horses)
        h = next(h for h in result if h['rl_rank_raw'] == 1)
        assert 'cal_prob_raw' in h
        assert abs(h['cal_prob_raw'] - 0.40) < 1e-9
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


def test_correction_off_no_change():
    """テスト3: OFF時は補正されない（cal_prob_raw == cal_prob）。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = False
    try:
        horses = _make_horses([(0.40, 1, 12), (0.30, 2, 1)])
        result = apply_market_correction(horses)
        for h in result:
            assert not h['correction_applied']
            assert abs(h['cal_prob'] - h['cal_prob_raw']) < 1e-9
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


def test_cal_prob_sum_normalized():
    """テスト4: 補正後の cal_prob の合計が 3.0 に正規化される（3頭以上の場合）。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = True
    try:
        horses = _make_horses([
            (0.40, 1, 12),
            (0.30, 2, 1),
            (0.20, 3, 2),
            (0.10, 4, 5),
        ])
        result = apply_market_correction(horses)
        total = sum(h['cal_prob'] for h in result)
        assert abs(total - 3.0) < 0.01, f"cal_prob sum={total:.4f} should be ~3.0"
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


def test_rl_rank_raw_saved():
    """テスト5: rl_rank_raw に補正前の順位が保存されている。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = True
    try:
        horses = _make_horses([(0.50, 1, 8), (0.30, 2, 1), (0.20, 3, 2)])
        result = apply_market_correction(horses)
        raws = {h['rl_rank_raw'] for h in result}
        assert 1 in raws and 2 in raws and 3 in raws
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


def test_band_classifiers():
    """classify_rl_band / classify_pop_band が正しい帯を返す。"""
    assert classify_rl_band(1) == 'top'
    assert classify_rl_band(3) == 'top'
    assert classify_rl_band(4) == 'mid'
    assert classify_rl_band(6) == 'mid'
    assert classify_rl_band(7) == 'low'
    assert classify_rl_band(99) == 'low'

    assert classify_pop_band(1) == 'popular'
    assert classify_pop_band(3) == 'popular'
    assert classify_pop_band(4) == 'mid'
    assert classify_pop_band(6) == 'mid'
    assert classify_pop_band(7) == 'low'
    assert classify_pop_band(9) == 'low'
    assert classify_pop_band(10) == 'unpopular'
    assert classify_pop_band(99) == 'unpopular'


def test_low_popular_boosted():
    """RL下位×1-3番人気は cal_prob が強調される（×1.2）。"""
    orig_enabled = mc.MARKET_CORRECTION_ENABLED
    mc.MARKET_CORRECTION_ENABLED = True
    try:
        horses = _make_horses([(0.10, 8, 1), (0.30, 1, 10)])
        result = apply_market_correction(horses)
        h_low_pop = next(h for h in result if h['rl_rank_raw'] == 8)
        assert h_low_pop['correction_factor'] == 1.2
        assert h_low_pop['correction_applied']
    finally:
        mc.MARKET_CORRECTION_ENABLED = orig_enabled


if __name__ == '__main__':
    test_rl1_unpopular_suppressed()
    print('✅ test_rl1_unpopular_suppressed passed')
    test_cal_prob_raw_preserved()
    print('✅ test_cal_prob_raw_preserved passed')
    test_correction_off_no_change()
    print('✅ test_correction_off_no_change passed')
    test_cal_prob_sum_normalized()
    print('✅ test_cal_prob_sum_normalized passed')
    test_rl_rank_raw_saved()
    print('✅ test_rl_rank_raw_saved passed')
    test_band_classifiers()
    print('✅ test_band_classifiers passed')
    test_low_popular_boosted()
    print('✅ test_low_popular_boosted passed')
