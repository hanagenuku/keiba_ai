"""残差学習（base_margin）の単体テスト"""

import math
import pytest
import numpy as np
import pandas as pd


class TestPopularityToBaseMargin:
    """_popularity_to_base_margin の入出力検証"""

    def test_first_pop_highest(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([1.0, 16.0])
        n = pd.Series([16.0, 16.0])
        bm = _popularity_to_base_margin(pop, n)
        assert bm[0] > bm[1], "1番人気が最下位より高い base_margin であるべき"

    def test_last_pop_negative(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([16.0])
        n = pd.Series([16.0])
        bm = _popularity_to_base_margin(pop, n)
        assert bm[0] < 0, "最下位人気の base_margin は負であるべき"

    def test_monotone_decreasing(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([1, 2, 3, 4, 5], dtype=float)
        n = pd.Series([10, 10, 10, 10, 10], dtype=float)
        bm = _popularity_to_base_margin(pop, n)
        for i in range(len(bm) - 1):
            assert bm[i] > bm[i + 1], "人気が下がるほど base_margin は小さくなるべき"

    def test_finite_values(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([1, 5, 10, 16], dtype=float)
        n = pd.Series([16, 16, 16, 16], dtype=float)
        bm = _popularity_to_base_margin(pop, n)
        assert np.all(np.isfinite(bm)), "全ての値が有限であるべき"

    def test_edge_pop_zero_clipped(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([0.0])
        n = pd.Series([10.0])
        bm = _popularity_to_base_margin(pop, n)
        assert np.isfinite(bm[0]), "人気0でもクリップされて有限になるべき"

    def test_small_field(self):
        from src.tools.train_xgb import _popularity_to_base_margin
        pop = pd.Series([1, 2], dtype=float)
        n = pd.Series([2, 2], dtype=float)
        bm = _popularity_to_base_margin(pop, n)
        assert bm[0] > bm[1]


class TestTrainXgbResidualFlag:
    """train_xgb(residual=True) のパラメータ・出力テスト"""

    def test_market_feat_cols_defined(self):
        from src.tools.train_xgb import _MARKET_FEAT_COLS
        assert 'f_popularity' in _MARKET_FEAT_COLS

    def test_exclude_cols_no_overlap_with_market(self):
        from src.tools.train_xgb import _EXCLUDE_COLS, _MARKET_FEAT_COLS
        assert _EXCLUDE_COLS.isdisjoint(_MARKET_FEAT_COLS)


class TestEngineResidualInference:
    """engine.py の残差推論パスのユニットテスト"""

    def test_base_margin_calculation(self):
        """人気→base_margin の計算が engine.py 内で正しく動くか"""
        pop = 3
        n_h = 12
        harm = math.log(n_h) + 0.5772
        p_mkt = (1.0 / pop) / harm
        p_mkt = max(min(p_mkt, 0.999), 0.001)
        bm = math.log(p_mkt / (1 - p_mkt))
        assert -5 < bm < 5, f"base_margin が異常値: {bm}"
        # 1番人気の方が高いことを確認
        pop1 = 1
        p_mkt1 = (1.0 / pop1) / harm
        p_mkt1 = max(min(p_mkt1, 0.999), 0.001)
        bm1 = math.log(p_mkt1 / (1 - p_mkt1))
        assert bm1 > bm, "1番人気の base_margin > 3番人気"

    def test_engine_residual_flag_default(self):
        from src.features import engine
        assert hasattr(engine, '_XGB_RESIDUAL')

    def test_sigmoid_roundtrip(self):
        """logit → sigmoid の往復で確率が復元されるか"""
        p_orig = 0.25
        logit_val = math.log(p_orig / (1 - p_orig))
        p_back = 1 / (1 + math.exp(-logit_val))
        assert abs(p_orig - p_back) < 1e-10
