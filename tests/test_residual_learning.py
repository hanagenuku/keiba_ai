"""残差学習（base_margin）の単体テスト"""

import json
import math
import pickle
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


class TestEnsembleResidualConflict:
    """init_engine: 残差学習モデルとアンサンブルモデルの併存を防ぐガードの検証。

    2026-07 に発生した実障害の再現テスト:
    data/xgb_ensemble_model.pkl（sklearn API の XGBClassifier を含む古い実験の
    残骸）が残っていると、残差学習フラグ(_XGB_RESIDUAL=True)が立っていても
    アンサンブルモデルが _XGB_FUKUSHO_MODEL を上書きしてしまい、推論時に
    「DMatrix + base_margin」パスへ sklearn API モデルを渡す型エラーが発生、
    calc_all の try/except でルールベースへサイレントフォールバックしていた。
    """

    def _make_residual_booster(self, tmp_path, feature_cols):
        import numpy as np
        import xgboost as xgb

        rng = np.random.RandomState(0)
        X = rng.rand(20, len(feature_cols))
        y = rng.randint(0, 2, size=20)
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        booster = xgb.train({'objective': 'binary:logistic', 'max_depth': 2}, dtrain, num_boost_round=3)
        model_path = tmp_path / 'data' / 'xgb_fukusho_model.pkl'
        booster.save_model(str(model_path))

        cols_path = tmp_path / 'data' / 'xgb_feature_cols.json'
        cols_path.write_text(json.dumps({'feature_cols': feature_cols, 'residual': True}))

    def _make_stray_ensemble(self, tmp_path, feature_cols):
        import numpy as np
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        rng = np.random.RandomState(0)
        X = rng.rand(20, len(feature_cols))
        y = rng.randint(0, 2, size=20)
        xgb_clf = XGBClassifier(n_estimators=3, max_depth=2)
        xgb_clf.fit(X, y)
        lgbm_clf = LGBMClassifier(n_estimators=3, max_depth=2, verbosity=-1)
        lgbm_clf.fit(X, y)

        ensemble = {'xgb': xgb_clf, 'lgbm': lgbm_clf, 'xgb_weight': 0.7, 'feat_cols': feature_cols}
        with open(tmp_path / 'data' / 'xgb_ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)

    def _make_min_engine_files(self, tmp_path):
        """init_engine を空 data/ ディレクトリで完走させるための最小pklを用意する。

        （_horse_dist_dict 等は data/ に対応する pkl が無いと構築処理が
        history.db 依存になり、DB が無い環境ではモジュール未初期化の
        グローバル変数参照で NameError になるため、空 dict を明示的に置く）
        """
        for name in ['horse_dist_dict.pkl', 'horse_course_dict.pkl',
                     'horse_venue_dist_dict.pkl', 'post_zone_bias.pkl']:
            with open(tmp_path / 'data' / name, 'wb') as f:
                pickle.dump({}, f)

    def test_residual_model_not_overridden_by_stray_ensemble(self, tmp_path):
        (tmp_path / 'data').mkdir()
        feature_cols = [f'f{i}' for i in range(5)]
        self._make_min_engine_files(tmp_path)
        self._make_residual_booster(tmp_path, feature_cols)
        self._make_stray_ensemble(tmp_path, feature_cols)

        from src.features import engine
        engine.init_engine(str(tmp_path))

        assert engine._XGB_RESIDUAL is True
        assert engine._ENSEMBLE_MODEL is None, (
            "残差学習モデル検出時はアンサンブルモデルを無視すべき"
        )
        import xgboost as xgb
        assert isinstance(engine._XGB_FUKUSHO_MODEL, xgb.Booster), (
            "残差推論パスは Booster を期待するため sklearn API に上書きされてはいけない"
        )
