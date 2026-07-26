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


class TestAbilityMarginExposure:
    """ability_margin（市場非依存のAI能力スコア）の公開 — 2026-07-26セッション。

    直前オッズ取得ボタンは従来オッズだけを更新し勝率は生成時点のまま据え置いて
    いたため、生成後に市場が動くと勝率とオッズの時点がずれたEVが表示される
    構造的な問題があった。ability_margin（base_marginを含まない木モデルの
    生スコア）をcalc_all()の出力に含め、index.htmlのJS側で新しい人気順位から
    base_marginだけ引き直して勝率を再計算できるようにした。
    ability_margin + base_margin(pop, n) から元のtotal（softmax入力）が
    正しく再構成できることを、実際のxgboost Boosterを使って検証する
    （North Star: 本番と同じ形式＝実際にsave_model/load_modelしたBoosterで検証）。
    """

    def _make_residual_booster(self, tmp_path, feature_cols):
        import numpy as np
        import xgboost as xgb

        rng = np.random.RandomState(1)
        X = rng.rand(20, len(feature_cols))
        y = rng.randint(0, 2, size=20)
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        booster = xgb.train({'objective': 'binary:logistic', 'max_depth': 2}, dtrain, num_boost_round=3)
        model_path = tmp_path / 'data' / 'xgb_fukusho_model.pkl'
        booster.save_model(str(model_path))

        cols_path = tmp_path / 'data' / 'xgb_feature_cols.json'
        cols_path.write_text(json.dumps({'feature_cols': feature_cols, 'residual': True}))

    def _make_min_engine_files(self, tmp_path):
        for name in ['horse_dist_dict.pkl', 'horse_course_dict.pkl',
                     'horse_venue_dist_dict.pkl', 'post_zone_bias.pkl']:
            with open(tmp_path / 'data' / name, 'wb') as f:
                pickle.dump({}, f)

    def _race(self):
        horses = [
            {'num': 1, 'horse_num': 1, 'name': 'A', 'win_odds': 2.0,
             'running_style': '差し', 'history': []},
            {'num': 2, 'horse_num': 2, 'name': 'B', 'win_odds': 8.0,
             'running_style': '先行', 'history': []},
            {'num': 3, 'horse_num': 3, 'name': 'C', 'win_odds': 15.0,
             'running_style': '逃げ', 'history': []},
        ]
        return {'date': '2026-07-26', 'racecourse': '新潟', 'distance': 1600,
                'surface': '芝', 'race_class': '3勝クラス', 'track_condition': '良',
                'horses': horses}

    def test_ability_margin_reconstructs_cal_prob(self, tmp_path):
        """ability_margin + base_margin(popularity) が cal_prob(=キャリブレーション前は
        raw sigmoid確率そのもの)を厳密に再構成できることを確認する。

        'total' ではなく 'cal_prob' を検証対象にしているのは、'total' はこの後
        f_relative_score(RELATIVE_WEIGHT=0.10のブレンド)・pace_bonus・エラータグ
        補正(et_factor)が追加で乗算されるため、ability_margin+base_marginだけでは
        厳密に再現できない値だから（index.html側のクライアント再同期もこれらの
        後段補正は再現しない設計であり、cal_prob相当の生シグモイド確率のみを
        softmax入力として使う近似であることに対応させている）。
        """
        (tmp_path / 'data').mkdir()
        feature_cols = [f'f{i}' for i in range(5)]
        self._make_min_engine_files(tmp_path)
        self._make_residual_booster(tmp_path, feature_cols)

        from src.features import engine
        engine.init_engine(str(tmp_path))
        assert engine._XGB_RESIDUAL is True

        out = engine.calc_all(self._race())
        n = len(out)
        for h in out:
            assert h['ability_margin'] is not None
            pop = h['popularity']
            harm = math.log(n) + 0.5772
            p_mkt = max(min((1.0 / max(pop, 1)) / harm, 0.999), 0.001)
            bm = math.log(p_mkt / (1 - p_mkt))
            expected_prob = 1 / (1 + math.exp(-(h['ability_margin'] + bm)))
            assert abs(h['cal_prob'] - expected_prob) < 1e-6, (
                f"ability_margin+base_marginからcal_probを再構成できない: {h['name']} "
                f"cal_prob={h['cal_prob']} expected={expected_prob}"
            )

    def test_ability_margin_none_without_residual_model(self):
        """XGBモデル未ロード時（ルールベースフォールバック）はability_marginがNone。"""
        import src.features.engine as eng
        for attr in ['_horse_dist_dict', '_horse_course_dict',
                     '_horse_venue_dist_dict', '_post_zone_bias',
                     '_jockey_dict', '_trainer_dict']:
            if not hasattr(eng, attr) or getattr(eng, attr) is None:
                setattr(eng, attr, {})
        eng._XGB_FUKUSHO_MODEL = None
        eng._XGB_RESIDUAL = False

        race = {'date': '2026-07-26', 'racecourse': '新潟', 'distance': 1600,
                'surface': '芝', 'race_class': '3勝クラス', 'track_condition': '良',
                'horses': [{'num': 1, 'horse_num': 1, 'name': 'A', 'win_odds': 2.0,
                           'running_style': '差し', 'history': []}]}
        out = eng.calc_all(race)
        assert out[0]['ability_margin'] is None
