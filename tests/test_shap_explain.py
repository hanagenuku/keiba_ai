"""shap_explain.py（ability_marginのカテゴリ別SHAP寄与度分解）の単体テスト"""

import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest


class TestFeatureCategoryMapCoverage:
    """FEATURE_CATEGORY_MAP が本番の特徴量リストを網羅しているか"""

    def test_covers_all_production_feature_cols(self):
        """data/xgb_feature_cols.json（本番の特徴量リスト）に存在する列が
        全てFEATURE_CATEGORY_MAPに登録されていることを確認する。

        未登録でもcompute_ability_breakdown自体は「その他」にフォールバックし
        壊れないが、新特徴量追加時にカテゴリ分類が漏れたまま気づかれないのを
        防ぐための早期警告テスト。
        """
        from src.features.shap_explain import FEATURE_CATEGORY_MAP

        with open('data/xgb_feature_cols.json') as f:
            cols = json.load(f)['feature_cols']

        missing = [c for c in cols if c not in FEATURE_CATEGORY_MAP]
        assert not missing, (
            f"本番特徴量{len(missing)}列がFEATURE_CATEGORY_MAPに未登録（'その他'に"
            f"フォールバックする）: {missing}。src/features/shap_explain.pyの"
            f"FEATURE_CATEGORY_MAPに追記すること"
        )


class TestComputeAbilityBreakdown:
    """compute_ability_breakdown の計算検証（実際のxgb.Boosterを使用）"""

    def _make_booster(self, feature_cols, base_margin_train):
        import xgboost as xgb

        rng = np.random.RandomState(0)
        n = 20
        X = rng.rand(n, len(feature_cols))
        y = rng.randint(0, 2, size=n)
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        dtrain.set_base_margin(base_margin_train)
        return xgb.train({'objective': 'binary:logistic', 'max_depth': 3},
                          dtrain, num_boost_round=10)

    def test_breakdown_sums_to_ability_margin(self):
        """breakdownの寄与度合計が ability_margin(=raw_margin - base_margin) と一致する"""
        import xgboost as xgb
        from src.features.shap_explain import compute_ability_breakdown

        feature_cols = ['f_jockey', 'f_dist_fukusho', 'f_speed_avg', 'f_post', 'f_blood']
        rng = np.random.RandomState(0)
        bm_train = rng.rand(20) * 2 - 1
        booster = self._make_booster(feature_cols, bm_train)

        Xtest = pd.DataFrame(rng.rand(1, len(feature_cols)), columns=feature_cols)
        bm_test = 0.42

        dmat = xgb.DMatrix(Xtest, feature_names=feature_cols)
        dmat.set_base_margin([bm_test])
        raw_margin = float(booster.predict(dmat, output_margin=True)[0])
        ability_margin = raw_margin - bm_test

        breakdown = compute_ability_breakdown(booster, Xtest, feature_cols, bm_test)
        total = sum(b['contrib'] for b in breakdown)
        assert abs(total - ability_margin) < 1e-3, (
            f"breakdown合計がability_marginと一致しない: {total} vs {ability_margin}"
        )

    def test_breakdown_groups_by_category(self):
        """既知カテゴリの特徴量は正しいカテゴリ名に集約される"""
        import xgboost as xgb
        from src.features.shap_explain import compute_ability_breakdown

        feature_cols = ['f_jockey', 'f_jockey_rate', 'f_dist_fukusho']
        rng = np.random.RandomState(1)
        bm_train = rng.rand(20) * 2 - 1
        booster = self._make_booster(feature_cols, bm_train)

        Xtest = pd.DataFrame(rng.rand(1, len(feature_cols)), columns=feature_cols)
        dmat = xgb.DMatrix(Xtest, feature_names=feature_cols)
        dmat.set_base_margin([0.0])

        breakdown = compute_ability_breakdown(booster, Xtest, feature_cols, 0.0)
        categories = {b['category'] for b in breakdown}
        # f_jockey/f_jockey_rateは'騎手'に、f_dist_fukushoは'距離適性'に集約される
        assert categories <= {'騎手', '距離適性', '基準値'}

    def test_breakdown_sorted_descending(self):
        """breakdownはcontrib降順でソートされている"""
        import xgboost as xgb
        from src.features.shap_explain import compute_ability_breakdown

        feature_cols = [f'f_speed_avg' if i == 0 else f'f_jockey' if i == 1 else f'f_post'
                        for i in range(3)]
        rng = np.random.RandomState(3)
        bm_train = rng.rand(20) * 2 - 1
        booster = self._make_booster(feature_cols, bm_train)

        Xtest = pd.DataFrame(rng.rand(1, len(feature_cols)), columns=feature_cols)
        breakdown = compute_ability_breakdown(booster, Xtest, feature_cols, 0.3)
        contribs = [b['contrib'] for b in breakdown]
        assert contribs == sorted(contribs, reverse=True)


class TestAbilityBreakdownWiredIntoEngine:
    """calc_all() が ability_breakdown を正しく出力に含めるかの統合テスト
    （North Star: 実際にsave_model/load_modelしたBoosterで検証）"""

    def _make_residual_booster(self, tmp_path, feature_cols):
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

    def test_calc_all_includes_ability_breakdown_for_residual_model(self, tmp_path):
        (tmp_path / 'data').mkdir()
        feature_cols = [f'f{i}' for i in range(5)]
        self._make_min_engine_files(tmp_path)
        self._make_residual_booster(tmp_path, feature_cols)

        from src.features import engine
        engine.init_engine(str(tmp_path))
        assert engine._XGB_RESIDUAL is True

        out = engine.calc_all(self._race())
        for h in out:
            assert h['ability_breakdown'] is not None
            assert isinstance(h['ability_breakdown'], list)
            assert len(h['ability_breakdown']) > 0
            for entry in h['ability_breakdown']:
                assert set(entry.keys()) == {'category', 'contrib'}
            # breakdown合計はability_marginとほぼ一致する
            total = sum(e['contrib'] for e in h['ability_breakdown'])
            assert abs(total - h['ability_margin']) < 1e-3

    def test_calc_all_ability_breakdown_none_without_residual_model(self):
        """XGBモデル未ロード時（ルールベースフォールバック）はability_breakdownがNone"""
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
        assert out[0]['ability_breakdown'] is None
