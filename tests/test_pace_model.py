"""展開予測モデル（train_pace_model / calc_pace_distribution 拡張）のテスト"""

import math
import pytest


class TestClassifyPace:
    """_classify_pace の分類ロジック検証"""

    def _set_cache(self):
        """テスト用にパーセンタイルキャッシュを設定"""
        import src.tools.train_pace_model as pm
        pm._PACE_PERCENTILE_CACHE = {
            ('芝', 'sprint'): (34.0, 35.5),
            ('芝', 'mile'): (35.0, 36.5),
            ('芝', 'mid'): (35.5, 37.0),
            ('芝', 'long'): (36.0, 37.5),
            ('ダート', 'sprint'): (35.5, 37.0),
            ('ダート', 'mile'): (36.0, 37.5),
        }

    def test_high_pace(self):
        self._set_cache()
        from src.tools.train_pace_model import _classify_pace
        assert _classify_pace(33.5, 1200, '芝') == 'high'

    def test_slow_pace(self):
        self._set_cache()
        from src.tools.train_pace_model import _classify_pace
        assert _classify_pace(36.0, 1200, '芝') == 'slow'

    def test_mid_pace(self):
        self._set_cache()
        from src.tools.train_pace_model import _classify_pace
        assert _classify_pace(34.5, 1200, '芝') == 'mid'

    def test_longer_distance_independent(self):
        self._set_cache()
        from src.tools.train_pace_model import _classify_pace
        # 2400m芝: 閾値は (36.0, 37.5) なので 36.5 は mid
        assert _classify_pace(36.5, 2400, '芝') == 'mid'
        # 35.5 は high
        assert _classify_pace(35.5, 2400, '芝') == 'high'

    def test_label_override(self):
        self._set_cache()
        from src.tools.train_pace_model import _classify_pace
        assert _classify_pace(33.0, 1200, '芝', 'slow') == 'slow'
        assert _classify_pace(33.0, 1200, '芝', 'S') == 'slow'
        assert _classify_pace(37.0, 1200, '芝', 'H') == 'high'
        assert _classify_pace(37.0, 1200, '芝', 'M') == 'mid'

    def test_missing_data_returns_none(self):
        from src.tools.train_pace_model import _classify_pace
        assert _classify_pace(0, 1200, '芝') is None
        assert _classify_pace(35.0, 0, '芝') is None
        assert _classify_pace(None, 1200, '芝') is None

    def test_no_cache_returns_mid(self):
        import src.tools.train_pace_model as pm
        pm._PACE_PERCENTILE_CACHE = {}
        assert pm._classify_pace(35.0, 1200, '芝') == 'mid'


class TestBuildPaceFeaturesForInference:
    """engine.py の _build_pace_features_for_inference 検証"""

    def test_basic_output_keys(self):
        from src.features.engine import _build_pace_features_for_inference
        race = {
            'horses': [
                {'horse_num': 1, 'running_style': '逃げ', 'jockey': 'A', 'agari3f': 34.5, 'popularity': 3},
                {'horse_num': 2, 'running_style': '先行', 'jockey': 'B', 'agari3f': 35.0, 'popularity': 1},
                {'horse_num': 3, 'running_style': '差し', 'jockey': 'C', 'agari3f': 33.8, 'popularity': 5},
            ],
            'escape_count': 1,
            'front_count': 1,
            'distance': 1600,
            'surface': '芝',
            'racecourse': '東京',
            'track_condition': '良',
        }
        feats = _build_pace_features_for_inference(race)
        expected_keys = {
            'escape_count', 'front_count', 'front_density',
            'avg_agari3f', 'std_agari3f', 'runner_count',
            'distance', 'surface_num',
            'escape_avg_pos', 'escape_outer_ratio', 'escape_avg_pop',
            'straight_length', 'straight_class', 'has_uphill', 'n_corners',
            'jockey_pace_median', 'jockey_escape_pct', 'condition_num',
        }
        assert set(feats.keys()) == expected_keys

    def test_escape_count_matches(self):
        from src.features.engine import _build_pace_features_for_inference
        race = {
            'horses': [
                {'horse_num': 1, 'running_style': '逃げ', 'jockey': 'A', 'popularity': 2},
                {'horse_num': 2, 'running_style': '逃げ', 'jockey': 'B', 'popularity': 5},
                {'horse_num': 3, 'running_style': '差し', 'jockey': 'C', 'popularity': 1},
            ],
            'escape_count': 2,
            'front_count': 0,
            'distance': 1200,
            'surface': '芝',
            'racecourse': '中山',
        }
        feats = _build_pace_features_for_inference(race)
        assert feats['escape_count'] == 2
        assert feats['escape_avg_pos'] == 1.5  # (1+2)/2
        assert feats['escape_avg_pop'] == 3.5  # (2+5)/2

    def test_no_escape_horses(self):
        from src.features.engine import _build_pace_features_for_inference
        race = {
            'horses': [
                {'horse_num': 1, 'running_style': '差し', 'jockey': 'A'},
                {'horse_num': 2, 'running_style': '追込', 'jockey': 'B'},
            ],
            'escape_count': 0,
            'front_count': 0,
            'distance': 2000,
            'surface': 'ダート',
            'racecourse': '阪神',
        }
        feats = _build_pace_features_for_inference(race)
        assert feats['escape_avg_pos'] == 1.0  # n/2 = 2/2
        assert feats['escape_outer_ratio'] == 0.0
        assert feats['escape_avg_pop'] == 8.0  # default

    def test_surface_num_encoding(self):
        from src.features.engine import _build_pace_features_for_inference
        race_turf = {'horses': [{'horse_num': 1, 'running_style': '差し', 'jockey': 'A'}],
                     'escape_count': 0, 'front_count': 0, 'distance': 1600, 'surface': '芝'}
        race_dirt = {'horses': [{'horse_num': 1, 'running_style': '差し', 'jockey': 'A'}],
                     'escape_count': 0, 'front_count': 0, 'distance': 1600, 'surface': 'ダート'}
        assert _build_pace_features_for_inference(race_turf)['surface_num'] == 1
        assert _build_pace_features_for_inference(race_dirt)['surface_num'] == 0

    def test_condition_num_encoding(self):
        from src.features.engine import _build_pace_features_for_inference
        base = {'horses': [{'horse_num': 1, 'running_style': '差し', 'jockey': 'A'}],
                'escape_count': 0, 'front_count': 0, 'distance': 1600, 'surface': '芝'}
        for cond, expected in [('良', 0), ('稍重', 1), ('重', 2), ('不良', 3)]:
            race = {**base, 'track_condition': cond}
            assert _build_pace_features_for_inference(race)['condition_num'] == expected


class TestCalcPaceDistribution:
    """calc_pace_distribution の後方互換・新モデル対応テスト"""

    def test_fallback_without_model(self):
        """ペースモデルなしでもルールベースで動く"""
        import src.features.engine as eng
        orig = eng._PACE_MODEL
        eng._PACE_MODEL = None
        try:
            race = {
                'horses': [
                    {'running_style': '逃げ', 'post_position': 1},
                    {'running_style': '先行', 'post_position': 3},
                    {'running_style': '差し', 'post_position': 5},
                ],
                'escape_count': 1,
                'front_count': 1,
                'distance': 1600,
                'surface': '芝',
                'racecourse': '東京',
            }
            result = eng.calc_pace_distribution(race)
            assert 'high' in result
            assert 'mid' in result
            assert 'slow' in result
            total = result['high'] + result['mid'] + result['slow']
            assert abs(total - 1.0) < 0.01
        finally:
            eng._PACE_MODEL = orig

    def test_probabilities_sum_to_one(self):
        """ルールベース出力が合計1になる"""
        import src.features.engine as eng
        orig = eng._PACE_MODEL
        eng._PACE_MODEL = None
        try:
            race = {
                'horses': [{'running_style': '逃げ', 'post_position': i} for i in range(1, 5)],
                'escape_count': 4,
                'front_count': 0,
                'distance': 1200,
                'surface': '芝',
                'racecourse': '中山',
            }
            result = eng.calc_pace_distribution(race)
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01
            assert result['high'] > result['slow']
        finally:
            eng._PACE_MODEL = orig


class TestJockeyPaceStats:
    """_JOCKEY_PACE_STATS のロードテスト"""

    def test_global_exists(self):
        from src.features import engine
        assert hasattr(engine, '_JOCKEY_PACE_STATS')

    def test_default_is_empty_dict(self):
        from src.features import engine
        assert isinstance(engine._JOCKEY_PACE_STATS, dict)


class TestTrainPaceModelModule:
    """train_pace_model モジュールのインポートテスト"""

    def test_import(self):
        from src.tools.train_pace_model import train_pace_model
        assert callable(train_pace_model)

    def test_percentile_cache_defined(self):
        from src.tools.train_pace_model import _PACE_PERCENTILE_CACHE
        assert isinstance(_PACE_PERCENTILE_CACHE, dict)


class TestFormatAccuracyDelta:
    """新旧モデルAccuracy比較ラベルの検証（同値ケースの回帰テスト）。

    2026-07-23、Colabでの実行時に新旧モデルのAccuracyが完全に同値(0.5436)
    だったにもかかわらず「↓悪化」と表示される不具合を発見した。厳密な `>` 比較
    だけだと acc == old_acc が常にelse節（悪化）に落ちることが原因だった。
    """

    def test_improved(self):
        from src.tools.train_pace_model import _format_accuracy_delta
        assert _format_accuracy_delta(0.60, 0.55) == '↑改善'

    def test_worsened(self):
        from src.tools.train_pace_model import _format_accuracy_delta
        assert _format_accuracy_delta(0.50, 0.55) == '↓悪化'

    def test_tied_is_not_worsened(self):
        from src.tools.train_pace_model import _format_accuracy_delta
        assert _format_accuracy_delta(0.5436, 0.5436) == '→変化なし'
