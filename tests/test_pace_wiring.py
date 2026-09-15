"""展開予想モデルを本番に繋いだ経路の回帰テスト（2026-09-15）。

🔴 ここで固定したいのは精度ではなく **パリティ**。
   旧ペースモデルは学習時に「そのレースの実際の脚質・実測agari3f」を見て、
   推論時には「過去走からの推定・定数36.0/1.5」を渡されていた。
   推定脚質と実際の脚質の一致率は 39.8% で、3分類の精度は
   学習時入力 52.80% に対し推論時入力では 50.45% しか出ていなかった。

   新経路は**レース条件だけ**を入力にするので、この違反が起こせない。
   「出走馬の中身を変えても結果が1ミリも動かない」ことをテストで固定する。
"""
import os
import pickle
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import engine
from src.features.race_shape import (
    PACE_INPUT_COLS, pace_model_inputs, pace_probs_from_seconds,
    class_level, dist_zone,
)


def _build_model():
    """本番と同じ形（xgboost回帰 + 閾値 + sigma）の小さなモデルを実際に学習して返す。

    North Star #6: 手打ちのスタブではなく、本番と同じ型の成果物で検証する。
    """
    import xgboost as xgb
    rows, y = [], []
    for dist in (1200, 1600, 2000):
        for surf in ('芝', 'ダート'):
            for cls in ('未勝利', '1勝', 'OP'):
                for n in (10, 14, 18):
                    r = pace_model_inputs(dist, surf, cls, n)
                    rows.append(r)
                    # 距離が延びるほど前半は遅く、ダートは速い、という単純な関係
                    y.append(33.0 + (dist - 1200) / 400.0 + (0.5 if surf == '芝' else 0.0))
    X = pd.DataFrame(rows)[PACE_INPUT_COLS]
    m = xgb.XGBRegressor(n_estimators=40, max_depth=3, random_state=0, verbosity=0)
    m.fit(X, y)
    th = {}
    for surf in ('芝', 'ダート'):
        for dz in ('~1400', '1401-1800', '1801-2200', '2201~'):
            th[f'{surf}|{dz}'] = [34.0, 35.0]
    return {'version': 1, 'pace_model': m, 'pace_cols': PACE_INPUT_COLS,
            'sigma': 0.8, 'thresholds': th,
            'lead': {'cols': [], 'coef': [], 'intercept': 0.0,
                     'mu': [], 'sd': [], 'median': [], 'auc': 0.78},
            'meta': {'rmse': 0.85}}


@pytest.fixture
def model(monkeypatch):
    m = _build_model()
    monkeypatch.setattr(engine, '_RACE_SHAPE_MODEL', m)
    return m


def _race(horses, **kw):
    base = {'id': 'R1', 'racecourse': '東京', 'surface': '芝', 'distance': 1600,
            'race_class': '1勝', 'horses': horses}
    base.update(kw)
    return base


def _horses(style, agari, n=12):
    return [{'horse_num': i + 1, 'running_style': style, 'agari3f': agari,
             'post_position': i + 1}
            for i in range(n)]


class TestParityIsStructural:
    def test_horse_composition_does_not_move_the_pace_prediction(self, model):
        """🔴 これが本命。出走馬の脚質・上がりを総入れ替えしても結果が同じであること。

        旧経路（escape_count / avg_agari3f を入力に持つ）では必ず値が動く。
        新経路は条件だけを見るので動いてはいけない。動いたら
        「学習時は実際の脚質・推論時は推定脚質」という違反が戻ってきている。
        """
        all_escape = engine.calc_pace_distribution(
            _race(_horses('逃げ', 33.0)))
        all_closer = engine.calc_pace_distribution(
            _race(_horses('追込', 38.5)))
        assert all_escape == all_closer

    def test_track_condition_does_not_move_it(self, model):
        """馬場状態は推論時つねに『良』（既知のパリティ違反）なので入力にしない。"""
        good = engine.calc_pace_distribution(
            _race(_horses('先行', 35.0), track_condition='良'))
        heavy = engine.calc_pace_distribution(
            _race(_horses('先行', 35.0), track_condition='不良'))
        assert good == heavy

    def test_only_the_four_declared_columns_are_used(self, model):
        """モデルに渡る列が宣言した4つだけであること（列が増えたら気づく）。"""
        seen = {}
        orig = model['pace_model'].predict

        def spy(X, *a, **k):
            seen['cols'] = list(X.columns)
            return orig(X, *a, **k)

        model['pace_model'].predict = spy
        engine.calc_pace_distribution(_race(_horses('逃げ', 33.0)))
        assert seen['cols'] == ['dist', 'surface_num', 'cls', 'n_horses']


class TestPaceDistribution:
    def test_faster_predicted_pace_means_more_high(self, model):
        """予測した前半600mが速いほど P(high) が大きいこと（向きの固定）。"""
        short = engine.calc_pace_distribution(_race(_horses('先行', 35.0), distance=1200))
        long_ = engine.calc_pace_distribution(_race(_horses('先行', 35.0), distance=2000))
        assert short['high'] > long_['high']
        assert long_['slow'] > short['slow']

    def test_probabilities_sum_to_one(self, model):
        d = engine.calc_pace_distribution(_race(_horses('差し', 36.0)))
        assert abs(sum(d.values()) - 1.0) < 0.01
        assert set(d) == {'high', 'mid', 'slow'}

    def test_prediction_is_exposed_for_inspection(self, model):
        r = _race(_horses('差し', 36.0))
        engine.calc_pace_distribution(r)
        assert 25.0 < r['_early_pace_pred'] < 45.0

    def test_number_of_runners_is_allowed_to_matter(self, model):
        """頭数は**レース条件**なので入力してよい（脚質と違い出走表から取れる）。"""
        r = _race(_horses('先行', 35.0, n=8))
        assert engine.calc_pace_distribution(r) is not None


class TestFallback:
    def test_falls_back_when_model_absent(self, monkeypatch):
        monkeypatch.setattr(engine, '_RACE_SHAPE_MODEL', None)
        monkeypatch.setattr(engine, '_PACE_MODEL', None)
        d = engine.calc_pace_distribution(_race(_horses('逃げ', 33.0)))
        assert set(d) == {'high', 'mid', 'slow'}
        assert abs(sum(d.values()) - 1.0) < 0.01

    def test_broken_model_does_not_stop_the_prediction(self, monkeypatch):
        """モデルが壊れていても予想自体は止めない（警告して旧経路へ）。"""
        class Boom:
            def predict(self, X):
                raise RuntimeError('boom')
        monkeypatch.setattr(engine, '_RACE_SHAPE_MODEL',
                            {'pace_model': Boom(), 'pace_cols': PACE_INPUT_COLS,
                             'sigma': 0.8, 'thresholds': {'芝|1401-1800': [34.0, 35.0]}})
        monkeypatch.setattr(engine, '_PACE_MODEL', None)
        d = engine.calc_pace_distribution(_race(_horses('逃げ', 33.0)))
        assert set(d) == {'high', 'mid', 'slow'}

    def test_unknown_distance_zone_falls_back(self, model):
        """閾値の無い組み合わせは None を返し、旧経路へ落ちること。"""
        model['thresholds'] = {}
        d = engine.calc_pace_distribution(_race(_horses('逃げ', 33.0)))
        assert set(d) == {'high', 'mid', 'slow'}


class TestSharedInputBuilder:
    def test_class_is_ordinal_not_a_category_code(self):
        """クラスは序数。カテゴリ番号だとDBと出馬表で採番がずれる。"""
        assert class_level('未勝利') < class_level('1勝') < class_level('OP')
        assert class_level('1勝') == class_level('1勝クラス')

    def test_unknown_class_gets_a_defined_default(self):
        assert class_level('') == class_level(None) == 3.0
        assert class_level('謎クラス') == 3.0

    def test_dist_zone_boundaries(self):
        assert dist_zone(1400) == '~1400'
        assert dist_zone(1401) == '1401-1800'
        assert dist_zone(1800) == '1401-1800'
        assert dist_zone(2200) == '1801-2200'
        assert dist_zone(2201) == '2201~'

    def test_inputs_have_exactly_the_declared_keys(self):
        row = pace_model_inputs(1600, '芝', '1勝', 16)
        assert sorted(row) == sorted(PACE_INPUT_COLS)

    def test_surface_flag(self):
        assert pace_model_inputs(1600, '芝', '1勝', 16)['surface_num'] == 1.0
        assert pace_model_inputs(1600, 'ダート', '1勝', 16)['surface_num'] == 0.0

    def test_probs_none_without_thresholds(self):
        assert pace_probs_from_seconds(34.0, 0.8, None) is None
        assert pace_probs_from_seconds(None, 0.8, [34.0, 35.0]) is None
