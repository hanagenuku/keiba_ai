"""展開特徴量のテスト。

North Star #6 に従い、本番と同じ形（corner_all は '3-3-2-1' の文字列、
4地点そろわない '5-3' も混ざる）で検証する。
"""
import math

import pytest

from src.features.race_shape import (
    first_corner_position, summarize_horse_shape, race_shape_features,
    RACE_FEATURE_COLS, CAN_LEAD_POS,
)


def run(corner, fs=12, agari=35.0):
    return {'corner_all': corner, 'field_size': fs, 'agari3f': agari}


class TestFirstCornerPosition:
    def test_leader_is_zero_and_last_is_one(self):
        assert first_corner_position('1-1-2-3', 12) == 0.0
        assert first_corner_position('12-11-10-9', 12) == 1.0

    def test_two_point_records_use_the_first_available_corner(self):
        """4地点そろうのは42.9%。残りは3-4角のみで、そこも同じ列として扱う。"""
        assert first_corner_position('5-3', 11) == pytest.approx(0.4)

    def test_rejects_impossible_and_missing(self):
        assert first_corner_position('', 12) is None
        assert first_corner_position(None, 12) is None
        assert first_corner_position('0-1', 12) is None     # 0着順はありえない
        assert first_corner_position('13-1', 12) is None    # 頭数超過
        assert first_corner_position('1-1', 1) is None      # 1頭立ては正規化不能


class TestSummarizeHorseShape:
    def test_front_runner_and_closer_are_separated(self):
        front = summarize_horse_shape([run('1-1-1-1'), run('2-2-1-1')])
        closer = summarize_horse_shape([run('12-12-11-9'), run('11-10-10-8')])
        assert front['p_mean'] < closer['p_mean']
        assert front['p_led'] > closer['p_led']

    def test_no_history_returns_none_not_a_fabricated_value(self):
        """履歴が無い馬に 0.5 等を捏造しない（レース側で既定値にする）。"""
        s = summarize_horse_shape([])
        assert s['p_mean'] is None and s['p_min'] is None
        assert s['p_n'] == 0

    def test_broken_rows_are_skipped_not_counted(self):
        s = summarize_horse_shape([run('1-1-1-1', agari=None), run('', agari=None),
                                   run(None, agari=None), run('3-3', agari=34.0)])
        assert s['p_n'] == 2, s

    def test_agari_is_collected_even_when_the_corner_is_unreadable(self):
        """通過順が壊れていても上がりは有効なデータなので独立に拾う。"""
        s = summarize_horse_shape([run('', agari=35.0), run(None, agari=34.0)])
        assert s['p_n'] == 0 and s['p_mean'] is None
        assert s['p_ag'] == pytest.approx(34.5)

    def test_min_captures_the_horse_can_lead_even_if_usually_behind(self):
        """『ハナを切れる』は平均ではなく最小で見る。1回でも逃げた馬を拾う。"""
        s = summarize_horse_shape([run('9-9-8-8'), run('1-1-1-2'), run('8-7-7-6')])
        assert s['p_min'] == 0.0
        assert s['p_mean'] > CAN_LEAD_POS


class TestRaceShapeFeatures:
    def _field(self):
        return [summarize_horse_shape([run('1-1-1-1')]),
                summarize_horse_shape([run('2-2-2-2')]),
                summarize_horse_shape([run('10-9-9-8')])]

    def test_returns_every_declared_column(self):
        f = race_shape_features(self._field(), [0.5, 0.3, 0.2], [1, 2, 3])
        assert set(f) == set(RACE_FEATURE_COLS)

    def test_solo_leader_and_contested_lead_are_different(self):
        """🔑 これが作り直しの核心。

        「逃げ馬が2頭」は旧実装だと escape_count=2 の一値になるが、
        単騎に近い(0.8/0.1) のと やり合い(0.45/0.45) は別の展開になる。
        """
        s = self._field()
        solo = race_shape_features(s, [0.80, 0.10, 0.10])
        fight = race_shape_features(s, [0.45, 0.45, 0.10])
        assert solo['P_gap'] > fight['P_gap']
        assert fight['P_ent'] > solo['P_ent']
        assert fight['P_hhi'] < solo['P_hhi']
        # P_top2 は同じでも区別できることを固定する
        assert solo['P_top2'] == pytest.approx(fight['P_top2'], abs=0.02)

    def test_probabilities_are_normalised(self):
        s = self._field()
        a = race_shape_features(s, [5.0, 3.0, 2.0])
        b = race_shape_features(s, [0.5, 0.3, 0.2])
        assert a['P_max'] == pytest.approx(b['P_max'])

    def test_empty_field_does_not_raise(self):
        f = race_shape_features([], [], [])
        assert set(f) == set(RACE_FEATURE_COLS)

    def test_horses_without_history_fall_back_at_race_level(self):
        s = [summarize_horse_shape([]), summarize_horse_shape([])]
        f = race_shape_features(s, [0.5, 0.5])
        assert f['pos_mean'] == 0.5 and f['ag_mean'] == 36.0
        assert f['n_can_lead'] == 0

    def test_leader_popularity_uses_the_most_likely_leader(self):
        f = race_shape_features(self._field(), [0.2, 0.7, 0.1], [5, 3, 9])
        assert f['pop_of_leader'] == 3.0

    def test_sentinel_popularity_is_not_used(self):
        f = race_shape_features(self._field(), [0.7, 0.2, 0.1], [99, 3, 9])
        assert f['pop_of_leader'] is None
