"""コース物理データの契約テスト。

🔑 ユーザー指示「推定値を勝手に feature として採用しないこと」を構造で守る。
"""
import json
import os

import pytest

USABLE = {'official', 'stated'}


@pytest.fixture(scope='module')
def P():
    with open(os.path.join('data', 'course_physical.json'), encoding='utf-8') as f:
        return json.load(f)


def test_every_distance_cell_declares_a_source(P):
    for k, e in P['distances'].items():
        assert e.get('src') in P['_src_levels'], f'{k} に src が無い'


def test_no_numeric_value_without_a_usable_source(P):
    """数値があるのに出典が web/未確定、という状態を作らせない。"""
    for k, e in P['distances'].items():
        if 'start_to_first_corner_m' in e:
            assert e['src'] in USABLE, f'{k} は src={e["src"]} なのに数値を持っている'


def test_qualitative_cells_carry_no_number(P):
    """「長い」だけの距離に数値を捏造していないこと。"""
    for k, e in P['distances'].items():
        if e['src'] == 'na':
            assert 'start_to_first_corner_m' not in e, k


def test_straight_course_has_no_first_corner(P):
    """新潟芝1000は直線コース。1角という概念が無いので None のまま。"""
    e = P['distances']['新潟_芝_1000']
    assert e['first_corner'] is None
    assert 'start_to_first_corner_m' not in e


def test_merge_distance_is_not_filed_as_first_corner(P):
    """🔴 中京芝1600の200mは『本線合流まで』。1角までではない。"""
    assert '中京_芝_1600' not in P['distances']
    sp = P['special']['中京_芝_1600']
    assert 'start_to_mainline_merge_m' in sp
    assert 'start_to_first_corner_m' not in sp


def test_lower_bound_is_not_stored_as_a_point_estimate(P):
    """阪神芝1600外は『500m以上』。点推定にしない。"""
    sp = P['special']['阪神_芝_1600']
    assert 'start_to_first_corner_min_m' in sp
    assert 'start_to_first_corner_m' not in sp


def test_inner_and_outer_are_separate_courses(P):
    """内回りと外回りで直線長・高低差が別物であること（本番の20キーには無い区別）。"""
    for v, inner, outer in (('京都', 328.4, 403.7), ('阪神', 356.5, 473.6),
                            ('新潟', 358.7, 658.7)):
        c = P['courses']
        assert c[f'{v}_芝_内回り']['final_straight_m'] == inner
        assert c[f'{v}_芝_外回り']['final_straight_m'] == outer
        assert (c[f'{v}_芝_内回り']['course_elevation_range_m']
                != c[f'{v}_芝_外回り']['course_elevation_range_m'])


def test_elevation_range_is_not_the_final_slope(P):
    """🔴 2つの別の量。中山芝は全体5.3m / ゴール前坂2.2m。混ぜない。"""
    assert P['courses']['中山_芝']['course_elevation_range_m'] == 5.3
    with open(os.path.join('data', 'course_distance_profiles.json'), encoding='utf-8') as f:
        slope = json.load(f)['hill']['中山_芝']['elevation_diff_m']
    assert slope == 2.2
    assert '_elevation_warning' in P


def test_web_values_are_kept_for_cross_checking(P):
    """既存web由来と一致したことが後から確認できること。"""
    ov = {k: e for k, e in P['distances'].items() if 'web_delta_m' in e}
    assert len(ov) >= 4
    assert all(abs(e['web_delta_m']) <= 5 for e in ov.values()), ov
