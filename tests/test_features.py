import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (
    calc_performance_index, f_recent, calc_chaos_score,
    auto_comment, dist_zone_label, dz,
    calc_course_aptitude_features, load_course_profiles, get_course_profile,
)

ROOT = os.path.join(os.path.dirname(__file__), '..')


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


# ── コース適性特徴量 ──────────────────────────────────────────────
def test_course_profiles_loads_all():
    profiles = load_course_profiles(ROOT)
    assert profiles is not None
    assert len(profiles['courses']) >= 20  # 10競馬場 × 芝/ダート
    assert get_course_profile('東京', '芝', ROOT)['straight_class'] == 'long'
    assert get_course_profile('中山', '芝', ROOT)['straight_class'] == 'short'


def test_course_aptitude_tokyo_specialist():
    history = [
        {'racecourse': '東京', 'surface': '芝', 'place': 1, 'agari3f': 33.2},
        {'racecourse': '東京', 'surface': '芝', 'place': 2, 'agari3f': 33.5},
        {'racecourse': '中山', 'surface': '芝', 'place': 12, 'agari3f': 36.8},
        {'racecourse': '中山', 'surface': '芝', 'place': 10, 'agari3f': 37.1},
    ]
    # 今日が東京芝 → 東京で2戦2好走
    feats = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    assert feats['f_same_course_rate'] == 1.0
    assert feats['f_course_coverage'] == 2
    # 今日が中山芝 → 中山で2戦2凡走
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', history, ROOT)
    assert feats['f_same_course_rate'] == 0.0
    assert feats['f_course_coverage'] == 2


def test_course_aptitude_straight_match():
    # 東京(long)で好走。今日が新潟(very_long)なら直線クラスは異なるが、
    # 同じlong同士のマッチを検証するため今日も東京で確認する。
    history = [
        {'racecourse': '東京', 'surface': '芝', 'place': 1, 'agari3f': 33.2},
        {'racecourse': '函館', 'surface': '芝', 'place': 8, 'agari3f': 35.0},
    ]
    feats = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    # straight_class=long の過去走は東京の1走（好走）のみ → 1.0
    assert feats['f_straight_match'] == 1.0
    # long コースの最速上がりは 33.2
    assert feats['f_agari_at_similar'] == 33.2


def test_course_aptitude_no_history():
    feats = calc_course_aptitude_features('新馬', '東京', '芝', [], ROOT)
    assert feats['f_same_course_rate'] == 0.0
    assert feats['f_course_coverage'] == 0
    assert feats['f_agari_at_similar'] == 99.0


def test_course_aptitude_unknown_course():
    # 未定義の競馬場（地方など）はデフォルト返却
    feats = calc_course_aptitude_features('テスト馬', '大井', 'ダート', [], ROOT)
    assert feats == {
        'f_same_course_rate': 0.0, 'f_same_turn_rate': 0.0,
        'f_straight_match': 0.0, 'f_uphill_match': 0.0,
        'f_agari_at_similar': 99.0, 'f_course_coverage': 0,
    }


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
    test_course_profiles_loads_all()
    print('✅ test_course_profiles_loads_all passed')
    test_course_aptitude_tokyo_specialist()
    print('✅ test_course_aptitude_tokyo_specialist passed')
    test_course_aptitude_straight_match()
    print('✅ test_course_aptitude_straight_match passed')
    test_course_aptitude_no_history()
    print('✅ test_course_aptitude_no_history passed')
    test_course_aptitude_unknown_course()
    print('✅ test_course_aptitude_unknown_course passed')
