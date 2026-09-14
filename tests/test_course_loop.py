"""内回り/外回りの解決（#255 で見つかった 407レースの誤りの回帰テスト）。

🔴 修正前は venue×surface の20キーしか引けず、**外回りの直線長が内回りの
   レースにも適用されていた**（新潟芝内回りは実際358.7mなのに659m）。
"""
import pytest

from src.features.engine import get_course_profile, resolve_course_loop


class TestResolveCourseLoop:
    def test_resolves_unambiguous_distances(self):
        assert resolve_course_loop('新潟', '芝', 1200, '.', strict=True) == '内回り'
        assert resolve_course_loop('新潟', '芝', 1600, '.', strict=True) == '外回り'
        assert resolve_course_loop('阪神', '芝', 2000, '.', strict=True) == '内回り'
        assert resolve_course_loop('京都', '芝', 1200, '.', strict=True) == '内回り'

    def test_straight_course_is_its_own_case(self):
        assert resolve_course_loop('新潟', '芝', 1000, '.', strict=True) == '直線'

    def test_strict_refuses_to_guess_both(self):
        """🔑 '両' は同じ距離で内外どちらも使う。距離からは決まらないので推定しない。

        直線長は内外で 33〜84% 違う。ここを推測すると誤った値を静かに配る。
        """
        for rc, dist in (('阪神', 1400), ('京都', 1400), ('京都', 2000),
                         ('新潟', 1400), ('新潟', 2000)):
            assert resolve_course_loop(rc, '芝', dist, '.', strict=True) is None, (rc, dist)

    def test_non_strict_keeps_the_older_approximation(self):
        """坂・コーナー用の既存挙動（'両'も内回り扱い）は変えていない。"""
        assert resolve_course_loop('阪神', '芝', 1400, '.', strict=False) == '内回り'

    def test_dirt_and_venues_without_loops_return_none(self):
        assert resolve_course_loop('新潟', 'ダート', 1200, '.', strict=True) is None
        assert resolve_course_loop('東京', '芝', 2000, '.', strict=True) is None


class TestGetCourseProfile:
    @pytest.mark.parametrize('rc,dist,expected,old', [
        ('新潟', 1200, 358.7, 659),    # +84% の誤りだった
        ('阪神', 2000, 356.5, 474),    # +33%
        ('京都', 1200, 328.4, 404),    # +23%
    ])
    def test_inner_loop_no_longer_gets_the_outer_straight(self, rc, dist, expected, old):
        p = get_course_profile(rc, '芝', '.', distance=dist)
        assert p['straight_length'] == expected
        assert get_course_profile(rc, '芝', '.')['straight_length'] == old

    def test_straight_course_uses_its_full_length(self):
        p = get_course_profile('新潟', '芝', 1000 and '.', distance=1000)
        assert p['straight_length'] == 1000.0

    def test_straight_class_changes_where_the_length_changed(self):
        """直線長だけ直して class を直し忘れると、分類だけ古いまま残る。"""
        assert get_course_profile('京都', '芝', '.', distance=1200)['straight_class'] == 'short'
        assert get_course_profile('新潟', '芝', '.', distance=1200)['straight_class'] == 'medium'

    def test_ambiguous_distance_falls_back_to_the_base_key(self):
        base = get_course_profile('阪神', '芝', '.')
        assert get_course_profile('阪神', '芝', '.', distance=1400) == base

    def test_distance_omitted_is_backward_compatible(self):
        for rc, sf in (('新潟', '芝'), ('東京', '芝'), ('中山', 'ダート')):
            assert get_course_profile(rc, sf, '.') is not None

    def test_venues_without_a_loop_are_unaffected(self):
        for rc, dist in (('東京', 2000), ('中山', 2000), ('小倉', 1800)):
            a = get_course_profile(rc, '芝', '.', distance=dist)
            b = get_course_profile(rc, '芝', '.')
            assert a['straight_length'] == b['straight_length'], rc
