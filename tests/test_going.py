"""当日の馬場状態の取得と反映（2026-09-15）。

固定したいこと:
  - **時間を遡らない**（先に終わったレースの情報で、それより前のレースを塗らない）
  - 反映すると f_track_cond が実際に動く（＝パリティ違反が解消している）
  - 会場×馬場が違えば別物として扱う（芝とダートの馬場は別に発表される）
  - 上限を超えたら黙って諦め、**予想は止めない**（2026-07-18 のタイムアウト事故）
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.going import (
    apply_going_to_races, collect_today_going,
    GOING_MISS_STREAK, GOING_FETCH_BUDGET_PER_VENUE,
)
from src.features.engine import calc_features_for_xgb


def _race(num, rc='東京', surf='芝'):
    return {'id': f'r{num}', 'race_num': num, 'racecourse': rc, 'surface': surf,
            'distance': 1600, 'race_class': '1勝', 'horses': []}


class TestApply:
    def test_applies_to_later_races_only(self):
        going = {('東京', '芝'): {'track_condition': '重', 'weather': '雨', 'race_num': 5}}
        races = [_race(4), _race(5), _race(6)]
        n = apply_going_to_races(races, going)
        assert n == 1
        assert 'track_condition' not in races[0]   # R4 は参照元より前
        assert 'track_condition' not in races[1]   # R5 は参照元そのもの
        assert races[2]['track_condition'] == '重'
        assert races[2]['weather'] == '雨'

    def test_surface_is_not_mixed(self):
        going = {('東京', 'ダート'): {'track_condition': '不良', 'race_num': 3}}
        races = [_race(8, surf='芝'), _race(9, surf='ダート')]
        apply_going_to_races(races, going)
        assert 'track_condition' not in races[0]
        assert races[1]['track_condition'] == '不良'

    def test_venue_is_not_mixed(self):
        going = {('東京', '芝'): {'track_condition': '重', 'race_num': 1}}
        races = [_race(9, rc='阪神')]
        assert apply_going_to_races(races, going) == 0

    def test_unknown_venue_leaves_race_untouched(self):
        races = [_race(9)]
        assert apply_going_to_races(races, {}) == 0
        assert 'track_condition' not in races[0]

    def test_records_where_the_value_came_from(self):
        going = {('東京', '芝'): {'track_condition': '稍重', 'race_num': 2}}
        races = [_race(7)]
        apply_going_to_races(races, going)
        assert races[0]['_going_source'] == '東京芝 R2'

    def test_missing_race_num_is_still_applied(self):
        """出馬表に番号が無いケースでも落ちない（安全側に反映する）。"""
        going = {('東京', '芝'): {'track_condition': '重', 'race_num': 3}}
        races = [{'racecourse': '東京', 'surface': '芝'}]
        assert apply_going_to_races(races, going) == 1


class TestItActuallyMovesTheFeature:
    def test_f_track_cond_stops_being_stuck_at_zero(self):
        """🔴 これが本題。反映前は必ず 0.0（良）だった。"""
        horse = {'name': 'テスト', 'horse_num': 1, 'history': [],
                 'jockey': '', 'trainer': '', 'weight_load': 56.0,
                 'age': 4, 'sex': '牡', 'popularity': 3, 'win_odds': 5.0,
                 'jockey_rate': 0.15, 'trainer_rate': 0.12}
        race = _race(9)
        before = calc_features_for_xgb(horse, dict(race, horses=[horse]))
        assert before['f_track_cond'] == 0.0

        apply_going_to_races(
            [race], {('東京', '芝'): {'track_condition': '不良', 'race_num': 1}})
        after = calc_features_for_xgb(horse, dict(race, horses=[horse]))
        assert after['f_track_cond'] == 3.0


class _Sess:
    def post(self, *a, **k):
        raise RuntimeError('ネットワークに出てはいけない')


class TestCollect:
    def _patch(self, monkeypatch, results_by_race, r01=0xA0):
        import src.scraper.jra_scraper as js
        import src.scraper.calendar as cal
        calls = {'n': 0}
        monkeypatch.setattr(cal, 'get_kaisai_on_date',
                            lambda d, s=None: ['pw01dde0105202601011'])
        monkeypatch.setattr(js, 'find_r01_result', lambda *a, **k: r01)
        monkeypatch.setattr(js, 'calc_suffix', lambda a, b: '00')
        monkeypatch.setattr(js, 'PLACE_NAMES', {'05': '東京'})

        def _try(sess, base, r, date, sx):
            calls['n'] += 1
            return 'soup' if r in results_by_race else None

        monkeypatch.setattr(js, '_try_fetch_result', _try)
        monkeypatch.setattr(js, 'parse_result_soup',
                            lambda soup, rc, r, d, pc: results_by_race.get(r))
        return calls

    def test_takes_the_latest_race_per_surface(self, monkeypatch):
        res = {
            1: {'surface': '芝', 'track_condition': '良', 'weather': '晴'},
            2: {'surface': 'ダート', 'track_condition': '稍重', 'weather': '晴'},
            3: {'surface': '芝', 'track_condition': '稍重', 'weather': '雨'},
        }
        self._patch(monkeypatch, res)
        going = collect_today_going(_Sess(), '20260915')
        assert going[('東京', '芝')]['track_condition'] == '稍重'
        assert going[('東京', '芝')]['race_num'] == 3
        assert going[('東京', 'ダート')]['track_condition'] == '稍重'

    def test_stops_at_the_first_unfinished_races(self, monkeypatch):
        """まだ終わっていないレースに連続で当たったら打ち切る（上限を溶かさない）。"""
        res = {1: {'surface': '芝', 'track_condition': '良', 'weather': '晴'}}
        calls = self._patch(monkeypatch, res)
        collect_today_going(_Sess(), '20260915')
        assert calls['n'] <= 1 + GOING_MISS_STREAK

    def test_never_exceeds_the_per_venue_budget(self, monkeypatch):
        """全レース取れてしまう日でも会場あたりの上限を超えない。"""
        res = {r: {'surface': '芝', 'track_condition': '良', 'weather': '晴'}
               for r in range(1, 30)}
        calls = self._patch(monkeypatch, res)
        collect_today_going(_Sess(), '20260915')
        assert calls['n'] <= GOING_FETCH_BUDGET_PER_VENUE

    def test_no_kaisai_returns_empty_without_raising(self, monkeypatch):
        import src.scraper.calendar as cal
        monkeypatch.setattr(cal, 'get_kaisai_on_date', lambda d, s=None: [])
        assert collect_today_going(_Sess(), '20260915') == {}

    def test_time_budget_aborts_quietly(self, monkeypatch):
        res = {r: {'surface': '芝', 'track_condition': '良', 'weather': '晴'}
               for r in range(1, 30)}
        self._patch(monkeypatch, res)
        assert collect_today_going(_Sess(), '20260915', time_budget_sec=-1) == {}


class TestWiring:
    def test_refresh_only_collects_after_10am(self):
        """8時台のrunでは1レースも終わっていないので実行しない。"""
        src = open(os.path.join(os.path.dirname(__file__), '..',
                                'scripts', 'weekend.py'), encoding='utf-8').read()
        assert 'collect_today_going' in src
        assert 'jst_now.hour >= 10' in src

    def test_refresh_applies_only_to_upcoming_races(self):
        """発走済みレースには触らない（確定オッズ混入と同じ線引き）。"""
        src = open(os.path.join(os.path.dirname(__file__), '..',
                                'scripts', 'weekend.py'), encoding='utf-8').read()
        assert 'apply_going_to_races(upcoming' in src

    def test_has_a_kill_switch(self):
        src = open(os.path.join(os.path.dirname(__file__), '..',
                                'scripts', 'weekend.py'), encoding='utf-8').read()
        assert 'COLLECT_GOING' in src
