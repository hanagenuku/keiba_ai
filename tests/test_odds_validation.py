"""単勝オッズの下限バリデーション（2026-07-27⑧導入）。

JRAの単勝オッズは最低1.0倍（元返し）で1.0未満は原理的に存在しない。
オッズ取得が部分的に失敗した際に0.1〜0.4倍のゴミ値が混入し、
オッズ昇順で導出する popularity が壊れてレース全体の base_margin が
誤る事故が複数回発生した（2026-06-28 / 07-05 / 07-26）。

North Starに従い、**2026-07-26に実際に起きた形**
（オッズ欠損と0.1倍台が同一レース内で同居）を再現して検証する。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (MIN_VALID_WIN_ODDS, _sanitize_win_odds,
                                 _INVALID_ODDS_WARNED)
from src.scraper.jra_scraper import apply_odds_to_races


@pytest.fixture(autouse=True)
def _clear_warn_cache():
    _INVALID_ODDS_WARNED.clear()
    yield
    _INVALID_ODDS_WARNED.clear()


class TestSanitizeWinOdds:

    def test_sub_one_odds_nulled(self):
        horses = [
            {'num': 1, 'win_odds': 0.1},
            {'num': 2, 'win_odds': 0.2},
            {'num': 3, 'win_odds': 3.5},
        ]
        n = _sanitize_win_odds(horses, 'R1')
        assert n == 2
        assert horses[0]['win_odds'] is None
        assert horses[1]['win_odds'] is None
        assert horses[2]['win_odds'] == 3.5, "正常なオッズは触らない"

    def test_exactly_one_is_valid(self):
        """1.0倍（元返し）は正当な値なので残す。"""
        horses = [{'num': 1, 'win_odds': 1.0}]
        assert _sanitize_win_odds(horses, 'R1') == 0
        assert horses[0]['win_odds'] == 1.0

    def test_none_and_zero_untouched(self):
        """オッズ未取得(None/0)は元々「不明」扱いなので変更しない。"""
        horses = [{'num': 1, 'win_odds': None}, {'num': 2, 'win_odds': 0}]
        assert _sanitize_win_odds(horses, 'R1') == 0
        assert horses[0]['win_odds'] is None
        assert horses[1]['win_odds'] == 0

    def test_no_warning_spam_for_same_race(self, capsys):
        horses = [{'num': 1, 'win_odds': 0.1}]
        _sanitize_win_odds(horses, 'R1')
        out1 = capsys.readouterr().out
        horses2 = [{'num': 1, 'win_odds': 0.1}]
        _sanitize_win_odds(horses2, 'R1')
        out2 = capsys.readouterr().out
        assert 'オッズ異常' in out1
        assert out2 == '', "同一レース・同一頭数の警告は1回だけ"


class TestPopularityNotCorruptedByBadOdds:
    """人気順位が壊れないこと — このバグの本丸。"""

    def _race(self):
        # 2026-07-26の実データ形状を模す:
        # オッズ欠損の馬と0.1倍台の馬が同一レースに同居している
        return {
            'id': '20260726_06_11',
            'date': '2026-07-26', 'racecourse': '新潟', 'distance': 1600,
            'surface': '芝', 'race_class': '3勝クラス', 'track_condition': '良',
            'horses': [
                {'num': 1, 'horse_num': 1, 'name': 'ゴミ値馬', 'win_odds': 0.1,
                 'running_style': '差し', 'history': []},
                {'num': 2, 'horse_num': 2, 'name': '真の1番人気', 'win_odds': 2.4,
                 'running_style': '先行', 'history': []},
                {'num': 3, 'horse_num': 3, 'name': '2番人気', 'win_odds': 5.0,
                 'running_style': '逃げ', 'history': []},
                {'num': 4, 'horse_num': 4, 'name': 'オッズ欠損馬', 'win_odds': None,
                 'running_style': '差し', 'history': []},
            ],
        }

    def test_bad_odds_horse_does_not_become_favorite(self):
        """0.1倍の馬が「1番人気」扱いされないこと（修正前は誤って1番人気になった）。"""
        import src.features.engine as eng
        for attr in ['_horse_dist_dict', '_horse_course_dict',
                     '_horse_venue_dist_dict', '_post_zone_bias',
                     '_jockey_dict', '_trainer_dict']:
            if not hasattr(eng, attr) or getattr(eng, attr) is None:
                setattr(eng, attr, {})
        eng._XGB_FUKUSHO_MODEL = None
        eng._XGB_RESIDUAL = False

        race = self._race()
        eng.calc_all(race)
        pops = {h['name']: h['popularity'] for h in race['horses']}

        assert pops['真の1番人気'] == 1, (
            f"実オッズ2.4倍の馬が1番人気になるべき: {pops}"
        )
        assert pops['2番人気'] == 2
        assert pops['ゴミ値馬'] > 2, (
            f"0.1倍のゴミ値馬が上位人気になってはいけない: {pops}"
        )

    def test_bad_odds_value_itself_is_cleared(self):
        """ゴミ値がそのまま表示・EV計算に流れないこと。"""
        import src.features.engine as eng
        eng._XGB_FUKUSHO_MODEL = None
        eng._XGB_RESIDUAL = False
        race = self._race()
        eng.calc_all(race)
        bad = next(h for h in race['horses'] if h['name'] == 'ゴミ値馬')
        assert bad['win_odds'] is None


class TestApplyOddsToRacesRejectsBadValues:
    """スクレイパー側（ゴミ値の入口）でも弾くこと。"""

    def test_sub_one_odds_not_applied(self):
        races = [{'id': 'R1', 'horses': [
            {'num': 1, 'win_odds': 0}, {'num': 2, 'win_odds': 0}]}]
        omap = {'R1': {1: {'tansho': 0.1}, 2: {'tansho': 4.5}}}
        updated = apply_odds_to_races(races, omap)
        hs = races[0]['horses']
        assert updated == 1, "正常な1頭だけが反映されるべき"
        assert hs[0]['win_odds'] == 0, "0.1倍は反映しない（既存値を保持）"
        assert hs[1]['win_odds'] == 4.5

    def test_valid_odds_still_applied(self):
        """正常データでは挙動が一切変わらないこと（回帰防止）。"""
        races = [{'id': 'R1', 'horses': [
            {'num': 1, 'win_odds': 0}, {'num': 2, 'win_odds': 0}]}]
        omap = {'R1': {1: {'tansho': 1.0}, 2: {'tansho': 128.0}}}
        updated = apply_odds_to_races(races, omap)
        assert updated == 2
        assert races[0]['horses'][0]['win_odds'] == 1.0
        assert races[0]['horses'][1]['win_odds'] == 128.0

    def test_min_valid_constant(self):
        assert MIN_VALID_WIN_ODDS == 1.0
