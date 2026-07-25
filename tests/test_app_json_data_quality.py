"""to_app_json() の data_quality（オッズ取得率・パース失敗一覧）出力の回帰テスト。

2026-07-25、新潟R11のparse失敗・直前オッズ取得が全滅した日曜予想を調査した際、
どちらの問題もコンソールログにしか残らず、latest.json経由でアプリ側から
気づく手段が無かったことが判明した。to_app_json()にdata_qualityセクションを
追加し、オッズ反映率(odds_coverage)とfetch_races_on_date()が返す取得失敗
レース一覧(parse_failures)をアプリ用JSONに含めるようにした。

非選択レースの処理はcalc_all()の本物のXGB推論を必要とするため、ここでは
calc_all()を空リスト（＝スコア計算不可）にモックしてdata_quality計算自体を
分離してテストする。races_all側の'horses'件数はcalc_allの戻り値と無関係
（to_app_json内でrace辞書から直接lenを取るため）、モック後も総頭数の検証は
そのまま有効。
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.app_json import to_app_json

JST = timezone(timedelta(hours=9))


def _races(*horse_counts):
    return [
        {'racecourse': '新潟', 'race_num': i + 1, 'id': f'r{i + 1}', 'race_name': 'テストR',
         'distance': 1200, 'surface': 'ダート', 'horses': [{} for _ in range(n)]}
        for i, n in enumerate(horse_counts)
    ]


def test_data_quality_reports_odds_coverage_ratio():
    races_all = _races(3, 2)  # 計5頭
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday',
                             odds_updated_count=0)
    assert result['data_quality']['odds_coverage'] == 0.0


def test_data_quality_odds_coverage_partial_ratio():
    races_all = _races(4)  # 計4頭
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday',
                             odds_updated_count=2)
    assert result['data_quality']['odds_coverage'] == 0.5


def test_data_quality_odds_coverage_none_when_not_provided():
    """odds_updated_count省略時（既存呼び出し元の後方互換）はNoneのまま
    （0%取得と区別するため、意図的にNoneで「未計測」を表す）。
    """
    races_all = _races(3)
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday')
    assert result['data_quality']['odds_coverage'] is None


def test_data_quality_odds_coverage_none_when_no_horses():
    """全レース0頭（races_all自体が空等）で0除算しないこと。"""
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], [], None, jst_now, day_type='sunday',
                             odds_updated_count=0)
    assert result['data_quality']['odds_coverage'] is None


def test_data_quality_carries_parse_failures_through():
    races_all = _races(2)
    parse_failures = [{'racecourse': '新潟', 'race_num': 11, 'reason': 'parse失敗'}]
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday',
                             parse_failures=parse_failures)
    assert result['data_quality']['parse_failures'] == parse_failures


def test_data_quality_parse_failures_defaults_to_empty_list():
    races_all = _races(2)
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday')
    assert result['data_quality']['parse_failures'] == []
