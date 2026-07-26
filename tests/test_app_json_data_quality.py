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


# ── 表示日付（same_day）─────────────────────────────────────────
# 2026-07-26、refresh_today()（当日実行）が生成したlatest.jsonの表示日付が
# 「7月27日(月)」という実在しない開催日になっていた不具合の回帰テスト。
# display_dtの+1日ロジックは「前夜に実行し翌日の予想を生成する」既存フロー
# （friday_predict.py / predict_next_day）専用の前提だったが、refresh_today()
# は当日に実行するため、day_type='sunday'だからと無条件に+1日すると
# 実在しない翌日（月曜）を表示してしまっていた。

def test_display_date_adds_one_day_for_night_before_generation():
    """従来フロー（前夜生成、same_day省略=False）は土曜夜にjst_now=土曜で
    day_type='sunday'を渡すと、表示日付は翌日（日曜）になる。"""
    races_all = _races(1)
    jst_now = datetime(2026, 7, 25, 20, 0, tzinfo=JST)  # 土曜夜
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday')
    assert result['date'] == '7月26日(日)'


def test_display_date_uses_jst_now_directly_when_same_day():
    """refresh_today()（当日実行、same_day=True）はjst_now自身の日付を
    そのまま表示する。day_type='sunday'でも+1日せず、実在しない
    「7月27日(月)」にならないことを確認する。"""
    races_all = _races(1)
    jst_now = datetime(2026, 7, 26, 9, 0, tzinfo=JST)  # 日曜朝（当日）
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], races_all, None, jst_now, day_type='sunday',
                             same_day=True)
    assert result['date'] == '7月26日(日)'
