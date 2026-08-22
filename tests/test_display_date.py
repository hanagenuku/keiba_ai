"""アプリに出す日付が「レース自身の日付」と一致することの回帰テスト。

2026-08-22、金曜予想のボタンが**土曜の未明(04:10 JST)**に押された結果、
中身は土曜(20260822)のレースなのに表示が「8月23日(日)」になった。

原因は表示日を `jst_now + 1日`（day_typeがsaturday/sundayのとき）で
推測していたこと。前夜に実行する前提に依存しており、日をまたいで実行すると外れる。
2026-07-26① の `same_day` 引数も同じ症状への対処だったが、呼び出し側が
正しい真偽値を渡すことに依存していた。日付はレース自身が持っているので、
そちらを一次情報にする。
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.app_json import to_app_json, _display_date_from_races

JST = timezone(timedelta(hours=9))


def _races(date_str='2026-08-22', n=2):
    """本番と同じ形（`date` は YYYY-MM-DD、`id` は YYYYMMDD_vv_rr）。"""
    d = date_str.replace('-', '')
    return [
        {'racecourse': '中京', 'race_num': i + 1, 'id': f'{d}_07_{i + 1:02d}',
         'race_id': f'{d}_07_{i + 1:02d}', 'date': date_str,
         'race_name': 'テストR', 'distance': 1200, 'surface': 'ダート',
         'horses': [{}, {}, {}]}
        for i in range(n)
    ]


def _run(races, jst_now, **kw):
    with patch('src.betting.app_json.calc_all', return_value=[]):
        return to_app_json([], races, None, jst_now, **kw)


def test_saturday_races_generated_after_midnight_show_saturday():
    """今回の事故の再現。土曜04:10に金曜予想を実行しても「8月22日(土)」。"""
    got = _run(_races('2026-08-22'),
               datetime(2026, 8, 22, 4, 10, tzinfo=JST), day_type='saturday')
    assert got['date'] == '8月22日(土)', got['date']


def test_saturday_races_generated_friday_night_show_saturday():
    """従来フロー（金曜夜に実行）でも同じ結果になること。"""
    got = _run(_races('2026-08-22'),
               datetime(2026, 8, 21, 21, 30, tzinfo=JST), day_type='saturday')
    assert got['date'] == '8月22日(土)', got['date']


def test_sunday_races_generated_saturday_night_show_sunday():
    got = _run(_races('2026-08-23'),
               datetime(2026, 8, 22, 19, 30, tzinfo=JST), day_type='sunday')
    assert got['date'] == '8月23日(日)', got['date']


def test_refresh_same_day_still_correct():
    """当日refresh（same_day=True）でも変わらないこと。"""
    got = _run(_races('2026-08-22'),
               datetime(2026, 8, 22, 8, 14, tzinfo=JST),
               day_type='saturday', same_day=True)
    assert got['date'] == '8月22日(土)', got['date']


def test_falls_back_to_jst_now_when_no_races():
    """レースが無ければ従来の推測にフォールバックする（後方互換）。"""
    got = _run([], datetime(2026, 8, 21, 21, 30, tzinfo=JST), day_type='saturday')
    assert got['date'] == '8月22日(土)', got['date']


def test_helper_reads_date_then_id():
    assert _display_date_from_races(_races('2026-08-23')).isoformat() == '2026-08-23'
    # date が壊れていても id から復元できる
    r = _races('2026-08-23')
    for x in r:
        x['date'] = ''
    assert _display_date_from_races(r).isoformat() == '2026-08-23'
    assert _display_date_from_races([]) is None
