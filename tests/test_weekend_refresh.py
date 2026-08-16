"""scripts/weekend.py の refresh_today() 回帰テスト。

前日夜（金曜/土曜）に生成された予想は、その時点でまだ薄いオッズを土台に
しているため、残差学習モデルのbase_margin（popularity由来）が未成熟な
情報のまま推論される（2026-07-24⑤で記録済みの既知課題）。この課題への
即効対応として、当日・発走前にオッズを再取得して予想（latest.json）を
更新するrefresh_todayを追加した。

keiba.db側（bets/bet_simulation等の実績記録）は書き換えない設計のため、
save_race_db/save_bets_db/log_bet_simulationが呼ばれないことも確認する。
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scripts.weekend as weekend

JST = timezone(timedelta(hours=9))


def _race(rid='r1', start_time='10:10'):
    # start_time は parse_header が実際に返すキー（2026-08-16 追加）。
    # 本番の出馬表には必ず発走時刻が載るので、フィクスチャにも持たせる。
    return {'id': rid, 'racecourse': '東京', 'race_num': 1, 'race_name': 'テストR',
            'distance': 1600, 'surface': '芝', 'num_horses': 2,
            'start_time': start_time,
            'date': '2026-07-26', 'horses': [{'num': 1, 'name': 'テストウマ'}]}


def test_refresh_today_skips_on_non_weekend(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(weekend, 'fetch_races_on_date',
                        lambda *a, **k: calls.append('called') or ([], []))
    jst_now = datetime(2026, 7, 27, 8, 0, tzinfo=JST)  # 月曜
    weekend.refresh_today(object(), 'dummy_hist.db', None, jst_now)
    assert calls == []


def test_refresh_today_skips_when_no_races(monkeypatch, tmp_path):
    app_path = tmp_path / 'latest.json'
    monkeypatch.setattr(weekend, 'APP_PATH', str(app_path))
    monkeypatch.setattr(weekend, 'fetch_races_on_date', lambda *a, **k: ([], []))

    def _fail_odds(*a, **k):
        raise AssertionError('races=[]の場合はオッズ取得まで進んではいけない')
    monkeypatch.setattr(weekend, 'fetch_odds_map', _fail_odds)

    jst_now = datetime(2026, 7, 26, 8, 0, tzinfo=JST)  # 日曜
    weekend.refresh_today(object(), 'dummy_hist.db', None, jst_now)
    assert not app_path.exists()


def test_refresh_today_updates_latest_json_without_touching_bets_db(monkeypatch, tmp_path):
    app_path = tmp_path / 'latest.json'
    monkeypatch.setattr(weekend, 'APP_PATH', str(app_path))

    race = _race()
    monkeypatch.setattr(weekend, 'fetch_races_on_date',
                        lambda sess, target_date, hist_path: ([race], []))
    monkeypatch.setattr(weekend, 'fetch_odds_map', lambda sess, races: {'r1': {1: {'tansho': 3.0}}})
    monkeypatch.setattr(weekend, 'apply_odds_to_races', lambda races, mom: 1)
    monkeypatch.setattr(weekend, 'calc_all', lambda race, bias: [{'num': 1, 'name': 'テストウマ',
                                                                    'total': 5.0, 'pn': 0.5}])
    saved_predictions = []
    monkeypatch.setattr(
        weekend, 'save_race_predictions',
        lambda race, scored, root, snapshot='initial':
            saved_predictions.append((race['id'], snapshot)))
    monkeypatch.setattr(weekend, 'select_quality_races', lambda races, bias: [])
    monkeypatch.setattr(weekend, 'build_market_odds_from_races', lambda races: {})

    captured = {}

    def _fake_to_app_json(selected, races_all, bias_data, jst_now, day_type=None,
                          market_odds_map=None, odds_updated_count=None, parse_failures=None,
                          same_day=False):
        captured['day_type'] = day_type
        captured['odds_updated_count'] = odds_updated_count
        captured['parse_failures'] = parse_failures
        captured['same_day'] = same_day
        return {'ok': True, 'day_type': day_type}
    monkeypatch.setattr(weekend, 'to_app_json', _fake_to_app_json)

    for fn_name in ('save_race_db', 'save_bets_db', 'log_bet_simulation'):
        def _fail(*a, _name=fn_name, **k):
            raise AssertionError(f'refresh_todayは{_name}を呼んではいけない（実績記録を書き換えない設計）')
        monkeypatch.setattr(weekend, fn_name, _fail)

    jst_now = datetime(2026, 7, 26, 9, 0, tzinfo=JST)  # 日曜
    weekend.refresh_today(object(), 'dummy_hist.db', None, jst_now)

    assert saved_predictions == [('r1', 'refresh')], (
        'refreshモードは snapshot="refresh" で保存し、前夜の予想(initial)を'
        'prediction_snapshots 側に残すべき')
    assert captured['day_type'] == 'sunday'
    assert captured['odds_updated_count'] == 1
    assert captured['parse_failures'] == []
    # 当日実行のため表示日付を+1日せずjst_nowそのまま使うことを
    # to_app_json()に明示的に伝えている（実在しない翌日表示バグの回帰防止）
    assert captured['same_day'] is True
    assert app_path.exists()
    with open(app_path, encoding='utf-8') as f:
        assert json.load(f) == {'ok': True, 'day_type': 'sunday'}


def test_refresh_today_infers_saturday_day_type(monkeypatch, tmp_path):
    app_path = tmp_path / 'latest.json'
    monkeypatch.setattr(weekend, 'APP_PATH', str(app_path))

    race = _race()
    monkeypatch.setattr(weekend, 'fetch_races_on_date',
                        lambda sess, target_date, hist_path: ([race], []))
    monkeypatch.setattr(weekend, 'fetch_odds_map', lambda sess, races: {})
    monkeypatch.setattr(weekend, 'apply_odds_to_races', lambda races, mom: 0)
    monkeypatch.setattr(weekend, 'calc_all', lambda race, bias: [])
    monkeypatch.setattr(weekend, 'save_race_predictions',
                        lambda race, scored, root, snapshot='initial': None)
    monkeypatch.setattr(weekend, 'select_quality_races', lambda races, bias: [])
    monkeypatch.setattr(weekend, 'build_market_odds_from_races', lambda races: {})

    captured = {}

    def _fake_to_app_json(selected, races_all, bias_data, jst_now, day_type=None, **kw):
        captured['day_type'] = day_type
        return {}
    monkeypatch.setattr(weekend, 'to_app_json', _fake_to_app_json)

    jst_now = datetime(2026, 7, 25, 8, 0, tzinfo=JST)  # 土曜
    weekend.refresh_today(object(), 'dummy_hist.db', None, jst_now)
    assert captured['day_type'] == 'saturday'


def test_main_supports_refresh_mode_choice():
    """argparseのmode選択肢にrefreshが追加されていることを確認する。"""
    import inspect
    src = inspect.getsource(weekend.main)
    assert "'refresh'" in src
    assert 'refresh_today' in src
