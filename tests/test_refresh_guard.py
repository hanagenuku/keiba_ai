"""当日refreshを増やしたことで生まれる「確定オッズ混入」を防ぐガードのテスト。

2026-08-16に refresh を 08:00 の1回から 08:00 / 11:30 / 14:00 の3回に増やした。
午後のrunは**すでに終わったレース**のオッズページも読めてしまう。そこには
確定オッズが載っているため、そのまま race_predictions に保存すると
「結果が出た後の市場情報」が予想スナップショットに混入する
（2026-07-06 の shadow_bets と同じ型のリーク）。

そこで weekend.refresh_today は
  ・発走時刻を過ぎたレース
  ・発走時刻が読めなかったレース（安全側）
を更新しない。ここではその挙動を固定する。
"""
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.parser import parse_header  # noqa: E402


# ── 発走時刻のパース ─────────────────────────────────────────────────────
class TestParseStartTime:
    def test_extracts_start_time(self):
        # 2026-08-16の実HTML調査でログに出た見出し文の形
        t = ('2026年7月26日（日曜） 2回新潟2日 発走時刻： 18時00分 '
             '3歳以上1勝クラス 1,000 メートル（芝・左）')
        assert parse_header(t)['start_time'] == '18:00'

    def test_zero_pads(self):
        t = '2026年8月16日（日曜） 札幌 発走時刻：9時50分 未勝利 1700メートル（ダ・右）'
        assert parse_header(t)['start_time'] == '09:50'

    def test_absent_when_not_written(self):
        t = '2026年8月16日（日曜） 札幌 未勝利 1700メートル（ダ・右）'
        assert parse_header(t).get('start_time') is None

    def test_ignores_impossible_time(self):
        t = '2026年8月16日 札幌 発走時刻：99時99分 未勝利 1700メートル（ダ・右）'
        assert parse_header(t).get('start_time') is None


# ── 発走済みレースを更新しないこと ────────────────────────────────────────
def _race(rid, start_time, num=1):
    r = {'id': rid, 'race_num': num, 'racecourse': '札幌', 'surface': 'ダート',
         'distance': 1700, 'horses': [], 'num_horses': 0}
    if start_time is not None:
        r['start_time'] = start_time
    return r


@pytest.fixture
def wk(monkeypatch, tmp_path):
    import scripts.weekend as wk
    monkeypatch.setattr(wk, 'ROOT', str(tmp_path))
    monkeypatch.setattr(wk, 'APP_PATH', str(tmp_path / 'data' / 'latest.json'))
    return wk


def _run(wk, monkeypatch, races, now, saved, app_written):
    monkeypatch.setattr(wk, 'fetch_races_on_date', lambda *a, **k: (races, []))
    monkeypatch.setattr(wk, 'fetch_odds_map', lambda *a, **k: {})
    monkeypatch.setattr(wk, 'apply_odds_to_races', lambda *a, **k: 0)
    monkeypatch.setattr(wk, 'calc_all', lambda race, bias: [{'num': 1}])
    monkeypatch.setattr(wk, 'select_quality_races', lambda *a, **k: [])
    monkeypatch.setattr(wk, 'build_market_odds_from_races', lambda *a, **k: {})

    def _save(race, scored, base_dir=None, snapshot='initial', **kw):
        saved.append((race['id'], snapshot))
    monkeypatch.setattr(wk, 'save_race_predictions', _save)

    def _app(selected, races_all, bias, jst_now, **kw):
        app_written.append([r['id'] for r in races_all])
        return {'races': {'札幌': [{'race_id': r['id'], 'v': 'new'}
                                   for r in races_all]}}
    monkeypatch.setattr(wk, 'to_app_json', _app)
    wk.refresh_today(None, 'h.db', {}, now)


class TestRefreshSkipsStartedRaces:
    def test_saves_only_upcoming(self, wk, monkeypatch):
        saved, app = [], []
        races = [_race('R1', '09:50', 1), _race('R2', '14:30', 9),
                 _race('R3', '16:10', 11)]
        _run(wk, monkeypatch, races, datetime(2026, 8, 15, 14, 0), saved, app)
        assert [s[0] for s in saved] == ['R2', 'R3'], (
            '発走済みのR1を更新している（確定オッズが混入する）')

    def test_unknown_start_time_is_not_updated(self, wk, monkeypatch):
        # 発走時刻が読めない＝発走済みか判定できない。安全側に倒して触らない。
        saved, app = [], []
        races = [_race('R1', None, 1), _race('R2', '16:10', 11)]
        _run(wk, monkeypatch, races, datetime(2026, 8, 15, 14, 0), saved, app)
        assert [s[0] for s in saved] == ['R2']

    def test_morning_run_updates_everything(self, wk, monkeypatch):
        saved, app = [], []
        races = [_race('R1', '09:50', 1), _race('R2', '16:10', 11)]
        _run(wk, monkeypatch, races, datetime(2026, 8, 15, 8, 0), saved, app)
        assert [s[0] for s in saved] == ['R1', 'R2']

    def test_snapshot_label_records_the_hour(self, wk, monkeypatch):
        """08時台は従来通り 'refresh'。以降は時刻つきで別枠に残す。

        どの時点の予想が一番当たるかを compare_prediction_snapshots で
        後から測れるようにするため（増やしたrefreshが本当に効いたのかを
        「理屈」ではなく実測で判定できる状態にしておく）。
        """
        for hour, want in [(8, 'refresh'), (11, 'refresh_11'), (14, 'refresh_14')]:
            saved, app = [], []
            _run(wk, monkeypatch, [_race('R2', '23:59', 11)],
                 datetime(2026, 8, 15, hour, 0), saved, app)
            assert saved and saved[0][1] == want, f'{hour}時: {saved}'


class TestStartedRacesKeepTheirDisplay:
    def test_finished_race_display_is_preserved(self, wk, monkeypatch, tmp_path):
        """終わったレースの買い目を後から書き換えない。

        latest.json のgit履歴は「発走前に実際に何を推奨したか」の記録なので、
        午後のrunが過去のレースを上書きすると記録として使えなくなる。
        """
        d = tmp_path / 'data'
        d.mkdir()
        (d / 'latest.json').write_text(json.dumps(
            {'races': {'札幌': [{'race_id': 'R1', 'v': 'morning'},
                                {'race_id': 'R2', 'v': 'morning'}]}},
            ensure_ascii=False), encoding='utf-8')

        saved, app = [], []
        races = [_race('R1', '09:50', 1), _race('R2', '16:10', 11)]
        _run(wk, monkeypatch, races, datetime(2026, 8, 15, 14, 0), saved, app)

        got = json.loads((d / 'latest.json').read_text(encoding='utf-8'))
        by_id = {r['race_id']: r for r in got['races']['札幌']}
        assert by_id['R1']['v'] == 'morning', '発走済みR1が上書きされている'
        assert by_id['R2']['v'] == 'new', '発走前R2が更新されていない'

    def test_missing_old_file_does_not_crash(self, wk, monkeypatch):
        saved, app = [], []
        races = [_race('R1', '09:50', 1)]
        _run(wk, monkeypatch, races, datetime(2026, 8, 15, 14, 0), saved, app)
        # 例外を出さずに完走すればよい（新しい方をそのまま書く）
        assert saved == []
