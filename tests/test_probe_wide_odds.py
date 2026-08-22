"""ワイドオッズ盤プローブの開催日探索が「前方」を向いていることを固定する。

2026-08-17 にこのプローブを2回走らせて2回とも1件も盤を見られなかった。
2回目（引数順の修正後）の原因は探索の向きで、`get_kaisai_on_date` が読む
出走表一覧は「今週これからの開催」しか載せないのに、プローブは直近9日を
**過去へ**遡っていた。月曜に走らせると必ず空振りする。
"""
import datetime as dt

import pytest

from scripts import probe_wide_odds as P


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(P.time, 'sleep', lambda *_: None)


def _fake_calendar(monkeypatch, kaisai):
    """kaisai: {date_str: {base: date_str}} を返す get_kaisai_on_date を差し込む。"""
    asked = []

    def fake(date_str, sess, calendar=None):
        asked.append(date_str)
        return kaisai.get(date_str, {})

    monkeypatch.setattr(P, 'get_kaisai_on_date', fake)
    return asked


def test_searches_forward_not_backward(monkeypatch):
    """月曜に走らせたとき、直前の土日ではなく次の土日を見つける。"""
    monday = dt.date(2026, 8, 17)
    kaisai = {
        # 直前の週末（過去）。この経路では本来見えないが、
        # 万一見えても採用してはいけない
        '20260815': {'pw01dde0102026 0301': '20260815'},
        '20260816': {'pw01dde0102026 0302': '20260816'},
        # 次の週末（これから）
        '20260822': {'pw01dde0102026 0401': '20260822'},
    }
    asked = _fake_calendar(monkeypatch, kaisai)

    found = P.find_kaisai_forward(sess=object(), today=monday)

    assert [d for d, _ in found] == ['20260822'], found
    # 過去日は一度も問い合わせていない
    assert all(d >= '20260817' for d in asked), asked
    # 昇順に前方へ進んでいる
    assert asked == sorted(asked)


def test_includes_today(monkeypatch):
    """当日開催（土曜に走らせた場合）も拾う。"""
    saturday = dt.date(2026, 8, 22)
    asked = _fake_calendar(monkeypatch, {'20260822': {'base_a': '20260822'}})

    found = P.find_kaisai_forward(sess=object(), today=saturday)

    assert found == [('20260822', 'base_a')]
    assert asked[0] == '20260822'


def test_returns_all_venues_of_the_day(monkeypatch):
    """同じ日に複数会場あれば全部返す（1会場でオッズが取れない時の予備）。"""
    monkeypatch.setattr(P, 'PROBE_DAYS_AHEAD', 3)
    _fake_calendar(monkeypatch, {
        '20260823': {'base_x': '20260823', 'base_y': '20260823'}})

    found = P.find_kaisai_forward(sess=object(), today=dt.date(2026, 8, 21),
                                  days_ahead=3)

    assert sorted(b for _, b in found) == ['base_x', 'base_y']
    assert all(d == '20260823' for d, _ in found)


def test_no_kaisai_returns_empty(monkeypatch):
    """開催が無ければ空を返す（例外にしない）。"""
    _fake_calendar(monkeypatch, {})
    assert P.find_kaisai_forward(sess=object(), today=dt.date(2026, 8, 17)) == []


def test_calendar_exception_does_not_abort_the_scan(monkeypatch):
    """1日ぶんの取得失敗で探索全体を止めない。"""
    def fake(date_str, sess, calendar=None):
        if date_str == '20260817':
            raise RuntimeError('boom')
        return {'base_z': date_str} if date_str == '20260822' else {}

    monkeypatch.setattr(P, 'get_kaisai_on_date', fake)
    found = P.find_kaisai_forward(sess=object(), today=dt.date(2026, 8, 17))
    assert found == [('20260822', 'base_z')]
