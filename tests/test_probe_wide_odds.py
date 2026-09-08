"""ワイドオッズ盤プローブの開催日探索が「前方」を向いていることを固定する。

2026-08-17 にこのプローブを2回走らせて2回とも1件も盤を見られなかった。
2回目（引数順の修正後）の原因は探索の向きで、`get_kaisai_on_date` が読む
出走表一覧は「今週これからの開催」しか載せないのに、プローブは直近9日を
**過去へ**遡っていた。月曜に走らせると必ず空振りする。
"""
import datetime as dt

import pytest

from scripts import probe_wide_odds as P
from src.scraper.jra_scraper import _to_odds_base


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


def test_uses_production_odds_base_builder():
    """オッズ用CNAMEの組み立てを本番と別に書かない。

    2026-08-23 の自動実行では、前方探索は成功したのに find_r01_odds が
    256件すべてパラメータエラーになった。同時刻の本番refreshは同じレースの
    オッズを100%取得しており、盤は存在していた。原因は組み立ての取り違え:

        本番 _to_odds_base() : pw151ouS3 0420260302
        プローブの自前置換    : pw151ous  010420260302   ← 別物

    同じ導出を2箇所に書くとこうなる（2026-08-09③の「対になっている処理」と同型）。
    """
    src = open(P.__file__, encoding='utf-8').read()
    body = src.split('"""', 2)[-1]          # ヘッダのdocstringは説明のため除外
    assert "replace('pw01dde'" not in body, '自前の文字列置換が残っている'
    assert '_to_odds_base(' in body


def test_production_odds_base_shape():
    """本番の変換が期待どおりの形であること（プローブが依存する前提の固定）。"""
    assert _to_odds_base('pw01dde010420260302') == 'pw151ouS30420260302'


# ── 券種リンクの棚卸し（2026-09-08 に A〜Z 総当たりから方針変更）──────────
# JRADB は href ではなく onclick の doAction('/JRADB/accessX.html','CNAME')
# に本体を埋める。実機のこの形を模したフィクスチャで検証する（North Star #6）。
_ODDS_PAGE_HTML = """
<html><head><title>単勝・複勝オッズ(馬番順) JRA</title></head><body>
<a href="/JRADB/accessS.html">出馬表</a>
<a href="javascript:void(0);"
   onclick="return doAction('/JRADB/accessO.html','pw151ouB01620260906Z/15');">馬連</a>
<a href="javascript:void(0);"
   onclick="return doAction('/JRADB/accessO.html','pw151ouD01620260906Z/15');">ワイド</a>
<a href="javascript:void(0);"
   onclick="return doAction('/JRADB/accessO.html','pw151ouF01620260906Z/15');">3連複</a>
<a href="#">ログイン</a>
</body></html>
"""


def test_extract_links_reads_cname_from_onclick():
    """href が javascript:void(0) でも onclick から CNAME と endpoint を復元する。"""
    links = P.extract_links(_ODDS_PAGE_HTML)
    wide = [l for l in links if l['text'] == 'ワイド']
    assert len(wide) == 1, links
    assert wide[0]['cname'] == 'pw151ouD01620260906Z/15', wide[0]
    assert wide[0]['endpoint'] == 'accessO.html', wide[0]


def test_extract_links_keeps_links_without_cname():
    """CNAME が取れないリンクも件数として残す。
    「探した結果ゼロ」と「探せていない」を混同しないための担保。"""
    links = P.extract_links(_ODDS_PAGE_HTML)
    assert len(links) == 5, links
    assert sum(1 for l in links if not l['cname']) == 2, links


def test_bet_words_cover_the_target_bet_types():
    for w in ['ワイド', '馬連', '3連複']:
        assert w in P.BET_WORDS


def test_probe_no_longer_brute_forces_a_to_z():
    """A〜Z 総当たりは 2026-09-06 の実行で決着済み。復活していないこと。"""
    body = open('scripts/probe_wide_odds.py', encoding='utf-8').read()
    assert "for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'" not in body
    assert 'extract_links(' in body
