"""過去レースの配当埋め戻しのテスト。

North Star #6 に従い、手打ちdictではなく **実際に `save_dividends_db()` が書いた行**を
sqlite3 から読み直して検証する。配当の形も `parse_dividends()` が返す形に合わせる
（単勝は dict、複勝・ワイドは list、順不同券種は昇順に正規化される）。
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backfill_dividends import _done_dates, _hist_dates, backfill
from src.utils.db import init_db, save_dividends_db


def _dividends():
    """parse_dividends が返すのと同じ形。"""
    return {
        'tansho': {'num': 7, 'payout': 480},
        'fukusho': [{'num': 7, 'payout': 180}, {'num': 3, 'payout': 210},
                    {'num': 11, 'payout': 330}],
        'umaren': {'nums': [7, 3], 'payout': 1450},
        'wide': [{'nums': [7, 3], 'payout': 620}],
        'sanrenpuku': {'nums': [7, 3, 11], 'payout': 5400},
        'sanrentan': {'nums': [7, 3, 11], 'payout': 24300},
    }


def _make_base(tmp_path, dates, done=()):
    """history.db（開催日あり）と keiba.db（一部だけ配当済み）を実際に作る。"""
    base = tmp_path / 'base'
    (base / 'data').mkdir(parents=True)
    hist = base / 'data' / 'history.db'
    conn = sqlite3.connect(hist)
    conn.execute('CREATE TABLE race_history (race_id TEXT, date TEXT)')
    for d in dates:
        conn.execute('INSERT INTO race_history VALUES (?, ?)', (f'{d}_05_01', d))
    conn.commit()
    conn.close()

    init_db(str(base))
    if done:
        save_dividends_db(
            [{'race_id': f'{d}_05_01', 'dividends': _dividends()} for d in done],
            base_dir=str(base))
    return base


def test_hist_dates_normalises_hyphenated_dates(tmp_path):
    """history.db の date は 'YYYY-MM-DD' の場合がある。YYYYMMDD に揃うこと。"""
    base = _make_base(tmp_path, ['20260704', '2026-07-05'])
    assert _hist_dates(str(base / 'data' / 'history.db')) == {'20260704', '20260705'}


def test_done_dates_reads_from_real_race_dividends(tmp_path):
    base = _make_base(tmp_path, ['20260704', '20260705'], done=['20260705'])
    assert _done_dates(str(base / 'data' / 'keiba.db')) == {'20260705'}


def test_done_dates_empty_when_db_missing(tmp_path):
    """keiba.db がまだ無い環境でも落ちないこと。"""
    assert _done_dates(str(tmp_path / 'nope.db')) == set()


def test_skips_already_saved_dates(tmp_path, monkeypatch):
    """配当済みの開催日は再取得しない（何度でも追加実行できるための冪等性）。"""
    base = _make_base(tmp_path, ['20260704', '20260705'], done=['20260705'])
    called = []

    def fake_fetch(sess, date):
        called.append(date)
        return [{'race_id': f'{date}_05_01', 'dividends': _dividends()}]

    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', fake_fetch)
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    r = backfill(str(base), budget=10)

    assert called == ['20260704']          # 済みの07-05は呼ばれない
    assert r['processed'] == 1
    assert r['remaining'] == 0


def test_budget_limits_days_and_leaves_rest(tmp_path, monkeypatch):
    """上限を超えて取得しない。残りは次回に回る（CIタイムアウト事故の再発防止）。"""
    base = _make_base(tmp_path, ['20260704', '20260705', '20260711', '20260712'])
    called = []

    def fake_fetch(sess, date):
        called.append(date)
        return [{'race_id': f'{date}_05_01', 'dividends': _dividends()}]

    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', fake_fetch)
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    r = backfill(str(base), budget=2)

    assert len(called) == 2
    assert called == ['20260712', '20260711']   # 既定は新しい順
    assert r['processed'] == 2 and r['remaining'] == 2


def test_oldest_first_reverses_order(tmp_path, monkeypatch):
    base = _make_base(tmp_path, ['20260704', '20260712'])
    called = []
    monkeypatch.setattr('scripts.backfill_dividends.fetch_results',
                        lambda s, d: called.append(d) or [])
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    backfill(str(base), budget=1, oldest_first=True)
    assert called == ['20260704']


def test_one_bad_day_does_not_stop_the_rest(tmp_path, monkeypatch):
    """1日が失敗しても残りは処理される（1日の失敗で全体を止めない）。"""
    base = _make_base(tmp_path, ['20260704', '20260705'])

    def fake_fetch(sess, date):
        if date == '20260705':
            raise RuntimeError('JRA側が落ちている想定')
        return [{'race_id': f'{date}_05_01', 'dividends': _dividends()}]

    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', fake_fetch)
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    r = backfill(str(base), budget=10)

    assert r['processed'] == 1
    assert _done_dates(str(base / 'data' / 'keiba.db')) == {'20260704'}


def test_zero_race_day_is_retried_next_time(tmp_path, monkeypatch):
    """0レースの日は保存しない＝次回また対象になる（取りこぼしを諦めない）。"""
    base = _make_base(tmp_path, ['20260704'])
    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', lambda s, d: [])
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    backfill(str(base), budget=10)
    assert _done_dates(str(base / 'data' / 'keiba.db')) == set()


def test_day_without_dividends_is_retried_next_time(tmp_path, monkeypatch):
    """レースは取れたが配当が空の日は済み扱いにしない（取り直す価値があるため）。"""
    base = _make_base(tmp_path, ['20260704'])
    monkeypatch.setattr(
        'scripts.backfill_dividends.fetch_results',
        lambda s, d: [{'race_id': f'{d}_05_01', 'dividends': {}}])
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    r = backfill(str(base), budget=10)
    assert r['processed'] == 0 and r['remaining'] == 1
    assert _done_dates(str(base / 'data' / 'keiba.db')) == set()


def test_saves_all_bet_types_with_real_round_trip(tmp_path, monkeypatch):
    """埋め戻し後、8券種が実際にDBから読み出せること（これが本来の目的）。"""
    base = _make_base(tmp_path, ['20260704'])
    monkeypatch.setattr(
        'scripts.backfill_dividends.fetch_results',
        lambda s, d: [{'race_id': f'{d}_05_01', 'dividends': _dividends()}])
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    backfill(str(base), budget=10)

    conn = sqlite3.connect(base / 'data' / 'keiba.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT bet_type, combo, payout FROM race_dividends '
        "WHERE race_id = '20260704_05_01'").fetchall()
    conn.close()
    got = {(r['bet_type'], r['combo']): r['payout'] for r in rows}

    # 組み合わせ券種の配当が実際に取り出せる = 任意の買い方を検証できる状態
    assert got[('umaren', '3-7')] == 1450        # 順不同なので昇順に正規化される
    assert got[('wide', '3-7')] == 620
    assert got[('sanrenpuku', '3-7-11')] == 5400
    assert got[('sanrentan', '7-3-11')] == 24300  # 着順券種は並べ替えない
    assert got[('tansho', '7')] == 480
    assert got[('fukusho', '11')] == 330


def test_time_limit_stops_cleanly_after_saving(tmp_path, monkeypatch):
    """時間上限に達したら、保存済みぶんを残したまま止まること。"""
    base = _make_base(tmp_path, ['20260704', '20260705', '20260711'])
    called = []
    times = iter([0.0] + [10_000.0] * 20)   # 1日目の保存直後に上限超過とみなす

    def fake_fetch(sess, date):
        called.append(date)
        return [{'race_id': f'{date}_05_01', 'dividends': _dividends()}]

    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', fake_fetch)
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    monkeypatch.setattr('scripts.backfill_dividends.time.time', lambda: next(times))
    r = backfill(str(base), budget=10, max_minutes=1)

    assert len(called) == 1
    assert r['processed'] == 1 and r['remaining'] == 2
    # 打ち切られても、取得できた日の配当は残っている
    assert _done_dates(str(base / 'data' / 'keiba.db')) == {'20260711'}


def test_dry_run_does_not_scrape(tmp_path, monkeypatch):
    base = _make_base(tmp_path, ['20260704'])

    def boom(*a, **k):
        raise AssertionError('--dry-run でスクレイピングしてはいけない')

    monkeypatch.setattr('scripts.backfill_dividends.fetch_results', boom)
    monkeypatch.setattr('scripts.backfill_dividends.create_session', boom)
    r = backfill(str(base), budget=10, dry_run=True)
    assert r['processed'] == 0 and r['remaining'] == 1


def test_never_writes_to_history_db(tmp_path, monkeypatch):
    """history.db には一切書き込まないこと（読み取り専用であることの担保）。"""
    base = _make_base(tmp_path, ['20260704'])
    hist = base / 'data' / 'history.db'
    before = hist.read_bytes()
    monkeypatch.setattr(
        'scripts.backfill_dividends.fetch_results',
        lambda s, d: [{'race_id': f'{d}_05_01', 'dividends': _dividends()}])
    monkeypatch.setattr('scripts.backfill_dividends.create_session', lambda: object())
    backfill(str(base), budget=10)
    assert hist.read_bytes() == before
