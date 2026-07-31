"""全券種の実配当保存（race_dividends）のテスト（2026-07-31導入）。

parse_dividends は8券種すべてを取得しているのに、保存していたのは
単勝・複勝だけ（results テーブル）で、ワイド・馬連・馬単・三連複などの
組み合わせ配当はパースした直後に捨てられていた。
そのため単勝・複勝以外は回収率を一度も検証できない状態が続いていた。

North Star に従い、手打ちのdictではなく **実際に parse_dividends() が
返す形** を使って保存→読み出し→決済まで通す。
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (init_db, save_dividends_db, get_dividends_for_races,
                          settle_bet_simulation, _dividend_rows)
from src.scraper.jra_scraper import parse_dividends
from src.betting.make_bets import log_bet_simulation

try:
    from bs4 import BeautifulSoup
except ImportError:                     # pragma: no cover
    BeautifulSoup = None


# 実機の払戻欄に相当するテキスト（全券種）
_PAYOUT_HTML = """
<html><body><div>
払戻金
単勝 3 450円
複勝 3 180円 8 220円 10 310円
枠連 2-5 900円
馬連 3-8 1450円
馬単 3-8 3200円
ワイド 3-8 620円 3-10 780円 8-10 1100円
3連複 3-8-10 5400円
3連単 3-8-10 28900円
</div></body></html>
"""


@pytest.fixture
def db(tmp_path):
    p = str(tmp_path / 'keiba.db')
    init_db(db_path=p)
    return p


@pytest.fixture
def divs():
    if BeautifulSoup is None:
        pytest.skip('bs4 未導入')
    return parse_dividends(BeautifulSoup(_PAYOUT_HTML, 'html.parser'))


def _result(race_id, divs, order=(3, 8, 10, 5)):
    return {'race_id': race_id, 'date': '2026-07-25',
            'finishers': [{'num': n} for n in order],
            'dividends': divs}


class TestDividendRows:

    def test_all_bet_types_are_captured(self, divs):
        types = {bt for bt, _, _ in _dividend_rows(divs)}
        assert types == {'tansho', 'fukusho', 'wakuren', 'umaren',
                         'umatan', 'wide', 'sanrenpuku', 'sanrentan'}

    def test_multi_entry_types_keep_every_combo(self, divs):
        rows = _dividend_rows(divs)
        assert len([r for r in rows if r[0] == 'fukusho']) == 3
        assert len([r for r in rows if r[0] == 'wide']) == 3

    def test_unordered_types_are_normalised(self):
        """馬連・ワイド等は順不同なので昇順に正規化する（照合漏れ防止）。"""
        rows = _dividend_rows({'umaren': {'nums': [8, 3], 'payout': 1450}})
        assert rows == [('umaren', '3-8', 1450)]

    def test_ordered_types_keep_finishing_order(self):
        """馬単・三連単は着順に意味があるので並べ替えない。"""
        rows = _dividend_rows({'umatan': {'nums': [8, 3], 'payout': 3200}})
        assert rows == [('umatan', '8-3', 3200)]
        rows = _dividend_rows({'sanrentan': {'nums': [10, 3, 8], 'payout': 28900}})
        assert rows == [('sanrentan', '10-3-8', 28900)]


class TestSaveAndLoad:

    def test_saves_every_bet_type(self, db, divs):
        out = save_dividends_db([_result('R1', divs)], db_path=db)
        assert out['races'] == 1
        conn = sqlite3.connect(db)
        n = conn.execute('SELECT COUNT(*) FROM race_dividends').fetchone()[0]
        types = {r[0] for r in conn.execute(
            'SELECT DISTINCT bet_type FROM race_dividends')}
        conn.close()
        assert n == out['rows'] == 12      # 単1+複3+枠1+馬連1+馬単1+ワイド3+三複1+三単1
        assert 'wide' in types and 'umaren' in types and 'sanrenpuku' in types

    def test_idempotent(self, db, divs):
        save_dividends_db([_result('R1', divs)], db_path=db)
        second = save_dividends_db([_result('R1', divs)], db_path=db)
        conn = sqlite3.connect(db)
        n = conn.execute('SELECT COUNT(*) FROM race_dividends').fetchone()[0]
        conn.close()
        assert second['rows'] == 0 and n == 12

    def test_roundtrip_matches_parse_dividends(self, db, divs):
        """保存→読み出しで parse_dividends と同じ形に戻ること。

        settle_bet_simulation がそのまま食える形でなければ意味がない。
        """
        save_dividends_db([_result('R1', divs)], db_path=db)
        back = get_dividends_for_races(['R1'], db_path=db)['R1']
        assert back['umaren']['nums'] == [3, 8]
        assert back['umaren']['payout'] == 1450
        assert back['umatan']['nums'] == [3, 8]      # 元データが既に着順
        assert back['sanrenpuku']['payout'] == 5400
        assert {tuple(w['nums']): w['payout'] for w in back['wide']} == {
            (3, 8): 620, (3, 10): 780, (8, 10): 1100}
        assert len(back['fukusho']) == 3

    def test_ignores_results_without_dividends(self, db):
        out = save_dividends_db(
            [{'race_id': 'R9', 'finishers': [{'num': 1}], 'dividends': {}}],
            db_path=db)
        assert out == {'races': 0, 'rows': 0}

    def test_missing_race_ids_return_empty(self, db, divs):
        save_dividends_db([_result('R1', divs)], db_path=db)
        assert get_dividends_for_races(['NOPE'], db_path=db) == {}


class TestUnlocksExoticSettlement:
    """本機能の目的: 保存した配当で組み合わせ券種が実配当決済できること。"""

    def _log(self, db, race_id, nums):
        scored = [{'num': n, 'name': f'H{n}', 'win_odds': 2.0 + i * 3.0,
                   'pn': 0.30 - i * 0.05} for i, n in enumerate(nums)]
        log_bet_simulation('2026-07-25', {
            'race': {'id': race_id, 'racecourse': '東京', 'race_num': 11,
                     'num_horses': len(nums)},
            'scored': scored, 'chaos_score': 0.4, 'score_gap': 1.2,
        }, db_path=db)

    def test_exotics_settle_with_real_payouts(self, db, divs):
        """従来 payout=0 のままだったワイド・馬連・三連複に実配当が入る。"""
        self._log(db, 'R1', [3, 8, 10])
        save_dividends_db([_result('R1', divs)], db_path=db)
        stored = get_dividends_for_races(['R1'], db_path=db)['R1']
        settle_bet_simulation(
            [{'race_id': 'R1', 'finishers': [{'num': n} for n in (3, 8, 10, 5)],
              'dividends': stored, 'num_runners': 12}], db_path=db)

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        got = {r['bet_type']: (r['is_hit'], r['payout']) for r in conn.execute(
            'SELECT bet_type, is_hit, payout FROM bet_simulation')}
        conn.close()
        assert got['馬連'] == (1, 1450)
        assert got['ワイド'] == (1, 620)
        assert got['三連複'] == (1, 5400)
        assert got['単勝'] == (1, 450)
        assert got['複勝'] == (1, 180)
