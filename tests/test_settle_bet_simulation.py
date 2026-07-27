"""bet_simulation の決済処理のテスト（2026-07-27⑤導入）。

log_bet_simulation は全券種を「買った想定」で記録するが、決済処理が
どこにも存在せず2,160行が is_hit=-1 のまま放置されていた。
North Starに従い、実際に log_bet_simulation で書き込んだ行を
実際の sqlite3 から読み出して決済する形で検証する
（手打ちINSERTではなく本番と同じ書き込み経路を使う）。
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (init_db, settle_bet_simulation,
                          _parse_sim_horse_nums)
from src.betting.make_bets import log_bet_simulation


def _make_candidate(race_id, nums, names=None):
    """log_bet_simulation が受け取る候補辞書を本番と同じ形で作る。"""
    names = names or [f'H{n}' for n in nums]
    scored = [
        {'num': n, 'name': nm, 'win_odds': 2.0 + i * 3.0, 'pn': 0.30 - i * 0.05}
        for i, (n, nm) in enumerate(zip(nums, names))
    ]
    return {
        'race': {'id': race_id, 'racecourse': '東京', 'race_num': 11,
                 'num_horses': len(nums)},
        'scored': scored,
        'chaos_score': 0.4,
        'score_gap': 1.2,
    }


def _make_result(race_id, order, dividends):
    """fetch_results が返す形のレース結果を作る。"""
    return {
        'race_id': race_id,
        'racecourse': '東京',
        'race_num': 11,
        'finishers': [{'num': n} for n in order],
        'dividends': dividends,
    }


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / 'keiba.db')
    init_db(db_path=path)
    return path


class TestParseSimHorseNums:
    """horse_num 文字列のパース（本番のlog_bet_simulationが書く形式）。"""

    def test_single(self):
        assert _parse_sim_horse_nums('単勝', '3') == [3]

    def test_pair(self):
        assert _parse_sim_horse_nums('馬連', '3-8') == [3, 8]

    def test_umatan_arrow_preserves_order(self):
        # 馬単は '->' 区切りで順序に意味がある
        assert _parse_sim_horse_nums('馬単', '8->3') == [8, 3]

    def test_trio(self):
        assert _parse_sim_horse_nums('三連複', '3-8-10') == [3, 8, 10]

    def test_empty_and_garbage(self):
        assert _parse_sim_horse_nums('単勝', '') == []
        assert _parse_sim_horse_nums('単勝', None) == []
        assert _parse_sim_horse_nums('単勝', 'abc') == []


class TestSettleBetSimulation:

    def _log(self, db, race_id, nums):
        log_bet_simulation('2026-07-27', _make_candidate(race_id, nums),
                           db_path=db)

    def _unsettled(self, db):
        conn = sqlite3.connect(db)
        n = conn.execute(
            'SELECT COUNT(*) FROM bet_simulation WHERE is_hit=-1').fetchone()[0]
        conn.close()
        return n

    def _fetch(self, db, bet_type):
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            'SELECT is_hit, payout FROM bet_simulation WHERE bet_type=?',
            (bet_type,)).fetchone()
        conn.close()
        return r

    def test_all_bet_types_get_settled(self, db):
        """未決済(-1)が残らないこと。決済処理の存在自体の回帰テスト。"""
        self._log(db, 'R1', [3, 8, 10])
        assert self._unsettled(db) > 0, "前提: 記録直後は未決済"

        result = _make_result('R1', [3, 8, 10, 5], {
            'tansho': {'num': 3, 'payout': 450},
            'fukusho': [{'num': 3, 'payout': 180},
                        {'num': 8, 'payout': 240},
                        {'num': 10, 'payout': 310}],
            'wide': [{'nums': [3, 8], 'payout': 620}],
            'umaren': {'nums': [3, 8], 'payout': 1450},
            'umatan': {'nums': [3, 8], 'payout': 2900},
            'sanrenpuku': {'nums': [3, 8, 10], 'payout': 8800},
        })
        out = settle_bet_simulation([result], db_path=db)
        assert out['settled'] == 6, f"6券種すべて決済されるべき: {out}"
        assert self._unsettled(db) == 0, "未決済が残っている"

    def test_actual_payouts_used_not_estimates(self, db):
        """payoutが推定オッズではなく実配当で入ること。

        推定オッズで決済すると「推定が甘い券種ほど成績が良く見える」
        バイアスが入るため、実配当を使うことを固定する。
        """
        self._log(db, 'R1', [3, 8, 10])
        result = _make_result('R1', [3, 8, 10], {
            'tansho': {'num': 3, 'payout': 450},
            'fukusho': [{'num': 3, 'payout': 180}],
            'umaren': {'nums': [3, 8], 'payout': 1450},
        })
        settle_bet_simulation([result], db_path=db)
        assert self._fetch(db, '単勝')['payout'] == 450
        assert self._fetch(db, '複勝')['payout'] == 180
        assert self._fetch(db, '馬連')['payout'] == 1450

    def test_umatan_requires_correct_order(self, db):
        """馬単は着順が逆なら不的中（馬連との差を明確にする）。

        log_bet_simulation は scored[0]->scored[1] の順で記録するため、
        実際の着順が逆（8着->3着ではなく 8が1着, 3が2着）の場合、
        馬連は的中・馬単は不的中になるのが正しい。
        """
        self._log(db, 'R1', [3, 8, 10])   # 3->8 で記録される
        result = _make_result('R1', [8, 3, 10], {   # 実際は 8が1着, 3が2着
            'umaren': {'nums': [3, 8], 'payout': 1450},
            'umatan': {'nums': [8, 3], 'payout': 3200},
        })
        settle_bet_simulation([result], db_path=db)
        assert self._fetch(db, '馬連')['is_hit'] == 1, "馬連は順不同なので的中"
        assert self._fetch(db, '馬単')['is_hit'] == 0, (
            "馬単は着順が逆なので不的中でなければならない"
        )
        assert self._fetch(db, '馬単')['payout'] == 0

    def test_miss_recorded_as_zero_not_left_unsettled(self, db):
        """外れは is_hit=0 で確定させる（-1のまま残さない）。"""
        self._log(db, 'R1', [3, 8, 10])
        result = _make_result('R1', [5, 7, 9], {
            'tansho': {'num': 5, 'payout': 900},
        })
        settle_bet_simulation([result], db_path=db)
        assert self._unsettled(db) == 0
        assert self._fetch(db, '単勝')['is_hit'] == 0
        assert self._fetch(db, '単勝')['payout'] == 0

    def test_missing_dividend_marks_hit_without_payout(self, db):
        """配当が取得できない場合、的中扱いでもpayout=0に留める。

        分母には入るが分子に入らないため、回収率を過大評価しない。
        """
        self._log(db, 'R1', [3, 8, 10])
        result = _make_result('R1', [3, 8, 10], {})   # dividends 空
        settle_bet_simulation([result], db_path=db)
        r = self._fetch(db, '単勝')
        assert r['is_hit'] == 1, "着順的には的中"
        assert r['payout'] == 0, "配当不明なのでpayoutは0"

    def test_race_without_result_left_unsettled(self, db):
        """結果が無いレースは触らない（次回以降に決済される）。"""
        self._log(db, 'R1', [3, 8, 10])
        out = settle_bet_simulation([], db_path=db)
        assert out['settled'] == 0
        assert self._unsettled(db) == 6

    def test_idempotent_second_run_settles_nothing(self, db):
        """決済済みの行は再実行で二重計上されない（is_hit=-1のみ対象）。"""
        self._log(db, 'R1', [3, 8, 10])
        result = _make_result('R1', [3, 8, 10], {
            'tansho': {'num': 3, 'payout': 450}})
        first = settle_bet_simulation([result], db_path=db)
        second = settle_bet_simulation([result], db_path=db)
        assert first['settled'] == 6
        assert second['settled'] == 0, "2回目は決済対象なしであるべき"

    def test_by_type_summary(self, db):
        """券種別サマリが返ること（ワークフローのログ表示に使う）。"""
        self._log(db, 'R1', [3, 8, 10])
        result = _make_result('R1', [3, 8, 10], {
            'tansho': {'num': 3, 'payout': 450},
            'fukusho': [{'num': 3, 'payout': 180}],
        })
        out = settle_bet_simulation([result], db_path=db)
        assert out['by_type']['単勝']['n'] == 1
        assert out['by_type']['単勝']['hit'] == 1
        assert out['by_type']['単勝']['payout'] == 450


class TestCheckAndUpdateBetsUmatanOrder:
    """bets テーブル側の馬単的中判定（2026-07-27⑤修正）。

    `check_and_update_bets` は馬連・馬単を同じ条件
    `h1 in top3[:2] and h2 in top3[:2]` で判定しており、馬単で
    着順が逆でも的中扱いになっていた。この関数にはテストが1件も
    無かったため、修正と合わせて回帰テストを新設する。
    """

    def _insert_bet(self, db, bet_type, h1, h2, race_id='R1'):
        conn = sqlite3.connect(db)
        conn.execute(
            'INSERT INTO bets (date, race_id, bet_type, horse_num, horse_num2, '
            'amount, is_hit) VALUES (?,?,?,?,?,?,-1)',
            ('2026-07-27', race_id, bet_type, h1, h2, 100))
        conn.commit()
        conn.close()

    def _fetch(self, db, bet_type):
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        r = conn.execute('SELECT is_hit, payout FROM bets WHERE bet_type=?',
                         (bet_type,)).fetchone()
        conn.close()
        return r

    def test_umatan_wrong_order_is_miss(self, db):
        from src.utils.db import check_and_update_bets
        # 3->8 で買ったが、実際は 8が1着・3が2着
        self._insert_bet(db, '馬単', 3, 8)
        result = _make_result('R1', [8, 3, 10], {
            'umatan': {'nums': [8, 3], 'payout': 3200}})
        check_and_update_bets([result], db_path=db)
        r = self._fetch(db, '馬単')
        assert r['is_hit'] == 0, "馬単は着順が逆なら不的中でなければならない"
        assert r['payout'] == 0

    def test_umatan_correct_order_is_hit(self, db):
        from src.utils.db import check_and_update_bets
        self._insert_bet(db, '馬単', 8, 3)
        result = _make_result('R1', [8, 3, 10], {
            'umatan': {'nums': [8, 3], 'payout': 3200}})
        check_and_update_bets([result], db_path=db)
        r = self._fetch(db, '馬単')
        assert r['is_hit'] == 1
        assert r['payout'] == 3200

    def test_umaren_order_insensitive(self, db):
        """馬連は順不同なので着順が逆でも的中（修正で壊れていないこと）。"""
        from src.utils.db import check_and_update_bets
        self._insert_bet(db, '馬連', 3, 8)
        result = _make_result('R1', [8, 3, 10], {
            'umaren': {'nums': [3, 8], 'payout': 1450}})
        check_and_update_bets([result], db_path=db)
        r = self._fetch(db, '馬連')
        assert r['is_hit'] == 1, "馬連は順不同で的中するべき"
        assert r['payout'] == 1450
