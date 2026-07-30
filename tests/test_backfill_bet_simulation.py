"""蓄積済み未決済 bet_simulation のDB内結果による一括決済のテスト
（2026-07-28導入）。

settle_bet_simulation() は fetch_results() が返す「その回スクレイプした結果」
しか参照しないため、過去に溜まった未決済行は週次ワークフローを何度回しても
決済されないまま残る。backfill_bet_simulation() はDBに既にある結果から
決済情報を組み立てて未決済行だけを決済する。

North Starに従い、手打ちINSERTではなく実際に log_bet_simulation() /
save_history_db() で書き込んだ行を、実際の sqlite3 から読み出して検証する。
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (init_db, save_history_db, settle_bet_simulation,
                          build_results_from_db, backfill_bet_simulation)
from src.betting.make_bets import log_bet_simulation


def _make_candidate(race_id, nums):
    scored = [
        {'num': n, 'name': f'H{n}', 'win_odds': 2.0 + i * 3.0,
         'pn': 0.30 - i * 0.05}
        for i, n in enumerate(nums)
    ]
    return {
        'race': {'id': race_id, 'racecourse': '東京', 'race_num': 11,
                 'num_horses': len(nums)},
        'scored': scored,
        'chaos_score': 0.4,
        'score_gap': 1.2,
    }


def _save_result(hist_db, race_id, order, tansho=0, fukusho=None,
                 date='2026-07-25'):
    """本番と同じ save_history_db 経由で history.db に結果を書く。

    order   : 着順どおりの馬番リスト
    fukusho : {馬番: 配当} （複勝配当が付いた馬のみ）
    """
    fukusho = fukusho or {}
    finishers = []
    for i, num in enumerate(order, 1):
        finishers.append({
            'place': i,
            'num': num,
            'name': f'H{num}',
            'popularity': i,
            'tansho_payout': tansho if i == 1 else 0,
            'fukusho_payout': fukusho.get(num, 0),
        })
    save_history_db([{
        'id': race_id,
        'race_id': race_id,
        'date': date,
        'racecourse': '東京',
        'race_num': 11,
        'race_name': 'テストS',
        'distance': 1600,
        'surface': '芝',
        'finishers': finishers,
    }], db_path=hist_db)


@pytest.fixture
def dbs(tmp_path):
    main = str(tmp_path / 'keiba.db')
    hist = str(tmp_path / 'history.db')
    init_db(db_path=main)
    return main, hist


def _unsettled(db):
    conn = sqlite3.connect(db)
    n = conn.execute(
        'SELECT COUNT(*) FROM bet_simulation WHERE is_hit=-1').fetchone()[0]
    conn.close()
    return n


def _row(db, bet_type):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    r = conn.execute(
        'SELECT is_hit, payout FROM bet_simulation WHERE bet_type=?',
        (bet_type,)).fetchone()
    conn.close()
    return r


class TestBuildResultsFromDb:

    def test_reconstructs_finishing_order_and_dividends(self, dbs):
        main, hist = dbs
        _save_result(hist, 'R1', [8, 3, 10, 5], tansho=450,
                     fukusho={8: 180, 3: 220, 10: 310})

        out = build_results_from_db(db_path=main, hist_db_path=hist)
        assert len(out) == 1
        r = out[0]
        assert r['race_id'] == 'R1'
        # 着順どおりに並ぶこと（settle 側が order[:3] で上位を取るため必須）
        assert [h['num'] for h in r['finishers']] == [8, 3, 10, 5]
        assert r['num_runners'] == 4
        assert r['dividends']['tansho']['payout'] == 450
        assert {f['num']: f['payout'] for f in r['dividends']['fukusho']} == {
            8: 180, 3: 220, 10: 310}

    def test_omits_combination_dividends(self, dbs):
        """ワイド・馬連等の組み合わせ配当はDBに存在しないので載らない。"""
        main, hist = dbs
        _save_result(hist, 'R1', [8, 3, 10], tansho=450, fukusho={8: 180})
        divs = build_results_from_db(db_path=main, hist_db_path=hist)[0]['dividends']
        for key in ('wide', 'umaren', 'umatan', 'sanrenpuku'):
            assert key not in divs

    def test_race_ids_filter(self, dbs):
        main, hist = dbs
        _save_result(hist, 'R1', [8, 3, 10], tansho=450)
        _save_result(hist, 'R2', [1, 2, 3], tansho=300)
        out = build_results_from_db(race_ids={'R2'}, db_path=main,
                                    hist_db_path=hist)
        assert [r['race_id'] for r in out] == ['R2']


class TestBackfillBetSimulation:

    def _log(self, db, race_id, nums):
        log_bet_simulation('2026-07-25', _make_candidate(race_id, nums),
                           db_path=db)

    def test_settles_backlog_that_live_path_would_never_touch(self, dbs):
        """本機能の目的そのもの。

        settle_bet_simulation に「今回の結果」を渡しても、過去レースの行は
        race_id が一致しないため決済されない。その状態から backfill が
        DB内の結果を使って決済できることを確認する。
        """
        main, hist = dbs
        self._log(main, 'OLD_RACE', [3, 8, 10])
        _save_result(hist, 'OLD_RACE', [3, 8, 10, 5], tansho=450,
                     fukusho={3: 180, 8: 220, 10: 310})
        assert _unsettled(main) == 6

        # 今週スクレイプした別レースの結果を渡しても過去分は決済されない
        settle_bet_simulation(
            [{'race_id': 'NEW_RACE', 'finishers': [{'num': 1}],
              'dividends': {}}], db_path=main)
        assert _unsettled(main) == 6, '前提: liveパスでは過去分が決済されない'

        out = backfill_bet_simulation(db_path=main, hist_db_path=hist)
        assert _unsettled(main) == 0
        assert out['settled'] == 6
        assert out['races_covered'] == 1

    def test_real_payouts_used_for_tansho_and_fukusho(self, dbs):
        main, hist = dbs
        self._log(main, 'R1', [3, 8, 10])
        # 複勝が発売される頭数（8頭以上）で組む
        _save_result(hist, 'R1', [3, 8, 10, 5, 1, 2, 4, 6], tansho=450,
                     fukusho={3: 180, 8: 220, 10: 310})
        backfill_bet_simulation(db_path=main, hist_db_path=hist)
        assert _row(main, '単勝')['payout'] == 450
        assert _row(main, '複勝')['payout'] == 180

    def test_combination_hit_recorded_without_payout(self, dbs):
        """組み合わせ配当が無くても的中判定はできること。

        回収率は算出できないが的中率は測れる、という本機能の
        中心的な制約を固定する。
        """
        main, hist = dbs
        self._log(main, 'R1', [3, 8, 10])   # 馬連 3-8 が記録される
        _save_result(hist, 'R1', [3, 8, 10, 5], tansho=450,
                     fukusho={3: 180, 8: 220, 10: 310})
        out = backfill_bet_simulation(db_path=main, hist_db_path=hist)
        umaren = _row(main, '馬連')
        assert umaren['is_hit'] == 1, '1着3・2着8なので馬連は的中'
        assert umaren['payout'] == 0, '実配当が無いので払戻は0のまま'
        assert out['payout_unavailable'].get('馬連') == 1

    def test_idempotent(self, dbs):
        """再実行しても二重計上しないこと（is_hit=-1 のみが対象）。"""
        main, hist = dbs
        self._log(main, 'R1', [3, 8, 10])
        _save_result(hist, 'R1', [3, 8, 10, 5], tansho=450,
                     fukusho={3: 180})
        first = backfill_bet_simulation(db_path=main, hist_db_path=hist)
        second = backfill_bet_simulation(db_path=main, hist_db_path=hist)
        assert first['settled'] == 6
        assert second['settled'] == 0
        assert _row(main, '単勝')['payout'] == 450

    def test_race_without_result_left_unsettled(self, dbs):
        """結果がまだ無いレースは触らず未決済のまま残すこと。"""
        main, hist = dbs
        self._log(main, 'NO_RESULT', [3, 8, 10])
        _save_result(hist, 'OTHER', [1, 2, 3], tansho=300)
        out = backfill_bet_simulation(db_path=main, hist_db_path=hist)
        assert out['settled'] == 0
        assert out['races_missing'] == 1
        assert _unsettled(main) == 6

    def test_fukusho_zone_shrinks_in_small_field(self, dbs):
        """7頭立ては複勝が2着までしか配当対象にならない。

        3着馬を的中扱いにすると券種別評価が汚れるため、
        num_runners が渡された場合は厳密に判定する。
        """
        main, hist = dbs
        # scored[0] が複勝の対象。3着になる馬番を先頭にして記録する
        self._log(main, 'SMALL', [10, 3, 8])
        _save_result(hist, 'SMALL', [3, 8, 10, 5, 1, 2, 4], tansho=450,
                     fukusho={3: 180, 8: 220})   # 7頭立て → 2着までが配当
        backfill_bet_simulation(db_path=main, hist_db_path=hist)
        fuku = _row(main, '複勝')
        assert fuku['is_hit'] == 0, (
            '7頭立ての3着は複勝の配当対象外なので不的中でなければならない'
        )

    def test_fukusho_zone_is_top3_in_normal_field(self, dbs):
        """8頭以上なら従来どおり3着までが複勝の対象。"""
        main, hist = dbs
        self._log(main, 'BIG', [10, 3, 8])
        _save_result(hist, 'BIG', [3, 8, 10, 5, 1, 2, 4, 6], tansho=450,
                     fukusho={3: 180, 8: 220, 10: 310})
        backfill_bet_simulation(db_path=main, hist_db_path=hist)
        fuku = _row(main, '複勝')
        assert fuku['is_hit'] == 1
        assert fuku['payout'] == 310
