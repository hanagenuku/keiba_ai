"""予想の時点別記録（prediction_snapshots）のテスト — 2026-07-27⑩導入。

当日朝のrefreshは race_predictions を上書きするため、前夜の予想がDBから
完全に消える。実際に2026-07-26のオッズ障害を「DBだけ見て起きていない」と
誤判定する事故が起きた（latest.jsonのgit履歴を辿って初めて判明）。

North Starに従い、手打ちINSERTではなく実際に save_race_predictions() で
書き込んだ行を実際の sqlite3 から読み出して検証する。
"""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (init_db, save_race_predictions,
                          compare_prediction_snapshots)


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / 'keiba.db')
    init_db(db_path=path)
    return path


def _race(race_id='R1', date='2026-08-01'):
    return {'id': race_id, 'date': date, 'racecourse': '新潟',
            'race_num': 11, 'num_horses': 3}


def _scored(specs):
    """specs: [(馬番, 名前, 人気, オッズ, RL順位, 勝率), ...]"""
    return [{'horse_num': n, 'num': n, 'name': nm, 'popularity': p,
             'win_odds': o, 'rl_rank': rl, 'win_prob': wp,
             'cal_prob': 0.4, 'top3_prob': 0.5}
            for n, nm, p, o, rl, wp in specs]


class TestSnapshotCoexistence:

    def test_both_snapshots_survive(self, db):
        """前夜(initial)と朝(refresh)の両方が残ること — 本機能の目的。"""
        save_race_predictions(_race(), _scored([
            (1, 'A', 1, 2.0, 1, 0.30), (2, 'B', 2, 5.0, 2, 0.20)]),
            db_path=db, snapshot='initial')
        save_race_predictions(_race(), _scored([
            (1, 'A', 2, 6.0, 2, 0.18), (2, 'B', 1, 2.2, 1, 0.33)]),
            db_path=db, snapshot='refresh')

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT snapshot, horse_num, rl_rank, tansho_odds FROM prediction_snapshots '
            'ORDER BY snapshot, horse_num').fetchall()
        conn.close()
        got = [(r['snapshot'], r['horse_num'], r['rl_rank'], r['tansho_odds']) for r in rows]
        assert got == [
            ('initial', 1, 1, 2.0), ('initial', 2, 2, 5.0),
            ('refresh', 1, 2, 6.0), ('refresh', 2, 1, 2.2),
        ], f"両時点が併存すべき: {got}"

    def test_race_predictions_still_single_row(self, db):
        """race_predictions は従来通り1頭1行（既存集計を壊さない）。"""
        save_race_predictions(_race(), _scored([(1, 'A', 1, 2.0, 1, 0.30)]),
                              db_path=db, snapshot='initial')
        save_race_predictions(_race(), _scored([(1, 'A', 2, 6.0, 2, 0.18)]),
                              db_path=db, snapshot='refresh')
        conn = sqlite3.connect(db)
        n = conn.execute('SELECT COUNT(*) FROM race_predictions').fetchone()[0]
        odds = conn.execute('SELECT tansho_odds FROM race_predictions').fetchone()[0]
        conn.close()
        assert n == 1, "race_predictions は1頭1行を維持すべき"
        assert odds == 6.0, "race_predictions は最新(refresh)で上書きされるべき"

    def test_same_snapshot_rerun_overwrites(self, db):
        """同一snapshotの再実行（--force/リトライ）は上書きし増殖しない。"""
        for odds in (2.0, 2.5, 3.0):
            save_race_predictions(_race(), _scored([(1, 'A', 1, odds, 1, 0.30)]),
                                  db_path=db, snapshot='initial')
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT tansho_odds FROM prediction_snapshots WHERE snapshot='initial'").fetchall()
        conn.close()
        assert len(rows) == 1, f"同一snapshotは1行のはず: {rows}"
        assert rows[0][0] == 3.0

    def test_default_snapshot_is_initial(self, db):
        """snapshot省略時は 'initial'（既存の呼び出しを壊さない）。"""
        save_race_predictions(_race(), _scored([(1, 'A', 1, 2.0, 1, 0.30)]), db_path=db)
        conn = sqlite3.connect(db)
        s = conn.execute('SELECT snapshot FROM prediction_snapshots').fetchone()[0]
        conn.close()
        assert s == 'initial'


class TestComparePredictionSnapshots:

    def _setup_two_snapshots(self, db):
        # 前夜: #1が本命(RL1)。朝: オッズが動いて #2 が本命に
        save_race_predictions(_race(), _scored([
            (1, 'A', 1, 2.0, 1, 0.30),
            (2, 'B', 2, 5.0, 2, 0.20),
            (3, 'C', 3, 9.0, 3, 0.12)]), db_path=db, snapshot='initial')
        save_race_predictions(_race(), _scored([
            (1, 'A', 3, 8.0, 2, 0.19),
            (2, 'B', 1, 2.1, 1, 0.32),
            (3, 'C', 2, 4.5, 3, 0.13)]), db_path=db, snapshot='refresh')
        # 結果: #2 が1着（＝朝の判断が正しかったケース）
        conn = sqlite3.connect(db)
        conn.execute("UPDATE race_predictions SET actual_place=1 WHERE horse_num=2")
        conn.execute("UPDATE race_predictions SET actual_place=5 WHERE horse_num=1")
        conn.execute("UPDATE race_predictions SET actual_place=3 WHERE horse_num=3")
        conn.commit(); conn.close()

    def test_detects_favorite_change(self, db):
        self._setup_two_snapshots(db)
        out = compare_prediction_snapshots(db_path=db)
        assert out is not None
        assert out['n_races'] == 1
        assert out['n_horses'] == 3
        assert out['fav_changed'] == 1, "本命が#1→#2に変わったことを検出すべき"
        assert out['rank_changed'] == 2, "RL順位が動いた2頭を検出すべき"

    def test_scores_which_snapshot_was_right(self, db):
        """どちらの時点の本命が勝ったかを集計できること（本機能の狙い）。"""
        self._setup_two_snapshots(db)
        out = compare_prediction_snapshots(db_path=db)
        assert out['initial_rl1_win'] == 0, "前夜の本命#1は5着なので0"
        assert out['refresh_rl1_win'] == 1, "朝の本命#2が1着なので1"

    def test_detail_rows_contain_both_values(self, db):
        self._setup_two_snapshots(db)
        out = compare_prediction_snapshots(db_path=db)
        row = next(r for r in out['rows'] if r['horse_num'] == 1)
        assert row['odds'] == (2.0, 8.0)
        assert row['rl'] == (1, 2)
        assert row['actual_place'] == 5

    def test_returns_none_without_refresh(self, db):
        """refreshが無い（前夜のみ）なら比較対象なしでNone。"""
        save_race_predictions(_race(), _scored([(1, 'A', 1, 2.0, 1, 0.30)]),
                              db_path=db, snapshot='initial')
        assert compare_prediction_snapshots(db_path=db) is None
