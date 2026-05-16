import os
import sqlite3
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import init_db, save_bets_db, get_db_path


def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        conn = sqlite3.connect(get_db_path(tmp))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert 'races' in tables
        assert 'bets' in tables
        assert 'results' in tables
        assert 'bet_simulation' in tables


def test_save_bets_db_fukusho():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        bets = [{'type': '複勝', 'nums': [3], 'horse_name': 'テスト馬', 'odds_est': 2.5, 'amount': 500}]
        save_bets_db('20260510', '20260510_05_01', bets, base_dir=tmp)
        conn = sqlite3.connect(get_db_path(tmp))
        rows = conn.execute('SELECT * FROM bets').fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][3] == '複勝'  # bet_type
        assert rows[0][4] == 3       # horse_num


def test_save_bets_db_no_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        bets = [{'type': '複勝', 'nums': [3], 'horse_name': 'テスト馬', 'odds_est': 2.5, 'amount': 500}]
        save_bets_db('20260510', '20260510_05_01', bets, base_dir=tmp)
        save_bets_db('20260510', '20260510_05_01', bets, base_dir=tmp)  # 2回目は無視される
        conn = sqlite3.connect(get_db_path(tmp))
        rows = conn.execute('SELECT * FROM bets').fetchall()
        conn.close()
        assert len(rows) == 1  # 重複しない


if __name__ == '__main__':
    test_init_db_creates_tables()
    print('✅ test_init_db_creates_tables passed')
    test_save_bets_db_fukusho()
    print('✅ test_save_bets_db_fukusho passed')
    test_save_bets_db_no_duplicate()
    print('✅ test_save_bets_db_no_duplicate passed')
