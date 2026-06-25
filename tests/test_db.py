import os
import sqlite3
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (init_db, save_bets_db, get_db_path,
                          save_odds_snapshots, get_latest_odds_snapshot_time,
                          save_race_predictions,
                          save_race_notes, get_latest_note_time,
                          calc_handicap_from_notes, recalc_all_handicaps)


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
        assert 'odds_snapshots' in tables
        assert 'race_notes' in tables


# テスト用の最小スキーマ（feature と weight の挙動を確認できる構成）
_NOTE_SCHEMA = {
    'version': 1,
    'categories': [
        {'id': 'start', 'feature': True, 'weight': 1.0},
        {'id': 'blocked', 'feature': True, 'weight': 1.0},
        {'id': 'condition', 'feature': True, 'weight': 1.0},
        {'id': 'memo_only', 'feature': False, 'weight': 1.0},
    ],
}


def test_calc_handicap_from_notes():
    # feature=true の項目だけ value×weight で合算、feature=false は無視
    notes = {'start': 2, 'blocked': 3, 'condition': -1, 'memo_only': 5}
    assert calc_handicap_from_notes(notes, _NOTE_SCHEMA) == 4.0
    # スキーマに無いキー・欠損キーは 0 扱い
    assert calc_handicap_from_notes({}, _NOTE_SCHEMA) == 0.0
    assert calc_handicap_from_notes({'unknown': 9}, _NOTE_SCHEMA) == 0.0


def test_save_race_notes_and_features():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        assert get_latest_note_time(get_db_path(tmp)) == ''

        rows = [
            {'date': '2026-06-20', 'race_id': '20260620_05_11', 'horse_num': 3,
             'horse_name': 'テスト馬', 'notes_data': '{"start": 2, "blocked": 3}',
             'free_memo': '出遅れ大', 'captured_at': '2026-06-20 18:00:00'},
        ]
        assert save_race_notes(rows, base_dir=tmp, schema=_NOTE_SCHEMA) == 1
        assert get_latest_note_time(get_db_path(tmp)) == '2026-06-20 18:00:00'

        conn = sqlite3.connect(get_db_path(tmp))
        hcap = conn.execute(
            "SELECT total_handicap FROM race_notes WHERE horse_num=3").fetchone()[0]
        conn.close()
        assert hcap == 5.0  # 2 + 3

        # 同じ馬の再入力は上書き（重複行を作らない）
        rows2 = [
            {'date': '2026-06-20', 'race_id': '20260620_05_11', 'horse_num': 3,
             'horse_name': 'テスト馬', 'notes_data': {'start': 1},
             'free_memo': '修正', 'captured_at': '2026-06-20 18:30:00'},
        ]
        save_race_notes(rows2, base_dir=tmp, schema=_NOTE_SCHEMA)
        conn = sqlite3.connect(get_db_path(tmp))
        cnt = conn.execute(
            "SELECT COUNT(*) FROM race_notes WHERE horse_num=3").fetchone()[0]
        hcap2 = conn.execute(
            "SELECT total_handicap FROM race_notes WHERE horse_num=3").fetchone()[0]
        conn.close()
        assert cnt == 1
        assert hcap2 == 1.0


def test_recalc_all_handicaps():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        rows = [
            {'date': '2026-06-20', 'race_id': 'r1', 'horse_num': 1,
             'notes_data': {'start': 2, 'blocked': 1}, 'captured_at': '2026-06-20 18:00:00'},
        ]
        save_race_notes(rows, base_dir=tmp, schema=_NOTE_SCHEMA)
        # weight を 2倍にしたスキーマで再計算 → total_handicap が追従する
        heavy = {'version': 2, 'categories': [
            {'id': 'start', 'feature': True, 'weight': 2.0},
            {'id': 'blocked', 'feature': True, 'weight': 2.0},
        ]}
        import json as _json
        schema_path = os.path.join(tmp, 'data', 'note_schema.json')
        with open(schema_path, 'w') as f:
            _json.dump(heavy, f)
        n = recalc_all_handicaps(base_dir=tmp, schema_path=schema_path)
        assert n == 1
        conn = sqlite3.connect(get_db_path(tmp))
        hcap = conn.execute("SELECT total_handicap FROM race_notes").fetchone()[0]
        conn.close()
        assert hcap == 6.0  # (2+1) × 2


def test_save_odds_snapshots_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        assert get_latest_odds_snapshot_time(get_db_path(tmp)) == ''
        rows = [
            {'race_id': '20260620_05_11', 'horse_num': 3, 'tansho': 2.8,
             'fukusho': 1.5, 'captured_at': '2026-06-20 14:30:00'},
            {'race_id': '20260620_05_11', 'horse_num': 5, 'tansho': 6.0,
             'fukusho': 2.1, 'captured_at': '2026-06-20 14:30:00'},
        ]
        assert save_odds_snapshots(rows, base_dir=tmp) == 2
        # 同一 (race_id, horse_num, captured_at) は重複取込されない
        assert save_odds_snapshots(rows, base_dir=tmp) == 0
        # 別時刻のスナップショットは別行として追加される
        rows2 = [{'race_id': '20260620_05_11', 'horse_num': 3, 'tansho': 2.5,
                  'fukusho': 1.4, 'captured_at': '2026-06-20 14:45:00'}]
        assert save_odds_snapshots(rows2, base_dir=tmp) == 1
        assert get_latest_odds_snapshot_time(get_db_path(tmp)) == '2026-06-20 14:45:00'


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


def test_save_race_predictions_stores_real_probs():
    # 旧バグ: cal_prob/fuku_prob が常に0で保存され乖離学習が機能しなかった。
    # calc_all 出力の cal_prob と top3_prob が実値で保存されることを確認する。
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        race = {'id': '20260101_05_11', 'date': '2026-01-01',
                'racecourse': '東京', 'race_num': 11}
        out = [
            {'horse_num': 1, 'name': 'A', 'win_prob': 0.45, 'cal_prob': 0.62,
             'top3_prob': 0.88, 'rl_rank': 1, 'popularity': 1, 'win_odds': 3.0},
            {'horse_num': 2, 'name': 'B', 'win_prob': 0.30, 'cal_prob': 0.40,
             'top3_prob': 0.71, 'rl_rank': 2, 'popularity': 2, 'win_odds': 6.0},
        ]
        save_race_predictions(race, out, base_dir=tmp)
        conn = sqlite3.connect(get_db_path(tmp))
        rows = conn.execute(
            'SELECT cal_prob, fuku_prob FROM race_predictions ORDER BY rl_rank'
        ).fetchall()
        conn.close()
        assert rows[0] == (0.62, 0.88)   # cal_prob と top3_prob が実値
        assert all(c > 0 and f > 0 for c, f in rows)  # 0埋めバグの回帰防止


def test_save_race_predictions_fuku_pct_fallback():
    # 旧経路（fuku_pct が 0-100 で来る）は /100 で 0-1 に吸収される
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'data'))
        init_db(base_dir=tmp)
        race = {'id': '20260101_05_12', 'date': '2026-01-01',
                'racecourse': '東京', 'race_num': 12}
        out = [{'horse_num': 1, 'name': 'A', 'win_prob': 0.4, 'cal_prob': 0.5,
                'fuku_pct': 85.0, 'rl_rank': 1, 'popularity': 1, 'win_odds': 3.0}]
        save_race_predictions(race, out, base_dir=tmp)
        conn = sqlite3.connect(get_db_path(tmp))
        v = conn.execute('SELECT fuku_prob FROM race_predictions').fetchone()[0]
        conn.close()
        assert abs(v - 0.85) < 1e-6


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
