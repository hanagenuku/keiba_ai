"""race_notes.date を 'YYYY-MM-DD' に揃える回帰テスト。

🔴 GAS はスプレッドシートのセルを Date オブジェクトとして返すため、
getNotesLog 経由の date は JavaScript の Date.toString() 形式
'Sun Sep 06 2026 00:00:00 GMT+0900 (Japan Standard Time)' で届いていた。
他のテーブル（bets / shadow_bets / displayed_bets）は全て 'YYYY-MM-DD' なので、
この列だけ日付で結合できず、本番で210行すべてが壊れていた。
2026-08-31 に shadow_bets.was_recommended が全683行0だった事故と同じ型。
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import (normalize_note_date, repair_note_dates,
                          save_race_notes, init_db)

JS = 'Sun Sep 06 2026 00:00:00 GMT+0900 (Japan Standard Time)'
SCHEMA = {'version': 3, 'categories': [
    {'id': 'paddock_score', 'weight': 1.0, 'feature': True, 'phase': 'pre'}]}


class TestNormalize:
    def test_js_date_string_is_parsed(self):
        assert normalize_note_date(JS) == '2026-09-06'

    def test_single_digit_day_is_zero_padded(self):
        assert normalize_note_date('Tue Jan 06 2026 00:00:00 GMT+0900') == '2026-01-06'

    def test_already_iso_is_untouched(self):
        assert normalize_note_date('2026-09-06') == '2026-09-06'

    def test_falls_back_to_race_id(self):
        assert normalize_note_date('', '20260906_01_11') == '2026-09-06'
        assert normalize_note_date('ゴミ', '20261231_05_03') == '2026-12-31'

    def test_js_form_wins_over_race_id(self):
        # 日付が両方あるときは date 側を信じる（race_id は最後の手段）
        assert normalize_note_date(JS, '20250101_01_01') == '2026-09-06'

    def test_unparseable_and_no_race_id_returns_input(self):
        assert normalize_note_date('???', '') == '???'


class TestSaveNormalizes:
    def test_save_race_notes_stores_iso_date(self, tmp_path):
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        save_race_notes([{
            'date': JS, 'race_id': '20260906_01_11', 'racecourse': '札幌',
            'race_num': 11, 'horse_num': 7, 'horse_name': 'テスト',
            'notes_data': json.dumps({'paddock_score': 8}), 'free_memo': '',
        }], db_path=db, schema=SCHEMA)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT date FROM race_notes').fetchone()
        assert row['date'] == '2026-09-06'

    def test_date_matches_other_tables_format(self, tmp_path):
        """他テーブルと同じ形式で入ること（結合できることの回帰テスト）"""
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        save_race_notes([{
            'date': JS, 'race_id': '20260906_01_11', 'horse_num': 7,
            'notes_data': '{"paddock_score": 8}',
        }], db_path=db, schema=SCHEMA)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO bets (date, race_id, bet_type, horse_num, amount) "
            "VALUES ('2026-09-06', '20260906_01_11', 'fukusho', 7, 500)")
        conn.commit()
        joined = conn.execute(
            "SELECT COUNT(*) FROM race_notes n JOIN bets b "
            "ON n.date = b.date AND n.race_id = b.race_id").fetchone()[0]
        assert joined == 1, 'date が揃っていないと結合が0件になる'


class TestRepairExisting:
    def test_repair_fixes_broken_rows(self, tmp_path):
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        conn = sqlite3.connect(db)
        for n in (1, 2, 3):
            conn.execute(
                "INSERT INTO race_notes (date, race_id, horse_num, notes_data) "
                "VALUES (?, ?, ?, ?)", (JS, '20260906_01_11', n, '{}'))
        conn.commit()
        conn.close()

        assert repair_note_dates(db) == 3
        conn = sqlite3.connect(db)
        dates = [r[0] for r in conn.execute('SELECT date FROM race_notes')]
        assert dates == ['2026-09-06'] * 3

    def test_repair_is_idempotent(self, tmp_path):
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO race_notes (date, race_id, horse_num, notes_data) "
                     "VALUES (?, '20260906_01_11', 7, '{}')", (JS,))
        conn.commit()
        conn.close()
        assert repair_note_dates(db) == 1
        assert repair_note_dates(db) == 0

    def test_repair_drops_broken_duplicate_of_good_row(self, tmp_path):
        """正しい日付の同一行が既にあるなら、壊れた側を捨てる（UNIQUE衝突）"""
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO race_notes (date, race_id, horse_num, notes_data) "
                     "VALUES ('2026-09-06', '20260906_01_11', 7, '{\"a\":1}')")
        conn.execute("INSERT INTO race_notes (date, race_id, horse_num, notes_data) "
                     "VALUES (?, '20260906_01_11', 7, '{\"b\":2}')", (JS,))
        conn.commit()
        conn.close()
        repair_note_dates(db)
        conn = sqlite3.connect(db)
        rows = conn.execute('SELECT date, notes_data FROM race_notes').fetchall()
        assert len(rows) == 1
        assert rows[0][0] == '2026-09-06'

    def test_init_db_repairs_automatically(self, tmp_path):
        db = str(tmp_path / 'k.db')
        init_db(db_path=db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO race_notes (date, race_id, horse_num, notes_data) "
                     "VALUES (?, '20260906_01_11', 7, '{}')", (JS,))
        conn.commit()
        conn.close()
        init_db(db_path=db)          # 2回目の初期化で自動修復される
        conn = sqlite3.connect(db)
        assert conn.execute('SELECT date FROM race_notes').fetchone()[0] == '2026-09-06'
