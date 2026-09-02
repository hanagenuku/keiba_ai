"""パドック気配（phase='pre'）が回顧の補正値に混ざらないことを固定する。

race_notes は notes_data を丸ごと差し替える設計なので、
「レース前の見立て」と「レース後の不利」を同じ合計に足すと
total_handicap が何も指さない数になる。North Star #6 に従い、
手打ち dict ではなく実際に save_race_notes で書き込んだ行を
sqlite3.Row として読み出して検証する。
"""
import json
import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import calc_handicap_from_notes, save_race_notes, init_db

SCHEMA = {
    'version': 2,
    'categories': [
        {'id': 'start', 'weight': 1.0, 'feature': True, 'phase': 'post'},
        {'id': 'blocked', 'weight': 1.0, 'feature': True, 'phase': 'post'},
        {'id': 'paddock_score', 'weight': 1.0, 'feature': True, 'phase': 'pre'},
        {'id': 'memo_only', 'weight': 1.0, 'feature': False, 'phase': 'post'},
    ],
}


class TestHandicapExcludesPaddock:
    def test_paddock_score_is_not_summed(self):
        # 出遅れ2 + 詰まり1 = 3。パドック +2 は足さない
        total = calc_handicap_from_notes(
            {'start': 2, 'blocked': 1, 'paddock_score': 8}, SCHEMA)
        assert total == 3.0

    def test_high_paddock_score_does_not_inflate_handicap(self):
        # 除外し損ねると 3 + 10 = 13 になり、不利が実際の4倍に見える
        total = calc_handicap_from_notes(
            {'start': 2, 'blocked': 1, 'paddock_score': 10}, SCHEMA)
        assert total == 3.0

    def test_paddock_only_is_zero_handicap(self):
        assert calc_handicap_from_notes({'paddock_score': 7}, SCHEMA) == 0.0

    def test_feature_false_still_excluded(self):
        assert calc_handicap_from_notes({'memo_only': 5}, SCHEMA) == 0.0

    def test_missing_phase_is_treated_as_post(self):
        # phase 未指定の古いスキーマは従来どおり合算する（後方互換）
        legacy = {'categories': [{'id': 'start', 'weight': 1.0, 'feature': True}]}
        assert calc_handicap_from_notes({'start': 2}, legacy) == 2.0


class TestProductionSchemaShape:
    def test_real_schema_splits_pre_and_post(self):
        root = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(root, 'data/note_schema.json'), encoding='utf-8') as f:
            schema = json.load(f)
        cats = schema['categories']
        assert all('phase' in c for c in cats), 'phase 未指定の項目があると両画面に出る'
        assert any(c['phase'] == 'pre' for c in cats)
        assert any(c['phase'] == 'post' for c in cats)

    def test_retired_key_is_not_reused(self):
        """旧 condition(1=良/0=普通/-1=不安) を 0〜10 の点数で再利用しないこと。

        再利用すると、保存済みの 1 が「良」なのか「1点＝最低評価」なのかを
        後から区別できなくなる（本番の race_notes に旧スケールの行が9行ある）。
        """
        root = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(root, 'data/note_schema.json'), encoding='utf-8') as f:
            schema = json.load(f)
        ids = [c['id'] for c in schema['categories']]
        assert 'condition' not in ids
        assert any(r['id'] == 'condition' for r in schema.get('retired', [])), \
            '旧キーの意味を retired に残しておかないと、過去データが読めなくなる'

    def test_real_schema_paddock_score_is_zero_to_ten(self):
        root = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(root, 'data/note_schema.json'), encoding='utf-8') as f:
            schema = json.load(f)
        cat = next(c for c in schema['categories'] if c['id'] == 'paddock_score')
        vals = [o['value'] for o in cat['options']]
        assert vals == list(range(11)), '0〜10の11段階'
        assert cat['phase'] == 'pre'

    def test_real_schema_paddock_score_does_not_inflate_handicap(self):
        root = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(root, 'data/note_schema.json'), encoding='utf-8') as f:
            schema = json.load(f)
        assert calc_handicap_from_notes({'paddock_score': 7}, schema) == 0.0


class TestRoundTrip:
    def test_saved_row_keeps_both_and_handicap_is_post_only(self, tmp_path):
        db = str(tmp_path / 'keiba.db')
        init_db(db_path=db)
        n = save_race_notes([{
            'date': '2026-09-05', 'race_id': '20260905_06_11',
            'racecourse': '中山', 'race_num': 11, 'horse_num': 7,
            'horse_name': 'テスト', 'free_memo': '',
            'notes_data': json.dumps({'start': 2, 'paddock_score': 9}),
        }], db_path=db, schema=SCHEMA)
        assert n == 1

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT notes_data, total_handicap FROM race_notes '
            'WHERE date=? AND race_id=? AND horse_num=?',
            ('2026-09-05', '20260905_06_11', 7)).fetchone()
        nd = json.loads(row['notes_data'])
        assert nd['paddock_score'] == 9, 'パドック点数が保存されている'
        assert nd['start'] == 2, '不利メモも同じ行に共存する'
        assert row['total_handicap'] == 2.0, 'total_handicap は回顧の不利だけ'
