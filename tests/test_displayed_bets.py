"""画面に出した買い目の記録・決済のテスト。

North Star #6 に従い、手打ちdictではなく **本番の to_app_json が実際に
吐く形**（data/latest.json と同じ構造）で検証する。
"""
import json
import os
import sqlite3
import tempfile
import unittest

from src.betting.displayed_bets import extract_tickets, rows_from_app_json
from src.utils.db import (init_db, save_displayed_bets, settle_displayed_bets,
                          save_dividends_db)


# 本番 latest.json（2026-08-30 新潟R3）から取った実物
REAL_GUMBEL_BETS = [
    {'tag': 'tan', 'label': '単勝（軸）', 'horse': '#5 エクスマキナ',
     'est': '2.3倍', 'amt': '¥100'},
    {'tag': 'wide', 'label': 'ワイド(3点)', 'axis': 5, 'horse': '#5 - #7・#2・#1',
     'mates': [{'n': 7, 'name': 'ハンデンブレイズ', 'odds': 4.9, 'prob': 0.224},
               {'n': 2, 'name': 'ニシノリリックス', 'odds': 9.3, 'prob': 0.1393},
               {'n': 1, 'name': 'ケイジャーダ', 'odds': 9.0, 'prob': 0.077}],
     'est': '4.9〜9.3倍', 'amt': '¥300'},
    {'tag': 'umaren', 'label': '馬連(3点)', 'axis': 5, 'horse': '#5 - #7・#2・#1',
     'mates': [{'n': 7, 'odds': 16.7, 'prob': 0.088},
               {'n': 2, 'odds': 26.1, 'prob': 0.05},
               {'n': 1, 'odds': 24.2, 'prob': 0.0263}],
     'est': '16.7〜26.1倍', 'amt': '¥300', 'syn_odds': 7.2},
    {'tag': 'sanfuku', 'label': '三連複(11点)', 'trio_type': 'formation',
     'legs': [[5], [2, 7], [1, 2, 3, 4, 6, 7, 8]],
     'nums': [1, 2, 3, 4, 5, 6, 7, 8],
     'combos': ['3-5-7', '2-3-5', '4-5-7', '2-4-5', '5-6-7', '2-5-6',
                '2-5-7', '1-5-7', '5-7-8', '2-5-8', '1-2-5'],
     'est': '¥1,270〜¥38,750', 'syn_odds': 3.3, 'amt': '¥1100'},
]

APP_JSON = {
    'date': '8月30日(日)',
    'races': {
        '新潟': [{'r': 3, 'race_id': '20260830_04_03', 'rec': True,
                  'gumbel_bets': REAL_GUMBEL_BETS}],
        '札幌': [{'r': 1, 'race_id': '20260830_01_01', 'rec': False,
                  'gumbel_bets': [REAL_GUMBEL_BETS[0]]}],
    },
}


class TestExtractTickets(unittest.TestCase):
    def test_splits_multi_point_rows_into_individual_tickets(self):
        t = extract_tickets(REAL_GUMBEL_BETS)
        self.assertEqual(len(t), 1 + 3 + 3 + 11)

    def test_amount_is_divided_per_ticket(self):
        """行の 'amt' は合計額。点あたりに割らないと投資を過大に数える。"""
        t = extract_tickets(REAL_GUMBEL_BETS)
        self.assertAlmostEqual(sum(a for _, _, a in t), 100 + 300 + 300 + 1100)
        san = [a for bt, _, a in t if bt == 'sanrenpuku']
        self.assertEqual(len(san), 11)
        self.assertAlmostEqual(san[0], 1100 / 11)

    def test_combo_is_normalised_ascending(self):
        t = extract_tickets(REAL_GUMBEL_BETS)
        for bt, combo, _ in t:
            nums = [int(x) for x in combo.split('-')]
            self.assertEqual(nums, sorted(nums), f'{bt} {combo} が昇順でない')

    def test_bet_type_names_match_race_dividends(self):
        """決済を素直なJOINにするため race_dividends と同じ表記にする。"""
        types = {bt for bt, _, _ in extract_tickets(REAL_GUMBEL_BETS)}
        self.assertEqual(types, {'tansho', 'wide', 'umaren', 'sanrenpuku'})

    def test_empty_and_malformed_rows_are_skipped(self):
        self.assertEqual(extract_tickets(None), [])
        self.assertEqual(extract_tickets([{'tag': 'wide', 'amt': '¥300'}]), [])
        self.assertEqual(extract_tickets([{'tag': 'unknown', 'amt': '¥100'}]), [])


class TestRowsFromAppJson(unittest.TestCase):
    def test_records_every_race_not_only_recommended(self):
        """bets / bet_simulation は推奨レースのみだった。ここは全レース。"""
        rows = rows_from_app_json(APP_JSON)
        self.assertEqual({r['race_id'] for r in rows},
                         {'20260830_04_03', '20260830_01_01'})
        self.assertTrue(any(r['is_recommended'] == 0 for r in rows))
        self.assertTrue(any(r['is_recommended'] == 1 for r in rows))

    def test_racecourse_comes_from_the_venue_key(self):
        rows = rows_from_app_json(APP_JSON)
        by_race = {r['race_id']: r['racecourse'] for r in rows}
        self.assertEqual(by_race['20260830_04_03'], '新潟')
        self.assertEqual(by_race['20260830_01_01'], '札幌')

    def test_date_is_derived_from_race_id(self):
        rows = rows_from_app_json(APP_JSON)
        self.assertTrue(all(r['date'] == '2026-08-30' for r in rows))


class TestSaveAndSettle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, 'data'), exist_ok=True)
        init_db(self.tmp)
        self.db = os.path.join(self.tmp, 'data', 'keiba.db')
        self.hist = os.path.join(self.tmp, 'data', 'history.db')
        # 着順: 5-7-2（軸#5が1着、ワイド5-7と5-2が的中、馬連5-7が的中、三連複2-5-7が的中）
        h = sqlite3.connect(self.hist)
        h.execute('CREATE TABLE horse_history (race_id TEXT, horse_num INT, place INT)')
        h.executemany('INSERT INTO horse_history VALUES (?,?,?)',
                      [('20260830_04_03', 5, 1), ('20260830_04_03', 7, 2),
                       ('20260830_04_03', 2, 3)])
        h.commit(); h.close()
        c = sqlite3.connect(self.db)
        c.executemany(
            'INSERT INTO race_dividends (race_id,bet_type,combo,payout) VALUES (?,?,?,?)',
            [('20260830_04_03', 'tansho', '5', 230),
             ('20260830_04_03', 'wide', '5-7', 250),
             ('20260830_04_03', 'wide', '2-5', 480),
             ('20260830_04_03', 'umaren', '5-7', 830),
             ('20260830_04_03', 'sanrenpuku', '2-5-7', 2140)])
        c.commit(); c.close()
        save_displayed_bets(rows_from_app_json(APP_JSON), self.tmp)

    def _rows(self, where=''):
        c = sqlite3.connect(self.db); c.row_factory = sqlite3.Row
        r = c.execute(f'SELECT * FROM displayed_bets {where}').fetchall()
        c.close(); return r

    def test_all_tickets_persisted(self):
        self.assertEqual(len(self._rows()), 18 + 1)

    def test_rerun_overwrites_instead_of_duplicating(self):
        """当日refreshやリトライで点数が二重に積み上がってはいけない。"""
        save_displayed_bets(rows_from_app_json(APP_JSON), self.tmp)
        self.assertEqual(len(self._rows()), 19)

    def test_snapshots_coexist(self):
        save_displayed_bets(rows_from_app_json(APP_JSON, snapshot='refresh'), self.tmp)
        self.assertEqual(len(self._rows()), 38)
        self.assertEqual(len(self._rows("WHERE snapshot='refresh'")), 19)

    def test_settlement_uses_real_dividends(self):
        r = settle_displayed_bets(self.tmp, hist_db_path=self.hist)
        self.assertEqual(r['settled'], 18)   # 札幌は着順が無いので持ち越し
        self.assertEqual(r['hit'], 1 + 2 + 1 + 1)
        got = {(x['bet_type'], x['combo']): x['payout']
               for x in self._rows('WHERE is_hit=1')}
        self.assertAlmostEqual(got[('tansho', '5')], 100 * 230 / 100)
        self.assertAlmostEqual(got[('wide', '5-7')], 100 * 250 / 100)
        self.assertAlmostEqual(got[('umaren', '5-7')], 100 * 830 / 100)
        self.assertAlmostEqual(got[('sanrenpuku', '2-5-7')], (1100 / 11) * 2140 / 100)

    def test_settlement_is_idempotent(self):
        a = settle_displayed_bets(self.tmp, hist_db_path=self.hist)
        b = settle_displayed_bets(self.tmp, hist_db_path=self.hist)
        self.assertEqual(b['settled'], 0)
        self.assertGreater(a['settled'], 0)

    def test_races_without_results_are_left_pending(self):
        settle_displayed_bets(self.tmp, hist_db_path=self.hist)
        pend = self._rows("WHERE is_hit=-1")
        self.assertEqual({r['race_id'] for r in pend}, {'20260830_01_01'})

    def test_hit_without_dividend_is_counted_as_invested_not_recovered(self):
        """配当が引けないのに payout を推定で埋めると回収率が嘘になる。"""
        c = sqlite3.connect(self.db)
        c.execute("DELETE FROM race_dividends WHERE bet_type='tansho'")
        c.commit(); c.close()
        r = settle_displayed_bets(self.tmp, hist_db_path=self.hist)
        self.assertEqual(r['no_payout'], 1)
        row = self._rows("WHERE bet_type='tansho'")[0]
        self.assertEqual(row['is_hit'], 1)
        self.assertEqual(row['payout'], 0)


class TestRealLatestJson(unittest.TestCase):
    """リポジトリの実 latest.json をそのまま通す（形が変わったら落ちる）。"""

    def test_production_latest_json_parses(self):
        p = os.path.join(os.path.dirname(__file__), '..', 'data', 'latest.json')
        if not os.path.exists(p):
            self.skipTest('latest.json が無い')
        with open(p, encoding='utf-8') as f:
            rows = rows_from_app_json(json.load(f))
        if not rows:
            self.skipTest('この世代には gumbel_bets が無い')
        self.assertTrue(all(r['bet_type'] in
                            ('tansho', 'fukusho', 'wide', 'umaren', 'sanrenpuku')
                            for r in rows))
        self.assertTrue(all(r['amount'] > 0 for r in rows))
        self.assertTrue(all(r['racecourse'] for r in rows))


if __name__ == '__main__':
    unittest.main()
