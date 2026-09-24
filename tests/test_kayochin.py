"""kayochinkeiba の指数記事パーサ。

North Star #6 に従い、手打ちHTMLではなく**実際の記事から切り出した本物の構造**
（`tests/fixtures/kayochin_*.html`）で検証する。このプロジェクトでは
「綺麗な自作HTMLだとテストは通り続けるが本番だけ壊れる」事故が複数回あった
（2026-07-22⑤ の Shift_JIS 未指定・2026-08-03② の結果ページ列崩れ）。

fixture は2本ある。**2023年版と2026年版で同じパーサが通ることが、過去分の
バックフィルが成立する前提**なので、両方を常に通す。
"""
import os
import unittest

from src.scraper.kayochin import (
    SOURCE_NAME, article_url, parse_article, parse_index_tables,
    parse_published_at, _circled_to_int,
)

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load(name):
    with open(os.path.join(FIX, name), encoding='utf-8') as f:
        return f.read()


class TestPublishedAt(unittest.TestCase):
    """🔴 9時間ずれると「予測時点で利用可能だったか」の答えが逆になる。

    実記事には同じ時刻が2つのオフセットで入っている（2026-09-24 に実測）:
      OG article:published_time  ...+00:00  ← RSS pubDate と一致。これが正
      JSON-LD datePublished      ...+09:00  ← 同じ時刻に違うオフセット＝バグ
    fixture には**両方**残してあるので、このテストは実際に両者を区別する。
    """

    def test_reads_og_not_json_ld(self):
        for name in ('kayochin_2026.html', 'kayochin_2023.html'):
            with self.subTest(name):
                html = _load(name)
                self.assertIn('"datePublished":', html, 'fixture に誤った方が残っていない')
                got = parse_published_at(html)
                self.assertTrue(got.endswith('+00:00'),
                                f'OG(UTC)ではなく JSON-LD を読んでいる: {got!r}')

    def test_returns_none_when_absent(self):
        self.assertIsNone(parse_published_at('<html><head></head></html>'))


class TestCircled(unittest.TestCase):
    def test_circled_numerals(self):
        self.assertEqual(_circled_to_int('①'), 1)
        self.assertEqual(_circled_to_int('⑧'), 8)
        self.assertEqual(_circled_to_int('⑱'), 18)

    def test_non_numeral(self):
        self.assertIsNone(_circled_to_int(''))
        self.assertIsNone(_circled_to_int('指'))


class TestParseBothEras(unittest.TestCase):
    """2023年版と2026年版で同じ結果の形になること（バックフィルの前提）。"""

    CASES = (
        ('kayochin_2026.html', '2026-09-20', 179),
        ('kayochin_2023.html', '2023-09-30', 180),
    )

    def test_twelve_races_no_rejects(self):
        for name, date, n_rows in self.CASES:
            with self.subTest(name):
                rows, meta = parse_article(_load(name), date, 'nakayama')
                self.assertEqual(meta['n_races'], 12)
                self.assertEqual(meta['rejected'], [])
                self.assertEqual(len(rows), n_rows)

    def test_race_id_matches_history_db_format(self):
        # history.db は f'{YYYYMMDD}_{場コード}_{R:02d}'。中山は '06'
        rows, _ = parse_article(_load('kayochin_2026.html'), '2026-09-20', 'nakayama')
        self.assertEqual(rows[0]['race_id'], '20260920_06_01')
        self.assertEqual(rows[-1]['race_id'], '20260920_06_12')

    def test_rank_is_descending_by_index(self):
        for name, date, _ in self.CASES:
            with self.subTest(name):
                rows, _ = parse_article(_load(name), date, 'nakayama')
                per = {}
                for r in rows:
                    per.setdefault(r['race_num'], []).append(r)
                for rn, rs in per.items():
                    vals = [r['index_value'] for r in rs]
                    self.assertEqual(vals, sorted(vals, reverse=True), f'{rn}R')
                    self.assertEqual([r['index_rank'] for r in rs],
                                     list(range(1, len(rs) + 1)), f'{rn}R')

    def test_horse_numbers_unique_and_in_range(self):
        for name, date, _ in self.CASES:
            with self.subTest(name):
                rows, _ = parse_article(_load(name), date, 'nakayama')
                per = {}
                for r in rows:
                    per.setdefault(r['race_num'], []).append(r['horse_num'])
                for rn, nums in per.items():
                    self.assertEqual(len(nums), len(set(nums)), f'{rn}R 馬番が重複')
                    self.assertTrue(all(1 <= n <= 18 for n in nums), f'{rn}R 馬番が範囲外')

    def test_source_and_published_on_every_row(self):
        rows, meta = parse_article(_load('kayochin_2026.html'), '2026-09-20', 'nakayama')
        self.assertTrue(all(r['source'] == SOURCE_NAME for r in rows))
        self.assertTrue(all(r['published_at'] == meta['published_at'] for r in rows))
        self.assertIsNotNone(meta['published_at'])


class TestCrossCheckRejects(unittest.TestCase):
    """一覧と各レース再掲が食い違うレースは採用しない（North Star #8 の検算）。"""

    def test_mutated_repeat_is_rejected(self):
        html = _load('kayochin_2026.html')
        races, rejected = parse_index_tables(html)
        self.assertEqual(rejected, [])
        # 12R の再掲テーブル（最後の3行テーブル）の指数を1つだけ書き換える
        last = html.rfind('<table')
        head, tail = html[:last], html[last:]
        self.assertIn('>42<', tail, '12R 再掲に想定の値がある')
        broken = head + tail.replace('>42<', '>41<', 1)
        races2, rejected2 = parse_index_tables(broken)
        self.assertEqual([rn for rn, _ in rejected2], [12])
        self.assertNotIn(12, races2)
        self.assertEqual(len(races2), 11, '他のレースは巻き込まれない')


class TestUrl(unittest.TestCase):
    def test_article_url(self):
        self.assertEqual(article_url('2026-09-20', 'nakayama'),
                         'https://kayochinkeiba.com/260920nakayama/')

    def test_rejects_non_jra_venue(self):
        with self.assertRaises(ValueError):
            parse_article(_load('kayochin_2026.html'), '2026-09-20', 'monbetsu')


if __name__ == '__main__':
    unittest.main()
