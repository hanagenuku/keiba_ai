"""バックフィル収集スクリプト。

North Star #6 に従い、sitemap の fixture は**実物から切り出したもの**を使う。
わざと **JRA以外（地方競馬）と `/login/` を残してある**ので、
「JRAの会場だけを拾う」ことが本物の混ざり方に対して検証される。

⚠ ネットワークには出ない。`_get` を差し替えて fixture を返させる。
"""
import os
import sqlite3
import tempfile
import unittest

from scripts import collect_web_signals as C

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')


def _fx(name):
    with open(os.path.join(FIX, name), encoding='utf-8') as f:
        return f.read()


def _fake_get(urls_seen=None):
    """sitemap と記事を fixture から返す偽の `_get`。"""
    def get(sess, url, timeout=30):
        if urls_seen is not None:
            urls_seen.append(url)
        if url.endswith('/sitemap.xml'):
            return _fx('kayochin_sitemap_index.xml')
        for ym in ('202309', '202609'):
            if url.endswith(f'post.{ym}.xml'):
                return _fx(f'kayochin_sitemap_{ym}.xml')
        if '2309' in url or '2310' in url:
            return _fx('kayochin_2023.html')
        return _fx('kayochin_2026.html')
    return get


class TestYmd(unittest.TestCase):
    def test_yymmdd(self):
        self.assertEqual(C._ymd_from_yymmdd('230930'), '2023-09-30')
        self.assertEqual(C._ymd_from_yymmdd('260920'), '2026-09-20')


class TestListArticles(unittest.TestCase):
    def setUp(self):
        self._orig = C._get
        C._get = _fake_get()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        C._get = self._orig
        self.tmp.cleanup()

    def test_picks_only_jra_venues(self):
        got = C.list_articles(None, '202309', '202309',
                              os.path.join(self.tmp.name, 'sm'))
        venues = {v for _d, v, _u in got}
        self.assertTrue(venues, 'JRAの記事が1本も拾えていない')
        self.assertLessEqual(venues, set(C.PLACE_ENG))
        # fixture に混ざっている地方競馬・ログインページを拾っていないこと
        urls = ' '.join(u for _d, _v, u in got)
        for bad in ('obihiro', 'kouchi', 'saga', 'morioka', 'funabashi', '/login/'):
            self.assertNotIn(bad, urls, f'JRA以外を拾っている: {bad}')

    def test_filters_by_month_range(self):
        got = C.list_articles(None, '202309', '202309',
                              os.path.join(self.tmp.name, 'sm'))
        # fixture には 2023-10-01 の記事も入っているが範囲外なので落ちる
        self.assertTrue(all(d.startswith('2023-09') for d, _v, _u in got),
                        f'範囲外の日付が混ざっている: {sorted({d for d,_,_ in got})}')

    def test_month_sitemap_is_cached(self):
        seen = []
        C._get = _fake_get(seen)
        cache = os.path.join(self.tmp.name, 'sm')
        C.list_articles(None, '202309', '202309', cache)
        n_first = len(seen)
        C.list_articles(None, '202309', '202309', cache)
        # 2回目は月別sitemapを取り直さない（indexのみ）
        self.assertEqual(len(seen) - n_first, 1,
                         '月別sitemapがキャッシュされていない')


class TestCollect(unittest.TestCase):
    def setUp(self):
        self._orig_get, self._orig_sess = C._get, C._session
        C._get = _fake_get()
        C._session = lambda: None
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name
        os.makedirs(os.path.join(self.base, 'data', 'private'), exist_ok=True)

    def tearDown(self):
        C._get, C._session = self._orig_get, self._orig_sess
        self.tmp.cleanup()

    def _run(self, **kw):
        kw.setdefault('start_ym', '202309')
        kw.setdefault('end_ym', '202309')
        kw.setdefault('interval_sec', 0)
        return C.collect(self.base, **kw)

    def _conn(self):
        return sqlite3.connect(os.path.join(self.base, C.DB_REL))

    def test_writes_signals_and_log(self):
        res = self._run(budget=2)
        self.assertEqual(res['fetched'], 2)
        self.assertGreater(res['rows'], 0)
        c = self._conn()
        n = c.execute('SELECT COUNT(*) FROM web_signals').fetchone()[0]
        self.assertEqual(n, res['rows'])
        logs = c.execute('SELECT status, n_races FROM web_fetch_log').fetchall()
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(s == 'ok' and r == 12 for s, r in logs))

    def test_budget_stops_the_run(self):
        """North Star #4: 件数の上限で必ず止まる。"""
        res = self._run(budget=1)
        self.assertEqual(res['fetched'], 1)
        self.assertGreater(res['left'], 0)

    def test_resumes_and_does_not_refetch(self):
        """2026-08-25 D-2① の教訓: 同じ場所を踏み続けない。"""
        first = self._run(budget=1)
        second = self._run(budget=1)
        self.assertEqual(second['done'], 1, '取得済みが数えられていない')
        self.assertEqual(second['todo'], first['todo'] - 1)

    def test_dry_run_writes_nothing(self):
        res = self._run(dry_run=True)
        self.assertNotIn('fetched', res)
        self.assertFalse(os.path.exists(os.path.join(self.base, C.DB_REL)))

    def test_http_failure_is_logged_not_silent(self):
        """失敗を無音にしない。理由付きで log に残り、再試行の対象に残る。"""
        def boom(sess, url, timeout=30):
            if url.endswith('.xml'):
                return _fake_get()(sess, url, timeout)
            raise RuntimeError('boom')
        C._get = boom
        res = self._run(budget=2)
        self.assertEqual(res['failed'], 2)
        c = self._conn()
        rows = c.execute('SELECT status FROM web_fetch_log').fetchall()
        self.assertTrue(all(s[0].startswith('http_error') for s in rows), rows)
        self.assertEqual(c.execute('SELECT COUNT(*) FROM web_signals').fetchone()[0], 0)
        # 失敗は done に数えない＝次回やり直される
        C._get = _fake_get()
        again = self._run(budget=1)
        self.assertEqual(again['done'], 0)

    def test_db_is_under_data_private(self):
        """取得データを公開リポジトリに置かない（netkeiba と同じ扱い）。"""
        self.assertEqual(C.DB_REL, os.path.join('data', 'private', 'web_signals.db'))
        with open(os.path.join(os.path.dirname(__file__), '..', '.gitignore'),
                  encoding='utf-8') as f:
            ign = f.read()
        self.assertIn('data/private/*', ign)


if __name__ == '__main__':
    unittest.main()
