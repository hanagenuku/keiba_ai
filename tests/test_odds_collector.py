"""オッズ時系列収集の回帰テスト。

設計書（オッズ時系列モデル）に**書かれていなかったが、後から復元できない**
6点を固定する。7ヶ月かけて集めた後に「取り損ねていた」を防ぐのが目的。

  ① 発走時刻を保存すること（無いと「発走何分前か」が出せない）
  ② 複勝の範囲(min/max)を潰さないこと
  ③ 取消・除外を記録すること
  ④ スナップショットの完全性を記録すること
  ⑤ minutes_to_post を必ず持たせること（リーク対策の切り口）
  ⑥ 発走直前まで取ること（最終オッズの唯一の入手経路）
"""
import os
import sqlite3
import tempfile
import unittest
import json
import shutil

from src.utils.db import init_db, save_odds_snapshots, save_race_schedule


class TestRaceScheduleIsStored(unittest.TestCase):
    """① 発走時刻。parse_header は取っていたがDBに保存されていなかった。"""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, 'data'), exist_ok=True)
        init_db(self.d)

    def _rows(self):
        c = sqlite3.connect(os.path.join(self.d, 'data', 'keiba.db'))
        c.row_factory = sqlite3.Row
        return c.execute('SELECT * FROM race_schedule').fetchall()

    def test_post_time_round_trips(self):
        save_race_schedule([{'race_id': '20260830_05_11', 'date': '2026-08-30',
                             'racecourse': '新潟', 'race_num': 11,
                             'post_time': '15:45', 'n_horses': 16}], base_dir=self.d)
        r = self._rows()[0]
        self.assertEqual(r['post_time'], '15:45',
                         '発走時刻が保存されないと時点別モデルが作れない')
        self.assertEqual(r['n_horses'], 16)

    def test_reruns_do_not_duplicate(self):
        row = {'race_id': 'R1', 'date': '2026-08-30', 'racecourse': '新潟',
               'race_num': 1, 'post_time': '10:05', 'n_horses': 12}
        save_race_schedule([row], base_dir=self.d)
        save_race_schedule([row], base_dir=self.d)
        self.assertEqual(len(self._rows()), 1)


class TestOddsSnapshotKeepsWhatCannotBeRecovered(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, 'data'), exist_ok=True)
        init_db(self.d)

    def _one(self):
        c = sqlite3.connect(os.path.join(self.d, 'data', 'keiba.db'))
        c.row_factory = sqlite3.Row
        return c.execute('SELECT * FROM odds_snapshots').fetchone()

    def test_parser_captures_the_range_not_just_the_average(self):
        """パーサ側が両端を返すこと。DBに列があってもパーサが潰していたら無意味。"""
        import inspect
        from src.scraper import jra_scraper
        src = inspect.getsource(jra_scraper.fetch_odds_for_race)
        self.assertIn('fukusho_min', src,
                      '複勝の下限をパースしていない')
        self.assertIn('fukusho_max', src,
                      '複勝の上限をパースしていない')

    def test_place_odds_range_is_kept(self):
        """② 複勝は範囲で出る。(min+max)/2 に潰すと市場の不確実性が消える。"""
        save_odds_snapshots([{
            'race_id': 'R1', 'horse_num': 3, 'tansho': 5.0,
            'fukusho': 2.0, 'fukusho_min': 1.6, 'fukusho_max': 2.4,
            'captured_at': '2026-08-30 15:00:00',
        }], base_dir=self.d)
        r = self._one()
        self.assertEqual(r['fukusho_min'], 1.6)
        self.assertEqual(r['fukusho_max'], 2.4)
        self.assertNotEqual(r['fukusho_min'], r['fukusho_max'],
                            '範囲幅が失われている')

    def test_scratch_is_recorded(self):
        """③ 取消。1頭消えると全馬のオッズが動くので記録が要る。"""
        save_odds_snapshots([{
            'race_id': 'R1', 'horse_num': 7, 'tansho': None, 'fukusho': None,
            'captured_at': '2026-08-30 15:00:00', 'is_scratched': True,
        }], base_dir=self.d)
        self.assertEqual(self._one()['is_scratched'], 1)

    def test_completeness_is_recorded(self):
        """④ 一部しか取れていない回で市場シェアを正規化すると静かに誤る。"""
        save_odds_snapshots([{
            'race_id': 'R1', 'horse_num': 3, 'tansho': 5.0,
            'captured_at': '2026-08-30 15:00:00',
            'n_captured': 9, 'n_expected': 14,
        }], base_dir=self.d)
        r = self._one()
        self.assertEqual(r['n_captured'], 9)
        self.assertEqual(r['n_expected'], 14)
        self.assertLess(r['n_captured'], r['n_expected'])

    def test_minutes_to_post_is_recorded(self):
        """⑤ リーク対策の切り口。T分前モデルはこの列で行を絞る。"""
        save_odds_snapshots([{
            'race_id': 'R1', 'horse_num': 3, 'tansho': 5.0,
            'captured_at': '2026-08-30 15:00:00', 'minutes_to_post': 30.0,
        }], base_dir=self.d)
        self.assertEqual(self._one()['minutes_to_post'], 30.0)

    def test_old_rows_without_new_columns_still_work(self):
        """既存8,440行を壊さないこと（新列は省略できる）。"""
        n = save_odds_snapshots([{
            'race_id': 'R1', 'horse_num': 3, 'tansho': 5.0, 'fukusho': 2.0,
            'captured_at': '2026-08-30 15:00:00',
        }], base_dir=self.d)
        self.assertEqual(n, 1)
        self.assertIsNone(self._one()['minutes_to_post'])


class TestCollectorWindow(unittest.TestCase):
    """⑥ 発走直前まで取る。結果ページからオッズ列は消滅済みなので
    直前スナップショットが唯一の最終オッズになる。"""

    def test_window_reaches_post_time(self):
        from scripts.collect_odds import WINDOW_END_MIN, WINDOW_START_MIN
        self.assertLessEqual(WINDOW_END_MIN, 1.0,
                             '発走直前まで取らないと最終オッズが手に入らない')
        self.assertGreaterEqual(WINDOW_START_MIN, 60.0,
                                '60分前より前から取らないと変動過程が見えない')

    def test_has_request_budget(self):
        """North Star #4: 新しいリクエスト元には必ず件数上限を設ける。"""
        from scripts.collect_odds import MAX_REQUESTS_DEFAULT
        self.assertGreater(MAX_REQUESTS_DEFAULT, 0)
        self.assertLessEqual(MAX_REQUESTS_DEFAULT, 20000)

    def test_post_dt_parses_and_rejects_garbage(self):
        from scripts.collect_odds import _post_dt
        self.assertIsNotNone(_post_dt('20260830', '15:45'))
        self.assertEqual(_post_dt('20260830', '15:45').hour, 15)
        self.assertIsNone(_post_dt('20260830', ''))
        self.assertIsNone(_post_dt('20260830', None))
        self.assertIsNone(_post_dt('20260830', '不明'))


if __name__ == '__main__':
    unittest.main()


class TestCollectWorkflow(unittest.TestCase):
    """ワークフロー側の取り決めを固定する。

    ⚠ 2026-08-25 に fetch-netkeiba-odds.yml で
    `timeout-minutes: ${{ fromJSON(...) + 10 }}` と式で連動させたら
    GitHubがワークフローを不正と判定して即failureになった。
    そのため数字は直書きし、ここで整合を検査する。
    """

    PATH = '.github/workflows/collect-odds.yml'

    def _yaml(self):
        import yaml
        with open(self.PATH, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_job_timeout_exceeds_script_budget(self):
        d = self._yaml()
        job_timeout = d['jobs']['collect']['timeout-minutes']
        src = open(self.PATH, encoding='utf-8').read()
        self.assertIn('if [ "$MIN" -gt 330 ]; then MIN=330; fi', src,
                      'スクリプト側の上限を丸める処理が無い')
        self.assertGreater(job_timeout, 330,
                           'ジョブのtimeoutがスクリプト上限以下だと途中で殺される')

    def test_no_expression_in_timeout(self):
        """式を書くとGitHubがワークフローごと不正と判定する（2026-08-25の事故）。"""
        src = open(self.PATH, encoding='utf-8').read()
        for line in src.split('\n'):
            if 'timeout-minutes' in line:
                self.assertNotIn('${{', line,
                                 'timeout-minutes に式を書いてはいけない')

    def test_pushes_only_data(self):
        src = open(self.PATH, encoding='utf-8').read()
        self.assertIn('git add data/', src)
        self.assertNotIn('git add -A', src,
                         '説明していない変更まで巻き込む（2026-08-27の反省）')


class TestJsonlSinkAndMerge(unittest.TestCase):
    """収集は keiba.db に触らず JSONL に落ち、取り込みは別工程で行う。

    🔴 これが崩れると、5.5時間走る collect-odds が、その間に走った
       weekend.yml の refresh（11:30 / 14:00 JST）の race_predictions を
       黙って上書きして消す。バイナリなので git はマージできない。
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, 'data'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_collect_odds_does_not_write_keiba_db(self):
        """収集モジュールが keiba.db への書き込み関数を持ち込んでいないこと。"""
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'scripts', 'collect_odds.py')).read()
        for banned in ('save_odds_snapshots', 'save_race_schedule', 'init_db'):
            self.assertNotIn(banned, src,
                             f'collect_odds.py が {banned} を使っている＝keiba.dbを書く')

    def test_append_writes_jsonl_and_survives_reopen(self):
        from scripts.collect_odds import _append, _ts_path
        p = _ts_path(self.d, '20260830')
        _append(p, 'schedule', [{'race_id': 'A', 'post_time': '15:00'}])
        _append(p, 'odds', [{'race_id': 'A', 'horse_num': 1, 'tansho': 3.0}])
        lines = [json.loads(x) for x in open(p, encoding='utf-8')]
        self.assertEqual([r['kind'] for r in lines], ['schedule', 'odds'])

    def test_out_dir_can_point_outside_the_repo(self):
        """CIでは作業ツリー外に書く。git reset --hard で巻き戻らないため。"""
        from scripts.collect_odds import _ts_path
        out = os.path.join(self.d, 'elsewhere')
        p = _ts_path(self.d, '20260830', out_dir=out)
        self.assertTrue(p.startswith(out))
        self.assertNotIn(os.path.join('data', 'odds_ts'), p)

    def test_merge_is_idempotent(self):
        from scripts.collect_odds import _append, _ts_path
        from scripts.merge_odds_ts import merge_odds_ts
        p = _ts_path(self.d, '20260830')
        _append(p, 'schedule', [{'race_id': '20260830_05_11', 'date': '2026-08-30',
                                 'racecourse': '新潟', 'race_num': 11,
                                 'post_time': '15:45', 'n_horses': 16}])
        _append(p, 'odds', [{'race_id': '20260830_05_11', 'horse_num': 1,
                             'tansho': 3.2, 'fukusho': 1.5,
                             'captured_at': '2026-08-30 15:10:00',
                             'minutes_to_post': 35.0, 'n_captured': 16,
                             'n_expected': 16, 'is_scratched': False,
                             'source': 'auto'}])
        first = merge_odds_ts(self.d)
        self.assertEqual(first['odds'], 1)
        self.assertEqual(first['schedule'], 1)
        self.assertFalse(os.path.exists(p), '取り込み済みJSONLは消える')

        # 同じ内容をもう一度流しても増えない
        _append(_ts_path(self.d, '20260830'), 'odds',
                [{'race_id': '20260830_05_11', 'horse_num': 1, 'tansho': 3.2,
                  'fukusho': 1.5, 'captured_at': '2026-08-30 15:10:00',
                  'minutes_to_post': 35.0, 'n_captured': 16, 'n_expected': 16,
                  'is_scratched': False, 'source': 'auto'}])
        self.assertEqual(merge_odds_ts(self.d)['odds'], 0)

    def test_merge_tolerates_truncated_last_line(self):
        """収集中にジョブが落ちると最終行が欠ける。1行捨てて続けること。"""
        from scripts.merge_odds_ts import merge_odds_ts
        d = os.path.join(self.d, 'data', 'odds_ts')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, '20260830.jsonl'), 'w', encoding='utf-8') as f:
            f.write(json.dumps({'kind': 'schedule', 'race_id': 'X',
                                'date': '2026-08-30', 'racecourse': '新潟',
                                'race_num': 1, 'post_time': '10:00',
                                'n_horses': 8}) + '\n')
            f.write('{"kind": "odds", "race_id": "X", "horse_nu')  # 途中で切れた
        self.assertEqual(merge_odds_ts(self.d)['schedule'], 1)

    def test_workflow_pushes_only_odds_ts(self):
        """ワークフローが data/ を丸ごと push していないこと（keiba.db 保護）。"""
        y = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))),
            '.github', 'workflows', 'collect-odds.yml')).read()
        self.assertNotIn('git add data/ ', y)
        self.assertIn('git add data/odds_ts/', y)
        self.assertNotIn('git status --porcelain -- data/)', y)


class TestCollectEndToEnd(unittest.TestCase):
    """`collect()` を実際に一度も呼んでいなかったので引数の取り違えを見逃した。

    2026-08-29 の初回実行は 57秒で終わり、ログにこう出ていた:
        ⚠ 出走表一覧取得失敗: 'str' object has no attribute 'post'
        ❌ <requests.sessions.Session object at 0x...>の開催情報が見つかりません
    `get_kaisai_on_date(sess, date)` と引数順を逆にしていた（正しくは (date, sess)）。
    さらに `fetch_races_on_date` は必須の第3引数 hist_db_path を渡していなかった。

    ⚠ 単体テスト（_append / _ts_path / merge / YAML）は全部通っていた。
      **経路を通していないテストは、経路のバグを見つけられない。**
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, 'data'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_collect_passes_arguments_in_the_right_order(self):
        import scripts.collect_odds as co

        seen = {}

        class FakeSession:
            def post(self, *a, **k):
                raise AssertionError('ネットワークに出てはいけない')

        def fake_kaisai(date_str, sess, calendar=None):
            # 本物と同じ順で受ける。逆なら型で分かる。
            seen['kaisai'] = (type(date_str).__name__, type(sess).__name__)
            assert isinstance(date_str, str), '第1引数が日付文字列でない'
            return {'pw01dde0105': '新潟'}

        def fake_races(sess, target_date, hist_db_path):
            seen['races'] = (type(sess).__name__, type(target_date).__name__,
                             hist_db_path)
            assert isinstance(target_date, str), '第2引数が日付文字列でない'
            return ([{'id': '20260829_05_11', 'date': '2026-08-29',
                      'racecourse': '新潟', 'race_num': 11,
                      'start_time': '15:45', 'horses': [{'num': 1}, {'num': 2}]}], [])

        orig = (co.get_kaisai_on_date, co.fetch_races_on_date, co.create_session)
        co.get_kaisai_on_date = fake_kaisai
        co.fetch_races_on_date = fake_races
        co.create_session = lambda *a, **k: FakeSession()
        try:
            co.collect(self.d, date_str='20260829', max_minutes=0)
        finally:
            (co.get_kaisai_on_date, co.fetch_races_on_date,
             co.create_session) = orig

        self.assertEqual(seen['kaisai'][0], 'str', 'get_kaisai_on_date の第1引数は日付')
        self.assertEqual(seen['races'][1], 'str', 'fetch_races_on_date の第2引数は日付')
        self.assertIn('history.db', seen['races'][2], 'hist_db_path が渡っていない')

        # 発走時刻が JSONL に落ちていること（これが無いと分析が成立しない）
        p = os.path.join(self.d, 'data', 'odds_ts', '20260829.jsonl')
        self.assertTrue(os.path.exists(p), 'スケジュールが書かれていない')
        rows = [json.loads(x) for x in open(p, encoding='utf-8')]
        sched = [r for r in rows if r['kind'] == 'schedule']
        self.assertEqual(len(sched), 1)
        self.assertEqual(sched[0]['post_time'], '15:45')
        self.assertEqual(sched[0]['n_horses'], 2)

    def test_swapped_arguments_raise_instead_of_returning_empty(self):
        """引数を逆にしたら黙って空を返さず落ちること（3度目の再発を止める）。"""
        from src.scraper.calendar import get_kaisai_on_date
        with self.assertRaises(TypeError):
            get_kaisai_on_date(object(), '20260829')


class TestOddsPathActuallyRuns(unittest.TestCase):
    """🔴 前回の統合テストは find_r01_odds を None に固定していた。
       つまり**壊れていた経路そのものを迂回**していたので見逃した。

    2026-08-30 の本番: 5.5時間・162周回して **0行**。原因は3つとも
    「jra_scraper の API を読み違えた」型:
      ① get_kaisai_on_date の戻り値は {base: **日付**}。会場名ではないので
         `if v == meta['racecourse']` は永久に一致しない
      ② find_r01_odds の引数順は (odds_base, date_str, sess)。逆にすると
         'str' object has no attribute 'post' で256件全滅
      ③ base はそのままでは使えず _to_odds_base() で変換が要る

    対策は「自前で引き直すのをやめる」こと。fetch_races_on_date が各レースに
    _odds_cn（base / date_str / race_num / odds_r01）を付けているので、
    既存の fetch_odds_map と同じくそれを使う。
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, 'data'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_collects_odds_rows_through_the_real_coordinate_path(self):
        import datetime as _dt
        import scripts.collect_odds as co

        now = _dt.datetime.now(co.JST)
        post = (now + _dt.timedelta(minutes=30)).strftime('%H:%M')   # 収集窓の中
        date_str = now.strftime('%Y%m%d')
        calls = []

        class FakeSession:
            pass

        def fake_races(sess, target_date, hist_db_path):
            return ([{
                'id': f'{date_str}_05_11', 'date': date_str, 'racecourse': '新潟',
                'race_num': 11, 'start_time': post,
                'horses': [{'num': 1}, {'num': 2}, {'num': 3}],
                '_odds_cn': {'base': 'pw01dde0104' + date_str, 'date_str': date_str,
                             'sx': 0x12, 'race_num': 11, 'odds_r01': 0x5B},
            }], [])

        def fake_fetch_odds(sess, odds_base, race_num, ds, sx):
            calls.append(dict(sess=type(sess).__name__, odds_base=odds_base,
                              race_num=race_num, date_str=ds, sx=sx))
            return {1: {'tansho': 3.2, 'fukusho': 1.5,
                        'fukusho_min': 1.4, 'fukusho_max': 1.6},
                    2: {'tansho': 5.0, 'fukusho': 2.0,
                        'fukusho_min': 1.8, 'fukusho_max': 2.2}}

        orig = (co.get_kaisai_on_date, co.fetch_races_on_date,
                co.fetch_odds_for_race, co.create_session)
        co.get_kaisai_on_date = lambda d, s, calendar=None: {'pw01dde0104' + d: d}
        co.fetch_races_on_date = fake_races
        co.fetch_odds_for_race = fake_fetch_odds
        co.create_session = lambda *a, **k: FakeSession()
        try:
            co.collect(self.d, date_str=date_str, max_minutes=5,
                       interval_sec=1, max_requests=1, out_dir=None)
        finally:
            (co.get_kaisai_on_date, co.fetch_races_on_date,
             co.fetch_odds_for_race, co.create_session) = orig

        self.assertTrue(calls, '🔴 fetch_odds_for_race が一度も呼ばれていない'
                               '（＝またオッズを1行も取れていない）')
        c = calls[0]
        self.assertEqual(c['sess'], 'FakeSession', '第1引数はセッション')
        self.assertTrue(c['odds_base'].startswith('pw151'),
                        f"_to_odds_base を通していない: {c['odds_base']}")
        self.assertEqual(c['race_num'], 11)
        self.assertEqual(c['date_str'], date_str)

        p = os.path.join(self.d, 'data', 'odds_ts', f'{date_str}.jsonl')
        rows = [json.loads(x) for x in open(p, encoding='utf-8')]
        odds = [r for r in rows if r['kind'] == 'odds']
        self.assertTrue(odds, '🔴 オッズ行が1行も書かれていない')
        self.assertEqual(odds[0]['race_id'], f'{date_str}_05_11')
        self.assertIsNotNone(odds[0]['minutes_to_post'])
        self.assertEqual(odds[0]['fukusho_min'], 1.4)
        # スキーマ外の内部キーを漏らしていないこと
        self.assertNotIn('_cn', rows[0])

    def test_does_not_rederive_odds_coordinates_itself(self):
        """自前で find_r01_odds を呼んだり会場名で base を引いたりしないこと。"""
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'scripts', 'collect_odds.py')).read()
        self.assertNotIn('find_r01_odds(', src,
                         'オッズR01の探索を二重に持っている')
        self.assertIn('_odds_cn', src,
                      'fetch_races_on_date が用意した座標を使っていない')
