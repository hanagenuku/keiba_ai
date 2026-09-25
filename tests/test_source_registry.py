"""情報源の台帳とゲートのテスト。

North Star #6 に従い、手打ち dict ではなく
**実際に JSON を書き / 実際に sqlite3 の web_signals.db を作って**検証する。
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.web_signals import (  # noqa: E402
    attach_web_signals, calc_web_signal_features, feature_cols_for, _ranks_and_z)
from src.utils import source_registry as sr  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_registry(base, sources):
    os.makedirs(os.path.join(base, 'data'), exist_ok=True)
    with open(os.path.join(base, 'data', 'source_registry.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'version': 2, 'sources': sources}, f, ensure_ascii=False)


def _make_db(base, rows, source='demo'):
    path = os.path.join(base, 'data', 'private', 'web_signals.db')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE web_signals (
        source TEXT, race_id TEXT, horse_num INTEGER, date TEXT, venue TEXT,
        race_num INTEGER, index_value INTEGER, index_rank INTEGER,
        n_horses INTEGER, published_at TEXT, collected_at TEXT,
        UNIQUE(source, race_id, horse_num))""")
    conn.executemany(
        'INSERT INTO web_signals (source, race_id, horse_num, date, venue, '
        'race_num, index_value, index_rank, n_horses, collected_at) '
        'VALUES (?,?,?,?,?,?,?,?,?,?)',
        [(source, r[0], r[1], '2026-09-21', 'nakayama', 5, r[2], 0, len(rows),
          '2026-09-25') for r in rows])
    conn.commit()
    conn.close()
    return path


def _adopted_entry(base, source='demo', cols=None):
    """裏付けが揃った採用エントリを作る（カード・結果ファイルも実際に置く）。"""
    os.makedirs(os.path.join(base, 'demo'), exist_ok=True)
    for name in ('CRITERIA.md', 'RESULTS.md'):
        with open(os.path.join(base, 'demo', name), 'w') as f:
            f.write('# dummy\n')
    return {
        'source': source,
        'status': 'adopted',
        'card': 'demo/CRITERIA.md',
        'results': 'demo/RESULTS.md',
        'measured': {'delta_auc_mean': 0.004},
        'feature_cols': cols or feature_cols_for(source),
    }


class TestGateDefaultsToNothing(unittest.TestCase):
    def setUp(self):
        sr.reset_warnings()

    def test_real_registry_has_no_adopted_source(self):
        """🔑 本番の台帳は採用ゼロ＝現状の挙動を変えない、を固定する。"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(sr.adopted_sources(base), [])
        self.assertEqual(sr.adopted_feature_cols(base), [])

    def test_missing_registry_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sr.adopted_sources(d), [])

    def test_broken_json_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'data'))
            with open(os.path.join(d, 'data', 'source_registry.json'), 'w') as f:
                f.write('{ this is not json')
            self.assertEqual(sr.adopted_sources(d), [])


class TestAdoptionRequiresEvidence(unittest.TestCase):
    """🔑 台帳に adopted と書くだけでは本番に届かないことを固定する。

    これが撤回された5件（市場補正レイヤー等）との構造的な違い。
    """

    def setUp(self):
        sr.reset_warnings()

    def test_adopted_with_full_evidence_is_honoured(self):
        with tempfile.TemporaryDirectory() as d:
            _write_registry(d, [_adopted_entry(d)])
            got = sr.adopted_sources(d)
            self.assertEqual(len(got), 1)
            self.assertEqual(sr.adopted_feature_cols(d),
                             ['f_ws_demo_rank', 'f_ws_demo_z'])

    def test_adopted_without_results_file_is_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            e = _adopted_entry(d)
            os.remove(os.path.join(d, 'demo', 'RESULTS.md'))
            _write_registry(d, [e])
            self.assertEqual(sr.adopted_sources(d), [])

    def test_adopted_without_card_field_is_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            e = _adopted_entry(d)
            e.pop('card')
            _write_registry(d, [e])
            self.assertEqual(sr.adopted_sources(d), [])

    def test_adopted_without_measured_number_is_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            e = _adopted_entry(d)
            e['measured'] = {'note': '効くと思う'}
            _write_registry(d, [e])
            self.assertEqual(sr.adopted_sources(d), [])

    def test_adopted_with_empty_feature_cols_is_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            e = _adopted_entry(d)
            e['feature_cols'] = []
            _write_registry(d, [e])
            self.assertEqual(sr.adopted_sources(d), [])

    def test_rejected_source_never_reaches_features(self):
        """不採用にした情報源は、DBにデータがあっても列を作らない。"""
        with tempfile.TemporaryDirectory() as d:
            e = _adopted_entry(d)
            e['status'] = 'rejected'
            _write_registry(d, [e])
            _make_db(d, [('20260921_06_05', 1, 80)])
            race = {'race_id': '20260921_06_05'}
            attach_web_signals(race, os.path.join(
                d, 'data', 'private', 'web_signals.db'))
            self.assertEqual(
                calc_web_signal_features({'horse_num': 1}, race, d), {})


class TestRaceInternalNormalisation(unittest.TestCase):
    def test_ties_get_average_rank(self):
        """同値のタイブレークで順位を捏造しない（Gate 0 の前例）。"""
        ranks, _ = _ranks_and_z({1: 80.0, 2: 60.0, 3: 80.0, 4: 50.0})
        self.assertEqual(ranks[1], 1.5)
        self.assertEqual(ranks[3], 1.5)
        self.assertEqual(ranks[2], 3.0)

    def test_z_is_zero_when_all_equal(self):
        _, zs = _ranks_and_z({1: 70.0, 2: 70.0})
        self.assertEqual(set(zs.values()), {0.0})


class TestAttachAndFeatures(unittest.TestCase):
    def setUp(self):
        sr.reset_warnings()

    def test_adopted_source_produces_columns_from_real_db(self):
        """感度検査: 採用されていれば実際に列が出る（ゲートが逆方向にも動く）。"""
        with tempfile.TemporaryDirectory() as d:
            _write_registry(d, [_adopted_entry(d)])
            db = _make_db(d, [('20260921_06_05', 1, 90),
                              ('20260921_06_05', 2, 70),
                              ('20260921_06_05', 3, 50)])
            race = {'race_id': '20260921_06_05'}
            attach_web_signals(race, db)
            self.assertIn('demo', race['web_signals'])
            f1 = calc_web_signal_features({'horse_num': 1}, race, d)
            f3 = calc_web_signal_features({'horse_num': 3}, race, d)
            self.assertEqual(f1['f_ws_demo_rank'], 1.0)
            self.assertEqual(f3['f_ws_demo_rank'], 3.0)
            self.assertGreater(f1['f_ws_demo_z'], f3['f_ws_demo_z'])

    def test_horse_without_signal_gets_nan_not_a_default(self):
        """指数が無い馬は既定値ではなく NaN（欠損と『悪い』を混同しない）。"""
        with tempfile.TemporaryDirectory() as d:
            _write_registry(d, [_adopted_entry(d)])
            db = _make_db(d, [('20260921_06_05', 1, 90)])
            race = {'race_id': '20260921_06_05'}
            attach_web_signals(race, db)
            f = calc_web_signal_features({'horse_num': 9}, race, d)
            self.assertNotEqual(f['f_ws_demo_rank'], f['f_ws_demo_rank'])  # NaN

    def test_missing_db_does_not_raise_and_yields_nan(self):
        with tempfile.TemporaryDirectory() as d:
            _write_registry(d, [_adopted_entry(d)])
            race = {'race_id': '20260921_06_05'}
            attach_web_signals(race, os.path.join(d, 'nope.db'))
            self.assertNotIn('web_signals', race)
            f = calc_web_signal_features({'horse_num': 1}, race, d)
            self.assertNotEqual(f['f_ws_demo_rank'], f['f_ws_demo_rank'])

    def test_no_x_gap_column_is_produced(self):
        """`x_gap`（確定人気が混ざる）を作らないことを固定する。"""
        with tempfile.TemporaryDirectory() as d:
            _write_registry(d, [_adopted_entry(d)])
            db = _make_db(d, [('20260921_06_05', 1, 90)])
            race = {'race_id': '20260921_06_05'}
            attach_web_signals(race, db)
            f = calc_web_signal_features({'horse_num': 1}, race, d)
            self.assertFalse([k for k in f if 'gap' in k])


if __name__ == '__main__':
    unittest.main()


class TestClosedUnmeasuredIsDistinctFromRejected(unittest.TestCase):
    """🔑 「測って不採用」と「測らずに閉じた」を台帳の上で混同させない。

    2026-09-18 の P3-02 で確定した規律:
    測っていないものについて書いてよいのは「測定予算を使う価値が無いと分かった」まで。
    「効かないと分かった」とは書かない。台帳の status もそれを分けて持つ。
    """

    def setUp(self):
        from src.utils import source_registry as sr
        sr.reset_warnings()

    def _rows(self):
        from src.utils.source_registry import source_status
        return {r['source']: r for r in source_status(BASE)['sources']}

    def test_closed_sources_carry_a_reason_and_a_recon_record(self):
        import json
        with open(os.path.join(BASE, 'data', 'source_registry.json'),
                  encoding='utf-8') as f:
            reg = json.load(f)
        closed = [e for e in reg['sources']
                  if e.get('status') == 'closed_unmeasured']
        self.assertGreaterEqual(len(closed), 1)
        for e in closed:
            self.assertTrue(e.get('closed_reason'),
                            f"{e['source']}: closed_unmeasured なのに理由が無い")
            self.assertTrue(e.get('closed_at'), f"{e['source']}: closed_at が無い")
            rec = e.get('recon')
            self.assertTrue(rec, f"{e['source']}: 調査記録へのリンクが無い")
            self.assertTrue(os.path.exists(os.path.join(BASE, rec)),
                            f"{e['source']}: recon={rec} が実在しない")
            # 測っていないので実測値を持たない。持っていたら rejected であるべき
            self.assertIsNone(e.get('measured'),
                              f"{e['source']}: 測っていないのに measured がある")
            self.assertIsNone(e.get('results'),
                              f"{e['source']}: 測っていないのに results がある")

    def test_closed_source_never_reaches_features(self):
        from src.utils.source_registry import adopted_sources
        rows = self._rows()
        closed = [s for s, r in rows.items() if r['status'] == 'closed_unmeasured']
        self.assertTrue(closed)
        for s in closed:
            self.assertNotIn(s, adopted_sources(BASE))

    def test_closed_reason_is_exposed_to_the_app(self):
        rows = self._rows()
        for s, r in rows.items():
            if r['status'] == 'closed_unmeasured':
                self.assertTrue(r.get('closed_reason'),
                                f'{s}: 画面に理由が出ない')

    def test_app_renders_the_closed_status_and_its_reason(self):
        with open(os.path.join(BASE, 'index.html'), encoding='utf-8') as f:
            html = f.read()
        self.assertIn('closed_unmeasured:', html)
        self.assertIn('測らずに閉じた', html)
        self.assertIn('closed_reason', html)
