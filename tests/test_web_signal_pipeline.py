"""収集→蓄積→学習→予想 の結線を固定するテスト。

このプロジェクトは「予想に効くと信じて繋いだ後付け層」を5回撤回している。
ここで固定するのは主に**繋がっていないこと**（採用されるまで本番に届かない）と、
繋いだときに**学習と推論が同じ経路を通ること**。
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    with open(os.path.join(BASE, rel), encoding='utf-8') as f:
        return f.read()


class TestCollectorDoesNotTouchSiteUnlessAdopted(unittest.TestCase):
    def test_collect_for_date_skips_when_not_adopted(self):
        """採用ゼロならサイトを1回も叩かない。"""
        from scripts.collect_web_signals import collect_for_date
        r = collect_for_date(BASE, '2026-09-21')
        self.assertEqual(r['skipped_reason'], 'not_adopted')
        self.assertEqual(r['fetched'], 0)

    def test_collect_for_date_uses_the_same_parser(self):
        """推論用に別のパース経路を作っていないこと。"""
        s = _src('scripts/collect_web_signals.py')
        tree = ast.parse(s)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'collect_for_date')
        called = {n.func.id for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn('parse_article', called)
        self.assertIn('_save', called)


class TestBothSidesShareOneAttachPath(unittest.TestCase):
    """🔑 学習側と推論側が同じ関数で貼ること。別経路だと値が静かにズレる。"""

    def test_training_side_calls_attach_for_adopted(self):
        s = _src('src/tools/build_training_data.py')
        self.assertIn('attach_for_adopted(race, base_dir)', s)

    def test_inference_side_calls_attach_for_adopted(self):
        s = _src('scripts/weekend.py')
        self.assertIn('attach_for_adopted(race, ROOT)', s)

    def test_normalisation_lives_in_one_module_only(self):
        """レース内正規化を他所で再実装していないこと。"""
        for rel in ('src/tools/build_training_data.py', 'scripts/weekend.py',
                    'src/features/engine.py'):
            self.assertNotIn('_ranks_and_z', _src(rel), rel)


class TestInferenceWiringIsUpstreamAndOptional(unittest.TestCase):
    def test_refresh_path_has_kill_switch(self):
        s = _src('scripts/weekend.py')
        self.assertIn("COLLECT_WEB_SIGNALS", s)

    @staticmethod
    def _func_src(rel, name):
        tree = ast.parse(_src(rel))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(_src(rel), fn)

    def test_refresh_path_attaches_before_calc_all(self):
        """貼るのは calc_all より前（＝上流の特徴量。後付け補正ではない）。"""
        s = self._func_src('scripts/weekend.py', 'refresh_today')
        i_attach = s.find('attach_for_adopted(race, ROOT)')
        i_calc = s.find('calc_all(race, avg_bias)')
        self.assertGreater(i_attach, 0, '貼っていない')
        self.assertGreater(i_calc, 0)
        self.assertLess(i_attach, i_calc, '貼るのが calc_all より後になっている')

    def test_previous_evening_path_is_deliberately_not_wired(self):
        """🔑 前日夜の生成（predict_next_day）には**意図的に**繋がない。

        kayochin の記事公開は実測で前日 19:17〜19:39 JST、本番の生成は 19:06。
        11分間に合わないので、拾えるのは当日朝の refresh だけ（CLAUDE.md）。
        ここを「念のため」繋ぐと、毎回空振りのリクエストを撃つうえ、
        取れた日と取れない日で学習/推論の母集団が変わる。
        """
        s = self._func_src('scripts/weekend.py', 'predict_next_day')
        self.assertNotIn('attach_for_adopted', s)
        self.assertNotIn('collect_for_date', s)

    def test_engine_reads_web_signals_in_feature_builder(self):
        s = _src('src/features/engine.py')
        self.assertIn('calc_web_signal_features(h, race', s)


class TestWorkflowGuardsAgainstSilentNoProgress(unittest.TestCase):
    """2026-08-25 D-2① の再発防止をワークフロー側で固定する。

    netkeiba 取得は artifact 復元に失敗して毎回同じ556レースを取り直していた。
    件数を印字するだけでは誰も気づけなかったので、**失敗させる**。
    """

    def setUp(self):
        import yaml
        with open(os.path.join(BASE, '.github/workflows/collect-web-signals.yml'),
                  encoding='utf-8') as f:
            self.wf = yaml.safe_load(f)
        self.steps = self.wf['jobs']['collect']['steps']

    def test_restore_uses_explicit_run_id(self):
        """名前だけの download-artifact は同じrunしか見ない（事故の原因）。"""
        dl = [s for s in self.steps
              if str(s.get('uses', '')).startswith('actions/download-artifact')]
        self.assertEqual(len(dl), 1)
        self.assertIn('run-id', dl[0]['with'])

    def test_job_fails_when_restore_made_no_progress(self):
        verify = [s for s in self.steps if 'Verify restore' in str(s.get('name'))]
        self.assertEqual(len(verify), 1)
        run = verify[0]['run']
        self.assertIn('exit 1', run)
        self.assertIn('web_fetch_log', run)

    def test_db_is_not_committed_to_the_public_repo(self):
        """他所のデータを公開リポジトリに置かない（netkeiba と同じ扱い）。"""
        joined = '\n'.join(str(s.get('run', '')) for s in self.steps)
        self.assertNotIn('git add', joined)
        self.assertNotIn('git commit', joined)
        up = [s for s in self.steps
              if str(s.get('uses', '')).startswith('actions/upload-artifact')]
        self.assertEqual(len(up), 1)
        self.assertIn('data/private/web_signals.db', up[0]['with']['path'])

    def test_private_dir_is_gitignored(self):
        with open(os.path.join(BASE, '.gitignore'), encoding='utf-8') as f:
            gi = f.read()
        self.assertIn('data/private/', gi)


class TestRetrainFailsLoudlyIfAdoptedDataMissing(unittest.TestCase):
    def test_retrain_verifies_adopted_source_data(self):
        import yaml
        with open(os.path.join(BASE, '.github/workflows/monthly-retrain.yml'),
                  encoding='utf-8') as f:
            wf = yaml.safe_load(f)
        steps = list(wf['jobs'].values())[0]['steps']
        names = [str(s.get('name')) for s in steps]
        self.assertTrue(any('Verify adopted sources' in n for n in names), names)
        verify = next(s for s in steps
                      if 'Verify adopted sources' in str(s.get('name')))
        self.assertIn('adopted_sources', verify['run'])
        self.assertIn('SystemExit(1)', verify['run'])

    def test_retrain_restores_before_training(self):
        import yaml
        with open(os.path.join(BASE, '.github/workflows/monthly-retrain.yml'),
                  encoding='utf-8') as f:
            wf = yaml.safe_load(f)
        names = [str(s.get('name')) for s in list(wf['jobs'].values())[0]['steps']]
        self.assertLess(names.index('Restore web signals DB'),
                        names.index('Run monthly retrain'))


class TestLedgerIsDisplayed(unittest.TestCase):
    def test_web_sources_section_reaches_the_app(self):
        self.assertIn('S.web_sources', _src('index.html'))

    def test_generate_stats_emits_web_sources(self):
        self.assertIn("stats['web_sources']", _src('scripts/generate_stats.py'))

    def test_app_warns_that_adoption_needs_evidence(self):
        """画面にも「採用と書くだけでは届かない」を出しておく。"""
        self.assertIn('採用と書くだけでは本番に届きません', _src('index.html'))


if __name__ == '__main__':
    unittest.main()


class TestCollectionAlsoRespectsTheLedger(unittest.TestCase):
    """🔑 台帳は「本番に届くもの」だけでなく「集めるもの」も決める。

    不採用と決めた情報源を毎週取りに行くと、相手のサーバに無駄な負荷をかける。
    逆に、測る前の情報源はデータが無いと測れないので untested も集める。
    """

    def test_rejected_source_is_not_collected(self):
        from scripts.collect_web_signals import collect
        r = collect(BASE, dry_run=True)
        self.assertEqual(r['skipped_reason'], 'status=rejected')
        self.assertEqual(r['fetched'], 0)

    def test_only_decided_against_statuses_block_collection(self):
        from scripts.collect_web_signals import NON_COLLECTABLE_STATUS
        self.assertIn('rejected', NON_COLLECTABLE_STATUS)
        self.assertIn('unusable', NON_COLLECTABLE_STATUS)
        self.assertIn('closed_unmeasured', NON_COLLECTABLE_STATUS)
        self.assertNotIn('untested', NON_COLLECTABLE_STATUS)
        self.assertNotIn('measuring', NON_COLLECTABLE_STATUS)
        self.assertNotIn('adopted', NON_COLLECTABLE_STATUS)

    def test_unregistered_source_is_not_blocked(self):
        """台帳に載っていない情報源は止めない（未登録は決定ではない）。"""
        import tempfile
        from scripts.collect_web_signals import _status_of, NON_COLLECTABLE_STATUS
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_status_of(d, 'kayochinkeiba'))
            self.assertNotIn(None, NON_COLLECTABLE_STATUS)

    def test_force_can_override_for_another_question(self):
        """別の問いのための再収集は --force で通せる（塞ぎきらない）。"""
        s = _src('scripts/collect_web_signals.py')
        self.assertIn("if not force and st in NON_COLLECTABLE_STATUS", s)
        self.assertIn("'--force'", s)

    def test_workflow_treats_nothing_to_collect_as_success(self):
        import yaml
        with open(os.path.join(BASE, '.github/workflows/collect-web-signals.yml'),
                  encoding='utf-8') as f:
            wf = yaml.safe_load(f)
        steps = wf['jobs']['collect']['steps']
        rep = next(s for s in steps if 'Report accumulated' in str(s.get('name')))
        self.assertIn('NON_COLLECTABLE_STATUS', rep['run'])
        self.assertIn('SystemExit(0)', rep['run'])
        up = next(s for s in steps
                  if str(s.get('uses', '')).startswith('actions/upload-artifact'))
        self.assertEqual(up['with']['if-no-files-found'], 'warn')
