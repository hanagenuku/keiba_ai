"""「対になっている処理」の片側だけが直る事故を防ぐための固定テスト。

2026-08-09の1日だけで、同じ形の事故が4件出た:
  - 結果ページ側の suffix スキャンには再試行があるのに、出走表側には無かった
    → 新潟が会場ごと予想から消えた
  - 単勝・馬連のEVは cal_prob 誤用を直したのに、三連複だけ直っていなかった
    → 「確率」が3.5を超えるEV8.81が表示されていた
  - サーバ側を軸1頭ベースに変えたのに、クライアント側が古いまま残っていた
    → オッズ取得ボタンで買い目が別物に置き換わっていた
  - 出走表側は取りこぼしをログに出すのに、結果側は無言だった
    → 札幌R10・R11が何の記録も残さず消えた

いずれも「同じことをする箇所が2つあって、片方だけ直した」という同一の型。
そこで**対になっている集合が一致していること自体**をテストで固定する。
"""
import inspect
import re

import pytest


# ── 券種: 確率・推定配当・EV の3者が同じ集合を扱っていること ────────────────
class TestBetTypeCoverage:
    """race_simulator / payout_estimator / ev_calculator の券種集合を揃える。

    2026-08-08にワイドを simulator と estimator に追加したとき、
    ev_calculator だけ追従し忘れていた（確率も配当も出るのにEVだけ無い状態）。
    """

    def _simulator_types(self):
        from src.betting import race_simulator
        # 確率を組み立てるのは calc_ticket_probabilities（simulate_race は着順生成）
        src = inspect.getsource(race_simulator.calc_ticket_probabilities)
        # 戻り値の辞書リテラルからキーを拾う
        return set(re.findall(r"'(\w+)':\s*\{\}", src))

    def _estimator_types(self):
        from src.betting.payout_estimator import _TAKEOUT
        return set(_TAKEOUT)

    def _ev_types(self):
        from src.betting import ev_calculator
        src = inspect.getsource(ev_calculator.calc_ev_all_tickets)
        m = re.search(r'bet_types\s*=\s*\[([^\]]*)\]', src)
        assert m, 'bet_types の定義が見つからない'
        return set(re.findall(r"'(\w+)'", m.group(1)))

    def test_simulator_and_estimator_agree(self):
        assert self._simulator_types() == self._estimator_types(), (
            'race_simulator と payout_estimator の券種がずれている')

    def test_ev_calculator_covers_every_simulated_type(self):
        sim, ev = self._simulator_types(), self._ev_types()
        assert sim <= ev, (
            f'確率は出るのにEVが計算されない券種がある: {sorted(sim - ev)}')

    def test_wide_is_present_everywhere(self):
        # 2026-08-08に追加した券種。回帰の名指し。
        for name, got in [('simulator', self._simulator_types()),
                          ('estimator', self._estimator_types()),
                          ('ev_calculator', self._ev_types())]:
            assert 'wide' in got, f'{name} にワイドが無い'


# ── スクレイパー: 出走表側と結果側の対称性 ──────────────────────────────────
class TestScraperPairSymmetry:
    """同じ処理を出走表用・結果用の2本持っている関数の対称性を固定する。"""

    def _src(self, fn):
        from src.scraper import jra_scraper
        return inspect.getsource(getattr(jra_scraper, fn))

    @pytest.mark.parametrize('fn', ['find_r01_shutuba', 'find_r01_result'])
    def test_r01_scan_has_retry(self, fn):
        # 1周失敗しただけで諦めると、その会場が丸ごと失われる。
        # 結果側は2026-08-02、出走表側は2026-08-09に再試行を入れた。
        sig = inspect.signature(__import__(
            'src.scraper.jra_scraper', fromlist=['x']).__dict__[fn])
        assert 'attempts' in sig.parameters, f'{fn} に再試行が無い'
        assert 'for attempt' in self._src(fn), f'{fn} の再試行ループが無い'

    @pytest.mark.parametrize('fn', ['_try_fetch_shutuba', '_try_fetch_result'])
    def test_single_race_fetch_swallows_network_errors(self, fn):
        # 近傍±60スキャン中に1回通信が切れただけで、そのvenueの残りが
        # 失われるのを防ぐ。結果側は元からtry、出走表側は2026-08-09に追加。
        src = self._src(fn)
        assert 'try:' in src and 'except Exception' in src, (
            f'{fn} が通信例外を握っていない（対の関数と非対称）')

    def test_result_loop_logs_every_skip(self):
        # 出走表側は取りこぼしを必ずログに出す。結果側も同じ粒度にする。
        # 2026-08-09に札幌R10・R11が無言で消えた件の回帰テスト。
        src = self._src('fetch_results')
        # 「soup が取れなかった」「parse できなかった」の2分岐がともに
        # 何かを print してから continue すること
        assert 'パラメータエラー/ページなし' in src, (
            '結果取得でページが取れなかった時のログが無い')
        assert '結果parse失敗' in src, (
            '結果のparseに失敗した時のログが無い')


# ── DB: INSERT した列は UPDATE でも直せること ──────────────────────────────
class TestHistoryWriteSymmetry:
    """再スクレイプで直せない列を作らない。

    2026-08-06の事故: popularity が UPDATE節に無く、列崩れで 99 が入った
    4,484行が再取得しても永久に直らなかった（アンカーが死んで AUC -0.11）。
    その時 trainer は追加されたが、同じ行から取る jockey は追従しなかった。
    """

    def _stmts(self):
        import src.utils.db as db
        src = inspect.getsource(db.save_history_db)
        ins = re.search(r'INSERT OR IGNORE INTO horse_history\s*"?\s*"?\(([^)]*)\)', src)
        upd = re.search(r'UPDATE horse_history SET([\s\S]*?)WHERE race_id', src)
        assert ins and upd, 'INSERT/UPDATE 文が見つからない'
        ins_cols = {c.strip() for c in ins.group(1).replace('"', ' ').split(',') if c.strip()}
        upd_cols = set(re.findall(r'^\s*"?\s*(\w+)\s*=', upd.group(1), re.M))
        return ins_cols, upd_cols

    # 直せなくても実害が無いと確認済みの列だけを免除する。
    # ここに足す時は「なぜ直せなくてよいか」を必ず書くこと。
    _EXEMPT = {
        'race_id', 'horse_num',      # 主キー（WHERE句側）
        'horse_name', 'date', 'racecourse', 'distance',  # レースの同一性を決める不変値
        'corner_3',                  # 常にNULL固定。corner_allから導出する（2026-08-03④）
    }

    def test_every_inserted_column_is_repairable(self):
        ins, upd = self._stmts()
        missing = sorted(ins - upd - self._EXEMPT)
        assert not missing, (
            '再スクレイプしても直せない列がある（popularity と同じ事故が起きうる）: '
            + ', '.join(missing))

    def test_jockey_and_trainer_are_both_repairable(self):
        # 結果ページの同じ行から取る対の列。片方だけ直された前例がある。
        _, upd = self._stmts()
        assert 'jockey' in upd and 'trainer' in upd

    def test_payout_columns_are_repairable(self):
        # 回収率の分子。壊れても「成績が悪い」としか見えない。
        _, upd = self._stmts()
        assert 'tansho_payout' in upd and 'fukusho_payout' in upd


# ── 健全性チェックが「壊れたら困る列」を網羅していること ────────────────────
class TestDataHealthCoverage:
    def test_payout_columns_are_monitored(self):
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'scripts', 'check_data_health.py')
        spec = importlib.util.spec_from_file_location('cdh', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cols = {e[0] for e in mod.COLUMN_CHECKS}
        assert {'tansho_payout', 'fukusho_payout'} <= cols, (
            '回収率の分子が点検対象に入っていない')

    def test_partial_columns_declare_their_scope(self):
        """一部の着順にしか入らない列は分母を絞ること。

        分母を全行にすると、**健全なDBでも**単勝90%・複勝70%が「欠損」と
        数えられ、毎回「⚠ 半分以上が欠損」が鳴る（実測）。この検知器は
        普段黙っていることに価値があるので、常時誤報させてはいけない。
        （tests/test_data_health.py::test_healthy_db_passes が実際に落ちる）
        """
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'scripts', 'check_data_health.py')
        spec = importlib.util.spec_from_file_location('cdh', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        by_col = {e[0]: e for e in mod.COLUMN_CHECKS}
        for col in ('tansho_payout', 'fukusho_payout'):
            assert len(by_col[col]) > 3 and 'place' in by_col[col][3], (
                f'{col} に対象行の絞り込みが無い（分母が全行だと検知できない）')
