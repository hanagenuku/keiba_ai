"""成績ページの「的中実績（軸1頭ベース・全券種）」を固定する。

ユーザー依頼:
  2026-10-19「各競馬場9から11レース及び自信のある推奨レースの的中実績、
              9から11レースに関しては、その中に推奨レースがあるならわかるように」
  2026-10-20「複勝以外の的中率と回収率も。それの累積も。全レースの的中率と回収率も」

ここで固定したいのは数字ではなく**どの規則で測っているか**:

1. 範囲の判定は `src/product/scope.py` を呼ぶ。閾値を集計側で作り直さない
   （2026-10-16「危険な人気馬」で、判定条件が2箇所に分かれると記事と台帳が
     静かに食い違うと分かっている）。
2. 「推奨」は**商品側の固定値 conf >= 66**であって、本番の `rec` でも
   `shadow_bets.was_recommended` でもない（2026-10-06 ユーザー決定）。
3. `conf` は `race_predictions` の軸行から復元する。復元式は
   `app_json.race_confidence()` と同一でなければならない。
4. 🔴 **的中と払戻は `race_dividends` を正とする**。`shadow_*_hit` /
   `shadow_*_payout` 列は使わない（2026-10-20 の監査で3種類の壊れ方を確認）。
5. 🔴 **回収率は必ず「3本抜き」と並べて出す**。100%超のマスは例外なく
   高配当1〜2本で出来ている。
6. 軸が無い行は「不的中」ではなく**評価不能**として母集団から除く。
"""
import ast
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_stats import (  # noqa: E402
    SHADOW_BETS, _axis_conf_map, _bet_stats, _combo_key, _dividend_map,
    calc_scope_accuracy,
)
from src.betting.app_json import race_confidence  # noqa: E402
from src.product import scope  # noqa: E402
from src.utils.db import init_db  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _db(tmp_path):
    """本番と同じスキーマのDBを作る（手打ちCREATEにしない。North Star #6）。"""
    os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
    init_db(str(tmp_path))
    conn = sqlite3.connect(os.path.join(tmp_path, 'data', 'keiba.db'))
    conn.row_factory = sqlite3.Row
    return conn


def _race(conn, race_id, venue, num, axis_top3, rl=(3, 4, 5),
          pays=None, date='2026-10-18'):
    """1レース分を本番と同じ3テーブルに書く。

    pays: {bet_type: {combo: payout}} をそのまま race_dividends へ。
          渡さない券種は「払戻表に行が無い」＝評価不能になる。
    """
    for rank, num_ in enumerate(rl, 1):
        conn.execute(
            'INSERT INTO race_predictions (date, race_id, racecourse, race_num, '
            'horse_num, rl_rank, fuku_prob) VALUES (?,?,?,?,?,?,?)',
            (date, race_id, venue, num, num_, rank,
             axis_top3 if rank == 1 else 0.1))
    conn.execute(
        'INSERT INTO shadow_bets (date, race_id, racecourse, race_num, '
        'rl1_num, rl2_num, rl3_num, was_recommended) VALUES (?,?,?,?,?,?,?,0)',
        (date, race_id, venue, num, rl[0], rl[1], rl[2]))
    for bt, combos in (pays or {}).items():
        for combo, pay in combos.items():
            conn.execute(
                'INSERT INTO race_dividends (race_id, date, bet_type, combo, '
                'payout) VALUES (?,?,?,?,?)', (race_id, date, bt, combo, pay))
    conn.commit()


def _all_pays(rl, tansho=None, fukusho=None, umaren=None, wide=None, san=None):
    """軸がちょうど当たる払戻表を作る。"""
    a, b, c = rl
    return {
        'tansho':     {str(a): tansho} if tansho else {},
        'fukusho':    {str(a): fukusho} if fukusho else {},
        'umaren':     {'-'.join(map(str, sorted([a, b]))): umaren} if umaren else {},
        'wide':       {'-'.join(map(str, sorted([a, b]))): wide} if wide else {},
        'sanrenpuku': {'-'.join(map(str, sorted([a, b, c]))): san} if san else {},
    }


class TestConfReconstruction:
    def test_matches_race_confidence(self, tmp_path):
        """復元式が app_json.race_confidence() と一致すること（North Star #8）。

        実データでも 2026-10-18 の34レースで latest.json の conf と 34/34 一致。
        """
        conn = _db(tmp_path)
        probs = [0.25, 0.555, 0.66, 0.664, 0.7, 0.999]
        for i, p in enumerate(probs):
            _race(conn, f'20261018_05_{i:02d}', '東京', 9, p)
        got = _axis_conf_map(conn)
        for i, p in enumerate(probs):
            scored = [{'rl_rank': 1, 'top3_prob': p}, {'rl_rank': 2, 'top3_prob': 0.1}]
            assert got[f'20261018_05_{i:02d}'] == race_confidence(scored)

    def test_ignores_non_axis_rows(self, tmp_path):
        conn = _db(tmp_path)
        _race(conn, '20261018_05_09', '東京', 9, 0.40, rl=(3, 4, 5))
        assert _axis_conf_map(conn)['20261018_05_09'] == 40


class TestScopeRulesComeFromScopeModule:
    @staticmethod
    def _code():
        """calc_scope_accuracy の**実行されるコードだけ**を返す。

        コメントと docstring は落とす（説明文に書いた 66 を「直書き」と
        誤検知しないため）。
        """
        src = open(os.path.join(BASE, 'scripts', 'generate_stats.py'),
                   encoding='utf-8').read()
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == 'calc_scope_accuracy')
        body = list(fn.body)
        if (isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        return '\n'.join(ast.unparse(n) for n in body)

    def test_threshold_is_not_hardcoded_in_stats(self):
        code = self._code()
        assert 'scope.FLAT_MIN_CONF' in code
        assert 'scope.is_main_race' in code
        assert 'scope.MAIN_RACE_NUMS' in code
        assert '66' not in code, '閾値を直書きしている。scope.FLAT_MIN_CONF を使うこと'

    def test_does_not_use_production_rec(self):
        """本番の rec / was_recommended に追従していないこと。"""
        code = self._code()
        assert 'was_recommended' not in code

    def test_flat_min_conf_is_tickets_conf_strong(self):
        from src.product import tickets
        assert scope.FLAT_MIN_CONF == tickets.CONF_STRONG == 66


class TestPayoutsComeFromDividends:
    """🔴 的中・払戻は race_dividends を正とする（shadow の列を信用しない）。"""

    def test_stats_never_reads_shadow_hit_columns(self):
        src = open(os.path.join(BASE, 'scripts', 'generate_stats.py'),
                   encoding='utf-8').read()
        head = src[:src.index('def calc_divergence_analysis')]
        code = '\n'.join(l for l in head.splitlines()
                         if not l.lstrip().startswith('#'))
        for col in ('shadow_fukusho_hit', 'shadow_tansho_hit',
                    'shadow_fukusho_payout', 'shadow_wide_payout'):
            assert col not in code, f'{col} を読んでいる。race_dividends を正とすること'

    def test_combo_key_is_sorted_ascending(self):
        """組番は昇順。実データで `bets` 側が馬連 10-8 を払戻表の 8-10 と
        突き合わせられず、唯一の的中なのに回収0%と表示していた。"""
        row = {'rl1_num': 10, 'rl2_num': 8, 'rl3_num': 3}
        assert _combo_key(row, 1) == '10'
        assert _combo_key(row, 2) == '8-10'
        assert _combo_key(row, 3) == '3-8-10'

    def test_combo_key_none_when_axis_incomplete(self):
        assert _combo_key({'rl1_num': 1, 'rl2_num': None, 'rl3_num': 3}, 2) is None
        assert _combo_key({'rl1_num': 1, 'rl2_num': 1, 'rl3_num': 3}, 2) is None

    def test_fukusho_without_a_payout_row_is_a_miss_not_a_hit(self, tmp_path):
        """7頭以下は複勝が2着までしか発売されない。3着は**外れ**。

        実データで4件、3着なのに「的中・払戻0円」と記録されていた
        （20260809_07_06 など）。払戻表に行があるかで判定する。
        """
        conn = _db(tmp_path)
        # 軸#3は3着に入ったが、複勝の払戻行は他の2頭ぶんしか無い
        _race(conn, '20261018_05_09', '東京', 9, 0.70, rl=(3, 4, 5),
              pays={'fukusho': {'4': 150, '5': 120},
                    'sanrenpuku': {'3-4-5': 900}})
        divs = _dividend_map(conn)
        rows = conn.execute('SELECT * FROM shadow_bets').fetchall()
        fuku = _bet_stats(rows, divs, '複勝', 'fukusho', 1)
        assert fuku['races'] == 1 and fuku['hit'] == 0, '3着を複勝的中にしない'
        san = _bet_stats(rows, divs, '三連複', 'sanrenpuku', 3)
        assert san['hit'] == 1, '三連複は当たっている'

    def test_missing_dividend_table_is_unevaluable_not_a_miss(self, tmp_path):
        """ワイドの払戻行が1行も無いレースは**評価不能**（不的中にしない）。

        実データで22レース。スクレイプ漏れであって外れではない。
        """
        conn = _db(tmp_path)
        _race(conn, '20261018_05_09', '東京', 9, 0.70, rl=(3, 4, 5),
              pays={'fukusho': {'3': 130}})       # wide の行が無い
        divs = _dividend_map(conn)
        rows = conn.execute('SELECT * FROM shadow_bets').fetchall()
        wide = _bet_stats(rows, divs, 'ワイド', 'wide', 2)
        assert wide['races'] == 0 and wide['na'] == 1


class TestAllBetTypesAndCumulative:
    def test_every_bet_type_has_hit_rate_and_recovery(self, tmp_path):
        """複勝以外も的中率と回収率を出す（2026-10-20 の依頼）。"""
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        _race(conn, '20261018_05_09', '東京', 9, 0.70, rl=rl,
              pays=_all_pays(rl, tansho=300, fukusho=130, umaren=800,
                             wide=250, san=1500))
        sa = calc_scope_accuracy(conn)
        g = sa['groups'][0]
        assert [b['type'] for b in g['bets']] == list(sa['bet_types'])
        assert len(g['bets']) == len(SHADOW_BETS) == 5
        for b in g['bets']:
            assert b['hit'] == 1 and b['hit_rate'] == 100.0
            assert b['roi'] is not None and b['roi_cut3'] is not None

    def test_cumulative_row_sums_all_bet_types(self, tmp_path):
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        _race(conn, '20261018_05_09', '東京', 9, 0.70, rl=rl,
              pays=_all_pays(rl, tansho=300, fukusho=130, umaren=800,
                             wide=250, san=1500))
        g = calc_scope_accuracy(conn)['groups'][0]
        t = g['total']
        assert t['races'] == 5 and t['hit'] == 5
        # 5点買って払戻合計 2980円 → 2980 / (5点×100円) = 596%
        assert t['roi'] == 596.0

    def test_monthly_cumulative_is_running_total(self, tmp_path):
        """「それの累積」＝その月までの累計。"""
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        _race(conn, '20260905_05_09', '東京', 9, 0.70, rl=rl, date='2026-09-05',
              pays=_all_pays(rl, tansho=300, fukusho=130))
        _race(conn, '20261018_05_09', '東京', 9, 0.70, rl=rl, date='2026-10-18',
              pays=_all_pays(rl, tansho=0, fukusho=0))   # 払戻行はあるが全外れ
        m = calc_scope_accuracy(conn)['monthly']
        assert [x['month'] for x in m] == ['2026-09', '2026-10']
        assert m[0]['cum_bets'] == m[0]['bets']
        assert m[1]['cum_bets'] == m[0]['bets'] + m[1]['bets']
        assert m[1]['cum_hit'] == m[0]['hit'] + m[1]['hit']


class TestGrouping:
    def test_main_races_and_rec_marked_inside(self, tmp_path):
        """9〜11R の中の推奨が分かれること（依頼の核心）。"""
        conn = _db(tmp_path)
        thr = scope.FLAT_MIN_CONF / 100.0
        rl = (3, 4, 5)
        p = _all_pays(rl, tansho=300, fukusho=130, umaren=800, wide=250, san=900)
        _race(conn, '20261018_05_09', '東京', 9, thr + 0.04, rl=rl, pays=p)
        _race(conn, '20261018_05_10', '東京', 10, thr - 0.10, rl=rl, pays=p)
        _race(conn, '20261018_05_11', '東京', 11, thr - 0.20, rl=rl, pays=p)
        _race(conn, '20261018_05_03', '東京', 3, thr + 0.10, rl=rl, pays=p)
        _race(conn, '20261018_05_04', '東京', 4, thr - 0.30, rl=rl, pays=p)

        sa = calc_scope_accuracy(conn)
        g = {x['label'].strip(): x for x in sa['groups']}
        assert g['9〜11R 全体']['races'] == 3
        assert g[f'うち推奨（信頼度{scope.FLAT_MIN_CONF}以上）']['races'] == 1
        assert g['うち推奨以外']['races'] == 2
        assert g[f'推奨レース 全体（信頼度{scope.FLAT_MIN_CONF}以上）']['races'] == 2
        assert g['うち平場（9〜11R以外）']['races'] == 1
        assert g['全レース']['races'] == 5

        venue = {v['venue']: v for v in sa['by_venue']}['東京']
        assert venue['races'] == 3 and venue['rec_races'] == 1
        nums = {d['num']: d for d in venue['by_num']}
        assert sorted(nums) == list(scope.MAIN_RACE_NUMS)
        assert nums[9]['rec'] == 1 and nums[10]['rec'] == 0

        recent = {(r['num'], r['rec']) for r in sa['recent']}
        assert (9, 1) in recent and (10, 0) in recent and (3, 1) in recent
        assert 4 not in {r['num'] for r in sa['recent']}

    def test_conf_exactly_on_the_threshold_is_recommended(self, tmp_path):
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        _race(conn, '20261018_05_01', '東京', 1, scope.FLAT_MIN_CONF / 100.0,
              rl=rl, pays=_all_pays(rl, fukusho=110))
        sa = calc_scope_accuracy(conn)
        rec = [g for g in sa['groups'] if g['label'].startswith('推奨レース 全体')][0]
        assert rec['races'] == 1
        assert scope.in_scope({'r': 1, 'conf': scope.FLAT_MIN_CONF})[0] is True


class TestEvaluableRowsOnly:
    def test_race_without_axis_is_excluded_not_counted_as_miss(self, tmp_path):
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        _race(conn, '20261018_05_09', '東京', 9, 0.60, rl=rl,
              pays=_all_pays(rl, fukusho=120))
        conn.execute(
            "INSERT INTO shadow_bets (date, race_id, racecourse, race_num, "
            "rl1_num) VALUES ('2026-10-18','20261018_05_10','東京',10,NULL)")
        conn.execute(
            "INSERT INTO race_predictions (date, race_id, racecourse, race_num, "
            "horse_num, rl_rank, fuku_prob) "
            "VALUES ('2026-10-18','20261018_05_10','東京',10,3,1,0.6)")
        conn.commit()
        sa = calc_scope_accuracy(conn)
        assert sa['excluded']['no_axis'] == 1
        assert sa['races'] == 1
        main = [g for g in sa['groups'] if g['label'] == '9〜11R 全体'][0]
        fuku = [b for b in main['bets'] if b['type'] == '複勝'][0]
        assert fuku['races'] == 1 and fuku['hit_rate'] == 100.0

    def test_race_without_conf_is_excluded(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            "INSERT INTO shadow_bets (date, race_id, racecourse, race_num, "
            "rl1_num) VALUES ('2026-10-18','20261018_05_09','東京',9,3)")
        conn.commit()
        sa = calc_scope_accuracy(conn)
        assert sa is None or sa['excluded']['no_conf'] == 1


class TestRobustnessIsAlwaysShown:
    def test_cut3_is_computed_and_lower_when_one_payout_dominates(self, tmp_path):
        """高配当1本で作られた回収率が「3本抜き」で崩れること。"""
        conn = _db(tmp_path)
        rl = (3, 4, 5)
        for i in range(5):
            _race(conn, f'20261018_05_{i:02d}', '東京', 9, 0.70, rl=rl,
                  pays={'sanrenpuku': ({'3-4-5': 50000} if i == 0 else {'1-2-6': 100})})
        divs = _dividend_map(conn)
        rows = conn.execute('SELECT * FROM shadow_bets').fetchall()
        san = _bet_stats(rows, divs, '三連複', 'sanrenpuku', 3)
        assert san['roi'] == 10000.0        # 50000 / (5点×100)
        assert san['roi_cut3'] == 0.0       # 1本抜けば何も残らない
        assert san['top_pay'] == 50000


class TestAppRendering:
    @staticmethod
    def _card():
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        i = html.index('的中実績（軸1頭ベース・全券種）')
        return html[i:html.index('// ── 直前オッズの変動', i)]

    def test_card_is_in_the_page(self):
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        assert 'scope_accuracy' in html

    def test_card_shows_every_bet_type_and_the_cumulative_row(self):
        card = self._card()
        assert 'g.bets' in card and 'g.total' in card
        assert 'sa.monthly' in card and 'cum_roi' in card

    def test_card_warns_hit_rate_is_not_profit_and_shows_cut3(self):
        card = self._card()
        assert '的中率と回収率は別物' in card
        assert 'roi_cut3' in card and '3本抜き' in card

    def test_card_does_not_colour_recovery_rates(self):
        """回収率に緑（pos）を付けない。

        「成績表の良いセルだけ買う」は3度否定済みで、緑のセルがそれを呼んだ。
        """
        card = self._card()
        assert 'roi' in card
        assert "'pos'" not in card and 'class="pos"' not in card

    def test_removed_cards_are_gone(self):
        """ユーザー依頼（2026-10-20「今までの表示は要りません」）。

        消した表示が stats.json 側にも残っていないこと（溜めるだけにしない）。
        """
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        src = open(os.path.join(BASE, 'scripts', 'generate_stats.py'),
                   encoding='utf-8').read()
        for key in ('rl_accuracy', 'by_type', 'upset_patterns', 'recent_races',
                    'divergence_analysis'):
            assert key not in html, f'{key} の表示が残っている'
            assert f"stats['{key}']" not in src, f'{key} を stats.json に出している'


class TestLatestDayIsSeparateFromCumulative:
    """ユーザー依頼（2026-10-31）:「累計は累計、当日分は日付など一目でわかるように」。

    それまでカード全体が累計で、日付はグレーの小さな期間表記しか無かった。
    ここで固定するのは「当日分と累計が混ざらないこと」と
    「結果がまだ無い日に 0% を出さないこと」の2点。
    """

    @staticmethod
    def _two_days(tmp_path):
        """2開催日ぶん。古い日は全部的中、新しい日は全部ハズレにして取り違えを検出する。"""
        conn = _db(tmp_path)
        # 10-24: 9R/10R とも軸が的中（複勝のみ払戻あり）
        for num in (9, 10):
            _race(conn, f'20261024_05_{num:02d}', '東京', num, 0.70, rl=(1, 2, 3),
                  pays=_all_pays((1, 2, 3), fukusho=200), date='2026-10-24')
        # 10-31: 払戻表に軸が載っていない＝ハズレ
        for num in (9, 10):
            _race(conn, f'20261031_05_{num:02d}', '京都', num, 0.70, rl=(1, 2, 3),
                  pays={'fukusho': {'7': 200}}, date='2026-10-31')
        return conn

    def test_latest_day_is_the_newest_race_day(self, tmp_path):
        sa = calc_scope_accuracy(self._two_days(tmp_path))
        assert sa['latest_day']['date'] == '2026-10-31'
        assert sa['latest_day']['venues'] == ['京都']
        assert sa['latest_day']['races'] == 2

    def test_latest_day_does_not_mix_in_other_days(self, tmp_path):
        """当日分に前の開催日の的中が混ざらないこと（この取り違えが一番こわい）。"""
        sa = calc_scope_accuracy(self._two_days(tmp_path))
        main = sa['latest_day']['groups'][0]
        fuku = [b for b in main['bets'] if b['type'] == '複勝'][0]
        assert fuku['races'] == 2
        assert fuku['hit'] == 0, '10-24 の的中が 10-31 に混ざっている'
        # 累計のほうは両日ぶん入っている
        cum = [g for g in sa['groups'] if g['label'] == '9〜11R 全体'][0]
        cum_fuku = [b for b in cum['bets'] if b['type'] == '複勝'][0]
        assert cum_fuku['races'] == 4 and cum_fuku['hit'] == 2

    def test_cumulative_period_still_covers_every_day(self, tmp_path):
        sa = calc_scope_accuracy(self._two_days(tmp_path))
        assert sa['period'] == {'from': '2026-10-24', 'to': '2026-10-31'}
        assert sa['races'] == 4

    def test_day_without_payouts_is_marked_unsettled_not_zero(self, tmp_path):
        """🔴 夜の結果取得前に「的中0%」と出さない。外れたと誤読させるため。"""
        conn = _db(tmp_path)
        _race(conn, '20261024_05_09', '東京', 9, 0.70, rl=(1, 2, 3),
              pays=_all_pays((1, 2, 3), fukusho=200), date='2026-10-24')
        _race(conn, '20261031_05_09', '京都', 9, 0.70, rl=(1, 2, 3),
              pays=None, date='2026-10-31')          # 払戻がまだ1件も無い
        sa = calc_scope_accuracy(conn)
        assert sa['latest_day']['date'] == '2026-10-31'
        assert sa['latest_day']['settled'] is False
        assert sa['daily'][0]['settled'] is False
        # 結果が入っている日は True
        assert [d for d in sa['daily'] if d['date'] == '2026-10-24'][0]['settled'] is True

    def test_daily_is_newest_first_and_counts_match(self, tmp_path):
        sa = calc_scope_accuracy(self._two_days(tmp_path))
        dates = [d['date'] for d in sa['daily']]
        assert dates == sorted(dates, reverse=True), '日別は新しい順'
        assert sum(d['races'] for d in sa['daily']) == sa['races']

    def test_daily_counts_the_product_scope_separately(self, tmp_path):
        """対象（9〜11R＋推奨）とその日の全レースを別に数えること。"""
        conn = _db(tmp_path)
        _race(conn, '20261031_05_09', '東京', 9, 0.70, rl=(1, 2, 3),
              pays=_all_pays((1, 2, 3), fukusho=200), date='2026-10-31')
        _race(conn, '20261031_05_01', '東京', 1, 0.40, rl=(1, 2, 3),   # 平場・信頼度40
              pays=_all_pays((1, 2, 3), fukusho=200), date='2026-10-31')
        sa = calc_scope_accuracy(conn)
        assert sa['latest_day']['races'] == 2
        assert sa['latest_day']['scope_races'] == 1, '信頼度40の平場は対象外'


class TestPageSeparatesTodayFromCumulative:
    @staticmethod
    def _html():
        with open(os.path.join(BASE, 'index.html'), encoding='utf-8') as f:
            return f.read()

    def test_today_block_is_labelled_with_the_date(self):
        html = self._html()
        assert '📅 当日分' in html
        assert 'sa.latest_day' in html
        assert '_ymdw(ld.date)' in html, '当日分の見出しに日付を出すこと'

    def test_cumulative_block_says_it_is_cumulative(self):
        assert '📊 累計' in self._html()

    def test_unsettled_day_shows_waiting_not_a_rate(self):
        """結果待ちの日に的中率を出さないこと。"""
        html = self._html()
        assert 'ld.settled' in html
        assert '結果待ち' in html

    def test_daily_table_exists_with_dates(self):
        html = self._html()
        assert 'sa.daily' in html
        assert '開催日' in html
