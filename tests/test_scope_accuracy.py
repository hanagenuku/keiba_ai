"""成績ページの「対象範囲の的中実績」を固定する。

ユーザー依頼（2026-10-19）:
    「各競馬場9から11レース及び自信のある推奨レースの的中実績、
      9から11レースに関しては、その中に推奨レースがあるならそれがわかるように」

ここで固定したいのは数字ではなく**どの規則で範囲を分けているか**:

1. 範囲の判定は `src/product/scope.py` を呼ぶ。閾値を集計側で作り直さない
   （2026-10-16「危険な人気馬」で、判定条件が2箇所に分かれると記事と台帳が
     静かに食い違うと分かっている）。
2. 「推奨」は**商品側の固定値 conf >= 66**であって、本番の `rec` でも
   `shadow_bets.was_recommended` でもない（2026-10-06 ユーザー決定）。
3. `conf` は `race_predictions` の軸行から復元する。復元式は
   `app_json.race_confidence()` と同一でなければならない。
4. 軸が無い行は「不的中」ではなく**評価不能**として母集団から除く。
5. 的中率の隣に必ず複勝回収率を置く（North Star #9）。的中したのに配当が
   取れていない行は0円として混ぜず分母から外す（2026-08-18③）。
"""
import ast
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_stats import (  # noqa: E402
    _axis_conf_map, _scope_bucket, calc_scope_accuracy,
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


def _race(conn, race_id, venue, num, axis_top3, fuku_hit, tan_hit=0,
          fuku_pay=None, rl1=3, date='2026-10-18'):
    conn.execute(
        'INSERT INTO race_predictions (date, race_id, racecourse, race_num, '
        'horse_num, rl_rank, fuku_prob) VALUES (?,?,?,?,?,1,?)',
        (date, race_id, venue, num, rl1, axis_top3))
    # 軸以外の馬（rl_rank != 1）も入れて、軸行だけを拾えていることを確かめる
    conn.execute(
        'INSERT INTO race_predictions (date, race_id, racecourse, race_num, '
        'horse_num, rl_rank, fuku_prob) VALUES (?,?,?,?,?,2,?)',
        (date, race_id, venue, num, rl1 + 1, 0.99))
    conn.execute(
        'INSERT INTO shadow_bets (date, race_id, racecourse, race_num, rl1_num, '
        'shadow_tansho_hit, shadow_fukusho_hit, shadow_fukusho_payout, '
        'shadow_wide_hit, shadow_umaren_hit, shadow_sanrenp_hit, was_recommended) '
        'VALUES (?,?,?,?,?,?,?,?,0,0,0,0)',
        (date, race_id, venue, num, rl1, tan_hit, fuku_hit, fuku_pay))
    conn.commit()


class TestConfReconstruction:
    def test_matches_race_confidence(self, tmp_path):
        """復元式が app_json.race_confidence() と一致すること。

        別実装で作って値を突合する（North Star #8）。実データでも
        2026-10-18 の34レースで latest.json の conf と 34/34 一致した。
        """
        conn = _db(tmp_path)
        for i, p in enumerate([0.25, 0.555, 0.66, 0.664, 0.7, 0.999]):
            _race(conn, f'20261018_05_{i:02d}', '東京', 9, p, 0)
        got = _axis_conf_map(conn)
        for i, p in enumerate([0.25, 0.555, 0.66, 0.664, 0.7, 0.999]):
            scored = [{'rl_rank': 1, 'top3_prob': p}, {'rl_rank': 2, 'top3_prob': 0.1}]
            assert got[f'20261018_05_{i:02d}'] == race_confidence(scored)

    def test_ignores_non_axis_rows(self, tmp_path):
        """軸以外の馬の fuku_prob を拾わないこと。"""
        conn = _db(tmp_path)
        _race(conn, '20261018_05_09', '東京', 9, 0.40, 0, rl1=3)
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
        """閾値の出どころが scope.py であること（集計側で発明しない）。"""
        code = self._code()
        assert 'scope.FLAT_MIN_CONF' in code
        assert 'scope.is_main_race' in code
        assert 'scope.MAIN_RACE_NUMS' in code
        assert '66' not in code, '閾値を直書きしている。scope.FLAT_MIN_CONF を使うこと'

    def test_does_not_use_production_rec(self):
        """本番の rec / was_recommended に追従していないこと。

        🔴 `rec` は「軸の3着内確率>=0.55 のうちその日の上位6本」で、商品の
        固定閾値とは別物（2026-10-05 に実データで食い違いを確認・10-06 に
        ユーザーが固定値を選択）。
        """
        code = self._code()
        assert 'was_recommended' not in code
        assert "'rec'" not in code.replace("'rec':", '')

    def test_flat_min_conf_is_tickets_conf_strong(self):
        from src.product import tickets
        assert scope.FLAT_MIN_CONF == tickets.CONF_STRONG == 66


class TestGrouping:
    def test_main_races_and_rec_marked_inside(self, tmp_path):
        """9〜11R の中の推奨が分かれること（依頼の核心）。"""
        conn = _db(tmp_path)
        thr = scope.FLAT_MIN_CONF / 100.0
        # 東京 9R: 推奨・複勝的中 / 10R: 非推奨・的中 / 11R: 非推奨・不的中
        _race(conn, '20261018_05_09', '東京', 9, thr + 0.04, 1, fuku_pay=130)
        _race(conn, '20261018_05_10', '東京', 10, thr - 0.10, 1, fuku_pay=110)
        _race(conn, '20261018_05_11', '東京', 11, thr - 0.20, 0)
        # 平場の推奨（9〜11R ではない）
        _race(conn, '20261018_05_03', '東京', 3, thr + 0.10, 1, fuku_pay=150)
        # 平場の非推奨（どちらのグループにも入らない）
        _race(conn, '20261018_05_04', '東京', 4, thr - 0.30, 1, fuku_pay=120)

        sa = calc_scope_accuracy(conn)
        g = {x['label'].strip(): x for x in sa['groups']}
        main = g['9〜11R 全体']
        assert main['races'] == 3 and main['fuku_hit'] == 2
        assert g[f'うち推奨（信頼度{scope.FLAT_MIN_CONF}以上）']['races'] == 1
        assert g['うち推奨以外']['races'] == 2
        rec = g[f'推奨レース 全体（信頼度{scope.FLAT_MIN_CONF}以上）']
        assert rec['races'] == 2, '9〜11R の推奨 + 平場の推奨'
        assert g['うち平場（9〜11R以外）']['races'] == 1
        assert g['（参考）全レース']['races'] == 5

        venue = {v['venue']: v for v in sa['by_venue']}['東京']
        assert venue['races'] == 3 and venue['rec_races'] == 1
        nums = {d['num']: d for d in venue['by_num']}
        assert sorted(nums) == list(scope.MAIN_RACE_NUMS)
        assert nums[9]['rec'] == 1 and nums[10]['rec'] == 0 and nums[11]['rec'] == 0
        assert nums[9]['fuku_hit'] == 1 and nums[11]['fuku_hit'] == 0

        # 直近一覧にも推奨フラグが出る（平場の非推奨は載らない）
        recent = {(r['num'], r['rec']) for r in sa['recent']}
        assert (9, 1) in recent and (10, 0) in recent and (3, 1) in recent
        assert 4 not in {r['num'] for r in sa['recent']}

    def test_conf_exactly_on_the_threshold_is_recommended(self, tmp_path):
        """境界（conf == 66）は推奨に含む（scope.in_scope と同じ >= 判定）。"""
        conn = _db(tmp_path)
        _race(conn, '20261018_05_01', '東京', 1, scope.FLAT_MIN_CONF / 100.0, 1,
              fuku_pay=110)
        sa = calc_scope_accuracy(conn)
        rec = [g for g in sa['groups'] if g['label'].startswith('推奨レース 全体')][0]
        assert rec['races'] == 1
        assert scope.in_scope({'r': 1, 'conf': scope.FLAT_MIN_CONF})[0] is True


class TestEvaluableRowsOnly:
    def test_race_without_axis_is_excluded_not_counted_as_miss(self, tmp_path):
        """軸が無い行（オッズ全滅等）は不的中ではなく評価不能。"""
        conn = _db(tmp_path)
        _race(conn, '20261018_05_09', '東京', 9, 0.60, 1, fuku_pay=120)
        conn.execute(
            'INSERT INTO shadow_bets (date, race_id, racecourse, race_num, '
            'rl1_num, shadow_tansho_hit, shadow_fukusho_hit) '
            "VALUES ('2026-10-18','20261018_05_10','東京',10,NULL,0,0)")
        conn.execute(
            'INSERT INTO race_predictions (date, race_id, racecourse, race_num, '
            "horse_num, rl_rank, fuku_prob) VALUES ('2026-10-18','20261018_05_10',"
            "'東京',10,3,1,0.6)")
        conn.commit()
        sa = calc_scope_accuracy(conn)
        assert sa['excluded']['no_axis'] == 1
        assert sa['races'] == 1
        main = [g for g in sa['groups'] if g['label'] == '9〜11R 全体'][0]
        assert main['races'] == 1 and main['fuku_rate'] == 100.0

    def test_race_without_conf_is_excluded(self, tmp_path):
        conn = _db(tmp_path)
        conn.execute(
            'INSERT INTO shadow_bets (date, race_id, racecourse, race_num, '
            'rl1_num, shadow_tansho_hit, shadow_fukusho_hit) '
            "VALUES ('2026-10-18','20261018_05_09','東京',9,3,0,1)")
        conn.commit()
        sa = calc_scope_accuracy(conn)
        assert sa is None or sa['excluded']['no_conf'] == 1


class TestRecoveryRateIsAlwaysShownNextToHits:
    def test_bucket_has_both_hit_rate_and_recovery(self):
        """的中率だけ出さない（North Star #9）。"""
        rows = [{'shadow_tansho_hit': 1, 'shadow_fukusho_hit': 1,
                 'shadow_fukusho_payout': 200.0},
                {'shadow_tansho_hit': 0, 'shadow_fukusho_hit': 0,
                 'shadow_fukusho_payout': None}]
        b = _scope_bucket(rows, 'x')
        assert b['fuku_rate'] == 50.0
        assert b['fuku_roi'] == 100.0   # 200円 / 2点 = 100%
        assert b['roi_races'] == 2 and b['roi_missing'] == 0

    def test_hit_without_payout_is_excluded_from_the_denominator(self):
        """的中なのに配当が無い行を0円として混ぜない（2026-08-18③）。"""
        rows = [{'shadow_tansho_hit': 0, 'shadow_fukusho_hit': 1,
                 'shadow_fukusho_payout': None},
                {'shadow_tansho_hit': 0, 'shadow_fukusho_hit': 1,
                 'shadow_fukusho_payout': 150.0}]
        b = _scope_bucket(rows, 'x')
        assert b['fuku_hit'] == 2
        assert b['roi_missing'] == 1
        assert b['roi_races'] == 1
        assert b['fuku_roi'] == 150.0, '欠損を0円で混ぜると75%になる'


class TestAppRendering:
    def test_card_is_in_the_page(self):
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        assert 'scope_accuracy' in html
        assert '対象範囲の的中実績' in html

    def test_card_warns_hit_rate_is_not_profit(self):
        """的中率と回収率が別物であることを画面に書く。"""
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        i = html.index('対象範囲の的中実績')
        card = html[i:i + 6000]
        assert '的中率と回収率は別物' in card
        assert '100%に届いていません' in card

    def test_card_does_not_colour_recovery_rates(self):
        """回収率に緑（pos）を付けない。

        「成績表の良いセルだけ買う」は 2026-08-15・08-31 に3度否定済みで、
        緑のセルがその提案を呼んだ。色で選ばせる作りに戻さない。
        """
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        i = html.index('対象範囲の的中実績')
        card = html[i:html.index('// ── RL精度', i)]
        assert 'fuku_roi' in card
        assert "'pos'" not in card and 'pos"' not in card
