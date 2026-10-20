"""2026-10-20 の監査で見つかった4件の修正を固定する。

ユーザー依頼: 成績ページの数字を全部検証 →「全部なおして」

① 波乱度(chaos_grade)が機能していなかった
   `shadow.py` が `classify_race_chaos` に popularity を渡しておらず、
   `top1_pop` が既定の 99 に落ちて `top1_pop >= 8 → C` が必ず成立していた。
   実データでは **2026-07-05 以降 B が1件も生成されず**、1,069件中 C が880件(82%)。

② 結果が無くて決済できない displayed_bets が620点放置されていた
   原因は④（結果のスクレイプ失敗で history.db にレースが入らない）。
   決済側は黙って次回に持ち越すので、誰も気づけなかった。

③ bet_simulation が 2026-10-04 で止まっていた
   2026-10-07 にボタン運用をやめて `predict_next_day` が走らなくなり、
   呼び出し元が消えた。当日refreshから呼べるよう**冪等化**してから繋ぐ
   （log_bet_simulation は重複排除が無く、実データで459組が重複していた）。

④ 結果側だけ取得失敗が無言だった
   出走表側は 2026-09-12 に「カード途中の穴」を検知するようにしたが、
   結果側は無言の continue のまま。片側だけ直した典型例。
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.scraper.jra_scraper as js  # noqa: E402
from src.betting.ev_filter import classify_race_chaos  # noqa: E402
from src.betting.make_bets import log_bet_simulation  # noqa: E402
from src.betting.shadow import record_all_shadow_bets  # noqa: E402
from src.utils.db import (init_db, save_displayed_bets,  # noqa: E402
                          save_race_predictions, settle_displayed_bets,
                          unsettled_displayed_bets)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── ① 波乱度 ────────────────────────────────────────────────────────────
def _race_and_morning(pop_of_top_pn):
    """pn最大の馬の人気だけを振れるレースを作る。"""
    race = {
        'id': '20261018_05_09', 'race_id': '20261018_05_09',
        'date': '2026-10-18', 'racecourse': '東京', 'race_num': 9,
        'race_name': 'テストS', 'surface': '芝', 'distance': 1600,
        'finishers': [
            {'num': n, 'place': i + 1, 'name': f'ウマ{n}', 'win_odds': 2.0 + i}
            for i, n in enumerate([1, 2, 3, 4, 5, 6, 7, 8])],
        'dividends': {
            'tansho':  {'num': 1, 'payout': 200},
            'fukusho': [{'num': n, 'payout': 130} for n in (1, 2, 3)],
            'umaren':  {'nums': [1, 2], 'payout': 800},
            'wide':    [{'nums': [1, 2], 'payout': 300}],
            'sanrenpuku': {'nums': [1, 2, 3], 'payout': 1200},
        },
    }
    # 「A」にも「gap/top3 によるC」にも落ちない領域を選ぶ。
    #   gap_1_2 = 0.06（>=0.04 なのでCではない / <0.10 なのでAでもない）
    #   top3_sum = 0.45（>=0.32 なのでCではない / <0.50 なのでAでもない）
    # → **人気だけ**が B と C を分ける。
    pns = [0.20, 0.14, 0.11] + [0.01] * 5
    horses = [{'num': n, 'name': f'ウマ{n}', 'rl_rank': i + 1,
               'win_prob': pns[i], 'cal_prob': 0.4, 'top3_prob': 0.5,
               'popularity': pop_of_top_pn if i == 0 else i + 2,
               'win_odds': 2.0 + i}
              for i, n in enumerate([1, 2, 3, 4, 5, 6, 7, 8])]
    return race, horses


class TestChaosGradeUsesPopularity:
    def test_classifier_itself_depends_on_popularity(self):
        """前提の確認: popularity が無いと必ず C に落ちる。"""
        scored = [{'pn': 0.20}, {'pn': 0.14}, {'pn': 0.11}]
        assert classify_race_chaos(scored) == 'C'
        with_pop = [dict(h, popularity=p)
                    for h, p in zip(scored, [1, 2, 3])]
        assert classify_race_chaos(with_pop) != 'C'

    def test_shadow_records_a_grade_other_than_c(self, tmp_path):
        """🔴 修正前は popularity を渡しておらず、必ず C になっていた。"""
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        race, horses = _race_and_morning(pop_of_top_pn=1)
        save_race_predictions(
            {'id': race['id'], 'date': race['date'],
             'racecourse': race['racecourse'], 'race_num': race['race_num']},
            horses, base_dir=str(tmp_path))
        record_all_shadow_bets([race], str(tmp_path))
        conn = sqlite3.connect(os.path.join(tmp_path, 'data', 'keiba.db'))
        grade = conn.execute('SELECT chaos_grade FROM shadow_bets').fetchone()[0]
        conn.close()
        assert grade != 'C', 'popularity が届いておらず全部Cになっている'

    def test_unpopular_top_pick_is_still_c(self, tmp_path):
        """人気薄をAIが一番手に推したレースは引き続き C（判定自体は変えない）。"""
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        race, horses = _race_and_morning(pop_of_top_pn=9)
        save_race_predictions(
            {'id': race['id'], 'date': race['date'],
             'racecourse': race['racecourse'], 'race_num': race['race_num']},
            horses, base_dir=str(tmp_path))
        record_all_shadow_bets([race], str(tmp_path))
        conn = sqlite3.connect(os.path.join(tmp_path, 'data', 'keiba.db'))
        grade = conn.execute('SELECT chaos_grade FROM shadow_bets').fetchone()[0]
        conn.close()
        assert grade == 'C'


# ── ③ bet_simulation の冪等化 ───────────────────────────────────────────
def _candidate():
    scored = [{'num': 3, 'name': 'ウマ3', 'pn': 0.30, 'win_odds': 2.0},
              {'num': 4, 'name': 'ウマ4', 'pn': 0.20, 'win_odds': 4.0},
              {'num': 5, 'name': 'ウマ5', 'pn': 0.10, 'win_odds': 8.0}]
    return {'race': {'id': '20261018_05_09', 'date': '2026-10-18',
                     'racecourse': '東京', 'race_num': 9, 'num_horses': 12},
            'scored': scored, 'chaos_score': 0.5, 'score_gap': 0.1}


class TestBetSimulationIsIdempotent:
    def test_logging_twice_does_not_duplicate(self, tmp_path):
        """当日refreshは1日3回走る。冪等でないと3倍に膨らむ。"""
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        c = _candidate()
        log_bet_simulation('2026-10-18', c, str(tmp_path))
        n1 = self._count(tmp_path)
        log_bet_simulation('2026-10-18', c, str(tmp_path))
        assert self._count(tmp_path) == n1 > 0

    def test_init_db_removes_existing_duplicates(self, tmp_path):
        """既に溜まっている重複（実データで459組）を掃除する。"""
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        path = os.path.join(tmp_path, 'data', 'keiba.db')
        conn = sqlite3.connect(path)
        conn.execute('DROP INDEX IF EXISTS idx_bet_sim_uniq')
        for _ in range(3):
            conn.execute(
                "INSERT INTO bet_simulation (date, race_id, bet_type, horse_num) "
                "VALUES ('2026-10-18','20261018_05_09','単勝','3')")
        conn.commit()
        conn.close()
        assert self._count(tmp_path) == 3
        init_db(str(tmp_path))          # マイグレーションが走る
        assert self._count(tmp_path) == 1

    @staticmethod
    def _count(tmp_path):
        conn = sqlite3.connect(os.path.join(tmp_path, 'data', 'keiba.db'))
        n = conn.execute('SELECT COUNT(*) FROM bet_simulation').fetchone()[0]
        conn.close()
        return n


class TestRefreshRecordsSimulation:
    def test_refresh_today_calls_log_bet_simulation(self):
        """🔴 2026-10-04 を最後に記録が止まっていた件。"""
        src = open(os.path.join(BASE, 'scripts', 'weekend.py'),
                   encoding='utf-8').read()
        body = src[src.index('def refresh_today('):src.index('def _keep_started_races(')]
        assert 'log_bet_simulation(' in body

    def test_refresh_today_still_does_not_write_bets(self):
        """`bets` は再実行で行が増えるので refresh からは書かない。"""
        src = open(os.path.join(BASE, 'scripts', 'weekend.py'),
                   encoding='utf-8').read()
        body = src[src.index('def refresh_today('):src.index('def _keep_started_races(')]
        code = '\n'.join(l for l in body.splitlines()
                         if not l.lstrip().startswith('#'))
        assert 'save_bets_db(' not in code


# ── ④ 結果側の取得失敗を記録する ────────────────────────────────────────
def _setup_results(monkeypatch, ok_suffix_by_race):
    monkeypatch.setattr('src.scraper.calendar.get_kaisai_on_date',
                        lambda d, s=None: {'pw01dde0108X': d})
    monkeypatch.setattr(js, 'find_r01_result', lambda b, d, s: 0x4B)
    monkeypatch.setattr(js, '_fill_pedigree', lambda *a, **k: None)
    monkeypatch.setattr(js.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(js, 'SUFFIX_SCAN_HALF_RANGE', 2)
    monkeypatch.setattr(js, 'RACE_RETRY_WAITS', (0,))
    monkeypatch.setattr(
        js, '_try_fetch_result',
        lambda sess, base, r, d, sx: (object()
                                      if ok_suffix_by_race.get(r) == sx else None))

    class _Sess:                        # Step1 を落として出走表一覧へフォールバック
        def post(self, *a, **k):
            raise RuntimeError('step1 unavailable')
    monkeypatch.setattr(js, 'parse_result_soup',
                        lambda soup, rc, r, d, pc: {
                            'race_name': f'R{r}', 'finishers': [
                                {'place': 1, 'num': 1, 'name': 'ウマ',
                                 'running_style': '逃'}]})
    return _Sess()


class TestResultFailuresAreReported:
    def test_a_hole_in_the_middle_of_the_card_is_recorded(self, monkeypatch):
        """Rn が取れず Rn+1 以降が取れている＝実在するのに落ちた。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        del ok[7]                      # R07 だけどの suffix でも取れない
        sess = _setup_results(monkeypatch, ok)
        fails = []
        results = js.fetch_results(sess, '20261010', failures=fails)
        assert len(results) == 11
        assert [f['race_num'] for f in fails] == [7]
        assert 'カード途中' in fails[0]['reason']

    def test_a_short_card_is_not_reported_as_a_failure(self, monkeypatch):
        """11レース開催（末尾が無い）を失敗にしない。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 12)}
        sess = _setup_results(monkeypatch, ok)
        fails = []
        results = js.fetch_results(sess, '20261010', failures=fails)
        assert len(results) == 11 and fails == []

    def test_failures_argument_is_optional(self, monkeypatch):
        """既存の呼び出し（backfill等）を壊さない。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        sess = _setup_results(monkeypatch, ok)
        assert len(js.fetch_results(sess, '20261010')) == 12

    def test_weekend_passes_a_failures_list(self):
        src = open(os.path.join(BASE, 'scripts', 'weekend.py'),
                   encoding='utf-8').read()
        assert 'failures=result_failures' in src
        # 未決済の点検も必ず通す（別名でごまかせないよう import 文を固定する）
        assert 'from src.utils.db import unsettled_displayed_bets' in src
        assert 'pend = unsettled_displayed_bets(' in src


# ── ② 決済できない表示買い目を可視化する ────────────────────────────────
class TestScanBudgetIsPerVenue:
    """🔴 予算を実行全体で共有すると、先の会場が使い切った時に後続会場が全滅する。

    2026-10-18 に 新潟R10 が朝の生成から落ち、`parse_failures` にも
    1件も残らなかった（原因は特定できていない）。会場ごとに割れば、
    少なくとも「他会場の巻き添えで1件も走査できない」は構造的に起きない。
    """

    def test_a_starved_first_venue_does_not_block_the_second(self, monkeypatch):
        asked = []
        monkeypatch.setattr('src.scraper.calendar.get_kaisai_on_date',
                            lambda d, s=None: {'pw01dde0104A': d,
                                               'pw01dde0105B': d})
        monkeypatch.setattr(js, 'find_r01_result', lambda b, d, s: 0x4B)
        monkeypatch.setattr(js, '_fill_pedigree', lambda *a, **k: None)
        monkeypatch.setattr(js.time, 'sleep', lambda *a: None)
        monkeypatch.setattr(js, 'SUFFIX_SCAN_HALF_RANGE', 128)
        monkeypatch.setattr(js, 'SUFFIX_SCAN_BUDGET_DEFAULT', 20)
        monkeypatch.setattr(js, 'RACE_RETRY_WAITS', (0,))

        seen = []

        def fake(sess, base, r, d, sx):
            asked.append((base, r))
            if base not in seen:
                seen.append(base)
            # **最初に処理された会場**は何をしても取れない＝予算を食い潰す。
            if base == seen[0]:
                return None
            # 2つ目の会場は計算式から +1 ずれている＝走査が要る。
            # 予算を会場ごとに持っていなければ、ここで1件も走査できない。
            want = f'{(int(js.calc_suffix(0x4B, r), 16) + 1) % 256:02X}'
            return object() if want == sx else None

        monkeypatch.setattr(js, '_try_fetch_result', fake)
        monkeypatch.setattr(js, 'parse_result_soup',
                            lambda soup, rc, r, d, pc: {
                                'race_name': f'R{r}', 'finishers': [
                                    {'place': 1, 'num': 1, 'name': 'ウマ',
                                     'running_style': '逃'}]})

        class _Sess:
            def post(self, *a, **k):
                raise RuntimeError('step1 unavailable')

        results = js.fetch_results(_Sess(), '20261018')
        assert len(results) == 12, '2会場目が1会場目の予算切れに巻き込まれている'

    def test_budget_is_reset_for_each_venue_in_both_paths(self):
        """出走表側・結果側とも、会場ループの**中**で予算を作り直すこと。"""
        src = open(os.path.join(BASE, 'src', 'scraper', 'jra_scraper.py'),
                   encoding='utf-8').read()
        assert src.count("scan_budget = {'remaining': SUFFIX_SCAN_BUDGET_DEFAULT}") == 2
        for fn in ('def fetch_races_on_date(', 'def fetch_results('):
            body = src[src.index(fn):]
            body = body[:body.index('\n\ndef ', 10)]
            init = body.index("scan_budget = {'remaining'")
            loop = body.index('for base')
            assert loop < init, f'{fn}: 予算が会場ループの外で作られている'


class TestUnsettledDisplayedBets:
    def _setup(self, tmp_path):
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        save_displayed_bets([
            {'date': '2026-10-04', 'race_id': '20261004_05_05',
             'racecourse': '東京', 'race_num': 5, 'is_recommended': 0,
             'snapshot': 'refresh', 'bet_type': 'fukusho', 'combo': '3',
             'amount': 100},
        ], str(tmp_path))
        return str(tmp_path)

    def test_lists_races_whose_results_never_arrived(self, tmp_path):
        base = self._setup(tmp_path)
        pend = unsettled_displayed_bets(base)
        assert [p['race_id'] for p in pend] == ['20261004_05_05']
        assert pend[0]['bets'] == 1

    def test_settled_races_drop_out(self, tmp_path):
        base = self._setup(tmp_path)
        hp = os.path.join(base, 'data', 'history.db')
        hc = sqlite3.connect(hp)
        hc.execute('CREATE TABLE IF NOT EXISTS horse_history '
                   '(race_id TEXT, horse_num INTEGER, place INTEGER)')
        for n, pl in [(3, 1), (4, 2), (5, 3)]:
            hc.execute('INSERT INTO horse_history VALUES (?,?,?)',
                       ('20261004_05_05', n, pl))
        hc.commit()
        hc.close()
        assert unsettled_displayed_bets(base, hist_db_path=hp) == []

    def test_fukusho_outside_the_payout_table_is_a_miss(self, tmp_path):
        """7頭以下で3着の複勝は外れ（settle_displayed_bets 側）。"""
        base = self._setup(tmp_path)
        conn = sqlite3.connect(os.path.join(base, 'data', 'keiba.db'))
        # 払戻表はあるが #3 の行は無い（＝複勝は2着まで）
        for combo, pay in [('4', 150), ('5', 120)]:
            conn.execute('INSERT INTO race_dividends '
                         '(race_id, date, bet_type, combo, payout) '
                         "VALUES ('20261004_05_05','2026-10-04','fukusho',?,?)",
                         (combo, pay))
        conn.commit()
        conn.close()
        hp = os.path.join(base, 'data', 'history.db')
        hc = sqlite3.connect(hp)
        hc.execute('CREATE TABLE IF NOT EXISTS horse_history '
                   '(race_id TEXT, horse_num INTEGER, place INTEGER)')
        for n, pl in [(4, 1), (5, 2), (3, 3)]:
            hc.execute('INSERT INTO horse_history VALUES (?,?,?)',
                       ('20261004_05_05', n, pl))
        hc.commit()
        hc.close()
        out = settle_displayed_bets(base, hist_db_path=hp)
        assert out['settled'] == 1 and out['hit'] == 0
