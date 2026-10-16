"""商品化レイヤー（Phase 4）のテスト。

North Star #6 に従い、手打ちの理想的な dict ではなく **本番と同じ形**で検証する:
- `data/latest.json` の実ファイルをそのまま読む
- history.db は実際に sqlite3 で作り `sqlite3.Row` として読み出す
"""

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import (  # noqa: E402
    archive, article, commentary, facts, marks, scope, tickets,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEST = os.path.join(BASE, 'data', 'latest.json')


def _load_real_races():
    with open(LATEST, encoding='utf-8') as f:
        d = json.load(f)
    out = []
    for venue, rs in d['races'].items():
        for r in rs:
            r = dict(r)
            r['_venue'] = venue
            out.append(r)
    return out


@pytest.fixture(scope='module')
def real_races():
    if not os.path.exists(LATEST):
        pytest.skip('data/latest.json が無い')
    races = _load_real_races()
    if not races:
        pytest.skip('latest.json にレースが無い')
    return races


# ---------------------------------------------------------------------------
# 本番の予想を書き換えないこと（最重要の不変条件）
# ---------------------------------------------------------------------------

class TestDoesNotMutatePrediction:
    def test_assign_marks_does_not_mutate_input(self, real_races):
        race = real_races[0]
        before = json.dumps(race['horses'], sort_keys=True, ensure_ascii=False)
        marks.assign_marks(race['horses'])
        after = json.dumps(race['horses'], sort_keys=True, ensure_ascii=False)
        assert before == after

    def test_full_pipeline_does_not_change_prediction_fields(self, real_races):
        """記事を生成しても score/rl_rank/bets/win_prob 等が1つも変わらない。"""
        keys = ('score', 'rl_rank', 'solo_rank', 'cal_prob', 'tan_pct',
                'fuku_pct', 'ability_margin', 'mark')
        for race in real_races:
            if not race.get('horses'):
                continue
            before = [{k: h.get(k) for k in keys} for h in race['horses']]
            bets_before = json.dumps(race.get('bets'), sort_keys=True,
                                     ensure_ascii=False)
            article.build_race_section(race, race['horses'], {}, paid=True)
            after = [{k: h.get(k) for k in keys} for h in race['horses']]
            bets_after = json.dumps(race.get('bets'), sort_keys=True,
                                    ensure_ascii=False)
            assert before == after, f"{race['_venue']}{race.get('r')}R で馬の値が変わった"
            assert bets_before == bets_after


# ---------------------------------------------------------------------------
# 存在しない情報を書かないこと
# ---------------------------------------------------------------------------

class TestNoFabrication:
    def test_generated_article_has_no_forbidden_expressions(self, real_races):
        """全レースの記事に陣営コメント・調教・パドックの語が出ない。"""
        for race in real_races:
            if not race.get('horses'):
                continue
            sec = article.build_race_section(race, race['horses'], {}, paid=True)
            hit = commentary.find_forbidden(sec['markdown'])
            assert hit is None, (
                f"{race['_venue']}{race.get('r')}R の記事に禁止表現「{hit}」"
            )

    def test_trainer_is_not_a_false_positive(self):
        """「調教師」は実データで持っているので禁止語に当たらない。"""
        assert commentary.find_forbidden('調教師は友道康夫。') is None
        assert commentary.find_forbidden('調教タイムは52.5秒。') is not None

    def test_join_raises_on_forbidden(self):
        with pytest.raises(AssertionError):
            commentary._join(['陣営によれば好調とのこと。'])

    def test_no_history_means_no_past_claims(self):
        """過去走が無い馬について過去の話を書かない。"""
        horse = {'n': 1, 'name': 'テスト', 'rl_rank': 1, 'solo_rank': 1, 'pop': 1}
        f = {'n_past_runs': 0}
        text = commentary.horse_comment(horse, f, {'dist': '1600m芝'})
        assert '前走' not in text
        assert '初出走' in text


# ---------------------------------------------------------------------------
# リーク防止（対象日以降の履歴を使わない）
# ---------------------------------------------------------------------------

def _make_history(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE horse_history (
        race_id TEXT, date TEXT, racecourse TEXT, surface TEXT, distance INTEGER,
        place INTEGER, horse_name TEXT, running_style TEXT, agari3f REAL,
        popularity INTEGER, jockey TEXT, class_grade TEXT, agari_rank INTEGER,
        field_size INTEGER, time_diff_sec REAL, weight_load REAL)""")
    conn.execute("""CREATE TABLE race_history (
        race_id TEXT, track_condition TEXT)""")
    for r in rows:
        conn.execute(
            'INSERT INTO horse_history (race_id,date,racecourse,surface,distance,'
            'place,horse_name,running_style,agari3f,popularity,jockey,class_grade,'
            'agari_rank,field_size,time_diff_sec,weight_load) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (r['race_id'], r['date'], r.get('racecourse', '中山'),
             r.get('surface', 'ダート'), r.get('distance', 1800),
             r.get('place'), r['horse_name'], r.get('running_style', '先行'),
             r.get('agari3f'), r.get('popularity'), r.get('jockey', 'A'),
             r.get('class_grade', '1勝'), r.get('agari_rank'),
             r.get('field_size', 16), r.get('time_diff_sec'), r.get('weight_load')))
        conn.execute('INSERT INTO race_history (race_id,track_condition) VALUES (?,?)',
                     (r['race_id'], r.get('track_condition', '良')))
    conn.commit()
    conn.close()


class TestNoLeakage:
    def test_target_race_itself_is_excluded(self, tmp_path):
        """history.db に対象レース自身の結果が入っていても使わない。"""
        db = str(tmp_path / 'h.db')
        _make_history(db, [
            # 対象レース当日（結果取得後に入った行）
            {'race_id': 'T', 'date': '2026-09-27', 'horse_name': 'ウマ',
             'place': 1, 'distance': 1200},
            # それより前の走り
            {'race_id': 'P', 'date': '2026-08-01', 'horse_name': 'ウマ',
             'place': 5, 'distance': 1800},
        ])
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        f = facts.build_facts({'name': 'ウマ'}, {'dist': '1200mダート', '_venue': '中山'},
                              '2026-09-27', conn)
        assert f['n_past_runs'] == 1
        assert f['past_runs'][0]['date'] == '2026-08-01'
        # 前走は 1800m なので「短縮」になる（当日の 1200m を前走と誤認していない）
        assert f['dist_change']['from'] == 1800
        assert f['last_place'] == 5


class TestSmallSampleGuards:
    def test_agari_trend_needs_three_runs(self, tmp_path):
        """1〜2走で末脚の傾向を書かない（North Star #7）。"""
        db = str(tmp_path / 'h.db')
        rows = [{'race_id': f'R{i}', 'date': f'2026-0{i}-01', 'horse_name': 'ウマ',
                 'place': 5, 'agari_rank': 1, 'distance': 1800} for i in (1, 2)]
        _make_history(db, rows)
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        f = facts.build_facts({'name': 'ウマ'}, {'dist': '1800mダート', '_venue': '中山'},
                             '2026-09-01', conn)
        assert f['n_past_runs'] == 2
        assert 'agari_top3' not in f

        rows.append({'race_id': 'R3', 'date': '2026-03-01', 'horse_name': 'ウマ',
                     'place': 5, 'agari_rank': 2, 'distance': 1800})
        db2 = str(tmp_path / 'h2.db')
        _make_history(db2, rows)
        conn2 = sqlite3.connect(f'file:{db2}?mode=ro', uri=True)
        conn2.row_factory = sqlite3.Row
        f2 = facts.build_facts({'name': 'ウマ'}, {'dist': '1800mダート', '_venue': '中山'},
                               '2026-09-01', conn2)
        assert f2['n_past_runs'] == 3
        assert 'agari_top3' in f2


# ---------------------------------------------------------------------------
# 危険な人気馬（測定した母集団を超えない）
# ---------------------------------------------------------------------------

class TestDangerFavorites:
    def test_only_market_first_favorite(self):
        """測定は市場1番人気でしか行っていないので2番人気に広げない。"""
        hs = [
            {'n': 1, 'name': 'A', 'pop': 1, 'solo_rank': 7, 'rl_rank': 1},
            {'n': 2, 'name': 'B', 'pop': 2, 'solo_rank': 9, 'rl_rank': 2},
        ]
        got = marks.danger_favorites(hs)
        assert [d['num'] for d in got] == [1]

    def test_needs_low_solo_rank(self):
        hs = [{'n': 1, 'name': 'A', 'pop': 1, 'solo_rank': 5, 'rl_rank': 1}]
        assert marks.danger_favorites(hs) == []


# ---------------------------------------------------------------------------
# 買い目（固定ルール）
# ---------------------------------------------------------------------------

class TestTickets:
    def _horses(self, conf_axis_pop=3, axis_solo=1):
        return marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': conf_axis_pop,
             'solo_rank': axis_solo, 'odds': 5.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 7.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3, 'odds': 9.0},
        ])

    def _wide_field(self, conf_axis_pop=3, axis_solo=1):
        """固め3頭・中穴2頭・大穴1頭が取れる6〜7頭立て。"""
        return marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': conf_axis_pop,
             'solo_rank': axis_solo, 'odds': 3.2},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 1, 'solo_rank': 3,
             'odds': 2.8},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 2,
             'odds': 6.5},
            {'n': 4, 'name': 'D', 'rl_rank': 4, 'pop': 6, 'solo_rank': 5,
             'odds': 12.0},
            {'n': 5, 'name': 'E', 'rl_rank': 5, 'pop': 8, 'solo_rank': 4,
             'odds': 22.0},
            {'n': 6, 'name': 'F', 'rl_rank': 6, 'pop': 3, 'solo_rank': 6,
             'odds': 5.0},
            {'n': 7, 'name': 'G', 'rl_rank': 7, 'pop': 12, 'solo_rank': 9,
             'odds': 60.0},
        ])

    def test_at_most_max_bet_lines(self):
        """🟢 2026-10-15: 2券種固定をやめ、並べる行数の上限で縛る。"""
        for conf in (30, 50, 58, 66, 80):
            t = tickets.build_tickets({'conf': conf}, self._wide_field())
            assert tickets.bet_type_count(t) <= tickets.MAX_BET_TYPES

    def test_strong_lists_every_bet_type(self):
        """🟢 2026-10-15 ユーザー指示「全券種を記載する」。"""
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        groups = {b['group'] for b in t['bets']}
        assert groups == {'単勝', '複勝', '馬連', 'ワイド', '三連複', '三連単'}
        lines = [b['type'] for b in t['bets']]
        assert lines == ['単勝', '複勝', '馬連', 'ワイド', '三連複',
                         '三連複(2頭軸)', '三連単', '三連単マルチ',
                         '三連単フォーメーション']

    def test_bet_lines_match_the_requested_format(self):
        """ユーザー例の書式（①－②③④⑤⑥ / ①→②③ など）で出す。"""
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        by = {b['type']: b['line'] for b in t['bets']}
        assert by['馬連'] == '馬連　1 － 2・3・4・5・6（5点）'
        assert by['三連複'] == '三連複　1 － 2・3・4・5・6（10点）'
        assert by['三連複(2頭軸)'] == '三連複(2頭軸)　1 － 2 － 3・4・5・6（4点）'
        assert by['三連単'] == '三連単　1 → 2・3・4・5・6（20点）'
        assert by['三連単マルチ'] == '三連単マルチ　1・2 － 3・4・5・6（24点）'
        assert by['三連単フォーメーション'] == \
            '三連単フォーメーション　1 → 2・3 → 4・5・6（6点）'

    def test_displayed_numbers_are_sorted(self):
        """🔑 並びに意味の無い列は馬番の昇順で書く（馬券の書き方）。

        相手を選ぶ順は rl_rank だが、そのまま書くと
        「11 － 1・7・5・10・2」のように読めない行になる（実データで確認）。
        """
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 2, 'solo_rank': 1,
             'odds': 3.2},
            {'n': 6, 'name': 'B', 'rl_rank': 2, 'pop': 1, 'solo_rank': 3,
             'odds': 2.8},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 2,
             'odds': 6.5},
            {'n': 9, 'name': 'D', 'rl_rank': 4, 'pop': 6, 'solo_rank': 5,
             'odds': 12.0},
            {'n': 2, 'name': 'E', 'rl_rank': 5, 'pop': 8, 'solo_rank': 4,
             'odds': 22.0},
            {'n': 7, 'name': 'F', 'rl_rank': 6, 'pop': 3, 'solo_rank': 6,
             'odds': 5.0},
        ])
        t = tickets.build_tickets({'conf': 70}, hs)
        by = {b['type']: b['line'] for b in t['bets']}
        assert by['馬連'] == '馬連　1 － 2・3・6・7・9（5点）'
        # 2頭目の軸は rl_rank で決まる（馬番の昇順ではない）
        assert by['三連複(2頭軸)'] == '三連複(2頭軸)　1 － 6 － 2・3・7・9（4点）'

    def test_partners_mix_solid_and_longshot(self):
        """🟢 「固めと中穴がバランスよく入る」。両方が相手に入ること。"""
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        assert t['solid_partners'] and t['ana_partners'], \
            '固めだけ／中穴だけになっている'
        assert len(t['partners']) == tickets.PARTNERS_STRONG
        assert set(t['partners']) == set(t['solid_partners']) | set(t['ana_partners'])

    def test_longshot_is_pulled_in_even_when_rl_order_is_all_solid(self):
        """🔴 ここが「バランスよく」の本体。

        相手を rl_rank の上位から順に取るだけだと、AI+市場の順位は市場に
        引っ張られるので**上位が固めで埋まって中穴が1頭も入らない**。
        中穴枠を別に確保していることをこのケースで固定する。
        """
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 1, 'solo_rank': 1,
             'odds': 2.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2,
             'odds': 3.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 3, 'solo_rank': 3,
             'odds': 3.5},
            {'n': 4, 'name': 'D', 'rl_rank': 4, 'pop': 4, 'solo_rank': 4,
             'odds': 4.0},
            {'n': 5, 'name': 'E', 'rl_rank': 5, 'pop': 5, 'solo_rank': 5,
             'odds': 5.0},
            {'n': 6, 'name': 'F', 'rl_rank': 6, 'pop': 6, 'solo_rank': 6,
             'odds': 6.0},
            {'n': 7, 'name': 'G', 'rl_rank': 7, 'pop': 7, 'solo_rank': 7,
             'odds': 9.0},
            {'n': 8, 'name': 'H', 'rl_rank': 8, 'pop': 8, 'solo_rank': 8,
             'odds': 18.0},
        ])
        t = tickets.build_tickets({'conf': 70}, hs)
        assert t['ana_partners'] == [7, 8], \
            f'中穴枠が確保されていない: {t["partners"]}'
        assert t['solid_partners'] == [2, 3, 4]
        assert t['partners'] == [2, 3, 4, 7, 8]

    def test_longshots_over_the_band_are_never_partners(self):
        """30倍超（大穴）は相手に入れない。実測で明確に悪い側。

        中穴枠が1頭ぶん埋まらない並びにしてある。上限を外すと
        空いた中穴枠に60倍の馬が入るので、ここで落ちる。
        """
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 2, 'solo_rank': 1,
             'odds': 3.2},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 1, 'solo_rank': 2,
             'odds': 2.8},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3,
             'odds': 6.5},
            {'n': 4, 'name': 'D', 'rl_rank': 4, 'pop': 3, 'solo_rank': 4,
             'odds': 5.0},
            {'n': 5, 'name': 'E', 'rl_rank': 5, 'pop': 6, 'solo_rank': 5,
             'odds': 12.0},
            {'n': 6, 'name': 'F', 'rl_rank': 6, 'pop': 10, 'solo_rank': 6,
             'odds': 60.0},
            {'n': 7, 'name': 'G', 'rl_rank': 7, 'pop': 12, 'solo_rank': 7,
             'odds': 80.0},
        ])
        t = tickets.build_tickets({'conf': 70}, hs)
        assert t['ana_partners'] == [5]
        assert 6 not in t['partners'] and 7 not in t['partners'], \
            f'30倍超の馬が相手に入っている: {t["partners"]}'
        # 枠が埋まらなかったことは記事側で断っている
        assert t['narrow_field'] is True

    def test_wide_is_one_solid_plus_one_longshot(self):
        """ワイドは固め筆頭＋中穴筆頭の2点。"""
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        wide = [b for b in t['bets'] if b['type'] == 'ワイド'][0]
        nums = [c[1] for c in wide['combos']]
        assert len(nums) == 2
        assert nums[0] in t['solid_partners']
        assert nums[1] in t['ana_partners']

    def test_band_thresholds_are_not_invented_here(self):
        """固め／中穴の境界は本番 axis_bets の実測値をそのまま使う。"""
        from src.betting import axis_bets
        assert tickets.ANA_ODDS_MIN == axis_bets.ANA_ODDS_MIN
        assert tickets.ANA_ODDS_MAX == axis_bets.ANA_ODDS_MAX
        src = open(os.path.join(BASE, 'src', 'product', 'tickets.py'),
                   encoding='utf-8').read()
        assert 'from src.betting.axis_bets import' in src, \
            '商品側で中穴の閾値を作り直している'

    def test_partner_needs_usable_odds(self):
        """オッズ未取得の馬は相手にしない（記事に「—倍」の相手を出さない）。"""
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1,
             'odds': 3.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2,
             'odds': 0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3,
             'odds': 8.0},
        ])
        t = tickets.build_tickets({'conf': 70}, hs)
        assert t['partners'] == [3]

    def test_point_count_and_flat_total_are_reported(self):
        """点数が多いので総点数と100円均等の総額を必ず持つ（§10 各自調整）。"""
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        assert t['total_points'] == sum(b['points'] for b in t['bets'])
        assert t['flat_unit_total'] == t['total_points'] * 100

    def test_low_confidence_is_skip(self):
        t = tickets.build_tickets({'conf': 40}, self._horses())
        assert t['verdict'] == 'skip'
        assert t['bets'] == []
        assert t['skip_reason']

    def test_missing_conf_is_skip(self):
        t = tickets.build_tickets({'conf': None}, self._horses())
        assert t['verdict'] == 'skip'

    def test_light_has_no_trifecta(self):
        """conf が最上位の帯に届かなければ三連単は出さない（点数を増やさない）。"""
        t = tickets.build_tickets({'conf': 60}, self._wide_field())
        assert t['verdict'] == 'light'
        groups = {b['group'] for b in t['bets']}
        assert '三連単' not in groups
        assert groups == {'単勝', '複勝', '馬連', 'ワイド', '三連複'}
        assert len(t['partners']) == tickets.PARTNERS_LIGHT

    def test_light_has_fewer_points_than_strong(self):
        strong = tickets.build_tickets({'conf': 70}, self._wide_field())
        light = tickets.build_tickets({'conf': 60}, self._wide_field())
        assert light['total_points'] < strong['total_points']

    def test_axis_danger_favorite_is_downgraded(self):
        """◎が市場1番人気かつ素のAI下位なら1段下げる（§10 買わないも商品）。"""
        hs = self._horses(conf_axis_pop=1, axis_solo=8)
        t = tickets.build_tickets({'conf': 70}, hs)
        assert t['axis_is_danger_favorite'] is True
        assert t['verdict'] == 'light'
        assert t['downgrade_reason']

    def test_axis_danger_favorite_low_conf_becomes_skip(self):
        hs = self._horses(conf_axis_pop=1, axis_solo=8)
        t = tickets.build_tickets({'conf': 58}, hs)
        assert t['verdict'] == 'skip'

    def test_no_odds_is_skip(self):
        """オッズ未取得（土曜夜の日曜予想で毎週起きる）なら見送り。"""
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1, 'odds': 0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 0},
        ])
        t = tickets.build_tickets({'conf': 80}, hs)
        assert t['verdict'] == 'skip'
        assert 'オッズ' in t['skip_reason']

    def test_allocation_sums_to_total(self):
        for hs in (self._horses(), self._wide_field()):
            for conf in (60, 70):
                t = tickets.build_tickets({'conf': conf}, hs)
                for amt, detail in t['allocation_examples'].items():
                    assert sum(detail.values()) == amt, (conf, amt, detail)

    def test_allocation_keeps_the_ratio_order(self):
        """🔴 券種が増えても金額の大小が比率と逆転しない。

        各券種を独立に四捨五入していたとき、1,000円・6券種で
        複勝100円 < ワイド200円 になった（比率は 0.25 vs 0.15）。
        最大剰余法に直した回帰テスト。
        """
        t = tickets.build_tickets({'conf': 70}, self._wide_field())
        ratio = t['allocation_ratio']
        for amt, detail in t['allocation_examples'].items():
            top = max(ratio, key=lambda k: ratio[k])
            assert detail[top] == max(detail.values()), (amt, detail)
            for a in ratio:
                for b in ratio:
                    if ratio[a] > ratio[b]:
                        assert detail[a] >= detail[b], (amt, a, b, detail)


# ---------------------------------------------------------------------------
# §10 資金配分は各自調整 / §11 なぜこの券種か
# ---------------------------------------------------------------------------

class TestAllocationIsNotAsserted:
    """§10「資金配分は、オッズと資金に応じて各自調整 とする」を文面で守る。"""

    def _horses(self):
        return marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1, 'odds': 5.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 7.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3, 'odds': 9.0},
        ])

    def test_note_is_attached_when_allocation_exists(self):
        t = tickets.build_tickets({'conf': 70}, self._horses())
        assert t['allocation_ratio']
        assert t['allocation_note'] == tickets.ALLOCATION_NOTE
        assert '各自調整' in t['allocation_note']

    def test_note_absent_when_skipping(self):
        t = tickets.build_tickets({'conf': 30}, self._horses())
        assert t['allocation_note'] is None
        assert t['bet_type_reasons'] == []

    def test_article_frames_amounts_as_conversion_not_recommendation(self):
        """円の表は「換算例」であって推奨額ではない、と本文に書く。"""
        md = article._bets_md(tickets.build_tickets({'conf': 70}, self._horses()))
        assert '換算例' in md
        assert '推奨額ではありません' in md
        assert '各自調整' in md
        # 断定形（旧文面）が復活していないこと
        assert '金額の目安' not in md

    def test_full_reason_appears_once_per_article_not_per_race(self):
        """期待値が動かない理由の全文は記事末尾に1回だけ（7回くり返さない）。"""
        md = article._bets_md(tickets.build_tickets({'conf': 70}, self._horses()))
        assert tickets.ALLOCATION_NOTE_SHORT in md
        assert '加重平均' not in md                       # レースごとには出さない
        assert '加重平均' in commentary.disclaimer()       # 記事末尾には出す
        assert '各自調整' in commentary.disclaimer()

    def test_allocation_note_does_not_claim_edge_from_allocation(self):
        note = tickets.ALLOCATION_NOTE
        assert '期待値は動きません' in note
        for bad in ('勝てる', '儲か', '必ず'):
            assert bad not in note


class TestBetTypeReason:
    """§11「この予想なら、なぜこの券種なのか」まで説明する。"""

    def _horses(self):
        return marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1, 'odds': 5.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 7.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3, 'odds': 9.0},
        ])

    def test_strong_explains_every_type(self):
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '複勝に配分を最も厚く' in joined
        assert 'ワイドは押さえ' in joined
        for kind in ('馬連', '三連複', '三連単', '単勝'):
            assert kind in joined, f'{kind} の理由が書かれていない'

    def test_reasons_state_that_spreading_is_a_preference_not_an_edge(self):
        """🔴 「広げたから当たる」と読まれない文にする。

        控除率は広げるほど不利（単複20% → 三連単27.5%）という事実を
        同じ場所に書く。2026-10-15 にユーザーの好みで広げた構成なので、
        根拠を取り違えられないようにする。
        """
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '読者の好み' in joined
        assert '当たりやすくなる、という意味ではありません' in joined
        assert '27.5' in joined and '20%' in joined

    def test_reasons_do_not_print_a_recovery_rate_over_100(self):
        """🔴 100%を超える生の数字を有料記事に出さない（自ら禁じた誇大表示）。

        7〜30倍の 102.5% も三連複の 100.1% も高配当1本で作られた数字で、
        1本抜くと 87.4% / 58.9% に落ちる。帯の比較は1本抜き後の数字で書く。
        """
        import re
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        nums = [float(x) for x in re.findall(r'([0-9]+(?:\.[0-9]+)?)%', joined)]
        over = [n for n in nums if n > 100]
        assert over == [], f'100%超の回収率が記事文面に出ている: {over}'

    def test_trifecta_reason_admits_it_is_unmeasured(self):
        """三連単は大規模に測っていない。測っていないものを良いと書かない。"""
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '測っていません' in joined

    def test_light_explains_why_no_trifecta(self):
        t = tickets.build_tickets({'conf': 60}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '今回は三連単を出していません' in joined
        assert '三連単は1着固定の流し' not in joined

    def test_skip_explains_nothing(self):
        """買っていない券種の理由は書かない。"""
        assert tickets.bet_type_reason({'verdict': 'skip', 'bets': []}) == []

    def test_reasons_never_claim_over_100_percent(self):
        """§8・§20 引用する回収率はすべて100%未満。誇張しない。"""
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '100%を下回っています' in joined
        for bad in ('必ず', '儲か', '確実'):
            assert bad not in joined

    def test_reasons_have_no_forbidden_expressions(self):
        """取得経路の無い情報（調教・パドック等）を混ぜていない。"""
        t = tickets.build_tickets({'conf': 70}, self._horses())
        for r in t['bet_type_reasons'] + [tickets.ALLOCATION_NOTE]:
            assert commentary.find_forbidden(r) is None

    def test_article_contains_reason_section(self):
        md = article._bets_md(tickets.build_tickets({'conf': 70}, self._horses()))
        assert 'なぜこの券種か' in md

    def test_single_bet_type_shows_no_allocation_table(self):
        """券種1つなら自明な比率表を載せず、各自調整だけを書く。

        2026-10-15 以降、通常の買い目は券種が複数になるので、
        1券種しか作れなかった場合（相手が1頭も取れない等）を直接作って確認する。
        """
        md = article._bets_md({
            'verdict': 'light',
            'bets': [{'type': '複勝', 'group': '複勝', 'axis': 1,
                      'combos': [[1]], 'points': 1, 'line': '複勝　1'}],
            'total_points': 1, 'flat_unit_total': 100,
            'allocation_ratio': {'複勝': 1.0}, 'allocation_examples': {},
            'bet_type_reasons': [],
        })
        assert '各自調整' in md
        assert '換算例' not in md
        assert '配分の目安' not in md

    def test_article_shows_point_count_and_flat_total(self):
        """🔴 点数が多いので「全部買うと幾らか」を記事に必ず出す。"""
        md = article._bets_md(tickets.build_tickets({'conf': 70}, self._horses()))
        assert '合計' in md and '点' in md
        assert '1点100円で買うと' in md

    def test_article_shows_which_partners_are_solid_or_longshot(self):
        """固め・中穴の内訳を読者に見せる（2026-10-15 指示の実体）。"""
        hs = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1,
             'odds': 3.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2,
             'odds': 4.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 6, 'solo_rank': 3,
             'odds': 11.0},
        ])
        md = article._bets_md(tickets.build_tickets({'conf': 70}, hs))
        assert '相手の内訳' in md
        assert '固め 2' in md and '中穴 3' in md

    def test_partner_breakdown_is_sorted(self):
        """内訳も馬番の昇順で書く（実データで「中穴 7・4・10」と出ていた）。"""
        hs = marks.assign_marks([
            {'n': 5, 'name': 'A', 'rl_rank': 1, 'pop': 1, 'solo_rank': 1,
             'odds': 2.5},
            {'n': 9, 'name': 'B', 'rl_rank': 2, 'pop': 3, 'solo_rank': 2,
             'odds': 6.0},
            {'n': 2, 'name': 'C', 'rl_rank': 3, 'pop': 2, 'solo_rank': 3,
             'odds': 5.0},
            {'n': 7, 'name': 'D', 'rl_rank': 4, 'pop': 4, 'solo_rank': 4,
             'odds': 9.0},
            {'n': 10, 'name': 'E', 'rl_rank': 5, 'pop': 6, 'solo_rank': 5,
             'odds': 20.0},
        ])
        md = article._bets_md(tickets.build_tickets({'conf': 70}, hs))
        assert '固め 2・9' in md, md
        assert '中穴 7・10' in md, md

    def test_published_text_is_frozen_in_archive(self):
        """記事に載せた文面もアーカイブで凍結する（§18 後から変えない）。"""
        assert 'allocation_note' in archive._FROZEN_KEYS
        assert 'bet_type_reasons' in archive._FROZEN_KEYS
        t = tickets.build_tickets({'conf': 70}, self._horses())
        snap = archive.build_snapshot(
            {'race_id': '20260927_06_09', 'r': 9, 'name': 'テスト',
             'dist': '1600m芝', 'conf': 70, '_venue': '中山'},
            self._horses(), [], t, [], {}, 'レース見解')
        assert snap['allocation_note'] == tickets.ALLOCATION_NOTE
        assert snap['bet_type_reasons'] == t['bet_type_reasons']


class TestArticleNeverShowsBrokenOdds:
    def test_unusable_odds_render_as_dash(self):
        """1.0倍未満（＝未取得）は「—」。50.0倍のような正常値は壊さない。"""
        assert article._odds_text(0) == '—'
        assert article._odds_text(0.0) == '—'
        assert article._odds_text(None) == '—'
        assert article._odds_text(0.4) == '—'
        assert article._odds_text(1.0) == '1.0倍'
        assert article._odds_text(50.0) == '50.0倍'

    def test_no_bare_zero_odds_cell_in_real_articles(self, real_races):
        """実データの記事に「0.0倍」というセル/文が出ない（部分一致ではなく境界で見る）。"""
        for race in real_races:
            if not race.get('horses'):
                continue
            sec = article.build_race_section(race, race['horses'], {}, paid=True)
            md = sec['markdown']
            assert '| 0.0倍 |' not in md
            assert '（単勝0.0倍）' not in md
            assert '・0.0倍）' not in md


# ---------------------------------------------------------------------------
# 予想を後から変更しない（§18）
# ---------------------------------------------------------------------------

class TestArchiveImmutability:
    def _snap(self, race_id='20260927_06_09'):
        return archive.build_snapshot(
            {'race_id': race_id, 'r': 9, 'name': 'テスト', 'dist': '1600m芝',
             'conf': 70, '_venue': '中山'},
            [{'n': 1, 'rl_rank': 1, 'solo_rank': 1, 'pop': 1, 'odds': 2.0,
              'fuku_pct': 60}],
            [{'num': 1, 'name': 'A', 'mark': '◎', 'rl_rank': 1}],
            {'stars': 4, 'verdict': 'strong', 'bets': [{'type': '複勝'}],
             'allocation_ratio': {'複勝': 1.0}, 'allocation_examples': {}},
            [], {1: 'コメント'}, 'レース見解')

    def test_second_save_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            archive.save_snapshot(d, '2026-09-27', self._snap())
            with pytest.raises(archive.AlreadyPublished):
                archive.save_snapshot(d, '2026-09-27', self._snap())

    def test_result_attach_does_not_change_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            archive.save_snapshot(d, '2026-09-27', self._snap())
            before = archive.load_snapshot(d, '2026-09-27', '20260927_06_09')
            archive.attach_result(d, '2026-09-27', '20260927_06_09',
                                  {'winner': 3, 'hit': False})
            after = archive.load_snapshot(d, '2026-09-27', '20260927_06_09')
            for k in archive._FROZEN_KEYS:
                assert before.get(k) == after.get(k), f'{k} が変わった'
            assert after['result'] == {'winner': 3, 'hit': False}

    def test_losing_prediction_is_kept(self):
        """外れた予想も消さない（§19）。"""
        with tempfile.TemporaryDirectory() as d:
            archive.save_snapshot(d, '2026-09-27', self._snap())
            archive.attach_result(d, '2026-09-27', '20260927_06_09', {'hit': False})
            assert archive.list_published(d, '2026-09-27') == ['20260927_06_09']

    def test_fingerprint_changes_when_input_changes(self):
        race = {'race_id': 'X', 'conf': 70}
        a = archive.input_fingerprint(race, [{'n': 1, 'rl_rank': 1}])
        b = archive.input_fingerprint(race, [{'n': 1, 'rl_rank': 2}])
        assert a != b


class TestDisclaimer:
    def test_disclaimer_has_no_guarantee_language(self):
        text = commentary.disclaimer()
        for bad in ('必ず', '保証します', '絶対'):
            assert bad not in text
        assert '自己責任' in text

    def test_sources_note_states_what_is_not_used(self):
        note = commentary.sources_note(has_web_info=False)
        assert 'JRA公式' in note
        # 使っていないことを明示する文なので、ここでは禁止語チェックを通さない
        assert '使用していません' in note


class TestThresholdsAreNotInvented:
    """商品の閾値が、決めたとおりの値から動いていないこと。

    - CONF_LIGHT は本番の `ev_filter.MIN_AXIS_FUKU_PROB` と一致していること
    - CONF_STRONG は**商品側で選んだ固定値 66**（2026-10-06 ユーザー決定）
    🔴 ここが動くと、結果を見てから閾値を最適化したのと同じになる。
    """

    def test_conf_light_matches_production_gate(self):
        from src.betting import ev_filter
        assert tickets.CONF_LIGHT == int(round(ev_filter.MIN_AXIS_FUKU_PROB * 100))

    def test_recommended_races_always_clear_conf_light(self, real_races):
        """本番が推奨した（rec=True）レースは必ず CONF_LIGHT 以上。

        これが `rec` と conf の**真の関係**。`rec` は
        「軸の3着内確率 >= MIN_AXIS_FUKU_PROB を満たすレースのうち
        その日の上位 max_races 本」（`ev_filter.select_quality_races`）なので、
        下限は CONF_LIGHT と一致するが、上限側は閾値ではない。
        """
        bad = [(r['_venue'], r.get('r'), r.get('conf'))
               for r in real_races
               if r.get('rec') and r.get('conf') is not None
               and r['conf'] < tickets.CONF_LIGHT]
        assert not bad, f'rec=True なのに conf<{tickets.CONF_LIGHT}: {bad}'

    def test_conf_strong_is_not_claimed_to_be_the_rec_boundary(self):
        """🔴 「CONF_STRONG は rec の境界」と書き戻さないための歯止め。

        2026-09-27 の1日だけ一致していたのを根拠にそう書いていたが、
        10/03 は食い違い4件・10/04 は3件で**偶然だった**。
        `rec` は当日の上位N本なので、固定の conf 値と一致しようがない。
        """
        for name in ('tickets.py', 'scope.py'):
            src = open(os.path.join(BASE, 'src', 'product', name),
                       encoding='utf-8').read()
            assert '2026-10-05 訂正' in src, f'{name} から訂正の記録が消えている'
            assert 'rec=True）の境界そのもの。2026-09-27' not in src

    def test_odds_gate_reason_is_not_about_expected_value(self):
        """🔴 オッズ無しの見送り理由に「妙味（＝期待値）」を使わない。

        `EV = AI確率 × オッズ` で買い目を選ぶのは 2026-07-05 / 07-30 / 08-31 に
        3回否定済み（EV>=1.0 で回収57.9% < 買わない側75.0%）。商品は EV で
        選んでいないので、EV を理由に見送ると自分の記録と矛盾する。
        2026-10-06 にユーザーが指摘して訂正した。書き戻しの歯止め。
        """
        src = open(os.path.join(BASE, 'src', 'product', 'tickets.py'),
                   encoding='utf-8').read()
        assert 'オッズが無い状態では妙味の判断ができない' not in src
        # 見送り自体は残っていること（ゲートを外したわけではない）
        assert 'オッズが取得できていないため見送り' in src

    def test_conf_strong_is_a_fixed_value_not_tied_to_rec(self):
        """🟢 2026-10-06 ユーザー決定「固定の数値でよい。上位6本という選び方はしない」。

        商品の対象範囲は**固定の conf 値**で決める。本番の `rec`
        （その日の上位 max_races 本）に差し替えないための歯止め。
        値を動かすのも、`rec` を見る形に変えるのも、どちらもここで落ちる。
        """
        assert tickets.CONF_STRONG == 66
        for name in ('tickets.py', 'scope.py'):
            src = open(os.path.join(BASE, 'src', 'product', name),
                       encoding='utf-8').read()
            assert '2026-10-06' in src, f'{name} に固定値の決定が記録されていない'
        # 対象範囲の判定が rec を読み始めると「その日の上位N本」に戻ってしまう
        scope_src = open(os.path.join(BASE, 'src', 'product', 'scope.py'),
                         encoding='utf-8').read()
        assert "get('rec')" not in scope_src and 'get("rec")' not in scope_src, \
            'scope.py が rec を参照している（固定の閾値で決める方針に反する）'


class TestScope:
    """対象範囲（ユーザー指示: 各競馬場9R〜11R ＋ 自信のある平場）。"""

    def test_main_races_are_always_in_scope_even_with_low_conf(self):
        # 2026-09-27 中山11R スプリンターズSは conf 54 だが 9〜11R なので対象。
        for num in scope.MAIN_RACE_NUMS:
            ok, why = scope.in_scope({'r': num, 'conf': 10})
            assert ok, f'{num}R が対象外になった: {why}'

    def test_flat_race_needs_confidence(self):
        assert not scope.in_scope({'r': 6, 'conf': scope.FLAT_MIN_CONF - 1})[0]
        assert scope.in_scope({'r': 6, 'conf': scope.FLAT_MIN_CONF})[0]

    def test_flat_race_without_conf_is_out(self):
        ok, why = scope.in_scope({'r': 6, 'conf': None})
        assert not ok
        assert '信頼度' in why

    def test_flat_threshold_is_not_invented(self):
        """平場の閾値をここで別に作らず tickets.CONF_STRONG（固定値66）を使う。"""
        assert scope.FLAT_MIN_CONF == tickets.CONF_STRONG

    def test_select_races_keeps_order(self):
        races = [
            {'r': 1, 'conf': 30},
            {'r': 9, 'conf': 30},
            {'r': 3, 'conf': 99},
            {'r': 11, 'conf': None},
        ]
        assert [r['r'] for r in scope.select_races(races)] == [9, 3, 11]

    def test_scope_on_real_data_matches_the_spec(self, real_races):
        """実データで「9〜11R は全部入る」「平場は rec 相当だけ」を確認する。"""
        for r in real_races:
            ok, _ = scope.in_scope(r)
            if int(r['r']) in scope.MAIN_RACE_NUMS:
                assert ok
            elif ok:
                assert r.get('conf') >= scope.FLAT_MIN_CONF


# ---------------------------------------------------------------------------
# 「履歴が読めない」を「初出走」と書かない（2026-09-30）
# ---------------------------------------------------------------------------

class TestUnknownHistoryIsNotDebut:
    """🔴 商品に嘘を書かないためのガード（§20）。

    `build_facts()` は history.db を引けたときだけ `n_past_runs` を作る。
    LFS 実体が無い等で `conn=None` だと facts が空のまま全馬に渡るため、
    キー欠落を 0 と同じに扱うと **キャリアのある馬にまで「初出走」と書く**。
    """

    HORSE = {'n': 1, 'name': 'テスト', 'rl_rank': 1, 'solo_rank': 1, 'pop': 1}
    RACE = {'dist': '1600m芝'}

    def test_empty_facts_does_not_claim_debut(self):
        text = commentary.horse_comment(self.HORSE, {}, self.RACE)
        assert '初出走' not in text
        assert '前走' not in text

    def test_measured_zero_runs_still_claims_debut(self):
        """実際に引いて0走だったときは従来どおり書く（黙らせてはいない）。"""
        text = commentary.horse_comment(self.HORSE, {'n_past_runs': 0}, self.RACE)
        assert '初出走' in text

    def test_real_data_debut_claims_match_measured_runs(self, real_races):
        """実データ・実 history.db で、初出走と書く馬が実測0走の馬に一致する。"""
        conn = facts.open_history(BASE)
        if conn is None:
            pytest.skip('history.db の実体が無い（LFS未取得）')
        race = dict(real_races[0])
        for h in race.get('horses') or []:
            f = facts.build_facts(h, race, '2026-09-27', conn)
            text = commentary.horse_comment(h, f, race)
            assert ('初出走' in text) == (f.get('n_past_runs') == 0)


# ---------------------------------------------------------------------------
# 発走後のレースを「公開」として保存しない（2026-09-30）
# ---------------------------------------------------------------------------

class TestStalePublicationIsRefused:
    """アーカイブの価値は「発走前に公開したものが残っている」ことに尽きる。

    結果が出た後に保存すると `published_at` が発走日より後になり、記録自体が嘘になる。
    """

    def test_past_date_raises(self):
        with pytest.raises(archive.StalePublication):
            archive.assert_publishable(
                '2026-09-27', now=archive.datetime(2026, 9, 30, tzinfo=archive.JST))

    def test_same_day_is_allowed(self):
        archive.assert_publishable(
            '2026-09-27', now=archive.datetime(2026, 9, 27, 8, tzinfo=archive.JST))

    def test_future_date_is_allowed(self):
        """前日夜に翌日ぶんを生成する運用があるので未来日は通す。"""
        archive.assert_publishable(
            '2026-09-27', now=archive.datetime(2026, 9, 26, 19, tzinfo=archive.JST))

    def test_guard_is_date_only_and_says_so(self):
        """時刻までは見ない（latest.json に発走時刻が無い）。その射程を明記してある。"""
        assert '時刻は見ていない' in archive.assert_publishable.__doc__

    def test_cli_refuses_to_publish_stale_and_writes_nothing(self):
        """CLI が実際に中止し、1件も保存しないこと。"""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'data'))
            with open(LATEST, encoding='utf-8') as f:
                raw = f.read()
            with open(os.path.join(d, 'data', 'latest.json'), 'w',
                      encoding='utf-8') as f:
                f.write(raw)  # 日付は 2026-09-27（既に発走済み）
            r = subprocess.run(
                [sys.executable, os.path.join(BASE, 'scripts',
                                              'build_product_article.py'),
                 '--base-dir', d, '--publish',
                 '--out', os.path.join(d, 'out.md')],
                capture_output=True, text=True)
            assert r.returncode == 1
            assert '公開を中止' in r.stderr
            assert not os.path.exists(os.path.join(d, archive.ARCHIVE_DIRNAME))


# ---------------------------------------------------------------------------
# 公開済みの予想に結果を追記する（§12 の「結果」）
#
# 🔴 的中判定は払戻表（race_dividends）が正。着順から自分で判定しない。
#    実データの反例: 2026-09-27 中山9R は7頭立てで複勝が2着までしか発売されず、
#    3着（馬番2）に複勝配当が無い。一方ワイドは同じレースで3着内の2頭組が
#    成立している。この非対称を手書きの規則で再現しようとすると必ず誤判定する。
# ---------------------------------------------------------------------------

from src.product import results as results_mod  # noqa: E402
from src.utils import db as db_mod  # noqa: E402


def _make_result_dbs(base_dir, placings, dividends, race_id, date='2026-09-27'):
    """本番と同じ経路・同じスキーマで着順と払戻を用意する。

    払戻は `src/utils/db.py::save_dividends_db`（本番の書き込み口）を通すので、
    combo の正規化もテスト側で手打ちしない。
    """
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    hp = os.path.join(base_dir, 'data', 'history.db')
    conn = sqlite3.connect(hp)
    conn.execute('CREATE TABLE horse_history ('
                 'race_id TEXT, horse_num INTEGER, place INTEGER, '
                 'fukusho_payout INTEGER)')
    for num, place in placings.items():
        conn.execute('INSERT INTO horse_history (race_id,horse_num,place,'
                     'fukusho_payout) VALUES (?,?,?,?)',
                     (race_id, num, place, None))
    conn.commit()
    conn.close()
    db_mod.init_db(base_dir)
    if dividends is not None:
        db_mod.save_dividends_db(
            [{'race_id': race_id, 'date': date, 'dividends': dividends}],
            base_dir=base_dir)
    return hp


def _publish(base_dir, race_id, bets, date='2026-09-27', ranks=None):
    snap = archive.build_snapshot(
        {'race_id': race_id, 'r': 9, 'name': 'テスト', 'dist': '1600m芝',
         'conf': 70, '_venue': '中山'},
        [{'n': 1, 'rl_rank': 1, 'solo_rank': 1, 'pop': 1, 'odds': 2.0,
          'fuku_pct': 60}],
        ranks if ranks is not None else [{'num': 2, 'name': 'A', 'mark': '◎'}],
        {'stars': 4, 'verdict': 'strong', 'bets': bets,
         'allocation_ratio': {}, 'allocation_examples': {}},
        [], {}, 'レース見解')
    archive.save_snapshot(base_dir, date, snap)


# 2026-09-27 中山9R の実データ（7頭立て・複勝は2着まで）
_REAL_PLACINGS = {1: 4, 2: 3, 3: 1, 4: 5, 5: 6, 6: 7, 7: 2}
_REAL_DIVS = {
    'fukusho': [{'num': 3, 'payout': 180}, {'num': 7, 'payout': 110}],
    'wide': [{'nums': [2, 3], 'payout': 390}, {'nums': [2, 7], 'payout': 170},
             {'nums': [3, 7], 'payout': 180}],
}


class TestAttachResults:
    RID = '20260927_06_09'

    def test_small_field_third_place_fukusho_is_not_a_hit(self, tmp_path):
        """🔴 7頭立ての3着。複勝は発売されていないので的中にしない。

        着順だけで「3着内だから複勝的中」と判定すると、ここで嘘の的中が記録される。
        """
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [
            {'type': '複勝', 'axis': 2, 'combos': [[2]]},
            {'type': 'ワイド', 'axis': 2, 'combos': [[2, 3]]},
        ])
        hist, keiba = results_mod.open_history(d), results_mod.open_keiba(d)
        results_mod.attach_for_date(d, '2026-09-27', hist, keiba)
        r = archive.load_snapshot(d, '2026-09-27', self.RID)['result']
        fuku = [b for b in r['bets'] if b['type'] == '複勝'][0]
        wide = [b for b in r['bets'] if b['type'] == 'ワイド'][0]
        assert r['placings']['2'] == 3          # 3着である
        assert fuku['hit'] is False             # それでも複勝は的中ではない
        assert fuku['payout_per_100'] is None
        assert wide['hit'] is True              # ワイドは3着内の2頭組で成立
        assert wide['payout_per_100'] == 390

    def test_hit_and_payout_come_from_dividends(self, tmp_path):
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        results_mod.attach_for_date(d, '2026-09-27', results_mod.open_history(d),
                                    results_mod.open_keiba(d))
        r = archive.load_snapshot(d, '2026-09-27', self.RID)['result']
        assert r['bets'][0] == {'type': '複勝', 'combo': [7], 'hit': True,
                                'payout_per_100': 110, 'payout_known': True}
        assert r['top3'] == [3, 7, 2]
        assert r['cross_check'] == 'ok'

    def test_prediction_is_untouched(self, tmp_path):
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        before = archive.load_snapshot(d, '2026-09-27', self.RID)
        results_mod.attach_for_date(d, '2026-09-27', results_mod.open_history(d),
                                    results_mod.open_keiba(d))
        after = archive.load_snapshot(d, '2026-09-27', self.RID)
        for k in archive._FROZEN_KEYS:
            assert before.get(k) == after.get(k), f'{k} が変わった'

    def test_missing_dividends_writes_nothing(self, tmp_path):
        """払戻が未取得なら結果を書かない（全部外れとして記録しない）。"""
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, None, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        out = results_mod.attach_for_date(d, '2026-09-27',
                                         results_mod.open_history(d),
                                         results_mod.open_keiba(d))
        assert out == {'attached': 0, 'already': 0, 'pending': 1, 'mismatch': []}
        assert archive.load_snapshot(d, '2026-09-27', self.RID)['result'] is None

    def test_rerun_is_idempotent(self, tmp_path):
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        hist, keiba = results_mod.open_history(d), results_mod.open_keiba(d)
        first = results_mod.attach_for_date(d, '2026-09-27', hist, keiba)
        snap1 = archive.load_snapshot(d, '2026-09-27', self.RID)
        second = results_mod.attach_for_date(d, '2026-09-27', hist, keiba)
        snap2 = archive.load_snapshot(d, '2026-09-27', self.RID)
        assert first['attached'] == 1 and second['attached'] == 0
        assert second['already'] == 1
        assert snap1 == snap2          # 2回目で1バイトも変わらない

    def test_cross_check_detects_disagreement(self, tmp_path):
        """着順（history.db）と払戻（keiba.db）が食い違えば mismatch と記録する。"""
        d = str(tmp_path)
        wrong = dict(_REAL_PLACINGS)
        wrong[1], wrong[2] = 3, 4      # 3着を別の馬にすり替える
        _make_result_dbs(d, wrong, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        out = results_mod.attach_for_date(d, '2026-09-27',
                                         results_mod.open_history(d),
                                         results_mod.open_keiba(d))
        assert out['mismatch'] == [self.RID]
        r = archive.load_snapshot(d, '2026-09-27', self.RID)['result']
        assert r['cross_check'] == 'mismatch'

    def test_result_has_no_roi(self, tmp_path):
        """回収率は保存しない（記事の金額は換算例なので賭け金の仮定が要る）。"""
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        _publish(d, self.RID, [{'type': '複勝', 'axis': 7, 'combos': [[7]]}])
        results_mod.attach_for_date(d, '2026-09-27', results_mod.open_history(d),
                                    results_mod.open_keiba(d))
        blob = json.dumps(archive.load_snapshot(d, '2026-09-27',
                                                self.RID)['result'],
                          ensure_ascii=False)
        for word in ('roi', 'ROI', '回収', 'recovered', 'invested'):
            assert word not in blob, f'結果に {word} が入っている'

    def test_marks_and_dangers_get_their_placings(self, tmp_path):
        d = str(tmp_path)
        _make_result_dbs(d, _REAL_PLACINGS, _REAL_DIVS, self.RID)
        snap = archive.build_snapshot(
            {'race_id': self.RID, 'r': 9, 'conf': 70, '_venue': '中山'},
            [{'n': 7, 'rl_rank': 1}],
            [{'num': 7, 'name': 'A', 'mark': '◎'}, {'num': 1, 'name': 'B', 'mark': ''}],
            {'stars': 4, 'verdict': 'strong', 'bets': [], 'allocation_ratio': {},
             'allocation_examples': {}},
            [{'num': 1, 'name': 'B', 'pop': 1}], {}, '見解')
        archive.save_snapshot(d, '2026-09-27', snap)
        results_mod.attach_for_date(d, '2026-09-27', results_mod.open_history(d),
                                    results_mod.open_keiba(d))
        r = archive.load_snapshot(d, '2026-09-27', self.RID)['result']
        assert r['marks'] == [{'mark': '◎', 'num': 7, 'place': 2}]
        assert r['danger_favorites'] == [{'num': 1, 'place': 4}]

    def test_cli_is_noop_on_empty_archive(self, tmp_path):
        """公開実績0件でも失敗しない（ワークフローに入れても無害）。"""
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(BASE, 'scripts',
                                          'attach_product_results.py'),
             '--base-dir', str(tmp_path)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert 'prediction_archive' in r.stderr


# ---------------------------------------------------------------------------
# 根拠の文章を書く印の数は選べる（ユーザー指示 2026-10-05: ◎と○のみでよい）
#
# 🔴 減らすのは**文章だけ**。印・比較表・買い目・危険な人気馬は §9 のフォーマット
#    どおり残す（商品として「印を並べるだけ」にしないという制約がある）。
# ---------------------------------------------------------------------------

class TestCommentMarkCount:
    def _sec(self, race, n):
        return article.build_race_section(race, race['horses'], {},
                                          paid=True, n_comment_marks=n)

    def _race_with_three_marks(self, real_races):
        want = {marks.MARKS[0], marks.MARKS[1], marks.MARKS[2]}
        for race in real_races:
            if not race.get('horses'):
                continue
            got = {h.get('product_mark') for h in marks.assign_marks(race['horses'])}
            if want <= got:
                return race
        pytest.skip('◎○▲ が揃うレースが latest.json に無い')

    def test_default_is_unchanged_and_writes_three(self, real_races):
        race = self._race_with_three_marks(real_races)
        md = self._sec(race, 3)['markdown']
        for mark, label in ((marks.MARKS[0], '本命'), (marks.MARKS[1], '対抗'),
                            (marks.MARKS[2], '単穴')):
            assert f'### {mark} {label}' in md
        # 既定値を渡さない従来の呼び方と完全に同じ出力であること
        plain = article.build_race_section(race, race['horses'], {}, paid=True)
        assert plain['markdown'] == md

    def test_two_drops_only_the_third_comment(self, real_races):
        race = self._race_with_three_marks(real_races)
        md = self._sec(race, 2)['markdown']
        assert f'### {marks.MARKS[0]} 本命' in md
        assert f'### {marks.MARKS[1]} 対抗' in md
        assert f'### {marks.MARKS[2]} 単穴' not in md

    def test_marks_table_and_bets_are_not_reduced(self, real_races):
        race = self._race_with_three_marks(real_races)
        full = self._sec(race, 3)
        trimmed = self._sec(race, 2)
        assert full['rows'] == trimmed['rows']
        assert full['tickets'] == trimmed['tickets']
        assert full['dangers'] == trimmed['dangers']
        md = trimmed['markdown']
        for section in ('### 印', '### 能力・適性・市場の比較', '### 最終結論',
                        '### 買い目'):
            assert section in md
        third = next(r for r in trimmed['rows'] if r['mark'] == marks.MARKS[2])
        assert f"{marks.MARKS[2]} {third['num']}" in md

    def test_out_of_range_falls_back_to_all(self, real_races):
        race = self._race_with_three_marks(real_races)
        for bad in (0, -1, 9, None, 'x'):
            md = self._sec(race, bad)['markdown']
            assert f'### {marks.MARKS[0]} 本命' in md

    def test_trimmed_article_has_no_forbidden_expressions(self, real_races):
        for race in real_races:
            if not race.get('horses'):
                continue
            md = self._sec(race, 2)['markdown']
            assert commentary.find_forbidden(md) is None

    def test_cli_option_trims_the_third_comment(self):
        """CLI を実際に起動して確認する（配線漏れを見逃さないため）。"""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'data'))
            with open(LATEST, encoding='utf-8') as f:
                raw = f.read()
            with open(os.path.join(d, 'data', 'latest.json'), 'w',
                      encoding='utf-8') as f:
                f.write(raw)
            out = os.path.join(d, 'out.md')
            r = subprocess.run(
                [sys.executable, os.path.join(BASE, 'scripts',
                                              'build_product_article.py'),
                 '--base-dir', d, '--comment-marks', '2', '--out', out],
                capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            text = open(out, encoding='utf-8').read()
            assert f'### {marks.MARKS[0]} 本命' in text
            assert f'### {marks.MARKS[2]} 単穴' not in text
            assert '### 買い目' in text


# ---------------------------------------------------------------------------
# 記事を Google ドキュメントに変換するための HTML 化
#
# 🔴 汎用 Markdown ではなく、`article.py` が実際に出す構造だけを対象にする。
#    週ごとに手で HTML を組むと見出しが段落になったり表が崩れるので、
#    変換を1箇所に固定してここで押さえる。
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(BASE, 'scripts'))
import md_to_html as mdh  # noqa: E402


class TestMarkdownToHtml:
    def test_headings_become_h1_h2_h3(self):
        out = mdh.md_to_html('# 題\n\n## レース\n\n### 印\n')
        assert '<h1>題</h1>' in out
        assert '<h2>レース</h2>' in out
        assert '<h3>印</h3>' in out

    def test_table_becomes_table_with_header_row(self):
        md = '| 印 | 馬番 |\n|---|---:|\n| ◎ | 6 |\n'
        out = mdh.md_to_html(md)
        assert '<table' in out and '</table>' in out
        assert '<th>印</th>' in out
        assert '<td>6</td>' in out
        assert '|---|' not in out

    def test_header_row_is_wrapped_in_thead(self):
        """🔴 `<thead>` が無いと Drive が空の見出し行を1行足す（実機で確認）。"""
        out = mdh.md_to_html('| 印 | 馬番 |\n|---|---:|\n| ◎ | 6 |\n')
        assert '<thead><tr><th>印</th>' in out
        assert '<tbody>' in out and '</tbody>' in out

    def test_horizontal_rule_is_dropped(self):
        """🔴 `<hr>` の直後が見出しだと Drive が `-----見出し` に潰す（実機で確認）。"""
        out = mdh.md_to_html('---\n\n## 京都9R\n')
        assert '<hr' not in out
        assert '<h2>京都9R</h2>' in out
        assert '-----' not in out

    def test_bold_and_italic(self):
        out = mdh.md_to_html('**強調**と*斜体*。\n')
        assert '<b>強調</b>' in out
        assert '<i>斜体</i>' in out

    def test_bullets_become_ul(self):
        out = mdh.md_to_html('- 一つ\n- 二つ\n')
        assert out.count('<li>') == 2
        assert '<ul>' in out

    def test_renders_source_links(self):
        """web見解の `[出典](url)` を生のまま出さない（2026-10-07 追加）。"""
        out = mdh.md_to_html(
            '出典：[1](https://a.example/x?y=1&z=2) / [出典](https://b.example/p)')
        assert '<a href="https://a.example/x?y=1&amp;z=2">1</a>' in out
        assert '<a href="https://b.example/p">出典</a>' in out
        assert '[1](' not in out
        assert '&amp;amp;' not in out      # & を二重にエスケープしない

    def test_does_not_linkify_non_http_schemes(self):
        out = mdh.md_to_html('[x](javascript:alert(1))')
        assert '<a ' not in out

    def test_html_is_escaped(self):
        out = mdh.md_to_html('a < b & c > d\n')
        assert '&lt;' in out and '&amp;' in out
        assert '<p>a &lt; b &amp; c &gt; d</p>' in out

    def test_real_article_leaves_no_raw_markdown(self, real_races):
        """実データの記事を変換して、記法が生のまま残らないこと。"""
        secs = []
        for race in real_races[:6]:
            if not race.get('horses'):
                continue
            secs.append(article.build_race_section(
                race, race['horses'], {}, paid=True, n_comment_marks=2))
        if not secs:
            pytest.skip('レースが無い')
        md = article.build_article('2026-01-01', secs, paid=True)
        out = mdh.md_to_html(md)
        for leftover in ('|---', '**', '### ', '## '):
            assert leftover not in out, f'生の記法が残っている: {leftover}'
        assert '<table' in out and '<h2>' in out and '<h3>' in out


# ---------------------------------------------------------------------------
# 対象日は race_id の先頭8桁から取る
#
# 🔴 2026-10-05 にリハーサルで発覚した実害: 日曜の記事に土曜の日付が載っていた。
#    `date` は `10月4日(日)`（年が無い表示用）、`generated_at` は前夜（土 17:57 に
#    日曜ぶんを生成）なので、どちらからも対象日は決まらない。
#    公開記録ではこのズレが `assert_publishable` の誤判定にもなる。
# ---------------------------------------------------------------------------

import build_product_article as bpa  # noqa: E402


class TestTargetDate:
    def test_race_id_wins_over_generated_at_and_display_date(self):
        data = {
            'date': '10月4日(日)',
            'generated_at': '2026-10-03T17:57:03.773859+09:00',
            'races': {'東京': [{'race_id': '20261004_05_01'}],
                      '京都': [{'race_id': '20261004_08_01'}]},
        }
        assert bpa._target_date(data) == '2026-10-04'

    def test_majority_wins_when_race_ids_disagree(self):
        data = {'races': {'A': [{'race_id': '20261004_05_01'},
                                {'race_id': '20261004_05_02'}],
                          'B': [{'race_id': '20261003_08_01'}]}}
        assert bpa._target_date(data) == '2026-10-04'

    def test_falls_back_when_no_race_id(self):
        data = {'generated_at': '2026-10-03T17:57:00+09:00', 'races': {}}
        assert bpa._target_date(data) == '2026-10-03'

    def test_display_date_string_without_year_is_not_used(self):
        """`10月4日(日)` は10文字以上あるが年が無い。日付として使わない。"""
        data = {'date': '10月4日(日)', 'races': {'A': [{'race_id': '20261004_05_01'}]}}
        got = bpa._target_date(data)
        assert got == '2026-10-04'
        assert not got.startswith('10月')

    def test_real_latest_json_matches_its_race_ids(self, real_races):
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        got = bpa._target_date(data)
        rid = next(r['race_id'] for r in real_races if r.get('race_id'))
        assert got == f'{rid[:4]}-{rid[4:6]}-{rid[6:8]}'

    def test_article_title_uses_the_race_date(self, real_races):
        """記事の見出しの日付が race_id と一致すること（読者に見える値）。"""
        secs = [article.build_race_section(real_races[0],
                                           real_races[0]['horses'], {},
                                           paid=True, n_comment_marks=2)]
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        text = article.build_article(bpa._target_date(data), secs, paid=True)
        rid = next(r['race_id'] for r in real_races if r.get('race_id'))
        assert text.splitlines()[0].startswith(
            f'# {rid[:4]}-{rid[4:6]}-{rid[6:8]} ')


# ---------------------------------------------------------------------------
# 根拠（2026-10-15 ユーザー指示「軸に選ぶ馬の根拠と危険な人気馬の根拠は
# 読者が一番欲しい情報」）。盛らない形で書けていることを固定する。
# ---------------------------------------------------------------------------

class TestAxisAndDangerBasis:
    """◎ と 危険な人気馬 の根拠。文面そのものを固定する。"""

    def _danger_axis_race(self):
        """◎ 自身が危険な人気馬に当たる形（本番11件中9件がこれだった）。"""
        race = {'_venue': '京都', 'r': 9, 'name': 'テスト特別',
                'dist': '芝1600m', 'conf': 70, 'cmt': ''}
        horses = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 1, 'solo_rank': 7,
             'fit_rank': 10, 'odds': 3.6},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 1,
             'fit_rank': 2, 'odds': 4.2},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 3, 'solo_rank': 2,
             'fit_rank': 3, 'odds': 9.0},
        ])
        return race, horses

    # --- ◎ の根拠 --------------------------------------------------------
    def test_axis_section_states_why_it_is_the_axis(self):
        race, horses = self._danger_axis_race()
        md = article.build_race_section(race, horses, {}, paid=True)['markdown']
        assert '◎にした根拠' in md
        assert 'レース信頼度（このレースは70）' in md

    def test_only_the_axis_gets_the_basis(self):
        """○▲ に同じ根拠文を繰り返さない（◎ の根拠だから1回だけ）。"""
        race, horses = self._danger_axis_race()
        md = article.build_race_section(race, horses, {}, paid=True)['markdown']
        assert md.count('◎にした根拠') == 1

    def test_axis_basis_does_not_overstate(self):
        """市場に勝っていない部分を隠さない。禁止語も出さない。"""
        text = commentary.axis_basis(66)
        assert '回収率が100%を超える根拠はありません' in text
        assert '市場＋0.0004' in text
        for word in ('勝てる', '儲か', '必ず', '確実'):
            assert word not in text
        assert commentary.find_forbidden(text) is None

    def test_axis_basis_does_not_sell_conf_as_a_true_probability(self):
        """conf は RL1 だけ ECE 0.1018 で過信。確率として売らない。"""
        text = commentary.axis_basis(80)
        assert '過信' in text
        assert '実測0.566' in text

    # --- この印（旧「危険な人気馬」）の根拠 ---------------------------------
    def test_danger_basis_says_half_still_come_in(self):
        """🔴 実測は 51.4% / 48.6%。「来ない」と書いたら測っていないことを書く。"""
        assert 'それでも半分近くは3着内に来ます' in commentary.DANGER_BASIS
        assert '51.4%' in commentary.DANGER_BASIS
        assert '48.6%' in commentary.DANGER_BASIS       # B期間6,077レースの実測
        assert '242頭' in commentary.DANGER_BASIS
        assert '849頭' in commentary.DANGER_BASIS
        assert commentary.find_forbidden(commentary.DANGER_BASIS) is None

    def test_danger_basis_does_not_claim_the_threshold_can_be_tightened(self):
        """🔴 2026-10-16: 条件を厳しくしても書ける水準にならなかった（6条件とも不採用）。

        「もっと絞れば危険と書ける」と読める文を商品に出さないための歯止め。
        """
        assert 'なりませんでした' in commentary.DANGER_BASIS

    def test_danger_basis_states_the_filter_value_without_overclaiming(self):
        """避ける価値 +1.6pt は実測。ただし100%未満であることを必ず併記する。"""
        assert '84.2%' in commentary.DANGER_BASIS
        assert '100%には届きません' in commentary.DANGER_BASIS

    def test_danger_basis_says_it_is_not_about_winning_either(self):
        """🔴 2026-10-16: 1着でも測って不採用（`danger/RESULTS_WIN.md`）。

        V0 の勝率は 23.0%（1番人気全体 32.2%）＝4頭に1頭は勝つ。
        事前登録の合格線（「来ない」10%未満 / 「勝ちきれない」16%未満）に
        N>=300 の条件は1つも届かなかった。読者が「1着には来ない印」と
        誤読しないよう、勝率を必ず併記する。
        """
        assert '1着に来ないという意味でもありません' in commentary.DANGER_BASIS
        assert '23.0%' in commentary.DANGER_BASIS      # 該当849頭の勝率
        assert '32.2%' in commentary.DANGER_BASIS      # 1番人気全体の勝率
        assert '4頭に1頭は勝っています' in commentary.DANGER_BASIS
        assert commentary.find_forbidden(commentary.DANGER_BASIS) is None

    def test_danger_basis_never_says_the_horse_will_not_win(self):
        """🔴 測定が支持しない断定を商品に出さないための歯止め。"""
        for banned in ('1着に来ません', '1着には来ません', '勝てません',
                       '勝つことはありません', '1着はありません'):
            assert banned not in commentary.DANGER_BASIS

    def test_danger_thresholds_are_unchanged_after_two_measurements(self):
        """🔴 3着内（6条件）と1着（6条件）で測って、どちらも不採用。

        「もっと絞れば書ける」を測定が支持しなかったので閾値をいじる根拠が無い。
        値が動いたら落とす（結果を見てから閾値を変えないため）。
        """
        assert marks.DANGER_MAX_POP == 1
        assert marks.DANGER_MIN_SOLO == 6

    def test_article_does_not_call_them_dangerous_favorites(self):
        """🔴 カード §7 の事前決定。断定的な名前をやめ事実の提示に留める。

        3着内率 48.6%（半分近く来る）で「危険」と書く根拠が無いため、
        見出しが「危険な人気馬」に戻ったら落とす。
        """
        race, horses = self._danger_axis_race()
        md = article.build_race_section(race, horses, {}, paid=True)['markdown']
        assert '危険な人気馬' not in md
        assert '市場の支持ほどAIは評価していない馬' in md

    def test_danger_is_axis_note_cites_the_large_sample(self):
        """🔴 ◎自身が 88.2%（849頭中749頭）は B期間で測った構造。N=11 ではない。"""
        assert '88.2%' in commentary.DANGER_IS_AXIS
        assert '強い警告としては読めません' in commentary.DANGER_IS_AXIS

    def test_danger_section_shows_the_three_ranks(self):
        race, horses = self._danger_axis_race()
        md = article.build_race_section(race, horses, {}, paid=True)['markdown']
        assert '市場を見ないAI単独の評価は7位' in md
        assert 'AI総合（市場込み）では1位' in md
        assert '今回条件への適性は10位' in md

    def test_danger_section_says_it_is_the_same_horse_as_the_axis(self):
        """🔴 B期間849頭中749頭(88.2%)が ◎ 自身。黙っていると同じ馬で印が並ぶ。"""
        race, horses = self._danger_axis_race()
        md = article.build_race_section(race, horses, {}, paid=True)['markdown']
        assert 'この馬は本記事の◎と同じ馬です' in md

    def test_danger_section_omits_the_axis_note_for_other_horses(self):
        """◎ ではない馬には「◎と同じ馬」と書かない。"""
        race = {'_venue': '京都', 'r': 9, 'name': 'T', 'dist': '芝1600m',
                'conf': 70, 'cmt': ''}
        horses = marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 3, 'solo_rank': 1,
             'odds': 3.6},
            {'n': 2, 'name': 'B', 'rl_rank': 4, 'pop': 1, 'solo_rank': 8,
             'odds': 2.2},
            {'n': 3, 'name': 'C', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2,
             'odds': 6.0},
            {'n': 4, 'name': 'D', 'rl_rank': 3, 'pop': 4, 'solo_rank': 3,
             'odds': 9.0},
        ])
        sec = article.build_race_section(race, horses, {}, paid=True)
        assert [d['num'] for d in sec['dangers']] == [2]
        assert 'この馬は本記事の◎と同じ馬です' not in sec['markdown']

    def test_race_comment_does_not_repeat_the_axis_danger_three_times(self):
        """◎ 自身の分は skip_reason / downgrade_reason が説明済み。"""
        race, horses = self._danger_axis_race()
        tk = tickets.build_tickets(race, horses)
        assert tk.get('axis_is_danger_favorite') is True
        text = commentary.race_comment(race, horses, tk, marks.danger_favorites(horses))
        assert 'は市場が上位人気に推しているが' not in text

    # --- アプリコメントの持ち込み ----------------------------------------
    def test_cmt_drops_expected_value_claims(self):
        """EV で買い目を選ぶ方向は3回否定済み。「期待値十分」を商品に出さない。"""
        got = commentary._clean_cmt('馬場はフラット。中穴(3.6倍)で期待値十分。')
        assert '馬場はフラット' in got
        assert '期待値' not in got

    def test_real_article_does_not_claim_expected_value(self, real_races):
        for race in real_races:
            horses = race.get('horses') or []
            if not horses:
                continue
            md = article.build_race_section(race, horses, {}, paid=True)['markdown']
            assert '期待値十分' not in md
