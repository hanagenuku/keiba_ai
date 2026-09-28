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

from src.product import archive, article, commentary, facts, marks, tickets  # noqa: E402

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

    def test_at_most_two_bet_types(self):
        for conf in (30, 50, 58, 66, 80):
            t = tickets.build_tickets({'conf': conf}, self._horses())
            assert tickets.bet_type_count(t) <= tickets.MAX_BET_TYPES

    def test_never_uses_high_takeout_types(self):
        """三連単・三連複・馬連・単勝は標準構成に入れない。"""
        banned = {'三連単', '三連複', '馬連', '馬単', '枠連', '単勝'}
        for conf in (58, 66, 80):
            t = tickets.build_tickets({'conf': conf}, self._horses())
            assert not ({b['type'] for b in t['bets']} & banned)

    def test_low_confidence_is_skip(self):
        t = tickets.build_tickets({'conf': 40}, self._horses())
        assert t['verdict'] == 'skip'
        assert t['bets'] == []
        assert t['skip_reason']

    def test_missing_conf_is_skip(self):
        t = tickets.build_tickets({'conf': None}, self._horses())
        assert t['verdict'] == 'skip'

    def test_strong_gets_two_types(self):
        t = tickets.build_tickets({'conf': 70}, self._horses())
        assert t['verdict'] == 'strong'
        assert {b['type'] for b in t['bets']} == {'複勝', 'ワイド'}

    def test_light_gets_one_type(self):
        t = tickets.build_tickets({'conf': 60}, self._horses())
        assert t['verdict'] == 'light'
        assert {b['type'] for b in t['bets']} == {'複勝'}

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
        t = tickets.build_tickets({'conf': 70}, self._horses())
        for amt, detail in t['allocation_examples'].items():
            assert sum(detail.values()) == amt


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
        note = commentary.sources_note(has_web_notes=False)
        assert 'JRA公式' in note
        # 使っていないことを明示する文なので、ここでは禁止語チェックを通さない
        assert '使用していません' in note


class TestThresholdsAreNotInvented:
    """商品の閾値が「本番で既に使われている値」と一致していること。

    🔴 ここが食い違うと、商品側で勝手に閾値を最適化したのと同じになる。
    """

    def test_conf_light_matches_production_gate(self):
        from src.betting import ev_filter
        assert tickets.CONF_LIGHT == int(round(ev_filter.MIN_AXIS_FUKU_PROB * 100))

    def test_conf_strong_matches_rec_flag_on_real_data(self, real_races):
        """conf >= CONF_STRONG が latest.json の rec と一致すること。"""
        mism = [
            (r['_venue'], r.get('r'), r.get('conf'), r.get('rec'))
            for r in real_races
            if r.get('conf') is not None
            and bool(r.get('rec')) != (r['conf'] >= tickets.CONF_STRONG)
        ]
        assert not mism, f'rec と conf>={tickets.CONF_STRONG} が食い違う: {mism}'
