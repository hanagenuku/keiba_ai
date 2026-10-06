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

    def test_strong_explains_both_types(self):
        t = tickets.build_tickets({'conf': 70}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '複勝を本線' in joined
        assert 'ワイドは押さえ' in joined
        assert '馬連' in joined and '三連複' in joined   # 除外した券種の理由も書く

    def test_light_explains_why_only_one_type(self):
        t = tickets.build_tickets({'conf': 60}, self._horses())
        joined = '\n'.join(t['bet_type_reasons'])
        assert '複勝を本線' in joined
        assert '今回は複勝だけ' in joined
        assert 'ワイドは押さえ' not in joined

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
        """券種1つなら自明な比率表を載せず、各自調整だけを書く。"""
        md = article._bets_md(tickets.build_tickets({'conf': 60}, self._horses()))
        assert '各自調整' in md
        assert '換算例' not in md
        assert '配分の目安' not in md

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
        note = commentary.sources_note(has_web_notes=False)
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
