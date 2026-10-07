"""web見解の別枠セクション（`src/product/web_block.py`）のテスト。

North Star #6 に従い、手打ちの理想的な dict ではなく **本番と同じ形**で検証する:
`data/latest.json` の実レース（10R/11R）と `claude_web.build_claude_view()` の
実際の戻り値を通す。

🔴 このテストの主目的は「web見解を足しても印・比較表・買い目が1つも変わらない」を
   コードで固定すること。撤回された後付け層5件と同じ形にならないための歯止め。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import claude_web  # noqa: E402
from src.product import article, commentary, web_block  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEST = os.path.join(BASE, 'data', 'latest.json')


def _real_main_race():
    """latest.json から web見解の対象（10R/11R）を1つ返す。"""
    with open(LATEST, encoding='utf-8') as f:
        d = json.load(f)
    for venue, rs in (d.get('races') or {}).items():
        for r in rs:
            if r.get('r') in claude_web.WEB_RACE_NUMS and (r.get('horses') or []):
                r = dict(r)
                r['_venue'] = venue
                return r
    pytest.skip('latest.json に 10R/11R が無い')


def _findings_for(race, *, with_sources=True, extra_horses=None):
    """その実レースの馬番を使って findings を組む（存在しない馬番を書かない）。"""
    horses = race['horses']
    nums = [h['num'] if 'num' in h else h['n'] for h in horses]
    note = {'note': '内が伸びる馬場。前々で運べる形が有利',
            'direction': 'up'}
    if with_sources:
        note['sources'] = ['https://www.bloodline-trackbias.work/entry/test']
    hs = {str(nums[0]): note}
    if extra_horses:
        hs.update(extra_horses)
    return {
        'schema_version': claude_web.SCHEMA_VERSION,
        'date': str(race['race_id'])[:8],
        'collected_at': '2026-10-10T08:50:00+09:00',
        'races': {
            race['race_id']: {
                'going': {'surface': '芝', 'state': '良', 'expected': 'やや重'},
                'summary': '内枠の先行勢を上に見る',
                'horses': hs,
                'sources': ['https://www.bloodline-trackbias.work/entry/test'],
            }
        },
    }


def _view(race, findings):
    # latest.json の馬は馬番キーが `n`。claude_web はスクレイパー側の `num` を
    # 前提にしているので、本番（build_product_article.py）と同じ形で渡し替える。
    return claude_web.build_claude_view(
        race['race_id'], race['r'],
        [{'num': h.get('n')} for h in race['horses']], findings,
        race_date=str(race['race_id'])[:8])


# ---------------------------------------------------------------------------
# 🔴 最重要: web見解は印・比較表・買い目を1つも変えない
# ---------------------------------------------------------------------------

class TestItDoesNotChangeThePrediction:

    def test_web_notes_do_not_change_marks_or_tickets(self):
        """web見解あり／なしで、印・比較表・買い目・危険な人気馬が完全一致する。"""
        race = _real_main_race()
        view = _view(race, _findings_for(race))
        assert view is not None, 'テストの前提（見解が組めること）が崩れている'

        without = article.build_race_section(race, race['horses'], {})
        with_web = article.build_race_section(race, race['horses'], {},
                                              web_view=view)

        assert with_web['marked'] == without['marked']
        assert with_web['rows'] == without['rows']
        assert with_web['tickets'] == without['tickets']
        assert with_web['dangers'] == without['dangers']

    def test_markdown_differs_only_by_the_appended_block(self):
        """本文の差分は末尾に足した別枠だけ（上の節を書き換えていない）。"""
        race = _real_main_race()
        view = _view(race, _findings_for(race))
        without = article.build_race_section(race, race['horses'], {})['markdown']
        with_web = article.build_race_section(race, race['horses'], {},
                                              web_view=view)['markdown']
        assert with_web.startswith(without), 'web見解が既存の本文を書き換えている'
        assert '🌐 web情報' in with_web[len(without):]

    def test_none_view_is_byte_identical_to_before(self):
        race = _real_main_race()
        a = article.build_race_section(race, race['horses'], {})['markdown']
        b = article.build_race_section(race, race['horses'], {},
                                       web_view=None)['markdown']
        assert a == b

    def test_web_block_never_receives_tickets_or_probabilities(self):
        """`build_web_block` が印・確率・買い目を触っていないことをソースで固定。

        docstring では各名前に言及するので、**呼び出しの形**で判定する。
        """
        src = open(os.path.join(BASE, 'src', 'product', 'article.py'),
                   encoding='utf-8').read()
        assert 'build_web_block(web_view, marked)' in src
        wb = open(os.path.join(BASE, 'src', 'product', 'web_block.py'),
                  encoding='utf-8').read()
        code = '\n'.join(ln for ln in wb.splitlines()
                         if not ln.lstrip().startswith('#'))
        code = code.split('"""', 2)[-1]        # モジュール docstring を除く
        for bad in ('import', 'build_tickets(', 'assign_marks(', 'rank_table(',
                    'win_prob', 'cal_prob', 'rl_rank', 'tickets', 'marks'):
            assert bad not in code, f'web_block が {bad} を触っている'
        # 読んでよいキーだけを使っている
        for ok in ("get('going')", "get('summary')", "get('horse_notes')",
                   "get('sources')"):
            assert ok in code


# ---------------------------------------------------------------------------
# 載せるもの・載せないもの
# ---------------------------------------------------------------------------

class TestWhatItShows:

    def test_shows_going_summary_and_sources(self):
        race = _real_main_race()
        md = '\n'.join(web_block.build_web_block(
            _view(race, _findings_for(race)), race['horses']))
        assert '馬場' in md and '良' in md
        assert '内枠の先行勢' in md
        assert 'bloodline-trackbias.work' in md
        assert 'AIの評価には反映していません' in md

    def test_note_without_sources_is_dropped(self):
        """出典の無いメモは載せない（裏づけの無い文を商品に出さない）。"""
        race = _real_main_race()
        md = '\n'.join(web_block.build_web_block(
            _view(race, _findings_for(race, with_sources=False)),
            race['horses']))
        assert '内が伸びる馬場' not in md
        assert '掲載していません' in md

    def test_web_order_and_marks_are_not_printed(self):
        """web の序列・印は記事に出さない（AIの印と competing させない）。"""
        race = _real_main_race()
        f = _findings_for(race)
        nums = [h.get('num', h.get('n')) for h in race['horses']]
        f['races'][race['race_id']]['order'] = nums[:3]
        f['races'][race['race_id']]['marks'] = {str(nums[0]): '◎',
                                                str(nums[1]): '○'}
        md = '\n'.join(web_block.build_web_block(_view(race, f), race['horses']))
        assert '◎' not in md and '○' not in md
        assert '序列' not in md

    def test_empty_view_yields_nothing(self):
        assert web_block.build_web_block(None, []) == []
        assert web_block.build_web_block({'going': None, 'summary': '',
                                          'horse_notes': {}}, []) == []

    def test_long_note_is_truncated(self):
        race = _real_main_race()
        nums = [h.get('num', h.get('n')) for h in race['horses']]
        f = _findings_for(race, extra_horses={
            str(nums[1]): {'note': 'あ' * 500,
                           'sources': ['https://example.com/a']}})
        md = '\n'.join(web_block.build_web_block(_view(race, f), race['horses']))
        assert 'あ' * (web_block.MAX_NOTE_LEN + 1) not in md
        assert '…' in md


# ---------------------------------------------------------------------------
# 古い見解を使い回さない
# ---------------------------------------------------------------------------

class TestStaleFindings:

    def test_other_days_findings_are_refused(self):
        """date が対象日と違えば見解を使わない（前の開催の印を今日出さない）。"""
        race = _real_main_race()
        f = _findings_for(race)
        f['date'] = '20200101'
        assert claude_web.build_claude_view(
            race['race_id'], race['r'], race['horses'], f,
            race_date=str(race['race_id'])[:8]) is None

    def test_non_main_race_is_refused(self):
        """10R/11R 以外には web見解を出さない（対象を広げていないこと）。"""
        race = _real_main_race()
        f = _findings_for(race)
        assert claude_web.build_claude_view(
            race['race_id'], 3, race['horses'], f,
            race_date=str(race['race_id'])[:8]) is None


# ---------------------------------------------------------------------------
# commentary 側のガードは弱めていない
# ---------------------------------------------------------------------------

class TestGeneratedCommentaryGuardIsIntact:
    """`commentary.FORBIDDEN_PATTERNS` は1文字も緩めていない。

    web_block が扱うのは**出典つきの引用**で、`commentary.py` が禁じている
    「取得経路が無いのに生成する文」ではない。両者を混同しないための固定。
    """

    def test_commentary_still_rejects_training_words(self):
        for word in ('陣営', '調教', '追い切り', 'パドック', '坂路'):
            assert commentary.find_forbidden(f'{word}の評価が高い') is not None

    def test_web_block_requires_a_source_instead(self):
        wb = open(os.path.join(BASE, 'src', 'product', 'web_block.py'),
                  encoding='utf-8').read()
        assert '出典' in wb
        # 生成文のガードを web_block から呼んでいない（別の種類の文なので）
        assert 'find_forbidden' not in wb
        # 代わりの歯止めは「出典の無いメモは落とす」
        assert 'dropped += 1' in wb


# ---------------------------------------------------------------------------
# 2026-10-07 に実データで見つけた表示の欠陥2件（記事を自動生成する前に潰した）
# ---------------------------------------------------------------------------

class TestArticleDisplayDefects:

    def test_sub_marks_are_separated(self):
        """△が複数いても馬番が連結されない（14番と7番が「147」に見えていた）。"""
        from src.product import marks
        rows = marks.rank_table(marks.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 1, 'solo_rank': 1, 'odds': 2.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 4.0},
            {'n': 3, 'name': 'C', 'rl_rank': 3, 'pop': 3, 'solo_rank': 3, 'odds': 6.0},
            {'n': 14, 'name': 'D', 'rl_rank': 4, 'pop': 4, 'solo_rank': 4, 'odds': 9.0},
            {'n': 7, 'name': 'E', 'rl_rank': 5, 'pop': 5, 'solo_rank': 5, 'odds': 14.0},
        ]))
        line = article._marks_line(rows)
        sub = [ln for ln in line.splitlines() if ln.startswith('△')]
        assert sub, '△の行が無い'
        assert '147' not in sub[0], f'馬番が連結されている: {sub[0]}'
        assert '14D' in sub[0] and '7E' in sub[0]

    def test_skip_reason_is_not_duplicated_and_ends_with_a_period(self):
        """見送り理由が信頼度を名指しする場合、信頼度の文を二重に出さない。"""
        from src.product import marks as _m, tickets as _t
        horses = _m.assign_marks([
            {'n': 1, 'name': 'A', 'rl_rank': 1, 'pop': 1, 'solo_rank': 1, 'odds': 2.0},
            {'n': 2, 'name': 'B', 'rl_rank': 2, 'pop': 2, 'solo_rank': 2, 'odds': 4.0},
        ])
        race = {'conf': 52, 'cmt': ''}
        t = _t.build_tickets(race, horses)
        assert t['verdict'] == 'skip'
        txt = commentary.race_comment(race, horses, t, [])
        assert txt.count('レース信頼度') == 1, txt
        assert txt.rstrip().endswith('。'), txt
