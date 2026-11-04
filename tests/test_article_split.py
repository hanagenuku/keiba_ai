"""有料記事／無料記事の振り分けと S/A 表示のテスト（2026-11-04 のユーザー改正）。

ユーザー指示:
    「有料記事は10レースから11レースという縛りを無くします。純粋にAIが推奨する
      レースのみにします。推奨レースの中でも自信のあるレースをS、次点をAとして表示。
      10レース11レースの推奨漏れ分に関しては、無料予想として記事にします」
    確定（同日）:
    「9〜11の内推奨レースがあればそれは有料。推奨が無ければ無料。
      その他平場の推奨レースは有料。推奨外は記事の必要なし」

North Star #6 に従い、手打ちの理想的な dict ではなく **本番の `data/latest.json`**
をそのまま読んで検証する。
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))

from src.product import article, scope, tickets  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEST = os.path.join(BASE, 'data', 'latest.json')


def _real_races():
    if not os.path.exists(LATEST):
        pytest.skip('data/latest.json が無い')
    with open(LATEST, encoding='utf-8') as f:
        d = json.load(f)
    out = []
    for venue, rs in (d.get('races') or {}).items():
        for r in rs:
            r = dict(r)
            r['_venue'] = venue
            out.append(r)
    if not out:
        pytest.skip('latest.json にレースが無い')
    return out


class TestArticleKind:
    """どちらの記事に載せるか。"""

    def test_recommended_race_is_paid_regardless_of_race_number(self):
        # 🔴 改正の核心: 9〜11R という縛りを外した。平場でも推奨なら有料。
        for r in ({'r': 11, 'conf': 80}, {'r': 3, 'conf': 66},
                  {'r': 1, 'conf': 99}, {'r': 12, 'conf': 70}):
            kind, why = scope.article_kind(r)
            assert kind == scope.PAID, (r, why)

    def test_main_race_without_recommendation_is_free(self):
        for r in ({'r': 9, 'conf': 65}, {'r': 10, 'conf': 0},
                  {'r': 11, 'conf': None}):
            kind, why = scope.article_kind(r)
            assert kind == scope.FREE, (r, why)

    def test_flat_race_without_recommendation_is_not_published(self):
        for r in ({'r': 1, 'conf': 65}, {'r': 8, 'conf': None},
                  {'r': 12, 'conf': 55}):
            kind, why = scope.article_kind(r)
            assert kind is None, (r, why)

    def test_boundary_is_exactly_conf_strong(self):
        assert scope.article_kind({'r': 5, 'conf': tickets.CONF_STRONG})[0] == scope.PAID
        assert scope.article_kind({'r': 5, 'conf': tickets.CONF_STRONG - 1})[0] is None
        assert scope.article_kind({'r': 9, 'conf': tickets.CONF_STRONG - 1})[0] == scope.FREE

    def test_in_scope_still_means_the_same_set(self):
        """🔑 改正で `in_scope` の意味を変えていないこと。

        改正前 = 9〜11R または conf>=66
        改正後 = 有料 ∪ 無料
        この2つが**実データで1件も食い違わない**ことを固定する。
        ずれると `danger_log` が残してきた `in_scope` の記録が、過去と
        今後で別の意味になり、台帳が静かに壊れる。
        """
        races = _real_races()
        for r in races:
            union = scope.article_kind(r)[0] is not None
            assert scope.in_scope(r)[0] == union, r.get('race_id')

    def test_every_real_race_lands_in_exactly_one_place(self):
        races = _real_races()
        kinds = [scope.article_kind(r)[0] for r in races]
        assert len(kinds) == len(races)
        assert set(kinds) <= {scope.PAID, scope.FREE, None}
        # 無料に入ったものは必ず 9〜11R かつ推奨外
        for r, k in zip(races, kinds):
            if k == scope.FREE:
                assert scope.is_main_race(r)
                assert (r.get('conf') or 0) < tickets.CONF_STRONG
            if k == scope.PAID:
                assert r.get('conf') is not None
                assert r['conf'] >= tickets.CONF_STRONG


class TestGrade:
    """S / A の格付け。"""

    def test_grade_boundaries(self):
        assert tickets.grade(tickets.CONF_GRADE_S) == 'S'
        assert tickets.grade(tickets.CONF_GRADE_S + 10) == 'S'
        assert tickets.grade(tickets.CONF_GRADE_S - 1) == 'A'
        assert tickets.grade(tickets.CONF_STRONG) == 'A'
        assert tickets.grade(tickets.CONF_STRONG - 1) is None
        assert tickets.grade(None) is None

    def test_grade_s_threshold_is_the_existing_five_star_boundary(self):
        """🔴 S の境界は**新しく作った閾値ではない**。

        72 は `confidence_stars()` が 2026-09-28 の実装当初から持っていた
        ★5の境界で、2026-11-04 の S/A 導入ではそこに名前を付けただけ。
        値を動かすのは「結果を見てから閾値を当てにいく」形なので、
        このプロジェクトで何度も事故になっている（2026-07-27 / 08-15 / 08-20）。

        ⚠ `confidence_stars(CONF_GRADE_S) == 5` だけでは**何も固定できない**
          （confidence_stars 自身が CONF_GRADE_S を使っているので、値を
          いくつに変えても通る自己参照のテストになる）。実際 2026-11-04 に
          72→70 と動かして素通りすることを確認したので、**リテラルを直接**
          固定する。
        """
        assert tickets.CONF_GRADE_S == 72
        assert tickets.confidence_stars(72) == 5
        assert tickets.confidence_stars(71) == 4
        assert tickets.grade(72) == 'S'
        assert tickets.grade(71) == 'A'
        src = open(os.path.join(BASE, 'src', 'product', 'tickets.py'),
                   encoding='utf-8').read()
        assert '2026-11-04' in src, 'tickets.py に S/A の決定が記録されていない'

    def test_grade_boundary_is_not_a_new_threshold(self):
        """推奨の境界は CONF_STRONG ひとつだけ（別の値を作っていない）。"""
        lowest_graded = min(c for c in range(0, 100) if tickets.grade(c))
        assert lowest_graded == tickets.CONF_STRONG


class TestGradeIndex:
    def _sec(self, grade, conf, heading):
        return {'markdown': '## x', 'grade': grade, 'conf': conf,
                'heading': heading}

    def test_lists_s_before_a_and_counts_them(self):
        secs = [self._sec('A', 67, '東京8R'), self._sec('S', 74, '東京5R'),
                self._sec('S', 76, '東京3R'), self._sec('A', 71, '京都10R')]
        md = '\n'.join(article.grade_index(secs))
        assert 'S 2本 / A 2本' in md
        assert md.index('東京3R') < md.index('東京5R') < md.index('京都10R')

    def test_says_s_is_not_a_profit_claim(self):
        """🔴 S を「当たる」「儲かる」と読ませない但し書きを必ず付ける。"""
        md = '\n'.join(article.grade_index([self._sec('S', 80, '東京1R')]))
        assert 'S だから回収率が100%を超えるという根拠はありません' in md

    def test_empty_when_nothing_is_recommended(self):
        assert article.grade_index([self._sec(None, 54, '東京9R')]) == []
        assert article.grade_index([]) == []


class TestArticleText:
    def test_paid_article_shows_the_index(self):
        secs = [{'markdown': '## 東京3R', 'grade': 'S', 'conf': 76,
                 'heading': '東京3R　2歳未勝利'}]
        md = article.build_article('2026-11-01', secs, paid=True)
        assert '本日の推奨レース（S 1本 / A 0本）' in md
        assert 'AIが推奨したレースだけをまとめています' in md

    def test_free_article_says_these_are_not_recommended(self):
        """🔴 無料版を「お得な版」と誤読させない。"""
        secs = [{'markdown': '## 東京9R', 'grade': None, 'conf': 54,
                 'heading': '東京9R'}]
        md = article.build_article('2026-11-01', secs, paid=False)
        assert '無料版' in md
        assert '推奨しなかった' in md
        assert '有料版より当たりやすいという意味ではありません' in md
        # 無料版には S/A の一覧を出さない（推奨していないものに等級を付けない）
        assert '本日の推奨レース' not in md


class TestRaceHeading:
    def test_recommended_race_shows_grade(self):
        races = _real_races()
        rec = [r for r in races
               if (r.get('conf') or 0) >= tickets.CONF_STRONG and r.get('horses')]
        if not rec:
            pytest.skip('この latest.json に推奨レースが無い')
        for r in rec[:3]:
            md = article.build_race_section(r, r['horses'], {}, paid=True)['markdown']
            assert f"**推奨度 {tickets.grade(r['conf'])}**" in md

    def test_non_recommended_race_has_no_grade(self):
        races = _real_races()
        low = [r for r in races
               if (r.get('conf') or 0) < tickets.CONF_STRONG and r.get('horses')]
        if not low:
            pytest.skip('この latest.json に推奨外レースが無い')
        for r in low[:3]:
            md = article.build_race_section(r, r['horses'], {}, paid=True)['markdown']
            assert '推奨度' not in md
            assert '**AI評価' in md


class TestEndToEnd:
    """実際に `scripts/build_product_article.py` を回して振り分けを確かめる。"""

    def _run(self, *extra):
        import build_product_article as bpa
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'a.md')
            rc = bpa.main(['--base-dir', BASE, '--out', out, '--no-web', *extra])
            if rc != 0:
                pytest.skip('この latest.json では該当レースが無い')
            with open(out, encoding='utf-8') as f:
                return f.read()

    def _headings(self, md):
        return [ln for ln in md.splitlines() if ln.startswith('## ')]

    def test_paid_contains_only_recommended_races(self):
        md = self._run()
        races = {f"{r['_venue']}{r.get('r')}R": r for r in _real_races()}
        for h in self._headings(md):
            key = h[3:].split('　')[0]
            assert races[key]['conf'] >= tickets.CONF_STRONG, h

    def test_free_contains_only_main_races_without_recommendation(self):
        md = self._run('--free')
        races = {f"{r['_venue']}{r.get('r')}R": r for r in _real_races()}
        for h in self._headings(md):
            r = races[h[3:].split('　')[0]]
            assert scope.is_main_race(r), h
            assert (r.get('conf') or 0) < tickets.CONF_STRONG, h

    def test_paid_and_free_do_not_overlap(self):
        paid = set(self._headings(self._run()))
        free = set(self._headings(self._run('--free')))
        assert paid and free
        assert not (paid & free)

    def test_free_article_is_not_trimmed(self):
        """🔴 ユーザー指示「無料記事も有料と同じ（買い目まで全部）」。

        中身を削る旧仕様に戻ったら落ちる。
        """
        md = self._run('--free')
        assert '### 買い目' in md
        assert '### 能力・適性・市場の比較' in md
        assert '買い目・全頭評価・他レースは有料版で公開しています' not in md
