"""アプリの出走表の下に出す note 用テキスト（data/note_blocks.json）。

ユーザー依頼（2026-11-14）:
    「商品になる予定のレースの予想根拠コメントと買い目をアプリのレース出走表の
      下部に置いてくれないか。そのままコピーして note に貼り付けられるように」

固定したいのは文章そのものではなく**どこから来た文章か**:

1. 本文は記事と**同じ実装**（`article.build_race_section`）から作る。
   画面側（JS）でも、このスクリプトの中でも、文章を組み立て直さない。
   組み立てを分けると `commentary.FORBIDDEN_PATTERNS`（陣営コメント・調教・
   パドックに触れない）が片方だけ素通りする（2026-10-16 の「危険な人気馬」で、
   判定を2箇所に分けると記事と台帳が静かに食い違うと分かっている）。
2. 載せるレースは `scope.article_kind()` の判定だけで決める（閾値を作り直さない）。
3. 予想（◎・印・買い目・確率）を一切変えない。読むだけ。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import build_note_blocks as nb  # noqa: E402
from src.product import article, commentary, scope  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEST = os.path.join(BASE, 'data', 'latest.json')


@pytest.fixture(scope='module')
def built():
    """本番の `data/latest.json` をそのまま読んで作る（North Star #6）。"""
    if not os.path.exists(LATEST):
        pytest.skip('data/latest.json が無い')
    return nb.build_blocks(BASE)


def _races(built):
    return built['races']


class TestScope:
    def test_only_product_races_are_included(self, built):
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        want = set()
        for venue, rs in (data.get('races') or {}).items():
            for r in rs:
                if not (r.get('horses') or []):
                    continue
                kind, _ = scope.article_kind(r)
                if kind in (scope.PAID, scope.FREE):
                    want.add(r.get('race_id'))
        assert set(_races(built)) == want

    def test_kind_comes_from_article_kind(self, built):
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        by_id = {r.get('race_id'): r
                 for rs in (data.get('races') or {}).values() for r in rs}
        for rid, b in _races(built).items():
            kind, _ = scope.article_kind(by_id[rid])
            assert b['kind'] == kind

    def test_threshold_is_not_rebuilt_in_the_script(self):
        src = open(os.path.join(BASE, 'scripts', 'build_note_blocks.py'),
                   encoding='utf-8').read()
        assert 'article_kind' in src
        assert str(scope.FLAT_MIN_CONF) not in src     # 66 を直に書かない


class TestBodyComesFromTheArticle:
    def test_markdown_is_identical_to_the_article_section(self, built):
        """同じレースを記事側の関数で作り直して1文字まで一致すること。"""
        if not _races(built):
            pytest.skip('この latest.json には商品対象のレースが無い')
        rid, b = next(iter(_races(built).items()))
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        race = next(dict(r, _venue=v)
                    for v, rs in (data.get('races') or {}).items()
                    for r in rs if r.get('race_id') == rid)
        sec = article.build_race_section(race, race['horses'], {}, paid=True)
        # facts 無しで作ると過去走の記述だけ減る。見出しと買い目は一致する。
        assert b['md'].startswith(sec['markdown'].split('\n')[0])
        assert '### 買い目' in b['md']
        assert b['heading'] == sec['heading']

    def test_script_does_not_write_its_own_sentences(self):
        src = open(os.path.join(BASE, 'scripts', 'build_note_blocks.py'),
                   encoding='utf-8').read()
        assert 'build_race_section' in src
        for word in ('### 印', '### 買い目', '◎にした根拠'):
            assert word not in src, f'{word} をスクリプト側で組み立てている'

    def test_every_block_has_the_reasons_and_the_bets(self, built):
        for b in _races(built).values():
            assert '### 印' in b['md']
            assert '### ◎ 本命' in b['md']
            assert '### 買い目' in b['md']

    def test_no_forbidden_sentences(self, built):
        """陣営コメント・調教・パドックに触れる**生成文**を出さない。

        判定は `commentary.find_forbidden` をそのまま呼ぶ（ここで正規表現を
        書き直さない）。⚠ web見解の節は出典つきの引用であって生成文ではなく、
        `web_block.py` が別の歯止め（出典必須・不反映の明記）を課している層
        なので、生成文の部分だけを見る（commentary のガードは緩めていない）。
        """
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        names_by_id = {
            r.get('race_id'): tuple(h.get('name') for h in (r.get('horses') or []))
            for rs in (data.get('races') or {}).values() for r in rs}
        for rid, b in _races(built).items():
            body = b['md'].split('### 🌐 web情報')[0]
            # ⚠ 馬名・騎手名は判定から外す（2026-10-18「ベルウッドバイオ」）。
            #   こちらが書いた文の中の語は names を伏せても捕まる。
            hit = commentary.find_forbidden(body, names=names_by_id.get(rid, ()))
            assert hit is None, (b['heading'], hit)

    def test_no_overclaiming_words(self, built):
        for b in _races(built).values():
            body = b['md']
            for word in ('必ず勝てる', '確実に当たる', '儲かります'):
                assert word not in body


class TestFileShape:
    def test_keyed_by_race_id_with_the_fields_the_app_needs(self, built):
        for rid, b in _races(built).items():
            assert rid.count('_') == 2                  # 20261114_05_09 の形
            for key in ('kind', 'venue', 'r', 'name', 'conf', 'heading', 'md'):
                assert key in b

    def test_records_which_prediction_it_was_built_from(self, built):
        """予想を作り直したら古い本文だと分かるようにしておく。"""
        with open(LATEST, encoding='utf-8') as f:
            data = json.load(f)
        assert built['source_generated_at'] == data.get('generated_at')

    def test_date_comes_from_race_id_not_the_display_date(self, built):
        if not _races(built):
            pytest.skip('商品対象のレースが無い')
        rid = next(iter(_races(built)))
        assert built['date'] == f'{rid[:4]}-{rid[4:6]}-{rid[6:8]}'

    def test_building_does_not_touch_latest_json(self, built):
        """読むだけ。予想のファイルを書き換えない。"""
        before = os.path.getmtime(LATEST)
        nb.build_blocks(BASE)
        assert os.path.getmtime(LATEST) == before


class TestPage:
    def test_app_shows_the_block_and_a_copy_button(self):
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        assert 'note用テキスト' in html
        assert 'copyNoteBlock' in html
        assert 'note_blocks.json' in html

    def test_app_does_not_build_the_sentences_itself(self):
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        assert '### 買い目' not in html
        assert '◎にした根拠' not in html

    def test_workflow_builds_the_file_and_publishes_it(self):
        wf = open(os.path.join(BASE, '.github', 'workflows', 'weekend.yml'),
                  encoding='utf-8').read()
        assert 'build_note_blocks.py' in wf
        assert 'note_blocks.json' in wf       # Pages にも配る
