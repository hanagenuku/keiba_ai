"""比較表の画像化（2026-11-23 ユーザー添削「表が見難い」）。

🔴 守るのは「表の中身を1つも変えない」こと。見た目は変えてよい。
"""

import importlib
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import table_image  # noqa: E402

md_to_html = importlib.import_module('scripts.md_to_html')


ROWS = [
    ['印', '馬番', '馬名', 'AI+市場', 'AI単独', '適性', '市場人気', '単勝', '複勝率'],
    ['◎', '10', 'ショウナンガレオン', '1', '4', '4', '1', '2.3倍', '59%'],
    ['○', '1', 'ダノンダックス', '2', '3', '3', '2', '5.4倍', '48%'],
    ['', '12', 'オンユアマークス', '6', '6', '11', '6', '10.7倍', '22%'],
]


class TestTableHtml:
    def test_every_cell_survives(self):
        """🔴 中身を書き換えない。全セルがそのまま入っていること。"""
        html = table_image.table_html(ROWS)
        for row in ROWS:
            for cell in row:
                if cell:
                    assert cell in html, cell

    def test_header_row_is_thead(self):
        html = table_image.table_html(ROWS)
        assert len(re.findall(r'<th[ >]', html)) == len(ROWS[0])
        assert '<thead>' in html
        assert len(re.findall(r'<tr[ >]', html)) == len(ROWS)

    def test_axis_row_is_marked(self):
        """◎の行だけ色を敷く（アプリと同じ見え方）。"""
        html = table_image.table_html(ROWS)
        assert html.count('<tr class="axis">') == 1

    def test_numeric_columns_are_right_aligned(self):
        html = table_image.table_html(ROWS)
        # 馬名（左寄せ）には num クラスを付けない
        assert '<td>ショウナンガレオン</td>' in html
        assert '<td class="num">2.3倍</td>' in html

    def test_cells_are_escaped(self):
        html = table_image.table_html([['印', '馬名'], ['◎', '<script>']])
        assert '<script>' not in html
        assert '&lt;script&gt;' in html

    def test_too_few_rows_is_an_error(self):
        with pytest.raises(table_image.TableImageError):
            table_image.table_html([['印', '馬番']])


class TestMdToHtmlHook:
    def test_renderer_replaces_the_table(self):
        md = '## 見出し\n\n| 印 | 馬番 |\n|---|---:|\n| ◎ | 7 |\n'
        seen = []

        def render(rows):
            seen.append(rows)
            return '<p><img src="data:image/png;base64,AAA"></p>'

        html = md_to_html.md_to_html(md, table_renderer=render)
        assert seen == [[['印', '馬番'], ['◎', '7']]]
        assert '<img src="data:image/png;base64,AAA">' in html
        assert '<table' not in html

    def test_falling_back_keeps_the_table(self):
        """🔴 画像にできなくても記事は出す（表のまま出る）。"""
        md = '| 印 | 馬番 |\n|---|---:|\n| ◎ | 7 |\n'
        html = md_to_html.md_to_html(md, table_renderer=lambda rows: None)
        assert '<table' in html
        assert '<th>印</th>' in html

    def test_default_is_unchanged(self):
        md = '| 印 | 馬番 |\n|---|---:|\n| ◎ | 7 |\n'
        assert md_to_html.md_to_html(md) == md_to_html.md_to_html(
            md, table_renderer=None)


def _playwright_available():
    if not shutil.which('node'):
        return False
    try:
        table_image._playwright_path()
    except table_image.TableImageError:
        return False
    return True


@pytest.mark.skipif(not _playwright_available(),
                    reason='Playwright が無い環境（記事は表のまま出る）')
class TestRender:
    def test_data_uri_is_a_png(self):
        uri = table_image.data_uri(ROWS)
        assert uri.startswith('data:image/png;base64,')
        assert len(uri) > 5000           # 空画像ではない

    def test_png_is_wide_enough_to_read(self):
        """潰れた画像を貼らない。9列なら 800px 以上にはなる。"""
        png = table_image.render_png(ROWS)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'
        width = int.from_bytes(png[16:20], 'big')
        assert width >= 800, width
