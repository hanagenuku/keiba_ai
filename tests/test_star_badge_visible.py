"""☆がスマホで見えること（2026-11-17: 馬名セルの省略表示に切られて1本も見えていなかった）。"""
import os
import re

HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html'),
            encoding='utf-8').read()


def test_star_badge_comes_before_the_name_so_ellipsis_cannot_hide_it():
    # 馬名セルは max-width + overflow:hidden + ellipsis。名前の後ろに置いたバッジは切られる
    assert re.search(r'td\.col-name\{[^}]*overflow:hidden[^}]*text-overflow:ellipsis', HTML)
    assert """<td class="col-name">'+starBadge+h.name+""" in HTML
    assert "h.name+starBadge" not in HTML


def test_star_note_is_rendered_as_text_not_only_as_a_tooltip():
    # 併記文は省略不可。title 属性だけではスマホで読めない
    assert 'class="star-note"' in HTML and '_starTitle + \'</div>\'' in HTML
