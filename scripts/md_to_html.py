#!/usr/bin/env python3
"""記事の Markdown を Google ドキュメント変換用の HTML にする。

🔴 汎用の Markdown 実装ではない。`src/product/article.py` が実際に出す構造だけを
   対象にする（見出し h1-h3 / 表 / 箇条書き / 太字 / 斜体 / 段落。水平線は落とす）。
   Drive の `create_file` に `contentMimeType: "text/html"` で渡すと、Google が
   `<h1>`〜`<h3>`・`<table>`・`<ul>`・`<b>`・`<i>` をネイティブの書式に変換する。

なぜスクリプトにするか: 土日朝の Routine が毎回 HTML を手で組むと、週によって
見出しが段落になったり表が崩れたりする。変換を1箇所に固定してテストで押さえる。

使い方:
    python3 scripts/md_to_html.py article.md > article.html
"""

import html
import re
import sys

_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ITALIC = re.compile(r'(?<!\*)\*([^*\n]+)\*(?!\*)')
_CODE = re.compile(r'`([^`\n]+)`')
_TABLE_SEP = re.compile(r'^\|[\s:\-|]+\|$')


def _inline(text):
    """行内の装飾。エスケープしてから太字・斜体・コードを戻す。"""
    out = html.escape(text, quote=False)
    out = _BOLD.sub(r'<b>\1</b>', out)
    out = _ITALIC.sub(r'<i>\1</i>', out)
    out = _CODE.sub(r'<code>\1</code>', out)
    return out


def _split_row(line):
    cells = line.strip().split('|')
    if cells and cells[0] == '':
        cells = cells[1:]
    if cells and cells[-1] == '':
        cells = cells[:-1]
    return [c.strip() for c in cells]


def _table_html(rows):
    """1行目を見出し行として扱う（article.py の表は必ず見出し付き）。

    🔴 `<thead>` で包む。包まずに `<tr><th>` だけを置くと、Drive の変換が
       **空の見出し行を1行足して**こちらの見出しをデータ行に落とす
       （2026-10-05 に実際の Google ドキュメントで確認）。
    """
    head, body = rows[0], rows[1:]
    out = ['<table border="1">', '<thead><tr>' + ''.join(
        f'<th>{_inline(c)}</th>' for c in head) + '</tr></thead>', '<tbody>']
    for cells in body:
        out.append('<tr>' + ''.join(
            f'<td>{_inline(c)}</td>' for c in cells) + '</tr>')
    out.append('</tbody>')
    out.append('</table>')
    return '\n'.join(out)


def md_to_html(text):
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # 水平線は落とす。
        # 🔴 `<hr>` の直後が見出しだと、Drive の変換が見出しの前に `-----` を
        #    くっつけて `## -----京都9R…` という行にしてしまう
        #    （2026-10-05 に実際の Google ドキュメントで確認）。
        #    区切りは見出し（h2）が担っているので、水平線は不要。
        if re.fullmatch(r'-{3,}', line.strip()):
            i += 1
            continue

        # 見出し（#### 以降は article.py が出さないので h3 に寄せる）
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            level = min(len(m.group(1)), 3)
            out.append(f'<h{level}>{_inline(m.group(2))}</h{level}>')
            i += 1
            continue

        # 表（次の行が区切り行ならヘッダ付きの表）
        if line.lstrip().startswith('|') and i + 1 < n and \
                _TABLE_SEP.match(lines[i + 1].strip()):
            rows = [_split_row(line)]
            i += 2
            while i < n and lines[i].lstrip().startswith('|'):
                rows.append(_split_row(lines[i]))
                i += 1
            out.append(_table_html(rows))
            continue

        # 箇条書き
        if re.match(r'^\s*[-*]\s+', line):
            items = []
            while i < n and re.match(r'^\s*[-*]\s+', lines[i].rstrip()):
                items.append(re.sub(r'^\s*[-*]\s+', '', lines[i].rstrip()))
                i += 1
            out.append('<ul>' + ''.join(
                f'<li>{_inline(x)}</li>' for x in items) + '</ul>')
            continue

        # 段落（空行までをまとめる。記事は1行1段落だが改行を潰さない）
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r'^(#{1,6}\s|\s*[-*]\s|\|)', lines[i]) and \
                not re.fullmatch(r'-{3,}', lines[i].strip()):
            para.append(lines[i].rstrip())
            i += 1
        out.append('<p>' + '<br>'.join(_inline(x) for x in para) + '</p>')

    body = '\n'.join(out)
    return ('<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head>\n'
            f'<body>\n{body}\n</body></html>\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print('使い方: python3 scripts/md_to_html.py <file.md>', file=sys.stderr)
        return 1
    with open(argv[0], encoding='utf-8') as f:
        sys.stdout.write(md_to_html(f.read()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
