"""比較表を「アプリの出走表に似た画像」にする。

🟢 2026-11-23 ユーザー添削「表の部分、すごく見難いので、別で作った見やすい表
   （アプリの表のような）を貼り付ける形にできないか？」→ 選択は「表を画像にして貼る」。

なぜ画像か: Google ドキュメントの表変換は列幅を揃えてくれないので、
9列の表が折り返して読めなくなる。印の色分けも消える。画像なら崩れない。

🔴 **表の中身は1つも変えない。** `article._rank_table_md()` が出す markdown の表を
   そのまま読んで描き直すだけ（数字の作り直し・並べ替え・丸めはしない）。
   note 用の .md は markdown の表のまま残す（note は画像を貼り直せる）。

⚠ 描画は Playwright（Chromium）に依存する。使えない環境では `render_png()` が
   `TableImageError` を上げる。呼ぶ側（`scripts/md_to_html.py`）は失敗したら
   HTML の表にそのまま戻す＝記事が出なくなることはない。
"""

import base64
import html as _html
import os
import shutil
import subprocess
import tempfile

#: 画像の倍率。文字が潰れない最小。
SCALE = 2

#: Playwright の実体を探す場所（この環境の既定 → 環境変数で上書き可）。
_PW_CANDIDATES = (
    os.environ.get('KEIBA_PLAYWRIGHT_PATH') or '',
    '/opt/node-tools/node_modules/playwright/index.js',
)

#: アプリの出走表に合わせた配色。
#: ◎の行だけ薄い黄色で、印の列は太字（アプリと同じ見え方）。
_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 12px; background: #fff;
       font-family: "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif; }
table { border-collapse: collapse; font-size: 15px; }
th, td { border: 1px solid #d7dcd9; padding: 6px 10px; white-space: nowrap; }
th { background: #0f3d2e; color: #fff; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.axis td { background: #fff8e1; }
td.mark { font-weight: 700; text-align: center; }
tbody tr:nth-child(even) td { background: #f7f9f8; }
tbody tr.axis:nth-child(even) td { background: #fff8e1; }
"""


class TableImageError(RuntimeError):
    """画像を作れなかった。呼ぶ側は markdown/HTML の表に戻すこと。"""


def _is_num_col(name):
    return name in ('馬番', 'AI+市場', 'AI単独', '適性', '市場人気', '単勝', '複勝率')


def table_html(rows):
    """markdown の表（1行目が見出し）から、画像化用の単体 HTML を作る。

    `rows` は `[[セル, ...], ...]`。中身は文字列のまま使う（数値に直さない）。
    """
    if not rows or len(rows) < 2:
        raise TableImageError('表の行が足りない')
    head = [str(c) for c in rows[0]]
    num = [_is_num_col(c) for c in head]
    out = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
           f'<style>{_CSS}</style></head><body><table id="t"><thead><tr>']
    for i, c in enumerate(head):
        cls = ' class="num"' if num[i] else ''
        out.append(f'<th{cls}>{_html.escape(c)}</th>')
    out.append('</tr></thead><tbody>')
    for r in rows[1:]:
        cells = [str(c) for c in r]
        # 1列目が印。◎の行はアプリと同じく色を敷く。
        axis = ' class="axis"' if cells and cells[0].strip() == '◎' else ''
        out.append(f'<tr{axis}>')
        for i, c in enumerate(cells):
            cls = 'num' if i < len(num) and num[i] else ('mark' if i == 0 else '')
            attr = f' class="{cls}"' if cls else ''
            out.append(f'<td{attr}>{_html.escape(c)}</td>')
        out.append('</tr>')
    out.append('</tbody></table></body></html>')
    return ''.join(out)


_NODE_SCRIPT = """
import pw from %(pw)r;
import fs from 'fs';
const { chromium } = pw;
const b = await chromium.launch();
const p = await b.newPage({ deviceScaleFactor: %(scale)d });
await p.setContent(fs.readFileSync(%(html)r, 'utf8'), { waitUntil: 'load' });
const el = await p.$('#t');
if (!el) { await b.close(); process.exit(3); }
await el.screenshot({ path: %(png)r });
await b.close();
"""


def _playwright_path():
    for c in _PW_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise TableImageError('Playwright が見つからない')


def render_png(rows, *, timeout=120):
    """表の PNG バイト列を返す。作れなければ `TableImageError`。"""
    node = shutil.which('node')
    if not node:
        raise TableImageError('node が無い')
    pw = _playwright_path()
    with tempfile.TemporaryDirectory() as d:
        hp = os.path.join(d, 't.html')
        pp = os.path.join(d, 't.png')
        mp = os.path.join(d, 'shot.mjs')
        with open(hp, 'w', encoding='utf-8') as f:
            f.write(table_html(rows))
        with open(mp, 'w', encoding='utf-8') as f:
            f.write(_NODE_SCRIPT % {'pw': pw, 'scale': SCALE,
                                    'html': hp, 'png': pp})
        r = subprocess.run([node, mp], capture_output=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(pp):
            raise TableImageError(
                f'描画に失敗（rc={r.returncode}）: '
                + (r.stderr.decode('utf-8', 'replace')[-300:] or '出力なし'))
        with open(pp, 'rb') as f:
            return f.read()


def data_uri(rows, **kw):
    """`<img src="...">` にそのまま入れられる data URI。"""
    png = render_png(rows, **kw)
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')
