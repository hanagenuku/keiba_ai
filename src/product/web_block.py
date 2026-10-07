"""Claude が当日朝に web から集めた見解を、記事の**別枠**セクションにする層。

＝＝ この層が何をしないか（ここを崩すと6件目の撤回になる）＝＝

このプロジェクトでは「予想の出力を後から書き換える層」が **5件すべて撤回**されている
（市場補正レイヤー / error_tags週次補正 / rank_matrix_filter / AI xx%バッジ /
ROI予測150%）。したがってこの層は **印・買い目・確率を1つも変えない**:

  - `assign_marks` / `rank_table` / `build_tickets` の結果を**読まないし渡さない**
  - 出力は記事の末尾に足す markdown の行だけ
  - `tests/test_product_web_block.py::test_web_notes_do_not_change_marks_or_tickets`
    が「web見解あり／なしで印・比較表・買い目が完全一致」をコードで固定する

⚠ **上乗せを期待していない。** 2026-09-27 の23レース実測で、web の能力指数は
  指数〜市場 ρ+0.825 に対し AI〜市場 +0.627 ＝ **web のほうが市場に近い**（独立情報ではない）。
  websig（134,874頭）では複勝回収率が 79.7% → 79.1% と**悪化**して不採用になった。
  この層を作ってあるのは「AI単体 vs AI＋web」を後から実配当で比べられる形にするため。

＝＝ 何を載せるか ＝＝

`data/claude_web_findings.json` のうち、記事に載せるのは次の4つだけ。

  1. 馬場状態・トラックバイアス（`going`）
  2. レース全体の一言（`summary`）
  3. 馬ごとのメモ（`horses[].note`）。**出典 URL を持つものだけ**
  4. 出典（`sources`）

🔴 **web の序列・印（`order` / `marks`）は記事に載せない。** AI の印と competing する
   2つ目の印を同じ記事に並べると「どちらに従うのか」が読者に分からなくなる。
   しかも web の序列は上の実測どおり**市場人気の言い換えに近い**ので、
   載せても読者が得る情報は増えない。アプリ（`index.html`）側は別列で出しているので
   比較したいときはそちらを見る。

🔴 **出典の無いメモは落とす。** `src/utils/source_registry.py` と同じ思想＝
   「書くだけでは届かない。裏づけの実体が要る」。

＝＝ `commentary.FORBIDDEN_PATTERNS` との関係（重要・弱めていない）＝＝

`src/product/commentary.py` は 陣営・調教・パドック に触れる文を**生成**できないよう
ガードしている。理由は `PHASE4_DESIGN.md` の「**取得経路が無い**」。
この層が扱うのは Claude が web で読んだ**出典つきの引用**で、生成文ではない。
よってここに同じ正規表現は当てないが、**`commentary.py` のガードは1文字も緩めていない**
（生成文は従来どおり落ちる）。代わりの歯止めとして、この層は

  - メモ1件ごとに **出典 URL を必須**にする（無ければ落とす）
  - セクション見出しに「**AIの評価には反映していません**」と毎回書く

の2つを課す。
"""

MAX_NOTE_LEN = 200


def _going_line(going):
    if not isinstance(going, dict):
        return None
    surface = str(going.get('surface') or '').strip()
    state = str(going.get('state') or '').strip()
    expected = str(going.get('expected') or '').strip()
    if not (state or expected):
        return None
    head = f'{surface}' if surface else '馬場'
    if state and expected and state != expected:
        return f'- **{head}の馬場**：{state}（事前の想定は{expected}）'
    return f'- **{head}の馬場**：{state or expected}'


def _note_lines(view, marked):
    """出典を持つ馬メモだけを行にする。落とした件数も返す。"""
    names = {h.get('n'): h.get('name') for h in (marked or [])}
    lines, dropped = [], 0
    for num in sorted((view.get('horse_notes') or {}).keys()):
        item = view['horse_notes'][num] or {}
        note = str(item.get('note') or '').strip()
        srcs = [s for s in (item.get('sources') or []) if isinstance(s, str) and s.strip()]
        if not note:
            continue
        if not srcs:
            # 出典の無いメモは載せない（裏づけの無い文を商品に出さないため）
            dropped += 1
            continue
        if len(note) > MAX_NOTE_LEN:
            note = note[:MAX_NOTE_LEN] + '…'
        arrow = {'up': '↑', 'down': '↓'}.get(item.get('direction'), '')
        name = names.get(num) or ''
        lines.append(f'- **{num}{name}**{arrow}：{note}（[出典]({srcs[0]})）')
    return lines, dropped


def build_web_block(view, marked):
    """記事に足す markdown の行リストを返す。載せるものが無ければ `[]`。

    Args:
        view: `src.features.claude_web.build_claude_view()` の戻り値（または None）。
        marked: そのレースの馬（`n` / `name` を持つ dict のリスト）。馬名の表示にだけ使う。
                🔴 印・確率・買い目は**読まない**。
    """
    if not view:
        return []

    body = []
    gl = _going_line(view.get('going'))
    if gl:
        body.append(gl)

    summary = str(view.get('summary') or '').strip()
    if summary:
        body.append(f'- {summary}')

    notes, dropped = _note_lines(view, marked)
    body += notes

    if not body:
        return []

    out = ['### 🌐 web情報（AIの評価には反映していません）', '']
    out += body
    out.append('')
    srcs = [s for s in (view.get('sources') or []) if isinstance(s, str) and s.strip()]
    if srcs:
        out.append('出典：' + ' / '.join(f'[{i + 1}]({s})' for i, s in enumerate(srcs)))
        out.append('')
    if dropped:
        out.append(f'※出典が確認できなかった記述 {dropped} 件は掲載していません。')
        out.append('')
    out.append('※この節は当日朝に web から集めた参考情報です。'
               '上の印・比較表・買い目はこの情報を使わずに算出しています。')
    out.append('')
    return out
