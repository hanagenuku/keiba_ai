"""note 用の記事（無料版／有料版）を組み立てる。

構成は Phase 4 §11 の順序に合わせてある。
無料版と有料版の差は **質ではなく量**（無料は1レースだけ・買い目なし）。
"""

from . import commentary as _cm
from . import marks as _marks
from . import tickets as _tk


def _stars(n):
    if not n:
        return '—'
    return '★' * n + '☆' * (5 - n)


def _odds_text(v):
    """1.0倍未満（＝未取得）は「—」。有料記事に 0.0倍 を載せない。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '—'
    if f < _tk.MIN_VALID_ODDS:
        return '—'
    return f'{f:.1f}倍'


def _rank_table_md(rows, show_fit):
    head = '| 印 | 馬番 | 馬名 | AI+市場 | AI単独 |'
    sep = '|---|---:|---|---:|---:|'
    if show_fit:
        head += ' 適性 |'
        sep += '---:|'
    head += ' 市場人気 | 単勝 | 複勝率 |'
    sep += '---:|---:|---:|'
    out = [head, sep]
    for r in rows:
        cells = [r['mark'] or '', str(r['num']), r['name'] or '',
                 str(r['rl_rank']), str(r['solo_rank'] or '—')]
        if show_fit:
            cells.append(str(r['fit_rank'] or '—'))
        fk = r.get('fuku_pct')
        cells += [str(r['pop'] or '—'), _odds_text(r['odds']),
                  f'{fk:.0f}%' if isinstance(fk, (int, float)) else '—']
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def _marks_line(rows):
    by_mark = {}
    for r in rows:
        if r['mark']:
            by_mark.setdefault(r['mark'], []).append(r)
    lines = []
    for m in (_marks.MARKS[0], _marks.MARKS[1], _marks.MARKS[2]):
        for r in by_mark.get(m, []):
            lines.append(f"{m} {r['num']}{r['name']}")
    subs = by_mark.get(_marks.SUB_MARK, [])
    if subs:
        lines.append(_marks.SUB_MARK + ' ' + ''.join(str(r['num']) for r in subs))
    return '\n'.join(lines)


def _bets_md(tickets):
    if tickets.get('verdict') == 'skip':
        return '**今回は見送り。**\n\n' + (tickets.get('skip_reason') or '')
    lines = []
    for b in tickets.get('bets', []):
        combos = b.get('combos') or []
        if b['type'] == '複勝':
            lines.append(f"複勝　{combos[0][0]}")
        else:
            partners = '・'.join(str(c[1]) for c in combos)
            lines.append(f"{b['type']}　{b['axis']} － {partners}")
    body = '\n'.join(lines)
    if tickets.get('downgrade_reason'):
        body = '⚠ ' + tickets['downgrade_reason'] + '。\n\n' + body
    # §11「この予想なら、なぜこの券種なのか」まで書く
    reasons = tickets.get('bet_type_reasons') or []
    if reasons:
        body += '\n\n**なぜこの券種か**\n\n' + '\n\n'.join(reasons)

    # 🔴 §10「資金配分は、オッズと資金に応じて各自調整 とする」。
    #    比率は目安・金額は換算例として出し、断定形にしない。
    # 券種が1つなら配る先が無いので比率・換算例は出さない（自明な表を載せない）。
    ratio = tickets.get('allocation_ratio') or {}
    if len(ratio) >= 2:
        rt = ' / '.join(f'{k} {int(v * 100)}%' for k, v in ratio.items())
        body += f'\n\n**配分の目安（比率）**：{rt}'
        ex = tickets.get('allocation_examples') or {}
        if ex:
            body += '\n\n比率を円に直した換算例（推奨額ではありません）：'
            for amt in sorted(ex):
                detail = '・'.join(f'{k} {v}円' for k, v in ex[amt].items())
                body += f'\n- 合計{amt}円で買う場合　{detail}'
        # レースごとは短縮版。期待値が動かない理由の全文は記事末尾に1回だけ出す。
        if tickets.get('allocation_note'):
            body += '\n\n' + _tk.ALLOCATION_NOTE_SHORT
    elif ratio:
        body += ('\n\n金額はオッズと手持ち資金に応じて各自調整してください'
                 '（券種が1つなので配分はありません）。')
    body += '\n\n※券種は2種類以内に抑えています。点数を減らすのは自由です。'
    return body


def build_race_section(race, horses, facts_by_num, *, paid=True,
                       n_comment_marks=3):
    """1レースぶんの本文を返す。

    `n_comment_marks` は**根拠の文章を書く印の数**（1=◎のみ / 2=◎○ / 3=◎○▲）。
    印そのもの・比較表・買い目は減らない（§9 のフォーマットは崩さない）。
    既定は 3 で、従来の出力と完全に同じ。
    """
    marked = _marks.assign_marks(horses)
    rows = _marks.rank_table(marked)
    dangers = _marks.danger_favorites(marked)
    tickets = _tk.build_tickets(race, marked)
    show_fit = _marks.has_fit_rank(marked)

    venue = race.get('_venue') or ''
    title = f"## {venue}{race.get('r')}R　{race.get('name') or ''}　{race.get('dist') or ''}"

    parts = [title, '']
    parts.append(f"**AI評価　{_stars(tickets.get('stars'))}**（レース信頼度 {race.get('conf')}）")
    parts.append('')
    parts.append('### 印')
    parts.append('')
    parts.append(_marks_line(rows) or '（印なし）')
    parts.append('')

    # 本命・対抗・単穴の理由
    order = [(_marks.MARKS[0], '本命'), (_marks.MARKS[1], '対抗'), (_marks.MARKS[2], '単穴')]
    try:
        n_comment = max(1, min(len(order), int(n_comment_marks)))
    except (TypeError, ValueError):
        n_comment = len(order)
    for mark, label in order[:n_comment]:
        h = next((x for x in marked if x.get('product_mark') == mark), None)
        if h is None:
            continue
        if not paid and mark != _marks.MARKS[0]:
            continue
        body = _cm.horse_comment(h, facts_by_num.get(h.get('n'), {}), race)
        parts.append(f"### {mark} {label}　{h.get('n')}{h.get('name')}"
                     f"（単勝{_odds_text(h.get('odds') or h.get('tansho_odds'))}）")
        parts.append('')
        parts.append(body or '（書ける事実がありません）')
        parts.append('')

    if paid:
        parts.append('### 能力・適性・市場の比較')
        parts.append('')
        parts.append(_rank_table_md(rows, show_fit))
        parts.append('')
        if not show_fit:
            parts.append('※適性順位はこの日の生成には含まれていません。')
            parts.append('')

        if dangers:
            parts.append('### ⚠ 危険な人気馬')
            parts.append('')
            for d in dangers:
                parts.append(
                    f"- **{d['num']}{d['name']}**（市場{d['pop']}番人気・"
                    f"{_odds_text(d['odds'])}）：市場を見ないAIの評価は{d['solo_rank']}位。"
                )
            parts.append('')
            parts.append('市場が上位人気に推している馬のうち、AIが独自評価で下位に置いた馬です。'
                         'この条件に当たった市場1番人気は、過去の検証で3着内率が'
                         '68.5%から51.4%まで落ちました。')
            parts.append('⚠ ただし「代わりにどの馬を買えばよいか」までは言えません'
                         '（AIの推しに入れ替えても回収率は改善しませんでした）。')
            parts.append('')

        parts.append('### 最終結論')
        parts.append('')
        parts.append(_cm.race_comment(race, marked, tickets, dangers))
        parts.append('')
        parts.append('### 買い目')
        parts.append('')
        parts.append(_bets_md(tickets))
        parts.append('')
    else:
        parts.append('*買い目・全頭評価・他レースは有料版で公開しています。*')
        parts.append('')

    return {
        'markdown': '\n'.join(parts),
        'marked': marked,
        'rows': rows,
        'dangers': dangers,
        'tickets': tickets,
    }


def build_article(day_label, sections, *, paid=True, has_web_notes=False):
    """1日ぶんの記事。"""
    kind = '有料版' if paid else '無料版'
    head = [f'# {day_label} AI競馬分析　{kind}', '']
    if paid:
        head.append('AIが過去データと今回の条件を分析し、印・根拠・買い目までまとめています。')
    else:
        head.append('AIが本命にした理由だけを無料で公開します。')
    head.append('')
    head.append('---')
    head.append('')
    body = ('\n---\n\n').join(s['markdown'] for s in sections)
    tail = ['', '---', '', '### データ出典', '', _cm.sources_note(has_web_notes),
            '', '### 注意事項', '', _cm.disclaimer()]
    return '\n'.join(head) + body + '\n'.join(tail) + '\n'
