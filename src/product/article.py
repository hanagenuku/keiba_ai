"""note 用の記事（無料版／有料版）を組み立てる。

構成は Phase 4 §11 の順序に合わせてある。
無料版と有料版の差は **質ではなく量**（無料は1レースだけ・買い目なし）。
"""

from . import commentary as _cm
from . import marks as _marks
from . import tickets as _tk
from . import web_block as _wb


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
        # 🔴 2026-10-07 修正。区切りなしで連結していたため、14番と7番が「147」と
        #    1頭の馬番のように見えていた（実データで確認）。馬名まで出して確定させる。
        lines.append(_marks.SUB_MARK + ' '
                     + '・'.join(f"{r['num']}{r['name']}" for r in subs))
    return '\n'.join(lines)


def _course_md(course):
    """1コースぶんの本文。点数・総額・配分はコースの中で閉じる。"""
    head = (f"**【{course['name']}】{course['lead']}**　"
            f"{course['points']}点"
            f"（1点100円で{course['flat_unit_total']:,}円）")
    body = [head, '']
    body += [b['line'] for b in course['lines']]

    # 🔴 §10「資金配分は、オッズと資金に応じて各自調整 とする」。
    #    金額は換算例として出し、断定形にしない。
    #    券種が1つなら配る先が無いので出さない（自明な表を載せない）。
    ex = course.get('allocation_examples') or {}
    outcomes = course.get('outcomes') or {}
    ratio = course.get('allocation_ratio') or {}
    if len(ratio) >= 2 and ex:
        body.append('')
        if outcomes:
            # 単勝・複勝（各1点）。金額の自由度があるので円で示す。
            # 🔴 単勝の比率は「複勝だけ当たっても元本を割らない」上限
            #    （= 1 − 1/複勝オッズ）で抑えている。tickets.py 冒頭 (A)。
            body.append('掛け金の換算例（推奨額ではありません。'
                        '◎が2〜3着＝複勝だけ当たったときに元本を割らない範囲で'
                        '単勝に回しています）：')
            for amt in sorted(ex):
                detail = '・'.join(f'{k} {v:,}円' for k, v in ex[amt].items() if v)
                line = f'- 合計{amt:,}円　{detail}'
                oc = outcomes.get(amt)
                if oc:
                    line += (f'　→ ◎が1着なら{oc["win"]:,}円 / '
                             f'2〜3着なら{oc["place"]:,}円 / 4着以下なら0円')
                body.append(line)
            zero = [a for a in sorted(ex) if not ex[a].get('単勝')]
            if zero and any(ex[a].get('単勝') for a in ex):
                body.append(f'- ⚠ 合計{zero[-1]:,}円以下では単勝を混ぜると'
                            f'「複勝は当たったのに元本割れ」になるため、'
                            f'複勝だけにしています。')
            elif zero and len(zero) == len(ex):
                body.append('- ⚠ 今回の複勝オッズでは、単勝を混ぜると'
                            '「複勝は当たったのに元本割れ」になります。'
                            'このレースは複勝だけにしています。')
            body.append('- ⚠ 複勝オッズは発売中は幅で出ます。'
                        '上の戻りは記事作成時点の表示値での計算です。')
        else:
            # 多点のコース。配分できるのは1点あたりの単価だけ。
            body.append('掛け金の換算例（推奨額ではありません。'
                        '券種ごとに単価を変える根拠が無いので1点の単価をそろえています）：')
            for amt in sorted(ex):
                pts = course['points'] or 1
                up = amt // pts
                detail = '・'.join(f'{k} {v:,}円' for k, v in ex[amt].items() if v)
                body.append(f'- 1点{up}円（合計{amt:,}円）　{detail}')
    return '\n'.join(body)


def _bets_md(tickets):
    if tickets.get('verdict') == 'skip':
        return '**今回は見送り。**\n\n' + (tickets.get('skip_reason') or '')

    # 🟢 2026-10-26 ユーザー指示「全部の買い目は推奨しない。2種程度を推奨する」。
    #    9行を1つの塊として出すのをやめ、コース単位で出す。
    #    🔴 コースをまたいだ合計点数・合計金額・配分は出さない。出すと
    #       「全部買う」形に戻るうえ、重複する行を二重に数えることになる
    #       （三連複(2頭軸) ⊂ 三連複、三連単フォーメーション ⊂ 三連単）。
    blocks = []
    courses = tickets.get('courses') or []
    for c in courses:
        blocks.append(_course_md(c))
    if not blocks:
        # コースが組めないとき（相手が取れない等）は行をそのまま出す
        blocks.append('\n'.join(b['line'] for b in tickets.get('bets', [])))
    body = '\n\n'.join(blocks)

    if tickets.get('course_note'):
        body += '\n\n' + tickets['course_note']
    # §10「資金配分は各自調整」。コースが組めなかった場合でも落とさない。
    body += '\n\n' + _tk.ALLOCATION_NOTE_SHORT

    # 🟢 固めと中穴がどれなのかを読者に見せる（2026-10-15 ユーザー指示の実体）。
    solid = tickets.get('solid_partners') or []
    ana = tickets.get('ana_partners') or []
    if solid or ana:
        bits = []
        # 馬番の昇順で書く（買い目の行と同じ規約。選んだ順そのままだと読めない）
        if solid:
            bits.append('固め ' + '・'.join(str(n) for n in sorted(solid)))
        if ana:
            bits.append('中穴 ' + '・'.join(str(n) for n in sorted(ana)))
        body += (f'\n\n相手の内訳：{" ／ ".join(bits)}'
                 f'（固めは単勝{_tk.ANA_ODDS_MIN:.0f}倍未満、'
                 f'中穴は{_tk.ANA_ODDS_MIN:.0f}〜{_tk.ANA_ODDS_MAX:.0f}倍）')

    # 参考：同じ券種の絞り込み版。コースと重ねて買うものではない。
    alts = tickets.get('alternatives') or []
    if alts:
        body += '\n\n**参考：同じ券種を点数を絞って買う場合**\n\n'
        body += (tickets.get('alternative_note') or '') + '\n\n'
        body += '\n'.join(b['line'] for b in alts)

    if tickets.get('downgrade_reason'):
        body = '⚠ ' + tickets['downgrade_reason'] + '。\n\n' + body
    # §11「この予想なら、なぜこの券種なのか」まで書く
    reasons = tickets.get('bet_type_reasons') or []
    if reasons:
        body += '\n\n**なぜこの券種か**\n\n' + '\n\n'.join(reasons)
    return body


def build_race_section(race, horses, facts_by_num, *, paid=True,
                       n_comment_marks=3, web_view=None):
    """1レースぶんの本文を返す。

    `n_comment_marks` は**根拠の文章を書く印の数**（1=◎のみ / 2=◎○ / 3=◎○▲）。
    印そのもの・比較表・買い目は減らない（§9 のフォーマットは崩さない）。
    既定は 3 で、従来の出力と完全に同じ。

    `web_view` は当日朝に web から集めた見解
    （`src.features.claude_web.build_claude_view()` の戻り値）。
    🔴 **印・比較表・買い目の計算に一切渡さない。** 記事の末尾に別枠で足すだけ。
    `None` なら出力は従来と完全に同じ。
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
        # 🔴 ◎ だけ「なぜ軸にしたか」を必ず書く（2026-10-15 ユーザー指示。
        #    読者が最も欲しい情報は印そのものではなく根拠）。
        if mark == _marks.MARKS[0]:
            parts.append('')
            parts.append(_cm.axis_basis(race.get('conf')))
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
            # 🔴 2026-10-16: 「危険な人気馬」という断定的な見出しをやめた。
            #    B期間6,077レースで測った3着内率は 48.6%（半分近く来る）で、
            #    事前登録した6条件すべてが不採用（danger/RESULTS.md）。
            #    カード §7 の事前決定どおり、事実の提示に留める名前にする。
            parts.append('### ⚠ 市場の支持ほどAIは評価していない馬')
            parts.append('')
            _axis = _marks.axis_horse(marked)
            _axis_num = _axis.get('n') if _axis else None
            for d in dangers:
                bits = [f"市場を見ないAI単独の評価は{d['solo_rank']}位"]
                _rl = d.get('rl_rank')
                if isinstance(_rl, int) and _rl < 99:
                    bits.append(f'AI総合（市場込み）では{_rl}位')
                _fit = next((h.get('fit_rank') for h in marked
                             if h.get('n') == d['num']), None)
                if isinstance(_fit, int) and _fit < 99:
                    bits.append(f'今回条件への適性は{_fit}位')
                parts.append(
                    f"- **{d['num']}{d['name']}**（市場{d['pop']}番人気・"
                    f"{_odds_text(d['odds'])}）：" + '、'.join(bits) + '。'
                )
                # 🔴 ◎ 自身が当たることが多い（2026-10-15 実測で本番11件中9件）。
                #    黙っていると「本命」と「危険」が同じ馬について並ぶだけになる。
                if _axis_num is not None and d['num'] == _axis_num:
                    parts.append('')
                    parts.append(_cm.DANGER_IS_AXIS)
            parts.append('')
            parts.append(_cm.DANGER_BASIS)
            parts.append('')

        parts.append('### 最終結論')
        parts.append('')
        parts.append(_cm.race_comment(race, marked, tickets, dangers))
        parts.append('')
        parts.append('### 買い目')
        parts.append('')
        parts.append(_bets_md(tickets))
        parts.append('')

        # 🔴 web見解は**全部の算出が終わった後**に足す。上の marked / rows /
        #    tickets / dangers はこの時点で確定済みで、web_view を一度も見ていない。
        web_md = _wb.build_web_block(web_view, marked)
        if web_md:
            parts += web_md
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


def build_article(day_label, sections, *, paid=True, has_web_info=False):
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
    tail = ['', '---', '', '### データ出典', '', _cm.sources_note(has_web_info),
            '', '### 注意事項', '', _cm.disclaimer()]
    return '\n'.join(head) + body + '\n'.join(tail) + '\n'
