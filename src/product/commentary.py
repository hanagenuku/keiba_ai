"""事実 → 自然文。**事実が無ければ書かない。**

🔴 このモジュールが守る最重要ルール（`biz/PHASE4_DESIGN.md` §0①）:
存在しない陣営コメント・調教・パドックを作らない。書けるのは
`facts.build_facts()` が history.db と latest.json から実際に取り出した値だけ。

そのため文の組み立ては「事実があればその節を足す」方式にしてある。
事実が1つも無ければ「過去の実績から書けることが無い」旨をそのまま書く
（もっともらしい文章を作らない）。

🔴 禁止語をモジュール定数として持ち、テストで「生成文に絶対に現れない」ことを固定する。
将来テンプレートを増やしたときに、うっかり推測を混ぜたら落ちる。
"""

import re

# 生成文に現れてはいけない表現。いずれも「こちらが取得経路を持っていない情報」を
# 指すもので、出てきた時点で推測を書いたことになる。
#
# ⚠ 正規表現で持つ。素の部分一致にすると「調教師」（trainer・実データで持っている）が
#    「調教」に引っかかって誤検知する。
FORBIDDEN_PATTERNS = (
    r'陣営',
    r'厩舎コメント',
    r'関係者',
    r'談話',
    r'調教(?!師)',      # 調教タイムは持っていない。調教師名は持っている
    r'追い切り',
    r'坂路',
    r'ウッド',
    r'馬なり',
    r'パドック',
    r'気配',
    r'仕上がり',
    r'とのこと',
    r'と話して',
)

_FORBIDDEN_RE = tuple((p, re.compile(p)) for p in FORBIDDEN_PATTERNS)

_STYLE_NOTE = {
    '逃げ': 'ハナを取れれば持ち味を出せる',
    '先行': '前めの位置を取れれば力を出せる',
    '差し': '流れが速くなれば末脚を生かせる',
    '追込': '前が止まる展開になれば浮上の余地がある',
}


def _fmt_odds(v):
    try:
        return f'{float(v):.1f}倍'
    except (TypeError, ValueError):
        return None


def horse_comment(horse, facts, race):
    """1頭ぶんの見解。事実のある節だけを連ねる。"""
    parts = []

    # --- 順位の関係（latest.json の実値。ここは必ず書ける） ----------------
    rl = horse.get('rl_rank')
    solo = horse.get('solo_rank')
    fit = horse.get('fit_rank')
    pop = horse.get('pop')
    rank_bits = []
    if isinstance(rl, int) and rl < 99:
        rank_bits.append(f'AI評価（市場を含む）は{rl}位')
    if isinstance(solo, int) and solo < 99:
        rank_bits.append(f'市場を見ないAI単独の評価では{solo}位')
    if isinstance(fit, int) and fit < 99:
        rank_bits.append(f'今回条件への適性は{fit}位')
    if isinstance(pop, int) and pop < 99:
        rank_bits.append(f'市場は{pop}番人気に置いている')
    if rank_bits:
        parts.append('、'.join(rank_bits) + '。')

    # 市場を見ないAIと市場のズレ。
    # 🔴 「AIが市場より高く評価している馬を狙う」方向は3回測って否定されている
    #    （EV>=1.0 は回収 57.9% / AI強気+3以上は複勝率 12.7% / corr(gap,誤差) = −0.1213）。
    #    だから「狙い目」とは書かず、情報として出したうえで注意を添える。
    if isinstance(solo, int) and isinstance(pop, int) and solo < 99 and pop < 99:
        d = pop - solo
        if d >= 3:
            parts.append('市場の評価より、市場を見ないAIの評価のほうが上。'
                         'ただし過去の検証では、AIが市場より高く評価した側に賭けても'
                         '回収率は改善しなかったため、狙い目としては扱わない。')
        elif d <= -3:
            parts.append('市場の評価のほうが、市場を見ないAIの評価より上。'
                         'AIが見ている範囲では市場ほど高くは評価できない。')

    # 能力と適性のズレ（§5 の読みどころ）
    if isinstance(solo, int) and isinstance(fit, int) and solo < 99 and fit < 99:
        gap = fit - solo
        if gap >= 3:
            parts.append('能力の評価は上位だが、今回の条件への適性はそれより下。'
                         '力を出しきれるかは条件面がどう出るか次第。')
        elif gap <= -3:
            parts.append('総合的な能力より、今回の条件への適性が上に出ている。'
                         'この条件なら能力上位馬を逆転する余地がある。')
        elif abs(gap) <= 1:
            parts.append('能力と今回条件への適性が一致しており、扱いやすい。')

    # 🔴 「履歴を読めていない」と「本当に初出走」を区別する。
    #    `build_facts()` は history.db を引けたときだけ `n_past_runs` を作る。
    #    キーが無い状態（LFS実体が無い等で conn=None だった）を 0 と同じに扱うと、
    #    キャリアのある馬にまで「初出走」と書いてしまう＝商品に嘘が載る（§20）。
    n_past = facts.get('n_past_runs')
    if n_past is None:
        return _join(parts)
    if n_past == 0:
        parts.append('初出走のため、過去の実績から書けることはない。')
        return _join(parts)

    # --- 近走の安定感 -----------------------------------------------------
    if facts.get('placed_streak'):
        parts.append(f"直近{facts['placed_streak']}戦は続けて3着以内に入っており、"
                     '近走の安定感は高い。')
    elif facts.get('last_lost_but_good_agari'):
        pass  # 下の「着外だが末脚は評価」で前走に触れるので二重に書かない
    elif facts.get('last_place'):
        p = facts['last_place']
        if p <= 3:
            parts.append(f'前走は{p}着。')
        else:
            td = facts.get('last_time_diff')
            if td is not None and td <= 0.5:
                parts.append(f'前走は{p}着だったが着差は{td:.1f}秒で、'
                             '大きく崩れたわけではない。')
            else:
                parts.append(f'前走は{p}着。')

    # --- 着外だが末脚は評価（2026-08-18 で市場残差 +0.87〜1.18pt と実測） ---
    if facts.get('last_lost_but_good_agari'):
        f = facts['last_lost_but_good_agari']
        parts.append(f"前走は{f['place']}着だが上がり3Fは{f['agari_rank']}位で、"
                     '内容そのものは悪くない。')

    # --- 末脚 -------------------------------------------------------------
    elif facts.get('agari_top3'):
        a = facts['agari_top3']
        line = f"直近{a['of']}走のうち{a['count']}走で上がり3F上位3位以内。"
        if a['count'] >= 3:
            line += '末脚は安定して使えている。'
        parts.append(line)

    # --- 距離の変化 -------------------------------------------------------
    if facts.get('dist_change'):
        d = facts['dist_change']
        if d['direction'] == '延長':
            parts.append(f"前走の{d['from']}mから{d['to']}mへの距離延長。"
                         '追走に余裕を持てるかが鍵。')
        else:
            parts.append(f"前走の{d['from']}mから{d['to']}mへの距離短縮。"
                         'より速い流れへの対応が問われる。')

    # --- 芝ダ替わり -------------------------------------------------------
    if facts.get('surface_change'):
        s = facts['surface_change']
        parts.append(f"前走の{s['from']}から{s['to']}への替わり。")

    # --- コース替わり -----------------------------------------------------
    if facts.get('venue_change'):
        vc = facts['venue_change']
        past_venues = {r.get('racecourse') for r in facts.get('past_runs', [])}
        if vc['to'] in past_venues:
            parts.append(f"{vc['to']}は経験がある。")
        else:
            parts.append(f"前走の{vc['from']}から{vc['to']}へ。当コースは初。")

    # --- 脚質 -------------------------------------------------------------
    if facts.get('style'):
        st = facts['style']
        note = _STYLE_NOTE.get(st)
        parts.append(f'脚質は{st}。' + (note + '。' if note else ''))

    # --- 道悪 -------------------------------------------------------------
    if facts.get('heavy_record'):
        h = facts['heavy_record']
        if h['placed'] >= 1:
            parts.append(f"道悪は{h['runs']}走で{h['placed']}回3着以内。")
        elif h['runs'] >= 2:
            parts.append(f"道悪は{h['runs']}走して3着以内が無い。")

    # --- 市場の見立てを超えた走り -----------------------------------------
    if facts.get('beat_market'):
        b = facts['beat_market']
        if b['count'] >= 2:
            parts.append(f"直近{b['of']}走のうち{b['count']}走で人気以上に走っている。")

    # --- 休養 -------------------------------------------------------------
    gap = facts.get('days_since_last')
    if isinstance(gap, int):
        if gap >= 180:
            parts.append(f'前走から{gap}日空いており、久々の一戦。')
        elif gap <= 14:
            parts.append(f'前走から{gap}日での連戦。')

    # --- 騎手替わり -------------------------------------------------------
    if facts.get('jockey_change'):
        j = facts['jockey_change']
        parts.append(f"騎手は{j['from']}から{j['to']}へ乗り替わり。")

    return _join(parts)


# ---------------------------------------------------------------------------
# 根拠（§9）。2026-10-15 にユーザーが「軸に選ぶ馬の根拠と危険な人気馬の根拠は
# 読者が一番欲しい情報」と明言したので、毎レース必ず書く固定文として持つ。
#
# 🔴 ここに書く数字は CLAUDE.md に実測として残っているものだけ。
#    盛らない。測っていない数字を足さない。
# ---------------------------------------------------------------------------

#: ◎ をその馬にした根拠。`axis_basis()` が信頼度の数字を差し込んで返す。
AXIS_BASIS_HEAD = (
    '**◎にした根拠**：このレースでAIの総合評価（過去走の実データ＋市場のオッズ）が'
    '1位だった馬です。下の比較表は「能力（市場を見ないAI単独）」「今回条件への適性」'
    '「市場人気」を分けて並べてあるので、◎の評価が何に支えられているかが読めます。'
)

#: 信頼度の読み方。⚠ 確率としては上側で過信なので、そう書く。
AXIS_BASIS_CONF = (
    'レース信頼度{c}は「◎が3着以内に来る確率」としてAIが出した数字です。'
    '⚠ 確率そのものとしては上の帯で過信があり（0.65を超えた帯は予測0.758に対し'
    '実測0.566）、レース間の比較として読んでください。'
    '当てやすいレースを見分ける力は過去3,560レースの検証で市場の指標を上回りました'
    '（判別力 0.6041 対 0.5708）。'
)

#: 期待を盛らないための一文。◎の根拠の最後に必ず付ける。
AXIS_BASIS_TAIL = (
    '⚠ 一方で「どの馬が来るか」の精度は市場とほぼ同じです'
    '（複勝の判別力で市場＋0.0004）。この予想が市場より上だと確認できているのは'
    '「どのレースを見送るか」の判断で、回収率が100%を超える根拠はありません。'
)


def axis_basis(conf=None):
    """◎ を軸にした根拠。読者が最も欲しい情報なので毎レース出す。"""
    parts = [AXIS_BASIS_HEAD]
    parts.append(AXIS_BASIS_CONF.format(
        c=f'（このレースは{conf}）' if conf is not None else ''))
    parts.append(AXIS_BASIS_TAIL)
    return _join(parts)


#: この印の根拠。
#: 🔴 2026-10-16 に B期間（2025-01-05〜2026-10-12・6,077レース）で測り直した。
#:    結果は `danger/RESULTS.md`。事前登録した6条件（閾値を上げる／◎を除く／
#:    適性を足す／2番人気まで広げる）は**すべて不採用**で、
#:    「高い確率で3着内に来ない人気馬」は現在のデータでは特定できなかった。
#:    よって**断定的な名前をやめ、事実の提示に留める**（カード §7 の事前決定）。
#: 🔴 「3着内に来ない」と書かない。実測は 51.4% / 48.6% ＝ 半分近くは来る。
DANGER_BASIS = (
    '**この印の根拠**：市場が1番人気に推している馬のうち、オッズを一切見ない'
    'AI単独の評価で6位以下に沈んだ馬を挙げています。\n'
    '2つの期間で測ってあります。242頭・73開催日では3着内率が 68.5% → **51.4%**、'
    '6,077レース・849頭（別期間・予測時点より後のデータを使わない形）では'
    '63.6% → **48.6%** でした。落ち幅はどちらも15〜17ポイントで再現しています。\n'
    '🔴 **それでも半分近くは3着内に来ます。**「来ない」という意味ではなく'
    '「1番人気としては過剰に信頼できない」という意味で読んでください。'
    '条件をもっと厳しくすれば「来ない」と書ける水準になるかも検証しましたが、'
    '**なりませんでした**（厳しくすると該当が月に数頭まで減り、期間を前半・後半に'
    '割ると数字が入れ替わります）。\n'
    '⚠ 代わりにどの馬を買えばよいかは言えません。AIの推しに入れ替えても'
    '複勝回収率は 77.5% 対 75.9% で、避けた1番人気と区別がつきませんでした。'
    'この印を避けることで複勝回収率は 82.6% → 84.2% と 1.6ポイント上がりますが、'
    '**100%には届きません**。'
)

#: この印が ◎ 自身だったときに足す説明。
#: 🔴 2026-10-16 の B期間測定で**構造だと確定した**。849頭のうち
#:    **749頭（88.2%）が ◎ 自身**（`rl_rank == 1`）で、その3着内率は 53.8%。
#:    ◎ ではない 100頭（11.8%）だけが 10.0% と低い。
#:    何も書かないと「本命」と「危険」が同じ馬について並ぶだけで読者が混乱する。
DANGER_IS_AXIS = (
    '※**この馬は本記事の◎と同じ馬です。** AIの総合評価（市場込み）では1位ですが、'
    '市場を抜くと下位＝この馬の高い評価は市場の支持に大きく乗っている、という形です。'
    'そのためこのレースは勝負度を1段下げています。\n'
    '⚠ この形は例外ではありません。6,077レースで測ると、この印が付いた849頭のうち'
    '**88.2%（749頭）が◎自身**で、その3着内率は53.8%でした。'
    '◎ではない馬に付いた場合（全体の11.8%）だけ3着内率が10.0%まで落ちます。'
    'つまり**◎自身に付いたこの印は、強い警告としては読めません**。'
)


def find_forbidden(text):
    """禁止表現が含まれていれば最初に見つかったパターンを返す。無ければ None。"""
    if not text:
        return None
    for pat, rx in _FORBIDDEN_RE:
        if rx.search(text):
            return pat
    return None


def _join(parts):
    text = ''.join(parts)
    hit = find_forbidden(text)
    if hit:
        # 実装のミスを黙って商品に出さない。落として気づかせる
        raise AssertionError(
            f'生成文に禁止表現「{hit}」が含まれた。'
            'このシステムは陣営コメント・調教タイム・パドックを取得していないため、'
            'これらに触れる文を書いてはいけない（biz/PHASE4_DESIGN.md §0①）'
        )
    return text


# `cmt`（アプリ表示用のレースコメント）から落とす文。
# 🔴 落とす理由: ①信頼度はこちらが conf から出すので二重に言わせない
#    （実データで conf 34 のレースに「信頼度高い」と書かれている例がある）
#    ②オッズ未取得のレースで「(0.0倍)」と書かれる
#    ③「中穴(3.6倍)で期待値十分」のような EV の断定が混ざる。
#      EV = AI確率 × オッズ で買い目を選ぶ方向は 2026-07-05 / 07-30 / 08-31 の
#      3回とも否定されている（EV>=1.0 は回収 57.9% < 買わない側 75.0%）。
#      自分の記録が否定したものを商品の文章で断定しない
_CMT_DROP = ('信頼度', '0.0倍', '期待値')


def _clean_cmt(cmt):
    """アプリ用コメントから、有料記事に載せられない文を落とす。"""
    if not cmt:
        return ''
    kept = []
    for sent in str(cmt).split('。'):
        sent = sent.strip()
        if not sent:
            continue
        if any(w in sent for w in _CMT_DROP):
            continue
        kept.append(sent)
    return ('。'.join(kept) + '。') if kept else ''


def _ensure_period(text):
    """文末に句点が無ければ足す。見解を連結したときに文が繋がって読めなくなるため。"""
    t = (text or '').strip()
    if t and t[-1] not in '。！？':
        t += '。'
    return t


def race_comment(race, horses, tickets, dangers):
    """レース全体の見解。AI が既に出している `cmt` を使い、判断だけ足す。"""
    parts = []
    cmt = _clean_cmt(race.get('cmt'))
    if cmt:
        parts.append(cmt)

    conf = race.get('conf')
    skip_reason = tickets.get('skip_reason') or ''
    # 🔴 2026-10-07 修正。見送り理由が既に信頼度を名指ししている場合、
    #    「レース信頼度は52。レース信頼度が低い（52）ため見送り。」と二重になっていた。
    if conf is not None and not (tickets.get('verdict') == 'skip'
                                 and 'レース信頼度' in skip_reason):
        parts.append(f'レース信頼度は{conf}。')

    if tickets.get('verdict') == 'skip':
        parts.append(_ensure_period(skip_reason or '今回は見送り。'))
    elif tickets.get('verdict') == 'light':
        parts.append('軸は取れるが相手を広げる根拠までは無い帯。'
                     '点数を絞って対応したい。')
    else:
        parts.append('軸の信頼度は比較的高い帯。◎から相手を絞って狙える。')

    if dangers:
        # 🔴 ◎ 自身が危険な人気馬だった場合、その説明は skip_reason /
        #    downgrade_reason が既にしている。同じ話を1レースで3回書かない。
        axis_num = next((h.get('n') for h in (horses or [])
                         if h.get('rl_rank') == 1), None)
        covered = bool(tickets.get('axis_is_danger_favorite')) and bool(
            skip_reason or tickets.get('downgrade_reason'))
        rest = [d for d in dangers
                if not (covered and d.get('num') == axis_num)]
        if rest:
            names = '・'.join(f"{d['num']}番{d['name']}" for d in rest)
            parts.append(f'⚠ {names} は市場が上位人気に推しているが、'
                         '市場を見ないAIの評価は下位。'
                         'ここを信頼しすぎない方がよい（詳細は上の節）。')

    return _join(parts)


def disclaimer():
    """商品に必ず添える文（§22）。"""
    return (
        '※本記事はAIによる分析をもとにした予想です。的中や利益を保証するものではありません。\n'
        '※過去の成績は将来の結果を保証しません。馬券の購入は自己責任でお願いします。\n'
        '※予想は公開時点の内容で、結果を見てからの修正は行いません。\n'
        '※資金配分で期待値は動きません（全体の期待値は各買い目の期待値の加重平均に'
        'なるため、どう振り分けても平均より上には行きません）。記事に書いている比率は'
        '当たり外れの振れ幅を抑えるための目安で、オッズと手持ち資金に応じて'
        '各自調整してください。'
    )


def sources_note(has_web_info=False):
    """出典の明示（§12）。Web情報を使っていない場合はそう書く。"""
    base = ('データ出典：JRA公式（出馬表・レース結果）。'
            '各馬の記述は過去走の実データ（着順・上がり3F・距離・コース・脚質・'
            '馬場状態・人気）から機械的に抽出したものです。')
    if has_web_info:
        return base + '\nWebから取得した情報は本文中に出典を明記しています。'
    return base + '\n本記事では陣営コメント・調教・パドックの情報は使用していません。'
