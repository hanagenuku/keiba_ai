"""買い目と資金配分。**固定ルール**（後から最適化しない）。

根拠は `biz/PHASE4_DESIGN.md` §3・§4。要点だけ再掲する。

🔴 複勝を基本にする理由は「数字が高いから」ではない。2026-08-18③ で高自信レースの
券種を総当たりしたとき、**上位3本の配当を抜いても崩れないのは複勝だけ**
（複勝 −1.4pt / ワイド −8.5pt）。ワイドは控除率 22.5% で複勝 20% より不利。
単勝は RL1 が 86.0% で 1番人気 86.1% と区別がつかない。

🔴 三連単・三連複・馬連は標準構成に入れない。2026-08-03 に総当たりして
馬連 70.2% / ワイド 71.4% / 三連複は万馬券1本を除くと 58.9%。

🔴 傾斜配分で期待値は動かない（総期待値 = 各EVの加重平均。全EV<1ならどう配分しても<1）。
配分を示すのは分散を抑えるためであって、「配分で勝てる」とは書かない。
"""

from . import marks as _marks

# 勝負度の境界。conf は race_confidence()（軸の3着内確率 × 100）。
#
# 🔴 どちらも **こちらが発明した数字ではなく、本番で既に使われている閾値**に合わせてある。
#    新しい閾値を勝手に作らない（それをやると後から最適化したのと同じになる）。
#
#  CONF_STRONG = 66 : 🔴 **2026-10-05 訂正。これは本番 `rec` の境界ではない。**
#                     当初「2026-09-27 の23レースで rec と conf>=66 が1件も食い違わない」
#                     ことを根拠にしたが、**その日だけの偶然だった**。
#                     10/03 は食い違い4件（rec=True の conf が 62/62/63/64）、
#                     10/04 は3件（55/63/63）。
#                     機序: `rec` は閾値ではなく「軸の3着内確率>=0.55 を満たすレースのうち
#                     **その日の上位 max_races=6 本**」（`ev_filter.select_quality_races`）。
#                     9/27 は上位6本がたまたま全部 conf>=66 だっただけ。
#                     ⚠ したがって 66 は**こちらが選んだ閾値**であり、本番由来ではない。
#                     どう扱うかはユーザー判断待ち（候補: `rec` そのものを使う /
#                     CONF_LIGHT=55 に寄せる / 66 を選択として明記したまま使う）。
#                     **結果を見てから動かさないこと。**
#  CONF_LIGHT  = 55 : `ev_filter.MIN_AXIS_FUKU_PROB = 0.55` と同じ。
#                     本番が「そもそも買い目を出すか」を判定している閾値で、
#                     根拠は 2026-08-08②（軸の複勝確率が下位半分の帯はROIが一貫して悪い）
CONF_STRONG = 66   # これ以上: 複勝 + ワイド
CONF_LIGHT = 55    # これ以上: 複勝のみ
# CONF_LIGHT 未満: 見送り

MAX_BET_TYPES = 2  # §8 1レース2券種以内

# 単勝オッズの下限。JRAの単勝は 1.0 倍未満が存在しないので、それ未満は「未取得」
# （2026-07-27⑧ でエンジン側にも同じ下限を入れてある）。
# 🔴 有料商品に 0.0倍 を載せない。土曜夜の日曜予想は毎週 odds_coverage=0.0 になる
MIN_VALID_ODDS = 1.0

# 配分（比率）。複勝を厚く、ワイドは押さえ。
# 🔴 これは**指示ではなく目安**。§10 は「資金配分は、オッズと資金に応じて各自調整 とする」
#    と定めており、PHASE4_DESIGN.md §4 も「比率で示す・金額は換算例」としている。
#    比率を断定形で出すと、期待値を動かせない操作（下記）を効くもののように見せてしまう。
ALLOC_STRONG = {'複勝': 0.70, 'ワイド': 0.30}
ALLOC_LIGHT = {'複勝': 1.00}

# 金額は「比率を円に直したときの換算例」であって推奨額ではない。
AMOUNT_EXAMPLES = (1000, 3000, 5000)

# 配分に必ず添える注記。記事・アーカイブの両方へこの文のまま載せる。
# 🔴 2026-08-18③: 総期待値 = 各EVの加重平均。全EV<1ならどう配分しても<1。
#    配分は分散を変えるだけで期待値を1ミリも動かさない（算数で決着している）。
ALLOCATION_NOTE = (
    '配分は目安です。資金配分で期待値は動きません'
    '（全体の期待値は各買い目の期待値の加重平均になるため、'
    'どう振り分けても平均より上には行きません）。'
    '比率を示しているのは当たり外れの振れ幅を抑えるためで、'
    'オッズと手持ち資金に応じて各自調整してください。'
)

# レースごとに載せる短縮版。理由の全文（上）は記事末尾の注意事項に1回だけ出す。
# 🔴 同じ段落を1記事に7回以上くり返すと読み飛ばされる。短縮版でも
#    「目安であること」と「各自調整」は落とさない（そこが §10 の要件）。
ALLOCATION_NOTE_SHORT = (
    '配分は目安です。オッズと手持ち資金に応じて各自調整してください。'
)


# 券種を選んだ理由（§11「この予想なら、なぜこの券種なのか」まで説明する）。
#
# 🔴 出どころは全部「過去データで買い方を総当たりした測定」で、こちらの成績ではない。
#    どの数字も100%を超えていない。券種を選べば控除率を超えられる、とは書かない。
REASON_FUKUSHO = (
    '複勝を本線にしているのは、過去データで買い方を総当たりしたときに'
    '**上位の高配当を3本抜いても順位が崩れなかったのが複勝だけ**だったためです'
    '（3本抜きでの下落は複勝 −1.4pt に対しワイド −8.5pt）。'
    '控除率も20%で、対象にしている券種のなかで最も低いです。'
)
REASON_WIDE = (
    'ワイドは押さえです。控除率が22.5%で複勝より不利なぶん、'
    '◎から○▲への数点に留めています。'
)
REASON_LIGHT_ONLY = (
    '今回は複勝だけです。このレースはレース信頼度が相手を広げる帯に届いていないため、'
    '点数を増やしていません。'
)
REASON_EXCLUDED = (
    '馬連・三連複・三連単は入れていません。同じ過去データで総当たりした結果、'
    '馬連は70.2%、三連複は最高配当1本を除くと58.9%で、'
    '控除率が22.5〜25%と複勝より不利でした。'
)
REASON_CAVEAT = (
    'なお、ここに挙げた回収率はいずれも100%を下回っています。'
    '券種の選び方で控除率を超えられるとは考えていません。'
    '複勝を選んでいるのは「勝てるから」ではなく'
    '「測った数字が偶然の高配当に左右されにくいから」です。'
)


def bet_type_reason(tickets):
    """この買い目でその券種になった理由を文のリストで返す（§11）。

    見送りなら空リスト（買っていないものの理由は書かない）。
    """
    if tickets.get('verdict') == 'skip':
        return []
    kinds = {b['type'] for b in tickets.get('bets', [])}
    if not kinds:
        return []
    out = []
    if '複勝' in kinds:
        out.append(REASON_FUKUSHO)
    if 'ワイド' in kinds:
        out.append(REASON_WIDE)
    elif kinds == {'複勝'}:
        out.append(REASON_LIGHT_ONLY)
    out.append(REASON_EXCLUDED)
    out.append(REASON_CAVEAT)
    return out


def usable_odds(horse):
    """使えるオッズ（1.0倍以上）を返す。無ければ None。"""
    for key in ('odds', 'tansho_odds'):
        v = horse.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f >= MIN_VALID_ODDS:
            return f
    return None


def confidence_stars(conf):
    """conf → 勝負度の星。conf が無ければ None（作り話をしない）。"""
    if conf is None:
        return None
    if conf >= 72:
        return 5
    if conf >= CONF_STRONG:
        return 4
    if conf >= CONF_LIGHT:
        return 3
    if conf >= 46:
        return 2
    return 1


def verdict(conf):
    """買うか見送るか。conf が無ければ見送り（不明なら勝負しない）。"""
    if conf is None:
        return 'skip'
    if conf >= CONF_STRONG:
        return 'strong'
    if conf >= CONF_LIGHT:
        return 'light'
    return 'skip'


def _round_unit(amount, unit=100):
    return int(max(unit, round(amount / unit) * unit))


def allocate(total, ratios):
    """比率 → 100円単位の金額。端数は最も比率の大きい券種に寄せる。"""
    if not ratios:
        return {}
    out = {k: _round_unit(total * v) for k, v in ratios.items()}
    diff = total - sum(out.values())
    if diff:
        top = max(ratios, key=lambda k: ratios[k])
        out[top] = max(100, out[top] + diff)
    return out


def build_tickets(race, horses):
    """買い目・配分・見送り判断を返す。

    入力は変更しない。券種は必ず MAX_BET_TYPES 以内。

    🔑 ◎ 自身が「危険な人気馬」に当たった場合は勝負度を1段下げる。
       これが §10「買わない」も予想として販売する、の実体。
       根拠: 市場1番人気のうち素のAIが6位以下の馬は複勝率 68.5%→51.4%
       （N=242・73開催日・walk-forward）。◎がそこに当たっているなら、
       同じレースを強気に買う理由が無い。
    """
    conf = race.get('conf')
    v = verdict(conf)
    axis = _marks.axis_horse(horses)

    # ◎ が危険な人気馬か
    axis_is_danger = False
    if axis is not None:
        danger_nums = {d['num'] for d in _marks.danger_favorites(horses)}
        axis_is_danger = axis.get('n') in danger_nums

    result = {
        'conf': conf,
        'stars': confidence_stars(conf),
        'verdict': v,
        'bets': [],
        'allocation_ratio': {},
        'allocation_examples': {},
        'allocation_note': None,
        'bet_type_reasons': [],
        'skip_reason': None,
    }

    if axis is None:
        result['verdict'] = 'skip'
        result['skip_reason'] = '軸（AI+市場1位）が特定できないため見送り'
        return result

    if usable_odds(axis) is None:
        result['verdict'] = 'skip'
        result['skip_reason'] = (
            'オッズが取得できていないため見送り。'
            'オッズが無い状態では妙味の判断ができない'
        )
        return result

    # ◎ が危険な人気馬なら1段下げる（strong→light、light→skip）
    downgraded = False
    if axis_is_danger and v != 'skip':
        v = 'light' if v == 'strong' else 'skip'
        downgraded = True
    result['verdict'] = v
    result['axis_is_danger_favorite'] = axis_is_danger

    if v == 'skip':
        if downgraded:
            result['skip_reason'] = (
                '◎は市場1番人気だが、市場を見ないAIの評価は下位。'
                'この形に当たった市場1番人気は過去の検証で3着内率が'
                '68.5%から51.4%まで落ちている。'
                'レース信頼度も高くないため今回は見送り'
            )
        else:
            result['skip_reason'] = (
                f'レース信頼度が低い（{conf}）ため見送り。'
                f'軸の3着内確率が下位の帯では回収率が一貫して悪い'
                if conf is not None else
                'レース信頼度が算出できなかったため見送り'
            )
        return result

    if downgraded:
        result['downgrade_reason'] = (
            '◎は市場1番人気だが、市場を見ないAIの評価は下位。'
            'この形は過去の検証で3着内率が落ちているため、'
            '相手を広げず点数を絞った'
        )

    axis_num = axis.get('n')
    partners = [h for h in horses
                if h.get('product_mark') in (_marks.MARKS[1], _marks.MARKS[2])]
    partner_nums = [h.get('n') for h in partners if h.get('n') is not None]

    bets = [{
        'type': '複勝',
        'axis': axis_num,
        'combos': [[axis_num]],
        'note': '本線',
    }]

    if v == 'strong' and partner_nums:
        bets.append({
            'type': 'ワイド',
            'axis': axis_num,
            'combos': [[axis_num, p] for p in partner_nums],
            'note': '押さえ（◎から○▲へ）',
        })

    result['bets'] = bets[:MAX_BET_TYPES]

    ratios = dict(ALLOC_STRONG) if len(result['bets']) == 2 else dict(ALLOC_LIGHT)
    # 実際に作られた券種だけを残す
    kinds = {b['type'] for b in result['bets']}
    ratios = {k: v2 for k, v2 in ratios.items() if k in kinds}
    total_ratio = sum(ratios.values()) or 1.0
    ratios = {k: round(v2 / total_ratio, 2) for k, v2 in ratios.items()}
    result['allocation_ratio'] = ratios
    result['allocation_examples'] = {
        amt: allocate(amt, ratios) for amt in AMOUNT_EXAMPLES
    }
    result['allocation_note'] = ALLOCATION_NOTE
    result['bet_type_reasons'] = bet_type_reason(result)
    return result


def bet_type_count(tickets):
    return len({b['type'] for b in tickets.get('bets', [])})
