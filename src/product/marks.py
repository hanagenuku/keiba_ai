"""印（◎○▲△）と「危険な人気馬」。固定ルール。

根拠は `biz/PHASE4_DESIGN.md` §1・§2。要点だけ再掲する。

🔴 ◎ は rl_rank 1 から動かさない。唯一「市場を上回っている」と測れているのは
`conf`（軸の3着内確率・当てやすさ AUC 0.6041 vs 市場 0.5708・3,560レース）で、
それは **rl_rank1 を軸としたときの値**。◎を別の馬に動かすと商品に載せる
信頼度の根拠そのものが消える。

🔴 「AIが市場より高く評価している馬を穴として推す」形は入れない。
3回測って否定されている（EV>=1.0 は回収 57.9% / AI強気+3以上は複勝率 12.7%・回収 53.0% /
corr(gap, 符号つき誤差) = −0.1213 ＝ 市場と食い違うほどその方向にAIが間違っている）。
使うのは **逆向き**（市場が高く見ている馬をAIが低く見ている＝疑う側）だけ。
"""

MARKS = ('◎', '○', '▲')
SUB_MARK = '△'

# 危険な人気馬の条件。
# 🔴 測定は「**市場1番人気** かつ 素のAI順位 >= 6」でしか行っていない
#    （N=242・73開催日・walk-forward 5窓 → 複勝率 68.5%→51.4%）。
#    2番人気まで広げると測っていない母集団に結論を持ち出すことになるので広げない。
DANGER_MAX_POP = 1
DANGER_MIN_SOLO = 6


def _rank(horse, key):
    v = horse.get(key)
    if isinstance(v, int) and 0 < v < 99:
        return v
    return None


def assign_marks(horses):
    """rl_rank に基づいて印を割り当てた新しいリストを返す。

    入力の dict は変更しない（本番の予想を書き換えないため）。
    """
    out = []
    for h in horses:
        h2 = dict(h)
        r = _rank(h2, 'rl_rank')
        if r == 1:
            h2['product_mark'] = MARKS[0]
        elif r == 2:
            h2['product_mark'] = MARKS[1]
        elif r == 3:
            h2['product_mark'] = MARKS[2]
        elif r in (4, 5):
            h2['product_mark'] = SUB_MARK
        else:
            h2['product_mark'] = ''
        out.append(h2)
    return out


def axis_horse(horses):
    """◎（rl_rank 1）。見つからなければ None。

    `scored[0]` とは限らないので明示的に探す（ev_filter と同じ規約）。
    """
    for h in horses:
        if _rank(h, 'rl_rank') == 1:
            return h
    return None


def marked_horses(horses):
    """印つきの馬を ◎○▲△ の順に返す。"""
    order = {MARKS[0]: 0, MARKS[1]: 1, MARKS[2]: 2, SUB_MARK: 3}
    marked = [h for h in horses if h.get('product_mark')]
    return sorted(marked, key=lambda h: (order.get(h['product_mark'], 9),
                                         _rank(h, 'rl_rank') or 99))


def danger_favorites(horses):
    """危険な人気馬（市場上位人気 × 市場ゼロのAI順位が下位）。

    ⚠ 「代わりに誰を買えばいいか」は言えない。素のAIの推しは複勝回収 77.5% で、
    避けた1番人気 75.9% と区別がつかない（2026-08-29・walk-forward）。
    だから返すのは「疑う対象」だけ。
    """
    out = []
    for h in horses:
        pop = _rank(h, 'pop')
        solo = _rank(h, 'solo_rank')
        if pop is None or solo is None:
            continue
        if pop <= DANGER_MAX_POP and solo >= DANGER_MIN_SOLO:
            out.append({
                'num': h.get('n'), 'name': h.get('name'),
                'pop': pop, 'solo_rank': solo,
                'odds': h.get('odds') or h.get('tansho_odds'),
                'rl_rank': _rank(h, 'rl_rank'),
            })
    return sorted(out, key=lambda x: x['pop'])


def rank_table(horses, limit=8):
    """能力（AI単独）・適性・市場・AI+市場 の順位を並べた表。

    §5 の「能力順位と適性順位のズレ」を読者に見せるための素材。
    `fit_rank` は 2026-09-28 に新設した列なので、それ以前の latest.json には無い。
    無いときは None のまま返し、表示側で列を落とす。
    """
    rows = []
    for h in horses:
        rl = _rank(h, 'rl_rank')
        if rl is None or rl > limit:
            continue
        rows.append({
            'num': h.get('n'), 'name': h.get('name'),
            'mark': h.get('product_mark', ''),
            'rl_rank': rl,
            'solo_rank': _rank(h, 'solo_rank'),
            'fit_rank': _rank(h, 'fit_rank'),
            'pop': _rank(h, 'pop'),
            'odds': h.get('odds') or h.get('tansho_odds'),
            'fuku_pct': h.get('fuku_pct'),
        })
    return sorted(rows, key=lambda r: r['rl_rank'])


def has_fit_rank(horses):
    """適性順位が入っているか（古い latest.json では入っていない）。"""
    return any(_rank(h, 'fit_rank') for h in horses)
