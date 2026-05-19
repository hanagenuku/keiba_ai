"""
アプリ用 JSON 生成。
ノートブックの to_app_json を分離。
"""
from src.betting.ev_filter import VENUE_ORDER
from src.betting.make_bets import calc_ev, make_bets
from src.features.engine import auto_comment, calc_all


def _build_horse_ranks(scored):
    """EV・AIスコア・クラス適性の各馬ランクを返す。"""
    ev_scores = []
    for h in scored:
        p = h.get('pn', 0.1)
        o = h.get('win_odds', 10) or 10
        ev_scores.append(round(min(0.88, p * 3.2) * max(1.2, o * 0.32), 3))

    def rank_by(key_fn):
        order = sorted(range(len(scored)), key=key_fn, reverse=True)
        return {scored[i]['num']: r + 1 for r, i in enumerate(order)}

    cl_scores = [
        scored[i].get('scores', {}).get('jockey', 5)
        + scored[i].get('scores', {}).get('blood', 5)
        + scored[i].get('scores', {}).get('distance', 5)
        for i in range(len(scored))
    ]
    return (
        {scored[i]['num']: r + 1 for r, i in enumerate(
            sorted(range(len(scored)), key=lambda x: ev_scores[x], reverse=True))},
        rank_by(lambda x: scored[x]['total']),
        {scored[i]['num']: r + 1 for r, i in enumerate(
            sorted(range(len(scored)), key=lambda x: cl_scores[x], reverse=True))},
    )


def _build_horses_list(scored, top1, by_odds):
    """アプリ表示用の馬リストを生成する（馬番順）。"""
    ev_ranks, rl_ranks, cl_ranks = _build_horse_ranks(scored)
    horses = []
    for h in scored:
        pop = next((i + 1 for i, x in enumerate(by_odds) if x['name'] == h['name']), 99)
        pn  = h.get('pn', 0)
        wo  = h.get('win_odds', 0) or 0
        horses.append({
            'n':        h['num'],
            'name':     h['name'],
            'odds':     wo,
            'score':    round(h['total'], 1),
            'pop':      pop,
            'style':    h.get('running_style', '差し'),
            'tan_pct':  round(min(60,  pn * 100),       1),
            'ren_pct':  round(min(80,  pn * 2.0 * 100), 1),
            'fuku_pct': round(min(88,  pn * 3.2 * 100), 1),
            'ev_rank':  ev_ranks.get(h['num'], 99),
            'rl_rank':  rl_ranks.get(h['num'], 99),
            'cl_rank':  cl_ranks.get(h['num'], 99),
            'mark': ('推' if h['num'] == top1['num'] else
                     '高' if round(min(88, pn * 3.2 * 100), 1) >= 38 else
                     '穴' if (rl_ranks.get(h['num'], 99) <= 4 and wo >= 10) else ''),
        })
    horses.sort(key=lambda x: x['n'])
    return horses


def _build_bet_list(bets):
    """ベット辞書リストをアプリ表示形式に変換する。"""
    result = []
    for b in bets:
        tag = ('fuku' if b['type'] == '複勝' else
               'tan'  if b['type'] == '単勝' else 'wide')
        est = (f'推定{b["odds_est"]:.1f}倍' if b['type'] == '複勝'
               else f'{b["odds"]:.1f}倍')
        result.append({
            'tag':   tag,
            'label': b['type'],
            'horse': f'#{b["nums"][0]} {b["horse_name"]}',
            'est':   est,
            'amt':   f'¥{b["amount"]}',
        })
    return result


def to_app_json(selected, races_all, bias_data, jst_now, day_type='friday'):
    """厳選レース＋全レース情報をアプリ用 JSON 形式で返す。

    Args:
        selected  : ability_first_loose が返す候補リスト
        races_all : その日の全レースリスト
        bias_data : 馬場バイアス辞書（省略可）
        jst_now   : datetime（JST）
        day_type  : 'friday' | 'saturday' | 'sunday'

    Returns:
        dict（JSON シリアライズ可能）
    """
    all_venues   = sorted({r['racecourse'] for r in races_all},
                          key=lambda v: VENUE_ORDER.get(v, 99))
    total_inv    = sum(sum(b['amount'] for b in make_bets(c)) for c in selected)
    races_by_venue = {}
    selected_ids = {c['race']['id'] for c in selected}

    # ── 厳選レース ─────────────────────────────────────────────────
    for c in selected:
        race   = c['race']
        top1   = c['top1']
        scored = c['scored']
        bets   = make_bets(c)
        rc     = race['racecourse']
        if rc not in races_by_venue:
            races_by_venue[rc] = []

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(50, int(60 + (pop_rank - 2) * 4 + c['score_gap'] * 20)))

        races_by_venue[rc].append({
            'r':    race['race_num'],
            'name': race['race_name'],
            'dist': f'{race["distance"]}m{race["surface"]}',
            'rec':  True,
            'conf': conf,
            'honmei': {
                'n':     top1['num'],
                'name':  top1['name'],
                'odds':  top1.get('win_odds', 0) or 0,
                'score': top1['total'],
                'style': top1.get('running_style', '差し'),
            },
            'horses': _build_horses_list(scored, top1, by_odds),
            'bets':   _build_bet_list(bets),
            'chaos_label': ('A' if c.get('chaos_score', 0) >= 0.65 else
                            'B' if c.get('chaos_score', 0) >= 0.45 else
                            'C' if c.get('chaos_score', 0) >= 0.25 else 'D'),
            'cmt': auto_comment(c, bias_data),
        })

    # ── 非厳選レース（全レース買い目つき）─────────────────────────
    for race in sorted(races_all,
                       key=lambda r: (VENUE_ORDER.get(r['racecourse'], 99),
                                      r['race_num'])):
        if race['id'] in selected_ids:
            continue
        rc = race['racecourse']
        if rc not in races_by_venue:
            races_by_venue[rc] = []

        scored = calc_all(race, bias_data)
        if not scored:
            continue
        top1  = scored[0]
        odds  = top1.get('win_odds') or 99
        gap   = top1['total'] - scored[1]['total'] if len(scored) > 1 else 0

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(1, int(60 + (pop_rank - 2) * 4 + gap * 20)))

        win_prob = top1.get('pn', 0)
        ev_fuku  = calc_ev(min(1.0, win_prob * 3), odds * 0.28)
        ev_tan   = calc_ev(win_prob, odds)
        c_ref    = {
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          max(ev_fuku, ev_tan),
        }
        bets = make_bets(c_ref)

        races_by_venue[rc].append({
            'r':    race['race_num'],
            'name': race['race_name'],
            'dist': f'{race["distance"]}m{race["surface"]}',
            'rec':  False,
            'conf': conf,
            'honmei': {
                'n':     top1['num'],
                'name':  top1['name'],
                'odds':  top1.get('win_odds', 0) or 0,
                'score': top1['total'],
                'style': top1.get('running_style', '差し'),
            },
            'horses': _build_horses_list(scored, top1, by_odds),
            'bets':   _build_bet_list(bets),
            'cmt':    auto_comment(c_ref, bias_data),
        })

    bias_txt = '内外:フラット ペース:±0 時計:±0'
    bias_tag = 'フラット'
    if bias_data:
        bias_txt = bias_data.get('summary', bias_txt)
        spd      = bias_data.get('track_speed', 0)
        bias_tag = ('時計速め' if spd > 0.3 else '時計遅め' if spd < -0.3 else 'フラット')

    jday = ['月', '火', '水', '木', '金', '土', '日'][jst_now.weekday()]
    return {
        'generated_at': jst_now.isoformat(),
        'date':         f'{jst_now.month}月{jst_now.day}日({jday})',
        'type':         day_type,
        'venues':       all_venues,
        'bias':         {'txt': bias_txt, 'tag': bias_tag},
        'stats':        {'invest': total_inv, 'rec': len(selected), 'roi': 150},
        'races':        races_by_venue,
    }
