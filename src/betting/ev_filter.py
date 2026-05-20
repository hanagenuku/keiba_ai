"""
EVフィルタ：予想候補レースの厳選ロジック。
ノートブックの ability_first_loose を分離。
"""
from src.betting.make_bets import calc_ev
from src.features.engine import calc_all, calc_chaos_score
from src.utils.config import VENUE_ORDER

EV_THRESHOLD = 1.05
ODDS_MIN     = 1.3
ODDS_MAX     = 20.0
WIN_PROB_MIN = 0.06
SKIP_CLASSES = ('未勝利', '新馬')


def ability_first_loose(races, bias_data=None, top_n=6):
    """純粋EV判断でレースを厳選する（オッズ上限 {ODDS_MAX} 倍）。

    Args:
        races     : fetch_races_on_date 等が返すレースリスト
        bias_data : 馬場バイアス辞書（省略可）
        top_n     : 最大厳選数

    Returns:
        候補辞書のリスト（競馬場・レース番号順に並べ替え済み）
    """
    cands = []
    for race in races:
        scored = calc_all(race, bias_data)
        if len(scored) < 3:
            continue
        top1  = scored[0]
        odds  = top1.get('win_odds') or 99
        if odds < ODDS_MIN or odds > ODDS_MAX:
            continue
        if race.get('class', '') in SKIP_CLASSES:
            continue
        gap = top1['total'] - scored[1]['total']
        if gap < 0.01:
            continue
        win_prob = top1.get('pn', 0)
        if win_prob < WIN_PROB_MIN:
            continue

        ev_fuku = calc_ev(min(1.0, win_prob * 3), odds * 0.28)
        ev_tan  = calc_ev(win_prob, odds)
        ev_max  = max(ev_fuku, ev_tan)
        if ev_max < EV_THRESHOLD:
            continue

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        cands.append({
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'priority':        ev_max * gap,
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          ev_max,
            'chaos_score':     calc_chaos_score(race, scored),
        })

    cands.sort(key=lambda x: x['priority'], reverse=True)
    selected = cands[:top_n]
    selected.sort(key=lambda x: (
        VENUE_ORDER.get(x['race']['racecourse'], 99),
        x['race']['race_num'],
    ))
    return selected
