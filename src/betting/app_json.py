"""
アプリ用 JSON 生成。
ノートブックの to_app_json を分離。
"""
from src.betting.ev_filter import VENUE_ORDER, calc_market_probs, calc_value_score
from src.betting.make_bets import calc_ev, make_bets
from src.features.engine import auto_comment, calc_all


def _build_horse_ranks(scored):
    """EV の各馬ランクを返す。"""
    ev_scores = []
    for h in scored:
        p = h.get('pn', 0.1)
        o = h.get('win_odds', 10) or 10
        ev_scores.append(round(min(0.88, p * 3.2) * max(1.2, o * 0.32), 3))
    return {scored[i]['num']: r + 1 for r, i in enumerate(
        sorted(range(len(scored)), key=lambda x: ev_scores[x], reverse=True))}
def _build_horses_list(scored, top1, by_odds):
    """アプリ表示用の馬リストを生成する（馬番順）。

    バリュースコアが付与済みの場合は EV ベースのマーク（高/推/穴）を使用。
    未付与の場合は旧ロジックにフォールバック。
    """
    ev_ranks = _build_horse_ranks(scored)

    # バリュースコア未付与の場合は計算する
    has_value_scores = any('ev' in h for h in scored)
    if not has_value_scores:
        market_probs = calc_market_probs(scored)
        for i, h in enumerate(scored):
            ai_prob = h.get('pn', 0) or 0
            m_prob  = market_probs[i]
            odds    = h.get('win_odds') or 0
            vs      = calc_value_score(ai_prob, m_prob, odds)
            h['prob_gap'] = vs['prob_gap']
            h['ev']       = vs['ev']
            h['is_value'] = vs['is_value']

    horses = []
    for h in scored:
        pop     = next((i + 1 for i, x in enumerate(by_odds) if x['name'] == h['name']), 99)
        pn      = h.get('pn', 0)
        wo      = h.get('win_odds', 0) or 0
        fuku_pct = round(min(88, pn * 3.2 * 100), 1)
        is_value = h.get('is_value', False)

        # バリューベースのマーク（高/推/穴）
        # 穴馬: AI複勝確率 > 市場複勝確率×1.5 かつ 人気8位以下
        market_probs_list = calc_market_probs(scored)
        mkt_fuku = market_probs_list[scored.index(h)] * 3.2  # 市場複勝確率の近似
        is_ana   = (pn * 3.2 > mkt_fuku * 1.5 and pop > 8)

        if is_value and pop <= 3:
            mark = '高'
        elif is_value and pop <= 8:
            mark = '推'
        elif is_value and is_ana:
            mark = '穴'
        else:
            mark = ''

        horses.append({
            'n':        h['num'],
            'name':     h['name'],
            'odds':     wo,
            'score':    round(h['total'], 1),
            'pop':      pop,
            'style':    h.get('running_style', '差し'),
            'tan_pct':  round(min(60,  pn * 100),       1),
            'ren_pct':  round(min(80,  pn * 2.0 * 100), 1),
            'fuku_pct': fuku_pct,
            'ev_rank':  ev_ranks.get(h['num'], 99),
            'rl_rank':  h.get('rl_rank', 99),
            'cl_rank':  h.get('cl_rank', 99),
            'ev':       h.get('ev', 0.0),
            'prob_gap': h.get('prob_gap', 0.0),
            'mark':     mark,
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
