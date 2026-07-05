"""
EVフィルタ：予想候補レースの厳選ロジック。
ノートブックの ability_first_loose を分離。
"""
import copy

from src.betting.make_bets import calc_ev
from src.utils.config import VENUE_ORDER

EV_THRESHOLD      = 1.05
ODDS_MIN          = 1.3
ODDS_MAX          = 30.0
WIN_PROB_MIN      = 0.06
SKIP_CLASSES      = ('未勝利', '新馬')
VALUE_EV_MIN      = 1.3   # バリュー判定の最低期待値（pn × win_odds）
VALUE_GAP_THRESHOLD = 0.10  # 後方互換で残存（フィルタには使用しない）

# ── 合成オッズ閾値 ────────────────────────────────────────────────────────
SYNTHETIC_SAFE_MIN = 2.5    # 2.5倍未満: 低配当警戒域
SYNTHETIC_SAFE_MAX = 6.0    # 6.0倍超 : 高配当警戒域
SYNTHETIC_SKIP_MIN = 1.5    # 1.5倍未満: 旨みなし → スキップ
SYNTHETIC_RISK_MAX = 10.0   # 10.0倍超 : 高リスク表示

# 各券種の相対的中確率（複勝を1.0基準とした近似値）
_REL_HIT_PROB = {
    '複勝':  1.0,
    '単勝':  0.8,
    'ワイド': 0.5,
    '馬連':  0.3,
    '馬単':  0.25,
    '三連複': 0.1,
}


def classify_race_chaos(scored):
    """pnベースの波乱度分類（A/B/C）。旧 classify_chaos_grade を置き換える。

    判定ルール（優先順）:
        gap_1_2 >= 0.10 かつ top3_sum >= 0.50 → 'A'（堅い）
        gap_1_2 <  0.04 または top3_sum < 0.32 または top1_pop >= 8 → 'C'（大荒れ）
        それ以外 → 'B'（中荒れ）

    Args:
        scored : calc_all が返す馬リスト。必須キー: pn（AI勝率）
                 popularity / _pop があれば top1_pop チェックに使用。

    Returns:
        'A' | 'B' | 'C'
    """
    if not scored:
        return 'B'

    # pn降順ソート（rl_rank=1 と一致するとは限らないが近似として使用）
    by_pn = sorted(scored, key=lambda h: h.get('pn', 0) or 0, reverse=True)
    probs = [h.get('pn', 0) or 0 for h in by_pn]

    gap_12  = probs[0] - probs[1] if len(probs) >= 2 else probs[0]
    top3_s  = sum(probs[:3])

    # pn最大の馬の人気（_pop > popularity の順で参照）
    top1    = by_pn[0]
    top1_pop = top1.get('_pop') or top1.get('popularity') or 99

    if gap_12 >= 0.10 and top3_s >= 0.50:
        return 'A'
    if gap_12 < 0.04 or top3_s < 0.32 or top1_pop >= 8:
        return 'C'
    return 'B'


def build_market_odds_from_races(races):
    """races の win_odds から market_odds_map を構築する。

    複勝オッズは単勝 × 0.344 で推定
    （JRA控除率: 複勝20%, 単勝22.5% → fuku ≈ tansho × 0.80/(3×0.775)）。

    Args:
        races : fetch_races_on_date が返すレースリスト

    Returns:
        {race_id: {horse_num: {'tansho': float, 'fukusho': float}}}
    """
    result = {}
    for race in races:
        odds_map = {}
        for h in race.get('horses', []):
            num = h.get('num') or h.get('horse_num')
            wo  = h.get('win_odds') or 0
            if num and wo > 0:
                odds_map[int(num)] = {
                    'tansho':  float(wo),
                    'fukusho': round(float(wo) * 0.344, 2),
                }
        if odds_map:
            result[race['id']] = odds_map
    return result


def detect_value_horses(horses, market_odds_map):
    """バリュー馬を検出する。

    EV（pn × win_odds）>= VALUE_EV_MIN をバリュー馬の基準とする（⑤）。
    value_gap（AI複勝確率 - 市場複勝逆算確率）も計算して付与するが、
    フィルタリングには使用しない（後方互換用）。

    Args:
        horses           : 各馬の予測結果。必須キー: horse_num, pn, win_odds
        market_odds_map  : {horse_num: fukusho_odds} または
                           {horse_num: {'tansho': float, 'fukusho': float}} の形式。
                           空dictの場合は value_gap=0.0 を全馬に設定して続行。

    Returns:
        is_value / value_gap / ev_direct を付与した馬リスト（EV降順ソート済み）。
    """
    result = []
    for h in horses:
        hnum      = h.get('horse_num', h.get('num'))
        fuku_prob = h.get('top3_prob') or h.get('pn', 0)
        wo        = h.get('win_odds', h.get('odds', 0)) or 0
        pn        = h.get('pn', 0) or 0

        # ── 市場オッズから value_gap（後方互換） ──────────────────────────
        odds = market_odds_map.get(hnum) if market_odds_map else None
        tansho_odds = None
        if isinstance(odds, dict):
            fuku_odds   = odds.get('fukusho')
            tansho_odds = odds.get('tansho')
        else:
            fuku_odds = odds

        if fuku_odds and fuku_odds >= 1.0:
            market_prob = 0.8 / fuku_odds
        else:
            market_prob = 0.0

        value_gap = round(fuku_prob - market_prob, 4) if market_odds_map else 0.0

        # ── EV ベースのバリュー判定（⑤ 主判定） ─────────────────────────
        direct_ev = round(pn * wo, 3)
        is_value  = direct_ev >= VALUE_EV_MIN

        entry = dict(h)
        entry['value_gap']    = value_gap
        entry['market_prob']  = round(market_prob, 4)
        entry['fukusho_odds'] = fuku_odds
        entry['tansho_odds']  = tansho_odds
        entry['ev_direct']    = direct_ev
        entry['is_value']     = is_value
        result.append(entry)

    result.sort(key=lambda x: x['ev_direct'], reverse=True)
    return result


def is_maiden_race(race):
    """新馬戦かどうか判定する（race_class / race_name の両方をチェック）。"""
    rc = race.get('race_class', race.get('class', '')) or ''
    rn = race.get('race_name', '') or ''
    return '新馬' in rc or '新馬' in rn


def calc_market_probs(horses):
    """全馬のオッズから市場確率を計算（JRA控除率補正込み）。"""
    raw = [1.0 / h.get('win_odds', 0) if (h.get('win_odds') or 0) > 0 else 0.0
           for h in horses]
    total = sum(raw) or 1.0
    return [p / total for p in raw]


def calc_value_score(ai_prob, market_prob, odds):
    """バリュースコアを計算する。

    Args:
        ai_prob     : AI勝率
        market_prob : 市場確率（calc_market_probs の出力）
        odds        : 単勝オッズ

    Returns:
        dict(prob_gap, ev, is_value)
    """
    if not market_prob or not odds:
        return {'prob_gap': 0.0, 'ev': None, 'is_value': False}
    prob_gap = ai_prob - market_prob
    ev = ai_prob * odds
    return {
        'prob_gap': round(prob_gap, 4),
        'ev':       round(ev, 3),
        'is_value': ev >= VALUE_EV_MIN,
    }


def ability_first_loose(races, bias_data=None, top_n=6):
    """純粋EV判断でレースを厳選する（オッズ上限 {ODDS_MAX} 倍）。"""
    from src.features.engine import calc_all, calc_chaos_score
    cands = []
    for race in races:
        scored = calc_all(race, bias_data)
        if len(scored) < 3:
            continue
        top1  = scored[0]
        odds  = top1.get('win_odds') or 99
        if odds < ODDS_MIN or odds > ODDS_MAX:
            continue
        rc = race.get('race_class', race.get('class', '')) or ''
        if any(s in rc for s in SKIP_CLASSES) or is_maiden_race(race):
            continue
        gap = top1['total'] - scored[1]['total']
        if gap < 0.005:
            continue
        win_prob = top1.get('pn', 0)
        if win_prob < WIN_PROB_MIN:
            continue

        fuku_prob = top1.get('top3_prob', min(0.80, win_prob * 3))
        ev_fuku = calc_ev(fuku_prob, odds * 0.28)
        ev_tan  = calc_ev(win_prob, odds)
        ev_max  = max(ev_fuku, ev_tan)
        if ev_max < EV_THRESHOLD:
            continue

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
            _h.setdefault('_pop', _rank)
        chaos_score_val = calc_chaos_score(race, scored)
        chaos_lvl = classify_race_chaos(scored)
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
            'chaos_score':     chaos_score_val,
            'chaos_level':     chaos_lvl,
        })

    cands.sort(key=lambda x: x['priority'], reverse=True)
    selected = cands[:top_n]
    selected.sort(key=lambda x: (
        VENUE_ORDER.get(x['race']['racecourse'], 99),
        x['race']['race_num'],
    ))
    return selected


def ability_first_with_value(races, bias_data=None, top_n=6):
    """バリュースコア（AI確率 vs 市場確率の乖離）を考慮してレースを厳選する。"""
    from src.features.engine import calc_all, calc_chaos_score
    cands = []
    for race in races:
        scored = calc_all(race, bias_data)
        if len(scored) < 3:
            continue
        rc2 = race.get('race_class', race.get('class', '')) or ''
        if any(s in rc2 for s in SKIP_CLASSES) or is_maiden_race(race):
            continue

        market_probs = calc_market_probs(scored)

        for i, horse in enumerate(scored):
            ai_prob = horse.get('pn', 0) or 0
            m_prob  = market_probs[i]
            odds    = horse.get('win_odds') or 0
            vs      = calc_value_score(ai_prob, m_prob, odds)
            horse['prob_gap'] = vs['prob_gap']
            horse['ev']       = vs['ev']
            horse['is_value'] = vs['is_value']

        value_horses = [h for h in scored if h.get('is_value')]
        if not value_horses:
            continue

        top1 = scored[0]
        odds = top1.get('win_odds') or 99
        gap  = top1['total'] - scored[1]['total']
        best_ev = max(h['ev'] for h in value_horses)

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
            _h.setdefault('_pop', _rank)

        ev_fuku = calc_ev(min(1.0, (top1.get('pn', 0) or 0) * 3), odds * 0.28)
        ev_tan  = calc_ev(top1.get('pn', 0) or 0, odds)
        chaos_score_val2 = calc_chaos_score(race, scored)

        cands.append({
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'priority':        best_ev * gap,
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          max(ev_fuku, ev_tan),
            'best_ev':         best_ev,
            'value_horses':    value_horses,
            'chaos_score':     chaos_score_val2,
            'chaos_level':     classify_race_chaos(scored),
        })

    cands.sort(key=lambda x: x['priority'], reverse=True)
    selected = cands[:top_n]
    selected.sort(key=lambda x: (
        VENUE_ORDER.get(x['race']['racecourse'], 99),
        x['race']['race_num'],
    ))
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  select_quality_races: 品質閾値ベースのレース厳選
# ══════════════════════════════════════════════════════════════════════════════

def select_quality_races(races, bias_data=None,
                         min_ev=1.30,
                         min_gap=0.03,
                         min_win_prob=0.10,
                         odds_range=(1.5, 20.0),
                         max_races=6,
                         min_races=0):
    """品質閾値を満たすレースだけを推奨する。

    gap は pn[0] - pn[1]（AI勝率差）で計算する（②）。
    波乱度は classify_race_chaos（pnベース）で判定（③）。
    EV は本命だけでなくフィールド内の最良馬から算出する。

    Parameters
    ----------
    min_ev       : フィールド最良馬のEV下限（デフォルト1.30）
    min_gap      : RL1位〜2位の pn 差下限（デフォルト0.03）
    min_win_prob : RL1位馬の AI勝率下限（デフォルト0.10）
    odds_range   : 本命オッズの許容範囲（デフォルト 1.5〜20.0）
    max_races    : 最大推奨レース数（デフォルト6）
    min_races    : 最小推奨レース数（デフォルト0 → 0件も許容）
    """
    from src.features.engine import calc_all, calc_chaos_score
    odds_min, odds_max = odds_range
    cands = []

    for race in races:
        scored = calc_all(race, bias_data)
        if len(scored) < 3:
            continue

        rc = race.get('race_class', race.get('class', '')) or ''
        if any(s in rc for s in SKIP_CLASSES) or is_maiden_race(race):
            continue

        top1 = scored[0]
        top2 = scored[1]

        # ② gap = pn差（AI勝率の差）
        pn1 = top1.get('pn', 0) or 0
        pn2 = top2.get('pn', 0) or 0
        gap = pn1 - pn2

        if gap < min_gap:
            continue

        odds = top1.get('win_odds') or 99
        if odds < odds_min or odds > odds_max:
            continue

        win_prob = pn1
        if win_prob < min_win_prob:
            continue

        # ── EVはフィールド全馬から最良を選択 ─────────────────────────────
        market_probs = calc_market_probs(scored)
        best_ev    = 0.0
        best_horse = top1
        for i, h in enumerate(scored):
            ai_prob = h.get('pn', 0) or 0
            m_prob  = market_probs[i]
            h_odds  = h.get('win_odds') or 0
            vs      = calc_value_score(ai_prob, m_prob, h_odds)
            h['prob_gap'] = vs['prob_gap']
            h['ev']       = vs['ev']
            h['is_value'] = vs['is_value']
            if (vs['ev'] or 0) > best_ev:
                best_ev    = vs['ev']
                best_horse = h

        fuku_prob = top1.get('top3_prob', min(0.80, win_prob * 3))
        ev_fuku   = calc_ev(fuku_prob, odds * 0.28)
        ev_tan    = calc_ev(win_prob, odds)
        ev_max    = max(best_ev, ev_fuku, ev_tan)

        if ev_max < min_ev:
            continue

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
            _h.setdefault('_pop', _rank)

        # ③ 波乱度は classify_race_chaos（pnベース）
        chaos_lvl = classify_race_chaos(scored)
        chaos_score_val = calc_chaos_score(race, scored)

        cands.append({
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,          # pn差（②変更後）
            'priority':        ev_max * (1 + gap * 10),
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          ev_max,
            'best_ev_horse':   best_horse.get('name', ''),
            'chaos_score':     chaos_score_val,
            'chaos_level':     chaos_lvl,
        })

    cands.sort(key=lambda x: x['priority'], reverse=True)
    selected = cands[:max_races]
    selected.sort(key=lambda x: (
        VENUE_ORDER.get(x['race']['racecourse'], 99),
        x['race']['race_num'],
    ))
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  合成オッズ計算・買い目調整
# ══════════════════════════════════════════════════════════════════════════════

def calc_synthetic_odds(bets, value_horses=None):
    """1レースの買い目全体の合成オッズを計算する。"""
    total_invest = sum(b['amount'] for b in bets)
    if total_invest == 0:
        return 0.0

    vh_by_num = {}
    if value_horses:
        for h in value_horses:
            num = h.get('horse_num', h.get('num'))
            if num is not None:
                vh_by_num[num] = h

    expected_return = 0.0
    for b in bets:
        bet_type = b['type']
        amount   = b['amount']
        rel_prob = _REL_HIT_PROB.get(bet_type, 0.3)

        est_payout = None
        if vh_by_num:
            try:
                from src.betting.make_bets import estimate_payout as _ep
                nums      = b.get('nums', [])[:3]
                odds_list = [vh_by_num.get(n, {}) for n in nums]
                est_payout = _ep(bet_type, odds_list)
            except Exception:
                pass

        if est_payout is not None:
            payout = amount * est_payout / 100
        elif bet_type == '複勝':
            odds_est = b.get('odds_est') or b.get('odds') or 0
            payout   = amount * max(1.1, odds_est * 0.28)
        else:
            odds_est = b.get('odds_est') or b.get('odds') or 0
            payout   = amount * max(1.0, odds_est)

        expected_return += payout * rel_prob

    return round(expected_return / total_invest, 3)


def adjust_bets_for_synthetic_odds(bets, value_horses=None):
    """合成オッズを計算し、範囲外なら金額を調整する（ソフト版）。"""
    bets = copy.deepcopy(bets)
    synthetic = calc_synthetic_odds(bets, value_horses)

    if synthetic <= 0:
        return bets, None

    if synthetic < SYNTHETIC_SKIP_MIN:
        return [], 'skip'

    if synthetic > SYNTHETIC_RISK_MAX:
        return bets, 'risk'

    if synthetic < SYNTHETIC_SAFE_MIN:
        for b in bets:
            if b['type'] == '複勝':
                b['amount'] = max(100, int(b['amount'] * 0.6 / 100) * 100)
            elif b['type'] in ('馬連', 'ワイド'):
                b['amount'] = max(100, int(b['amount'] * 1.5 / 100) * 100)
        new_syn = calc_synthetic_odds(bets, value_horses)
        return bets, ('low' if new_syn < 2.0 else None)

    if synthetic > SYNTHETIC_SAFE_MAX:
        for b in bets:
            if b['type'] in ('馬連', '三連複', '馬単'):
                b['amount'] = max(100, int(b['amount'] * 0.6 / 100) * 100)
            elif b['type'] == '複勝':
                b['amount'] = max(100, int(b['amount'] * 1.5 / 100) * 100)
        new_syn = calc_synthetic_odds(bets, value_horses)
        return bets, ('high' if new_syn > SYNTHETIC_SAFE_MAX else None)

    return bets, None
