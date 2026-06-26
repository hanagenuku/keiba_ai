"""
EVフィルタ：予想候補レースの厳選ロジック。
ノートブックの ability_first_loose を分離。
"""
import copy

from src.betting.make_bets import calc_ev, classify_chaos_grade
from src.features.engine import calc_all, calc_chaos_score
from src.utils.config import VENUE_ORDER

EV_THRESHOLD      = 1.05
ODDS_MIN          = 1.3
ODDS_MAX          = 30.0
WIN_PROB_MIN      = 0.06
SKIP_CLASSES      = ('未勝利', '新馬')
VALUE_EV_MIN      = 1.2   # バリュー判定の最低期待値
VALUE_GAP_THRESHOLD = 0.10  # detect_value_horses のバリューギャップ閾値

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


def detect_value_horses(horses, market_odds_map):
    """AI複勝確率と市場オッズから乖離（バリューギャップ）を計算する。

    Args:
        horses           : 各馬の予測結果。必須キー: horse_num, cal_prob
        market_odds_map  : {horse_num: fukusho_odds} または
                           {horse_num: {'tansho': float, 'fukusho': float}} の形式。
                           空dictの場合は value_gap=0.0 を全馬に設定して続行。

    Returns:
        value_gap（AI確率 - 市場逆算確率）を付与した馬リスト（降順ソート済み）。
        value_gap > VALUE_GAP_THRESHOLD の馬が「バリュー馬」。
    """
    result = []
    for h in horses:
        hnum  = h.get('horse_num', h.get('num'))
        # ⑤ 複勝確率は top3_prob（Harville）を正とする。cal_prob は別途保持。
        fuku_prob = h.get('top3_prob', h.get('cal_prob', h.get('pn', 0))) or 0.0
        odds  = market_odds_map.get(hnum) if market_odds_map else None

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
        entry = dict(h)
        entry['value_gap']   = value_gap
        entry['market_prob'] = round(market_prob, 4)
        entry['fukusho_odds'] = fuku_odds
        entry['tansho_odds']  = tansho_odds
        result.append(entry)

    result.sort(key=lambda x: x['value_gap'], reverse=True)
    return result


def classify_race_chaos(scored):
    """レースの波乱度をA/B/Cに分類する。

    A: 堅い（本命有力）  B: 中荒れ  C: 大荒れ（混戦）

    Args:
        scored: calc_all が返す馬リスト（win_prob付き）

    Returns:
        'A' | 'B' | 'C'
    """
    probs = sorted([h.get('win_prob', h.get('pn', 0)) for h in scored], reverse=True)
    if not probs:
        return 'B'
    gap_1_2 = probs[0] - probs[1] if len(probs) >= 2 else probs[0]
    top3_sum = sum(probs[:3])
    if gap_1_2 > 0.10 and top3_sum > 0.50:
        return 'A'
    if gap_1_2 < 0.03 or top3_sum < 0.35:
        return 'C'
    return 'B'


def is_maiden_race(race):
    """新馬戦かどうか判定する（race_class / race_name の両方をチェック）。"""
    rc = race.get('race_class', race.get('class', '')) or ''
    rn = race.get('race_name', '') or ''
    return '新馬' in rc or '新馬' in rn


def calc_market_probs(horses):
    """全馬のオッズから市場確率を計算（JRA控除率補正込み）。

    Args:
        horses: win_odds キーを持つ馬辞書のリスト

    Returns:
        市場確率リスト（horsesと同順）
    """
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
        return {'prob_gap': 0.0, 'ev': 0.0, 'is_value': False}
    prob_gap = ai_prob - market_prob
    ev = ai_prob * odds
    return {
        'prob_gap': round(prob_gap, 4),
        'ev':       round(ev, 3),
        'is_value': ev >= VALUE_EV_MIN,
    }


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
        # ④ 波乱度分類器を classify_chaos_grade に統一（popularity を win_odds 順位で補完）
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
        chaos_score_val = calc_chaos_score(race, scored)
        chaos_lvl = classify_chaos_grade(scored, chaos_score_val)
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
    """バリュースコア（AI確率 vs 市場確率の乖離）を考慮してレースを厳選する。

    EV >= VALUE_EV_MIN のバリュー馬が存在するレースを優先し、
    各馬に value_score / ev / prob_gap / is_value を付与する。

    Args:
        races     : fetch_races_on_date 等が返すレースリスト
        bias_data : 馬場バイアス辞書（省略可）
        top_n     : 最大厳選数

    Returns:
        ability_first_loose と同形式の候補辞書リスト
    """
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
            'chaos_level':     classify_chaos_grade(scored, chaos_score_val2),
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
                         min_ev=1.20,
                         min_gap=0.02,
                         min_win_prob=0.10,
                         odds_range=(1.5, 25.0),
                         max_races=10,
                         min_races=0):
    """品質閾値を満たすレースだけを推奨する。

    ability_first_loose() の厳格版。固定数ではなく質で切る（0〜max_races）。
    ability_first_with_value() のバリュー検出ロジックを統合：
    EVは本命だけでなくフィールド内の最良馬から算出する。

    priority の計算式: ev_max × (1 + gap × 10)
    （EVを主軸にし、gap は抜け感ボーナスとして乗算）

    Parameters
    ----------
    min_ev       : フィールド最良馬のEV下限（デフォルト1.20）
    min_gap      : RL1位〜2位の totalスコア差下限（デフォルト0.02）
    min_win_prob : RL1位馬の AI勝率下限（デフォルト0.10）
    odds_range   : 本命オッズの許容範囲（デフォルト 1.5〜25.0）
    max_races    : 最大推奨レース数（デフォルト10）
    min_races    : 最小推奨レース数（デフォルト0 → 0件も許容）
    """
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
        gap  = top1['total'] - top2['total']

        if gap < min_gap:
            continue

        odds = top1.get('win_odds') or 99
        if odds < odds_min or odds > odds_max:
            continue

        win_prob = top1.get('pn', 0)
        if win_prob < min_win_prob:
            continue

        # ── EVはフィールド全馬から最良を選択（ability_first_with_value 統合）──
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
            if vs['ev'] > best_ev:
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
        chaos_score_val3 = calc_chaos_score(race, scored)

        cands.append({
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'priority':        ev_max * (1 + gap * 10),   # EV主軸・gap ボーナス
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          ev_max,
            'best_ev_horse':   best_horse.get('name', ''),
            'chaos_score':     chaos_score_val3,
            'chaos_level':     classify_chaos_grade(scored, chaos_score_val3),
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
    """1レースの買い目全体の合成オッズを計算する。

    合成オッズ = 複勝ベースの期待リターン / 総投資額
    各券種の的中確率は複勝に対する相対値で近似する。

    Returns
    -------
    float: 合成オッズ（0.0 なら計算不能）
    """
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
                est_payout = _ep(bet_type, odds_list)    # 100円あたりの払戻額
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
    """合成オッズを計算し、範囲外なら金額を調整する（ソフト版）。

    調整方針:
      2.5〜6.0倍  : 調整不要
      1.5〜2.5倍  : 複勝を減額・馬連を増額して調整を試みる。
                     調整後も2倍未満なら note='low'（スキップしない）
      6.0〜10.0倍 : 馬連・三連複を減額・複勝を増額して調整を試みる。
                     調整後も6倍超なら note='high'（スキップしない）
      1.5倍未満   : 旨みなし → [] / note='skip'
      10.0倍超    : note='risk'（スキップしない）

    Returns
    -------
    (adjusted_bets, note)
      note: None / 'low' / 'high' / 'risk' / 'skip'
    """
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
