"""
RL上位ベースの買い目最適化

AIのRL順位（win_prob順）を起点に、オッズ妙味のある馬を選ぶ。
「AIが上位評価した馬の中から、オッズに旨味がある馬を買う」設計。

旧設計（EV最大化）の問題:
    Gumbel T=2.5 で確率がフラット化 → EV = prob × odds が
    オッズに単調増加 → 常に最低人気を推す構造的欠陥。

新設計:
    1. RL上位馬（AIが強いと判断した馬）を候補とする
    2. その中でオッズが本命すぎない馬（妙味あり）を選ぶ
    3. EV は足切り（控除率超え）にのみ使う

使い方（Colab / スクリプト）:
    from src.betting.bet_optimizer import build_optimal_bets
    from src.betting.payout_estimator import estimate_payouts_from_win_odds

    win_odds = {h['horse_num']: h['win_odds'] for h in horses if h.get('win_odds')}
    odds_map = estimate_payouts_from_win_odds(win_odds)
    bets = build_optimal_bets(probs, odds_map, horses, race)
"""

import numpy as np

# EV足切り（控除率を下回る買い目を除外するだけ）
MIN_EV = {
    'win':      1.0,
    'place':    1.0,
    'quinella': 1.0,
    'wide':     1.0,
    'trio':     1.0,
}

# 券種別の最低的中確率（低すぎる買い目を除外）
MIN_PROB = {
    'win':      0.05,
    'place':    0.10,
    'quinella': 0.02,
    'wide':     0.05,
    'trio':     0.005,
}

# 単勝: 本命すぎるオッズ（1.x倍）は妙味なし
WIN_MIN_ODDS = 2.0
# 単勝: オッズ上限（穴すぎる馬は避ける）
WIN_MAX_ODDS = 30.0

TRIO_MIN_POINTS = 4
TRIO_MAX_POINTS = 15

SYN_ODDS_TARGET = (2.5, 6.0)  # 三連複合成オッズの目安

# Gumbelシミュレーションに入れる rating の温度。
# rating（XGBマージン）をそのまま使うと P(勝利)=softmax(rating, T=1) となり
# 過信する（フォワード実測: RL1平均35% vs 実勝率16%）。
# 2026-06-27〜07-04 の96レースで log-loss 最適だった T=2.5 をデフォルトとし、
# rating_temperature.json の "gumbel_rating" キーがあればそちらを優先する。
DEFAULT_GUMBEL_RATING_T = 2.5

_GUMBEL_T_CACHE = {}


def _load_gumbel_rating_temperature(base_dir):
    """rating_temperature.json から Gumbel 用温度を読む（キャッシュ付き）。"""
    if base_dir in _GUMBEL_T_CACHE:
        return _GUMBEL_T_CACHE[base_dir]
    T = DEFAULT_GUMBEL_RATING_T
    try:
        import os, json
        path = os.path.join(base_dir, 'data', 'rating_temperature.json')
        with open(path) as f:
            T = float(json.load(f)['calibration']['gumbel_rating']['T'])
    except Exception:
        pass
    _GUMBEL_T_CACHE[base_dir] = T
    return T


def build_optimal_bets(probs, odds_map, horses, race):
    """
    RL上位馬を起点に、オッズ妙味のある買い目を生成する。

    Parameters
    ----------
    probs    : シミュレーション確率（win/place/quinella/trio の確率）
               {bet_type: {key: float}}
    odds_map : 推定配当 or 実オッズ（estimate_payouts_from_win_odds と同じ構造）
               {bet_type: {key: float}}
    horses   : calc_all() の出力リスト（horse_num / win_odds / name / rl_rank 等）
    race     : レース辞書

    Returns
    -------
    dict: {
        'win':      [{'key', 'prob', 'odds', 'ev', 'amount'}],
        'place':    [...],
        'quinella': [...],
        'trio':     [...],
        'summary':  {'total_points', 'total_amount', 'payout_min', 'payout_max', 'syn_odds'},
    }
    """
    from src.betting.ev_calculator import calc_ev_all_tickets

    ev_results = calc_ev_all_tickets(probs, odds_map)

    rl_ranking = _build_rl_ranking(horses)

    UNIT = 100

    result = {
        'win':      _select_win(ev_results.get('win', []), UNIT, rl_ranking),
        'place':    _select_place(ev_results.get('place', []), UNIT, rl_ranking),
        'quinella': _select_quinella(ev_results.get('quinella', []), UNIT, rl_ranking),
        'trio':     _build_trio(
                        probs.get('trio', {}),
                        odds_map.get('trio', {}),
                        ev_results.get('trio', []),
                        UNIT,
                        probs=probs,
                        rl_ranking=rl_ranking,
                    ),
    }

    result['summary'] = _calc_summary(result, UNIT)
    return result


def _build_rl_ranking(horses):
    """horses リストから {horse_num: rl_rank} マップを作る。"""
    ranking = {}
    for h in horses:
        num = h.get('horse_num') or h.get('num')
        rl = h.get('rl_rank')
        if num is not None and rl is not None:
            ranking[num] = rl
    return ranking


def _select_win(ev_list, unit, rl_ranking):
    """
    単勝: RL上位3頭の中から、オッズに妙味がある馬を最大1点。

    - RL1が本命すぎ(2倍未満)ならスキップ → RL2-3から選ぶ
    - RL1-3のうちオッズ妙味がある馬(2〜30倍)をEV順で1点
    - 全員本命すぎ or 全員穴すぎなら買わない
    """
    rl_top3 = {num for num, rank in rl_ranking.items() if rank <= 3}
    if not rl_top3:
        rl_top3 = {e['key'] for e in sorted(ev_list, key=lambda x: x['prob'], reverse=True)[:3]}

    candidates = [e for e in ev_list
                  if e['key'] in rl_top3
                  and e['ev'] >= MIN_EV['win']
                  and e['prob'] >= MIN_PROB['win']
                  and WIN_MIN_ODDS <= e['odds'] <= WIN_MAX_ODDS]

    candidates.sort(key=lambda x: x['ev'], reverse=True)
    return [dict(e, amount=unit) for e in candidates[:1]]


def _select_place(ev_list, unit, rl_ranking):
    """
    複勝: RL上位5頭からEV足切りを通る馬を最大2点。

    RL順で優先し、EV >= 1.0 で足切り。
    """
    rl_top5 = {num for num, rank in rl_ranking.items() if rank <= 5}
    if not rl_top5:
        rl_top5 = {e['key'] for e in sorted(ev_list, key=lambda x: x['prob'], reverse=True)[:5]}

    candidates = [e for e in ev_list
                  if e['key'] in rl_top5
                  and e['ev'] >= MIN_EV['place']
                  and e['prob'] >= MIN_PROB['place']]

    candidates.sort(key=lambda x: rl_ranking.get(x['key'], 99))
    return [dict(e, amount=unit) for e in candidates[:2]]


def _select_quinella(ev_list, unit, rl_ranking):
    """
    馬連: RL上位5頭の組み合わせから、EV足切りを通るもの最大5点。

    少なくとも1頭がRL上位3頭に入る組み合わせを優先。
    """
    rl_top3 = {num for num, rank in rl_ranking.items() if rank <= 3}
    rl_top5 = {num for num, rank in rl_ranking.items() if rank <= 5}

    def _has_rl_horse(e):
        if isinstance(e['key'], tuple):
            return any(k in rl_top5 for k in e['key'])
        return e['key'] in rl_top5

    def _has_rl_top3(e):
        if isinstance(e['key'], tuple):
            return any(k in rl_top3 for k in e['key'])
        return e['key'] in rl_top3

    core = [e for e in ev_list
            if _has_rl_horse(e) and _has_rl_top3(e)
            and e['ev'] >= MIN_EV['quinella'] and e['prob'] >= MIN_PROB['quinella']]
    core.sort(key=lambda x: x['ev'], reverse=True)

    if len(core) < 3:
        extra = [e for e in ev_list
                 if _has_rl_horse(e)
                 and e['ev'] >= MIN_EV['quinella'] and e['prob'] >= MIN_PROB['quinella']
                 and e not in core]
        extra.sort(key=lambda x: x['ev'], reverse=True)
        core = core + extra

    return [dict(e, amount=unit) for e in core[:5]]


def _build_trio(trio_probs, trio_odds, ev_list, unit, probs=None,
                 rl_ranking=None):
    """
    三連複を軸構造ベースで組む。

    1. determine_axis_structure() で複勝確率から軸を判定
    2. 軸を含む組み合わせに限定してEV最大化
       - single_axis: 軸馬を含む組み合わせのみ → EV順
       - double_axis: 2頭軸を含む組み合わせのみ → EV順
       - box: 上位馬の全組み合わせ（EVフィルタなし）
    3. 軸が不明確（拮抗）ならボックスにする
    4. 相手は複勝確率8%以上の馬に限定（低確率の穴馬を除外）
    """
    if rl_ranking is None:
        rl_ranking = {}
    MIN_PARTNER_PLACE_PROB = 0.08

    # ── 軸構造の判定 ──
    if probs and probs.get('place'):
        structure, axis_nums = determine_axis_structure(probs, None)
    else:
        structure, axis_nums = 'list', []

    # ── 相手候補の複勝確率フィルタ ──
    place_probs = probs.get('place', {}) if probs else {}
    qualified_partners = set()
    if place_probs:
        for num, pp in place_probs.items():
            if pp >= MIN_PARTNER_PLACE_PROB:
                qualified_partners.add(num)
        for a in axis_nums:
            qualified_partners.add(a)

    def _combo_has_qualified_partners(e):
        if not qualified_partners:
            return True
        return all(n in qualified_partners for n in e['key'])

    # ── 軸に基づくフィルタリング ──
    if structure == 'single_axis' and axis_nums:
        axis = axis_nums[0]
        candidates = [e for e in ev_list
                      if axis in e['key'] and _combo_has_qualified_partners(e)]
    elif structure == 'double_axis' and len(axis_nums) >= 2:
        candidates = [e for e in ev_list
                      if all(a in e['key'] for a in axis_nums)
                      and _combo_has_qualified_partners(e)]
    elif structure == 'box' and axis_nums:
        box_set = set(axis_nums) & qualified_partners if qualified_partners else set(axis_nums)
        if len(box_set) < 3:
            box_set = set(axis_nums)
        candidates = [e for e in ev_list
                      if set(e['key']).issubset(box_set)]
    else:
        candidates = [e for e in ev_list if _combo_has_qualified_partners(e)]

    # ── 選択 ──
    if structure == 'box' and candidates:
        value = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    else:
        value = [e for e in candidates
                 if e['ev'] >= MIN_EV['trio'] and e['prob'] >= MIN_PROB['trio']]
        if len(value) < TRIO_MIN_POINTS:
            relaxed = [e for e in candidates
                       if e['ev'] >= 1.0 and e['prob'] >= MIN_PROB['trio']]
            if len(relaxed) >= len(value):
                value = relaxed

    # ── フォールバック（軸制約が厳しすぎる場合）──
    if len(value) < TRIO_MIN_POINTS:
        fallback_pool = [
            {'key': k, 'prob': p,
             'odds': trio_odds.get(k, 0),
             'ev':   p * trio_odds.get(k, 0)}
            for k, p in trio_probs.items()
            if trio_odds.get(k, 0) > 0 and _combo_has_qualified_partners(
                {'key': k})
        ]
        if len(fallback_pool) < TRIO_MIN_POINTS:
            fallback_pool = [
                {'key': k, 'prob': p,
                 'odds': trio_odds.get(k, 0),
                 'ev':   p * trio_odds.get(k, 0)}
                for k, p in trio_probs.items()
                if trio_odds.get(k, 0) > 0
            ]
        value = sorted(fallback_pool, key=lambda x: x['prob'],
                        reverse=True)[:TRIO_MIN_POINTS]

    value = value[:TRIO_MAX_POINTS]

    # ── 合成オッズ ──
    syn = _calc_synthetic_odds(value)
    syn_note = None
    if syn > 0:
        if syn < 1.5:
            syn_note = '低配当注意'
        elif syn > 12.0:
            syn_note = '高リスク'

    result = []
    for e in value:
        entry = dict(e, amount=unit, syn_odds=round(syn, 2))
        if syn_note:
            entry['syn_note'] = syn_note
        result.append(entry)

    return result


def _calc_synthetic_odds(combos):
    """
    合成オッズ = 確率加重平均オッズ。

    「この点数を全部買ったとき、当たれば平均何倍の配当か」の目安。
    """
    total_prob = sum(c['prob'] for c in combos)
    if total_prob <= 0:
        return 0.0
    return sum(c['odds'] * c['prob'] for c in combos) / total_prob


def _calc_summary(bets, unit):
    """合計投資額・想定配当レンジ・合成オッズ。"""
    total_pts = sum(len(bets.get(bt, [])) for bt in ['win', 'place', 'quinella', 'trio'])
    total_amt = total_pts * unit

    trio = bets.get('trio', [])
    if trio:
        payout_min = int(min(c['odds'] for c in trio) * unit)
        payout_max = int(max(c['odds'] for c in trio) * unit)
        syn_odds   = round(_calc_synthetic_odds(trio), 2)
    else:
        payout_min = payout_max = 0
        syn_odds = 0.0

    return {
        'total_points': total_pts,
        'total_amount': total_amt,
        'payout_min':   payout_min,
        'payout_max':   payout_max,
        'syn_odds':     syn_odds,
    }


# ── タスク3: 軸構造の自動判定（自信度ベース）─────────────────────────────────

def determine_axis_structure(probs, horses):
    """
    シミュレーション確率から三連複の軸構造を判定する。

    型を先に決めるのではなく、複勝確率の分布から導く。
    build_optimal_trio の EVが拮抗している場合に補助的に使う。

    Returns
    -------
    structure : 'single_axis' / 'double_axis' / 'box'
    axis_nums : 軸馬番リスト（box の場合は上位5頭）
    """
    place_probs = probs.get('place', {})
    sorted_horses = sorted(place_probs.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_horses) < 3:
        return 'box', [h[0] for h in sorted_horses]

    p1 = sorted_horses[0][1]
    p2 = sorted_horses[1][1]
    p3 = sorted_horses[2][1]

    gap_1_2 = p1 - p2
    gap_2_3 = p2 - p3

    if gap_1_2 >= 0.15:
        return 'single_axis', [sorted_horses[0][0]]
    elif gap_2_3 >= 0.10:
        return 'double_axis', [sorted_horses[0][0], sorted_horses[1][0]]
    else:
        return 'box', [h[0] for h in sorted_horses[:5]]


# ── タスク4: make_bets_v2（Gumbel確率ベース） ───────────────────────────────

def make_bets_v2(horses, race, base_dir, market_odds_map=None,
                 n_sims=20000, **_kwargs):
    """
    Gumbel確率ベースの買い目生成。

    Parameters
    ----------
    horses         : calc_all() の出力リスト（horse_num / rating / win_odds 必須）
    race           : レース辞書
    base_dir       : プロジェクトルート
    market_odds_map: {race_id: {horse_num: {'tansho', 'fukusho'}}} または None
    n_sims         : Gumbelシミュレーション回数

    Returns
    -------
    bets     : build_optimal_bets の出力
    probs    : シミュレーション確率 dict
    odds_map : 使用したオッズ dict
    meta     : dict
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
    from src.betting.payout_estimator import estimate_payouts_from_win_odds

    horse_nums = [h.get('horse_num') or h.get('num') for h in horses]
    meta = {}

    # ── シミュレーション ──────────────────────────────────────────────────────
    T = _load_gumbel_rating_temperature(base_dir)
    ratings = [h.get('rating', 0.0) / T for h in horses]
    orders  = simulate_race(ratings, n_sims=n_sims)
    probs   = calc_ticket_probabilities(orders, horse_nums)
    meta['T_gumbel'] = T

    # ── オッズ取得 ──────────────────────────────────────────────────────────
    race_id = race.get('id', '')
    real_odds = (market_odds_map or {}).get(race_id, {})

    if real_odds:
        win_odds = {num: info['tansho']
                    for num, info in real_odds.items()
                    if info.get('tansho', 0) > 0}
    else:
        win_odds = {h.get('horse_num') or h.get('num'): h.get('win_odds', 0) or 0
                    for h in horses}
        win_odds = {k: v for k, v in win_odds.items() if v > 0}

    odds_map = estimate_payouts_from_win_odds(win_odds, n_sims=min(n_sims, 10000))

    # ── 買い目生成 ─────────────────────────────────────────────────────────
    bets = build_optimal_bets(probs, odds_map, horses, race)
    return bets, probs, odds_map, meta
