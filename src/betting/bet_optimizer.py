"""
期待値ベースの買い目最適化

Gumbelシミュレーションの確率と実オッズ（or推定オッズ）から、
型に縛られず期待値最大の買い目を券種横断で生成する。

使い方（Colab / スクリプト）:
    from src.betting.bet_optimizer import build_optimal_bets
    from src.betting.payout_estimator import estimate_payouts_from_win_odds

    win_odds = {h['horse_num']: h['win_odds'] for h in horses if h.get('win_odds')}
    odds_map = estimate_payouts_from_win_odds(win_odds)
    bets = build_optimal_bets(probs, odds_map, horses, race)
"""

import numpy as np

# 券種別の最低期待値（控除率を超える水準）
MIN_EV = {
    'win':      1.15,   # 単勝（控除20%）
    'place':    1.10,   # 複勝（控除20%、堅いので低め）
    'quinella': 1.25,   # 馬連（控除22.5%）
    'wide':     1.20,   # ワイド（未実装、将来用）
    'trio':     1.30,   # 三連複（控除22.5%、荒れるので高め）
}

# 券種別の最低的中確率（低すぎる買い目を除外）
MIN_PROB = {
    'win':      0.05,
    'place':    0.15,
    'quinella': 0.02,
    'wide':     0.05,
    'trio':     0.005,
}

TRIO_MIN_POINTS = 4
TRIO_MAX_POINTS = 20

SYN_ODDS_TARGET = (2.5, 6.0)  # 三連複合成オッズの目安


def build_optimal_bets(probs, odds_map, horses, race):
    """
    期待値最大の買い目を券種横断で生成する。

    Parameters
    ----------
    probs    : build_dual_probs の出力（win/place/quinella/trio の確率）
               {bet_type: {key: float}}
    odds_map : 推定配当 or 実オッズ（estimate_payouts_from_win_odds と同じ構造）
               {bet_type: {key: float}}
    horses   : calc_all() の出力リスト（horse_num / win_odds / name 等）
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

    UNIT = 100

    result = {
        'win':      _select_win(ev_results.get('win', []), UNIT),
        'place':    _select_place(ev_results.get('place', []), UNIT),
        'quinella': _select_quinella(ev_results.get('quinella', []), UNIT),
        'trio':     _build_trio(
                        probs.get('trio', {}),
                        odds_map.get('trio', {}),
                        ev_results.get('trio', []),
                        UNIT,
                        probs=probs,
                    ),
    }

    result['summary'] = _calc_summary(result, UNIT)
    return result


def _select_win(ev_list, unit):
    """単勝: EV最大の1点（基準を満たせば）。"""
    hits = [e for e in ev_list
            if e['ev'] >= MIN_EV['win'] and e['prob'] >= MIN_PROB['win']]
    return [dict(e, amount=unit) for e in hits[:1]]


def _select_place(ev_list, unit):
    """複勝: EV基準を満たす馬、最大2点。"""
    hits = [e for e in ev_list
            if e['ev'] >= MIN_EV['place'] and e['prob'] >= MIN_PROB['place']]
    return [dict(e, amount=unit) for e in hits[:2]]


def _select_quinella(ev_list, unit):
    """馬連: EV上位5点まで。"""
    hits = [e for e in ev_list
            if e['ev'] >= MIN_EV['quinella'] and e['prob'] >= MIN_PROB['quinella']]
    return [dict(e, amount=unit) for e in hits[:5]]


def _build_trio(trio_probs, trio_odds, ev_list, unit, probs=None):
    """
    三連複を軸構造ベースで組む。

    1. determine_axis_structure() で複勝確率から軸を判定
    2. 軸を含む組み合わせに限定してEV最大化
       - single_axis: 軸馬を含む組み合わせのみ → EV順
       - double_axis: 2頭軸を含む組み合わせのみ → EV順
       - box: 上位馬の全組み合わせ（EVフィルタなし）
    3. 軸が不明確（拮抗）ならボックスにする
    """
    # ── 軸構造の判定 ──
    if probs and probs.get('place'):
        structure, axis_nums = determine_axis_structure(probs, None)
    else:
        structure, axis_nums = 'list', []

    # ── 軸に基づくフィルタリング ──
    if structure == 'single_axis' and axis_nums:
        axis = axis_nums[0]
        candidates = [e for e in ev_list if axis in e['key']]
    elif structure == 'double_axis' and len(axis_nums) >= 2:
        candidates = [e for e in ev_list
                      if all(a in e['key'] for a in axis_nums)]
    elif structure == 'box' and axis_nums:
        box_set = set(axis_nums)
        candidates = [e for e in ev_list
                      if set(e['key']).issubset(box_set)]
    else:
        candidates = ev_list

    # ── 選択 ──
    if structure == 'box' and candidates:
        # BOX: 全組み合わせを採用（中途半端にEVで絞らない）
        value = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    else:
        # 軸ベース: 軸内でEV順に選択
        value = [e for e in candidates
                 if e['ev'] >= MIN_EV['trio'] and e['prob'] >= MIN_PROB['trio']]
        if len(value) < TRIO_MIN_POINTS:
            relaxed = [e for e in candidates
                       if e['ev'] >= 1.0 and e['prob'] >= MIN_PROB['trio']]
            if len(relaxed) >= len(value):
                value = relaxed

    # ── フォールバック（軸制約が厳しすぎる場合）──
    if len(value) < TRIO_MIN_POINTS:
        value = sorted(
            [{'key': k, 'prob': p,
              'odds': trio_odds.get(k, 0),
              'ev':   p * trio_odds.get(k, 0)}
             for k, p in trio_probs.items()
             if trio_odds.get(k, 0) > 0],
            key=lambda x: x['prob'],
            reverse=True
        )[:TRIO_MIN_POINTS]

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
                 feat_df=None, n_sims=20000):
    """
    Gumbel確率ベースの新しい買い目生成。

    既存の make_bets() とは独立したパス。段階移行用。

    Parameters
    ----------
    horses         : calc_all() の出力リスト（horse_num / rating / win_odds 必須）
    race           : レース辞書
    base_dir       : プロジェクトルート（dual_model 用）
    market_odds_map: {race_id: {horse_num: {'tansho', 'fukusho'}}} または None
    feat_df        : XGB特徴量 DataFrame（horse_features.csv 行形式）
                     渡すと dual_model（B2_ndcg で単勝）が有効になる
                     None のとき horses['rating']（ルールベース）で単一シミュレーション
    n_sims         : Gumbelシミュレーション回数

    Returns
    -------
    bets     : build_optimal_bets の出力
    probs    : シミュレーション確率 dict
    odds_map : 使用したオッズ dict
    meta     : {'dual': bool, 'b2_available': bool, 'T_A', 'T_B2'}
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
    from src.betting.payout_estimator import estimate_payouts_from_win_odds

    horse_nums = [h.get('horse_num') or h.get('num') for h in horses]
    meta = {'dual': False, 'b2_available': False, 'T_A': None, 'T_B2': None}

    # ── シミュレーション ──────────────────────────────────────────────────────
    if feat_df is not None:
        from src.betting.dual_model import build_dual_probs
        probs, dm_meta = build_dual_probs(feat_df, horse_nums, base_dir, n_sims)
        meta.update({'dual': True,
                     'b2_available': dm_meta['b2_available'],
                     'T_A': dm_meta['T_A'],
                     'T_B2': dm_meta['T_B2']})
    else:
        # horses['rating'] はルールベース総合スコア → 直接使用
        ratings = [h.get('rating', 0.0) for h in horses]
        orders  = simulate_race(ratings, n_sims=n_sims)
        probs   = calc_ticket_probabilities(orders, horse_nums)

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
