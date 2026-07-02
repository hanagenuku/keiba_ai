"""
払戻推定（単勝オッズベース）

実際の確定オッズが単勝しかない場合、単勝オッズから
市場が暗示する各馬の勝率を逆算し、Gumbelシミュレーションで
全馬券種の理論配当を推定する。

注意: あくまで理論値。実配当とは 10〜30% ズレる。
ただし「モデルA vs モデルB」の相対比較には十分（両モデルに同じ誤差が乗る）。
"""

import numpy as np


# JRA の券種別控除率
_TAKEOUT = {
    'win':       0.200,   # 単勝
    'place':     0.200,   # 複勝
    'quinella':  0.225,   # 馬連
    'exacta':    0.225,   # 馬単
    'trio':      0.225,   # 三連複
    'trifecta':  0.275,   # 三連単
}


def estimate_payouts_from_win_odds(win_odds_map, n_sims=20000):
    """
    単勝オッズから全馬券種の理論配当を推定する。

    手順:
      1. 単勝オッズ → 市場勝率（正規化）
      2. logit 変換 → Gumbel 能力値
      3. simulate_race → 各馬券的中確率
      4. 理論配当 = (1 - 控除率) / 的中確率

    Parameters
    ----------
    win_odds_map : dict {馬番(int): 単勝オッズ(float)}
        例: {1: 3.5, 2: 12.0, 3: 5.6, ...}
    n_sims : int
        シミュレーション回数

    Returns
    -------
    dict  {
        'win':      {馬番: 推定配当},
        'place':    {馬番: 推定配当},
        'quinella': {(馬番A, 馬番B): 推定配当},
        'exacta':   {(1着, 2着): 推定配当},
        'trio':     {(a, b, c): 推定配当},
        'trifecta': {(1着, 2着, 3着): 推定配当},
    }
    オッズの低すぎる組合せ（確率 < 1e-6）は省略。
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities

    if not win_odds_map:
        return {bt: {} for bt in _TAKEOUT}

    nums  = list(win_odds_map.keys())
    odds  = np.array([win_odds_map[n] for n in nums], dtype=float)

    # 単勝オッズ → 市場勝率（正規化: 控除込みの合計を1に揃える）
    raw_prob    = 1.0 / odds
    market_prob = raw_prob / raw_prob.sum()

    # logit → Gumbel 能力値
    market_rating = np.log(np.clip(market_prob, 1e-9, 1.0))

    orders = simulate_race(market_rating, n_sims=n_sims)
    probs  = calc_ticket_probabilities(orders, nums)

    payouts = {}
    for bet_type, takeout in _TAKEOUT.items():
        payouts[bet_type] = {}
        for key, prob in probs.get(bet_type, {}).items():
            if prob > 1e-6:
                payouts[bet_type][key] = round((1.0 - takeout) / prob, 1)

    return payouts


def get_top_payout(payouts, bet_type, probs):
    """
    bet_type の「最も確率が高い買い目」の推定配当を返す。
    モデルの上位予測1点買い を想定したROI計算用。

    Parameters
    ----------
    payouts  : estimate_payouts_from_win_odds の戻り値
    bet_type : 'win' / 'place' / 'quinella' / 'trio' 等
    probs    : calc_ticket_probabilities の戻り値（モデルの確率）

    Returns
    -------
    top_key     : 最高確率の買い目
    est_payout  : 推定配当（取得できない場合は None）
    """
    bet_probs = probs.get(bet_type, {})
    if not bet_probs:
        return None, None

    top_key = max(bet_probs, key=bet_probs.get)
    est_payout = payouts.get(bet_type, {}).get(top_key)
    return top_key, est_payout
