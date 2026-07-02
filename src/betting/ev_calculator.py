"""
期待値計算

シミュレーションで得た確率と実オッズを比較し、
期待値（EV）のある馬券を見つける。

EV = 的中確率 × オッズ
JRA控除率は約20-25%なので、EV > 1.25 程度を狙う目安。
"""


def calc_ev_all_tickets(probabilities, odds_map):
    """
    全馬券種のEVを計算する。

    Parameters
    ----------
    probabilities : dict
        calc_ticket_probabilities の出力
    odds_map : dict
        実オッズ {
          'win': {馬番(int): オッズ(float)},
          'place': {馬番(int): オッズ(float)},
          'quinella': {(a,b): オッズ},
          'exacta': {(a,b): オッズ},
          'trio': {(a,b,c): オッズ},
          'trifecta': {(a,b,c): オッズ},
        }
        対応するキーのみ計算。未提供の券種はスキップ。

    Returns
    -------
    dict: 券種 → [{key, prob, odds, ev}] (EV降順)
    """
    ev_results = {}
    bet_types = ['win', 'place', 'quinella', 'exacta', 'trio', 'trifecta']

    for bet_type in bet_types:
        probs = probabilities.get(bet_type, {})
        odds  = odds_map.get(bet_type, {})

        evs = []
        for key, prob in probs.items():
            o = odds.get(key)
            if o is None or o <= 0:
                continue
            evs.append({
                'key':  key,
                'prob': round(prob, 4),
                'odds': o,
                'ev':   round(prob * o, 3),
            })

        evs.sort(key=lambda x: x['ev'], reverse=True)
        ev_results[bet_type] = evs

    return ev_results


def select_value_bets(ev_results, min_ev=1.25, min_prob=0.01):
    """
    EVのある馬券だけを選ぶ。

    Parameters
    ----------
    min_ev   : EV下限（デフォルト1.25 ≈ 控除率分を超える期待値）
    min_prob : 的中確率下限（低すぎる買い目を除外）

    Returns
    -------
    dict: 条件を満たす馬券のみ（券種 → リスト）
    """
    value_bets = {}
    for bet_type, evs in ev_results.items():
        selected = [e for e in evs if e['ev'] >= min_ev and e['prob'] >= min_prob]
        if selected:
            value_bets[bet_type] = selected
    return value_bets
