"""Plackett-Luce の閉形式（軸1頭ベース買い目が使う券種だけ）。

`race_simulator.simulate_race` は Gumbel-Max トリックで Plackett-Luce から
着順を**サンプリング**するが、必要な券種（単勝・ワイド・馬連・三連複）は
いずれも**厳密式で書ける**。このモジュールはその厳密式だけを持つ。

🔴 なぜ MC を厳密式に置き換えたか（2026-11-08）
----------------------------------------------------------------
① **同じ入力で実行ごとに答えが変わっていた。** `simulate_race` は
   `seed=None` なので、同じレースを3回組むと三連複の順序・表示される
   確率・推定配当・合成オッズが毎回変わる（実測: 2026-11-08 の34レースで
   「異なる順序」が5/5回）。点の集合は 33/34レースで一致していたので
   買う点そのものは安定していたが、**商品の画面に出る数字が再現しない**。
   2026-10-07 に脚質が PYTHONHASHSEED で変わっていたのを直したのと同じ扱い。
② **クライアント側（直前オッズ取得後）と一致させられない。** 乱数が違えば
   JS と Python は原理的に同じ値を出せず、「二重実装が黙って食い違う」
   （2026-08-09 に実際に起きた事故）を検出する手段が無い。厳密式なら
   `tests/test_axis_bets_exact.py` ↔ `tests/test_axis_bets_resync.mjs` で
   1e-9 まで突合でき、どちらかを変えればテストが落ちる。

⚠ 置き換えたのは**確率の計算方法だけ**。買い目の選び方（軸=RL1 /
  中穴=単勝7〜30倍の cal_prob 降順 / 合成オッズ3.0 / 点数上限）は
  `axis_bets.py` のまま1つも変えていない。閾値も動かしていない。

⚠ 回収率が上がる根拠はない。MC のばらつきを消しただけで、
  期待値は1ミリも動かない。
"""

import math


def weights_from_ratings(ratings):
    """rating（log空間の能力値）→ Plackett-Luce の重み。

    オーバーフロー対策に最大値を引く（確率は比なので不変）。
    """
    m = max(ratings) if ratings else 0.0
    return [math.exp(r - m) for r in ratings]


def weights_from_win_odds(odds):
    """単勝オッズ → 市場の重み。

    `payout_estimator.estimate_payouts_from_win_odds` と同じ経路:
      1/オッズ → 合計1に正規化 → log → Gumbel 能力値。
    exp(log(p)) = p なので、重みは正規化後の市場勝率そのもの。
    """
    inv = [(1.0 / o if o and o > 0 else 0.0) for o in odds]
    s = sum(inv)
    if s <= 0:
        return [0.0] * len(odds)
    return [x / s for x in inv]


def _p_order(w, total, idx):
    """P(1着=idx[0], 2着=idx[1], ...) を Plackett-Luce で厳密に。"""
    p, rest = 1.0, total
    for i in idx:
        if rest <= 0:
            return 0.0
        p *= w[i] / rest
        rest -= w[i]
    return p


def top2_set_prob(w, total, i, j):
    """P({i, j} が1-2着・順不同)。"""
    return _p_order(w, total, (i, j)) + _p_order(w, total, (j, i))


def top3_set_prob(w, total, i, j, k):
    """P({i, j, k} が1-2-3着・順不同) = 6通りの順列の和。"""
    return (_p_order(w, total, (i, j, k)) + _p_order(w, total, (i, k, j))
            + _p_order(w, total, (j, i, k)) + _p_order(w, total, (j, k, i))
            + _p_order(w, total, (k, i, j)) + _p_order(w, total, (k, j, i)))


def both_in_top3_prob(w, total, i, j):
    """P(i と j がともに3着内) = 3着の馬で場合分けして和を取る。"""
    s = 0.0
    for k in range(len(w)):
        if k == i or k == j:
            continue
        s += top3_set_prob(w, total, i, j, k)
    return s


def win_prob(w, total, i):
    """P(i が1着)。"""
    return (w[i] / total) if total > 0 else 0.0


# ── 券種別の確率・推定配当（axis_bets.py が読む形） ──────────────────────
# `race_simulator.calc_ticket_probabilities` /
# `payout_estimator.estimate_payouts_from_win_odds` と同じ辞書の形で返す。
# ただし `build_axis_bets` が実際に読む 'win' / 'wide' / 'quinella' / 'trio'
# だけを作る（'place' / 'exacta' / 'trifecta' は一度も読まれない）。

# JRA の券種別控除率（payout_estimator._TAKEOUT と同じ値。
# ⚠ 2箇所に分かれるので、変えるときは必ず両方直す）
TAKEOUT = {'quinella': 0.225, 'wide': 0.225, 'trio': 0.225}


def _ticket_probs(w, horse_nums):
    total = sum(w)
    n = len(horse_nums)
    out = {'win': {}, 'wide': {}, 'quinella': {}, 'trio': {}}
    if total <= 0 or n < 2:
        return out
    for i, num in enumerate(horse_nums):
        out['win'][int(num)] = win_prob(w, total, i)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted((int(horse_nums[i]), int(horse_nums[j])))
            out['quinella'][(a, b)] = top2_set_prob(w, total, i, j)
            out['wide'][(a, b)] = both_in_top3_prob(w, total, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                key = tuple(sorted((int(horse_nums[i]), int(horse_nums[j]),
                                    int(horse_nums[k]))))
                out['trio'][key] = top3_set_prob(w, total, i, j, k)
    return out


def exact_ticket_probabilities(ratings, horse_nums):
    """AIの rating から券種確率を厳密に出す（simulate_race の置き換え）。"""
    return _ticket_probs(weights_from_ratings(list(ratings)), list(horse_nums))


def exact_payouts_from_win_odds(win_odds_map):
    """単勝オッズから理論配当を厳密に出す（estimate_payouts... の置き換え）。

    床と丸めは `payout_estimator` と同一にしてある
    （JRAの払戻は元本100円を下回らないので 1.0 で床を張り、小数1桁に丸める）。
    """
    nums = [int(n) for n in win_odds_map]
    odds = [float(win_odds_map[n]) for n in win_odds_map]
    probs = _ticket_probs(weights_from_win_odds(odds), nums)
    out = {}
    for bet_type, takeout in TAKEOUT.items():
        out[bet_type] = {}
        for key, p in probs.get(bet_type, {}).items():
            if p > 1e-6:
                out[bet_type][key] = max(1.0, round((1.0 - takeout) / p, 1))
    return out
