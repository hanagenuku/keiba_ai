"""
レースシミュレーター（Plackett-Luce + Gumbel-Max）

能力値（rating）からレースをN回シミュレートし、
全馬券種（単勝・複勝・馬連・馬単・三連複・三連単）の的中確率を算出する。
"""

import numpy as np
from collections import Counter
from itertools import combinations


def simulate_race(ratings, n_sims=20000, seed=None):
    """
    能力値からレースをシミュレートする（Gumbel-Maxトリック）。

    Plackett-Luce分布に従う着順サンプルを生成する:
      能力値 + Gumbelノイズ を降順ソート → 1着〜最下位の着順

    Parameters
    ----------
    ratings : array-like
        各馬の能力値（大きいほど強い）
    n_sims : int
        シミュレーション回数

    Returns
    -------
    orders : np.ndarray (n_sims, n_horses)
        各シミュレーションの着順（馬のインデックス）
    """
    if seed is not None:
        np.random.seed(seed)

    ratings = np.asarray(ratings, dtype=float)
    n = len(ratings)

    # Gumbelノイズ: -log(-log(U)), U~Uniform(0,1)
    u = np.random.uniform(1e-12, 1.0, size=(n_sims, n))
    gumbel = -np.log(-np.log(u))

    scores = ratings[np.newaxis, :] + gumbel
    orders = np.argsort(-scores, axis=1)
    return orders


def calc_ticket_probabilities(orders, horse_nums):
    """
    シミュレーション結果から全馬券種の確率を集計する。

    Parameters
    ----------
    orders : np.ndarray (n_sims, n_horses)
        simulate_race の出力（各行が着順インデックス列）
    horse_nums : list of int
        各インデックスに対応する馬番

    Returns
    -------
    dict with keys: win / place / quinella / exacta / trio / trifecta
        各馬券の確率。キーは馬番（int）またはタプル。
    """
    n_sims = orders.shape[0]
    nums = np.array(horse_nums, dtype=int)

    # 着順インデックス → 馬番に変換
    order_nums = nums[orders]           # (n_sims, n_horses)
    first  = order_nums[:, 0]
    second = order_nums[:, 1]
    third  = order_nums[:, 2]
    top3   = order_nums[:, :3]

    result = {
        'win': {}, 'place': {}, 'quinella': {},
        'exacta': {}, 'trio': {}, 'trifecta': {},
    }

    # 単勝（1着確率）
    for num in horse_nums:
        result['win'][int(num)] = float((first == num).mean())

    # 複勝（3着以内確率）
    for num in horse_nums:
        result['place'][int(num)] = float((top3 == num).any(axis=1).mean())

    # 馬連（1-2着、順不同）— ベクトル化
    top2_sorted = np.sort(order_nums[:, :2], axis=1)
    for a, b in combinations(sorted(horse_nums), 2):
        mask = (top2_sorted[:, 0] == a) & (top2_sorted[:, 1] == b)
        p = float(mask.mean())
        if p > 0:
            result['quinella'][(a, b)] = p

    # 馬単（1-2着、順序あり）— ベクトル化
    for a in horse_nums:
        for b in horse_nums:
            if a == b:
                continue
            p = float(((first == a) & (second == b)).mean())
            if p > 0:
                result['exacta'][(int(a), int(b))] = p

    # 三連複（1-2-3着、順不同）— ベクトル化で高速化
    top3_sorted = np.sort(top3, axis=1)   # (n_sims, 3)
    for a, b, c in combinations(sorted(horse_nums), 3):
        mask = ((top3_sorted[:, 0] == a) &
                (top3_sorted[:, 1] == b) &
                (top3_sorted[:, 2] == c))
        p = float(mask.mean())
        if p > 0:
            result['trio'][(a, b, c)] = p

    # 三連単（1-2-3着、順序あり）— Counterで全シム集計
    trifecta_cnt = Counter(
        (int(order_nums[i, 0]), int(order_nums[i, 1]), int(order_nums[i, 2]))
        for i in range(n_sims)
    )
    for combo, cnt in trifecta_cnt.items():
        result['trifecta'][combo] = cnt / n_sims

    return result
