"""
馬の能力タイプ特徴量

過去走から「この馬はどういう武器を持つ馬か」を算出し、
今日の条件との相性を判定する。

重要（データリーク対策）:
  全ての算出は「対象レース日より前」の過去走のみを使用する。
  当日の結果・タイムは絶対に使わない。
"""

import numpy as np
from collections import defaultdict

DIST_SPRINT = (1000, 1400)
DIST_MILE = (1401, 1800)
DIST_MIDDLE = (1801, 2200)
DIST_LONG = (2201, 4000)


def calc_agari_ability(history):
    """末脚の強さ（上がり順位の相対値の平均、0〜1、高いほど強い）"""
    ranks = []
    for h in history:
        ar = h.get('agari_rank')
        fs = h.get('field_size') or h.get('finishers') or h.get('num_finishers')
        if ar and fs and fs > 1:
            ranks.append(1.0 - (ar - 1) / (fs - 1))

    if not ranks:
        return 0.5

    return round(float(np.mean(ranks)), 3)


def calc_stamina_score(history):
    """スタミナ指標（長距離での相対成績、0〜1、高いほどスタミナ型）"""
    long_results = []
    short_results = []

    for h in history:
        d = h.get('distance')
        p = h.get('place')
        fs = h.get('field_size') or h.get('finishers') or h.get('num_finishers')
        if not d or not p or p <= 0 or not fs or fs < 2:
            continue

        score = 1.0 - (p - 1) / (fs - 1)

        if d >= DIST_MIDDLE[0]:
            long_results.append(score)
        elif d <= DIST_SPRINT[1]:
            short_results.append(score)

    if not long_results:
        return 0.5

    long_avg = np.mean(long_results)

    if short_results:
        short_avg = np.mean(short_results)
        diff = long_avg - short_avg
        return round(float(np.clip(0.5 + diff, 0.0, 1.0)), 3)

    return round(float(long_avg), 3)


def calc_speed_score(history):
    """スピード指標（短距離〜マイルでの相対成績、0〜1、高いほどスピード型）"""
    short_results = []
    long_results = []

    for h in history:
        d = h.get('distance')
        p = h.get('place')
        fs = h.get('field_size') or h.get('finishers') or h.get('num_finishers')
        if not d or not p or p <= 0 or not fs or fs < 2:
            continue

        score = 1.0 - (p - 1) / (fs - 1)

        if d <= DIST_MILE[1]:
            short_results.append(score)
        elif d >= DIST_MIDDLE[0]:
            long_results.append(score)

    if not short_results:
        return 0.5

    short_avg = np.mean(short_results)

    if long_results:
        long_avg = np.mean(long_results)
        diff = short_avg - long_avg
        return round(float(np.clip(0.5 + diff, 0.0, 1.0)), 3)

    return round(float(short_avg), 3)


def calc_optimal_distance(history):
    """最適距離と信頼度を返す。200m刻みでビン化し、成績加重で最良帯を選ぶ。

    Returns (optimal_dist: int, confidence: float 0〜1)
    """
    dist_scores = defaultdict(list)

    for h in history:
        d = h.get('distance')
        p = h.get('place')
        fs = h.get('field_size') or h.get('finishers') or h.get('num_finishers')
        if not d or not p or p <= 0 or not fs or fs < 2:
            continue

        score = 1.0 - (p - 1) / (fs - 1)
        bin_dist = round(d / 200) * 200
        dist_scores[bin_dist].append(score)

    if not dist_scores:
        return 1600, 0.0

    best_dist = None
    best_score = -1
    total_n = sum(len(v) for v in dist_scores.values())

    for d, scores in dist_scores.items():
        n = len(scores)
        avg = np.mean(scores)
        weighted = avg * (n / (n + 1))
        if weighted > best_score:
            best_score = weighted
            best_dist = d

    confidence = len(dist_scores[best_dist]) / total_n if total_n > 0 else 0.0

    return int(best_dist), round(float(confidence), 3)


def calc_distance_features(horse, race, history):
    """距離適性に関する特徴量をまとめて計算する（9個）。

    history は対象レース日より前の過去走のみを渡すこと（データリーク対策）。
    """
    today_dist = race.get('distance', 1600)

    agari = calc_agari_ability(history)
    stamina = calc_stamina_score(history)
    speed = calc_speed_score(history)
    optimal_dist, dist_conf = calc_optimal_distance(history)

    dist_vs_optimal = today_dist - optimal_dist

    prev_dist = history[0].get('distance') if history else today_dist
    dist_change = today_dist - (prev_dist or today_dist)

    if dist_change < 0:
        shortening = min(abs(dist_change) / 400.0, 1.0)
        speed_x_short = speed * shortening
    else:
        speed_x_short = 0.0

    if dist_change > 0:
        extension = min(dist_change / 400.0, 1.0)
        stamina_x_ext = stamina * extension
    else:
        stamina_x_ext = 0.0

    return {
        'f_agari_ability': agari,
        'f_stamina_score': stamina,
        'f_speed_score': speed,
        'f_optimal_distance': optimal_dist,
        'f_dist_vs_optimal': dist_vs_optimal,
        'f_dist_change': dist_change,
        'f_speed_x_shortening': round(speed_x_short, 3),
        'f_stamina_x_extension': round(stamina_x_ext, 3),
        'f_dist_confidence': dist_conf,
    }
