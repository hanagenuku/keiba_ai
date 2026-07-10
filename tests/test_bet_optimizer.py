"""
bet_optimizer.py のユニットテスト

外部モデル不要。確率・オッズのモックデータで検証する。
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.bet_optimizer import (
    build_optimal_bets,
    determine_axis_structure,
    _calc_synthetic_odds,
    MIN_EV, MIN_PROB, TRIO_MIN_POINTS, TRIO_MAX_POINTS,
)


# ── テスト用モックデータ ───────────────────────────────────────────────────────

def _make_probs_and_odds(n=8):
    """n頭レースのダミー確率とオッズを生成する。"""
    from itertools import combinations
    nums = list(range(1, n + 1))

    rng = np.random.default_rng(42)
    raw = rng.exponential(1.0, n)
    raw /= raw.sum()

    probs = {
        'win':      {h: float(raw[i])       for i, h in enumerate(nums)},
        'place':    {h: min(0.9, float(raw[i] * 3)) for i, h in enumerate(nums)},
        'quinella': {},
        'trio':     {},
    }

    for a, b in combinations(nums, 2):
        pa, pb = raw[a-1], raw[b-1]
        probs['quinella'][(a, b)] = float(pa * pb * 5)

    for a, b, c in combinations(nums, 3):
        pa, pb, pc = raw[a-1], raw[b-1], raw[c-1]
        probs['trio'][(a, b, c)] = float(pa * pb * pc * 50)

    # オッズ = (1 - 控除率) / 確率 の近似
    def _to_odds(prob, takeout=0.225):
        return max(1.1, (1 - takeout) / max(prob, 1e-6))

    odds_map = {
        'win':      {h: _to_odds(raw[i], 0.20) for i, h in enumerate(nums)},
        'place':    {h: _to_odds(min(0.9, raw[i]*3), 0.20) for i, h in enumerate(nums)},
        'quinella': {k: _to_odds(v, 0.225) for k, v in probs['quinella'].items()},
        'trio':     {k: _to_odds(v, 0.225) for k, v in probs['trio'].items()},
    }

    # RL順位 = win確率降順
    ranked = sorted(range(n), key=lambda i: raw[i], reverse=True)
    rl_ranks = {}
    for rank, idx in enumerate(ranked, 1):
        rl_ranks[nums[idx]] = rank

    horses = [{'horse_num': h, 'win_odds': odds_map['win'][h], 'name': f'馬{h}',
               'rl_rank': rl_ranks[h]} for h in nums]
    return probs, odds_map, horses


# ── build_optimal_bets の基本動作 ─────────────────────────────────────────────

def test_returns_all_keys():
    """戻り値に win/place/quinella/trio/summary が含まれる。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    for key in ('win', 'place', 'quinella', 'trio', 'summary'):
        assert key in bets, f'キー {key} がない'


def test_trio_min_4_points():
    """三連複は最低 4 点（「3頭1点」禁止）。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['trio']) >= TRIO_MIN_POINTS, (
        f"三連複 {len(bets['trio'])} 点 < 最低 {TRIO_MIN_POINTS} 点"
    )


def test_trio_max_points():
    """三連複は最大 TRIO_MAX_POINTS 点。"""
    probs, odds_map, horses = _make_probs_and_odds(16)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['trio']) <= TRIO_MAX_POINTS


def test_trio_has_ev():
    """三連複の各買い目に ev キーがある。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    for b in bets['trio']:
        assert 'ev' in b and b['ev'] >= 0


def test_win_at_most_1_point():
    """単勝は最大 1 点。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['win']) <= 1


def test_place_at_most_2_points():
    """複勝は最大 2 点。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['place']) <= 2


def test_quinella_at_most_5_points():
    """馬連は最大 5 点。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['quinella']) <= 5


def test_summary_total_amount():
    """summary.total_amount == 全買い目点数 × 100。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    total_pts = sum(len(bets[bt]) for bt in ['win', 'place', 'quinella', 'trio'])
    assert bets['summary']['total_points'] == total_pts
    assert bets['summary']['total_amount'] == total_pts * 100


def test_each_bet_has_amount_100():
    """各買い目の amount が 100 円。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})
    for bt in ['win', 'place', 'quinella', 'trio']:
        for b in bets[bt]:
            assert b['amount'] == 100, f'{bt} の amount が 100 でない'


# ── RL上位ベースの選択 ────────────────────────────────────────────────────────

def test_win_rejects_low_rank_high_ev():
    """RL8の馬はEVが高くても単勝に選ばれない（旧設計の構造的欠陥テスト）。"""
    nums = list(range(1, 10))
    probs = {
        'win': {h: 0.12 - h * 0.005 for h in nums},
        'place': {h: 0.5 - h * 0.03 for h in nums},
        'quinella': {}, 'trio': {},
    }
    # RL8（馬番8）のオッズを極端に高く → EV最大だが、RL下位なので選ばれない
    odds_map = {
        'win': {h: 5.0 for h in nums},
        'place': {}, 'quinella': {}, 'trio': {},
    }
    odds_map['win'][8] = 50.0  # EV最大だがRL8
    odds_map['win'][1] = 3.0   # RL1だが本命すぎ

    horses = [{'horse_num': h, 'rl_rank': h, 'name': f'馬{h}'} for h in nums]
    bets = build_optimal_bets(probs, odds_map, horses, {})

    if bets['win']:
        selected = bets['win'][0]['key']
        assert selected in {1, 2, 3}, f"RL下位の {selected} が選ばれた"


def test_win_skips_favorite_picks_rl2():
    """RL1がオッズ1.5倍（本命すぎ）→ RL2-3から選ぶ。"""
    probs = {
        'win': {1: 0.30, 2: 0.20, 3: 0.15, 4: 0.10, 5: 0.08},
        'place': {}, 'quinella': {}, 'trio': {},
    }
    odds_map = {
        'win': {1: 1.5, 2: 5.0, 3: 8.0, 4: 15.0, 5: 20.0},
        'place': {}, 'quinella': {}, 'trio': {},
    }
    horses = [{'horse_num': h, 'rl_rank': h} for h in range(1, 6)]
    bets = build_optimal_bets(probs, odds_map, horses, {})

    if bets['win']:
        selected = bets['win'][0]['key']
        assert selected in {2, 3}, f"RL1(1.5倍)をスキップして{selected}を選ぶべき"


def test_place_ordered_by_rl_rank():
    """複勝はRL順位の高い方が先に選ばれる。"""
    probs = {
        'win': {}, 'quinella': {}, 'trio': {},
        'place': {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.15},
    }
    odds_map = {
        'win': {}, 'quinella': {}, 'trio': {},
        'place': {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0},
    }
    horses = [{'horse_num': h, 'rl_rank': h} for h in range(1, 6)]
    bets = build_optimal_bets(probs, odds_map, horses, {})

    if len(bets['place']) >= 2:
        rl1 = bets['place'][0]['key']
        rl2 = bets['place'][1]['key']
        assert rl1 < rl2, f"RL{rl1}がRL{rl2}より先に来るべき"


def test_quinella_includes_rl_top():
    """馬連はRL上位3頭の少なくとも1頭を含む。"""
    probs, odds_map, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, odds_map, horses, {})

    rl_top3 = {h['horse_num'] for h in horses if h['rl_rank'] <= 3}
    for b in bets['quinella']:
        if isinstance(b['key'], tuple):
            has_top = any(k in rl_top3 for k in b['key'])
        else:
            has_top = b['key'] in rl_top3
        assert has_top, f"馬連 {b['key']} にRL上位3頭がいない"


# ── 空入力・頭数不足 ───────────────────────────────────────────────────────────

def test_empty_odds_map_returns_structure():
    """odds_map が空でも構造が返る（全券種が空リストでok）。"""
    probs, _, horses = _make_probs_and_odds(8)
    bets = build_optimal_bets(probs, {}, horses, {})
    for key in ('win', 'place', 'quinella', 'trio', 'summary'):
        assert key in bets


def test_small_field_4_horses():
    """4頭フィールドでも trio は 4 点保証（trio は1通りしかないが補充される）。"""
    from itertools import combinations
    nums = [1, 2, 3, 4]
    raw  = np.array([0.4, 0.3, 0.2, 0.1])

    probs = {
        'win':   {h: float(raw[i]) for i, h in enumerate(nums)},
        'place': {h: float(raw[i] * 2.5) for i, h in enumerate(nums)},
        'quinella': {(a, b): float(raw[a-1] * raw[b-1] * 6) for a, b in combinations(nums, 2)},
        'trio':  {(1, 2, 3): 0.3, (1, 2, 4): 0.2, (1, 3, 4): 0.1, (2, 3, 4): 0.05},
    }
    odds_map = {
        'win':      {h: max(1.1, 0.8/raw[i])  for i, h in enumerate(nums)},
        'place':    {h: max(1.1, 0.8/(raw[i]*2.5)) for i, h in enumerate(nums)},
        'quinella': {k: max(2.0, 0.775/v) for k, v in probs['quinella'].items()},
        'trio':     {k: max(3.0, 0.775/v) for k, v in probs['trio'].items()},
    }

    horses = [{'horse_num': h, 'win_odds': odds_map['win'][h], 'name': f'馬{h}'} for h in nums]
    bets = build_optimal_bets(probs, odds_map, horses, {})
    assert len(bets['trio']) >= TRIO_MIN_POINTS


# ── _calc_synthetic_odds ──────────────────────────────────────────────────────

def test_synthetic_odds_empty():
    assert _calc_synthetic_odds([]) == 0.0


def test_synthetic_odds_single():
    combos = [{'prob': 0.1, 'odds': 10.0, 'ev': 1.0}]
    syn = _calc_synthetic_odds(combos)
    assert syn == pytest.approx(10.0)


def test_synthetic_odds_weighted():
    """確率加重平均オッズが正しく計算される。"""
    combos = [
        {'prob': 0.3, 'odds': 5.0, 'ev': 1.5},
        {'prob': 0.1, 'odds': 15.0, 'ev': 1.5},
    ]
    expected = (0.3 * 5.0 + 0.1 * 15.0) / (0.3 + 0.1)
    assert _calc_synthetic_odds(combos) == pytest.approx(expected)


# ── determine_axis_structure ──────────────────────────────────────────────────

def test_determine_axis_single():
    """1頭が突出していれば single_axis。"""
    probs = {'place': {1: 0.80, 2: 0.45, 3: 0.35, 4: 0.25, 5: 0.20}}
    structure, axes = determine_axis_structure(probs, [])
    assert structure == 'single_axis'
    assert axes == [1]


def test_determine_axis_double():
    """上位2頭が抜け、3頭目との差が大きければ double_axis。"""
    probs = {'place': {1: 0.65, 2: 0.60, 3: 0.40, 4: 0.25}}
    structure, axes = determine_axis_structure(probs, [])
    assert structure == 'double_axis'
    assert set(axes) == {1, 2}


def test_determine_axis_box():
    """拮抗していれば box。"""
    probs = {'place': {1: 0.50, 2: 0.48, 3: 0.45, 4: 0.42, 5: 0.40}}
    structure, axes = determine_axis_structure(probs, [])
    assert structure == 'box'
    assert len(axes) <= 5


# ── 軸構造が三連複に反映されるか ──────────────────────────────────────────────

def _make_single_axis_probs(n=10):
    """1頭が突出する確率分布を作成。"""
    from itertools import combinations
    nums = list(range(1, n + 1))
    # #1 が突出
    place_p = {1: 0.85}
    for i in range(2, n + 1):
        place_p[i] = 0.30 + (n - i) * 0.02
    win_p = {h: p / sum(place_p.values()) for h, p in place_p.items()}
    trio_p = {}
    for a, b, c in combinations(nums, 3):
        trio_p[(a, b, c)] = win_p[a] * win_p[b] * win_p[c] * 50
    trio_odds = {k: max(3.0, 0.775 / v) for k, v in trio_p.items()}
    return {
        'win': win_p,
        'place': place_p,
        'quinella': {},
        'trio': trio_p,
    }, {'trio': trio_odds}


def test_single_axis_all_combos_contain_axis():
    """single_axis のとき、全三連複が軸馬を含む。"""
    probs, odds_map = _make_single_axis_probs(10)
    bets = build_optimal_bets(probs, odds_map, [], {})
    for b in bets['trio']:
        assert 1 in b['key'], f"軸馬 #1 が {b['key']} に含まれていない"


def test_box_all_combos_within_box_set():
    """box のとき、全三連複がbox馬で構成される。"""
    from itertools import combinations
    nums = list(range(1, 9))
    # 拮抗した確率
    place_p = {h: 0.45 + (8 - h) * 0.01 for h in nums}
    win_p = {h: p / sum(place_p.values()) for h, p in place_p.items()}
    trio_p = {}
    for a, b, c in combinations(nums, 3):
        trio_p[(a, b, c)] = win_p[a] * win_p[b] * win_p[c] * 50
    trio_odds = {k: max(3.0, 0.775 / v) for k, v in trio_p.items()}
    probs = {'win': win_p, 'place': place_p, 'quinella': {}, 'trio': trio_p}
    odds = {'trio': trio_odds}

    structure, axis_nums = determine_axis_structure(probs, None)
    assert structure == 'box'
    box_set = set(axis_nums)

    bets = build_optimal_bets(probs, odds, [], {})
    for b in bets['trio']:
        assert set(b['key']).issubset(box_set), \
            f"combo {b['key']} が box {box_set} 外"
