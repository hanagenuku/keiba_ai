"""軸1頭ベースの買い目（axis_bets.py）の検証。

North Star #6 に従い、手打ちの理想dictではなく **calc_all が実際に返す形**
（rl_rank / win_odds / cal_prob / rating を持つ馬辞書）で検証する。
"""
import pytest

from src.betting.axis_bets import (
    build_axis_bets, select_axis, select_anas, synthetic_odds,
    QUINELLA_MIN_SYNTHETIC_ODDS, TRIO_MAX_POINTS, ANA_ODDS_MIN, ANA_ODDS_MAX,
)


def _horse(num, rl, odds, cal):
    """calc_all の出力と同じキー構成の馬辞書。"""
    return {'num': num, 'name': f'馬{num}', 'rl_rank': rl,
            'win_odds': odds, 'cal_prob': cal, 'rating': 0.0}


@pytest.fixture
def field():
    """軸1頭＋中穴帯に複数頭いる12頭立て。"""
    return [
        _horse(1, 1, 2.5, 0.62),    # 軸（RL1・人気）
        _horse(2, 2, 4.0, 0.51),
        _horse(3, 3, 6.0, 0.44),
        _horse(4, 4, 9.0, 0.38),    # 中穴（3着内確率がこの帯で最上位）
        _horse(5, 5, 12.0, 0.33),   # 中穴
        _horse(6, 6, 20.0, 0.25),   # 中穴
        _horse(7, 7, 28.0, 0.20),   # 中穴
        _horse(8, 8, 45.0, 0.12),   # 帯の外（穴すぎ）
        _horse(9, 9, 60.0, 0.10),
        _horse(10, 10, 80.0, 0.08),
        _horse(11, 11, 5.0, 0.40),  # オッズは中穴帯の外（人気すぎ）
        _horse(12, 12, 120.0, 0.05),
    ]


def _probs_odds(field, *, wide=0.30, quin=0.08, trio=0.02, win=0.35,
                wide_o=6.0, quin_o=12.0, trio_o=40.0):
    """simulate 相当の確率・推定配当を、組み合わせ全部に一律で用意する。"""
    from itertools import combinations
    nums = [h['num'] for h in field]
    probs = {'win': {n: win for n in nums}, 'place': {}, 'wide': {},
             'quinella': {}, 'exacta': {}, 'trio': {}, 'trifecta': {}}
    odds = {'win': {}, 'place': {}, 'wide': {}, 'quinella': {},
            'exacta': {}, 'trio': {}, 'trifecta': {}}
    for a, b in combinations(sorted(nums), 2):
        probs['wide'][(a, b)] = wide;      odds['wide'][(a, b)] = wide_o
        probs['quinella'][(a, b)] = quin;  odds['quinella'][(a, b)] = quin_o
    for t in combinations(sorted(nums), 3):
        probs['trio'][t] = trio;           odds['trio'][t] = trio_o
    return probs, odds


class TestAxisAndAna:
    def test_axis_is_rl1(self, field):
        assert select_axis(field)['num'] == 1

    def test_ana_is_within_odds_band_and_sorted_by_fuku_prob(self, field):
        anas = select_anas(field, axis_num=1, n=3)
        nums = [h['num'] for h in anas]
        assert nums == [4, 5, 6], f'3着内確率順に中穴が並んでいない: {nums}'
        for h in anas:
            assert ANA_ODDS_MIN <= h['win_odds'] < ANA_ODDS_MAX

    def test_ana_excludes_favorites_and_extreme_longshots(self, field):
        nums = [h['num'] for h in select_anas(field, axis_num=1, n=99)]
        assert 11 not in nums, '5.0倍（人気すぎ）が中穴に混ざっている'
        assert 8 not in nums and 12 not in nums, '45倍以上（穴すぎ）が混ざっている'
        assert 1 not in nums, '軸自身が中穴候補に入っている'


class TestSyntheticOdds:
    def test_matches_hand_calculation(self):
        # 1/(1/10 + 1/15 + 1/20) = 4.615...
        assert synthetic_odds([10, 15, 20]) == pytest.approx(4.6154, abs=1e-3)

    def test_adding_a_short_price_drags_it_down(self):
        assert synthetic_odds([10, 15, 20, 4]) < synthetic_odds([10, 15, 20])


class TestBuildAxisBets:
    def test_win_is_axis_only(self, field):
        p, o = _probs_odds(field)
        b = build_axis_bets(field, p, o)
        assert [x['key'] for x in b['win']] == [1]

    def test_wide_is_axis_times_ana_capped(self, field):
        p, o = _probs_odds(field)
        b = build_axis_bets(field, p, o, wide_points=3)
        assert [x['key'] for x in b['wide']] == [(1, 4), (1, 5), (1, 6)]
        b1 = build_axis_bets(field, p, o, wide_points=1)
        assert [x['key'] for x in b1['wide']] == [(1, 4)], '1点指定が効いていない'

    def test_quinella_keeps_synthetic_odds_above_threshold(self, field):
        # 12倍を4点買うと合成3.0倍ちょうど。5点目で3.0を割るので止まるはず
        p, o = _probs_odds(field, quin_o=12.0)
        b = build_axis_bets(field, p, o)
        picked = [x['odds'] for x in b['quinella']]
        assert picked, '馬連が1点も出ていない'
        assert synthetic_odds(picked) >= QUINELLA_MIN_SYNTHETIC_ODDS
        assert b['summary']['quinella_syn_odds'] >= QUINELLA_MIN_SYNTHETIC_ODDS

    def test_quinella_skipped_when_even_one_point_is_too_short(self, field):
        """1点だけで合成2.0倍しか無いなら、取りガミ側なので買わない。"""
        p, o = _probs_odds(field, quin_o=2.0)
        b = build_axis_bets(field, p, o)
        assert b['quinella'] == []

    def test_trio_second_leg_is_ana_not_favorites(self, field):
        """全ての三連複が『軸 ＋ 中穴のどれか』を含むこと。

        2列目に上位人気(RL2,RL3)を置くと実測で崩れる（56.3%）ため、
        中穴が必ず1頭入る構造であることを固定する。
        """
        p, o = _probs_odds(field)
        b = build_axis_bets(field, p, o)
        assert b['trio'], '三連複が組めていない'
        ana2 = {h['num'] for h in select_anas(field, 1, 2)}
        for x in b['trio']:
            assert 1 in x['key'], f'軸が入っていない: {x["key"]}'
            assert ana2 & set(x['key']), f'2列目の中穴が入っていない: {x["key"]}'

    def test_trio_respects_max_points(self, field):
        p, o = _probs_odds(field)
        b = build_axis_bets(field, p, o, trio_max_points=5)
        assert len(b['trio']) == 5
        b2 = build_axis_bets(field, p, o)
        assert len(b2['trio']) <= TRIO_MAX_POINTS

    def test_no_duplicate_combinations(self, field):
        p, o = _probs_odds(field)
        b = build_axis_bets(field, p, o)
        for k in ('wide', 'quinella', 'trio'):
            keys = [x['key'] for x in b[k]]
            assert len(keys) == len(set(keys)), f'{k} に重複がある'

    def test_returns_empty_when_no_axis(self):
        f = [_horse(1, 2, 3.0, 0.5), _horse(2, 3, 5.0, 0.4)]   # RL1 が居ない
        p, o = _probs_odds(f)
        b = build_axis_bets(f, p, o)
        assert b['win'] == [] and b['trio'] == []

    def test_survives_missing_odds_and_probs(self, field):
        """オッズ欠損レースでも例外を投げず、買える分だけ返すこと。"""
        for h in field:
            h['win_odds'] = 0
        b = build_axis_bets(field, {'win': {}}, {})
        assert b['win'] == [] and b['wide'] == [] and b['trio'] == []


class TestSimulatorProvidesWide:
    """ワイドは確率・推定配当とも今回追加した。実際に出ることを固定する。"""

    def test_calc_ticket_probabilities_has_wide(self):
        import numpy as np
        from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
        orders = simulate_race([2.0, 1.0, 0.5, 0.0, -0.5], n_sims=3000)
        probs = calc_ticket_probabilities(orders, [1, 2, 3, 4, 5])
        assert probs['wide'], 'wide が空'
        # ワイド(2頭とも3着内) は 馬連(1-2着) より必ず起きやすい
        for k, v in probs['quinella'].items():
            assert probs['wide'][k] >= v, f'{k}: wide {probs["wide"][k]} < 馬連 {v}'

    def test_payout_estimator_has_wide(self):
        from src.betting.payout_estimator import estimate_payouts_from_win_odds
        po = estimate_payouts_from_win_odds({1: 2.0, 2: 4.0, 3: 8.0, 4: 15.0, 5: 30.0},
                                            n_sims=3000)
        assert po['wide'], 'wide の推定配当が空'
        for k, v in po['wide'].items():
            assert v >= 1.0, f'{k} の推定配当が1.0倍未満: {v}'
