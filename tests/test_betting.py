"""買い目・推奨フラグ設計統一のテスト"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.make_bets import (
    classify_chaos_grade, _select_bet_candidates, select_bet_type,
)
from src.betting.ev_filter import detect_value_horses


# ── ① _select_bet_candidates は空リストを返さない ────────────────────────────

def _make_horse(num, rl_rank, pn=0.10, odds=10.0, popularity=5):
    return {'num': num, 'horse_num': num, 'name': f'馬{num}',
            'rl_rank': rl_rank, 'pn': pn, 'win_odds': odds, 'popularity': popularity}


def test_select_bet_no_skip_c_no_value_9heads():
    """chaos_grade=C・バリュー馬なし・9頭 → 空リストを返さない（旧バグの回帰防止）"""
    horses = [_make_horse(i, i, pn=0.12, odds=8.0, popularity=i) for i in range(1, 10)]
    by_rl  = sorted(horses, key=lambda h: h['rl_rank'])
    bets = _select_bet_candidates(
        by_rl, by_rl[0], by_rl[1], by_rl[2],
        chaos_grade='C', value_horses=[], num_horses=9,
    )
    assert bets, '_select_bet_candidates が空リストを返した（スキップバグ）'


def test_select_bet_no_skip_c_no_value_13heads():
    """13頭 chaos_grade=C バリューなし → 買い目が出る"""
    horses = [_make_horse(i, i, pn=0.08, odds=10.0, popularity=i) for i in range(1, 14)]
    by_rl  = sorted(horses, key=lambda h: h['rl_rank'])
    bets = _select_bet_candidates(
        by_rl, by_rl[0], by_rl[1], by_rl[2],
        chaos_grade='C', value_horses=[], num_horses=13,
    )
    assert bets


def test_select_bet_no_skip_a_5heads():
    """5頭 chaos_grade=A → 少頭数・馬連が出る"""
    horses = [_make_horse(i, i) for i in range(1, 6)]
    by_rl  = sorted(horses, key=lambda h: h['rl_rank'])
    bets = _select_bet_candidates(
        by_rl, by_rl[0], by_rl[1], by_rl[2],
        chaos_grade='A', value_horses=[], num_horses=5,
    )
    assert bets
    assert any(b['type'] == '馬連' for b in bets)


# ── ④ classify_chaos_grade: popularity 未設定でも win_odds から推定 ────────────

def test_classify_chaos_grade_uses_odds_fallback():
    """popularity が未設定でも win_odds から人気順を推定して A/B/C を返す"""
    # RL1位が1番人気相当（odds=2.0）・chaos_score低い → 'A' になるべき
    horses = [
        {'num': 1, 'horse_num': 1, 'rl_rank': 1, 'pn': 0.40, 'win_odds': 2.0},
        {'num': 2, 'horse_num': 2, 'rl_rank': 2, 'pn': 0.25, 'win_odds': 5.0},
        {'num': 3, 'horse_num': 3, 'rl_rank': 3, 'pn': 0.15, 'win_odds': 8.0},
    ]
    # popularity キーがない状態でも動作すること
    for h in horses:
        assert 'popularity' not in h
    grade = classify_chaos_grade(horses, chaos_score=0.20)
    assert grade == 'A'


def test_classify_chaos_grade_c_high_popularity():
    """RL1位が6番人気以上 → 'C'"""
    horses = [
        {'num': 3, 'horse_num': 3, 'rl_rank': 1, 'pn': 0.15, 'win_odds': 15.0},
        {'num': 1, 'horse_num': 1, 'rl_rank': 2, 'pn': 0.30, 'win_odds': 3.0},
        {'num': 2, 'horse_num': 2, 'rl_rank': 3, 'pn': 0.20, 'win_odds': 5.0},
        {'num': 4, 'horse_num': 4, 'rl_rank': 4, 'pn': 0.12, 'win_odds': 7.0},
        {'num': 5, 'horse_num': 5, 'rl_rank': 5, 'pn': 0.10, 'win_odds': 8.0},
        {'num': 6, 'horse_num': 6, 'rl_rank': 6, 'pn': 0.08, 'win_odds': 10.0},
    ]
    grade = classify_chaos_grade(horses, chaos_score=0.40)
    assert grade == 'C'


# ── ⑤ detect_value_horses: value_gap は top3_prob ベース ─────────────────────

def test_detect_value_horses_uses_top3_prob():
    """top3_prob が存在するとき value_gap 計算に使われる（cal_prob より優先）"""
    horses = [
        {'num': 1, 'horse_num': 1, 'top3_prob': 0.60, 'cal_prob': 0.30, 'pn': 0.25},
        {'num': 2, 'horse_num': 2, 'top3_prob': 0.40, 'cal_prob': 0.25, 'pn': 0.20},
    ]
    market = {1: {'tansho': 4.0, 'fukusho': 1.8}, 2: {'tansho': 6.0, 'fukusho': 2.5}}
    result = detect_value_horses(horses, market)
    by_num = {h['horse_num']: h for h in result}

    # 馬1: top3_prob=0.60, market_prob=0.8/1.8≈0.444 → value_gap ≈ 0.156
    market_prob1 = 0.8 / 1.8
    expected_gap1 = round(0.60 - market_prob1, 4)
    assert abs(by_num[1]['value_gap'] - expected_gap1) < 0.001, \
        f'value_gap should use top3_prob: got {by_num[1]["value_gap"]}, expected {expected_gap1}'


def test_detect_value_horses_fallback_to_cal_prob():
    """top3_prob がないとき cal_prob にフォールバックする"""
    horses = [{'num': 1, 'horse_num': 1, 'cal_prob': 0.50, 'pn': 0.30}]
    market = {1: 2.0}  # fukusho_odds=2.0, market_prob=0.4
    result = detect_value_horses(horses, market)
    # cal_prob=0.50 - market_prob=0.4 = 0.10
    assert abs(result[0]['value_gap'] - 0.10) < 0.001


# ── select_bet_type: market_odds あり（ルールベースパス）───────────────────────

def test_select_bet_type_returns_bets_all_chaos_grades():
    """A/B/C すべての波乱度で bets が返ってくること"""
    horses = [_make_horse(i, i, pn=0.15, odds=8.0, popularity=i) for i in range(1, 10)]
    for grade in ('A', 'B', 'C'):
        bets = select_bet_type(horses, grade, [], num_horses=9)
        assert bets, f'grade={grade} で bets が空'


if __name__ == '__main__':
    test_select_bet_no_skip_c_no_value_9heads()
    print('✅ test_select_bet_no_skip_c_no_value_9heads passed')
    test_select_bet_no_skip_c_no_value_13heads()
    print('✅ test_select_bet_no_skip_c_no_value_13heads passed')
    test_select_bet_no_skip_a_5heads()
    print('✅ test_select_bet_no_skip_a_5heads passed')
    test_classify_chaos_grade_uses_odds_fallback()
    print('✅ test_classify_chaos_grade_uses_odds_fallback passed')
    test_classify_chaos_grade_c_high_popularity()
    print('✅ test_classify_chaos_grade_c_high_popularity passed')
    test_detect_value_horses_uses_top3_prob()
    print('✅ test_detect_value_horses_uses_top3_prob passed')
    test_detect_value_horses_fallback_to_cal_prob()
    print('✅ test_detect_value_horses_fallback_to_cal_prob passed')
    test_select_bet_type_returns_bets_all_chaos_grades()
    print('✅ test_select_bet_type_returns_bets_all_chaos_grades passed')
