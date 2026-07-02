"""買い目・推奨フラグ設計統一のテスト"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.make_bets import (
    _select_bet_candidates, select_bet_type,
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


def test_detect_value_horses_fallback_to_pn():
    """top3_prob がないとき pn にフォールバックする（app_json.fuku_pct と同じフォールバック）"""
    horses = [{'num': 1, 'horse_num': 1, 'cal_prob': 0.50, 'pn': 0.30}]
    market = {1: 2.0}  # fukusho_odds=2.0, market_prob=0.4
    result = detect_value_horses(horses, market)
    # pn=0.30 - market_prob=0.4 = -0.10  (app_json の fuku_pct = pn*100 = 30% と一致)
    assert abs(result[0]['value_gap'] - (-0.10)) < 0.001


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
    test_detect_value_horses_uses_top3_prob()
    print('✅ test_detect_value_horses_uses_top3_prob passed')
    test_detect_value_horses_fallback_to_pn()
    print('✅ test_detect_value_horses_fallback_to_pn passed')
    test_select_bet_type_returns_bets_all_chaos_grades()
    print('✅ test_select_bet_type_returns_bets_all_chaos_grades passed')
