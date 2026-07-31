"""AI強気ペア(A×D)の馬連・ワイド推奨のテスト（2026-07-31導入）。

「市場人気よりAIが高評価」の馬どうしを組む買い目。
片脚だけ（AI強気 × 中立馬）は実測で比1.26と基準未達だったため、
**両脚ともAI強気であること**が定義の核心。ここが崩れると別物になる。

⚠ 回収率は未検証（組み合わせ配当がDBに無い）。
   詳細は src/betting/ai_pair_bets.py の docstring を参照。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.ai_pair_bets import (build_ai_pair_bets, classify,
                                      format_for_app)


def _h(num, pop, ai, name=None):
    """calc_all が返す馬辞書と同じ形。"""
    return {'num': num, 'name': name or f'H{num}',
            'popularity': pop, 'rl_rank': ai}


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.delenv('AI_PAIR_BETS', raising=False)


class TestClassify:

    def test_axis_is_mid_popularity_and_ai_bullish(self):
        assert classify(_h(1, 4, 2)) == 'A'      # 4番人気→AI2位
        assert classify(_h(1, 6, 5)) == 'A'      # 6番人気→AI5位

    def test_partner_is_longshot_and_ai_bullish(self):
        assert classify(_h(1, 7, 3)) == 'D'
        assert classify(_h(1, 14, 9)) == 'D'

    def test_ai_not_bullish_is_excluded(self):
        """AI順位が市場人気以上＝AI強気でない馬は対象外。"""
        assert classify(_h(1, 5, 5)) is None     # 同じ
        assert classify(_h(1, 5, 8)) is None     # AI弱気

    def test_top_popularity_excluded_from_axis(self):
        """1-3番人気は軸にしない（実測で単勝回収率が伸びない帯）。"""
        assert classify(_h(1, 2, 1)) is None
        assert classify(_h(1, 3, 1)) is None

    def test_missing_or_invalid_fields(self):
        assert classify({'num': 1}) is None
        assert classify(_h(1, 99, 3)) is None    # 人気不明(99)
        assert classify(_h(1, 5, 99)) is None    # AI順位不明


class TestBuildAiPairBets:

    def _field(self):
        return [
            _h(1, 1, 1),      # 上位人気・AI一致 → 対象外
            _h(5, 5, 2),      # A群（AI最上位）
            _h(4, 4, 3),      # A群
            _h(9, 9, 4),      # D群（AI最上位）
            _h(12, 12, 6),    # D群
            _h(7, 7, 10),     # AI弱気 → 対象外
        ]

    def test_default_picks_ai_top_from_each_group(self):
        """既定は各群のAI最上位1頭ずつ（検証で的中率が最も高かった構成）。"""
        r = build_ai_pair_bets(self._field())
        assert [p['nums'] for p in r['umaren']] == [[5, 9]]
        assert r['axis'] == [5] and r['partners'] == [9]

    def test_umaren_and_wide_are_same_pairs(self):
        r = build_ai_pair_bets(self._field())
        assert r['umaren'] == r['wide']

    def test_expanding_counts_produces_cross_product(self):
        r = build_ai_pair_bets(self._field(), max_axis=2, max_partner=2)
        assert sorted(p['nums'] for p in r['umaren']) == [[4, 9], [4, 12], [5, 9], [5, 12]]

    def test_marks_itself_unverified(self):
        """回収率未検証であることを表示側に必ず伝える。"""
        r = build_ai_pair_bets(self._field())
        assert r['unverified'] is True
        assert r['measured']['breakeven_ratio'] == 1.29

    def test_empty_when_no_axis(self):
        """軸(中位人気×AI強気)が居なければ推奨を出さない。"""
        assert build_ai_pair_bets([_h(9, 9, 4), _h(12, 12, 6)]) == {}

    def test_empty_when_no_partner(self):
        assert build_ai_pair_bets([_h(5, 5, 2), _h(4, 4, 3)]) == {}

    def test_empty_for_empty_input(self):
        assert build_ai_pair_bets([]) == {}
        assert build_ai_pair_bets(None) == {}

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv('AI_PAIR_BETS', '0')
        assert build_ai_pair_bets(self._field()) == {}

    def test_pair_carries_reason_fields_for_display(self):
        """アプリで「なぜこの組か」を出せるだけの情報を持つこと。"""
        p = build_ai_pair_bets(self._field())['umaren'][0]
        assert p['axis_pop'] == 5 and p['axis_ai'] == 2
        assert p['partner_pop'] == 9 and p['partner_ai'] == 4

    def test_format_for_app(self):
        lines = format_for_app(build_ai_pair_bets(self._field()))
        assert len(lines) == 1
        assert '5-9' in lines[0] and '5番人気' in lines[0]
        assert format_for_app({}) == []


class TestWiredIntoAppJson:

    def test_app_json_exposes_ai_pair_bets(self):
        """to_app_json のレースエントリに ai_pair_bets が載ること。"""
        import src.betting.app_json as aj
        src = open(aj.__file__).read()
        # 推奨レース側・推奨外レース側の両方に入っていること
        assert src.count("'ai_pair_bets': _build_ai_pair_bets(scored)") == 2
