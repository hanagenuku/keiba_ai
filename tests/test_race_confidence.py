"""レースの自信度（画面の「軸3着内 xx%」）を固定する。

🔴 2026-08-18 まで、この数字は手書きの式だった:

    conf = 60 + (AI本命の市場人気順位 - 2) * 4 + score_gap * 20

**主項の符号が逆**で、AI本命が不人気なほど自信度が上がる設計になっていた。
実データ（2025-07以降・32,240頭）はその逆:

    人気1: 複勝率66.1% → 旧式conf 56
    人気9: 複勝率 9.6% → 旧式conf 88   ← 当たらないほど高く出ていた

差し替え後は `select_quality_races` の見送り判定が読むのと同じ
`top3_prob` を使う。**画面の数字と見送り判断が必ず一致する**ことが要点。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betting.app_json import race_confidence  # noqa: E402
from src.betting.ev_filter import MIN_AXIS_FUKU_PROB  # noqa: E402


def _horse(num, rl_rank, top3_prob, popularity, name=None):
    """calc_all() が実際に返す形に合わせた馬辞書（North Star #6）。"""
    return {
        'num': num, 'horse_num': num, 'name': name or f'ウマ{num}',
        'rl_rank': rl_rank, 'top3_prob': top3_prob,
        'popularity': popularity, '_pop': popularity,
        'win_odds': 1.0 + popularity, 'total': 10.0 - rl_rank,
        'cal_prob': top3_prob, 'tan_pct': 20.0, 'running_style': '差し',
    }


class TestRaceConfidence:
    def test_returns_axis_top3_prob_as_percent(self):
        scored = [_horse(1, 1, 0.62, 1), _horse(2, 2, 0.41, 2),
                  _horse(3, 3, 0.30, 3)]
        assert race_confidence(scored) == 62

    def test_uses_rl_rank_1_not_list_head(self):
        """軸は RL1。scored の先頭とは限らない（ev_filter と同じ規約）。"""
        scored = [_horse(1, 2, 0.41, 1), _horse(2, 1, 0.75, 3),
                  _horse(3, 3, 0.30, 2)]
        assert race_confidence(scored) == 75

    def test_unpopular_axis_does_not_raise_confidence(self):
        """旧式の逆向きバグの回帰テスト。

        軸が9番人気で3着内確率が低いレースの自信度が、
        軸が1番人気で確率が高いレースを**上回らない**こと。
        旧式では 88 vs 56 と逆転していた。
        """
        popular = [_horse(1, 1, 0.66, 1), _horse(2, 2, 0.40, 2)]
        longshot = [_horse(1, 1, 0.10, 9), _horse(2, 2, 0.40, 1)]
        assert race_confidence(longshot) < race_confidence(popular)

    def test_matches_the_skip_threshold(self):
        """画面の数字と見送り判断が一致すること。

        自信度が閾値ちょうどのレースは通過側、下回れば見送り側になる。
        両者が別々の値を見ていると、画面で高く出ているのに買い目が
        出ない（またはその逆）という食い違いが起きる。
        """
        at = [_horse(1, 1, MIN_AXIS_FUKU_PROB, 1)]
        below = [_horse(1, 1, MIN_AXIS_FUKU_PROB - 0.10, 1)]
        assert race_confidence(at) >= int(MIN_AXIS_FUKU_PROB * 100)
        assert race_confidence(below) < int(MIN_AXIS_FUKU_PROB * 100)

    def test_is_monotone_in_axis_probability(self):
        prev = -1
        for p in [0.20, 0.35, 0.50, 0.65, 0.80]:
            c = race_confidence([_horse(1, 1, p, 1)])
            assert c > prev
            prev = c

    def test_none_when_probability_missing(self):
        """確率が無いときに作り話をしないこと。

        旧式は人気順位だけで必ず50〜99の数字を出していた。
        """
        scored = [{'num': 1, 'name': 'ウマ1', 'rl_rank': 1, 'popularity': 1}]
        assert race_confidence(scored) is None

    def test_empty_input(self):
        assert race_confidence([]) is None
        assert race_confidence(None) is None

    def test_clamped_to_valid_range(self):
        assert race_confidence([_horse(1, 1, 1.5, 1)]) == 99
        assert race_confidence([_horse(1, 1, -0.2, 1)]) == 0
