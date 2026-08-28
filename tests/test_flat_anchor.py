"""オッズが取れない時に「馬番順の人気」を捏造しないこと（2026-08-29発見）。

■ 何が起きていたか
`calc_all` は人気順位をこう作っていた:

    for rank, h in enumerate(sorted(horses, key=lambda x: x.get('win_odds') or 999), 1):
        h['popularity'] = rank

オッズが全滅すると **全馬が999で同値** になり、Python の sort は安定なので
順位が **入力順＝馬番順** に付く。つまり「馬番1が1番人気、馬番2が2番人気…」
という **架空の市場** が base_margin（残差学習の予測の出発点）に入っていた。
アンカー無しより悪い。

■ いつ起きていたか
土曜夜の日曜予想は `odds_coverage = 0.0`（2026-08-15 C-2）。
`_sanitize_odds_book` が盤を無効化したレースも同じ状態になる。
2026-08-15 の土曜夜は 17レース中 15レースがこれに該当した。

■ 実測（5窓・複勝回収率 / 各レース1位を1点買い）
    健全な人気 82.6% ／ 馬番順(修正前) 73.6% ／ フラット(修正後) 83.0%
    AUC は5窓すべてで 馬番順 0.66台 ＜ フラット 0.76〜0.79
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (_base_margin_from_popularity,
                                 _flat_base_margin)


def _derive_popularity(horses):
    """修正前の calc_all と同じ式。バグの再現用。"""
    for rank, h in enumerate(
            sorted(horses, key=lambda x: x.get('win_odds') or 999), 1):
        if not h.get('popularity') or h.get('popularity') == 99:
            h['popularity'] = rank
    return horses


class TestTheBugItself:
    """修正前の式が実際に馬番順を人気にしてしまうことを固定する。

    この振る舞いが「仕様」に戻らないよう、バグ側を明示的にテストしておく。
    """

    def test_old_formula_assigns_popularity_by_horse_number(self):
        horses = [{'num': n, 'win_odds': None, 'popularity': None}
                  for n in range(1, 13)]
        _derive_popularity(horses)
        assert [h['popularity'] for h in horses] == list(range(1, 13))
        assert all(h['num'] == h['popularity'] for h in horses)


class TestFlatBaseMargin:

    def test_equals_mean_over_all_ranks(self):
        for n in (8, 12, 16, 18):
            expect = sum(_base_margin_from_popularity(k, n)
                         for k in range(1, n + 1)) / n
            assert math.isclose(_flat_base_margin(n), expect, rel_tol=1e-12)

    def test_is_between_top_and_bottom_popularity(self):
        """水準を保つための値なので、1番人気と最下位人気の間に入ること。"""
        n = 14
        assert (_base_margin_from_popularity(n, n)
                < _flat_base_margin(n)
                < _base_margin_from_popularity(1, n))

    def test_matches_the_inline_formula_it_replaced(self):
        """切り出した関数が、元のインライン式と同じ値を返すこと。"""
        for n in (8, 16):
            for pop in (1, 3, n):
                harm = math.log(n) + 0.5772
                p = max(min((1.0 / max(pop, 1)) / harm, 0.999), 0.001)
                assert math.isclose(_base_margin_from_popularity(pop, n),
                                    math.log(p / (1 - p)), rel_tol=1e-12)


class TestAnchorSelection:
    """calc_all がアンカーをどう選ぶか。engine を丸ごと動かさずに式だけ検証する。"""

    @staticmethod
    def _pick(horses):
        has_odds = any((h.get('win_odds') or 0) > 0 for h in horses)
        has_pop = any(0 < (h.get('popularity') or 0) < 99 for h in horses)
        return 'odds' if (has_odds or has_pop) else 'flat'

    def test_no_odds_no_popularity_gives_flat(self):
        horses = [{'num': n, 'win_odds': None, 'popularity': None}
                  for n in range(1, 13)]
        assert self._pick(horses) == 'flat'

    def test_sentinel_99_is_not_treated_as_known_popularity(self):
        """99は「不明」のセンチネル。これを人気とみなすと元の事故に戻る。"""
        horses = [{'num': n, 'win_odds': None, 'popularity': 99}
                  for n in range(1, 13)]
        assert self._pick(horses) == 'flat'

    def test_odds_present_keeps_the_market_anchor(self):
        """正常時は挙動を変えないこと（回帰）。"""
        horses = [{'num': 1, 'win_odds': 3.5, 'popularity': None},
                  {'num': 2, 'win_odds': 8.0, 'popularity': None}]
        assert self._pick(horses) == 'odds'

    def test_confirmed_popularity_without_odds_keeps_the_market_anchor(self):
        """結果ページ由来の確定人気があるなら、それは本物の市場情報。"""
        horses = [{'num': 1, 'win_odds': None, 'popularity': 3},
                  {'num': 2, 'win_odds': None, 'popularity': 1}]
        assert self._pick(horses) == 'odds'


class TestFlatAnchorLetsAbilityDecideTheOrder:
    """フラットにすると順位が ability_margin だけで決まること。

    ユーザーの指摘「素のAIのベースがあるのだから、オッズがなくても
    それなりの順位付けができるはず」がこれで成立する。
    """

    def test_flat_margin_preserves_ability_order(self):
        n = 10
        ability = [0.8, -0.3, 1.5, 0.1, -1.2, 0.4, -0.7, 2.0, -0.1, 0.6]
        f = _flat_base_margin(n)
        raw = [a + f for a in ability]
        assert (sorted(range(n), key=lambda i: -raw[i])
                == sorted(range(n), key=lambda i: -ability[i]))

    def test_umaban_margin_lets_horse_number_override_ability(self):
        """対照: 馬番順アンカーだと能力の順位が馬番に引きずられる。"""
        n = 10
        ability = [0.0] * n           # 全馬の能力が同じ
        raw = [ability[i] + _base_margin_from_popularity(i + 1, n)
               for i in range(n)]
        # 能力に差が無いのに、馬番順にきれいに並んでしまう
        assert sorted(range(n), key=lambda i: -raw[i]) == list(range(n))
