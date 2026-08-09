"""市場ゼロAI順位（solo_rank）の検証。

`solo_rank` は残差学習モデルの `ability_margin`（= raw_margin - base_margin、
市場アンカーを差し引いたAI単独の評価）から作る**表示専用**の順位。
買い目・軸・レース厳選・成績集計はすべて従来の `rl_rank` のままであることを、
このテストで固定する（表示のために馬券が変わってしまう退行の検知）。

North Star #6 に従い、手打ちの理想dictではなく **calc_all が実際に返す形**
（num / name / total / pn / rl_rank / ability_margin 等）の馬辞書で検証する。
"""
from src.betting.app_json import _build_solo_ranks, _build_horses_list


def _horse(num, rl, pn, ability, odds=5.0):
    """calc_all の出力と同じキー構成の馬辞書。"""
    return {
        'num': num, 'name': f'馬{num}', 'total': 8.0 - num * 0.1,
        'pn': pn, 'rl_rank': rl, 'cl_rank': rl, 'win_odds': odds,
        'cal_prob': pn, 'top3_prob': min(0.99, pn * 2.4),
        'running_style': '差し', 'ability_margin': ability,
        'ability_breakdown': None, 'rating': 0.0,
    }


def _field():
    """rl_rank と ability_margin の順序がわざと食い違う4頭立て。

    市場（オッズ）に支えられて RL1 になっている #1 は、市場を外すと最下位。
    逆に市場が見放している #4 が、AI単独では最上位になる。
    """
    return [
        _horse(1, 1, 0.40, -0.50, odds=1.8),
        _horse(2, 2, 0.30, +0.10, odds=4.0),
        _horse(3, 3, 0.20, +0.30, odds=9.0),
        _horse(4, 4, 0.10, +0.90, odds=30.0),
    ]


class TestBuildSoloRanks:
    def test_ranks_by_ability_margin_descending(self):
        assert _build_solo_ranks(_field()) == {4: 1, 3: 2, 2: 3, 1: 4}

    def test_empty_when_any_horse_lacks_ability_margin(self):
        # 非残差モデル運用時・推論フォールバック時に、一部の馬だけ別基準の
        # 順位が混ざるのを防ぐ（レース全体で出さない）。
        field = _field()
        field[2]['ability_margin'] = None
        assert _build_solo_ranks(field) == {}

    def test_empty_for_empty_field(self):
        assert _build_solo_ranks([]) == {}

    def test_ties_break_by_horse_number_not_by_existing_rank(self):
        # ⚠ タイブレークに着順や既存順位を使うとそこから情報が漏れる
        #    （2026-08-08のPLレーティングのリークがこの形だった）。
        #    同値なら馬番昇順という中立な順序であることを固定する。
        field = [_horse(3, 1, 0.4, 0.5), _horse(1, 2, 0.3, 0.5),
                 _horse(2, 3, 0.2, 0.5)]
        assert _build_solo_ranks(field) == {1: 1, 2: 2, 3: 3}


class TestSoloRankInAppJson:
    def test_horses_list_exposes_solo_rank(self):
        field = _field()
        by_odds = sorted(field, key=lambda h: h['win_odds'])
        horses = _build_horses_list(field, field[0], by_odds)
        got = {h['n']: h['solo_rank'] for h in horses}
        assert got == {4: 1, 3: 2, 2: 3, 1: 4}

    def test_rl_rank_is_untouched_by_solo_rank(self):
        # 併記であって置き換えではない。買い目・軸・集計はすべて rl_rank を
        # 見ているので、ここが動くと表示だけのつもりが馬券まで変わる。
        field = _field()
        by_odds = sorted(field, key=lambda h: h['win_odds'])
        horses = _build_horses_list(field, field[0], by_odds)
        assert {h['n']: h['rl_rank'] for h in horses} == {1: 1, 2: 2, 3: 3, 4: 4}

    def test_solo_rank_none_when_model_is_not_residual(self):
        # 非残差モデルでは ability_margin が None → 列ごと出さない（Noneのまま）
        field = [dict(h, ability_margin=None) for h in _field()]
        by_odds = sorted(field, key=lambda h: h['win_odds'])
        horses = _build_horses_list(field, field[0], by_odds)
        assert all(h['solo_rank'] is None for h in horses)

    def test_solo_rank_does_not_depend_on_odds(self):
        # ability_margin は市場アンカーを差し引いた値なので、オッズが動いても
        # solo_rank は動かない。直前オッズ取得の前後で順位が変わらないこと。
        field = _field()
        before = _build_solo_ranks(field)
        for h in field:
            h['win_odds'] *= 3.0   # 市場が総取っ替えになった想定
        assert _build_solo_ranks(field) == before
