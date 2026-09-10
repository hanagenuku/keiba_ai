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


# ── 市場ゼロAIの勝率（画面の「AI勝率」列・2026-09-10 ユーザー要望）──────────
# ユーザー要望「勝率を市場オッズ由来ではなくAI予想の勝率で表示して」への対応。
# base_margin をフラット（全馬同値）に置き換えたAI単独の確率。
# 🔴 表示専用であり、EV・買い目・軸・RL順位・レース厳選は従来のままであること。

def _scored(margins):
    return [_horse(i + 1, i + 1, 0.1, m, odds=2.0 + i) for i, m in enumerate(margins)]


def test_solo_probs_sum_to_one():
    from src.betting.app_json import _build_solo_probs
    r = _build_solo_probs(_scored([1.2, -0.4, 0.7, -1.1, 0.05, 2.0]))
    assert abs(sum(w for w, _ in r.values()) - 1.0) < 1e-6


def test_solo_probs_follow_ability():
    from src.betting.app_json import _build_solo_probs
    r = _build_solo_probs(_scored([1.2, -0.4, 0.7, -1.1, 0.05, 2.0]))
    assert max(r, key=lambda k: r[k][0]) == 6      # ability 2.0
    assert min(r, key=lambda k: r[k][0]) == 4      # ability -1.1


def test_solo_probs_independent_of_market():
    """🔴 市場ゼロであることの検査: オッズ・人気を変えても値が動かない。"""
    from src.betting.app_json import _build_solo_probs
    ms = [1.2, -0.4, 0.7, -1.1, 0.05, 2.0]
    a, b = _scored(ms), _scored(ms)
    for i, h in enumerate(a):
        h['win_odds'] = 2.0 + i * 5
        h['popularity'] = i + 1
    for i, h in enumerate(b):
        h['win_odds'] = 60.0 - i * 5
        h['popularity'] = len(b) - i
    assert _build_solo_probs(a) == _build_solo_probs(b)


def test_solo_probs_empty_when_any_ability_missing():
    from src.betting.app_json import _build_solo_probs
    sc = _scored([1.2, -0.4, 0.7])
    sc[1]['ability_margin'] = None
    assert _build_solo_probs(sc) == {}


def test_horse_dict_carries_ai_pct():
    """馬辞書に ai_tan_pct / ai_fuku_pct が乗り、従来の勝率も残っていること。"""
    scored = _scored([1.2, -0.4, 0.7, -1.1, 0.05, 2.0])
    hs = _build_horses_list(scored, scored[0],
                            sorted(scored, key=lambda x: x['win_odds']))
    assert all(h['ai_tan_pct'] is not None for h in hs)
    assert all(h['ai_fuku_pct'] is not None for h in hs)
    # 表示専用なので、EV・買い目が使う従来の勝率は消えていない
    assert all(h['tan_pct'] is not None for h in hs)


def test_ai_pct_none_for_non_residual_model():
    scored = _scored([1.2, -0.4, 0.7])
    for h in scored:
        h['ability_margin'] = None
    hs = _build_horses_list(scored, scored[0],
                            sorted(scored, key=lambda x: x['win_odds']))
    assert all(h['ai_tan_pct'] is None for h in hs)
    assert all(h['tan_pct'] is not None for h in hs)
