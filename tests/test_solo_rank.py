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


# ── EV用の絶対確率（2026-09-11）────────────────────────────────────────────
# 🔑 このプロジェクトには性質の違う確率が2種類あり、混ぜてはいけない。
#    EV用   : sigmoid(ability + フラット土台) → Isotonic  馬ごとの絶対値が正しい
#    順位用 : softmax(T=3.5) → Harville                  レース内合計が 1.0 / 3.0
# 順位用を絶対確率として読むと ECE 0.0757（EV用は 0.0080）。
# ⚠ EV用はレース内合計が 1.0 / 3.0 に**ならない**。それが正しい。

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.betting import app_json as _aj


@pytest.fixture
def fake_cal(tmp_path, monkeypatch):
    """本番と同じ形（Isotonic 2本入りの pkl）の較正器を置く。North Star #6。"""
    import pickle
    x = np.linspace(0.02, 0.98, 200)
    d = {'win': IsotonicRegression(out_of_bounds='clip').fit(x, x * 0.3),
         'fuku': IsotonicRegression(out_of_bounds='clip').fit(x, x * 0.8),
         'n_rows': 200, 'fitted_on': 'test'}
    (tmp_path / 'data').mkdir()
    with open(tmp_path / 'data' / 'ability_calibrator.pkl', 'wb') as f:
        pickle.dump(d, f)
    monkeypatch.setattr(_aj, '_ABILITY_CAL_DIR', None)
    monkeypatch.setattr(_aj, '_ABILITY_CAL', None)
    return str(tmp_path)


def _scored(margins):
    return [_horse(i + 1, i + 1, 0.1, m, odds=2.0 + i) for i, m in enumerate(margins)]


def test_ev_probs_follow_ability(fake_cal):
    r = _aj._build_ev_probs(_scored([1.2, -0.4, 0.7, -1.1, 0.05, 2.0]), fake_cal)
    assert max(r, key=lambda k: r[k][0]) == 6      # ability 2.0
    assert min(r, key=lambda k: r[k][0]) == 4      # ability -1.1


def test_ev_probs_independent_of_market(fake_cal):
    """🔴 当日のオッズ・人気を変えても値が動かない（EV設計の前提）。"""
    ms = [1.2, -0.4, 0.7, -1.1, 0.05, 2.0]
    a, b = _scored(ms), _scored(ms)
    for i, h in enumerate(a):
        h['win_odds'], h['popularity'] = 2.0 + i * 5, i + 1
    for i, h in enumerate(b):
        h['win_odds'], h['popularity'] = 60.0 - i * 5, len(b) - i
    assert _aj._build_ev_probs(a, fake_cal) == _aj._build_ev_probs(b, fake_cal)


def test_ev_probs_empty_without_calibrator(tmp_path, monkeypatch):
    """🔴 較正器が無ければ数字を作らない（softmax で代用しない）。"""
    monkeypatch.setattr(_aj, '_ABILITY_CAL_DIR', None)
    monkeypatch.setattr(_aj, '_ABILITY_CAL', None)
    assert _aj._build_ev_probs(_scored([1.2, -0.4, 0.7]), str(tmp_path)) == {}


def test_ev_probs_empty_when_any_ability_missing(fake_cal):
    sc = _scored([1.2, -0.4, 0.7])
    sc[1]['ability_margin'] = None
    assert _aj._build_ev_probs(sc, fake_cal) == {}


def test_horse_dict_carries_ev_fields(fake_cal):
    """馬辞書に EV用確率と必要オッズが乗り、順位用の確率も残っていること。"""
    scored = _scored([1.2, -0.4, 0.7, -1.1, 0.05, 2.0])
    hs = _build_horses_list(scored, scored[0],
                            sorted(scored, key=lambda x: x['win_odds']), base_dir=fake_cal)
    for h in hs:
        assert h['ev_tan_pct'] is not None and h['ev_fuku_pct'] is not None
        # 必要オッズ = 1.2 / EV用勝率
        assert h['need_odds'] == pytest.approx(1.2 / (h['ev_tan_pct'] / 100), rel=0.02)
        # 順位用は別に残っている（買い目・軸が使う）
        assert h['tan_pct'] is not None and h['fuku_pct'] is not None


def test_no_ev_field_in_horse_dict(fake_cal):
    """🔴 'ev' という名前を画面に出さない。順位用の量は sim_ev に分離した。"""
    scored = _scored([1.2, -0.4, 0.7])
    hs = _build_horses_list(scored, scored[0],
                            sorted(scored, key=lambda x: x['win_odds']), base_dir=fake_cal)
    assert 'ev' not in hs[0], "'ev' が残っている（期待値と誤読される）"
    assert 'sim_ev' in hs[0]


def test_ev_fields_none_without_calibrator(tmp_path, monkeypatch):
    monkeypatch.setattr(_aj, '_ABILITY_CAL_DIR', None)
    monkeypatch.setattr(_aj, '_ABILITY_CAL', None)
    scored = _scored([1.2, -0.4, 0.7])
    hs = _build_horses_list(scored, scored[0],
                            sorted(scored, key=lambda x: x['win_odds']), base_dir=str(tmp_path))
    assert all(h['ev_tan_pct'] is None and h['need_odds'] is None for h in hs)
    assert all(h['tan_pct'] is not None for h in hs)   # 順位用は影響を受けない


def test_marks_use_sim_ev_not_ev():
    """🔴 マーク付与が読む内部量は sim_ev（順位用）。名前で性質が分かること。"""
    import inspect
    src = inspect.getsource(_aj._assign_marks)
    assert "h.get('sim_ev')" in src
    assert "h.get('ev')" not in src, "順位用の量が 'ev' のまま残っている"
