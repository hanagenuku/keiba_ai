"""select_quality_races のゲート構成の回帰テスト。

2026-08-24に3,969レース（2025-07-05〜2026-08-23・完全out-of-sample）で
6本のゲートを1本ずつ外して測った結果、働いていたのは軸の3着内確率だけだった。

    ゲート                外した時の複勝回収の差   判定
    gap>=0.03                    -0.4pt        外す
    win_prob>=0.10               ±0.0pt        外す（完全なno-op）
    軸のtop3_prob>=0.55          -4.0pt        ★残す
    未勝利・新馬を除外             +0.4pt        外す（除外が逆効果）
    ev_max>=1.30(代理)           -0.5pt        外す
    軸オッズ1.5-20倍(代理)         +0.4pt        外す

ここでは「既定でOFFになっていること」と「引数を渡せば従来通り効くこと」の
両方を固定する。切り戻し手段を消さないため。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _scored(axis_fuku=0.70, axis_pn=0.30, second_pn=0.28, axis_odds=3.0):
    return [
        {'num': 1, 'name': '軸', 'rl_rank': 1, 'win_odds': axis_odds,
         'pn': axis_pn, 'top3_prob': axis_fuku, 'total': 8.0},
        {'num': 2, 'name': '二番手', 'rl_rank': 2, 'win_odds': 6.0,
         'pn': second_pn, 'top3_prob': 0.42, 'total': 6.0},
        {'num': 3, 'name': '三番手', 'rl_rank': 3, 'win_odds': 9.0,
         'pn': 0.12, 'top3_prob': 0.33, 'total': 5.0},
    ]


def _run(monkeypatch, scored, race_class='1勝クラス', race_name='テスト', **kw):
    import src.betting.ev_filter as ef
    import src.features.engine as eng
    monkeypatch.setattr(eng, 'calc_all', lambda race, bias=None: scored)
    monkeypatch.setattr(eng, 'calc_chaos_score', lambda race, s: 5.0)
    race = {'id': 'r1', 'race_num': 1, 'race_name': race_name,
            'distance': 1600, 'surface': '芝', 'race_class': race_class,
            'racecourse': '中京', 'horses': scored}
    return ef.select_quality_races([race], **kw)


class TestGatesOffByDefault:
    """効果が確認できなかった5本は既定でOFF。"""

    def test_maiden_and_newcomer_are_not_excluded(self, monkeypatch):
        """未勝利・新馬を既定で除外しないこと。

        実測: 未勝利・新馬 複勝回収82.7%（それ以外79.2%）。3本抜き・前後半・
        日ブロックCIすべてで上。除外していたのが逆効果だった。
        """
        for cls in ('未勝利', '3歳新馬'):
            assert len(_run(monkeypatch, _scored(), race_class=cls)) == 1, cls

    def test_maiden_detected_by_race_name_also_not_excluded(self, monkeypatch):
        assert len(_run(monkeypatch, _scored(), race_class='',
                        race_name='2歳新馬')) == 1

    def test_tiny_gap_passes(self, monkeypatch):
        """gap（AI勝率1位-2位差）で足切りしないこと。"""
        s = _scored(axis_pn=0.30, second_pn=0.2999)   # gap ≒ 0.0001
        assert len(_run(monkeypatch, s)) == 1

    def test_low_win_prob_passes(self, monkeypatch):
        s = _scored(axis_pn=0.05, second_pn=0.04)
        assert len(_run(monkeypatch, s)) == 1

    def test_odds_outside_old_range_passes(self, monkeypatch):
        for od in (1.1, 45.0):
            assert len(_run(monkeypatch, _scored(axis_odds=od))) == 1, od

    def test_low_ev_passes(self, monkeypatch):
        """ev = pn × odds は 07-05/07-30/08-06 と3回否定されている指標。"""
        s = _scored(axis_pn=0.30, axis_odds=1.05)
        for h in s[1:]:
            h['win_odds'] = 1.05
        assert len(_run(monkeypatch, s)) == 1


class TestSurvivingGate:
    """唯一働いていたゲートは残す。"""

    def test_low_axis_confidence_is_still_skipped(self, monkeypatch):
        assert _run(monkeypatch, _scored(axis_fuku=0.40)) == []

    def test_high_axis_confidence_is_kept(self, monkeypatch):
        assert len(_run(monkeypatch, _scored(axis_fuku=0.70))) == 1

    def test_threshold_unchanged(self):
        from src.betting.ev_filter import MIN_AXIS_FUKU_PROB
        assert MIN_AXIS_FUKU_PROB == 0.55


class TestBackwardCompatible:
    """引数を渡せば従来のゲートが復活すること（切り戻し手段を残す）。"""

    def test_skip_classes_can_be_re_enabled(self, monkeypatch):
        assert _run(monkeypatch, _scored(), race_class='未勝利',
                    skip_classes=True) == []

    def test_min_gap_can_be_re_enabled(self, monkeypatch):
        s = _scored(axis_pn=0.30, second_pn=0.2999)
        assert _run(monkeypatch, s, min_gap=0.03) == []

    def test_min_win_prob_can_be_re_enabled(self, monkeypatch):
        s = _scored(axis_pn=0.05, second_pn=0.04)
        assert _run(monkeypatch, s, min_win_prob=0.10) == []

    def test_odds_range_can_be_re_enabled(self, monkeypatch):
        assert _run(monkeypatch, _scored(axis_odds=45.0),
                    odds_range=(1.5, 20.0)) == []

    def test_min_ev_can_be_re_enabled(self, monkeypatch):
        s = _scored(axis_pn=0.30, axis_odds=1.05)
        for h in s[1:]:
            h['win_odds'] = 1.05
        assert _run(monkeypatch, s, min_ev=1.30) == []


class TestPriorityIsAxisWinProb:
    """並べ替えは軸のAI勝率。ev_max ベースに戻っていないこと。"""

    def test_priority_equals_axis_pn(self, monkeypatch):
        sel = _run(monkeypatch, _scored(axis_pn=0.31))
        assert sel[0]['priority'] == pytest.approx(0.31)

    def test_axis_is_rl_rank_1_not_first_element(self, monkeypatch):
        """軸は rl_rank==1。scored の先頭とは限らない（ev_filter 共通の規約）。"""
        s = _scored()
        s[0]['rl_rank'], s[1]['rl_rank'] = 2, 1
        s[1]['pn'] = 0.44
        s[1]['top3_prob'] = 0.70      # 軸になる馬なので確度ゲートを通す側に
        s[0]['top3_prob'] = 0.42
        sel = _run(monkeypatch, s)
        assert sel[0]['priority'] == pytest.approx(0.44)

    def test_higher_confidence_race_ranks_first(self, monkeypatch):
        import src.betting.ev_filter as ef
        import src.features.engine as eng
        weak, strong = _scored(axis_pn=0.20), _scored(axis_pn=0.45)
        by_id = {'weak': weak, 'strong': strong}
        monkeypatch.setattr(eng, 'calc_all', lambda race, bias=None: by_id[race['id']])
        monkeypatch.setattr(eng, 'calc_chaos_score', lambda race, s: 5.0)
        races = [{'id': k, 'race_num': i + 1, 'race_name': k, 'distance': 1600,
                  'surface': '芝', 'race_class': '1勝クラス', 'racecourse': '中京',
                  'horses': v} for i, (k, v) in enumerate(by_id.items())]
        sel = ef.select_quality_races(races, max_races=1)
        assert len(sel) == 1 and sel[0]['race']['id'] == 'strong'
