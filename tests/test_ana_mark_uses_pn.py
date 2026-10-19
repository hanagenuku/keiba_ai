"""穴マークが `pn`（= win_prob）を見ており `cal_prob` を見ないことを固定する。

2026-10-19、較正のパリティ違反（カードB）を調べる過程で、
`app_json.py` のコメントと CLAUDE.md が「穴マーク閾値 cal_prob>=0.15」と
書いているのに、実コードは `h.get('pn', 0) >= 0.15` を見ていると判明した。
`pn = win_prob`（`engine.py:3271`）で softmax T=3.5 の出力なので**較正器を通らない**。

この違いは実務上重い: `xgb_calibrator.pkl` を差し替えても穴マークは1つも動かない。
誤った記述が戻らないよう、ここで挙動として固定する。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betting.app_json import _assign_marks


def _horse(num, name, pn, odds, cal_prob):
    return {'num': num, 'name': name, 'pn': pn, 'win_odds': odds,
            'cal_prob': cal_prob, 'sim_ev': 5.0}


def _field():
    """穴の条件（6番人気以下・AI上位5位・EV>=1.5）を満たす馬を1頭だけ含む出走表。

    ana は AI 4位にしてある。3位以内だと「推」の条件（ai_rank <= _pop - 3）に
    先に拾われてしまい、穴の判定にたどり着かないため。
    """
    pns = [0.30, 0.29, 0.28, 0.10, 0.09]
    hs = [_horse(i, f'h{i}', pns[i - 1], float(i), 0.5) for i in range(1, 6)]
    hs.append(_horse(6, 'ana', 0.20, 20.0, 0.5))   # 6番人気・AI4位
    return hs


def test_ana_mark_fires_on_pn_even_when_cal_prob_is_tiny():
    """cal_prob が閾値を大きく下回っていても、pn が足りていれば穴が付く。"""
    hs = _field()
    hs[-1]['cal_prob'] = 0.001
    marks = _assign_marks(hs, sorted(hs, key=lambda h: h['win_odds']))
    assert marks[6] == '穴', marks


def test_ana_mark_does_not_fire_on_cal_prob_when_pn_is_tiny():
    """pn が閾値を下回れば、cal_prob がいくら大きくても穴は付かない。"""
    hs = _field()
    hs[-1]['pn'] = 0.14
    hs[-1]['cal_prob'] = 0.99
    marks = _assign_marks(hs, sorted(hs, key=lambda h: h['win_odds']))
    assert marks[6] != '穴', marks


def test_comment_does_not_claim_cal_prob_threshold():
    """コメントが「cal_prob>=0.15」に戻ったら落とす（記述と実装の食い違いの再発防止）。"""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'betting', 'app_json.py'), encoding='utf-8').read()
    i = src.index('# 穴マーク:')
    head = src[i:i + 400]
    assert 'cal_prob>=0.15' not in head.split('🔴')[0], head
