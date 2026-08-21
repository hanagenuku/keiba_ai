"""単勝オッズ盤が「盤ごと」壊れている場合の検出テスト。

2026-08-20 に判明した事故:
土曜夜の日曜予想は毎回、専用オッズページが全滅（odds_coverage=0.0）し、
出馬表側の値で補完していた。ところがその値は単勝オッズではなかった。
2026-08-16の日曜予想を翌朝の確定オッズと突合すると:

    1番人気が一致したレース  4 / 17
    log相関                 +0.24
    比の中央値              0.145

1頭ずつ見ると「1.0倍未満」ではないので既存の検査は素通りする。
盤全体の Σ(1/オッズ) を見ると 7.16（正常は約1.25）で一発で分かる。

実測（latest.json 全76世代・2,041レース）: 正常な盤は最大 1.577、
壊れた盤は最小 1.714 で二峰に完全に分離する。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (MAX_VALID_ODDS_BOOK_SUM, _INVALID_ODDS_WARNED,
                                 _sanitize_odds_book)


def _horses(odds):
    return [{'num': i + 1, 'win_odds': o} for i, o in enumerate(odds)]


def test_threshold_sits_in_the_measured_gap():
    """閾値は実測の空白帯（正常最大1.577〜異常最小1.714）の中にあること。"""
    assert 1.577 < MAX_VALID_ODDS_BOOK_SUM < 1.714


def test_normal_book_is_untouched():
    """正常な単勝盤（Σ≈1.25）はそのまま。"""
    hs = _horses([2.5, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 50.0])
    before = [h['win_odds'] for h in hs]
    assert _sanitize_odds_book(hs, 'r1') == 0
    assert [h['win_odds'] for h in hs] == before


def test_broken_book_is_cleared():
    """2026-08-16 大垣特別と同じ形（全馬2.0倍前後・Σ≈7）は盤ごと無効化する。"""
    hs = _horses([2.0, 2.0, 2.0, 2.0, 2.1, 2.2, 2.2, 2.3,
                  2.3, 2.4, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2])
    assert _sanitize_odds_book(hs, 'r2') == 16
    assert all(h['win_odds'] is None for h in hs)


def test_partial_book_is_untouched():
    """一部の馬しかオッズを持たない盤は Σ が小さくなるだけなので触らない。

    上限のみを見る設計であることの回帰テスト（下限で弾くと部分取得を誤検知する）。
    """
    hs = _horses([3.0, 9.0, 25.0]) + [{'num': 9, 'win_odds': None}] * 10
    assert _sanitize_odds_book(hs, 'r3') == 0
    assert hs[0]['win_odds'] == 3.0


def test_too_few_horses_is_untouched():
    hs = _horses([1.2, 1.2])
    assert _sanitize_odds_book(hs, 'r4') == 0
    assert hs[0]['win_odds'] == 1.2


def test_warns_once_per_race():
    _INVALID_ODDS_WARNED.clear()
    for _ in range(3):
        _sanitize_odds_book(_horses([2.0] * 12), 'r5')
    assert sum(1 for k in _INVALID_ODDS_WARNED if k[0] == 'r5') == 1


def test_calc_all_applies_the_book_check():
    """calc_all の popularity 導出より前に盤の検査が入っていること。"""
    src = open(os.path.join(os.path.dirname(__file__), '..',
                            'src/features/engine.py'), encoding='utf-8').read()
    i_book = src.index('_sanitize_odds_book(_horses_in')
    i_pop = src.index("_h['popularity'] = _rank")
    assert i_book < i_pop
