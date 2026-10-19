"""禁止表現の判定が**馬名・騎手名に当たらない**ことを固定する。

2026-10-18 の実データ（新潟9R）に「ベルウッドバイオ」がいて、
`FORBIDDEN_PATTERNS` の `ウッド`（調教コース）に一致していた。実害は2つ:

1. `tests/test_product_layer.py` の禁止表現テスト2件が落ちる
2. この馬が「市場の支持ほどAIは評価していない馬」に選ばれていたら、
   `race_comment` が馬名を本文に書くので `_join` が AssertionError を投げ、
   **そのレースの記事生成ごと落ちる**（実際は3番人気で条件を満たさず助かった）

guard の目的は「**こちらが**陣営コメント・調教タイム・パドックに触れる文を
書かないこと」なので、こちらが書いたのではない固有名詞は判定の対象外にする。
⚠ ただし guard を弱める方向の変更なので、パターン自体が減っていないことも固定する。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import commentary


def test_horse_name_containing_a_forbidden_word_is_not_a_hit():
    assert commentary.find_forbidden('ベルウッドバイオは3番人気。',
                                     names=('ベルウッドバイオ',)) is None


def test_the_same_word_outside_a_name_is_still_a_hit():
    """固有名詞を伏せても、こちらが書いた文の中の語は必ず捕まえる。"""
    assert commentary.find_forbidden('ベルウッドバイオはウッドで併せ馬。',
                                     names=('ベルウッドバイオ',)) == 'ウッド'
    assert commentary.find_forbidden('ウッドの時計が良い', names=()) == 'ウッド'


def test_names_argument_is_optional():
    """既存の呼び出し（names なし）の挙動を変えない。"""
    assert commentary.find_forbidden('調教タイムは52.5秒。') is not None
    assert commentary.find_forbidden('調教師は友道康夫。') is None


def test_pattern_list_is_not_shrunk():
    """「馬名に当たるから」とパターンを削る方向に逃げていないことを固定する。"""
    for pat in ('陣営', '厩舎コメント', '関係者', '談話', r'調教(?!師)', '追い切り',
                '坂路', 'ウッド', '馬なり', 'パドック', '気配', '仕上がり',
                'とのこと', 'と話して'):
        assert pat in commentary.FORBIDDEN_PATTERNS, pat
    assert len(commentary.FORBIDDEN_PATTERNS) == 14


def test_race_comment_does_not_crash_on_a_danger_horse_named_with_a_forbidden_word():
    """この修正がいちばん効く場所。馬名を本文に書いても落ちない。"""
    race = {'conf': 70, 'r': 9}
    horses = [{'n': 1, 'rl_rank': 1}]
    tickets = {'verdict': 'strong'}
    dangers = [{'num': 5, 'name': 'ベルウッドバイオ'}]
    text = commentary.race_comment(race, horses, tickets, dangers)
    assert 'ベルウッドバイオ' in text
