"""推奨フラグ（`rec`）はサーバの値を据え置く——クライアントで EV で付け外ししない。

## 何を固定するか

`index.html` の「🔄直前オッズ取得」は、2026-06 から `updateRecFlag()` で
推奨フラグを書き換えていた:

    推奨外し: RL1 の単勝オッズ < 1.5倍（妙味なし） or RL上位3頭全員の sim_ev < 0.8
    推奨追加: 非推奨レースで RL1〜3 のどれかが sim_ev >= 1.2 かつ 2.0〜30.0倍

🔴 どちらも根拠が `sim_ev` = AI確率 × オッズ だった。EV による選別は
2026-07-05 / 07-30 / 08-31 の3回とも否定されている（North Star #9）。
AI確率は意図的に平坦なので EV はほぼオッズの単調増加関数で、
「EV が高い」は「人気薄」とほぼ同義になる。だから
  - 「オッズ < 1.5倍だから推奨を外す」は**堅いレースを外す**操作
  - 「EV >= 1.2 だから推奨を足す**は人気薄のいるレースを足す**操作
でしかなく、どちらも推奨の定義（`conf >= 66`・`src/product/scope.py`）と無関係に
商品の範囲を動かしていた。

2026-11-10 にユーザーが「①EV判定ごと外して サーバの rec を据え置く」を選択し、
関数・呼び出し・表示（「推奨取消」「NEW推奨」バッジ）を丸ごと削除した。
軸1頭ベース買い目の組み直し（`recalcAxisBets`）が EV を一切使わないのと同じ理由。

⚠ ここで固定しているのは「クライアントが rec を触らない」ことだけ。
サーバ側（`article_kind()` / `conf`）が推奨を決める形は変えていない。
"""
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, 'index.html'), encoding='utf-8') as f:
    HTML = f.read()

JS = '\n'.join(re.findall(r'<script>(.*?)</script>', HTML, re.S))


def test_no_update_rec_flag():
    """EV で推奨を付け外しする関数を復活させない。"""
    assert 'updateRecFlag(' not in JS and 'function updateRecFlag' not in JS, (
        'updateRecFlag() が戻っている。推奨フラグを EV で動かすのは '
        '2026-11-10 にユーザー判断で外した（North Star #9）。'
    )


def test_client_never_assigns_rec():
    """クライアントは `race.rec` に代入しない（サーバの値を据え置く）。"""
    hits = re.findall(r'\brace\.rec\s*=(?!=)', JS)
    assert not hits, (
        'クライアントが race.rec に代入している: %r。'
        '推奨はサーバ（conf >= 66）だけが決める。' % hits
    )


def test_no_morning_rec_snapshot():
    """付け外しの表示用に朝の rec を退避する仕掛けも残さない。"""
    for dead in ('_morning_rec', '_rec_reason'):
        assert dead not in HTML, (
            '%s が残っている。推奨フラグを据え置くなら朝との差分は存在しない。' % dead
        )


def test_sim_ev_not_used_for_scope():
    """`sim_ev` が推奨・範囲の判定に使われていない（表示とEV欄だけ）。

    EV を画面に出すこと自体は禁じていない（人間が読む列）。
    禁じているのは EV で**買う/買わない・推奨する/しない**を決めること。
    """
    for m in re.finditer(r'[^\n]*sim_ev[^\n]*', JS):
        line = m.group(0)
        assert 'race.rec' not in line, '推奨判定に sim_ev が入っている: %s' % line.strip()
