"""AI強気ペア（A×D）の馬連・ワイド推奨を生成する。

## 背景

2026-07-31 の大規模ウォークフォワード検証（4,173レース）で、
「市場人気より AI が高く評価している馬（AI強気）」どうしを組む馬券が、
市場が示唆する的中率を上回ることが分かった。

単勝では AI強気 × 中位人気 が回収率 100.9%（N=922）＝損益分岐ちょうどで止まったが、
**両脚ともエッジを持つ組み合わせ**なら控除率を上回りうる、という発想。
片脚だけ（AI強気 × 中立馬）は実測でも 1.26 前後にとどまり基準未達だった。

## 定義

- **A群（軸）**: 市場4-6番人気 かつ AI順位 < 市場人気
- **D群（相手）**: 市場7番人気以下 かつ AI順位 < 市場人気

## 実測（4,173レース・月次ウォークフォワード）

| | 組数 | 実的中率 | 市場示唆 | 比 |
|---|---|---|---|---|
| 馬連 | 7,937 | 0.44% | 0.24% | **1.82** |
| ワイド | 7,937 | 1.45% | 0.93% | **1.56** |

控除率22.5%の損益分岐は比1.29なので、いずれも上回る。
各群のAI最上位1頭ずつに絞ると的中率はほぼ倍になる
（馬連 0.44%→0.80% / ワイド 1.45%→2.29%）ため、既定では絞る。

## ⚠ 重大な限界（実装者・利用者は必ず読むこと）

**回収率は検証できていない。** 馬連・ワイドの組み合わせ配当は
keiba.db にも history.db にも保存されていないため、
上表の「比」は Plackett-Luce 近似で推定した市場示唆確率との比率でしかない。
しかも A×D は的中率0.4〜1.5%という**低確率領域**で、
この近似の誤差が最も大きい場所にあたる。

したがって本推奨は「期待値がプラスであることが確認された買い目」ではなく
**「比の上では有望だが回収率未検証の仮説」**である。

同種の仮説は 2026-07-27 に本番投入され（人気×RL乖離マトリクス）、
翌日の大規模検証で否定された前例がある（182.8% → 71.3%）。
実配当が取れるようになるまでは、エア馬券での検証に留めること。

環境変数 `AI_PAIR_BETS=0` で無効化できる。
"""
import os

# A群（軸）: 市場4-6番人気 かつ AI強気
AXIS_POP_MIN, AXIS_POP_MAX = 4, 6
# D群（相手）: 市場7番人気以下 かつ AI強気
PARTNER_POP_MIN = 7

# 各群から取る頭数の既定値。検証で的中率が最も高かった 1×1 を既定とする。
DEFAULT_MAX_AXIS = 1
DEFAULT_MAX_PARTNER = 1

# 検証時の実測値（アプリ表示用。数字を変えるときは必ず再検証すること）
MEASURED = {
    'races': 4173,
    'umaren': {'ratio': 1.82, 'hit_rate': 0.80},
    'wide': {'ratio': 1.56, 'hit_rate': 2.29},
    'breakeven_ratio': 1.29,
}


def _enabled():
    """有効かどうか。既定ON、`AI_PAIR_BETS=0` で無効化。"""
    return os.environ.get('AI_PAIR_BETS', '1') not in ('0', 'false', 'False')


def _pop_of(h):
    """市場人気を取り出す。取得できなければ None。"""
    for key in ('popularity', '_pop'):
        v = h.get(key)
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 18:
            return v
    return None


def _ai_rank_of(h):
    for key in ('rl_rank', 'ai_rank'):
        v = h.get(key)
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= 18:
            return v
    return None


def classify(h):
    """馬を A（軸）/ D（相手）/ None に分類する。

    どちらも「AI順位 < 市場人気」＝AIが市場より高く評価していることが条件。
    """
    pop = _pop_of(h)
    ai = _ai_rank_of(h)
    if pop is None or ai is None:
        return None
    if ai >= pop:                       # AI強気でなければ対象外
        return None
    if AXIS_POP_MIN <= pop <= AXIS_POP_MAX:
        return 'A'
    if pop >= PARTNER_POP_MIN:
        return 'D'
    return None


def build_ai_pair_bets(scored, max_axis=DEFAULT_MAX_AXIS,
                       max_partner=DEFAULT_MAX_PARTNER):
    """A×D の馬連・ワイド推奨を組み立てる。

    Args:
        scored: calc_all() が返した馬リスト
        max_axis / max_partner: 各群から取る頭数（AI評価上位から）

    Returns:
        dict: {'umaren': [...], 'wide': [...], 'axis': [...], 'partners': [...]}
              対象馬がいない/無効時は空の dict
    """
    if not _enabled() or not scored:
        return {}

    axis, partners = [], []
    for h in scored:
        g = classify(h)
        if g == 'A':
            axis.append(h)
        elif g == 'D':
            partners.append(h)
    if not axis or not partners:
        return {}

    axis.sort(key=lambda x: _ai_rank_of(x) or 99)
    partners.sort(key=lambda x: _ai_rank_of(x) or 99)
    axis = axis[:max(1, max_axis)]
    partners = partners[:max(1, max_partner)]

    pairs = []
    for a in axis:
        for p in partners:
            na, np_ = a.get('num'), p.get('num')
            if na is None or np_ is None or na == np_:
                continue
            pairs.append({
                'nums': sorted([int(na), int(np_)]),
                'axis_num': int(na), 'axis_name': a.get('name', ''),
                'axis_pop': _pop_of(a), 'axis_ai': _ai_rank_of(a),
                'partner_num': int(np_), 'partner_name': p.get('name', ''),
                'partner_pop': _pop_of(p), 'partner_ai': _ai_rank_of(p),
            })
    if not pairs:
        return {}

    return {
        'umaren': pairs,
        'wide': pairs,          # 同じ組み合わせを両券種で買う想定
        'axis': [int(a.get('num')) for a in axis if a.get('num') is not None],
        'partners': [int(p.get('num')) for p in partners if p.get('num') is not None],
        'measured': MEASURED,
        'unverified': True,     # 回収率未検証であることを表示側に伝える
    }


def format_for_app(pair_bets):
    """アプリ表示用の文字列リストにする。"""
    if not pair_bets or not pair_bets.get('umaren'):
        return []
    out = []
    for p in pair_bets['umaren']:
        a, b = p['nums']
        out.append(
            f"馬連/ワイド {a}-{b}  "
            f"(軸 #{p['axis_num']} {p['axis_pop']}番人気→AI{p['axis_ai']}位 / "
            f"相手 #{p['partner_num']} {p['partner_pop']}番人気→AI{p['partner_ai']}位)"
        )
    return out
