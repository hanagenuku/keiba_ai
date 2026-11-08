"""軸1頭ベース買い目の Python↔JS 突合フィクスチャを作る。

`tests/fixtures/axis_bets_crosscheck.json` を書き出す。中身は
  入力（馬番・馬名・RL順位・単勝オッズ・rating）と
  Python が出した買い目（build_axis_bets の生出力と _format_axis_bets の表示形）。

`tests/test_axis_bets_resync.mjs` が同じ入力を index.html の JS に流し、
1e-9 まで一致することを確かめる。**どちらかを変えればテストが落ちる**のが
目的なので、フィクスチャは手で書き換えず必ずこのスクリプトで作り直すこと。

実行:  python tests/make_axis_bets_crosscheck.py
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.betting.axis_bets import build_axis_bets                  # noqa: E402
from src.betting.app_json import _format_axis_bets                # noqa: E402
from src.betting.pl_exact import (exact_ticket_probabilities,      # noqa: E402
                                  exact_payouts_from_win_odds)

OUT = os.path.join(ROOT, 'tests', 'fixtures', 'axis_bets_crosscheck.json')

# 2026-11-08 の実データから、形の違うレースを選んで固定した
# （6頭・9頭・18頭・中穴なし・オッズ欠損あり）。latest.json は毎週変わるので
# ここに値ごと写し取って、テストが日によって変わらないようにしてある。
CASES = [
    # name, [(馬番, 馬名, RL順位, 単勝オッズ, rating)]
    ('18頭・東京11R相当', [
        (1, 'A', 11, 42.7, -1.90), (2, 'B', 4, 9.4, -0.42),
        (3, 'C', 15, 97.2, -2.55), (4, 'D', 2, 5.1, -0.05),
        (5, 'E', 6, 13.6, -0.78), (6, 'F', 1, 2.8, 0.52),
        (7, 'G', 13, 61.0, -2.18), (8, 'H', 7, 18.2, -0.96),
        (9, 'I', 3, 6.9, -0.24), (10, 'J', 9, 28.4, -1.41),
        (11, 'K', 5, 11.1, -0.60), (12, 'L', 17, 148.3, -2.94),
        (13, 'M', 8, 22.5, -1.18), (14, 'N', 10, 33.9, -1.63),
        (15, 'O', 12, 50.8, -2.04), (16, 'P', 14, 78.6, -2.37),
        (17, 'Q', 16, 118.0, -2.74), (18, 'R', 18, 199.5, -3.20),
    ]),
    ('9頭', [
        (1, 'A', 4, 7.8, -0.55), (2, 'B', 1, 2.1, 0.80),
        (3, 'C', 6, 15.2, -1.05), (4, 'D', 2, 3.4, 0.31),
        (5, 'E', 8, 41.0, -1.92), (6, 'F', 3, 5.6, -0.12),
        (7, 'G', 5, 11.0, -0.80), (8, 'H', 9, 72.0, -2.40),
        (9, 'I', 7, 24.0, -1.40),
    ]),
    ('6頭', [
        (1, 'A', 2, 3.1, 0.40), (2, 'B', 1, 1.9, 0.95),
        (3, 'C', 4, 8.9, -0.62), (4, 'D', 3, 5.2, -0.10),
        (5, 'E', 6, 33.0, -1.80), (6, 'F', 5, 14.5, -1.05),
    ]),
    ('中穴が1頭もいない（全馬7倍未満か30倍以上）', [
        (1, 'A', 1, 1.8, 1.10), (2, 'B', 2, 3.0, 0.45),
        (3, 'C', 3, 4.4, 0.05), (4, 'D', 4, 6.2, -0.35),
        (5, 'E', 5, 35.0, -1.85), (6, 'F', 6, 60.0, -2.30),
        (7, 'G', 7, 88.0, -2.70), (8, 'H', 8, 120.0, -3.05),
    ]),
    ('中穴が1頭だけ（三連複が組めない）', [
        (1, 'A', 1, 1.7, 1.20), (2, 'B', 2, 2.9, 0.50),
        (3, 'C', 3, 4.1, 0.10), (4, 'D', 4, 12.0, -0.85),
        (5, 'E', 5, 40.0, -2.00), (6, 'F', 6, 66.0, -2.45),
    ]),
    # 🔑 馬連の合成オッズ3.0の打ち切りが**実際に効く**ケース。
    #   中穴が7〜9倍に6頭いると、4点目で合成オッズが3.0を割るので
    #   そこで止まる（止まらない実装だと点数が増えてテストが落ちる）。
    ('中穴が低めに密集（馬連の合成オッズ3.0で打ち切られる）', [
        (1, 'A', 1, 2.6, 0.60), (2, 'B', 2, 7.2, -0.18),
        (3, 'C', 3, 7.6, -0.24), (4, 'D', 4, 8.0, -0.30),
        (5, 'E', 5, 8.4, -0.36), (6, 'F', 6, 8.8, -0.42),
        (7, 'G', 7, 9.2, -0.48),
        # ↓ 密集帯より弱いが単勝29倍（＝帯の中）。確率の降順では最後に来る。
        #   打ち切りが break なら密集帯で止まるのでこの馬は買わないが、
        #   continue だと「合成オッズを割らないから」と後から拾ってしまう。
        (8, 'H', 8, 29.0, -1.60),
        (9, 'I', 9, 55.0, -2.25), (10, 'J', 10, 80.0, -2.60),
    ]),
    # 🔑 推定配当の床 max(1.0, ...) が**実際に効く**ケース。
    #   5頭立てで軸が1.2倍と抜けていると「軸と相手がともに3着内」の確率が
    #   0.775（=1-控除率0.225）を超え、式は1.0未満を返す。JRAの払戻は
    #   元本100円を下回らないので1.0で床を張る（床が無いと落ちる）。
    ('5頭・軸が抜けている（ワイドの推定配当が床1.0に当たる）', [
        (1, 'A', 1, 1.2, 2.60), (2, 'B', 2, 7.5, -0.10),
        (3, 'C', 3, 60.0, -2.40), (4, 'D', 4, 120.0, -3.10),
        (5, 'E', 5, 200.0, -3.60),
    ]),
    ('オッズ欠損あり（2頭が0倍）', [
        (1, 'A', 3, 6.0, -0.20), (2, 'B', 1, 2.4, 0.70),
        (3, 'C', 5, 13.0, -0.95), (4, 'D', 2, 3.8, 0.20),
        (5, 'E', 4, 9.0, -0.60), (6, 'F', 6, 0.0, -1.70),
        (7, 'G', 7, 0.0, -2.10), (8, 'H', 8, 26.0, -1.45),
    ]),
]


def build(rows):
    # 🔑 cal_prob は rating の単調増加関数として与える。
    #   本番の cal_prob は Isotonic較正済みの複勝確率
    #   = isotonic(sigmoid(rating)) で、Isotonic回帰は**単調非減少**なので
    #   cal_prob の順位は rating の順位と一致する（較正が平らな帯の同値を除く）。
    #   JS は較正器を再現できないので rating で並べる。ここで cal_prob を
    #   sigmoid(rating) にしておくと、その等価性ごと突合できる
    #   （どちらかが別のキーで並べ始めたらテストが落ちる）。
    scored = [{'num': n, 'name': nm, 'rl_rank': rl, 'win_odds': od,
               'rating': rt, 'cal_prob': 1.0 / (1.0 + math.exp(-rt))}
              for n, nm, rl, od, rt in rows]
    nums = [h['num'] for h in scored]
    probs = exact_ticket_probabilities([h['rating'] for h in scored], nums)
    win_odds = {h['num']: h['win_odds'] for h in scored
                if h['win_odds'] and h['win_odds'] > 0}
    ab = build_axis_bets(scored, probs, exact_payouts_from_win_odds(win_odds))
    # タプルキーは JSON にできないので "2,6" の形にそろえる（JS側と同じ）
    def key(k):
        return ','.join(str(x) for x in k) if isinstance(k, tuple) else k
    ab_json = {bt: [dict(b, key=key(b['key'])) for b in ab[bt]]
               for bt in ('win', 'wide', 'quinella', 'trio')}
    ab_json['summary'] = ab['summary']
    return {
        'horses': [{'n': n, 'name': nm, 'rl_rank': rl, 'odds': od,
                    'rating': rt} for n, nm, rl, od, rt in rows],
        'axis_bets': ab_json,
        'formatted': _format_axis_bets(ab, scored),
    }


def main():
    out = {
        '_note': ('tests/make_axis_bets_crosscheck.py が生成する。'
                  '手で書き換えないこと。'),
        'cases': [{'name': name, **build(rows)} for name, rows in CASES],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write('\n')
    print(f'wrote {OUT}')
    for c in out['cases']:
        s = c['axis_bets']['summary']
        print(f"  {c['name']}: 軸#{s.get('axis')} "
              f"単{len(c['axis_bets']['win'])} ワイド{len(c['axis_bets']['wide'])} "
              f"馬連{len(c['axis_bets']['quinella'])} "
              f"三連複{len(c['axis_bets']['trio'])} 計{s.get('total_points')}点")


if __name__ == '__main__':
    main()
