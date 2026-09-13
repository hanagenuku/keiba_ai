"""pace_bonus（手書きのペース補正）が既定OFFであることの回帰テスト。

🔴 なぜ切ったか（pace/RESULTS.md・2026-09-13）
32,904頭 / 2,388レース（train_end 2025-06-30 で 2026年は完全OOS）で測ると:

    3着内AUC  補正なし 0.7685 → 本番 0.7665  （−0.0019・3四半期とも負）
              対照 シャッフル 0.7648 / 対照 符号反転 0.7631
    係数掃引  探索期で選ぶ最良 α* = 0（α に対して単調に悪化）
    押し上げ  100%超のセル 0/6。10+人気では符号反転の対照(96.9%)が本番(68.3%)に勝つ

レース内相対の混合は単調変換なので順位を動かさない。**順位を動かしていたのは
pace_bonus だけ**（19.1%の馬・最大12位）。導入以来一度も検証されておらず、
2026-08-24 E-1 では「CSVに f_front_adv/f_back_adv が無く再現できない」として
検証対象からも外れていた。2026-09-01 の error_tags・2026-07-28 の
rank_matrix_filter と同じ扱い（既定OFF・環境変数で復帰）。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.features.engine as engine


def _make_out(n=6):
    """calc_all の最終ブロックが触る形だけを持った馬リスト。"""
    return [{'num': i + 1, 'total': 5.0 + i * 0.1,
             'f_relative_score': float(i), 'f_front_adv': 0.8 - 0.2 * i,
             'f_back_adv': -0.1 * i}
            for i in range(n)]


def _apply(out, on):
    """engine.calc_all の該当ブロックと同じ計算（環境変数だけを切り替える）。"""
    prev = os.environ.get('PACE_BONUS')
    os.environ['PACE_BONUS'] = '1' if on else '0'
    try:
        pace_on = os.environ.get('PACE_BONUS', '0') == '1'
        res = []
        for h in out:
            pb = (h['f_front_adv'] + h['f_back_adv']) * 0.5 if pace_on else 0.0
            res.append(round(h['total'] * 0.9 + h['f_relative_score'] * 0.1 + pb, 2))
        return res
    finally:
        if prev is None:
            os.environ.pop('PACE_BONUS', None)
        else:
            os.environ['PACE_BONUS'] = prev


class TestDefaultOff:
    def test_source_gates_pace_bonus_behind_env(self):
        src = open(engine.__file__, encoding='utf-8').read()
        assert "os.environ.get('PACE_BONUS', '0') == '1'" in src, \
            'pace_bonus が環境変数で切り替えられる形になっていない'
        # 既定は OFF（'1' がデフォルトになっていないこと）
        assert "os.environ.get('PACE_BONUS', '1')" not in src

    def test_off_by_default_changes_nothing_but_the_bonus(self):
        out = _make_out()
        off, on = _apply(out, False), _apply(out, True)
        assert off != on, 'フィクスチャが pace_bonus の効かない形になっている'
        for h, o in zip(out, off):
            assert math.isclose(o, round(h['total'] * 0.9 + h['f_relative_score'] * 0.1, 2))

    def test_env_restores_old_behaviour(self):
        out = _make_out()
        on = _apply(out, True)
        for h, o in zip(out, on):
            pb = (h['f_front_adv'] + h['f_back_adv']) * 0.5
            assert math.isclose(o, round(h['total'] * 0.9 + h['f_relative_score'] * 0.1 + pb, 2))


class TestRelativeBlendDoesNotReorder:
    """相対混合が単調変換であること（＝順位を動かすのは pace_bonus だけ）。

    この前提が崩れると「pace_bonus を切れば RL はモデルの順位になる」という
    今回の判断の土台が消えるので、テストで固定する。
    """

    def test_blend_preserves_order(self):
        import random
        rnd = random.Random(0)
        for _ in range(500):
            n = rnd.randint(5, 18)
            t = [rnd.uniform(0, 10) for _ in range(n)]
            mn, mx = min(t), max(t)
            rel = [(x - mn) / max(mx - mn, 0.1) * 10 for x in t]
            blended = [0.9 * x + 0.1 * r for x, r in zip(t, rel)]
            assert (sorted(range(n), key=lambda i: -t[i])
                    == sorted(range(n), key=lambda i: -blended[i]))
