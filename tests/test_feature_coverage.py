"""engine.py の f_* 関数が XGB に届いているかを固定する。

2026-08-26の棚卸しで、12個の f_* のうち6個がXGBに届いていないと分かった。
うち3個は別名のより厚い特徴量で置き換わっており穴ではなかったが、
残り3個は「繋がっていないことに誰も気づいていない」状態だった。

⚠ 「繋がっていない＝繋げば効くはず」ではない。f_bias は 2026-08-06 に
実際に繋いで測って **-0.0005**（悪化）だった。だからこのテストが求めるのは
「繋げ」ではなく **「測ったうえで、繋がない理由を書け」** である。
"""
import json
import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# XGBに入っていないことが正しい f_* と、その理由。
# 新しく免除を足すときは、必ず「測った結果」か「代替がどれか」を書くこと。
ALLOW = {
    'f_dist_v2':  '代替あり: f_dist_fukusho / f_optimal_distance など7列が担当',
    'f_rl':       '代替あり: f_rl_rank（レース内のRL順位）がモデルに入っている',
    'f_weight':   '代替あり: f_weight_load / f_weight_trend_avg など5列が担当',
    'f_bias':     '測って否定: 2026-08-06 に繋いで AUC -0.0005（132特徴量 0.7905 < 126特徴量 0.7910）',
    'f_maturity': '測って否定: 2026-08-26 に本番同型で3窓測り AUC -0.0002（2/3窓が悪化）。'
                 'そもそも history.db に G1/G2/G3/L が1行も無く重賞が全部 OP に潰れるため、'
                 '非ゼロになるのは全体の9.9%だけ',
    'f_rotation': '測って否定: 2026-08-26 に f_maturity と同時に測り AUC -0.0002。'
                  '重要度 0.45〜0.51%（106〜121位）',
}


def _rule_functions():
    src = open(os.path.join(_ROOT, 'src', 'features', 'engine.py'), encoding='utf-8').read()
    return sorted(set(re.findall(r'^def (f_[a-z0-9_]+)\(', src, re.M)))


def _model_features():
    with open(os.path.join(_ROOT, 'data', 'xgb_feature_cols.json'), encoding='utf-8') as f:
        cols = json.load(f)
    return set(cols['feature_cols'] if isinstance(cols, dict) else cols)


class TestRuleFeaturesReachTheModel(unittest.TestCase):
    def test_every_rule_function_is_wired_or_excused(self):
        model = _model_features()
        unexplained = [f for f in _rule_functions()
                       if f not in model and f not in ALLOW]
        self.assertEqual(
            unexplained, [],
            'engine.py にあるのにXGBに届いておらず、理由も書かれていない f_* がある。\n'
            '繋ぐ前に効果を測り、結果を ALLOW に理由付きで書くこと（f_bias の前例参照）:\n'
            f'  {unexplained}')

    def test_allow_entries_are_real_functions(self):
        """代替やモデル入りになった関数が ALLOW に残り続けないようにする。"""
        rules, model = set(_rule_functions()), _model_features()
        for name in ALLOW:
            self.assertIn(name, rules,
                          f'{name} は engine.py にもう存在しない。ALLOW から消すこと')
            self.assertNotIn(name, model,
                             f'{name} はモデルに入った。ALLOW から消すこと')

    def test_allow_entries_have_a_reason(self):
        for name, reason in ALLOW.items():
            self.assertGreater(len(reason), 20,
                               f'{name} の免除理由が短すぎる。測った数字か代替名を書くこと')


if __name__ == '__main__':
    unittest.main()
