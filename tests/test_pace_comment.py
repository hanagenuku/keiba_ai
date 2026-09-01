"""AIコメントのペース文が、ペースと矛盾しないことを固定する。

旧実装は `f'{pace_str}想定({pace_pct}%)で{style}有利な展開。'` で、この
`style` は **AI本命自身の脚質** だった。ペースを一切見ていないのに
「ハイペースだから逃げ有利」と読める文になっており、本番の実コメント
822件中146件(17.8%)がペースの向きと矛盾していた。

実例（2026-08-30 札幌R11 日高ステークス）:
    旧: 「ハイペース想定(85%)で逃げ有利な展開」
    実際: 前半3F 34.2 のハイペース。逃げ2頭は12着・13着。
"""
import unittest

from src.features.engine import _pace_sentence, predict_race_pace


class TestPaceSentenceMatchesPace(unittest.TestCase):
    def test_high_pace_never_says_escape_is_favored(self):
        """今回の事故そのもの。修正前はここで「逃げ有利」が出ていた。"""
        for style in ('逃げ', '先行', '差し', '追込'):
            s = _pace_sentence('high', 'ハイペース', 85, style)
            self.assertIn('差し・追込に有利', s)
            self.assertNotIn('逃げ有利', s)
            self.assertNotIn('先行有利', s)

    def test_slow_pace_never_says_closers_are_favored(self):
        for style in ('逃げ', '先行', '差し', '追込'):
            s = _pace_sentence('slow', 'スロー', 41, style)
            self.assertIn('逃げ・先行に有利', s)
            self.assertNotIn('追込有利', s)
            self.assertNotIn('差し有利', s)

    def test_warns_when_the_pick_does_not_fit_the_pace(self):
        """本命の脚質が展開と噛み合わない時は、その旨を明示する。"""
        s = _pace_sentence('high', 'ハイペース', 85, '逃げ')
        self.assertIn('向かない', s)
        s = _pace_sentence('high', 'ハイペース', 85, '追込')
        self.assertIn('噛み合う', s)

    def test_middle_pace_makes_no_claim(self):
        """ミドルは向きが決まらない。作り話をしない。"""
        s = _pace_sentence('mid', 'ミドル', 49, '差し')
        self.assertNotIn('有利', s)
        self.assertNotIn('向かない', s)

    def test_favored_style_table_agrees_with_predict_race_pace(self):
        """`predict_race_pace` が持つ対応と食い違わないこと。

        片方だけ直されると、また「同じことをする箇所が2つあって片方だけ
        直す」事故になる（2026-08-09③の監査で名指しした型）。
        """
        # estimate_horse_style は馬の history[:5] の running_style を見る。
        # 本番の calc_all が渡すのと同じ形にする（North Star #6）。
        def horse(style):
            return {'history': [{'running_style': style} for _ in range(5)]}
        many_escape = [horse('逃げ') for _ in range(4)] + \
                      [horse('追込') for _ in range(6)]
        p = predict_race_pace(many_escape)
        self.assertEqual(p['pace'], 'high')
        self.assertEqual(p['favored_style'], 'stalk')     # 差し
        s = _pace_sentence('high', 'ハイペース', 90, '逃げ')
        self.assertIn('差し', s)

    def test_no_style_still_produces_a_sentence(self):
        self.assertTrue(_pace_sentence('high', 'ハイペース', 85, ''))
        self.assertTrue(_pace_sentence('slow', 'スロー', 30, None))


if __name__ == '__main__':
    unittest.main()
