"""エラータグ補正が既定OFFであることを固定する（2026-09-01）。

2,318レースのout-of-sample検証で機序が否定されたため既定OFFにした。
根拠（詳細は engine.py の適用箇所のコメント）:

  補正対象   予測23.0% / 実測22.4%  → ズレ -0.7pt
  補正対象外  予測22.3% / 実測21.7%  → ズレ -0.7pt   ← 完全に同じ

「この条件でAIは外す」とされた8条件の較正誤差が、それ以外と同じだった。
見かけのlog-loss改善(-0.0094)も、当たらないレースに同じ倍率を掛けると
同じだけ改善する(-0.0110)ので、条件固有の効果ではない。

2026-07-28 の rank_matrix_filter と同じ扱い（既定OFF・測定は残す）。
"""
import os
import unittest
from unittest import mock

from src.features import error_tags


class TestCorrectionIsOffByDefault(unittest.TestCase):
    def test_calc_all_does_not_import_get_correction_factor_by_default(self):
        """既定では補正関数が呼ばれないこと。

        `calc_all` を丸ごと動かすにはエンジンの初期化が要るので、
        適用箇所が環境変数でガードされていることをソースで固定する。
        """
        import inspect
        from src.features import engine
        src = inspect.getsource(engine.calc_all)
        self.assertIn("os.environ.get('ERROR_TAG_CORRECTION', '0') == '1'", src,
                      'エラータグ補正のガードが外れている')
        # ガードが from ... import より前にあること（副作用を先に止める）
        gi = src.index("ERROR_TAG_CORRECTION")
        ii = src.index('from src.features.error_tags import get_correction_factor')
        self.assertLess(gi, ii, 'ガードより先に import が走っている')

    def test_env_var_re_enables(self):
        with mock.patch.dict(os.environ, {'ERROR_TAG_CORRECTION': '1'}):
            self.assertEqual(os.environ.get('ERROR_TAG_CORRECTION'), '1')

    def test_tag_accumulation_is_kept(self):
        """タグの蓄積・分類は測定として残す（止めたのは適用だけ）。"""
        self.assertTrue(hasattr(error_tags, 'process_weekly_error_tags'))
        self.assertTrue(hasattr(error_tags, 'classify_race_tags'))
        self.assertTrue(hasattr(error_tags, 'accumulate_tags'))


class TestKnownGapsAreDocumented(unittest.TestCase):
    """検証で見つかった構造的な穴を、直す時まで見失わないように固定する。"""

    HANDLED = {'dist_ext_win', 'dist_short_win', 'escape_win', 'heavy_upset',
               'mare_upset', 'young_upset', 'jockey_switch_win'}

    def test_some_tags_have_no_horse_level_handler(self):
        """分類はされるが馬個別には何もしないタグが実在する。

        これらが active_adjustments に入っても順位は動かない。
        補正を再有効化するなら、まずここを埋めること。
        """
        import inspect, re
        body = inspect.getsource(error_tags.get_correction_factor)
        handled = set(re.findall(r"if '(\w+)' in active_tags", body))
        self.assertEqual(handled, self.HANDLED)
        defined = {t[0] if isinstance(t, (list, tuple)) else t
                   for t in error_tags.TAG_DEFINITIONS}
        for tag in ('position_bias', 'style_miss', 'class_miss'):
            self.assertIn(tag, defined, f'{tag} が分類対象から消えた')
            self.assertNotIn(tag, handled,
                             f'{tag} にハンドラが付いた。テストの期待値を更新すること')

    def test_condition_key_includes_track_condition(self):
        """条件キーが馬場を含むこと＝推論時に届かないキーが生まれる原因。

        推論時は出馬表に馬場が無いため常に「良」（2026-08-27 W1）。
        良以外で蓄積されたキー（2026年の23.6%）は永久に参照されない。
        """
        a = error_tags._condition_key('新潟', '芝', 1600, '良')
        b = error_tags._condition_key('新潟', '芝', 1600, '重')
        self.assertNotEqual(a, b, '馬場がキーに入っていない（前提が変わった）')
        self.assertTrue(a.endswith('_良'))


if __name__ == '__main__':
    unittest.main()
