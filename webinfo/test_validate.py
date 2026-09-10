"""webinfo/validate.py の検証。North Star #6: 本番と同じ形の記録で試す。"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as V                                    # noqa: E402

OK = {
    'web_oikiri_grade': '上位', 'web_oikiri_grade_raw': 'A',
    'web_oikiri_note': '一杯に追われて反応良く、動きは軽い',
    'web_stable_comment': '中立', 'web_stable_comment_raw': '順調に来ています',
    'web_condition_note': 'なし', 'web_condition_note_raw': '',
    'web_captured_at': '2026-09-13T08:30:00+09:00',
    'web_post_time': '2026-09-13T15:40:00+09:00',
    'web_source_urls': ['https://example.invalid/oikiri'],
    'web_n_found': 3, 'web_model': 'claude-opus-5', 'web_session_id': 's1',
}


class TestValidate(unittest.TestCase):
    def test_valid_record_passes(self):
        self.assertEqual(V.validate(OK), [])

    def test_unknown_key_is_rejected(self):
        """🔴 これが項目凍結の本体。後から項目を足せないこと。"""
        r = dict(OK); r['web_paddock_vibe'] = 'good'
        errs = V.validate(r)
        self.assertTrue(any('FIELDS.json に無いキー' in e for e in errs), errs)

    def test_enum_value_outside_options_is_rejected(self):
        r = dict(OK); r['web_oikiri_grade'] = 'S'      # 原表記は raw 側に入れる規約
        self.assertTrue(any('選択肢に無い' in e for e in V.validate(r)))

    def test_record_after_post_time_is_rejected(self):
        r = dict(OK); r['web_captured_at'] = '2026-09-13T15:41:00+09:00'
        self.assertTrue(any('発走後の記録' in e for e in V.validate(r)))

    def test_empty_source_urls_is_rejected(self):
        r = dict(OK); r['web_source_urls'] = []
        errs = V.validate(r)
        self.assertTrue(any('source_urls' in e for e in errs), errs)

    def test_n_found_must_match_actual(self):
        r = dict(OK); r['web_n_found'] = 4
        self.assertTrue(any('n_found' in e for e in V.validate(r)))

    def test_nashi_and_fumei_are_not_counted_as_found(self):
        """『なし』と『不明』を充足に数えない（充足率の測定が壊れるため）。"""
        r = dict(OK)
        r['web_stable_comment'] = '不明'; r['web_n_found'] = 2
        self.assertEqual(V.validate(r), [])

    def test_frozen_hash_mismatch_raises(self):
        real = open(V.SHA_PATH).read()
        try:
            open(V.SHA_PATH, 'w').write('0' * 64 + '\n')
            with self.assertRaises(V.FrozenSchemaError):
                V.validate(OK)
        finally:
            open(V.SHA_PATH, 'w').write(real)

    def test_merge_keeps_other_channels(self):
        """🔴 📝・🐎 のキーを消さないこと（2026-09-02 と同型の defect）。"""
        existing = json.dumps({'paddock_score': 7, 'start': 1}, ensure_ascii=False)
        merged = json.loads(V.merge_into_notes(existing, {'web_oikiri_grade': '上位'}))
        self.assertEqual(merged['paddock_score'], 7)
        self.assertEqual(merged['start'], 1)
        self.assertEqual(merged['web_oikiri_grade'], '上位')

    def test_merge_rejects_unprefixed(self):
        with self.assertRaises(ValueError):
            V.merge_into_notes('{}', {'oikiri_grade': '上位'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
