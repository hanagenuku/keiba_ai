"""週末 web 情報の記録（2026-09-15）。

書き手は毎回ゼロから始まる別セッションの Claude なので、
**出典の無い値・スキーマ違反をコードで弾く**。
このプロジェクトでは「作り話の数字が画面に出ていた」事故が3回ある
（ROI予測150%のべた書き / AI xx%バッジの逆向きの式 / 穴馬率が常に0%）。
"""
import json
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.utils.web_notes import (
    validate_web_note, usable_going, load_schema, load_notes,
    CONFIDENCE_VALUES, GOING_VALUES,
)


def _ok(**kw):
    note = {'date': '2026-09-19', 'collected_at': '2026-09-19T08:40:00+09:00',
            'phase': 'pre', 'model_consumed': False,
            'venues': {'中山': {'going': {'芝': '重', 'ダート': '不良'},
                               'weather': '雨', 'source': 'search:radionikkei',
                               'confidence': 'reported'}},
            'races': []}
    note.update(kw)
    return note


class TestSchemaIsRealAndReadable:
    def test_schema_loads_and_declares_it_is_not_consumed(self):
        s = load_schema(BASE)
        assert s['model_consumed'] is False
        assert s['why_not_consumed']

    def test_notes_directory_exists(self):
        assert os.path.isdir(os.path.join(BASE, 'data', 'web_notes'))


class TestValidation:
    def test_a_well_formed_note_passes(self):
        assert validate_web_note(_ok()) == []

    def test_a_value_without_a_source_is_rejected(self):
        """🔴 本命。出典の無い値を作らせない。"""
        n = _ok()
        del n['venues']['中山']['source']
        assert any('source' in e for e in validate_web_note(n))

    def test_confidence_must_be_declared(self):
        n = _ok()
        n['venues']['中山']['confidence'] = 'たぶん'
        assert any('confidence' in e for e in validate_web_note(n))

    def test_model_consumed_true_is_rejected(self):
        """記録のみという契約を、ファイル側からも破れないようにする。"""
        assert any('model_consumed' in e
                   for e in validate_web_note(_ok(model_consumed=True)))

    def test_post_phase_is_rejected(self):
        assert any('phase' in e for e in validate_web_note(_ok(phase='post')))

    def test_bad_going_value_is_rejected(self):
        n = _ok()
        n['venues']['中山']['going']['芝'] = 'やや重'   # 正しくは '稍重'
        assert validate_web_note(n)

    def test_unknown_surface_is_rejected(self):
        n = _ok()
        n['venues']['中山']['going']['障害'] = '良'
        assert validate_web_note(n)

    def test_race_note_needs_a_kind_and_a_source(self):
        n = _ok(races=[{'race_id': 'x', 'note': 'y'}])
        errs = validate_web_note(n)
        assert any('kind' in e for e in errs)
        assert any('source' in e for e in errs)

    def test_missing_date_is_rejected(self):
        n = _ok()
        del n['date']
        assert any('date' in e for e in validate_web_note(n))

    def test_bad_date_format_is_rejected(self):
        assert validate_web_note(_ok(date='2026/09/19'))


class TestUsableGoing:
    def test_only_official_is_usable(self):
        """報道ベースの値は記録どまり。特徴量の候補にしない。"""
        assert usable_going(_ok()) == {}

    def test_official_is_exposed(self):
        n = _ok()
        n['venues']['中山']['confidence'] = 'official'
        assert usable_going(n) == {('中山', '芝'): '重', ('中山', 'ダート'): '不良'}


class TestEveryRecordedFileIsValid:
    def test_all_notes_on_disk_pass(self):
        """実際に溜まっていくファイルを毎回検証する（書き手がLLMなので）。"""
        for note in load_notes(BASE):
            assert validate_web_note(note) == [], note.get('date')


class TestItIsNotWiredIntoPrediction:
    def test_nothing_in_the_prediction_path_reads_web_notes(self):
        """🔴 『記録のみ』が本当に守られているかをコードで固定する。"""
        import subprocess
        out = subprocess.run(
            ['grep', '-rn', 'web_notes', 'src/features', 'src/betting',
             'src/models', 'scripts', 'index.html'],
            cwd=BASE, capture_output=True, text=True).stdout
        assert out.strip() == '', f'予想の経路が web_notes を読んでいる:\n{out}'
