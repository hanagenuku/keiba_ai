"""sunday_results.py の回帰テスト。

日曜結果ワークフローは record_all_shadow_bets() を recommended_race_ids 無しで
呼んでおり、実際に推奨・購入した日曜レースでも shadow_bets.was_recommended が
常に0で記録されていた（weekend.py の土曜側は bets テーブルから当日の
race_id を引いて正しく渡していたのに、日曜側だけこの手順が抜けていた）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scripts.sunday_results as sunday_results


def test_sunday_results_module_imports_without_error():
    assert hasattr(sunday_results, 'main')


def test_sunday_results_passes_recommended_race_ids_to_shadow_bets():
    """record_all_shadow_bets が recommended_race_ids 付きで呼ばれていることを
    ソースレベルで確認する（2026-07-21に発見した was_recommended 常時0バグの
    再発防止）。"""
    src_path = os.path.join(os.path.dirname(__file__), '..',
                             'scripts', 'sunday_results.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert 'recommended_race_ids=_rec_ids' in src
    assert "SELECT DISTINCT race_id FROM bets WHERE date=?" in src


def test_sunday_results_no_longer_calls_dead_correction_table():
    """update_correction_table() はcorrection_table.jsonを書くだけで、
    engine.py/app_json.py/make_bets.pyのどこからも読まれていないと判明したため
    呼び出しを削除した。再度呼び出しが復活していないことを確認する。"""
    assert not hasattr(sunday_results, 'update_correction_table')
    src_path = os.path.join(os.path.dirname(__file__), '..',
                             'scripts', 'sunday_results.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert 'update_correction_table' not in src
