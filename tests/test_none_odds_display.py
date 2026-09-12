"""オッズが1頭も取れなかった日に、予想生成が丸ごと落ちないことの回帰テスト。

🔴 2026-09-12（土）夜の本番実行（weekend.py --mode saturday）は、日曜23レースの
出走表まで正常に取得しながら、厳選レースの1件目を表示した直後に exit code 1 で
落ち、**土曜の結果と日曜の予想が丸ごと失われた**。

原因は表示行1つだけ:

    print(f'  ◎ ... {top1.get("win_odds", 0):.1f}倍 ...')

`win_odds` は「キーはあるが値が None」を取りうる（出馬表にオッズ列が無い／
`_sanitize_odds_book` が壊れた盤を丸ごと無効化した）。`.get()` の既定値は
**キーが無いときにしか効かない**ので None が素通りし、`f'{None:.1f}'` が
TypeError になる。実際その日は専用オッズページが両会場とも全滅し
（`オッズ反映: 0頭 / 23R`）、出馬表側のオッズも Σ(1/オッズ)=12.69 等で
無効化されていた。

2026-08-07④（`agari_rank=None` で calc_all が TypeError）と同じ型。
本番と同じ経路（predict_next_day / friday_predict.main の表示部）で固定する。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.fmt import format_odds


class TestFormatOdds:
    def test_none_is_not_formatted_as_a_number(self):
        # 修正前の実装（f'{v:.1f}倍'）はここで TypeError になっていた
        assert format_odds(None) == 'オッズ不明'

    def test_normal_odds(self):
        assert format_odds(3.14) == '3.1倍'
        assert format_odds(1.0) == '1.0倍'

    def test_sub_one_is_unknown_not_zero(self):
        # 単勝は元返し1.0倍が下限。0.0倍/0.2倍と表示するのは嘘になる
        assert format_odds(0) == 'オッズ不明'
        assert format_odds(0.2) == 'オッズ不明'

    def test_garbage_is_unknown(self):
        assert format_odds('x') == 'オッズ不明'


def _race():
    return {'id': '20260913_09_02', 'racecourse': '阪神', 'race_num': 2,
            'race_name': '2歳未勝利', 'distance': 1600, 'surface': '芝',
            'num_horses': 2, 'start_time': '10:10', 'date': '2026-09-13',
            'horses': [{'num': 1, 'name': 'ウマ1', 'win_odds': None},
                       {'num': 2, 'name': 'ウマ2', 'win_odds': None}]}


def _scored():
    # _sanitize_odds_book が盤ごと無効化した後の calc_all 出力の形。
    # win_odds キーは存在し、値が None であることが要点。
    return [{'num': 1, 'name': 'ウマ1', 'win_odds': None, 'total': 5.2,
             'pn': 0.30, 'rl_rank': 1, 'cal_prob': 0.5, 'ability_margin': 0.1},
            {'num': 2, 'name': 'ウマ2', 'win_odds': None, 'total': 4.1,
             'pn': 0.20, 'rl_rank': 2, 'cal_prob': 0.4, 'ability_margin': 0.0}]


def _patch_common(monkeypatch, mod, tmp_path, race, scored):
    monkeypatch.setattr(mod, 'APP_PATH', str(tmp_path / 'latest.json'))
    monkeypatch.setattr(mod, 'fetch_races_on_date',
                        lambda *a, **k: ([race], []))
    # 本番と同じ「オッズが1頭も取れなかった」状態
    monkeypatch.setattr(mod, 'fetch_odds_map', lambda sess, races: {})
    monkeypatch.setattr(mod, 'apply_odds_to_races', lambda races, mom: 0)
    monkeypatch.setattr(mod, 'calc_all', lambda race, bias: scored)
    monkeypatch.setattr(mod, 'save_race_predictions', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'save_race_db', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'save_bets_db', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'log_bet_simulation', lambda *a, **k: None)
    monkeypatch.setattr(mod, 'build_market_odds_from_races', lambda races: {})
    monkeypatch.setattr(mod, 'to_app_json', lambda *a, **k: {'races': []})
    # 厳選レースが1件出る = 表示行を必ず通る
    monkeypatch.setattr(mod, 'select_quality_races',
                        lambda races, bias: [{'race': race, 'top1': scored[0],
                                              'scored': scored}])
    monkeypatch.setattr(mod, 'make_bets', lambda c: [])


def test_sunday_prediction_survives_all_odds_missing(monkeypatch, tmp_path, capsys):
    """本番が落ちた経路そのもの（weekend.py --mode saturday の日曜予想生成）。"""
    import scripts.weekend as weekend
    from datetime import datetime, timedelta, timezone
    JST = timezone(timedelta(hours=9))

    race, scored = _race(), _scored()
    _patch_common(monkeypatch, weekend, tmp_path, race, scored)
    monkeypatch.setattr(weekend, '_already_generated', lambda *a, **k: False)

    # 修正前はここで TypeError: unsupported format string passed to NoneType
    weekend.predict_next_day(object(), 'dummy_hist.db', None,
                             datetime(2026, 9, 12, 18, 7, tzinfo=JST))

    out = capsys.readouterr().out
    assert 'オッズ不明' in out, 'オッズ不明を明示せず 0.0倍 と偽っていないか'
    assert '0.0倍' not in out
    assert (tmp_path / 'latest.json').exists(), 'latest.json が生成されていない'


def test_saturday_prediction_survives_all_odds_missing(monkeypatch, tmp_path, capsys):
    """対になっている friday_predict 側（片方だけ直す事故を防ぐ）。"""
    import scripts.friday_predict as fp

    race, scored = _race(), _scored()
    _patch_common(monkeypatch, fp, tmp_path, race, scored)
    monkeypatch.setattr(fp, 'init_engine', lambda *a, **k: None)
    monkeypatch.setattr(fp, 'init_betting', lambda *a, **k: None)
    monkeypatch.setattr(fp, 'init_db', lambda *a, **k: None)
    monkeypatch.setattr(fp, 'backup_db', lambda *a, **k: None)
    monkeypatch.setattr(fp, 'checkpoint_db', lambda *a, **k: None)
    monkeypatch.setattr(fp, 'create_session', lambda *a, **k: object())

    fp.main()

    out = capsys.readouterr().out
    assert 'オッズ不明' in out
    assert '0.0倍' not in out


def test_no_raw_win_odds_format_remains_in_production_scripts():
    """`.get('win_odds', 0):.1f` 型の書き方が復活していないこと。"""
    import re
    root = os.path.join(os.path.dirname(__file__), '..')
    bad = []
    for rel in ('scripts/weekend.py', 'scripts/friday_predict.py'):
        src = open(os.path.join(root, rel), encoding='utf-8').read()
        if re.search(r'win_odds["\']?(?:\s*,\s*0)?\s*\)\s*:\s*\.?\d*f', src):
            bad.append(rel)
    assert not bad, f'None で落ちる書き方が残っている: {bad}'
