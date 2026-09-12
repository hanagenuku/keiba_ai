"""EV用確率（AI勝率・AI複勝・必要オッズ）が本番で実際に計算されることを固定する。

🔴 2026-09-12 に本番 latest.json で ev_tan_pct / ev_fuku_pct / need_odds が
**全馬 null** だった。原因は `to_app_json(..., base_dir=...)` を本番の
3箇所（friday_predict×1 / weekend×2）が渡しておらず、既定の None のまま
`_load_ability_calibrator(None)` が None を返していたこと。
＝PR #239 以降、この3列は一度も本番で出ていなかった。

2026-08-26 の棚卸しで見つけた `prediction_snapshots`（8,464行を溜めていたのに
誰も読んでいなかった）と同じ「作ったのに配線していない」型。
"""
import ast
import pytest
from src.betting.app_json import _build_ev_probs, _load_ability_calibrator

PROD = ['scripts/friday_predict.py', 'scripts/weekend.py']


def _to_app_json_calls(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'to_app_json']


@pytest.mark.parametrize('path', PROD)
def test_production_passes_base_dir(path):
    """🔴 本番の呼び出しは必ず base_dir を渡すこと（渡さないと3列が黙って null になる）。"""
    calls = _to_app_json_calls(path)
    assert calls, f'{path} に to_app_json の呼び出しが無い'
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert 'base_dir' in kw, \
            f'{path}:{c.lineno} が base_dir を渡していない → AI勝率/必要オッズが null になる'


def test_calibrator_file_is_loadable():
    """較正器の実ファイルが読めること（無ければ3列は出ない）。"""
    cal = _load_ability_calibrator('.')
    assert cal, 'data/ability_calibrator.pkl が読めない'
    assert 'win' in cal and 'fuku' in cal


def test_ev_probs_are_produced_with_base_dir():
    """base_dir があれば実際に値が出ること。"""
    scored = [{'num': i, 'ability_margin': -1.0 + 0.3 * i} for i in range(1, 9)]
    got = _build_ev_probs(scored, '.')
    assert got, 'base_dir を渡しても EV用確率が空'
    for n, (w, f) in got.items():
        assert 0.0 <= w <= 1.0 and 0.0 <= f <= 1.0
    assert len({round(w, 6) for w, _ in got.values()}) > 1, '全馬同じ確率になっている'


def test_ev_probs_empty_without_base_dir():
    """base_dir が無いと空になる＝今回の不具合の姿。これが本番で起きていた。"""
    scored = [{'num': i, 'ability_margin': -1.0 + 0.3 * i} for i in range(1, 9)]
    assert _build_ev_probs(scored, None) == {}
