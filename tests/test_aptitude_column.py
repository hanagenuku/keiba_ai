"""コース・馬場適性順位（fit_rank）の回帰テスト。

🔴 この列は**表示専用**。買い目・軸・レース選択には一切使わない。
   Step 1（2026-09-22）で「3着内率への効果の67〜81%は市場評価の言い換え」
   「回収率は測った36マスすべて100%未満」と実測済み。ここでは
   「計算が正しいこと」と「買い目側から読まれていないこと」を固定する。
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.aptitude import APT_COLS, aptitude_scores, calc_fit_ranks


def _row(**kw):
    base = {c: 0.0 for c, _ in APT_COLS}
    base.update(kw)
    return base


def test_apt_cols_match_gate0():
    """🔴 研究側（betting/gate0.py の T_COLS）と列・符号が一致すること。

    片方だけ変えると「研究で通した軸」と「画面に出る軸」が静かに食い違う。
    2026-09-15 のペースモデル（学習と推論で入力が別物だった）と同型の事故。
    """
    src = open(os.path.join(os.path.dirname(__file__), '..', 'betting', 'gate0.py'),
              encoding='utf-8').read()
    tree = ast.parse(src)
    t_cols = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'T_COLS':
            t_cols = ast.literal_eval(node.value)
    assert t_cols is not None, 'betting/gate0.py に T_COLS が無い'
    assert list(APT_COLS) == [tuple(x) for x in t_cols], \
        f'適性軸が研究側とズレている: {APT_COLS} vs {t_cols}'


def test_higher_fit_columns_rank_first():
    xs = [_row(f_style_course_fit=v) for v in (1.0, 2.0, 3.0)]
    assert calc_fit_ranks([7, 8, 9], xs) == {9: 1, 8: 2, 7: 3}


def test_dist_vs_optimal_is_absneg():
    """距離適性は0から離れるほど不利（absneg）。符号を取り違えると順位が逆になる。"""
    xs = [{'f_dist_vs_optimal': v} for v in (0.0, 2.0, -5.0)]
    got = calc_fit_ranks([1, 2, 3], xs)
    assert got[1] == 1 and got[2] == 2 and got[3] == 3, got


def test_zero_variance_column_is_skipped():
    """レース内で全馬同値の列は使わない（z化できないため）。"""
    xs = [{'f_style_course_fit': 5.0, 'f_speed_x_shortening': v} for v in (1.0, 2.0)]
    scores, used = aptitude_scores(xs)
    assert used == ['f_speed_x_shortening'], used


def test_returns_empty_when_no_column_usable():
    """使える列が1つも無ければ空 → アプリ側は列ごと非表示（数字を作らない）。"""
    assert calc_fit_ranks([1, 2], [{}, {}]) == {}
    assert calc_fit_ranks([1, 2], [{'f_style_course_fit': None}] * 2) == {}


def test_nan_column_is_skipped():
    xs = [{'f_style_course_fit': float('nan')}, {'f_style_course_fit': 1.0}]
    assert calc_fit_ranks([1, 2], xs) == {}


def test_ties_share_the_same_rank():
    xs = [_row(f_style_course_fit=1.0), _row(f_style_course_fit=1.0),
          _row(f_style_course_fit=9.0)]
    got = calc_fit_ranks([1, 2, 3], xs)
    assert got[3] == 1 and got[1] == got[2] == 2, got


def test_betting_modules_do_not_read_fit_rank():
    """🔴 買い目・軸・レース選択が fit_rank を読んでいないこと。

    読み始めた瞬間に「測っていない軸で馬券を選ぶ」状態になる。app_json.py は
    画面へ渡すだけなので免除する。
    """
    root = os.path.join(os.path.dirname(__file__), '..', 'src', 'betting')
    offenders = []
    for name in sorted(os.listdir(root)):
        if not name.endswith('.py') or name == 'app_json.py':
            continue
        if 'fit_rank' in open(os.path.join(root, name), encoding='utf-8').read():
            offenders.append(name)
    assert not offenders, f'fit_rank を買い目側が読んでいる: {offenders}'
