"""距離単位のコース台帳 data/course_starts.json の契約テスト。

🔑 ユーザー指示（2026-09-15）を構造で守る:
   > 約○mと書かれていない距離について、推定値を勝手にfeatureとして採用しないこと。
   > 「公式記載値」「公式図からの算出値」「未確定」を分けたデータ表にする。
   > 特にAIに投入するなら、ここを曖昧にしない。
"""
import json
import os

import pytest

from src.features.race_shape import (
    course_start_record,
    start_to_first_corner_m,
)

USABLE = {'official', 'stated'}

# 資料が芝スタートと明記しているダート（これ以外を勝手に足させない）
DOCUMENTED_TURF_STARTS = {
    ('東京', 'ダート', 1600),
    ('中山', 'ダート', 1200),
    ('新潟', 'ダート', 1200),
    ('福島', 'ダート', 1150),
}


@pytest.fixture(scope='module')
def S():
    with open(os.path.join('data', 'course_starts.json'), encoding='utf-8') as f:
        return json.load(f)['starts']


def test_no_number_without_a_usable_source(S):
    """数値があるのに出典が無い／使えない、という状態を作らせない。"""
    for k, e in S.items():
        if e.get('start_to_first_corner_m') is not None:
            assert e.get('start_to_first_corner_src') in USABLE, \
                f'{k}: src={e.get("start_to_first_corner_src")} なのに数値を持っている'
        if e.get('turf_start_m') is not None:
            assert e.get('turf_start_src') in USABLE, \
                f'{k}: 芝区間の数値に出典が無い'


def test_getter_refuses_unsourced_numbers(S):
    """出典の無いセルは getter が None を返す（呼び出し側が埋められない）。"""
    for k, e in S.items():
        if e.get('start_to_first_corner_src') in USABLE:
            continue
        got = start_to_first_corner_m(
            e['racecourse'], e['surface'], e['distance'])
        # course_physical.json 側にも無いことを含めて、確かな値以外は出さない
        if got is not None:
            from src.features.race_shape import _load_physical
            ph = (_load_physical(None) or {}).get('distances', {})
            assert ph.get(k, {}).get('src') in USABLE, \
                f'{k}: 出典の無い数値が getter から出ている'


def test_documented_official_dirt_numbers_reach_the_getter():
    """ダート資料の【公式】3件が実際に届くこと（台帳に入れただけで終わらせない）。"""
    assert start_to_first_corner_m('東京', 'ダート', 1600) == 640.0
    assert start_to_first_corner_m('阪神', 'ダート', 1800) == 300.0
    assert start_to_first_corner_m('札幌', 'ダート', 1700) == 240.0


def test_turf_start_flag_is_only_the_documented_four(S):
    """芝スタートは独立フラグ。資料が明記した4つ以外を推定で足さない。"""
    flagged = {(e['racecourse'], e['surface'], e['distance'])
               for e in S.values() if e.get('turf_start')}
    assert flagged == DOCUMENTED_TURF_STARTS, f'差分: {flagged ^ DOCUMENTED_TURF_STARTS}'


def test_turf_start_distance_only_where_documented(S):
    """芝区間の長さが分かっているのは東京ダ1600だけ。他は None のまま。"""
    withm = {(e['racecourse'], e['surface'], e['distance'])
             for e in S.values() if e.get('turf_start_m') is not None}
    assert withm == {('東京', 'ダート', 1600)}
    assert course_start_record('東京', 'ダート', 1600)['turf_start_m'] == 150.0
    # 新潟ダ1200・福島ダ1150 は「芝区間の距離は未確定」と資料が言っている
    for rc, ds in [('新潟', 1200), ('福島', 1150)]:
        assert course_start_record(rc, 'ダート', ds)['turf_start_m'] is None


def test_dirt_has_no_inner_outer_loop(S):
    """ダートは内外回りの区別が無い（資料の注記）。loop を付けない。"""
    for k, e in S.items():
        if e['surface'] == 'ダート':
            assert e.get('loop') is None, f'{k} にダートなのに loop が入っている'


def test_merge_distance_is_not_filed_as_first_corner(S):
    """中京芝1600の200mは「本線合流まで」で1角までではない。同じ列に混ぜない。"""
    rec = course_start_record('中京', '芝', 1600)
    assert rec is not None
    assert rec.get('start_to_mainline_merge_m') is not None
    assert rec.get('start_to_first_corner_m') is None


def test_every_venue_surface_pair_is_covered(S):
    """20の会場×馬場が全部ある（ダート資料の反映で埋まった状態を固定する）。"""
    pairs = {(e['racecourse'], e['surface']) for e in S.values()}
    venues = {'札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'}
    assert pairs == {(v, s) for v in venues for s in ('芝', 'ダート')}


def test_starts_is_a_superset_of_course_physical():
    """course_starts は course_physical の上位互換で、値が食い違わないこと。

    getter が course_starts を先に見るので、ここがずれると静かに別の値になる。
    """
    with open(os.path.join('data', 'course_physical.json'), encoding='utf-8') as f:
        ph = json.load(f)['distances']
    with open(os.path.join('data', 'course_starts.json'), encoding='utf-8') as f:
        st = json.load(f)['starts']
    for k, e in ph.items():
        if e.get('src') not in USABLE or e.get('start_to_first_corner_m') is None:
            continue
        assert k in st, f'{k} が course_starts に無い'
        assert st[k].get('start_to_first_corner_src') in USABLE, f'{k} の出典が落ちている'
        assert abs(float(st[k]['start_to_first_corner_m'])
                   - float(e['start_to_first_corner_m'])) < 1e-6, f'{k} の値が食い違う'


def test_spiral_corner_matches_documented_venues(S):
    """スパイラルカーブは資料が名指しした会場だけに立てる。"""
    spiral_dirt = {e['racecourse'] for e in S.values()
                   if e['surface'] == 'ダート' and e.get('spiral_corner')}
    assert spiral_dirt == {'中京', '新潟', '福島', '函館'}
